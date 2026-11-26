"""Classical models: TF-IDF (word + character n-grams) with LR, linear SVM or Naive Bayes.

Hyperparameters are searched with Optuna, maximizing macro F1 on the fixed validation
split (FR-06). TF-IDF matrices are cached per vectorizer setting so the search mostly
fits classifiers, not vectorizers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from sentiment.evaluation.metrics import compute_metrics
from sentiment.predict import Predictor
from sentiment.utils import get_logger, path_size_mb, timer

logger = get_logger(__name__)

ClassifierName = Literal["logreg", "linear_svm", "naive_bayes"]
CLASSIFIER_DISPLAY_NAMES: dict[str, str] = {
    "logreg": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "naive_bayes": "TF-IDF + Complement Naive Bayes",
}
CHAR_NGRAM_RANGES: dict[str, tuple[int, int] | None] = {
    "none": None,
    "2-4": (2, 4),
    "2-5": (2, 5),
    "3-5": (3, 5),
    "3-6": (3, 6),
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class SearchSpace(BaseModel):
    """Optuna search space. Lists are categorical choices, pairs are log-uniform ranges."""

    model_config = ConfigDict(extra="forbid")

    word_ngram_max: list[int] = [1, 2]
    char_ngrams: list[str] = ["none", "2-5", "3-5"]
    min_df: list[int] = [1, 2, 3]
    sublinear_tf: list[bool] = [True, False]
    logreg_c: tuple[float, float] = (1e-2, 1e2)
    svm_c: tuple[float, float] = (1e-3, 1e2)
    nb_alpha: tuple[float, float] = (1e-3, 1.0)


class ClassicalParams(BaseModel):
    """Model-specific settings (``params`` section of ``configs/classical.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    classifiers: list[ClassifierName] = ["logreg", "linear_svm", "naive_bayes"]
    n_trials: int = Field(default=30, ge=1)
    search_space: SearchSpace = Field(default_factory=SearchSpace)
    svm_calibration_folds: int = Field(default=5, ge=2)
    max_df: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=20, ge=1)
    latency_samples: int = Field(default=50, ge=1)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VectorizerParams:
    """TF-IDF settings shared by the word and character vectorizers."""

    word_ngram_max: int
    char_ngrams: str
    min_df: int
    sublinear_tf: bool
    max_df: float = 1.0


def build_word_vectorizer(p: VectorizerParams) -> TfidfVectorizer:
    """Word unigram (and optionally bigram) TF-IDF; keeps tokens of 2+ letters."""
    return TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, p.word_ngram_max),
        min_df=p.min_df,
        max_df=p.max_df,
        sublinear_tf=p.sublinear_tf,
        dtype=np.float32,
    )


def build_char_vectorizer(p: VectorizerParams) -> TfidfVectorizer | None:
    """Character n-gram TF-IDF inside word boundaries, or None if disabled."""
    ngram_range = CHAR_NGRAM_RANGES[p.char_ngrams]
    if ngram_range is None:
        return None
    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=ngram_range,
        min_df=p.min_df,
        max_df=p.max_df,
        sublinear_tf=p.sublinear_tf,
        dtype=np.float32,
    )


def build_features(p: VectorizerParams) -> FeatureUnion:
    """Unfitted feature union of the word and (optional) character vectorizers."""
    parts: list[tuple[str, TfidfVectorizer]] = [("word", build_word_vectorizer(p))]
    char = build_char_vectorizer(p)
    if char is not None:
        parts.append(("char", char))
    return FeatureUnion(parts)


class FeatureCache:
    """Fit each word/char vectorizer setting once and reuse its matrices across trials."""

    def __init__(self, train_texts: Sequence[str], val_texts: Sequence[str]) -> None:
        """Store the texts; vectorizers are fitted lazily on first request."""
        self.train_texts = list(train_texts)
        self.val_texts = list(val_texts)
        self._word: dict[tuple[Any, ...], tuple[sp.csr_matrix, sp.csr_matrix]] = {}
        self._char: dict[tuple[Any, ...], tuple[sp.csr_matrix, sp.csr_matrix]] = {}

    def _fit(self, vec: TfidfVectorizer) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return vec.fit_transform(self.train_texts), vec.transform(self.val_texts)

    def get(self, p: VectorizerParams) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Return ``(X_train, X_val)`` for the given settings."""
        word_key = (p.word_ngram_max, p.min_df, p.sublinear_tf, p.max_df)
        if word_key not in self._word:
            self._word[word_key] = self._fit(build_word_vectorizer(p))
        blocks = [self._word[word_key]]
        char_vec = build_char_vectorizer(p)
        if char_vec is not None:
            char_key = (p.char_ngrams, p.min_df, p.sublinear_tf, p.max_df)
            if char_key not in self._char:
                self._char[char_key] = self._fit(char_vec)
            blocks.append(self._char[char_key])
        if len(blocks) == 1:
            return blocks[0]
        return (
            sp.hstack([b[0] for b in blocks], format="csr"),
            sp.hstack([b[1] for b in blocks], format="csr"),
        )


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #


def build_classifier(
    name: ClassifierName,
    hp: dict[str, float],
    seed: int,
    calibrate: bool = False,
    calibration_folds: int = 5,
) -> ClassifierMixin:
    """Create an unfitted classifier.

    Args:
        name: Classifier identifier.
        hp: Classifier hyperparameters (``C`` or ``alpha``).
        seed: Random seed.
        calibrate: For the SVM, wrap it in sigmoid (Platt) calibration so it exposes
            ``predict_proba``. Ignored for the other classifiers.
        calibration_folds: Folds used for calibration.

    Returns:
        Scikit-learn classifier.
    """
    if name == "logreg":
        return LogisticRegression(C=hp["C"], max_iter=5000, random_state=seed)
    if name == "linear_svm":
        svm = LinearSVC(C=hp["C"], random_state=seed, max_iter=10000)
        if not calibrate:
            return svm
        cv = StratifiedKFold(n_splits=calibration_folds, shuffle=True, random_state=seed)
        return CalibratedClassifierCV(svm, method="sigmoid", cv=cv)
    if name == "naive_bayes":
        return ComplementNB(alpha=hp["alpha"])
    raise ValueError(f"Unknown classifier {name!r}")


def suggest(
    trial: optuna.Trial, name: ClassifierName, space: SearchSpace, max_df: float
) -> tuple[VectorizerParams, dict[str, float]]:
    """Sample vectorizer and classifier hyperparameters for one Optuna trial."""
    vec = VectorizerParams(
        word_ngram_max=trial.suggest_categorical("word_ngram_max", space.word_ngram_max),
        char_ngrams=trial.suggest_categorical("char_ngrams", space.char_ngrams),
        min_df=trial.suggest_categorical("min_df", space.min_df),
        sublinear_tf=trial.suggest_categorical("sublinear_tf", space.sublinear_tf),
        max_df=max_df,
    )
    if name == "logreg":
        hp = {"C": trial.suggest_float("C", *space.logreg_c, log=True)}
    elif name == "linear_svm":
        hp = {"C": trial.suggest_float("C", *space.svm_c, log=True)}
    else:
        hp = {"alpha": trial.suggest_float("alpha", *space.nb_alpha, log=True)}
    return vec, hp


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #


def class_weights(clf: ClassifierMixin) -> np.ndarray:
    """Per-class feature weights of a fitted linear classifier, shape ``(n_classes, n_features)``.

    Higher weight means stronger evidence for that class. For binary linear models
    scikit-learn stores a single row (evidence for ``classes_[1]``); it is expanded to
    ``[-w, w]``. For calibrated SVMs the fold estimators' weights are averaged. For
    Naive Bayes the log-probabilities are centered across classes.
    """
    if isinstance(clf, CalibratedClassifierCV):
        coefs = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        flp = clf.feature_log_prob_
        return flp - flp.mean(axis=0, keepdims=True)
    else:
        raise TypeError(f"{type(clf).__name__} has no linear weights")
    coefs = np.asarray(coefs)
    return np.vstack([-coefs[0], coefs[0]]) if coefs.shape[0] == 1 else coefs


def _split_feature_name(name: str) -> tuple[str, str]:
    kind, _, feature = name.partition("__")
    return kind, feature


def feature_names(pipeline: Pipeline) -> np.ndarray:
    """Names of all features produced by the pipeline's feature union."""
    return pipeline.named_steps["features"].get_feature_names_out()


def top_features(pipeline: Pipeline, k: int = 20, kind: str | None = None) -> pd.DataFrame:
    """Most influential features per class.

    Args:
        pipeline: Fitted ``features`` + ``clf`` pipeline.
        k: Features per class.
        kind: Restrict to ``"word"`` or ``"char"`` features; None keeps both.

    Returns:
        DataFrame with ``label``, ``rank``, ``feature``, ``kind``, ``weight``.
    """
    names = pipeline.named_steps["features"].get_feature_names_out()
    split = [_split_feature_name(n) for n in names]
    kinds = np.array([s[0] for s in split])
    features = np.array([s[1] for s in split])
    weights = class_weights(pipeline.named_steps["clf"])
    mask = np.ones(len(names), bool) if kind is None else kinds == kind
    idx_all = np.flatnonzero(mask)

    rows = []
    for c, label in enumerate(pipeline.classes_):
        w = weights[c, idx_all]
        order = idx_all[np.argsort(-w)[:k]]
        rows += [
            {
                "label": label,
                "rank": r,
                "feature": features[i],
                "kind": kinds[i],
                "weight": float(weights[c, i]),
            }
            for r, i in enumerate(order, start=1)
        ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #


class ClassicalPredictor(Predictor):
    """:class:`Predictor` wrapper around a fitted TF-IDF pipeline."""

    MODEL_FILE = "model.joblib"
    METADATA_FILE = "metadata.json"

    def __init__(self, pipeline: Pipeline, name: str = "classical") -> None:
        """Wrap a fitted pipeline whose final step exposes ``predict_proba``."""
        self.pipeline = pipeline
        self.name = name
        self.labels = [str(c) for c in pipeline.classes_]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Class probabilities, columns ordered as :attr:`labels`."""
        return self.pipeline.predict_proba(list(texts))

    def explain(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        """Word features that pushed the prediction for ``text`` the most (FR-14).

        Contribution of a feature = its TF-IDF value times the weight difference
        between the predicted class and the mean of the other classes. Positive
        values support the predicted class, negative values oppose it. Only word
        features are returned because they can be highlighted in the text.

        Args:
            text: Input text.
            k: Number of features to return.

        Returns:
            ``(feature, contribution)`` pairs sorted by absolute contribution.
        """
        features = self.pipeline.named_steps["features"]
        x = features.transform([text]).tocsr()
        weights = class_weights(self.pipeline.named_steps["clf"])
        pred = int(np.argmax(self.predict_proba([text])[0]))
        others = np.delete(weights, pred, axis=0).mean(axis=0)
        direction = weights[pred] - others
        names = features.get_feature_names_out()
        contrib = [
            (_split_feature_name(names[i])[1], float(v * direction[i]))
            for i, v in zip(x.indices, x.data, strict=True)
            if names[i].startswith("word__")
        ]
        return sorted(contrib, key=lambda t: -abs(t[1]))[:k]

    def save(self, directory: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save the pipeline and a ``metadata.json`` (used by ``load_predictor``)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / self.MODEL_FILE, compress=3)
        meta = {"type": "classical", "name": self.name, "labels": self.labels, **(metadata or {})}
        (directory / self.METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ClassicalPredictor:
        """Load a predictor saved with :meth:`save`."""
        directory = Path(directory)
        meta = json.loads((directory / cls.METADATA_FILE).read_text(encoding="utf-8"))
        return cls(joblib.load(directory / cls.MODEL_FILE), name=meta.get("name", "classical"))


# --------------------------------------------------------------------------- #
# Search and training
# --------------------------------------------------------------------------- #


@dataclass
class SearchResult:
    """Best hyperparameters for one classifier and the full trial history."""

    classifier: ClassifierName
    vectorizer: VectorizerParams
    hyperparams: dict[str, float]
    best_val_macro_f1: float
    trials: pd.DataFrame
    search_seconds: float
    at_boundary: list[str] = field(default_factory=list)


def search(
    name: ClassifierName,
    cache: FeatureCache,
    y_train: np.ndarray,
    y_val: np.ndarray,
    params: ClassicalParams,
    seed: int,
) -> SearchResult:
    """Run the Optuna search for one classifier, maximizing validation macro F1.

    The SVM is searched without calibration (calibration does not change which
    hyperparameters separate the classes best, and is 5x slower).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        vec, hp = suggest(trial, name, params.search_space, params.max_df)
        x_train, x_val = cache.get(vec)
        clf = build_classifier(name, hp, seed)
        clf.fit(x_train, y_train)
        return compute_metrics(y_val, clf.predict(x_val), sorted(set(y_train)))["macro_f1"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"classical-{name}",
    )
    with timer() as t:
        study.optimize(objective, n_trials=params.n_trials)

    best = dict(study.best_params)
    vec = VectorizerParams(
        word_ngram_max=best.pop("word_ngram_max"),
        char_ngrams=best.pop("char_ngrams"),
        min_df=best.pop("min_df"),
        sublinear_tf=best.pop("sublinear_tf"),
        max_df=params.max_df,
    )
    trials = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    ranges = {"logreg": space_range(params, "logreg_c"), "linear_svm": space_range(params, "svm_c")}
    ranges["naive_bayes"] = space_range(params, "nb_alpha")
    low, high = ranges[name]
    at_boundary = [k for k, v in best.items() if near_log_boundary(v, low, high)]
    for k in at_boundary:
        logger.warning(
            "%s: best %s=%.4g is at the edge of [%g, %g]; consider widening the range",
            name,
            k,
            best[k],
            low,
            high,
        )
    return SearchResult(name, vec, best, float(study.best_value), trials, t["seconds"], at_boundary)


def space_range(params: ClassicalParams, field_name: str) -> tuple[float, float]:
    """Return the ``(low, high)`` log-uniform range of a search-space field."""
    low, high = getattr(params.search_space, field_name)
    return float(low), float(high)


def near_log_boundary(value: float, low: float, high: float, tolerance: float = 0.05) -> bool:
    """True if ``value`` lies within ``tolerance`` of either end of ``[low, high]`` on a log scale.

    A best value at the edge of the range suggests the optimum may lie outside it.
    """
    span = np.log(high) - np.log(low)
    position = (np.log(value) - np.log(low)) / span
    return bool(position < tolerance or position > 1 - tolerance)


@dataclass
class TrainedModel:
    """A final, refitted classical model with its validation results and costs (FR-10)."""

    name: ClassifierName
    predictor: ClassicalPredictor
    search: SearchResult
    val_metrics: dict[str, float]
    train_seconds: float
    inference_ms_per_example: float
    latency_ms_single: float
    extra: dict[str, Any] = field(default_factory=dict)


def fit_final(
    result: SearchResult,
    train: pd.DataFrame,
    val: pd.DataFrame,
    params: ClassicalParams,
    seed: int,
) -> TrainedModel:
    """Refit the best configuration from scratch on the training split and measure it.

    Training time covers the whole pipeline (vectorizers + classifier + calibration),
    so it is comparable with the neural models' training time.
    """
    pipeline = Pipeline(
        [
            ("features", build_features(result.vectorizer)),
            (
                "clf",
                build_classifier(
                    result.classifier,
                    result.hyperparams,
                    seed,
                    calibrate=True,
                    calibration_folds=params.svm_calibration_folds,
                ),
            ),
        ]
    )
    with timer() as t_train:
        pipeline.fit(train["text"].tolist(), train["label"].to_numpy())

    predictor = ClassicalPredictor(pipeline, name=CLASSIFIER_DISPLAY_NAMES[result.classifier])
    val_texts = val["text"].tolist()
    with timer() as t_batch:
        val_pred = predictor.predict(val_texts)
    latencies = []
    for text in val_texts[: params.latency_samples]:
        with timer() as t_one:
            predictor.predict_proba([text])
        latencies.append(t_one["seconds"])

    return TrainedModel(
        name=result.classifier,
        predictor=predictor,
        search=result,
        val_metrics=compute_metrics(val["label"].tolist(), val_pred, predictor.labels),
        train_seconds=t_train["seconds"],
        inference_ms_per_example=1000 * t_batch["seconds"] / len(val_texts),
        latency_ms_single=1000 * float(np.median(latencies)),
    )


def save_model(model: TrainedModel, directory: Path, top_k: int) -> dict[str, Any]:
    """Save model, metadata, trial history and top features; return the metadata dict."""
    top_all = top_features(model.predictor.pipeline, k=top_k)
    top_words = top_features(model.predictor.pipeline, k=top_k, kind="word")
    metadata = {
        "classifier": model.name,
        "vectorizer": model.search.vectorizer.__dict__,
        "hyperparams": model.search.hyperparams,
        "search_best_val_macro_f1": model.search.best_val_macro_f1,
        "val_metrics": model.val_metrics,
        "train_seconds": model.train_seconds,
        "search_seconds": model.search.search_seconds,
        "inference_ms_per_example": model.inference_ms_per_example,
        "latency_ms_single": model.latency_ms_single,
        "n_features": len(feature_names(model.predictor.pipeline)),
    }
    model.predictor.save(directory, metadata)
    metadata["model_size_mb"] = path_size_mb(directory / ClassicalPredictor.MODEL_FILE)
    model.predictor.save(directory, metadata)  # rewrite metadata now that the size is known
    model.search.trials.to_csv(directory / "trials.csv", index=False)
    top_all.to_csv(directory / "top_features.csv", index=False)
    top_words.to_csv(directory / "top_words.csv", index=False)
    model.extra.update(metadata=metadata, top_words=top_words, top_features=top_all)
    return metadata


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def render_report(models: list[TrainedModel], best: TrainedModel, n_trials: int) -> str:
    """Markdown summary of the validation results; every number comes from ``models``."""
    labels = best.predictor.labels
    n_val = best.extra["n_val"]

    metric_rows = []
    for m in models:
        meta = m.extra["metadata"]
        metric_rows.append(
            [
                m.predictor.name + (" **(best)**" if m is best else ""),
                f"{m.val_metrics['macro_f1']:.4f}",
                f"{m.val_metrics['accuracy']:.4f}",
                f"{m.train_seconds:.1f}",
                f"{m.search.search_seconds:.1f}",
                f"{m.inference_ms_per_example:.2f}",
                f"{m.latency_ms_single:.1f}",
                f"{meta['model_size_mb']:.1f}",
                f"{meta['n_features']:,}",
            ]
        )

    hp_rows = []
    for m in models:
        v = m.search.vectorizer
        hp = ", ".join(f"{k}={val:.4g}" for k, val in m.search.hyperparams.items())
        hp_rows.append(
            [
                m.predictor.name,
                f"1–{v.word_ngram_max}",
                v.char_ngrams,
                str(v.min_df),
                str(v.sublinear_tf),
                hp,
                f"{m.search.best_val_macro_f1:.4f}",
            ]
        )

    flagged = [
        f"{m.predictor.name}: {', '.join(m.search.at_boundary)}"
        for m in models
        if m.search.at_boundary
    ]
    boundary_note = (
        "**Warning:** best value at the edge of its search range (" + "; ".join(flagged) + ")."
        if flagged
        else "No regularization parameter ended at the edge of its search range."
    )

    top = best.extra["top_words"]
    k = int(top["rank"].max())
    word_rows = []
    for rank in range(1, k + 1):
        cells = [str(rank)]
        for lab in labels:
            row = top[(top["label"] == lab) & (top["rank"] == rank)]
            cell = f"{row['feature'].item()} ({row['weight'].item():.3f})" if len(row) else ""
            cells.append(cell)
        word_rows.append(cells)

    blocks = [
        "# Classical models: validation results",
        "_Generated by `scripts/train.py --config configs/classical.yaml` — do not edit by "
        "hand. Test-set results are produced separately in the evaluation phase._",
        f"Each classifier was tuned with {n_trials} Optuna (TPE) trials maximizing macro F1 "
        "on the validation split, then refitted from scratch on the training split.",
        "## Validation metrics",
        _md_table(
            [
                "Model",
                "Macro F1",
                "Accuracy",
                "Train time (s)",
                "Search time (s)",
                "Inference (ms/example)",
                "Single-text latency (ms)",
                "Size (MB)",
                "Features",
            ],
            metric_rows,
        ),
        f"- Validation macro F1 is measured on only {n_val} reviews, so one review changes it "
        f"by about {100 / n_val:.1f} points; differences of 1–2 points are within noise.\n"
        "- *Train time* is a fresh fit of the full pipeline (vectorizers, classifier and, for "
        "the SVM, 5-fold probability calibration).\n"
        "- *Search time* for the first classifier includes building the shared TF-IDF "
        "cache; later classifiers reuse it.",
        "## Selected hyperparameters",
        _md_table(
            [
                "Model",
                "Word n-grams",
                "Char n-grams",
                "min_df",
                "Sublinear TF",
                "Classifier",
                "Search best F1",
            ],
            hp_rows,
        ),
        "*Search best F1* is the uncalibrated score found during the search. The final SVM "
        "is wrapped in Platt calibration, so its validation score can differ slightly.",
        boundary_note,
        f"## Top {k} most influential words per class ({best.predictor.name})",
        "Word features ranked by model weight. Character n-gram weights are in "
        "`top_features.csv` in the model directory.",
        _md_table(["Rank", *[f"{lab} (weight)" for lab in labels]], word_rows),
    ]
    return "\n\n".join(blocks) + "\n"


def train_classical(config: Any, report_dir: str | Path = "reports/classical") -> dict[str, Any]:
    """Search, refit, save and log every configured classical model.

    Outputs:
        * ``<output_dir>/<classifier>/``: model, metadata, trials, top features
        * ``<output_dir>/best/``: copy of the best model by validation macro F1
        * ``<output_dir>/config.yaml`` and ``summary.json``
        * ``<report_dir>/validation.md`` and ``top_words.csv``
        * One MLflow parent run with a nested run per classifier

    Args:
        config: :class:`sentiment.utils.RunConfig` with ``model: classical``.
        report_dir: Where the git-tracked report is written.

    Returns:
        Summary dict (also saved as ``summary.json``).
    """
    import shutil

    from sentiment.tracking import flatten_params, mlflow, setup_mlflow
    from sentiment.utils import save_config, set_seed

    params = ClassicalParams.model_validate(config.params)
    set_seed(config.seed)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    train = pd.read_parquet(config.data.train)
    val = pd.read_parquet(config.data.val)
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    cache = FeatureCache(train["text"], val["text"])

    setup_mlflow(config)
    models: list[TrainedModel] = []
    with mlflow.start_run(run_name="classical") as parent:
        mlflow.log_params(flatten_params({"seed": config.seed, **params.model_dump()}))
        mlflow.log_artifact(str(out_dir / "config.yaml"))
        for name in params.classifiers:
            logger.info("Searching %s (%d trials)", name, params.n_trials)
            result = search(name, cache, y_train, y_val, params, config.seed)
            model = fit_final(result, train, val, params, config.seed)
            model.extra["n_val"] = len(val)
            meta = save_model(model, out_dir / name, params.top_k)
            logger.info(
                "%s: val macro F1 %.4f (search best %.4f), train %.1fs",
                name,
                model.val_metrics["macro_f1"],
                result.best_val_macro_f1,
                model.train_seconds,
            )
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(
                    flatten_params(
                        {
                            "classifier": name,
                            "vectorizer": result.vectorizer.__dict__,
                            **result.hyperparams,
                        }
                    )
                )
                mlflow.log_metrics(
                    {f"val_{k}": v for k, v in model.val_metrics.items()}
                    | {
                        "search_best_val_macro_f1": result.best_val_macro_f1,
                        "train_seconds": model.train_seconds,
                        "search_seconds": result.search_seconds,
                        "inference_ms_per_example": model.inference_ms_per_example,
                        "latency_ms_single": model.latency_ms_single,
                        "model_size_mb": meta["model_size_mb"],
                    }
                )
                mlflow.log_artifacts(str(out_dir / name), artifact_path=name)
            models.append(model)

        best = max(models, key=lambda m: m.val_metrics["macro_f1"])
        best_dir = out_dir / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(out_dir / best.name, best_dir)

        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "validation.md").write_text(
            render_report(models, best, params.n_trials), encoding="utf-8"
        )
        best.extra["top_words"].to_csv(report_dir / "top_words.csv", index=False)

        summary = {
            "best": best.name,
            "mlflow_run_id": parent.info.run_id,
            "models": {m.name: m.extra["metadata"] for m in models},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_metric("best_val_macro_f1", best.val_metrics["macro_f1"])
        mlflow.set_tag("best_classifier", best.name)
        mlflow.log_artifact(str(report_dir / "validation.md"))
    logger.info("Best classical model: %s -> %s", best.name, best_dir)
    return summary
