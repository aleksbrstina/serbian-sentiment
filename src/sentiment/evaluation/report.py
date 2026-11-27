"""Evaluate every model on the test split and write the comparison report (FR-09/10/12).

Models are loaded only through :func:`sentiment.predict.load_predictor`, so all of them
are evaluated under identical conditions: same test texts, same metric code, same
machine for timing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import binomtest

from sentiment.evaluation.metrics import compute_metrics, confusion
from sentiment.predict import Predictor, load_predictor
from sentiment.utils import get_logger, timer

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ModelSpec(BaseModel):
    """One model to evaluate."""

    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    path: Path
    hub_id: str | None = None  # used if ``path`` does not exist (transformer only)


class EvalConfig(BaseModel):
    """Settings for ``scripts/evaluate.py`` (``configs/evaluate.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelSpec]
    test: Path = Path("data/processed/test.parquet")
    train: Path = Path("data/processed/train.parquet")
    lexicon: Path | None = Path("reports/eda/informative_words.csv")
    lexicon_top_k: int = 30
    output_dir: Path = Path("reports/evaluation")
    comparison_path: Path = Path("reports/comparison.md")
    readme_path: Path | None = Path("README.md")
    n_bootstrap: int = Field(default=2000, ge=100)
    seed: int = 42
    latency_samples: int = Field(default=50, ge=1)
    review_sample: int = Field(default=40, ge=1)
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "serbian-sentiment"
    mlflow_artifact_root: Path = Path("mlruns")


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load and validate the evaluation config."""
    import yaml

    with Path(path).open(encoding="utf-8") as f:
        return EvalConfig.model_validate(yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def bootstrap_macro_f1(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    labels: Sequence[str],
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap of macro F1: every model is scored on the same resamples.

    Args:
        y_true: Gold labels.
        predictions: Mapping model -> predicted labels.
        labels: Class labels.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        DataFrame of shape ``(n_resamples, n_models)`` with macro F1 per resample.
    """
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = {name: np.empty(n_resamples) for name in predictions}
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        for name, pred in predictions.items():
            scores[name][b] = f1_score(
                y_true[idx], pred[idx], labels=list(labels), average="macro", zero_division=0
            )
    return pd.DataFrame(scores)


def confidence_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval."""
    alpha = (1 - level) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def mcnemar_p(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    """Exact McNemar test: are the two models' error patterns different?

    Only the discordant examples (one model right, the other wrong) matter; under the
    null hypothesis each discordant example is equally likely to favour either model.
    """
    a_only = int((errors_a & ~errors_b).sum())
    b_only = int((~errors_a & errors_b).sum())
    if a_only + b_only == 0:
        return 1.0
    return float(binomtest(a_only, a_only + b_only, 0.5).pvalue)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class ModelResult:
    """Test-set results and costs for one model."""

    spec: ModelSpec
    labels: list[str]
    proba: np.ndarray
    pred: np.ndarray
    metrics: dict[str, float]
    inference_ms_per_example: float
    latency_ms_single: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> np.ndarray:
        """Boolean array, True where the prediction is wrong (set by the caller)."""
        return self.metadata["_errors"]


def _load(spec: ModelSpec) -> tuple[Predictor, dict[str, Any]]:
    if spec.path.exists():
        meta_path = spec.path / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return load_predictor(spec.path), meta
    if spec.hub_id:
        from huggingface_hub import hf_hub_download

        from sentiment.models.transformer import TransformerPredictor

        logger.info("%s not found locally; loading %s from the Hub", spec.path, spec.hub_id)
        meta = json.loads(Path(hf_hub_download(spec.hub_id, "metadata.json")).read_text())
        return TransformerPredictor.load(spec.hub_id, name=spec.name), meta
    raise FileNotFoundError(f"{spec.path} does not exist; train the {spec.key} model first.")


def evaluate_model(
    spec: ModelSpec, texts: list[str], y_true: np.ndarray, latency_samples: int
) -> ModelResult:
    """Predict on the test texts, compute metrics and time inference."""
    predictor, meta = _load(spec)
    predictor.predict_proba(texts[:2])  # warm-up (lazy init, caches)
    with timer() as t_batch:
        proba = predictor.predict_proba(texts)
    latencies = []
    for text in texts[:latency_samples]:
        with timer() as t_one:
            predictor.predict_proba([text])
        latencies.append(t_one["seconds"])
    labels = list(predictor.labels)
    pred = np.array(labels)[proba.argmax(axis=1)]
    result = ModelResult(
        spec=spec,
        labels=labels,
        proba=proba,
        pred=pred,
        metrics=compute_metrics(y_true.tolist(), pred.tolist(), labels),
        inference_ms_per_example=1000 * t_batch["seconds"] / len(texts),
        latency_ms_single=1000 * float(np.median(latencies)),
        metadata=meta,
    )
    result.metadata["_errors"] = pred != y_true
    logger.info("%-32s macro F1 %.4f", spec.name, result.metrics["macro_f1"])
    return result


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_confusion_matrices(results: list[ModelResult], y_true: np.ndarray, path: Path) -> Path:
    """One normalized confusion matrix per model, counts annotated."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.2))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results, strict=True):
        cm = confusion(y_true, r.pred, r.labels)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if share[i, j] > 0.6 else "black"
                label = f"{cm[i, j]}\n{share[i, j]:.0%}"
                ax.text(j, i, label, ha="center", va="center", color=color)
        ax.set_xticks(range(len(r.labels)), r.labels)
        ax.set_yticks(range(len(r.labels)), r.labels)
        ax.set(title=r.spec.name, xlabel="Predicted", ylabel="Gold" if ax is axes[0] else "")
        ax.title.set_fontsize(9)
    fig.tight_layout()
    return _save(fig, path)


def plot_f1_intervals(summary: pd.DataFrame, path: Path) -> Path:
    """Macro F1 with 95% bootstrap intervals, best model highlighted."""
    import matplotlib.pyplot as plt

    data = summary.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(data) + 1.2))
    best = data["macro_f1"].idxmax()
    colors = ["#2471a3" if i == best else "#7f8c8d" for i in data.index]
    ax.errorbar(
        data["macro_f1"],
        data["name"],
        xerr=[data["macro_f1"] - data["ci_low"], data["ci_high"] - data["macro_f1"]],
        fmt="none",
        ecolor="#555555",
        capsize=4,
    )
    ax.scatter(data["macro_f1"], data["name"], color=colors, zorder=3, s=40)
    for _, row in data.iterrows():
        ax.annotate(
            f"{row['macro_f1']:.3f}",
            (row["macro_f1"], row["name"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set(title="Test macro F1 with 95% bootstrap intervals", xlabel="Macro F1")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_speed_quality(summary: pd.DataFrame, path: Path) -> Path:
    """Macro F1 against single-text latency (log scale); marker area ~ model size."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sizes = 30 + 300 * np.sqrt(summary["size_mb"] / summary["size_mb"].max())
    ax.scatter(
        summary["latency_ms_single"], summary["macro_f1"], s=sizes, color="#2471a3", alpha=0.6
    )
    # Models with near-identical scores (e.g. LR and SVM) get staggered labels.
    ordered = summary.sort_values(["macro_f1", "latency_ms_single"])
    previous = None
    shift = 0
    for _, row in ordered.iterrows():
        close = previous is not None and abs(row["macro_f1"] - previous) < 0.01
        shift = shift + 14 if close else 0
        previous = row["macro_f1"]
        ax.annotate(
            f"{row['name']}\n{row['size_mb']:.1f} MB",
            (row["latency_ms_single"], row["macro_f1"]),
            xytext=(6, -4 + shift),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set(
        title="Quality vs. single-text latency (CPU; marker area ~ size)",
        xlabel="Latency per text (ms, log scale)",
        ylabel="Test macro F1",
    )
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, path)


def plot_error_tags(rates: pd.DataFrame, path: Path) -> Path:
    """Error rate on tagged vs. untagged reviews per tag and model."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 0.45 * rates["tag"].nunique() * 2 + 1))
    sns.barplot(data=rates, y="tag", x="error_tagged", hue="model", ax=ax)
    ax.set(
        title="Error rate on reviews with each property (test set)",
        xlabel="Error rate (share misclassified)",
        ylabel="",
    )
    ax.legend(fontsize=7, title="")
    fig.tight_layout()
    return _save(fig, path)


def _save(fig: Any, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
MANUAL_CATEGORIES: dict[str, str] = {
    "negation": "Sentiment flipped or softened by negation",
    "irony": "Irony or sarcasm; literal wording opposite to the verdict",
    "slang": "Slang, informal or colloquial wording",
    "mixed_sentiment": "Weighs strong pros and cons; verdict hinges on the balance",
    "short_text": "Too little text to judge",
    "plot_summary": "Mostly plot retelling; evaluative language is sparse",
    "verdict_truncated": "Verdict sits in the middle of a long review, outside what the models see",
    "rare_vocabulary": "Verdict phrased in rare, elaborate or academic vocabulary",
    "label_ambiguous": "Reasonable readers could disagree with the gold label",
}


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "n/a"
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def build_summary(results: list[ModelResult], boot: pd.DataFrame) -> pd.DataFrame:
    """One row per model with metrics, confidence interval and costs."""
    rows = []
    for r in results:
        low, high = confidence_interval(boot[r.spec.key].to_numpy())
        meta = r.metadata
        rows.append(
            {
                "key": r.spec.key,
                "name": r.spec.name,
                "macro_f1": r.metrics["macro_f1"],
                "ci_low": low,
                "ci_high": high,
                "accuracy": r.metrics["accuracy"],
                "macro_precision": r.metrics["macro_precision"],
                "macro_recall": r.metrics["macro_recall"],
                "train_seconds": meta.get("train_seconds"),
                "train_device": meta.get("device", "cpu"),
                "inference_ms_per_example": r.inference_ms_per_example,
                "latency_ms_single": r.latency_ms_single,
                "size_mb": meta.get("model_size_mb", float("nan")),
            }
        )
    return pd.DataFrame(rows).set_index("key", drop=False)


def pairwise_vs_best(
    results: list[ModelResult], boot: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    """Compare each model with the best one: bootstrap difference and McNemar test."""
    best_key = summary["macro_f1"].idxmax()
    best = next(r for r in results if r.spec.key == best_key)
    rows = []
    for r in results:
        if r.spec.key == best_key:
            continue
        diff = boot[best_key].to_numpy() - boot[r.spec.key].to_numpy()
        low, high = confidence_interval(diff)
        rows.append(
            {
                "model": r.spec.name,
                "diff": summary.at[best_key, "macro_f1"] - summary.at[r.spec.key, "macro_f1"],
                "diff_low": low,
                "diff_high": high,
                "p_not_better": float((diff <= 0).mean()),
                "mcnemar_p": mcnemar_p(best.errors, r.errors),
                "best_only_right": int((~best.errors & r.errors).sum()),
                "other_only_right": int((best.errors & ~r.errors).sum()),
            }
        )
    return pd.DataFrame(rows)


def review_candidates(
    test: pd.DataFrame, results: list[ModelResult], tags: pd.DataFrame, k: int
) -> pd.DataFrame:
    """Misclassified test reviews for manual analysis, hardest first.

    Ranked by how many models got the review wrong, then by how confident the
    strongest model was in its wrong answer.
    """
    from sentiment.evaluation.errors import snippet

    errors = np.vstack([r.errors for r in results])
    n_wrong = errors.sum(axis=0)
    strongest = max(results, key=lambda r: r.metrics["macro_f1"])
    conf = strongest.proba.max(axis=1)
    frame = pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "label": test["label"].to_numpy(),
            "n_models_wrong": n_wrong,
            "strongest_confidence": conf,
            "words": test["text"].str.split().str.len().to_numpy(),
        }
    )
    for r in results:
        pos = r.labels.index("positive") if "positive" in r.labels else -1
        frame[f"{r.spec.key}_pred"] = r.pred
        frame[f"{r.spec.key}_p_positive"] = r.proba[:, pos].round(3)
    tag_cols = tags.columns
    frame["tags"] = tags.apply(
        lambda row: ";".join(c for c in tag_cols if row[c]), axis=1
    ).to_numpy()
    frame["snippet"] = test["text"].map(snippet).to_numpy()
    wrong = frame[frame["n_models_wrong"] > 0]
    return wrong.sort_values(
        ["n_models_wrong", "strongest_confidence"], ascending=[False, False]
    ).head(k)


def load_manual_review(path: Path) -> pd.DataFrame | None:
    """Load ``manual_review.csv`` (id, categories, notes) if it exists."""
    if not path.exists():
        return None
    review = pd.read_csv(path)
    unknown = {
        c.strip()
        for cats in review["categories"].dropna()
        for c in str(cats).split(";")
        if c.strip() and c.strip() not in MANUAL_CATEGORIES
    }
    if unknown:
        raise ValueError(f"Unknown manual categories in {path}: {sorted(unknown)}")
    return review


def manual_category_table(
    review: pd.DataFrame, results: list[ModelResult], test: pd.DataFrame
) -> pd.DataFrame:
    """Count manual categories overall and among each model's errors."""
    id_to_pos = {int(i): p for p, i in enumerate(test["id"])}
    rows = []
    for cat, description in MANUAL_CATEGORIES.items():
        ids = [
            int(i)
            for i, cats in zip(review["id"], review["categories"].fillna(""), strict=True)
            if cat in [c.strip() for c in str(cats).split(";")]
        ]
        row: dict[str, Any] = {"category": cat, "description": description, "reviewed": len(ids)}
        for r in results:
            row[r.spec.name] = int(sum(r.errors[id_to_pos[i]] for i in ids if i in id_to_pos))
        rows.append(row)
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def results_table(summary: pd.DataFrame) -> str:
    """The README/comparison results table (format fixed by the spec)."""
    best = summary["macro_f1"].idxmax()
    rows = []
    for key, s in summary.iterrows():
        f1 = f"{s['macro_f1']:.3f} [{s['ci_low']:.3f}–{s['ci_high']:.3f}]"
        if key == best:
            f1 = f"**{f1}**"
        device = "" if s["train_device"] == "cpu" else f" ({s['train_device']})"
        if str(s["train_device"]).startswith("cpu"):
            device = " (CPU)"
        rows.append(
            {
                "Model": s["name"],
                "Macro F1 [95% CI]": f1,
                "Accuracy": f"{s['accuracy']:.3f}",
                "Training time": _fmt_seconds(s["train_seconds"]) + device,
                "Inference per example": f"{s['latency_ms_single']:.1f} ms",
                "Model size": f"{s['size_mb']:.1f} MB",
            }
        )
    return _table(pd.DataFrame(rows))


def render_comparison(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    rates: pd.DataFrame,
    thresholds: Any,
    manual: pd.DataFrame | None,
    n_reviewed: int,
    n_test: int,
    figures: dict[str, Path],
    report_path: Path,
    n_bootstrap: int,
    reviewers: str = "",
) -> str:
    """Markdown comparison report; every number is computed by the caller's code."""
    from sentiment.evaluation.errors import TAG_DESCRIPTIONS

    def rel(name: str) -> str:
        return figures[name].relative_to(report_path.parent).as_posix()

    best = summary.loc[summary["macro_f1"].idxmax()]
    pw = pairwise.copy()
    pw_rows = pd.DataFrame(
        {
            "Compared with": pw["model"],
            f"{best['name']} − model (macro F1)": [
                f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                for d, lo, hi in zip(pw["diff"], pw["diff_low"], pw["diff_high"], strict=True)
            ],
            "P(not better)": pw["p_not_better"].map(lambda p: f"{p:.3f}"),
            "McNemar p": pw["mcnemar_p"].map(lambda p: f"{p:.3f}"),
            "Only best right / only other right": [
                f"{a} / {b}"
                for a, b in zip(pw["best_only_right"], pw["other_only_right"], strict=True)
            ],
        }
    )

    detail = summary[
        ["name", "accuracy", "macro_precision", "macro_recall", "inference_ms_per_example"]
    ].copy()
    detail.columns = ["Model", "Accuracy", "Macro precision", "Macro recall", "Batch ms/example"]
    for c in ["Accuracy", "Macro precision", "Macro recall"]:
        detail[c] = detail[c].map(lambda v: f"{v:.3f}")
    detail["Batch ms/example"] = detail["Batch ms/example"].map(lambda v: f"{v:.2f}")

    tag_table = rates.pivot_table(index="tag", columns="model", values="error_tagged", sort=False)
    counts = rates.groupby("tag", sort=False)["n_tagged"].first()
    tag_rows = pd.DataFrame(
        {
            "Tag": [f"`{t}`" for t in tag_table.index],
            "Definition": [TAG_DESCRIPTIONS[t] for t in tag_table.index],
            "Reviews": [int(counts[t]) for t in tag_table.index],
            **{m: tag_table[m].map(lambda v: f"{v:.0%}").to_numpy() for m in tag_table.columns},
        }
    )
    overall = {r["name"]: 1 - r["accuracy"] for _, r in summary.iterrows()}
    overall_row = pd.DataFrame(
        [{"Tag": "**all reviews**", "Definition": "", "Reviews": n_test}
         | {m: f"{overall[m]:.0%}" for m in tag_table.columns}]
    )  # fmt: skip
    tag_rows = pd.concat([overall_row, tag_rows], ignore_index=True)

    blocks = [
        "# Model comparison on the test set",
        "_Generated by `scripts/evaluate.py` — do not edit by hand._",
        f"All models are evaluated on the same {n_test} held-out test reviews, loaded through "
        "the common `Predictor` interface, with the same metric code. Latency is measured on "
        "the same CPU for every model (median of single-text calls). Intervals are 95% "
        f"percentile bootstrap intervals from {n_bootstrap:,} resamples of the test set, "
        "shared by all models (paired).",
        "## Results",
        results_table(summary),
        f"![Macro F1 with confidence intervals]({rel('f1_intervals')})",
        "## Is the best model really better?",
        f"Paired comparison against **{best['name']}**: the bootstrap interval of the "
        "difference in macro F1 (same resamples for both models), the share of resamples in "
        "which the best model is not ahead, and McNemar's exact test on the reviews where "
        "exactly one of the two models is right.",
        _table(pw_rows),
        "## Detailed metrics",
        _table(detail),
        f"![Confusion matrices]({rel('confusion')})",
        f"![Quality vs. latency]({rel('speed_quality')})",
        "## Error analysis: heuristic tags",
        "Each tag is a measurable property; thresholds are quartiles of the **training** "
        f"split (negation > {thresholds.negation:.2f} per 100 words, contrast > "
        f"{thresholds.contrast:.2f}, questions > {thresholds.questions:.2f}, exclamations > "
        f"{thresholds.exclamations:.2f}, short ≤ {thresholds.short_words:.0f} words, long > "
        f"{thresholds.long_words:.0f} words). The table shows each model's error rate on test "
        "reviews with the tag.",
        _table(tag_rows),
        f"![Error rate by tag]({rel('error_tags')})",
    ]
    blocks.append("## Error analysis: manual review")
    if manual is None:
        blocks.append(
            f"Candidates for manual review ({n_reviewed} misclassified reviews, hardest first) "
            "are in `reports/error_analysis/errors_for_review.md`. Add categories to "
            "`reports/error_analysis/manual_review.csv` and rerun `scripts/evaluate.py`."
        )
    else:
        man = manual.copy()
        reviewer_note = f" Reviewer: {reviewers}." if reviewers else ""
        blocks.append(
            f"{n_reviewed} misclassified test reviews were read and categorized (a review can "
            "have several categories). Columns per model count how many of that model's "
            "errors fall into each category. Per-review notes: "
            f"`reports/error_analysis/manual_review.csv`.{reviewer_note}"
        )
        blocks.append(_table(man))
    return "\n\n".join(blocks) + "\n"


def update_readme(readme: Path, table: str) -> bool:
    """Replace the block between the RESULTS markers in the README with ``table``."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    start = text.index(README_START) + len(README_START)
    end = text.index(README_END)
    readme.write_text(text[:start] + "\n" + table + "\n" + text[end:], encoding="utf-8")
    return True


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Evaluate all configured models and write every report artifact.

    Returns:
        Dict with the summary table, pairwise comparison and output paths.
    """
    from sentiment.evaluation.errors import error_rates_by_tag, fit_thresholds, tag_texts

    test = pd.read_parquet(config.test)
    train = pd.read_parquet(config.train)
    texts, y_true = test["text"].tolist(), test["label"].to_numpy()

    results = [evaluate_model(s, texts, y_true, config.latency_samples) for s in config.models]
    labels = results[0].labels
    boot = bootstrap_macro_f1(
        y_true, {r.spec.key: r.pred for r in results}, labels, config.n_bootstrap, config.seed
    )
    summary = build_summary(results, boot)
    pairwise = pairwise_vs_best(results, boot, summary)

    lexicon = None
    if config.lexicon and config.lexicon.exists():
        words = pd.read_csv(config.lexicon)
        lexicon = {
            lab: grp["word"].head(config.lexicon_top_k).tolist()
            for lab, grp in words.groupby("label")
        }
    thresholds = fit_thresholds(train["text"])
    tags = tag_texts(test["text"], test["label"], thresholds, lexicon)
    rates = error_rates_by_tag(tags, {r.spec.name: r.errors for r in results})

    out = config.output_dir
    err_dir = config.comparison_path.parent / "error_analysis"
    figures = {
        "confusion": plot_confusion_matrices(results, y_true, out / "confusion_matrices.png"),
        "f1_intervals": plot_f1_intervals(summary, out / "f1_intervals.png"),
        "speed_quality": plot_speed_quality(summary, out / "speed_quality.png"),
        "error_tags": plot_error_tags(rates, out / "error_tags.png"),
    }

    predictions = pd.DataFrame({"id": test["id"], "label": y_true})
    for r in results:
        predictions[f"{r.spec.key}_pred"] = r.pred
        pos = r.labels.index("positive") if "positive" in r.labels else -1
        predictions[f"{r.spec.key}_p_positive"] = r.proba[:, pos].round(4)
    predictions = pd.concat([predictions, tags.add_prefix("tag_")], axis=1)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pairwise.to_csv(out / "pairwise.csv", index=False)
    rates.to_csv(out / "error_rates_by_tag.csv", index=False)

    candidates = review_candidates(test, results, tags, config.review_sample)
    err_dir.mkdir(parents=True, exist_ok=True)
    candidates.drop(columns="snippet").to_csv(err_dir / "errors_for_review.csv", index=False)
    (err_dir / "errors_for_review.md").write_text(
        render_review_candidates(candidates, results), encoding="utf-8"
    )
    manual = load_manual_review(err_dir / "manual_review.csv")
    manual_table = manual_category_table(manual, results, test) if manual is not None else None
    n_reviewed = len(manual) if manual is not None else len(candidates)
    reviewers = ""
    if manual is not None and "reviewer" in manual:
        reviewers = "; ".join(sorted(manual["reviewer"].dropna().unique()))

    config.comparison_path.write_text(
        render_comparison(
            summary, pairwise, rates, thresholds, manual_table, n_reviewed, len(test),
            figures, config.comparison_path, config.n_bootstrap, reviewers,
        ),
        encoding="utf-8",
    )  # fmt: skip
    readme_updated = bool(config.readme_path) and update_readme(
        config.readme_path, results_table(summary)
    )
    _log_to_mlflow(config, summary, figures)
    return {
        "summary": summary,
        "pairwise": pairwise,
        "rates": rates,
        "readme_updated": readme_updated,
        "comparison": config.comparison_path,
    }


def render_review_candidates(candidates: pd.DataFrame, results: list[ModelResult]) -> str:
    """Markdown list of misclassified reviews for manual reading."""
    lines = [
        "# Misclassified test reviews for manual analysis",
        "",
        "_Generated by `scripts/evaluate.py`. Ranked by how many models are wrong, then by the "
        "strongest model's confidence. Categories go into `manual_review.csv`._",
        "",
        "Allowed categories: " + ", ".join(f"`{c}`" for c in MANUAL_CATEGORIES) + ".",
        "",
    ]
    for _, row in candidates.iterrows():
        preds = ", ".join(
            f"{r.spec.name}: {row[f'{r.spec.key}_pred']} (p+={row[f'{r.spec.key}_p_positive']:.2f})"
            for r in results
        )
        lines += [
            f"## id {row['id']} — gold **{row['label']}**, {row['n_models_wrong']} of "
            f"{len(results)} models wrong, {row['words']} words",
            "",
            f"*Predictions:* {preds}  ",
            f"*Heuristic tags:* {row['tags'] or '—'}",
            "",
            f"> {row['snippet']}",
            "",
        ]
    return "\n".join(lines)


def _log_to_mlflow(config: EvalConfig, summary: pd.DataFrame, figures: dict[str, Path]) -> None:
    from sentiment.tracking import mlflow
    from sentiment.utils import RunConfig

    run_cfg = RunConfig(
        model="evaluation",
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
        mlflow_artifact_root=config.mlflow_artifact_root,
    )
    from sentiment.tracking import setup_mlflow

    setup_mlflow(run_cfg)
    with mlflow.start_run(run_name="evaluation-test"):
        mlflow.log_params({"n_bootstrap": config.n_bootstrap, "seed": config.seed})
        for key, row in summary.iterrows():
            mlflow.log_metrics(
                {
                    f"test_macro_f1.{key}": row["macro_f1"],
                    f"test_macro_f1_ci_low.{key}": row["ci_low"],
                    f"test_macro_f1_ci_high.{key}": row["ci_high"],
                    f"test_accuracy.{key}": row["accuracy"],
                    f"latency_ms_single.{key}": row["latency_ms_single"],
                }
            )
        mlflow.log_artifact(str(config.comparison_path))
        for path in figures.values():
            mlflow.log_artifact(str(path), artifact_path="figures")
