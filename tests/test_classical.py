import json

import numpy as np
import pandas as pd
import pytest

from sentiment.models.classical import (
    ClassicalParams,
    ClassicalPredictor,
    FeatureCache,
    VectorizerParams,
    build_classifier,
    build_features,
    class_weights,
    search,
    top_features,
    train_classical,
)
from sentiment.predict import load_predictor
from sentiment.tracking import flatten_params
from sentiment.utils import RunConfig

POS = ["odličan", "sjajan", "remek", "savršeno", "genijalan"]
NEG = ["loš", "dosadan", "očajno", "nažalost", "traljavo"]
FILLER = ["film", "glumac", "scena", "priča", "reditelj", "kamera", "muzika"]


def _make(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = "positive" if i % 2 else "negative"
        cue = POS if label == "positive" else NEG
        words = list(rng.choice(FILLER, 8)) + list(rng.choice(cue, 2))
        rng.shuffle(words)
        rows.append({"id": i, "text": " ".join(words), "label": label, "source": "toy"})
    return pd.DataFrame(rows)


@pytest.fixture
def splits():
    return _make(80, 0), _make(30, 1)


VEC = VectorizerParams(word_ngram_max=2, char_ngrams="2-4", min_df=1, sublinear_tf=True)


def _logreg_pipeline():
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [("features", build_features(VEC)), ("clf", build_classifier("logreg", {"C": 10}, 0))]
    )


@pytest.mark.parametrize("name", ["logreg", "linear_svm", "naive_bayes"])
def test_each_classifier_learns_toy_task(splits, name):
    train, val = splits
    from sklearn.pipeline import Pipeline

    hp = {"C": 1.0} if name != "naive_bayes" else {"alpha": 0.1}
    pipe = Pipeline(
        [("features", build_features(VEC)), ("clf", build_classifier(name, hp, 0, calibrate=True))]
    )
    pipe.fit(train["text"], train["label"])
    predictor = ClassicalPredictor(pipe)
    proba = predictor.predict_proba(val["text"].tolist())
    assert proba.shape == (len(val), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    accuracy = np.mean(np.array(predictor.predict(val["text"].tolist())) == val["label"].to_numpy())
    assert accuracy > 0.9


def test_class_weights_point_the_right_way(splits):
    train, _ = splits

    pipe = _logreg_pipeline()
    pipe.fit(train["text"], train["label"])
    w = class_weights(pipe.named_steps["clf"])
    assert w.shape[0] == 2
    np.testing.assert_allclose(w[0], -w[1])
    top = top_features(pipe, k=5, kind="word")
    assert set(top[top["label"] == "positive"]["feature"]) & set(POS)
    assert set(top[top["label"] == "negative"]["feature"]) & set(NEG)
    assert (top["kind"] == "word").all()
    assert top.groupby("label").size().tolist() == [5, 5]


def test_feature_cache_reuses_matrices(splits):
    train, val = splits
    cache = FeatureCache(train["text"], val["text"])
    a = cache.get(VEC)
    b = cache.get(VEC)
    assert a[0].shape == b[0].shape
    assert len(cache._word) == 1 and len(cache._char) == 1
    word_only = VectorizerParams(word_ngram_max=2, char_ngrams="none", min_df=1, sublinear_tf=True)
    x_train, _ = cache.get(word_only)
    assert len(cache._word) == 1  # word block reused
    assert x_train.shape[1] < a[0].shape[1]


def test_search_is_deterministic(splits):
    train, val = splits
    params = ClassicalParams(n_trials=4)
    args = (train["label"].to_numpy(), val["label"].to_numpy(), params, 7)
    r1 = search("logreg", FeatureCache(train["text"], val["text"]), *args)
    r2 = search("logreg", FeatureCache(train["text"], val["text"]), *args)
    assert r1.hyperparams == r2.hyperparams
    assert r1.vectorizer == r2.vectorizer
    assert len(r1.trials) == 4


def test_explain_highlights_cue_words(splits):
    train, _ = splits

    pipe = _logreg_pipeline()
    pipe.fit(train["text"], train["label"])
    explanation = ClassicalPredictor(pipe).explain("film odličan sjajan scena", k=3)
    words = [w for w, _ in explanation]
    assert "odličan" in words or "sjajan" in words
    assert explanation[0][1] > 0  # top feature supports the prediction


def test_train_classical_end_to_end(splits, tmp_path):
    train, val = splits
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_parquet(data_dir / "train.parquet")
    val.to_parquet(data_dir / "val.parquet")
    config = RunConfig(
        model="classical",
        output_dir=tmp_path / "artifacts",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_artifact_root=tmp_path / "mlruns",
        data={"train": data_dir / "train.parquet", "val": data_dir / "val.parquet"},
        params={"n_trials": 2, "top_k": 5, "latency_samples": 3},
    )
    summary = train_classical(config, report_dir=tmp_path / "reports")

    assert set(summary["models"]) == {"logreg", "linear_svm", "naive_bayes"}
    best_dir = tmp_path / "artifacts" / "best"
    meta = json.loads((best_dir / "metadata.json").read_text())
    assert meta["classifier"] == summary["best"]
    for key in ("val_metrics", "train_seconds", "inference_ms_per_example", "model_size_mb"):
        assert key in meta
    assert (tmp_path / "artifacts" / "config.yaml").exists()

    predictor = load_predictor(best_dir)
    assert predictor.predict(["odličan sjajan film"]) == ["positive"]

    report = (tmp_path / "reports" / "validation.md").read_text()
    assert "**(best)**" in report
    top_words = pd.read_csv(tmp_path / "reports" / "top_words.csv")
    assert top_words.groupby("label").size().tolist() == [5, 5]

    import mlflow

    runs = mlflow.search_runs(experiment_names=["serbian-sentiment"])
    assert len(runs) == 4  # parent + one nested run per classifier
    assert runs["metrics.val_macro_f1"].notna().sum() == 3


def test_load_predictor_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictor(tmp_path)
    (tmp_path / "metadata.json").write_text('{"type": "unknown"}')
    with pytest.raises(ValueError, match="Unknown model type"):
        load_predictor(tmp_path)


def test_flatten_params():
    assert flatten_params({"a": 1, "b": {"c": [1, 2], "d": {"e": True}}}) == {
        "a": "1",
        "b.c": "[1, 2]",
        "b.d.e": "True",
    }


def test_near_log_boundary():
    from sentiment.models.classical import near_log_boundary

    assert near_log_boundary(9.4, 1e-3, 10)
    assert near_log_boundary(1.1e-3, 1e-3, 10)
    assert not near_log_boundary(1.0, 1e-3, 10)
