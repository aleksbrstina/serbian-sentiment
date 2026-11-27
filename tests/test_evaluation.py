from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from sentiment.evaluation.errors import (
    cue_counts,
    error_rates_by_tag,
    fit_thresholds,
    snippet,
    tag_texts,
)
from sentiment.evaluation.report import (
    MANUAL_CATEGORIES,
    ModelSpec,
    bootstrap_macro_f1,
    confidence_interval,
    load_manual_review,
    mcnemar_p,
    results_table,
    update_readme,
)
from sentiment.predict import Predictor

LABELS = ["negative", "positive"]


# --- heuristic tags -------------------------------------------------------------
def test_thresholds_come_from_training_quantiles():
    train = pd.Series(["a b c d"] * 3 + ["ne ne a b"])  # negation density 0,0,0,50 per 100
    th = fit_thresholds(train, quantile=0.75)
    assert th.negation == pytest.approx(12.5)
    assert th.short_words == 4 and th.long_words == 4


def test_tag_texts():
    train = pd.Series(["film je dobar i lep"] * 8)
    th = fit_thresholds(train)
    texts = pd.Series(["film nije dobar ne", "zašto? zašto?", "odličan loš loš", "a b c d e f g"])
    labels = pd.Series(["negative", "negative", "positive", "positive"])
    tags = tag_texts(texts, labels, th, lexicon={"positive": ["odličan"], "negative": ["loš"]})
    assert tags["negation_dense"].tolist() == [True, False, False, False]
    assert tags["questions"].tolist() == [False, True, False, False]
    assert tags["contradicting_cues"].tolist() == [False, False, True, False]
    assert tags["long"].tolist() == [False, False, False, True]


def test_cue_counts_case_insensitive():
    lexicon = {"positive": ["odličan"], "negative": ["Loš"]}
    counts = cue_counts(pd.Series(["Odličan i ODLIČAN, loš"]), lexicon)
    assert counts.iloc[0].to_dict() == {"cues_positive": 2, "cues_negative": 1}


def test_error_rates_by_tag():
    tags = pd.DataFrame({"t": [True, True, False, False]})
    rates = error_rates_by_tag(tags, {"m": np.array([True, False, False, False])})
    row = rates.iloc[0]
    assert (row["n_tagged"], row["error_tagged"], row["error_untagged"]) == (2, 0.5, 0.0)


def test_snippet():
    assert snippet("kratko") == "kratko"
    out = snippet("a" * 500 + "b" * 500, head=10, tail=10)
    assert out.startswith("a" * 10) and out.endswith("b" * 10) and "[…]" in out


# --- statistics -------------------------------------------------------------------
def test_bootstrap_is_paired_and_reproducible():
    y = np.array(["positive", "negative"] * 50)
    preds = {"perfect": y.copy(), "flipped": np.where(y == "positive", "negative", "positive")}
    boot = bootstrap_macro_f1(y, preds, LABELS, n_resamples=200, seed=1)
    assert boot.shape == (200, 2)
    assert (boot["perfect"] == 1.0).all() and (boot["flipped"] == 0.0).all()
    again = bootstrap_macro_f1(y, preds, LABELS, n_resamples=200, seed=1)
    pd.testing.assert_frame_equal(boot, again)


def test_confidence_interval_bounds():
    low, high = confidence_interval(np.arange(1001) / 1000)
    assert low == pytest.approx(0.025) and high == pytest.approx(0.975)


def test_mcnemar():
    a = np.array([False] * 20 + [True] * 2)  # a wrong on 2
    b = np.array([True] * 12 + [False] * 10)  # b wrong on 12
    assert mcnemar_p(a, b) < 0.05
    assert mcnemar_p(a, a) == 1.0


# --- reports ----------------------------------------------------------------------
def test_results_table_marks_best():
    summary = pd.DataFrame(
        {
            "name": ["A", "B"],
            "macro_f1": [0.8, 0.9],
            "ci_low": [0.7, 0.85],
            "ci_high": [0.85, 0.95],
            "accuracy": [0.8, 0.9],
            "train_seconds": [5.0, 600.0],
            "train_device": ["cpu", "cuda (Tesla T4)"],
            "latency_ms_single": [4.0, 100.0],
            "size_mb": [1.5, 443.0],
        },
        index=["a", "b"],
    )
    table = results_table(summary)
    assert "**0.900 [0.850–0.950]**" in table
    assert "5 s (CPU)" in table and "10.0 min (cuda (Tesla T4))" in table


def test_update_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n")
    assert update_readme(readme, "| new |")
    expected = "intro\n<!-- RESULTS:START -->\n| new |\n<!-- RESULTS:END -->\noutro\n"
    assert readme.read_text() == expected
    no_markers = tmp_path / "other.md"
    no_markers.write_text("nothing")
    assert not update_readme(no_markers, "x")


def test_manual_review_rejects_unknown_category(tmp_path):
    path = tmp_path / "review.csv"
    rows = {"id": [1, 2], "categories": ["irony;negation", "made_up"], "notes": ["", ""]}
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="made_up"):
        load_manual_review(path)
    assert load_manual_review(tmp_path / "missing.csv") is None
    assert {"negation", "irony", "slang", "mixed_sentiment", "short_text"} <= set(MANUAL_CATEGORIES)


def test_shipped_manual_review_is_valid():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "reports/error_analysis/manual_review.csv"
    if not path.exists():
        pytest.skip("manual review not generated")
    review = load_manual_review(path)
    assert len(review) >= 30  # spec: at least 30 manually reviewed examples
    assert review["id"].is_unique


def test_model_spec_rejects_unknown_fields():
    with pytest.raises(ValueError):
        ModelSpec(key="a", name="A", path="x", typo=1)


# --- demo app ---------------------------------------------------------------------
class FakePredictor(Predictor):
    labels = LABELS

    def __init__(self, p_pos: float) -> None:
        self.p_pos = p_pos

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([[1 - self.p_pos, self.p_pos] for _ in texts])

    def explain(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        return [("odličan", 0.5), ("loš", -0.25)]


def _app():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    spec = importlib.util.spec_from_file_location("demo_app", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered first
    spec.loader.exec_module(module)
    return module


def test_app_analyze_normalizes_and_scores():
    app = _app()
    models = [app.ModelSlot("classical", "TF-IDF", FakePredictor(0.8), "fake")]
    res = app.analyze("  Филм је одличан, али лош крај. ", models)
    assert res["normalized"] == "Film je odličan, ali loš kraj."
    out = res["models"]["classical"]
    assert out["label"] == "pozitivno" and out["probs"]["pozitivno"] == pytest.approx(0.8)
    scores = dict((w, s) for w, s in res["classical_highlights"] if s is not None)
    # Predicted positive: "odličan" pushes towards positive -> negative score (drawn blue)
    assert scores["odličan"] == pytest.approx(-1.0) and scores["loš"] == pytest.approx(0.5)
    assert "".join(w for w, _ in res["classical_highlights"]) == res["normalized"]
    assert app.analyze("   ", models) == {"empty": True}


def test_app_builds_interface():
    pytest.importorskip("gradio")
    app = _app()
    models = [app.ModelSlot("classical", "TF-IDF", FakePredictor(0.3), "fake")]
    demo = app.build_demo(models)
    assert type(demo).__name__ == "Blocks"
