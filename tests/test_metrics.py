import numpy as np
import pytest

from sentiment.evaluation.metrics import compute_metrics, confusion

LABELS = ["negative", "positive"]


def test_perfect_predictions():
    y = ["positive", "negative", "positive"]
    m = compute_metrics(y, y, LABELS)
    assert m["accuracy"] == m["macro_f1"] == m["macro_precision"] == m["macro_recall"] == 1.0


def test_known_values():
    # TP(pos)=2, FN(pos)=1, FP(pos)=1, TN=1
    y_true = ["positive", "positive", "positive", "negative", "negative"]
    y_pred = ["positive", "positive", "negative", "positive", "negative"]
    m = compute_metrics(y_true, y_pred, LABELS)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["f1_positive"] == pytest.approx(2 / 3)  # P=2/3, R=2/3
    assert m["f1_negative"] == pytest.approx(0.5)  # P=1/2, R=1/2
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert m["macro_precision"] == pytest.approx((2 / 3 + 0.5) / 2)


def test_macro_f1_differs_from_accuracy_on_imbalance():
    y_true = ["negative"] * 9 + ["positive"]
    y_pred = ["negative"] * 10
    m = compute_metrics(y_true, y_pred, LABELS)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["f1_positive"] == 0.0
    assert m["macro_f1"] == pytest.approx(0.9474 / 2, abs=1e-3)


def test_label_absent_from_predictions_does_not_crash():
    labels = ["negative", "neutral", "positive"]
    m = compute_metrics(["neutral", "positive"], ["positive", "positive"], labels)
    assert m["f1_neutral"] == 0.0
    assert m["f1_negative"] == 0.0


def test_metrics_validate_input():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_metrics(["positive"], [], LABELS)
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [], LABELS)


def test_confusion_matrix_orientation():
    y_true = ["positive", "positive", "negative"]
    y_pred = ["negative", "positive", "negative"]
    cm = confusion(y_true, y_pred, LABELS)
    # rows = gold, columns = predicted
    np.testing.assert_array_equal(cm, [[1, 0], [1, 1]])
