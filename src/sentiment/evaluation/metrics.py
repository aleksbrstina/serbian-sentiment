"""Classification metrics shared by every model (FR-09)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, float]:
    """Compute accuracy and macro-averaged precision, recall and F1.

    Macro averaging weights every class equally, so it stays meaningful if a
    dataset variant (e.g. three-class) is imbalanced.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        labels: All class labels, in a fixed order.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, ``macro_precision``, ``macro_recall``
        and per-class ``f1_<label>``.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} gold vs {len(y_pred)} predicted")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty set")

    labels = list(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }
    metrics.update({f"f1_{lab}": float(v) for lab, v in zip(labels, per_class, strict=True)})
    return metrics


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Confusion matrix with rows = gold labels and columns = predictions, in ``labels`` order."""
    return confusion_matrix(y_true, y_pred, labels=list(labels))
