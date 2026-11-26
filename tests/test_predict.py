from collections.abc import Sequence

import numpy as np
import pytest

from sentiment.predict import Predictor


class LengthPredictor(Predictor):
    """Toy model: long texts are positive."""

    labels = ["negative", "positive"]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        p = np.array([1.0 if len(t) > 5 else 0.0 for t in texts])
        return np.column_stack([1 - p, p])


def test_predict_uses_argmax_of_proba():
    model = LengthPredictor()
    assert model.predict(["loše", "odlično iskustvo"]) == ["negative", "positive"]


def test_predict_empty_input():
    assert LengthPredictor().predict([]) == []


def test_predictor_is_abstract():
    with pytest.raises(TypeError):
        Predictor()  # type: ignore[abstract]
