"""Common prediction interface shared by all models.

Evaluation and the demo app talk to models only through :class:`Predictor`, so all
three models are compared under identical conditions and adding a new model does
not require changes elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Predictor(ABC):
    """Abstract base class for a trained sentiment model.

    Subclasses must set :attr:`labels` and implement :meth:`predict_proba`.
    """

    #: Class labels, ordered to match the columns returned by ``predict_proba``.
    labels: list[str]

    #: Human-readable model name used in reports and the demo.
    name: str = "model"

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probabilities.

        Args:
            texts: Raw input texts.

        Returns:
            Array of shape ``(len(texts), len(self.labels))`` whose rows sum to 1.
        """

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Return the most probable label for each text.

        Args:
            texts: Raw input texts.

        Returns:
            One label per input text.
        """
        if len(texts) == 0:
            return []
        proba = self.predict_proba(texts)
        return [self.labels[i] for i in np.argmax(proba, axis=1)]
