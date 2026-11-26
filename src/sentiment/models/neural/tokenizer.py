"""Word tokenizer and vocabulary for the BiLSTM.

The tokenizer is deliberately simple (regex, lower-case) because the pretrained
fastText vectors are keyed by lower-cased surface forms; a fancier tokenizer would
only reduce embedding coverage.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import regex

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# Words (letters, optionally joined by an apostrophe or hyphen), numbers, and the
# punctuation marks the EDA found to be class-indicative.
_TOKEN_RE = regex.compile(r"\p{L}+(?:['\-]\p{L}+)*|\p{N}+|[!?]")


def tokenize(text: str) -> list[str]:
    """Split text into lower-cased word, number and ``!``/``?`` tokens.

    Args:
        text: Normalized input text.

    Returns:
        List of tokens.
    """
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Bidirectional token <-> id mapping with ``<pad>`` = 0 and ``<unk>`` = 1."""

    def __init__(self, tokens: Sequence[str]) -> None:
        """Create a vocabulary from an ordered token list (specials are prepended)."""
        specials = [PAD, UNK]
        self.itos: list[str] = specials + [t for t in tokens if t not in specials]
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(
        cls, texts: Iterable[str], min_freq: int = 2, max_size: int | None = None
    ) -> Vocabulary:
        """Build a vocabulary from training texts.

        Args:
            texts: Training texts.
            min_freq: Minimum count for a token to get its own id; rarer tokens map
                to ``<unk>``.
            max_size: Optional cap on the number of non-special tokens (most frequent
                kept).

        Returns:
            Vocabulary ordered by descending frequency (ties broken alphabetically,
            so the result is deterministic).
        """
        counts = Counter(tok for text in texts for tok in tokenize(text))
        ranked = sorted(
            (t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t)
        )
        return cls(ranked[:max_size] if max_size else ranked)

    def __len__(self) -> int:
        """Number of entries including special tokens."""
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        """Whether ``token`` has its own id."""
        return token in self.stoi

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map tokens to ids, using ``<unk>`` for unknown tokens."""
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def save(self, path: str | Path) -> None:
        """Write the vocabulary as a JSON list of tokens."""
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        """Load a vocabulary saved with :meth:`save`."""
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        if itos[:2] != [PAD, UNK]:
            raise ValueError(f"{path} does not start with the special tokens {PAD}, {UNK}")
        return cls(itos[2:])
