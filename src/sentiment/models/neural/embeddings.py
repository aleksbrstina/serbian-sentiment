"""Load pretrained fastText vectors (``cc.sr.300.vec.gz``) for a vocabulary.

The Common Crawl Serbian vectors mix Cyrillic and Latin entries, while our data is
normalized to Latin script. Each fastText word is therefore lower-cased and
transliterated before lookup. The file is sorted by frequency, so when several
entries map to the same key (e.g. ``Филм`` and ``film``) the most frequent one wins.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentiment.data.cleaning import cyrillic_to_latin
from sentiment.models.neural.tokenizer import PAD_ID, Vocabulary

FASTTEXT_INSTRUCTIONS = """\
fastText vectors not found: {path}

Download the Serbian Common Crawl vectors (text format, ~1.2 GB):
  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sr.300.vec.gz
and place the file at {path} (keep it gzipped).

License: CC BY-SA 3.0. Grave et al. (2018), Learning Word Vectors for 157 Languages.
Alternatively set `embedding_path: null` in configs/bilstm.yaml for random init.
"""


class EmbeddingsMissingError(FileNotFoundError):
    """Raised when the fastText file is absent; the message explains how to get it."""


@dataclass
class EmbeddingMatrix:
    """Embedding weights aligned with a vocabulary plus coverage statistics."""

    weights: np.ndarray
    found_ids: set[int] = field(default_factory=set)

    @property
    def found(self) -> int:
        """Number of vocabulary types initialized from pretrained vectors."""
        return len(self.found_ids)

    @property
    def coverage_types(self) -> float:
        """Share of regular (non-special) vocabulary types with a pretrained vector."""
        return self.found / max(len(self.weights) - 2, 1)


def _vocab_hash(vocab: Vocabulary, path: Path, transliterate: bool) -> str:
    h = hashlib.sha256()
    h.update("\n".join(vocab.itos).encode())
    h.update(f"{path.name}|{transliterate}".encode())
    return h.hexdigest()[:16]


def read_fasttext(
    path: str | Path, vocab: Vocabulary, transliterate: bool = True
) -> dict[int, np.ndarray]:
    """Stream a ``.vec(.gz)`` file and return vectors for vocabulary ids.

    Only lines whose (normalized) word is in the vocabulary are parsed, so a 2M-word
    file is read in about a minute without loading it into memory.

    Args:
        path: Path to ``cc.sr.300.vec.gz`` (or an uncompressed ``.vec``).
        vocab: Target vocabulary.
        transliterate: Lower-case and transliterate Cyrillic fastText words to Latin.

    Returns:
        Mapping vocabulary id -> vector (first, i.e. most frequent, match wins).

    Raises:
        EmbeddingsMissingError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise EmbeddingsMissingError(FASTTEXT_INSTRUCTIONS.format(path=path))
    opener = gzip.open if path.suffix == ".gz" else open
    vectors: dict[int, np.ndarray] = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().split()
        dim = int(header[1]) if len(header) == 2 else len(header) - 1
        for line in f:
            word, _, rest = line.rstrip("\n").partition(" ")
            key = cyrillic_to_latin(word.lower()) if transliterate else word
            idx = vocab.stoi.get(key)
            if idx is None or idx in vectors:
                continue
            vec = np.asarray(rest.split(" "), dtype=np.float32)
            if vec.shape[0] == dim:
                vectors[idx] = vec
    return vectors


def build_embedding_matrix(
    vocab: Vocabulary,
    path: str | Path | None,
    dim: int = 300,
    transliterate: bool = True,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> EmbeddingMatrix:
    """Create an embedding matrix for ``vocab``, initialized from fastText where possible.

    Words without a pretrained vector are drawn from a normal distribution with the
    same standard deviation as the pretrained vectors, so they start at a comparable
    scale; ``<pad>`` is all zeros.

    Args:
        vocab: Vocabulary.
        path: fastText file, or None for fully random initialization.
        dim: Embedding dimension (must match the file).
        transliterate: See :func:`read_fasttext`.
        seed: Seed for the random rows.
        cache_dir: If given, the matched rows are cached as ``.npz`` keyed by the
            vocabulary, so repeated runs skip reading the 1.2 GB file.

    Returns:
        Embedding matrix and how many vocabulary types were found.
    """
    rng = np.random.default_rng(seed)
    if path is None:
        weights = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
        weights[PAD_ID] = 0
        return EmbeddingMatrix(weights)

    path = Path(path)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"fasttext_{_vocab_hash(vocab, path, transliterate)}.npz"
    if cache is not None and cache.exists():
        data = np.load(cache)
        vectors = dict(zip(data["ids"].tolist(), data["vectors"], strict=True))
    else:
        vectors = read_fasttext(path, vocab, transliterate)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            ids = np.array(sorted(vectors), dtype=np.int64)
            stacked = np.stack([vectors[i] for i in ids]) if len(ids) else np.zeros((0, dim))
            np.savez_compressed(cache, ids=ids, vectors=stacked.astype(np.float32))

    found = np.stack(list(vectors.values())) if vectors else np.zeros((0, dim), np.float32)
    std = float(found.std()) if len(found) else 0.1
    weights = rng.normal(0, std, (len(vocab), dim)).astype(np.float32)
    for idx, vec in vectors.items():
        weights[idx] = vec
    weights[PAD_ID] = 0
    return EmbeddingMatrix(weights, found_ids=set(vectors))
