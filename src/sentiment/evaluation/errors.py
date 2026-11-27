"""Heuristic tagging of reviews for error analysis (FR-11).

Every tag is a measurable property of the text. Thresholds are computed on the
*training* split (e.g. "top quartile of negation density"), so the test set is
only ever labelled, never used to choose cut-offs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import regex

NEGATION_RE = regex.compile(
    r"\b(?:ne|nije|nisam|nisu|nismo|niste|nisi|nikad|nikada|nema)\b", regex.I
)
CONTRAST_RE = regex.compile(
    r"\b(?:ali|međutim|ipak|iako|mada|doduše|premda|nažalost|naprotiv)\b|s druge strane", regex.I
)
QUESTION_RE = regex.compile(r"\?")
EXCLAMATION_RE = regex.compile(r"!")
WORD_RE = regex.compile(r"\p{L}+")

TAG_DESCRIPTIONS: dict[str, str] = {
    "negation_dense": "Negation words per 100 words in the top quartile of training reviews",
    "contrast_dense": "Contrastive markers (ali, međutim, ipak, iako, ...) in the top quartile",
    "questions": "Question marks per 100 words in the top quartile (rhetorical questions)",
    "exclamations": "Exclamation marks per 100 words in the top quartile",
    "short": "Length in the bottom quartile of training reviews",
    "long": "Length in the top quartile of training reviews (most truncated by the models)",
    "contradicting_cues": "More class-indicative words of the opposite class than of its own",
}


def _per_100_words(texts: pd.Series, pattern: regex.Pattern[str]) -> pd.Series:
    words = texts.str.split().str.len().clip(lower=1)
    return texts.map(lambda t: len(pattern.findall(t))) / words * 100


@dataclass
class TagThresholds:
    """Cut-offs learned from the training split."""

    negation: float
    contrast: float
    questions: float
    exclamations: float
    short_words: float
    long_words: float


def fit_thresholds(train_texts: pd.Series, quantile: float = 0.75) -> TagThresholds:
    """Compute tag thresholds from training texts.

    Args:
        train_texts: Training-split texts.
        quantile: Upper quantile for "dense" tags; ``1 - quantile`` is used for short.

    Returns:
        Thresholds for :func:`tag_texts`.
    """
    words = train_texts.str.split().str.len()
    return TagThresholds(
        negation=float(_per_100_words(train_texts, NEGATION_RE).quantile(quantile)),
        contrast=float(_per_100_words(train_texts, CONTRAST_RE).quantile(quantile)),
        questions=float(_per_100_words(train_texts, QUESTION_RE).quantile(quantile)),
        exclamations=float(_per_100_words(train_texts, EXCLAMATION_RE).quantile(quantile)),
        short_words=float(words.quantile(1 - quantile)),
        long_words=float(words.quantile(quantile)),
    )


def cue_counts(texts: pd.Series, lexicon: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Count occurrences of each class's cue words in each text.

    Args:
        texts: Texts to scan.
        lexicon: Mapping label -> class-indicative words (e.g. top chi-square words).

    Returns:
        DataFrame with one ``cues_<label>`` column per label.
    """
    sets = {lab: {w.lower() for w in words} for lab, words in lexicon.items()}
    tokens = texts.map(lambda t: WORD_RE.findall(t.lower()))

    def count(tokens_of_text: list[str], cue_set: set[str]) -> int:
        return sum(t in cue_set for t in tokens_of_text)

    return pd.DataFrame(
        {f"cues_{lab}": tokens.map(lambda ts, s=s: count(ts, s)) for lab, s in sets.items()}
    )


def tag_texts(
    texts: pd.Series,
    labels: pd.Series,
    thresholds: TagThresholds,
    lexicon: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Assign boolean heuristic tags to each text.

    Args:
        texts: Texts (normalized, as the models see them).
        labels: Gold labels (needed for ``contradicting_cues``).
        thresholds: Output of :func:`fit_thresholds`.
        lexicon: Class cue words; without it ``contradicting_cues`` is omitted.

    Returns:
        DataFrame (same index as ``texts``) with one boolean column per tag.
    """
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    words = texts.str.split().str.len()
    tags = pd.DataFrame(
        {
            "negation_dense": _per_100_words(texts, NEGATION_RE) > thresholds.negation,
            "contrast_dense": _per_100_words(texts, CONTRAST_RE) > thresholds.contrast,
            "questions": _per_100_words(texts, QUESTION_RE) > thresholds.questions,
            "exclamations": _per_100_words(texts, EXCLAMATION_RE) > thresholds.exclamations,
            "short": words <= thresholds.short_words,
            "long": words > thresholds.long_words,
        }
    )
    if lexicon:
        cues = cue_counts(texts, lexicon)
        own = np.array([cues.at[i, f"cues_{lab}"] for i, lab in enumerate(labels)])
        other = cues.sum(axis=1).to_numpy() - own
        tags["contradicting_cues"] = other > own
    return tags


def error_rates_by_tag(tags: pd.DataFrame, errors: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Error rate of each model on texts with and without each tag.

    Args:
        tags: Output of :func:`tag_texts`.
        errors: Mapping model name -> boolean array (True = misclassified).

    Returns:
        Long DataFrame with ``tag``, ``n_tagged``, ``model``, ``error_tagged``,
        ``error_untagged`` and ``ratio`` (tagged / untagged error rate).
    """
    rows = []
    for tag in tags.columns:
        mask = tags[tag].to_numpy()
        for model, err in errors.items():
            tagged = float(err[mask].mean()) if mask.any() else float("nan")
            untagged = float(err[~mask].mean()) if (~mask).any() else float("nan")
            rows.append(
                {
                    "tag": tag,
                    "n_tagged": int(mask.sum()),
                    "model": model,
                    "error_tagged": tagged,
                    "error_untagged": untagged,
                    "ratio": tagged / untagged if untagged else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def snippet(text: str, head: int = 350, tail: int = 350) -> str:
    """Opening and closing part of a long review for manual reading."""
    if len(text) <= head + tail + 20:
        return text
    return f"{text[:head].rstrip()} […] {text[-tail:].lstrip()}"
