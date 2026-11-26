"""Descriptive statistics for exploratory data analysis.

All functions take DataFrames with ``text`` and ``label`` columns and return plain
pandas/Python objects so the same numbers feed the report, the plots and the notebook.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import regex
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from sentiment.data.loading import LABEL_ORDER

_WORD_RE = regex.compile(r"\p{L}+")

#: Small Serbian (Latin) stopword list, used only for the raw-frequency word lists.
#: The chi-square ranking does not need it because frequent function words are
#: equally common in both classes and get low scores anyway.
SERBIAN_STOPWORDS: frozenset[str] = frozenset(
    """
    a ali bi bih bila bile bili bilo bio biti bude da do dok ga gde i ih ili im ja je
    jer jedan jedna jedno joj još ju kad kada kako kao koja koje koji kojima koju kroz
    li me mi mu na nad nakon nam nas ne nego neki neko nešto ni nije nisam nisu niti
    njega njegov njegova njegove njegovo njen njena njih njihov njihova o od ova ove
    ovaj ovde ovo pa po pod pored posle pre preko pri s sa sam se si sve svi svoj svoje
    svojim ta taj tako takođe te tek ti tim to toga tom tu u uz vam već vi za zbog što
    šta čak će ću ćemo ćete ćeš da li samo još koliko onda ono ona oni one kojoj kojeg
    čiji film filma filmu filmom filmovi filmova
    """.split()  # noqa: SIM905
)

#: Lexical cues counted per class. Negation words follow the error-analysis
#: heuristics in the spec (FR-11).
CUE_PATTERNS: dict[str, str] = {
    "negation (ne/nije/nisam/nisu)": r"\b(?:ne|nije|nisam|nisu|nismo|niste)\b",
    "question mark": r"\?",
    "exclamation mark": r"!",
    "ellipsis": r"\.\.\.",
    "explicit score (e.g. 7/10)": r"\b\d{1,2}(?:[.,]\d)?\s*/\s*10\b",
}


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer keeping only letter sequences (Unicode-aware)."""
    return _WORD_RE.findall(text.lower())


def labels_in(df: pd.DataFrame) -> list[str]:
    """Return the labels present in ``df`` in canonical order."""
    present = set(df["label"])
    return [lab for lab in LABEL_ORDER if lab in present]


def class_distribution(splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Count records per class in each split.

    Args:
        splits: Mapping of split name to DataFrame.

    Returns:
        DataFrame indexed by split with one count column per label plus ``total``.
    """
    labels = labels_in(pd.concat(splits.values()))
    rows = {
        name: {lab: int((df["label"] == lab).sum()) for lab in labels}
        for name, df in splits.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")[labels]
    table["total"] = table.sum(axis=1)
    return table


def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``n_chars`` and ``n_words`` columns."""
    out = df.copy()
    out["n_chars"] = out["text"].str.len()
    out["n_words"] = out["text"].str.split().str.len()
    return out


@dataclass
class LengthComparison:
    """Per-class length summary and a test of whether lengths differ by class."""

    summary: pd.DataFrame
    test_name: str
    p_value: float


def length_stats(df: pd.DataFrame) -> LengthComparison:
    """Summarize word counts per class and test for a difference between classes.

    For two classes a two-sided Mann-Whitney U test is used: lengths are heavily
    right-skewed, so a rank test is more appropriate than a t-test. With more classes
    the Kruskal-Wallis test is used instead.

    Args:
        df: Data with ``text`` and ``label``.

    Returns:
        Summary table (count, mean, std, min, quartiles, max per class) and p-value.
    """
    df = add_length_columns(df)
    labels = labels_in(df)
    groups = [df.loc[df["label"] == lab, "n_words"] for lab in labels]
    summary = df.groupby("label")["n_words"].describe().loc[labels]
    if len(groups) == 2:
        p = float(mannwhitneyu(*groups, alternative="two-sided").pvalue)
        name = "Mann-Whitney U"
    else:
        from scipy.stats import kruskal

        p = float(kruskal(*groups).pvalue)
        name = "Kruskal-Wallis"
    return LengthComparison(summary=summary, test_name=name, p_value=p)


def share_longer_than(df: pd.DataFrame, thresholds: Iterable[int]) -> dict[int, float]:
    """Share of texts with more than ``t`` whitespace-separated words, per threshold.

    Subword tokenizers produce at least as many tokens as words for Serbian, so this
    is a lower bound on how many texts a transformer with that max length truncates.
    """
    n_words = df["text"].str.split().str.len()
    return {t: float((n_words > t).mean()) for t in thresholds}


def top_frequent_words(
    df: pd.DataFrame, n: int = 20, stopwords: frozenset[str] = SERBIAN_STOPWORDS
) -> dict[str, list[tuple[str, int]]]:
    """Most frequent non-stopword tokens per class.

    Args:
        df: Data with ``text`` and ``label``.
        n: Number of words per class.
        stopwords: Tokens to ignore.

    Returns:
        Mapping label -> list of ``(word, count)``.
    """
    result = {}
    for lab in labels_in(df):
        counts = Counter(
            tok
            for text in df.loc[df["label"] == lab, "text"]
            for tok in tokenize(text)
            if tok not in stopwords and len(tok) > 1
        )
        result[lab] = counts.most_common(n)
    return result


def informative_words(df: pd.DataFrame, n: int = 20, min_df: int = 5) -> pd.DataFrame:
    """Rank words by the chi-square statistic of word presence vs. class.

    Each word is assigned to the class in which it appears in the largest share of
    documents. Binary presence (not counts) is used so one very long review cannot
    dominate a word's score.

    Args:
        df: Data with ``text`` and ``label``.
        n: Number of words to return per class.
        min_df: Ignore words appearing in fewer documents than this.

    Returns:
        DataFrame with columns ``label``, ``word``, ``chi2``, ``p_value`` and one
        ``doc_share_<label>`` column per class, sorted by class then score.
    """
    labels = labels_in(df)
    vectorizer = CountVectorizer(
        tokenizer=tokenize, lowercase=False, binary=True, min_df=min_df, token_pattern=None
    )
    x = vectorizer.fit_transform(df["text"])
    y = df["label"].to_numpy()
    scores, p_values = chi2(x, y)
    vocab = vectorizer.get_feature_names_out()
    shares = np.vstack([np.asarray(x[y == lab].mean(axis=0)).ravel() for lab in labels])

    table = pd.DataFrame({"word": vocab, "chi2": scores, "p_value": p_values})
    for i, lab in enumerate(labels):
        table[f"doc_share_{lab}"] = shares[i]
    table["label"] = np.array(labels)[shares.argmax(axis=0)]
    top = (
        table.sort_values("chi2", ascending=False)
        .groupby("label", sort=False)
        .head(n)
        .assign(label=lambda t: pd.Categorical(t["label"], labels, ordered=True))
        .sort_values(["label", "chi2"], ascending=[True, False])
    )
    top["label"] = top["label"].astype(str)
    return top[["label", "word", "chi2", "p_value", *[f"doc_share_{lab}" for lab in labels]]]


@dataclass
class VocabularyStats:
    """Vocabulary size and how much of a held-out split is unseen in training."""

    n_tokens: int
    n_types: int
    hapax_share: float
    oov_token_rate: float
    oov_type_rate: float


def _oov_rate(tokens: Iterable[str], vocab: Counter[str]) -> float:
    """Share of ``tokens`` not present in ``vocab`` (0.0 for no tokens)."""
    flags = [t not in vocab for t in tokens]
    return float(np.mean(flags)) if flags else 0.0


def vocabulary_stats(train: pd.DataFrame, heldout: pd.DataFrame) -> VocabularyStats:
    """Compute vocabulary statistics on ``train`` and OOV rates on ``heldout``.

    Args:
        train: Training data.
        heldout: Validation (or test) data.

    Returns:
        Vocabulary statistics. ``hapax_share`` is the share of training word types
        seen exactly once.
    """
    counts = Counter(tok for text in train["text"] for tok in tokenize(text))
    held_tokens = [tok for text in heldout["text"] for tok in tokenize(text)]
    held_types = set(held_tokens)
    return VocabularyStats(
        n_tokens=sum(counts.values()),
        n_types=len(counts),
        hapax_share=sum(1 for c in counts.values() if c == 1) / max(len(counts), 1),
        oov_token_rate=_oov_rate(held_tokens, counts),
        oov_type_rate=_oov_rate(held_types, counts),
    )


def word_forms(train: pd.DataFrame, forms: Iterable[str]) -> dict[str, int]:
    """Count occurrences of specific word forms (e.g. inflections of one lemma)."""
    counts = Counter(tok for text in train["text"] for tok in tokenize(text))
    return {f: counts[f] for f in forms}


def cue_rates(df: pd.DataFrame, patterns: Mapping[str, str] = CUE_PATTERNS) -> pd.DataFrame:
    """Frequency of lexical cues per class.

    Args:
        df: Data with ``text`` and ``label``.
        patterns: Mapping cue name -> regex.

    Returns:
        DataFrame indexed by cue with, for each label, ``per100_<label>`` (mean
        occurrences per 100 words) and ``docs_<label>`` (share of documents with at
        least one occurrence), plus ``ratio`` = per100 of the first label divided by
        per100 of the last label.
    """
    df = add_length_columns(df)
    labels = labels_in(df)
    rows = {}
    for name, pattern in patterns.items():
        rx = regex.compile(pattern, regex.IGNORECASE)
        counts = df["text"].map(lambda t, rx=rx: len(rx.findall(t)))
        per100 = counts / df["n_words"] * 100
        row: dict[str, float] = {}
        for lab in labels:
            mask = df["label"] == lab
            row[f"per100_{lab}"] = float(per100[mask].mean())
            row[f"docs_{lab}"] = float((counts[mask] > 0).mean())
        last = row[f"per100_{labels[-1]}"]
        row["ratio"] = row[f"per100_{labels[0]}"] / last if last else float("nan")
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")
