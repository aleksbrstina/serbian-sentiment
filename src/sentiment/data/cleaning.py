"""Text normalization and record-level cleaning (deduplication, empty records)."""

from __future__ import annotations

import html
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import regex

# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #

# Serbian Cyrillic -> Latin (Gaj's alphabet). Digraph letters are handled separately
# so that all-caps words become "LJUBAV" rather than "LjUBAV".
_CYR_TO_LAT: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e", "ж": "ž",
    "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n",
    "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "ć", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "č", "џ": "dž", "ш": "š",
}  # fmt: skip
_CYR_UPPER_DIGRAPHS: dict[str, tuple[str, str]] = {
    "Љ": ("Lj", "LJ"),
    "Њ": ("Nj", "NJ"),
    "Џ": ("Dž", "DŽ"),
}
_CYRILLIC_RE = regex.compile(r"\p{Cyrillic}")

# Characters that look like Serbian diacritics but are different code points.
# Eth (ð/Ð) is the classic stand-in for đ/Đ in text typed on non-Serbian keyboards.
_DIACRITIC_FIXES: dict[str, str] = {"ð": "đ", "Ð": "Đ", "Ɖ": "Đ"}

_PUNCT_FIXES: dict[str, str] = {
    "„": '"', "“": '"', "”": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
    "–": "-", "—": "-", "‒": "-", "―": "-",
    "…": "...",
}  # fmt: skip

_ASCII_DIACRITICS = str.maketrans(
    {"č": "c", "ć": "c", "ž": "z", "š": "s", "đ": "dj",
     "Č": "C", "Ć": "C", "Ž": "Z", "Š": "S", "Đ": "Dj"}
)  # fmt: skip

_HTML_TAG_RE = regex.compile(r"<\s*/?\s*[a-zA-Z][^<>]*>")
_HTML_BREAK_RE = regex.compile(r"<\s*(br|p|div|li)\b[^<>]*>", regex.IGNORECASE)
# Control and format characters (BOM, zero-width spaces, C1 controls, ...), keeping \n and \t
# which are handled by whitespace normalization.
_INVISIBLE_RE = regex.compile(r"[^\S\n\t ]|[\p{Cc}\p{Cf}--[\n\t]]", flags=regex.V1)
_WHITESPACE_RE = regex.compile(r"\s+")


def _upper_context(text: str, i: int) -> bool:
    """Return True if the letter next to position ``i`` is uppercase (all-caps word)."""
    for j in (i + 1, i - 1):
        if 0 <= j < len(text) and text[j].isalpha():
            return text[j].isupper()
    return False


def cyrillic_to_latin(text: str) -> str:
    """Transliterate Serbian Cyrillic to Latin script; other characters are unchanged.

    Args:
        text: Input text in any mix of scripts.

    Returns:
        Text written in Latin script.
    """
    if not _CYRILLIC_RE.search(text):
        return text
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch in _CYR_UPPER_DIGRAPHS:
            title, caps = _CYR_UPPER_DIGRAPHS[ch]
            out.append(caps if _upper_context(text, i) else title)
        elif ch in _CYR_TO_LAT:
            out.append(_CYR_TO_LAT[ch])
        elif ch.lower() in _CYR_TO_LAT:
            out.append(_CYR_TO_LAT[ch.lower()].upper())
        else:
            out.append(ch)
    return "".join(out)


def normalize_diacritics(text: str) -> str:
    """Unify diacritic representation.

    Composes decomposed sequences (``c`` + combining caron -> ``č``) via Unicode NFC
    and replaces look-alike code points (``ð`` -> ``đ``). Digraphs such as ``dj`` are
    deliberately *not* converted to ``đ`` because they are ambiguous ("nadjačati").

    Args:
        text: Input text.

    Returns:
        Text with canonical Serbian diacritics.
    """
    text = unicodedata.normalize("NFC", text)
    return text.translate(str.maketrans(_DIACRITIC_FIXES))


def normalize_punctuation(text: str) -> str:
    """Map typographic quotes, dashes and ellipses to their ASCII equivalents."""
    for src, dst in _PUNCT_FIXES.items():
        text = text.replace(src, dst)
    return text


def strip_html(text: str) -> str:
    """Decode HTML entities and remove HTML tags, turning block tags into spaces."""
    text = html.unescape(text)
    text = _HTML_BREAK_RE.sub(" ", text)
    return _HTML_TAG_RE.sub("", text)


def remove_invisible(text: str) -> str:
    """Remove control/format characters (BOM, zero-width space) and odd spaces."""
    return _INVISIBLE_RE.sub(lambda m: " " if m.group().isspace() else "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and trim the ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_diacritics(text: str) -> str:
    """Map Serbian diacritics to ASCII (``č``/``ć`` -> ``c``, ``đ`` -> ``dj``, ...).

    Not part of the default pipeline; available as an experiment option because
    informal reviews often omit diacritics.
    """
    return text.translate(_ASCII_DIACRITICS)


#: Ordered normalization steps. Order matters: HTML entities may decode to Cyrillic or
#: invisible characters, and whitespace is collapsed last.
NORMALIZATION_STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("html", strip_html),
    ("invisible_chars", remove_invisible),
    ("cyrillic_to_latin", cyrillic_to_latin),
    ("diacritics", normalize_diacritics),
    ("punctuation", normalize_punctuation),
    ("whitespace", normalize_whitespace),
]


def normalize_text(text: str, ascii_diacritics: bool = False) -> str:
    """Apply the full normalization pipeline to one text.

    Args:
        text: Raw text.
        ascii_diacritics: If True, additionally strip diacritics (see
            :func:`strip_diacritics`).

    Returns:
        Normalized text.
    """
    for _, step in NORMALIZATION_STEPS:
        text = step(text)
    return strip_diacritics(text) if ascii_diacritics else text


def normalize_series(
    texts: pd.Series, ascii_diacritics: bool = False
) -> tuple[pd.Series, dict[str, int]]:
    """Normalize a column of texts and count how many records each step changed.

    Args:
        texts: Raw texts.
        ascii_diacritics: Whether to strip diacritics as a final step.

    Returns:
        Normalized texts and a mapping ``step name -> number of records changed``.
    """
    steps = list(NORMALIZATION_STEPS)
    if ascii_diacritics:
        steps.append(("ascii_diacritics", strip_diacritics))
    changed: dict[str, int] = {}
    current = texts.fillna("").astype(str)
    for name, step in steps:
        updated = current.map(step)
        changed[name] = int((updated != current).sum())
        current = updated
    return current, changed


# --------------------------------------------------------------------------- #
# Record-level cleaning
# --------------------------------------------------------------------------- #

#: Removal reasons, in the order they are applied.
REMOVAL_REASONS: tuple[str, ...] = (
    "empty_text",
    "missing_label",
    "conflicting_label",
    "exact_duplicate",
    "near_duplicate",
)

_NON_WORD_RE = regex.compile(r"[^\p{L}\p{N}]+")


def dedup_key(text: str) -> str:
    """Return a key that ignores case, punctuation and spacing, for near-duplicate detection."""
    return _NON_WORD_RE.sub(" ", text.casefold()).strip()


@dataclass
class CleaningResult:
    """Outcome of :func:`clean_records`."""

    data: pd.DataFrame
    removed: pd.DataFrame
    counts: dict[str, int] = field(default_factory=dict)


def clean_records(df: pd.DataFrame) -> CleaningResult:
    """Remove empty, unlabeled, conflicting and duplicate records.

    Expects already-normalized text. Steps, in order:

    1. ``empty_text``: text is empty after normalization.
    2. ``missing_label``: label is missing.
    3. ``conflicting_label``: the same text (up to case/punctuation) appears with
       different labels; *all* copies are dropped because the true label is unknown.
    4. ``exact_duplicate``: identical normalized text and label; the first copy is kept.
    5. ``near_duplicate``: identical up to case/punctuation/spacing; first copy kept.

    Args:
        df: DataFrame with at least ``text`` and ``label`` columns.

    Returns:
        Cleaned data, the removed rows with a ``reason`` column, and counts per reason.
    """
    work = df.copy()
    work["_key"] = work["text"].map(dedup_key)

    masks: dict[str, pd.Series] = {}
    remaining = pd.Series(True, index=work.index)

    def take(reason: str, mask: pd.Series) -> None:
        nonlocal remaining
        mask = mask & remaining
        masks[reason] = mask
        remaining = remaining & ~mask

    take("empty_text", work["text"].str.len() == 0)
    take("missing_label", work["label"].isna())
    labels_per_key = work[remaining].groupby("_key")["label"].transform("nunique")
    take("conflicting_label", (labels_per_key > 1).reindex(work.index, fill_value=False))
    take("exact_duplicate", work.duplicated(subset=["text", "label"], keep="first"))
    # Recompute on the surviving rows so a removed exact duplicate is not counted twice.
    near = work[remaining].duplicated(subset=["_key", "label"], keep="first")
    take("near_duplicate", near.reindex(work.index, fill_value=False))

    removed_parts = [work[m].assign(reason=r) for r, m in masks.items() if m.any()]
    removed = (
        pd.concat(removed_parts).drop(columns="_key")
        if removed_parts
        else work.iloc[0:0].drop(columns="_key").assign(reason=pd.Series(dtype=str))
    )
    return CleaningResult(
        data=work[remaining].drop(columns="_key").reset_index(drop=True),
        removed=removed.reset_index(drop=True),
        counts={r: int(m.sum()) for r, m in masks.items()},
    )
