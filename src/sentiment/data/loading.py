"""Load raw review datasets into a unified ``(id, text, label, source)`` DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Canonical label names used throughout the project.
NEGATIVE, NEUTRAL, POSITIVE = "negative", "neutral", "positive"
LABEL_ORDER: tuple[str, ...] = (NEGATIVE, NEUTRAL, POSITIVE)

#: Mapping from SerbMR label strings to canonical labels.
SERBMR_LABELS: dict[str, str] = {
    "NEGATIVE": NEGATIVE,
    "NEUTRAL": NEUTRAL,
    "POSITIVE": POSITIVE,
}

#: Expected raw file name for each supported SerbMR variant.
SERBMR_FILES: dict[str, str] = {
    "2C": "SerbMR-2C.csv",
    "3C": "SerbMR-3C.csv",
}

SERBMR_DOWNLOAD_INSTRUCTIONS = """\
SerbMR raw data not found: {path}

This project does not download datasets automatically. To get the data:
  1. Open https://github.com/vukbatanovic/SerbMR/releases/tag/v1.0
  2. Download {filename} (binary: SerbMR-2C.csv, three-class: SerbMR-3C.csv)
  3. Place it at {path}

License: CC BY-NC-SA 4.0 (non-commercial research use). Please cite:
  Batanović, Nikolić, Milosavljević (2016). Reliable Baselines for Sentiment Analysis
  in Resource-Limited Languages: The Serbian Movie Review Dataset. LREC 2016.
"""

UNIFIED_COLUMNS = ["id", "text", "label", "source"]


class RawDataMissingError(FileNotFoundError):
    """Raised when a raw dataset file is absent; the message explains how to get it."""


def serbmr_path(raw_dir: str | Path, variant: str) -> Path:
    """Return the expected path of a SerbMR variant inside ``raw_dir``.

    Args:
        raw_dir: Directory holding raw data files.
        variant: ``"2C"`` (binary) or ``"3C"`` (three-class).

    Returns:
        Path to the CSV file.

    Raises:
        ValueError: If the variant is unknown.
    """
    if variant not in SERBMR_FILES:
        expected = list(SERBMR_FILES)
        raise ValueError(f"Unknown SerbMR variant {variant!r}; expected one of {expected}")
    return Path(raw_dir) / SERBMR_FILES[variant]


def load_serbmr(path: str | Path, source: str | None = None) -> pd.DataFrame:
    """Load a SerbMR CSV file (columns ``Text,class-att``) into the unified format.

    Rows with an empty label are kept with ``label=None`` so the cleaning step can
    count and report them instead of silently dropping them here.

    Args:
        path: Path to ``SerbMR-2C.csv`` or ``SerbMR-3C.csv``.
        source: Value for the ``source`` column; defaults to the file stem.

    Returns:
        DataFrame with columns ``id``, ``text``, ``label``, ``source``. ``id`` is the
        zero-based row number in the raw file, kept for traceability.

    Raises:
        RawDataMissingError: If the file does not exist.
        ValueError: If expected columns are missing or a label is not recognised.
    """
    path = Path(path)
    if not path.exists():
        raise RawDataMissingError(
            SERBMR_DOWNLOAD_INSTRUCTIONS.format(path=path, filename=path.name)
        )

    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = {"Text", "class-att"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}; found {list(raw.columns)}")

    raw_labels = raw["class-att"].str.strip().str.upper()
    unknown = sorted(set(raw_labels) - set(SERBMR_LABELS) - {""})
    if unknown:
        raise ValueError(f"{path} contains unknown labels: {unknown}")

    return pd.DataFrame(
        {
            "id": range(len(raw)),
            "text": raw["Text"],
            "label": raw_labels.map(SERBMR_LABELS),  # "" -> NaN
            "source": source or path.stem,
        }
    )[UNIFIED_COLUMNS]
