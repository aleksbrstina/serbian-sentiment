"""End-to-end data preparation: load -> normalize -> clean -> split -> save + report."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from sentiment.data.cleaning import REMOVAL_REASONS, clean_records, normalize_series
from sentiment.data.loading import LABEL_ORDER, load_serbmr, serbmr_path
from sentiment.data.split import stratified_split
from sentiment.utils import get_logger

logger = get_logger(__name__)

SPLITS: tuple[str, ...] = ("train", "val", "test")


class DataConfig(BaseModel):
    """Configuration for :func:`prepare_data` (``configs/data.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["serbmr"] = "serbmr"
    variant: Literal["2C", "3C"] = "2C"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    report_path: Path = Path("reports/data_quality.md")
    seed: int = Field(default=42, ge=0)
    val_size: float = Field(default=0.1, gt=0, lt=1)
    test_size: float = Field(default=0.1, gt=0, lt=1)
    ascii_diacritics: bool = False


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate a data preparation config from YAML."""
    with Path(path).open(encoding="utf-8") as f:
        return DataConfig.model_validate(yaml.safe_load(f) or {})


@dataclass
class PrepResult:
    """Everything produced by :func:`prepare_data`."""

    splits: dict[str, pd.DataFrame]
    raw: pd.DataFrame
    removed: pd.DataFrame
    removal_counts: dict[str, int]
    normalization_changes: dict[str, int]
    raw_sha256: str
    raw_path: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_data(config: DataConfig) -> PrepResult:
    """Run the data pipeline in memory (no files written).

    Args:
        config: Data preparation settings.

    Returns:
        Splits plus the intermediate statistics needed for the quality report.

    Raises:
        RawDataMissingError: If the raw file is absent (message has download steps).
    """
    raw_path = serbmr_path(config.raw_dir, config.variant)
    raw = load_serbmr(raw_path, source=f"SerbMR-{config.variant}")
    logger.info("Loaded %d records from %s", len(raw), raw_path)

    normalized = raw.copy()
    normalized["text"], changes = normalize_series(raw["text"], config.ascii_diacritics)

    cleaned = clean_records(normalized)
    logger.info("Removed %d records: %s", len(cleaned.removed), cleaned.counts)

    splits = stratified_split(
        cleaned.data, val_size=config.val_size, test_size=config.test_size, seed=config.seed
    )
    return PrepResult(
        splits=splits,
        raw=raw,
        removed=cleaned.removed,
        removal_counts=cleaned.counts,
        normalization_changes=changes,
        raw_sha256=_sha256(raw_path),
        raw_path=raw_path,
    )


def split_fingerprint(splits: dict[str, pd.DataFrame]) -> str:
    """Short hash of which record ids landed in which split.

    Lets another machine (e.g. Colab) verify it reproduced exactly the same split,
    independent of library versions.
    """
    h = hashlib.sha256()
    for name in SPLITS:
        ids = sorted(int(i) for i in splits[name]["id"])
        h.update(f"{name}:{','.join(map(str, ids))};".encode())
    return h.hexdigest()[:16]


def _labels_present(*frames: pd.DataFrame) -> list[str]:
    present = set().union(*(set(f["label"].dropna()) for f in frames))
    return [label for label in LABEL_ORDER if label in present]


def _md_table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


_STEP_DESCRIPTIONS: dict[str, str] = {
    "html": "Decode HTML entities, remove HTML tags",
    "invisible_chars": "Remove BOM, zero-width and control characters; map odd spaces to space",
    "cyrillic_to_latin": "Transliterate Serbian Cyrillic to Latin",
    "diacritics": "Unicode NFC composition; replace look-alikes (ð -> đ)",
    "punctuation": "Typographic quotes/dashes/ellipsis -> ASCII",
    "whitespace": "Collapse whitespace runs, trim",
    "ascii_diacritics": "Strip diacritics (č -> c, đ -> dj)",
}

_REASON_DESCRIPTIONS: dict[str, str] = {
    "empty_text": "Text empty after normalization",
    "missing_label": "No label",
    "conflicting_label": "Same text with different labels (all copies dropped)",
    "exact_duplicate": "Identical normalized text and label (first kept)",
    "near_duplicate": "Identical up to case/punctuation/spacing (first kept)",
}


def render_quality_report(result: PrepResult, config: DataConfig) -> str:
    """Render the data quality report as Markdown. All numbers come from ``result``.

    Args:
        result: Output of :func:`prepare_data`.
        config: Config used for the run.

    Returns:
        Markdown document.
    """
    labels = _labels_present(result.raw, *result.splits.values())
    n_raw = len(result.raw)
    n_clean = sum(len(s) for s in result.splits.values())

    # Removal counts per reason and class.
    removal_rows = []
    for reason in REMOVAL_REASONS:
        sub = result.removed[result.removed["reason"] == reason] if len(result.removed) else []
        per_class = [int((sub["label"] == lab).sum()) if len(sub) else 0 for lab in labels]
        removal_rows.append(
            [f"`{reason}`", _REASON_DESCRIPTIONS[reason], result.removal_counts.get(reason, 0)]
            + per_class
        )
    removal_rows.append(
        ["**total**", "", len(result.removed)]
        + [int((result.removed["label"] == lab).sum()) for lab in labels]
    )

    norm_rows = [
        [i, f"`{name}`", _STEP_DESCRIPTIONS[name], count, f"{count / max(n_raw, 1):.1%}"]
        for i, (name, count) in enumerate(result.normalization_changes.items(), start=1)
    ]

    split_rows = []
    for name in SPLITS:
        df = result.splits[name]
        counts = [int((df["label"] == lab).sum()) for lab in labels]
        split_rows.append(
            [name, len(df), f"{len(df) / max(n_clean, 1):.1%}"]
            + [f"{c} ({c / max(len(df), 1):.1%})" for c in counts]
        )

    raw_counts = [int((result.raw["label"] == lab).sum()) for lab in labels]
    words = pd.concat(result.splits.values())["text"].str.split().str.len()

    return (
        "\n\n".join(
            [
                "# Data quality report",
                "_Generated by `scripts/prepare_data.py` — do not edit by hand._",
                _md_table(
                    ["Property", "Value"],
                    [
                        ["Dataset", f"SerbMR-{config.variant}"],
                        ["Raw file", f"`{result.raw_path}`"],
                        ["Raw file SHA-256", f"`{result.raw_sha256}`"],
                        ["Seed", config.seed],
                        ["Split (train/val/test)", _split_ratio(config)],
                        ["Split fingerprint", f"`{split_fingerprint(result.splits)}`"],
                        ["Generated at (UTC)", datetime.now(UTC).strftime("%Y-%m-%d %H:%M")],
                    ],
                ),
                "## 1. Raw data",
                _md_table(
                    ["Records", *labels, "Missing label"],
                    [[n_raw, *raw_counts, int(result.raw["label"].isna().sum())]],
                ),
                "## 2. Text normalization",
                "Steps are applied in order to every record. The count is the number of "
                "records whose text the step changed.",
                _md_table(["#", "Step", "Description", "Records changed", "Share"], norm_rows),
                "## 3. Record cleaning",
                "Rules are applied in order; each record is counted under the first rule "
                "it matches.",
                _md_table(["Reason", "Description", "Removed", *labels], removal_rows),
                f"Records kept: **{n_clean}** of {n_raw} ({n_clean / max(n_raw, 1):.1%}).",
                "## 4. Stratified split",
                _md_table(["Split", "Records", "Share", *labels], split_rows),
                "## 5. Cleaned text length (words)",
                _md_table(
                    ["Min", "Median", "Mean", "Max"],
                    [
                        [
                            int(words.min()),
                            int(words.median()),
                            f"{words.mean():.1f}",
                            int(words.max()),
                        ]
                    ],
                ),
            ]
        )
        + "\n"
    )


def _split_ratio(config: DataConfig) -> str:
    train = 1 - config.val_size - config.test_size
    return f"{train:.0%} / {config.val_size:.0%} / {config.test_size:.0%}"


def save_outputs(result: PrepResult, config: DataConfig) -> dict[str, Path]:
    """Write split Parquet files, removed records, the config and the quality report.

    Args:
        result: Output of :func:`prepare_data`.
        config: Config used for the run; saved next to the data (NFR-01).

    Returns:
        Mapping of output name to written path.
    """
    out_dir = config.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in SPLITS:
        paths[name] = out_dir / f"{name}.parquet"
        result.splits[name].to_parquet(paths[name], index=False)

    paths["removed"] = out_dir / "removed.parquet"
    result.removed.to_parquet(paths["removed"], index=False)

    paths["config"] = out_dir / "data_config.yaml"
    with paths["config"].open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(render_quality_report(result, config), encoding="utf-8")
    paths["report"] = config.report_path
    return paths
