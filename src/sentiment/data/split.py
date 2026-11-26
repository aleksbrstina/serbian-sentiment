"""Stratified train/validation/test splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """Split data into train/val/test, stratified by label.

    The test set is carved out first, then the validation set from the remainder,
    with ``val_size`` rescaled so both fractions refer to the full dataset.

    Args:
        df: Data to split.
        val_size: Fraction of the full dataset for validation.
        test_size: Fraction of the full dataset for testing.
        seed: Random seed for reproducibility.
        label_col: Column to stratify on.

    Returns:
        Mapping with keys ``"train"``, ``"val"``, ``"test"``.

    Raises:
        ValueError: If the fractions are not in (0, 1) or sum to 1 or more.
    """
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError(f"Invalid split sizes: val={val_size}, test={test_size}")

    rest, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    train, val = train_test_split(
        rest,
        test_size=val_size / (1 - test_size),
        stratify=rest[label_col],
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
