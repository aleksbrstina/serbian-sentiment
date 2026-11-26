"""Run exploratory data analysis on the processed splits.

Usage:
    python scripts/run_eda.py --data-dir data/processed --out-dir reports/eda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sentiment.eda.report import run_eda, write_report
from sentiment.utils import get_logger

logger = get_logger("run_eda")


def main(argv: list[str] | None = None) -> int:
    """Run EDA and write the report; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args(argv)

    paths = {name: args.data_dir / f"{name}.parquet" for name in ("train", "val", "test")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(f"Missing processed data: {missing}. Run `make prepare-data` first.", file=sys.stderr)
        return 1

    splits = {name: pd.read_parquet(path) for name, path in paths.items()}
    result = run_eda(splits)
    for name, path in write_report(result, splits, args.out_dir).items():
        logger.info("Wrote %-22s -> %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
