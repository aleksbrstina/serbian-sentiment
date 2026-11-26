"""Prepare processed train/val/test splits from raw data.

Usage:
    python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import sys

from sentiment.data.loading import RawDataMissingError
from sentiment.data.pipeline import load_data_config, prepare_data, save_outputs
from sentiment.utils import get_logger, set_seed

logger = get_logger("prepare_data")


def main(argv: list[str] | None = None) -> int:
    """Run data preparation; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data YAML config")
    args = parser.parse_args(argv)

    config = load_data_config(args.config)
    set_seed(config.seed)
    try:
        result = prepare_data(config)
    except RawDataMissingError as err:
        print(err, file=sys.stderr)
        return 1

    for name, path in save_outputs(result, config).items():
        logger.info("Wrote %-8s -> %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
