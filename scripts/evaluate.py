"""Evaluate all trained models on the test split and write the comparison report.

Usage:
    python scripts/evaluate.py --config configs/evaluate.yaml
"""

from __future__ import annotations

import argparse
import sys

from sentiment.evaluation.report import load_eval_config, run_evaluation
from sentiment.utils import get_logger

logger = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/evaluate.yaml")
    args = parser.parse_args(argv)
    config = load_eval_config(args.config)
    if not config.test.exists():
        print(f"Missing {config.test}. Run `make prepare-data` first.", file=sys.stderr)
        return 1
    try:
        result = run_evaluation(config)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1
    logger.info("Wrote %s (README updated: %s)", result["comparison"], result["readme_updated"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
