"""Train a model from a YAML config.

Usage:
    python scripts/train.py --config configs/classical.yaml
"""

from __future__ import annotations

import argparse
import sys

from sentiment.utils import get_logger, load_config

logger = get_logger("train")


def main(argv: list[str] | None = None) -> int:
    """Dispatch training on ``config.model``; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to a configs/<model>.yaml file")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    missing = [str(p) for p in (config.data.train, config.data.val) if not p.exists()]
    if missing:
        print(f"Missing processed data: {missing}. Run `make prepare-data` first.", file=sys.stderr)
        return 1

    if config.model == "classical":
        from sentiment.models.classical import train_classical

        train_classical(config)
    else:
        print(f"Model {config.model!r} is not implemented yet.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
