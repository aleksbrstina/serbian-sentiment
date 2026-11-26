"""Train a model from a YAML config.

Usage:
    python scripts/train.py --config configs/classical.yaml
"""

from __future__ import annotations

import argparse
import sys

import yaml

from sentiment.utils import RunConfig, get_logger, load_config

logger = get_logger("train")


def apply_overrides(config: RunConfig, overrides: list[str]) -> RunConfig:
    """Return a copy of ``config`` with dotted ``key=value`` overrides applied and re-validated.

    Values are parsed as YAML, so ``true``, ``0.002`` and ``[1, 2]`` get proper types.
    """
    data = config.model_dump(mode="json")
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override {item!r} is not KEY=VALUE")
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return RunConfig.model_validate(data)


def main(argv: list[str] | None = None) -> int:
    """Dispatch training on ``config.model``; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to a configs/<model>.yaml file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set params.lr=0.002 (repeatable; YAML-parsed)",
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name (default: model name)")
    parser.add_argument(
        "--report-dir", default=None, help="Report directory (default: reports/<model>)"
    )
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)

    missing = [str(p) for p in (config.data.train, config.data.val) if not p.exists()]
    if missing:
        print(f"Missing processed data: {missing}. Run `make prepare-data` first.", file=sys.stderr)
        return 1

    if config.model == "classical":
        from sentiment.models.classical import train_classical

        train_classical(config, report_dir=args.report_dir or "reports/classical")
    elif config.model == "bilstm":
        from sentiment.models.neural.train import train_bilstm

        train_bilstm(config, report_dir=args.report_dir or "reports/bilstm", run_name=args.run_name)
    elif config.model == "bertic":
        from sentiment.models.transformer import train_transformer

        train_transformer(
            config, report_dir=args.report_dir or "reports/bertic", run_name=args.run_name
        )
    else:
        print(f"Model {config.model!r} is not implemented yet.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
