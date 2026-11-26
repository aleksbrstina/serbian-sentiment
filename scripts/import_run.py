"""Import a run trained elsewhere (e.g. BERTić on Colab) into the local MLflow store.

The Colab notebook produces ``reports/bertic/summary.json`` and ``history.csv``.
After copying them into the repository, this script logs them as an MLflow run so
all three models can be compared in one local MLflow experiment.

Usage:
    python scripts/import_run.py --report-dir reports/bertic --run-name bertic-colab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from sentiment.tracking import mlflow, setup_mlflow
from sentiment.utils import load_config


def main(argv: list[str] | None = None) -> int:
    """Log summary metrics, per-epoch history and the report files; returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/bertic"))
    parser.add_argument("--config", default="configs/bertic.yaml")
    parser.add_argument("--run-name", default="bertic-colab")
    parser.add_argument("--tracking-uri", default=None, help="Override the config's MLflow URI")
    args = parser.parse_args(argv)

    summary_path = args.report_dir / "summary.json"
    if not summary_path.exists():
        print(f"{summary_path} not found; copy the Colab results first.", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = load_config(args.config)
    if args.tracking_uri:
        config = config.model_copy(update={"tracking_uri": args.tracking_uri})
    setup_mlflow(config)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({"source": "imported", "device": summary.get("device", "unknown")})
        if summary.get("hub_model_id"):
            mlflow.set_tag("hub_model_id", summary["hub_model_id"])
        mlflow.log_params(
            {
                k: summary[k]
                for k in ("model_name", "max_length", "truncation", "learning_rate")
                if k in summary
            }
        )
        history_path = args.report_dir / "history.csv"
        if history_path.exists():
            for row in pd.read_csv(history_path).to_dict("records"):
                metrics = {k: float(v) for k, v in row.items() if k != "epoch" and pd.notna(v)}
                mlflow.log_metrics(metrics, step=int(row["epoch"]))
        mlflow.log_metrics(
            {f"final_val_{k}": v for k, v in summary["val_metrics"].items()}
            | {
                k: float(summary[k])
                for k in (
                    "train_seconds",
                    "inference_ms_per_example",
                    "latency_ms_single",
                    "model_size_mb",
                    "best_epoch",
                )
                if k in summary
            }
        )
        for path in args.report_dir.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path))
    print(f"Imported {summary_path} as MLflow run {args.run_name!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
