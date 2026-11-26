"""Summarize MLflow runs whose name starts with a prefix into a Markdown table.

Usage:
    python scripts/summarize_experiments.py --prefix exp- --out reports/bilstm/experiments.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentiment.tracking import mlflow

PARAMS = [
    "lr",
    "dropout",
    "embedding_dropout",
    "pooling",
    "truncation",
    "max_len",
    "batch_size",
    "hidden_size",
]
METRICS = ["final_val_macro_f1", "best_epoch", "train_seconds"]


def main(argv: list[str] | None = None) -> int:
    """Write the experiment table; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default="exp-")
    parser.add_argument("--experiment", default="serbian-sentiment")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--out", type=Path, default=Path("reports/bilstm/experiments.md"))
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[args.experiment],
        filter_string=(
            f"attributes.run_name LIKE '{args.prefix}%' AND attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time ASC"],
    )
    if runs.empty:
        print(f"No finished runs starting with {args.prefix!r}", file=sys.stderr)
        return 1

    header = ["Run", *PARAMS, "Val macro F1", "Best epoch", "Train time (s)"]
    lines = [
        "# BiLSTM experiments",
        "",
        "_Generated from MLflow by `scripts/summarize_experiments.py` — do not edit by hand._",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    best = runs["metrics.final_val_macro_f1"].max()
    for _, r in runs.iterrows():
        f1 = r["metrics.final_val_macro_f1"]
        cells = [r["tags.mlflow.runName"], *[str(r.get(f"params.{p}", "")) for p in PARAMS]]
        cells += [
            f"**{f1:.4f}**" if f1 == best else f"{f1:.4f}",
            str(int(r["metrics.best_epoch"])),
            f"{r['metrics.train_seconds']:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
