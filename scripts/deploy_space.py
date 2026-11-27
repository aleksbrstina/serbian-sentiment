"""Publish the demo to Hugging Face Spaces.

1. Uploads the classical (logreg) and BiLSTM models to a model repository, because
   weights never go into git (NFR-08). BERTić is already on the Hub.
2. Builds a Space folder (app, ``sentiment`` package, pinned requirements, Space card)
   and uploads it. Library versions are pinned to the ones the models were trained
   with, so the pickled scikit-learn pipeline loads safely.

Authentication: ``huggingface_hub.login()`` or ``HF_TOKEN``.

Usage:
    python scripts/deploy_space.py --user <hf-username>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGES = [
    "numpy", "pandas", "regex", "pyyaml", "pydantic", "scikit-learn", "scipy", "joblib",
    "optuna", "transformers", "tokenizers", "safetensors", "huggingface-hub",
]  # fmt: skip

BASELINES_CARD = """---
license: cc-by-nc-sa-4.0
language: sr
tags: [sentiment-analysis, serbian, scikit-learn, pytorch]
---

# Serbian sentiment baselines (TF-IDF + logistic regression, BiLSTM)

Baseline models from the [serbian-sentiment](https://github.com/{github}) project, trained on
SerbMR-2C movie reviews. Used by the demo Space; load them with
`sentiment.predict.load_predictor("<folder>")` from the project package.

| Folder | Model |
| --- | --- |
| `logreg/` | TF-IDF (word unigrams + char 3–5-grams) + logistic regression (joblib) |
| `bilstm/` | BiLSTM over frozen fastText `cc.sr.300` embeddings (PyTorch state dict + vocabulary) |

The `model.joblib` file is a pickle: only load it from a source you trust.

Licenses: trained on SerbMR (CC BY-NC-SA 4.0); the BiLSTM embedding matrix is derived from
fastText vectors (CC BY-SA 3.0). Non-commercial research use.
"""

SPACE_CARD = """---
title: Serbian Review Sentiment
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {gradio}
python_version: "3.11"
app_file: app/app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: TF-IDF vs. BiLSTM vs. BERTić on Serbian movie reviews
models:
  - {user}/bertic-serbmr-sentiment
  - {user}/serbian-sentiment-baselines
---

# Serbian review sentiment: three approaches side by side

Demo for the [serbian-sentiment](https://github.com/{github}) project: a classical TF-IDF model,
a BiLSTM written from scratch and fine-tuned BERTić, trained on SerbMR movie reviews.
"""


def requirements() -> str:
    """Pinned runtime requirements (CPU PyTorch from the official CPU index)."""
    lines = ["--extra-index-url https://download.pytorch.org/whl/cpu"]
    lines += [f"{pkg}=={version(pkg)}" for pkg in RUNTIME_PACKAGES]
    lines.append(f"torch=={version('torch')}")
    return "\n".join(lines) + "\n"


def build_space(target: Path, user: str, github: str) -> Path:
    """Assemble the Space folder at ``target``."""
    (target / "app").mkdir(parents=True)
    shutil.copy(ROOT / "app" / "app.py", target / "app" / "app.py")
    shutil.copytree(
        ROOT / "src" / "sentiment",
        target / "src" / "sentiment",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (target / "requirements.txt").write_text(requirements(), encoding="utf-8")
    (target / "README.md").write_text(
        SPACE_CARD.format(gradio=version("gradio"), user=user, github=github), encoding="utf-8"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    """Upload the baseline models and the Space; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", required=True, help="Hugging Face username or organization")
    parser.add_argument("--github", default="aleksbrstina/serbian-sentiment")
    parser.add_argument("--space-name", default="serbian-sentiment-demo")
    parser.add_argument("--skip-models", action="store_true", help="Only update the Space")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    baselines = f"{args.user}/serbian-sentiment-baselines"
    space = f"{args.user}/{args.space_name}"

    if not args.skip_models:
        sources = {
            "logreg": ROOT / "artifacts/classical/logreg",
            "bilstm": ROOT / "artifacts/bilstm",
        }
        missing = [str(p) for p in sources.values() if not (p / "metadata.json").exists()]
        if missing:
            print(f"Train the models first; missing: {missing}", file=sys.stderr)
            return 1
        api.create_repo(baselines, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for folder, src in sources.items():
                shutil.copytree(src, staging / folder, ignore=shutil.ignore_patterns("*.png"))
            (staging / "README.md").write_text(
                BASELINES_CARD.format(github=args.github), encoding="utf-8"
            )
            api.upload_folder(
                folder_path=str(staging), repo_id=baselines, commit_message="Upload baselines"
            )
        print(f"Models: https://huggingface.co/{baselines}")

    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = build_space(Path(tmp) / "space", args.user, args.github)
        api.upload_folder(
            folder_path=str(folder), repo_id=space, repo_type="space", commit_message="Deploy demo"
        )
    print(f"Space: https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
