"""Hand-written PyTorch training loop for the BiLSTM (no Lightning or Trainer wrappers).

One epoch = forward, loss, backward, gradient clipping, optimizer step for every
batch, followed by a no-grad pass over the validation split. The best epoch's
weights are kept (checkpointing) and training stops after ``patience`` epochs
without improvement of the monitored metric (early stopping).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader

from sentiment.evaluation.metrics import compute_metrics
from sentiment.models.neural.dataset import ReviewDataset, Truncation, collate
from sentiment.models.neural.embeddings import build_embedding_matrix
from sentiment.models.neural.model import BiLSTMClassifier, Pooling
from sentiment.models.neural.tokenizer import Vocabulary
from sentiment.predict import Predictor
from sentiment.utils import get_logger, path_size_mb, timer

logger = get_logger(__name__)


class BiLSTMParams(BaseModel):
    """Model-specific settings (``params`` section of ``configs/bilstm.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    # Vocabulary and inputs
    min_freq: int = Field(default=2, ge=1)
    max_vocab: int | None = 50_000
    max_len: int = Field(default=512, ge=8)
    truncation: Truncation = "head_tail"
    # Embeddings
    embedding_path: Path | None = Path("data/raw/embeddings/cc.sr.300.vec.gz")
    embedding_dim: int = 300
    transliterate_embeddings: bool = True
    freeze_embeddings: bool = True
    normalize_embeddings: bool = True
    embedding_cache_dir: Path | None = Path("data/processed")
    # Architecture
    hidden_size: int = Field(default=128, ge=1)
    num_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.5, ge=0, lt=1)
    embedding_dropout: float = Field(default=0.2, ge=0, lt=1)
    pooling: Pooling = "max_mean"
    # Optimization
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    max_epochs: int = Field(default=30, ge=1)
    patience: int = Field(default=4, ge=1)
    grad_clip: float | None = 1.0
    monitor: Literal["val_loss", "val_macro_f1"] = "val_loss"
    lr_scheduler: Literal["none", "plateau"] = "plateau"
    lr_factor: float = Field(default=0.5, gt=0, lt=1)
    lr_patience: int = Field(default=1, ge=0)
    num_threads: int | None = None
    latency_samples: int = Field(default=50, ge=1)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #


class BiLSTMPredictor(Predictor):
    """:class:`Predictor` for a trained :class:`BiLSTMClassifier`."""

    MODEL_FILE = "model.pt"
    VOCAB_FILE = "vocab.json"
    METADATA_FILE = "metadata.json"

    def __init__(
        self,
        model: BiLSTMClassifier,
        vocab: Vocabulary,
        labels: list[str],
        params: BiLSTMParams,
        name: str = "BiLSTM",
    ) -> None:
        """Wrap a trained network with the preprocessing it was trained with."""
        self.model = model.eval()
        self.vocab = vocab
        self.labels = labels
        self.params = params
        self.name = name

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Softmax probabilities, columns ordered as :attr:`labels`."""
        self.model.eval()
        dataset = ReviewDataset(
            texts, None, self.vocab, {}, self.params.max_len, self.params.truncation
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        out = [torch.softmax(self.model(ids, lengths), dim=1) for ids, lengths, _ in loader]
        return torch.cat(out).numpy() if out else np.zeros((0, len(self.labels)))

    def save(self, directory: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save weights, vocabulary and metadata (architecture, labels, preprocessing)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / self.MODEL_FILE)
        self.vocab.save(directory / self.VOCAB_FILE)
        meta = {
            "type": "bilstm",
            "name": self.name,
            "labels": self.labels,
            "params": self.params.model_dump(mode="json"),
            **(metadata or {}),
        }
        (directory / self.METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> BiLSTMPredictor:
        """Load a predictor saved with :meth:`save` (CPU)."""
        directory = Path(directory)
        meta = json.loads((directory / cls.METADATA_FILE).read_text(encoding="utf-8"))
        params = BiLSTMParams.model_validate(meta["params"])
        vocab = Vocabulary.load(directory / cls.VOCAB_FILE)
        model = build_model(params, len(vocab), len(meta["labels"]), pretrained=None)
        state = torch.load(directory / cls.MODEL_FILE, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(model, vocab, meta["labels"], params, name=meta.get("name", "BiLSTM"))


def build_model(
    params: BiLSTMParams, vocab_size: int, num_classes: int, pretrained: np.ndarray | None
) -> BiLSTMClassifier:
    """Instantiate the network from config parameters."""
    return BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=params.embedding_dim,
        hidden_size=params.hidden_size,
        num_layers=params.num_layers,
        dropout=params.dropout,
        embedding_dropout=params.embedding_dropout,
        pooling=params.pooling,
        pretrained=pretrained,
        freeze_embeddings=params.freeze_embeddings,
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one pass over ``loader``; trains if an optimizer is given, else evaluates.

    Returns:
        Mean loss per example, predicted class ids, gold class ids.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    preds, golds = [], []
    with torch.set_grad_enabled(training):
        for ids, lengths, labels in loader:
            logits = model(ids, lengths)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
            preds.append(logits.argmax(dim=1))
            golds.append(labels)
    return total_loss / max(n, 1), torch.cat(preds).numpy(), torch.cat(golds).numpy()


@dataclass
class TrainingResult:
    """Outcome of :func:`fit`."""

    model: BiLSTMClassifier
    history: pd.DataFrame
    best_epoch: int
    stopped_early: bool
    train_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


def fit(
    model: BiLSTMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    labels: list[str],
    params: BiLSTMParams,
    on_epoch_end: Any = None,
) -> TrainingResult:
    """Train with early stopping, keeping the best epoch's weights.

    Args:
        model: Network to train (modified in place, then restored to the best epoch).
        train_loader: Shuffled training batches.
        val_loader: Validation batches.
        labels: Class labels in index order.
        params: Optimization settings.
        on_epoch_end: Optional callback ``f(epoch, row_dict)`` (used for MLflow logging).

    Returns:
        The best model, the per-epoch history and timing.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=params.lr_factor, patience=params.lr_patience
        )
        if params.lr_scheduler == "plateau"
        else None
    )
    higher_is_better = params.monitor == "val_macro_f1"
    best_score = -math.inf if higher_is_better else math.inf
    best_state: dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
    best_epoch, bad_epochs, rows = 0, 0, []

    with timer() as t:
        for epoch in range(1, params.max_epochs + 1):
            train_loss, train_pred, train_gold = run_epoch(
                model, train_loader, loss_fn, optimizer, params.grad_clip
            )
            val_loss, val_pred, val_gold = run_epoch(model, val_loader, loss_fn)
            to_labels = np.array(labels)
            train_m = compute_metrics(to_labels[train_gold], to_labels[train_pred], labels)
            val_m = compute_metrics(to_labels[val_gold], to_labels[val_pred], labels)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_macro_f1": train_m["macro_f1"],
                "val_macro_f1": val_m["macro_f1"],
                "val_accuracy": val_m["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            if scheduler is not None:
                scheduler.step(val_loss)
            score = row[params.monitor]
            improved = score > best_score if higher_is_better else score < best_score
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
            row["best_so_far"] = improved
            rows.append(row)
            logger.info(
                "epoch %2d | train loss %.4f F1 %.3f | val loss %.4f F1 %.3f%s",
                epoch,
                train_loss,
                train_m["macro_f1"],
                val_loss,
                val_m["macro_f1"],
                "  *" if improved else "",
            )
            if on_epoch_end is not None:
                on_epoch_end(epoch, row)
            if bad_epochs >= params.patience:
                logger.info(
                    "Early stopping: no %s improvement for %d epochs", params.monitor, bad_epochs
                )
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        stopped_early=bad_epochs >= params.patience,
        train_seconds=t["seconds"],
    )


def plot_history(history: pd.DataFrame, best_epoch: int, path: str | Path) -> Path:
    """Save train/validation loss and macro F1 curves with the best epoch marked."""
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax_loss.plot(history["epoch"], history["train_loss"], marker="o", label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    ax_loss.set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    ax_f1.plot(history["epoch"], history["train_macro_f1"], marker="o", label="train")
    ax_f1.plot(history["epoch"], history["val_macro_f1"], marker="o", label="validation")
    ax_f1.set(title="Macro F1", xlabel="Epoch", ylabel="Macro F1")
    for ax in (ax_loss, ax_f1):
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _seeded_loader(dataset: ReviewDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator
    )


def token_coverage(
    texts: Sequence[str], vocab: Vocabulary, found_ids: set[int]
) -> dict[str, float]:
    """Share of tokens that are in the vocabulary, and that have a pretrained vector."""
    from sentiment.models.neural.tokenizer import tokenize

    ids = [i for text in texts for i in vocab.encode(tokenize(text))]
    n = max(len(ids), 1)
    return {
        "in_vocab": sum(i > 1 for i in ids) / n,
        "pretrained": sum(i in found_ids for i in ids) / n,
    }


def render_report(summary: dict[str, Any], history: pd.DataFrame) -> str:
    """Markdown report of a BiLSTM run; all numbers come from ``summary``/``history``."""
    s = summary
    hist = history.copy()
    hist["best_so_far"] = hist["best_so_far"].map({True: "✓", False: ""})
    header = "| " + " | ".join(hist.columns) + " |"
    sep = "| " + " | ".join("---" for _ in hist.columns) + " |"
    rows = [
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |"
        for r in hist.itertuples(index=False)
    ]
    emb = s["embeddings"]
    blocks = [
        "# BiLSTM: training report",
        "_Generated by `scripts/train.py --config configs/bilstm.yaml` — do not edit by hand._",
        "## Result",
        "| Metric | Value |\n| --- | --- |\n"
        f"| Validation macro F1 (best epoch) | {s['val_metrics']['macro_f1']:.4f} |\n"
        f"| Validation accuracy | {s['val_metrics']['accuracy']:.4f} |\n"
        f"| Best epoch / epochs run | {s['best_epoch']} / {s['epochs_run']} |\n"
        f"| Early stopping triggered | {s['stopped_early']} |\n"
        f"| Training time | {s['train_seconds']:.1f} s |\n"
        f"| Inference | {s['inference_ms_per_example']:.2f} ms/example (batch), "
        f"{s['latency_ms_single']:.1f} ms single text |\n"
        f"| Parameters (trainable / total) | {s['parameters']['trainable']:,} / "
        f"{s['parameters']['total']:,} |\n"
        f"| Model size on disk | {s['model_size_mb']:.1f} MB |",
        "## Vocabulary and embeddings",
        "| Metric | Value |\n| --- | --- |\n"
        f"| Vocabulary size (incl. `<pad>`, `<unk>`) | {s['vocab_size']:,} |\n"
        f"| Vocabulary types with a fastText vector | {emb['found']:,} "
        f"({emb['coverage_types']:.1%}) |\n"
        f"| Validation tokens in vocabulary | {s['val_coverage']['in_vocab']:.1%} |\n"
        f"| Validation tokens with a fastText vector | {s['val_coverage']['pretrained']:.1%} |",
        "## Learning curves",
        "![Learning curves](learning_curves.png)",
        "\n".join([header, sep, *rows]),
    ]
    return "\n\n".join(blocks) + "\n"


def train_bilstm(
    config: Any, report_dir: str | Path = "reports/bilstm", run_name: str | None = None
) -> dict[str, Any]:
    """Train, checkpoint, evaluate on validation, save and log the BiLSTM.

    Outputs:
        * ``<output_dir>/``: ``model.pt``, ``vocab.json``, ``metadata.json``,
          ``history.csv``, ``learning_curves.png``, ``config.yaml``
        * ``<report_dir>/training.md`` and ``learning_curves.png`` (git-tracked)
        * One MLflow run with per-epoch metrics

    Args:
        config: :class:`sentiment.utils.RunConfig` with ``model: bilstm``.
        report_dir: Where the git-tracked report is written.
        run_name: MLflow run name; defaults to ``"bilstm"``.

    Returns:
        Summary dict (also stored in ``metadata.json``).
    """
    import shutil

    from sentiment.tracking import flatten_params, mlflow, setup_mlflow
    from sentiment.utils import save_config, set_seed

    params = BiLSTMParams.model_validate(config.params)
    set_seed(config.seed)
    if params.num_threads:
        torch.set_num_threads(params.num_threads)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    train = pd.read_parquet(config.data.train)
    val = pd.read_parquet(config.data.val)
    labels = sorted(train["label"].unique())
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    vocab = Vocabulary.build(train["text"], params.min_freq, params.max_vocab)
    emb = build_embedding_matrix(
        vocab,
        params.embedding_path,
        params.embedding_dim,
        params.transliterate_embeddings,
        config.seed,
        params.embedding_cache_dir,
    )
    if params.normalize_embeddings and emb.found:
        # fastText values are tiny (std ~0.05); rescaling to unit std gives the LSTM
        # inputs on a usable scale without changing directions between words.
        emb.weights[2:] /= emb.weights[2:].std()
    logger.info(
        "Vocabulary %d types; fastText vectors for %d (%.1f%%)",
        len(vocab),
        emb.found,
        100 * emb.coverage_types,
    )

    train_ds = ReviewDataset(
        train["text"], train["label"], vocab, label_to_id, params.max_len, params.truncation
    )
    val_ds = ReviewDataset(
        val["text"], val["label"], vocab, label_to_id, params.max_len, params.truncation
    )
    train_loader = _seeded_loader(train_ds, params.batch_size, True, config.seed)
    val_loader = _seeded_loader(val_ds, params.batch_size, False, config.seed)

    model = build_model(params, len(vocab), len(labels), emb.weights)

    setup_mlflow(config)
    with mlflow.start_run(run_name=run_name or "bilstm") as run:
        mlflow.log_params(flatten_params({"seed": config.seed, **params.model_dump(mode="json")}))
        mlflow.log_params({"vocab_size": len(vocab), "embedding_coverage": emb.coverage_types})

        def log_epoch(epoch: int, row: dict[str, Any]) -> None:
            mlflow.log_metrics(
                {k: float(v) for k, v in row.items() if k not in {"epoch", "best_so_far"}},
                step=epoch,
            )

        result = fit(model, train_loader, val_loader, labels, params, on_epoch_end=log_epoch)
        predictor = BiLSTMPredictor(result.model, vocab, labels, params)

        val_texts = val["text"].tolist()
        with timer() as t_batch:
            val_pred = predictor.predict(val_texts)
        latencies = []
        for text in val_texts[: params.latency_samples]:
            with timer() as t_one:
                predictor.predict_proba([text])
            latencies.append(t_one["seconds"])
        val_metrics = compute_metrics(val["label"].tolist(), val_pred, labels)

        summary: dict[str, Any] = {
            "val_metrics": val_metrics,
            "best_epoch": result.best_epoch,
            "epochs_run": len(result.history),
            "stopped_early": result.stopped_early,
            "train_seconds": result.train_seconds,
            "inference_ms_per_example": 1000 * t_batch["seconds"] / len(val_texts),
            "latency_ms_single": 1000 * float(np.median(latencies)),
            "parameters": result.model.count_parameters(),
            "vocab_size": len(vocab),
            "embeddings": {"found": emb.found, "coverage_types": emb.coverage_types},
            "val_coverage": token_coverage(val_texts, vocab, emb.found_ids),
            "mlflow_run_id": run.info.run_id,
        }
        predictor.save(out_dir, summary)
        summary["model_size_mb"] = path_size_mb(out_dir / BiLSTMPredictor.MODEL_FILE)
        predictor.save(out_dir, summary)

        result.history.to_csv(out_dir / "history.csv", index=False)
        plot_history(result.history, result.best_epoch, out_dir / "learning_curves.png")
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_dir / "learning_curves.png", report_dir / "learning_curves.png")
        (report_dir / "training.md").write_text(
            render_report(summary, result.history), encoding="utf-8"
        )

        mlflow.log_metrics(
            {f"final_val_{k}": v for k, v in val_metrics.items()}
            | {
                "train_seconds": summary["train_seconds"],
                "inference_ms_per_example": summary["inference_ms_per_example"],
                "latency_ms_single": summary["latency_ms_single"],
                "model_size_mb": summary["model_size_mb"],
                "best_epoch": result.best_epoch,
            }
        )
        mlflow.log_artifacts(str(out_dir), artifact_path="model")
        mlflow.log_artifact(str(report_dir / "training.md"))
    logger.info(
        "BiLSTM: val macro F1 %.4f at epoch %d, %.1fs",
        val_metrics["macro_f1"],
        result.best_epoch,
        result.train_seconds,
    )
    return summary
