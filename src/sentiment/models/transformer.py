"""Fine-tuning BERTić (``classla/bcms-bertic``) for sentiment with the Hugging Face Trainer.

Runs locally (CPU, slow; useful for smoke tests) and in Colab/Kaggle (GPU). Long
reviews are truncated head+tail by default: the first and last tokens are kept,
because the EDA and the BiLSTM experiments showed the verdict is usually at the end.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field

from sentiment.evaluation.metrics import compute_metrics
from sentiment.predict import Predictor
from sentiment.utils import get_logger, path_size_mb, timer

logger = get_logger(__name__)

Truncation = Literal["head", "tail", "head_tail"]
DEFAULT_MODEL = "classla/bcms-bertic"
FALLBACK_MODEL = "FacebookAI/xlm-roberta-base"


class TransformerParams(BaseModel):
    """Model-specific settings (``params`` section of ``configs/bertic.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = DEFAULT_MODEL
    max_length: int = Field(default=256, ge=8, le=512)
    truncation: Truncation = "head_tail"
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    num_train_epochs: float = Field(default=4, gt=0)
    train_batch_size: int = Field(default=16, ge=1)
    eval_batch_size: int = Field(default=32, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    early_stopping_patience: int | None = 2
    metric_for_best_model: Literal["eval_loss", "eval_macro_f1"] = "eval_loss"
    fp16: Literal["auto", "on", "off"] = "auto"
    max_train_samples: int | None = None  # smoke tests only
    max_eval_samples: int | None = None
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_private: bool = False
    latency_samples: int = Field(default=50, ge=1)


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def truncate_ids(ids: list[int], budget: int, strategy: Truncation) -> list[int]:
    """Truncate subword ids (without special tokens) to ``budget`` tokens.

    Args:
        ids: Subword ids of one text, without ``[CLS]``/``[SEP]``.
        budget: Number of content tokens allowed (``max_length`` minus specials).
        strategy: ``head``, ``tail`` or ``head_tail`` (first ``budget // 2`` and last
            ``budget - budget // 2`` tokens).

    Returns:
        Truncated ids.
    """
    if len(ids) <= budget:
        return ids
    if strategy == "head":
        return ids[:budget]
    if strategy == "tail":
        return ids[-budget:]
    head = budget // 2
    return ids[:head] + ids[len(ids) - (budget - head) :]


def special_token_frame(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return the special-token ids the tokenizer puts before and after a single text.

    Derived by tokenizing a probe word with and without special tokens, so it works
    for any architecture (``[CLS] ... [SEP]`` for BERT/ELECTRA, ``<s> ... </s>`` for
    XLM-R) and does not depend on tokenizer methods that differ between versions.
    """
    probe = "film"
    plain = tokenizer(probe, add_special_tokens=False)["input_ids"]
    full = tokenizer(probe, add_special_tokens=True)["input_ids"]
    for start in range(len(full) - len(plain) + 1):
        if full[start : start + len(plain)] == plain:
            return full[:start], full[start + len(plain) :]
    raise ValueError("Could not locate the probe tokens inside the special-token frame")


def encode(
    texts: Sequence[str], tokenizer: Any, max_length: int, strategy: Truncation
) -> dict[str, list[list[int]]]:
    """Tokenize texts with custom truncation and wrap them in the model's special tokens.

    Hugging Face tokenizers only truncate from one side, so the full sequence is
    tokenized first and truncated here.

    Returns:
        Dict with ``input_ids`` and ``attention_mask`` (unpadded lists).
    """
    prefix, suffix = special_token_frame(tokenizer)
    budget = max_length - len(prefix) - len(suffix)
    raw = tokenizer(list(texts), add_special_tokens=False, truncation=False, verbose=False)
    input_ids = [prefix + truncate_ids(ids, budget, strategy) + suffix for ids in raw["input_ids"]]
    return {"input_ids": input_ids, "attention_mask": [[1] * len(ids) for ids in input_ids]}


class EncodedDataset(torch.utils.data.Dataset):
    """Pre-encoded examples for the Trainer; padding is done per batch by the collator."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: Sequence[int]) -> None:
        """Store encodings and integer labels."""
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        """Number of examples."""
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Return one example as a dict of lists plus ``labels``."""
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #


class TransformerPredictor(Predictor):
    """:class:`Predictor` for a fine-tuned sequence classification transformer.

    Truncation settings are stored in the model config (``sentiment_max_length``,
    ``sentiment_truncation``), so a model downloaded from the Hub is self-describing.
    """

    METADATA_FILE = "metadata.json"

    def __init__(self, model: Any, tokenizer: Any, name: str = "BERTić") -> None:
        """Wrap a model and tokenizer; inference runs on GPU if available."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.name = name
        cfg = model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        self.max_length = int(getattr(cfg, "sentiment_max_length", 256))
        self.truncation: Truncation = getattr(cfg, "sentiment_truncation", "head_tail")

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Softmax probabilities, columns ordered as :attr:`labels`."""
        out = []
        for start in range(0, len(texts), batch_size):
            enc = encode(
                texts[start : start + batch_size], self.tokenizer, self.max_length, self.truncation
            )
            batch = self.tokenizer.pad(enc, return_tensors="pt").to(self.device)
            logits = self.model(**batch).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def attention(self, text: str, layer: int = -1) -> tuple[list[str], np.ndarray]:
        """Attention from ``[CLS]`` to every token, averaged over heads (FR-16 groundwork).

        Args:
            text: Input text.
            layer: Transformer layer (default: last).

        Returns:
            Tokens and one attention weight per token.
        """
        enc = encode([text], self.tokenizer, self.max_length, self.truncation)
        batch = self.tokenizer.pad(enc, return_tensors="pt").to(self.device)
        # Fused SDPA kernels (the default) do not return attention weights, so switch to
        # the eager implementation for this call only and restore it afterwards.
        previous = getattr(self.model.config, "_attn_implementation", None) or "sdpa"
        self.model.set_attn_implementation("eager")
        try:
            with torch.no_grad():
                out = self.model(**batch, output_attentions=True)
        finally:
            self.model.set_attn_implementation(previous)
        weights = out.attentions[layer][0].mean(dim=0)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        return tokens, weights

    def word_attention(self, text: str, layer: int = -1) -> list[tuple[str, float | None]]:
        """``[CLS]`` attention merged from subwords into words, scaled to a max of 1.0.

        If the text was truncated head+tail, a ``("[…]", None)`` entry marks exactly
        where the middle was cut out.

        Args:
            text: Input text.
            layer: Transformer layer (default: last).

        Returns:
            ``(word, weight)`` pairs in text order.
        """
        tokens, weights = self.attention(text, layer=layer)
        prefix, suffix = special_token_frame(self.tokenizer)
        content_tokens = tokens[len(prefix) : len(tokens) - len(suffix)]
        content_weights = weights[len(prefix) : len(weights) - len(suffix)]
        budget = self.max_length - len(prefix) - len(suffix)
        n_full = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        cut_at = budget // 2 if n_full > budget and self.truncation == "head_tail" else None

        words: list[tuple[str, float | None]] = []
        for i, (token, weight) in enumerate(zip(content_tokens, content_weights, strict=True)):
            if i == cut_at:
                words.append(("[…]", None))
            if token.startswith("##") and words and words[-1][1] is not None:
                word, total = words[-1]
                words[-1] = (word + token[2:], total + float(weight))
            else:
                words.append((token, float(weight)))
        top = max((w for _, w in words if w is not None), default=1.0) or 1.0
        return [(w, None if v is None else v / top) for w, v in words]

    def save(self, directory: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save model, tokenizer and ``metadata.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        meta = {"type": "transformer", "name": self.name, "labels": self.labels, **(metadata or {})}
        (directory / self.METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, source: str | Path, name: str | None = None) -> TransformerPredictor:
        """Load from a local directory or a Hugging Face Hub repo id."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(str(source))
        tokenizer = AutoTokenizer.from_pretrained(str(source))
        if name is None:
            meta_path = Path(source) / cls.METADATA_FILE
            name = "BERTić"
            if meta_path.exists():
                name = json.loads(meta_path.read_text(encoding="utf-8")).get("name", name)
        return cls(model, tokenizer, name=name)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def history_from_log(log_history: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn ``trainer.state.log_history`` into one row per epoch.

    Training loss is averaged over the logging steps of each epoch.
    """
    train = pd.DataFrame([r for r in log_history if "loss" in r and "eval_loss" not in r])
    evals = pd.DataFrame([r for r in log_history if "eval_loss" in r])
    if evals.empty:
        return pd.DataFrame()
    evals["epoch"] = evals["epoch"].round().astype(int)
    rows = evals[["epoch", "eval_loss", "eval_macro_f1", "eval_accuracy"]].rename(
        columns={"eval_loss": "val_loss", "eval_macro_f1": "val_macro_f1"}
    )
    rows = rows.rename(columns={"eval_accuracy": "val_accuracy"})
    if not train.empty:
        train["epoch"] = np.ceil(train["epoch"]).astype(int).clip(lower=1)
        rows = rows.merge(
            train.groupby("epoch")["loss"].mean().rename("train_loss"), on="epoch", how="left"
        )
    return rows[["epoch", *[c for c in rows.columns if c != "epoch"]]]


def plot_history(history: pd.DataFrame, best_epoch: int, path: str | Path) -> Path:
    """Save training/validation loss and validation macro F1 per epoch."""
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 3.5))
    if "train_loss" in history:
        ax_loss.plot(history["epoch"], history["train_loss"], marker="o", label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    ax_loss.set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    ax_f1.plot(
        history["epoch"], history["val_macro_f1"], marker="o", color="C1", label="validation"
    )
    ax_f1.set(title="Validation macro F1", xlabel="Epoch", ylabel="Macro F1")
    from matplotlib.ticker import MaxNLocator

    for ax in (ax_loss, ax_f1):
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render_report(summary: dict[str, Any], history: pd.DataFrame) -> str:
    """Markdown report of a fine-tuning run; all numbers come from the arguments."""
    s = summary
    table = history.copy()
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    lines += [
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    hub_id = s.get("hub_model_id")
    hub = f"[`{hub_id}`](https://huggingface.co/{hub_id})" if hub_id else "not published"
    blocks = [
        f"# {s['name']}: fine-tuning report",
        "_Generated by `scripts/train.py --config configs/bertic.yaml` — do not edit by hand._",
        "## Result",
        "| Metric | Value |\n| --- | --- |\n"
        f"| Base model | `{s['model_name']}` |\n"
        f"| Validation macro F1 (best epoch) | {s['val_metrics']['macro_f1']:.4f} |\n"
        f"| Validation accuracy | {s['val_metrics']['accuracy']:.4f} |\n"
        f"| Best epoch / epochs run | {s['best_epoch']} / {s['epochs_run']} |\n"
        f"| Max length / truncation | {s['max_length']} / {s['truncation']} |\n"
        f"| Training time | {s['train_seconds']:.1f} s on {s['device']} |\n"
        f"| Inference | {s['inference_ms_per_example']:.2f} ms/example (batch), "
        f"{s['latency_ms_single']:.1f} ms single text ({s['device']}) |\n"
        f"| Parameters | {s['parameters']:,} |\n"
        f"| Model size on disk | {s['model_size_mb']:.1f} MB |\n"
        f"| Reviews truncated at {s['max_length']} tokens (train) | {s['truncated_share']:.1%} |\n"
        f"| Hugging Face Hub | {hub} |",
        "## Learning curves",
        "![Learning curves](learning_curves.png)",
        "\n".join(lines),
    ]
    return "\n\n".join(blocks) + "\n"


def model_card(summary: dict[str, Any]) -> str:
    """Hugging Face model card (README.md) generated from the run summary."""
    m = summary["val_metrics"]
    repo = summary.get("hub_model_id") or "<user>/<repo>"
    return f"""---
language: sr
license: apache-2.0
base_model: {summary["model_name"]}
pipeline_tag: text-classification
tags:
  - sentiment-analysis
  - serbian
  - bertic
metrics:
  - f1
  - accuracy
---

# BERTić fine-tuned for Serbian movie review sentiment

[`{summary["model_name"]}`](https://huggingface.co/{summary["model_name"]}) fine-tuned for
binary sentiment classification (negative / positive) on the SerbMR-2C movie review
dataset (Batanović et al., 2016).

| Validation metric | Value |
| --- | --- |
| Macro F1 | {m["macro_f1"]:.4f} |
| Accuracy | {m["accuracy"]:.4f} |

Validation split: {summary.get("n_val", "n/a")} reviews (stratified 80/10/10 split, seed 42).
Test-set results are reported in the project repository.

## Preprocessing

Texts are normalized (Cyrillic transliterated to Latin, unified diacritics and
punctuation). Reviews longer than {summary["max_length"]} subword tokens are truncated
**{summary["truncation"]}** (first and last tokens kept); these settings are stored in
`config.json` as `sentiment_max_length` and `sentiment_truncation`.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{repo}")
clf("Film je odličan, gluma je sjajna.")
```

The plain pipeline truncates from the end only; the project's `TransformerPredictor`
applies the same head+tail truncation that was used in training.

## Training

Learning rate {summary.get("learning_rate", "n/a")}, linear warmup and decay, best epoch
{summary["best_epoch"]} of {summary["epochs_run"]} selected on validation loss, trained in
{summary["train_seconds"] / 60:.1f} min on {summary["device"]}.

## Data license

SerbMR is licensed CC BY-NC-SA 4.0 (non-commercial research use). Cite:
Batanović, Nikolić, Milosavljević (2016). *Reliable Baselines for Sentiment Analysis in
Resource-Limited Languages: The Serbian Movie Review Dataset.* LREC 2016.
"""


def push_to_hub(
    model_dir: Path, repo_id: str, summary: dict[str, Any], private: bool = False
) -> str:
    """Upload the saved model directory (weights, tokenizer, metadata, model card).

    Authentication uses the token from ``huggingface_hub.login()`` or ``HF_TOKEN``.

    Returns:
        URL of the model repository.
    """
    from huggingface_hub import HfApi

    (model_dir / "README.md").write_text(model_card(summary), encoding="utf-8")
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        commit_message=f"Upload fine-tuned model (val macro F1 "
        f"{summary['val_metrics']['macro_f1']:.4f})",
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Published model to %s", url)
    return url


def _resolve_fp16(setting: str) -> bool:
    if setting == "on":
        return True
    if setting == "off":
        return False
    return torch.cuda.is_available()


def train_transformer(
    config: Any, report_dir: str | Path = "reports/bertic", run_name: str | None = None
) -> dict[str, Any]:
    """Fine-tune, evaluate on validation, save, log and optionally publish the model.

    Outputs:
        * ``<output_dir>/model/``: model, tokenizer, ``metadata.json``
        * ``<output_dir>/history.csv``, ``learning_curves.png``, ``config.yaml``
        * ``<report_dir>/training.md`` and ``learning_curves.png``
        * One MLflow run with per-epoch metrics
        * Optional push to the Hugging Face Hub (``push_to_hub``/``hub_model_id``)

    Args:
        config: :class:`sentiment.utils.RunConfig` with ``model: bertic``.
        report_dir: Where the git-tracked report is written.
        run_name: MLflow run name; defaults to ``"bertic"``.

    Returns:
        Summary dict (also stored in ``metadata.json``).
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from sentiment.tracking import flatten_params, mlflow, setup_mlflow
    from sentiment.utils import save_config

    params = TransformerParams.model_validate(config.params)
    if params.push_to_hub and not params.hub_model_id:
        raise ValueError("push_to_hub requires params.hub_model_id (e.g. 'user/bertic-serbmr')")
    set_seed(config.seed)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    train = pd.read_parquet(config.data.train)
    val = pd.read_parquet(config.data.val)
    if params.max_train_samples:
        train = train.sample(params.max_train_samples, random_state=config.seed)
    if params.max_eval_samples:
        val = val.sample(params.max_eval_samples, random_state=config.seed)
    labels = sorted(train["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        params.model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label2id,
    )
    # Stored in config.json so the published model carries its preprocessing.
    model.config.sentiment_max_length = params.max_length
    model.config.sentiment_truncation = params.truncation

    train_enc = encode(train["text"].tolist(), tokenizer, params.max_length, params.truncation)
    val_enc = encode(val["text"].tolist(), tokenizer, params.max_length, params.truncation)
    n_full = [
        len(ids) for ids in tokenizer(train["text"].tolist(), add_special_tokens=False)["input_ids"]
    ]
    prefix, suffix = special_token_frame(tokenizer)
    budget = params.max_length - len(prefix) - len(suffix)
    truncated_share = float(np.mean([n > budget for n in n_full]))

    train_ds = EncodedDataset(train_enc, [label2id[x] for x in train["label"]])
    val_ds = EncodedDataset(val_enc, [label2id[x] for x in val["label"]])

    def metrics_fn(eval_pred: Any) -> dict[str, float]:
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        names = np.array(labels)
        return compute_metrics(names[label_ids], names[preds], labels)

    greater = params.metric_for_best_model == "eval_macro_f1"
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=params.metric_for_best_model,
        greater_is_better=greater,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        lr_scheduler_type="linear",
        # transformers 5.x: a float in [0, 1) is a ratio of total steps (was warmup_ratio).
        warmup_steps=float(params.warmup_ratio),
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.train_batch_size,
        per_device_eval_batch_size=params.eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        fp16=_resolve_fp16(params.fp16),
        logging_strategy="steps",
        logging_steps=max(1, len(train_ds) // (params.train_batch_size * 4)),
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
        disable_tqdm=True,
    )
    callbacks = (
        [EarlyStoppingCallback(early_stopping_patience=params.early_stopping_patience)]
        if params.early_stopping_patience
        else []
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_fn,
        callbacks=callbacks,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    setup_mlflow(config)
    with mlflow.start_run(run_name=run_name or "bertic") as run:
        mlflow.log_params(flatten_params({"seed": config.seed, **params.model_dump(mode="json")}))
        with timer() as t_train:
            trainer.train()
        history = history_from_log(trainer.state.log_history)
        for row in history.to_dict("records"):
            mlflow.log_metrics(
                {k: float(v) for k, v in row.items() if k != "epoch" and pd.notna(v)},
                step=int(row["epoch"]),
            )
        best_ckpt = trainer.state.best_model_checkpoint
        monitored = history["val_macro_f1"] if greater else history["val_loss"]
        best_row = monitored.idxmax() if greater else monitored.idxmin()
        best_epoch = int(history.loc[best_row, "epoch"])

        predictor = TransformerPredictor(trainer.model, tokenizer, name="BERTić")
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
            "name": predictor.name,
            "model_name": params.model_name,
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "best_checkpoint": best_ckpt,
            "epochs_run": len(history),
            "train_seconds": t_train["seconds"],
            "device": device,
            "inference_ms_per_example": 1000 * t_batch["seconds"] / len(val_texts),
            "latency_ms_single": 1000 * float(np.median(latencies)),
            "parameters": int(sum(p.numel() for p in trainer.model.parameters())),
            "max_length": params.max_length,
            "truncation": params.truncation,
            "truncated_share": truncated_share,
            "learning_rate": params.learning_rate,
            "n_train": len(train),
            "n_val": len(val),
            "hub_model_id": params.hub_model_id if params.push_to_hub else None,
            "mlflow_run_id": run.info.run_id,
        }
        model_dir = out_dir / "model"
        predictor.save(model_dir, summary)
        summary["model_size_mb"] = path_size_mb(model_dir)
        predictor.save(model_dir, summary)
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)

        history.to_csv(out_dir / "history.csv", index=False)
        plot_history(history, best_epoch, out_dir / "learning_curves.png")
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_dir / "learning_curves.png", report_dir / "learning_curves.png")
        history.to_csv(report_dir / "history.csv", index=False)
        (report_dir / "training.md").write_text(render_report(summary, history), encoding="utf-8")
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        mlflow.log_metrics(
            {f"final_val_{k}": v for k, v in val_metrics.items()}
            | {
                "train_seconds": summary["train_seconds"],
                "inference_ms_per_example": summary["inference_ms_per_example"],
                "latency_ms_single": summary["latency_ms_single"],
                "model_size_mb": summary["model_size_mb"],
                "best_epoch": best_epoch,
            }
        )
        mlflow.log_artifact(str(report_dir / "training.md"))
        mlflow.log_artifact(str(out_dir / "learning_curves.png"))

        if params.push_to_hub:
            push_to_hub(model_dir, params.hub_model_id, summary, private=params.hub_private)
            mlflow.set_tag("hub_model_id", params.hub_model_id)

    logger.info(
        "%s: val macro F1 %.4f at epoch %d, %.1fs on %s",
        predictor.name,
        val_metrics["macro_f1"],
        best_epoch,
        t_train["seconds"],
        device,
    )
    return summary
