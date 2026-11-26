import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from transformers import (  # noqa: E402
    BertTokenizerFast,
    ElectraConfig,
    ElectraForSequenceClassification,
)

from sentiment.models.transformer import (  # noqa: E402
    TransformerPredictor,
    encode,
    history_from_log,
    model_card,
    special_token_frame,
    train_transformer,
    truncate_ids,
)
from sentiment.predict import load_predictor  # noqa: E402
from sentiment.utils import RunConfig  # noqa: E402

WORDS = ["film", "je", "odličan", "sjajan", "loš", "dosadan", "glumac", "scena", "nije"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A tiny random ELECTRA classifier + WordPiece tokenizer saved locally (no network)."""
    root = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", ".", "!", *WORDS]
    # transformers 5.x takes the vocabulary as a dict (``vocab_file`` is ignored).
    # Cased like BERTić: lower-casing would also strip diacritics (odličan -> odlican).
    tokenizer = BertTokenizerFast(vocab={t: i for i, t in enumerate(vocab)}, do_lower_case=False)
    torch.manual_seed(0)
    config = ElectraConfig(
        vocab_size=len(vocab), embedding_size=16, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64, num_labels=2,
        id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1},
    )  # fmt: skip
    model = ElectraForSequenceClassification(config)
    model.config.sentiment_max_length = 8
    model.config.sentiment_truncation = "head_tail"
    model.save_pretrained(root)
    tokenizer.save_pretrained(root)
    return root


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("head", [0, 1, 2, 3]), ("tail", [6, 7, 8, 9]), ("head_tail", [0, 1, 8, 9])],
)
def test_truncate_ids(strategy, expected):
    assert truncate_ids(list(range(10)), 4, strategy) == expected
    assert truncate_ids([1, 2], 4, strategy) == [1, 2]


def test_special_token_frame_and_encode(tiny_model_dir):
    tok = BertTokenizerFast.from_pretrained(tiny_model_dir)
    prefix, suffix = special_token_frame(tok)
    assert prefix == [tok.cls_token_id] and suffix == [tok.sep_token_id]

    text = "film je odličan sjajan loš dosadan glumac scena nije film"  # 10 tokens
    enc = encode([text, "film"], tok, max_length=6, strategy="head_tail")
    ids = enc["input_ids"][0]
    assert len(ids) == 6
    assert tok.convert_ids_to_tokens(ids) == ["[CLS]", "film", "je", "nije", "film", "[SEP]"]
    assert enc["attention_mask"][1] == [1, 1, 1]  # short text is not padded here
    head = encode([text], tok, max_length=6, strategy="head")["input_ids"][0]
    assert tok.convert_ids_to_tokens(head)[1:-1] == ["film", "je", "odličan", "sjajan"]


def test_predictor_roundtrip_and_attention(tiny_model_dir, tmp_path):
    predictor = TransformerPredictor.load(tiny_model_dir, name="tiny")
    assert predictor.labels == ["negative", "positive"]
    assert (predictor.max_length, predictor.truncation) == (8, "head_tail")
    texts = ["film je odličan", "film je loš i dosadan glumac scena nije film"]
    proba = predictor.predict_proba(texts, batch_size=1)
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # Batching must not change results (padding is masked).
    np.testing.assert_allclose(predictor.predict_proba(texts, batch_size=2), proba, atol=1e-5)

    tokens, weights = predictor.attention(texts[1])
    assert tokens[0] == "[CLS]" and len(tokens) == 8 == len(weights)
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)

    predictor.save(tmp_path / "saved", {"note": "x"})
    loaded = load_predictor(tmp_path / "saved")
    assert isinstance(loaded, TransformerPredictor) and loaded.name == "tiny"
    np.testing.assert_allclose(loaded.predict_proba(texts), proba, atol=1e-5)


def test_history_from_log():
    log = [
        {"loss": 0.7, "epoch": 0.5},
        {"loss": 0.6, "epoch": 1.0},
        {"eval_loss": 0.65, "eval_macro_f1": 0.6, "eval_accuracy": 0.6, "epoch": 1.0},
        {"loss": 0.4, "epoch": 2.0},
        {"eval_loss": 0.55, "eval_macro_f1": 0.7, "eval_accuracy": 0.7, "epoch": 2.0},
        {"train_runtime": 10.0, "epoch": 2.0},
    ]
    history = history_from_log(log)
    assert history["epoch"].tolist() == [1, 2]
    assert history["train_loss"].tolist() == pytest.approx([0.65, 0.4])
    assert history["val_macro_f1"].tolist() == [0.6, 0.7]


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = "positive" if i % 2 else "negative"
        cue = ["odličan", "sjajan"] if i % 2 else ["loš", "dosadan"]
        words = list(rng.choice(["film", "je", "glumac", "scena"], 3)) + cue
        rng.shuffle(words)
        rows.append({"id": i, "text": " ".join(words), "label": label, "source": "toy"})
    return pd.DataFrame(rows)


def test_train_transformer_end_to_end(tiny_model_dir, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _toy(48, 0).to_parquet(data / "train.parquet")
    _toy(16, 1).to_parquet(data / "val.parquet")
    config = RunConfig(
        model="bertic",
        output_dir=tmp_path / "artifacts",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_artifact_root=tmp_path / "mlruns",
        data={"train": data / "train.parquet", "val": data / "val.parquet"},
        params={
            "model_name": str(tiny_model_dir),
            "max_length": 8,
            "learning_rate": 1e-3,
            "num_train_epochs": 3,
            "train_batch_size": 8,
            "early_stopping_patience": None,
            "latency_samples": 2,
        },
    )
    summary = train_transformer(config, report_dir=tmp_path / "reports", run_name="test")

    model_dir = tmp_path / "artifacts" / "model"
    assert (model_dir / "model.safetensors").exists()
    assert not (tmp_path / "artifacts" / "checkpoints").exists()  # cleaned up
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["type"] == "transformer" and meta["epochs_run"] == 3
    assert summary["best_epoch"] in {1, 2, 3}
    history = pd.read_csv(tmp_path / "reports" / "history.csv")
    assert len(history) == 3 and {"val_loss", "val_macro_f1", "train_loss"} <= set(history)
    assert (tmp_path / "reports" / "learning_curves.png").exists()
    assert "fine-tuning report" in (tmp_path / "reports" / "training.md").read_text()

    predictor = load_predictor(model_dir)
    cfg = predictor.model.config
    assert (cfg.sentiment_max_length, cfg.sentiment_truncation) == (8, "head_tail")


def test_push_requires_repo_id(tmp_path):
    config = RunConfig(model="bertic", params={"push_to_hub": True})
    with pytest.raises(ValueError, match="hub_model_id"):
        train_transformer(config)


def test_model_card_contains_metrics():
    summary = {
        "model_name": "classla/bcms-bertic",
        "val_metrics": {"macro_f1": 0.9123, "accuracy": 0.9},
        "max_length": 256,
        "truncation": "head_tail",
        "best_epoch": 2,
        "epochs_run": 4,
        "train_seconds": 600,
        "device": "cuda (T4)",
        "hub_model_id": "user/repo",
        "n_val": 169,
    }
    card = model_card(summary)
    assert card.startswith("---\nlanguage: sr")
    assert "0.9123" in card and "user/repo" in card and "CC BY-NC-SA" in card
