import gzip
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sentiment.models.neural.dataset import ReviewDataset, collate, truncate  # noqa: E402
from sentiment.models.neural.embeddings import (  # noqa: E402
    EmbeddingsMissingError,
    build_embedding_matrix,
    read_fasttext,
)
from sentiment.models.neural.model import BiLSTMClassifier  # noqa: E402
from sentiment.models.neural.tokenizer import PAD_ID, UNK_ID, Vocabulary, tokenize  # noqa: E402
from sentiment.models.neural.train import (  # noqa: E402
    BiLSTMParams,
    BiLSTMPredictor,
    build_model,
    fit,
    train_bilstm,
)
from sentiment.predict import load_predictor  # noqa: E402
from sentiment.utils import RunConfig  # noqa: E402


# --- Tokenizer and vocabulary ---------------------------------------------------
def test_tokenize():
    assert tokenize("Film NIJE loš?! Ocena: 7/10, rock-n-roll") == [
        "film", "nije", "loš", "?", "!", "ocena", "7", "10", "rock-n-roll",
    ]  # fmt: skip


def test_vocabulary_build_and_encode():
    vocab = Vocabulary.build(["a b b c", "b c d"], min_freq=2)
    assert vocab.itos[:2] == ["<pad>", "<unk>"]
    assert vocab.itos[2:] == ["b", "c"]  # by frequency, rare tokens dropped
    assert vocab.encode(["b", "zzz"]) == [2, UNK_ID]
    assert "a" not in vocab


def test_vocabulary_is_deterministic_on_ties():
    assert Vocabulary.build(["y x", "x y"], min_freq=1).itos == ["<pad>", "<unk>", "x", "y"]


def test_vocabulary_max_size_and_roundtrip(tmp_path):
    vocab = Vocabulary.build(["a a a b b c"], min_freq=1, max_size=2)
    assert vocab.itos[2:] == ["a", "b"]
    vocab.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").itos == vocab.itos


# --- Truncation and dataset -----------------------------------------------------
@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("head", [0, 1, 2, 3]), ("tail", [6, 7, 8, 9]), ("head_tail", [0, 1, 8, 9])],
)
def test_truncate(strategy, expected):
    assert truncate(list(range(10)), 4, strategy) == expected


def test_truncate_short_sequence_unchanged():
    assert truncate([1, 2], 4, "head_tail") == [1, 2]


def test_truncate_odd_length():
    assert truncate(list(range(10)), 5, "head_tail") == [0, 1, 7, 8, 9]


def test_dataset_and_collate_pad_batches():
    vocab = Vocabulary.build(["dobar film", "loš film"], min_freq=1)
    ds = ReviewDataset(
        ["dobar film", "loš", ""], ["positive", "negative", "negative"], vocab,
        {"negative": 0, "positive": 1}, max_len=10,
    )  # fmt: skip
    ids, lengths, labels = collate([ds[0], ds[1], ds[2]])
    assert ids.shape == (3, 2)
    assert lengths.tolist() == [2, 1, 1]  # empty text becomes a single <pad>
    assert ids[1, 1].item() == PAD_ID
    assert labels.tolist() == [1, 0, 0]


# --- Embeddings -------------------------------------------------------------------
def _write_vec(path, rows, dim=4):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(f"{len(rows)} {dim}\n")
        for word, value in rows:
            f.write(word + " " + " ".join([str(value)] * dim) + "\n")


def test_read_fasttext_transliterates_and_prefers_first(tmp_path):
    path = tmp_path / "v.vec.gz"
    _write_vec(path, [("Филм", 1.0), ("film", 2.0), ("loš", 3.0), ("other", 4.0)])
    vocab = Vocabulary(["film", "loš", "missing"])
    vectors = read_fasttext(path, vocab, transliterate=True)
    assert set(vectors) == {vocab.stoi["film"], vocab.stoi["loš"]}
    assert vectors[vocab.stoi["film"]][0] == 1.0  # Cyrillic entry came first
    without = read_fasttext(path, vocab, transliterate=False)
    assert without[vocab.stoi["film"]][0] == 2.0


def test_build_embedding_matrix_and_cache(tmp_path):
    path = tmp_path / "v.vec.gz"
    _write_vec(path, [("film", 1.0), ("loš", -1.0)])
    vocab = Vocabulary(["film", "loš", "missing"])
    emb = build_embedding_matrix(vocab, path, dim=4, cache_dir=tmp_path)
    assert emb.weights.shape == (5, 4)
    assert emb.found == 2 and emb.coverage_types == pytest.approx(2 / 3)
    np.testing.assert_array_equal(emb.weights[PAD_ID], 0)
    np.testing.assert_array_equal(emb.weights[vocab.stoi["film"]], 1.0)
    assert list(tmp_path.glob("fasttext_*.npz"))
    path.unlink()  # second call must use the cache
    again = build_embedding_matrix(vocab, path, dim=4, cache_dir=tmp_path)
    np.testing.assert_array_equal(again.weights, emb.weights)


def test_missing_embeddings_explain_download(tmp_path):
    with pytest.raises(EmbeddingsMissingError, match="cc.sr.300.vec.gz"):
        read_fasttext(tmp_path / "cc.sr.300.vec.gz", Vocabulary(["a"]))


# --- Model ------------------------------------------------------------------------
@pytest.mark.parametrize("pooling", ["last", "max", "mean", "max_mean"])
def test_model_output_shape_and_padding_invariance(pooling):
    torch.manual_seed(0)
    model = BiLSTMClassifier(20, 3, embedding_dim=8, hidden_size=4, pooling=pooling).eval()
    ids = torch.tensor([[2, 3, 4], [5, 6, PAD_ID]])
    logits = model(ids, torch.tensor([3, 2]))
    assert logits.shape == (2, 3)
    # Extra padding must not change the result for the shorter sequence.
    padded = torch.tensor([[5, 6, PAD_ID, PAD_ID, PAD_ID]])
    torch.testing.assert_close(model(padded, torch.tensor([2]))[0], logits[1])


def test_frozen_embeddings_are_not_trainable():
    pretrained = np.ones((10, 8), dtype=np.float32)
    model = BiLSTMClassifier(10, 2, embedding_dim=8, pretrained=pretrained, freeze_embeddings=True)
    assert not model.embedding.weight.requires_grad
    torch.testing.assert_close(model.embedding.weight[3], torch.ones(8))
    counts = model.count_parameters()
    assert counts["total"] - counts["trainable"] == 80


def test_pretrained_shape_mismatch():
    with pytest.raises(ValueError, match="Pretrained shape"):
        BiLSTMClassifier(10, 2, embedding_dim=8, pretrained=np.zeros((9, 8), np.float32))


# --- Training ---------------------------------------------------------------------
POS, NEG = ["odličan", "sjajan", "remek"], ["loš", "dosadan", "očajno"]
FILLER = ["film", "glumac", "scena", "priča", "reditelj"]


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = "positive" if i % 2 else "negative"
        words = list(rng.choice(FILLER, 6)) + list(rng.choice(POS if i % 2 else NEG, 2))
        rng.shuffle(words)
        rows.append({"id": i, "text": " ".join(words), "label": label, "source": "toy"})
    return pd.DataFrame(rows)


SMALL = {
    "embedding_path": None,
    "embedding_dim": 16,
    "hidden_size": 8,
    "batch_size": 8,
    "lr": 0.01,
    "max_epochs": 15,
    "patience": 3,
    "min_freq": 1,
    "latency_samples": 2,
    "num_threads": 1,
}


def _loaders(params, train, val, vocab):
    from sentiment.models.neural.train import _seeded_loader

    lab = {"negative": 0, "positive": 1}
    tr = ReviewDataset(train["text"], train["label"], vocab, lab, params.max_len)
    va = ReviewDataset(val["text"], val["label"], vocab, lab, params.max_len)
    return _seeded_loader(tr, params.batch_size, True, 0), _seeded_loader(va, 16, False, 0)


def test_fit_learns_toy_task_and_restores_best_epoch():
    torch.manual_seed(0)
    train, val = _toy(64, 0), _toy(32, 1)
    params = BiLSTMParams(**SMALL)
    vocab = Vocabulary.build(train["text"], min_freq=1)
    model = build_model(params, len(vocab), 2, pretrained=None)
    result = fit(model, *_loaders(params, train, val, vocab), ["negative", "positive"], params)
    history = result.history
    assert history["val_macro_f1"].max() > 0.9
    best = history.loc[history["val_loss"].idxmin()]
    assert result.best_epoch == best["epoch"]
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]


def test_early_stopping_triggers():
    torch.manual_seed(0)
    train, val = _toy(32, 0), _toy(16, 1)
    # Monitoring F1 on a trivially separable task plateaus at 1.0 quickly.
    params = BiLSTMParams(**{**SMALL, "max_epochs": 50, "patience": 2, "monitor": "val_macro_f1"})
    vocab = Vocabulary.build(train["text"], min_freq=1)
    model = build_model(params, len(vocab), 2, pretrained=None)
    result = fit(model, *_loaders(params, train, val, vocab), ["negative", "positive"], params)
    assert result.stopped_early
    assert len(result.history) < 50
    assert len(result.history) == result.best_epoch + params.patience


def test_train_bilstm_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _toy(64, 0).to_parquet(data / "train.parquet")
    _toy(32, 1).to_parquet(data / "val.parquet")
    config = RunConfig(
        model="bilstm",
        output_dir=tmp_path / "artifacts",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_artifact_root=tmp_path / "mlruns",
        data={"train": data / "train.parquet", "val": data / "val.parquet"},
        params=SMALL,
    )
    summary = train_bilstm(config, report_dir=tmp_path / "reports", run_name="test")
    out = tmp_path / "artifacts"
    for name in ("model.pt", "vocab.json", "metadata.json", "history.csv", "learning_curves.png"):
        assert (out / name).exists(), name
    assert (tmp_path / "reports" / "learning_curves.png").exists()
    assert "Learning curves" in (tmp_path / "reports" / "training.md").read_text()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["type"] == "bilstm" and meta["model_size_mb"] > 0

    predictor = load_predictor(out)
    assert isinstance(predictor, BiLSTMPredictor)
    proba = predictor.predict_proba(["odličan sjajan film", "loš dosadan film"])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    texts = ["odličan sjajan film", "loš dosadan film"]
    assert predictor.predict(texts) == ["positive", "negative"]
    assert summary["val_metrics"]["macro_f1"] > 0.9


def test_train_script_overrides():
    from scripts.train import apply_overrides

    base = RunConfig(model="bilstm", params={"lr": 0.001})
    new = apply_overrides(base, ["params.lr=0.003", "seed=7", "params.truncation=head"])
    assert new.params["lr"] == 0.003 and new.seed == 7 and new.params["truncation"] == "head"
    assert base.params["lr"] == 0.001
    with pytest.raises(ValueError):
        apply_overrides(base, ["no_equals_sign"])
