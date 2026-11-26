import pandas as pd
import pytest

from sentiment.data.loading import RawDataMissingError, load_serbmr, serbmr_path
from sentiment.data.split import stratified_split


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=["Text", "class-att"]).to_csv(path, index=False)


def test_load_serbmr_unified_format(tmp_path):
    path = tmp_path / "SerbMR-3C.csv"
    _write_csv(path, [('Dobar, "film".', "POSITIVE"), ("Loš", "NEGATIVE"), ("Ok", "NEUTRAL")])
    df = load_serbmr(path)
    assert list(df.columns) == ["id", "text", "label", "source"]
    assert df["label"].tolist() == ["positive", "negative", "neutral"]
    assert df["text"][0] == 'Dobar, "film".'
    assert (df["source"] == "SerbMR-3C").all()
    assert df["id"].tolist() == [0, 1, 2]


def test_load_serbmr_missing_label_becomes_nan(tmp_path):
    path = tmp_path / "x.csv"
    _write_csv(path, [("tekst", "")])
    assert load_serbmr(path)["label"].isna().all()


def test_load_serbmr_rejects_unknown_label(tmp_path):
    path = tmp_path / "x.csv"
    _write_csv(path, [("tekst", "MAYBE")])
    with pytest.raises(ValueError, match="unknown labels"):
        load_serbmr(path)


def test_load_serbmr_rejects_wrong_columns(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"review": ["a"], "y": ["POSITIVE"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_serbmr(path)


def test_missing_file_explains_how_to_download(tmp_path):
    with pytest.raises(RawDataMissingError, match="github.com/vukbatanovic/SerbMR"):
        load_serbmr(serbmr_path(tmp_path, "2C"))


def test_serbmr_path_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError):
        serbmr_path(tmp_path, "5C")


# --- Split ------------------------------------------------------------------------
@pytest.fixture
def balanced():
    n = 1000
    return pd.DataFrame(
        {
            "id": range(n),
            "text": [f"t{i}" for i in range(n)],
            "label": ["positive", "negative"] * (n // 2),
        }
    )


def test_split_sizes_and_disjointness(balanced):
    splits = stratified_split(balanced, seed=1)
    assert {k: len(v) for k, v in splits.items()} == {"train": 800, "val": 100, "test": 100}
    ids = [set(s["id"]) for s in splits.values()]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert set().union(*ids) == set(balanced["id"])


def test_split_is_stratified(balanced):
    for split in stratified_split(balanced, seed=1).values():
        share = split["label"].value_counts(normalize=True)["positive"]
        assert share == pytest.approx(0.5, abs=0.01)


def test_split_is_deterministic(balanced):
    a = stratified_split(balanced, seed=7)
    b = stratified_split(balanced, seed=7)
    c = stratified_split(balanced, seed=8)
    assert a["test"]["id"].tolist() == b["test"]["id"].tolist()
    assert a["test"]["id"].tolist() != c["test"]["id"].tolist()


def test_split_rejects_bad_sizes(balanced):
    with pytest.raises(ValueError):
        stratified_split(balanced, val_size=0.5, test_size=0.5)
