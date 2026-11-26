import pandas as pd
from scripts.prepare_data import main

from sentiment.data.pipeline import DataConfig, prepare_data, save_outputs


def _make_raw(raw_dir, n=60):
    rows = [
        (f"Филм број {i} је {'добар' if i % 2 else 'лош'}.", "POSITIVE" if i % 2 else "NEGATIVE")
        for i in range(n)
    ]
    rows += [rows[0], ("", "POSITIVE")]  # one duplicate, one empty
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Text", "class-att"]).to_csv(raw_dir / "SerbMR-2C.csv", index=False)


def _config(tmp_path):
    return DataConfig(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_path=tmp_path / "reports" / "data_quality.md",
    )


def test_end_to_end_pipeline(tmp_path):
    _make_raw(tmp_path / "raw")
    config = _config(tmp_path)
    result = prepare_data(config)
    paths = save_outputs(result, config)

    assert result.removal_counts["exact_duplicate"] == 1
    assert result.removal_counts["empty_text"] == 1
    assert result.normalization_changes["cyrillic_to_latin"] == 61

    splits = {n: pd.read_parquet(paths[n]) for n in ("train", "val", "test")}
    assert sum(len(s) for s in splits.values()) == 60
    assert list(splits["train"].columns) == ["id", "text", "label", "source"]
    assert splits["train"]["text"].str.contains("Film").all()

    report = paths["report"].read_text(encoding="utf-8")
    assert "Records kept: **60** of 62" in report
    assert paths["config"].exists()


def test_script_prints_instructions_when_data_missing(tmp_path, capsys):
    cfg = tmp_path / "data.yaml"
    cfg.write_text(f"raw_dir: {tmp_path / 'raw'}\n", encoding="utf-8")
    assert main(["--config", str(cfg)]) == 1
    assert "releases/tag/v1.0" in capsys.readouterr().err


def test_split_fingerprint_detects_changes(tmp_path):
    from sentiment.data.pipeline import split_fingerprint

    _make_raw(tmp_path / "raw")
    result = prepare_data(_config(tmp_path))
    fp = split_fingerprint(result.splits)
    assert fp == split_fingerprint(prepare_data(_config(tmp_path)).splits)
    swapped = dict(result.splits)
    swapped["val"], swapped["test"] = result.splits["test"], result.splits["val"]
    assert split_fingerprint(swapped) != fp
