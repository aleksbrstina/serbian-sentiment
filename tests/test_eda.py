import pandas as pd
import pytest

from sentiment.eda.report import insights, render_report, run_eda, write_report
from sentiment.eda.stats import (
    class_distribution,
    cue_rates,
    informative_words,
    length_stats,
    share_longer_than,
    tokenize,
    top_frequent_words,
    vocabulary_stats,
)


@pytest.fixture
def toy():
    pos = [f"odličan film broj {i} zaista odličan" for i in range(20)]
    neg = [f"loš film broj {i} nije dobar? nažalost" for i in range(20)]
    return pd.DataFrame({"text": pos + neg, "label": ["positive"] * 20 + ["negative"] * 20})


def test_tokenize_keeps_letters_with_diacritics():
    assert tokenize("Odličan, ŽIVOT! 10/10 čak-i") == ["odličan", "život", "čak", "i"]


def test_class_distribution(toy):
    table = class_distribution({"train": toy, "val": toy.head(5)})
    assert list(table.columns) == ["negative", "positive", "total"]
    assert table.loc["train"].tolist() == [20, 20, 40]
    assert table.loc["val", "positive"] == 5


def test_length_stats_detects_no_difference_for_identical_lengths():
    labels = ["positive", "negative"] * 10
    df = pd.DataFrame({"text": ["a b c"] * 10 + ["d e f"] * 10, "label": labels})
    result = length_stats(df)
    assert result.summary.loc["positive", "50%"] == 3
    assert result.p_value > 0.05


def test_share_longer_than():
    df = pd.DataFrame({"text": ["a", "a b c", "a b c d e"], "label": ["positive"] * 3})
    assert share_longer_than(df, [2, 4]) == {2: pytest.approx(2 / 3), 4: pytest.approx(1 / 3)}


def test_top_frequent_words_removes_stopwords(toy):
    top = top_frequent_words(toy, n=3)
    assert top["positive"][0] == ("odličan", 40)
    assert all(w not in {"nije", "film"} for w, _ in top["negative"])


def test_informative_words_assigns_words_to_correct_class(toy):
    table = informative_words(toy, n=10, min_df=1)
    assert table[table["word"] == "odličan"]["label"].item() == "positive"
    assert table[table["word"] == "nažalost"]["label"].item() == "negative"
    assert table.loc[table["word"] == "odličan", "doc_share_positive"].item() == pytest.approx(1.0)
    film = table[table["word"] == "film"]  # equally common in both classes
    assert film["chi2"].item() == pytest.approx(0.0)


def test_vocabulary_stats_oov():
    train = pd.DataFrame({"text": ["a b b"], "label": ["positive"]})
    val = pd.DataFrame({"text": ["a c"], "label": ["positive"]})
    v = vocabulary_stats(train, val)
    assert (v.n_tokens, v.n_types) == (3, 2)
    assert v.hapax_share == 0.5
    assert v.oov_token_rate == 0.5
    assert v.oov_type_rate == 0.5


def test_cue_rates(toy):
    cues = cue_rates(toy)
    neg = cues.loc["negation (ne/nije/nisam/nisu)"]
    assert neg["docs_negative"] == 1.0
    assert neg["docs_positive"] == 0.0
    assert cues.loc["question mark", "per100_negative"] > 0


def test_full_eda_report(toy, tmp_path):
    splits = {"train": toy, "val": toy.head(10), "test": toy.tail(10)}
    result = run_eda(splits)
    assert len(insights(result)) >= 3
    paths = write_report(result, splits, tmp_path)
    report = paths["report"].read_text(encoding="utf-8")
    assert "## Key insights" in report
    assert "![Class distribution](class_distribution.png)" in report
    assert all(p.exists() for p in paths.values())
    assert "Key insights" in render_report(result)
