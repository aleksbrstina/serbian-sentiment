import pandas as pd
import pytest

from sentiment.data.cleaning import (
    clean_records,
    cyrillic_to_latin,
    dedup_key,
    normalize_diacritics,
    normalize_punctuation,
    normalize_series,
    normalize_text,
    normalize_whitespace,
    remove_invisible,
    strip_diacritics,
    strip_html,
)


# --- Cyrillic -> Latin -------------------------------------------------------
@pytest.mark.parametrize(
    ("cyr", "lat"),
    [
        ("љубав", "ljubav"),
        ("њега", "njega"),
        ("џеп", "džep"),
        ("ђак", "đak"),
        ("ћуприја", "ćuprija"),
        ("Чаша жуте шљиве", "Čaša žute šljive"),
        ("Љубав", "Ljubav"),  # title case digraph
        ("ЉУБАВ", "LJUBAV"),  # all caps digraph
        ("Њ", "Nj"),  # isolated letter
        ("ЏЕП", "DŽEP"),
        ("Филм је одличан!", "Film je odličan!"),
    ],
)
def test_cyrillic_to_latin(cyr, lat):
    assert cyrillic_to_latin(cyr) == lat


def test_cyrillic_to_latin_keeps_latin_and_mixed_text():
    assert cyrillic_to_latin("Film 2024: супер") == "Film 2024: super"
    assert cyrillic_to_latin("već latinica") == "već latinica"


def test_full_serbian_alphabet_has_no_cyrillic_left():
    alphabet = "абвгдђежзијклљмнњопрстћуфхцчџш"
    out = cyrillic_to_latin(alphabet + alphabet.upper())
    assert not any("Ѐ" <= ch <= "ӿ" for ch in out)


# --- Diacritics ----------------------------------------------------------------
def test_combining_diacritics_are_composed():
    decomposed = "čaš i žut"  # c + caron, s + caron, z + caron
    assert normalize_diacritics(decomposed) == "čaš i žut"


def test_combining_acute_composes_to_c_acute():
    assert normalize_diacritics("kuća") == "kuća"


def test_eth_lookalike_becomes_dj():
    assert normalize_diacritics("ðak Ðoković") == "đak Đoković"


def test_dj_digraph_is_not_converted():
    assert normalize_diacritics("nadjačati") == "nadjačati"


def test_strip_diacritics():
    assert strip_diacritics("Čačak, Đorđe, šuma, žaba, ćup") == "Cacak, Djordje, suma, zaba, cup"


# --- HTML, invisible characters, punctuation, whitespace ----------------------------
def test_strip_html_tags_and_entities():
    out = strip_html("Odličan<br/>film &amp; gluma &quot;top&quot;")
    assert out == 'Odličan film & gluma "top"'
    assert strip_html("<p>Dobar</p><b>film</b>") == " Dobarfilm"


def test_strip_html_keeps_comparisons():
    assert strip_html("ocena 3 < 5 i 7 > 4") == "ocena 3 < 5 i 7 > 4"


def test_remove_invisible():
    assert remove_invisible("﻿Film​ je\x9e dobar") == "Film je dobar"


def test_normalize_punctuation():
    assert normalize_punctuation("„Film“ – ‘dobar’…") == "\"Film\" - 'dobar'..."


def test_normalize_whitespace():
    assert normalize_whitespace("  a \n\t b   c ") == "a b c"


def test_normalize_text_full_pipeline():
    raw = "﻿<p>Филм је  „одличан“</p>&nbsp;—  čista 10!"
    assert normalize_text(raw) == 'Film je "odličan" - čista 10!'


def test_normalize_text_is_idempotent():
    raw = "<b>Ђорђе</b> &amp; „ЉУБАВ“ …   test"
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_normalize_series_counts_changes_per_step():
    texts = pd.Series(["Филм", "plain", "a  b", "<i>x</i>"])
    out, changed = normalize_series(texts)
    assert out.tolist() == ["Film", "plain", "a b", "x"]
    assert changed["cyrillic_to_latin"] == 1
    assert changed["whitespace"] == 1
    assert changed["html"] == 1
    assert changed["diacritics"] == 0


# --- Record cleaning -------------------------------------------------------------
def _df(rows):
    return pd.DataFrame(rows, columns=["id", "text", "label"])


def test_dedup_key_ignores_case_and_punctuation():
    assert dedup_key("Odličan film!!!") == dedup_key("odličan, FILM")


def test_clean_records_counts_each_reason():
    df = _df(
        [
            (0, "dobar film", "positive"),
            (1, "dobar film", "positive"),  # exact duplicate
            (2, "Dobar film!", "positive"),  # near duplicate
            (3, "", "negative"),  # empty
            (4, "bez oznake", None),  # missing label
            (5, "sporno", "positive"),  # conflicting
            (6, "sporno", "negative"),  # conflicting
            (7, "loš film", "negative"),
        ]
    )
    result = clean_records(df)
    assert result.counts == {
        "empty_text": 1,
        "missing_label": 1,
        "conflicting_label": 2,
        "exact_duplicate": 1,
        "near_duplicate": 1,
    }
    assert sorted(result.data["id"]) == [0, 7]
    assert len(result.removed) == 6
    assert set(result.removed["reason"]) == set(result.counts)
    assert len(result.data) + len(result.removed) == len(df)


def test_clean_records_keeps_first_duplicate():
    df = _df([(10, "isti tekst", "positive"), (11, "isti tekst", "positive")])
    result = clean_records(df)
    assert result.data["id"].tolist() == [10]
    assert result.removed["id"].tolist() == [11]


def test_clean_records_no_removals():
    df = _df([(0, "a", "positive"), (1, "b", "negative")])
    result = clean_records(df)
    assert len(result.data) == 2
    assert len(result.removed) == 0
    assert "reason" in result.removed.columns
    assert sum(result.counts.values()) == 0
