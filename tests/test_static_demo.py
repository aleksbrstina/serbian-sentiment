"""Parity between the browser demo (app/static/sentiment.js) and the Python pipeline."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from sentiment.data.cleaning import normalize_text
from sentiment.export_static import export_pipeline, write_export
from sentiment.models.classical import VectorizerParams, build_classifier, build_features

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "app" / "static" / "sentiment.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

TRICKY = [
    "Odličan film, sjajna gluma!!!",
    "Филм је ОДЛИЧАН, а ЉУБАВ и Џеп... Љубав",
    "Loš &amp; dosadan &quot;film&quot;<br/>nažalost&#33; &nbsp;kraj",
    "čista žuta kuća, ðak Ðoković",
    "﻿NULL​byte\x9e  tabs\tand\nnewlines  here",
    "„Navodnici“ – crtice … i ‘apostrofi’",
    "a b c 7 10/10 rock-n-roll x",
    "",
    "      ",
    "Ne nije nisam nisu odličan loš loš loš",
]


def _toy_pipeline(word_ngram_max: int = 1) -> Pipeline:
    rng = np.random.default_rng(0)
    pos = ["odličan", "sjajan", "remek", "ljubav", "čista"]
    neg = ["loš", "dosadan", "očajno", "nažalost", "kraj"]
    filler = ["film", "je", "gluma", "scena", "priča", "žuta", "kuća", "đak"]
    texts, labels = [], []
    for i in range(80):
        cue = pos if i % 2 else neg
        words = list(rng.choice(filler, 6)) + list(rng.choice(cue, 2))
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append("positive" if i % 2 else "negative")
    vec = VectorizerParams(
        word_ngram_max=word_ngram_max, char_ngrams="3-5", min_df=1, sublinear_tf=True
    )
    clf = build_classifier("logreg", {"C": 5.0}, 0)
    pipe = Pipeline([("features", build_features(vec)), ("clf", clf)])
    return pipe.fit(texts, labels)


def _run_node(model_path: Path, texts: list[str], tmp_path: Path) -> list[dict]:
    runner = tmp_path / "run.js"
    runner.write_text(
        f"""
const S = require({json.dumps(str(JS))});
const fs = require("fs");
const model = JSON.parse(fs.readFileSync({json.dumps(str(model_path))}, "utf8"));
const texts = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = texts.map((t) => {{
  const r = S.predict(model, t);
  return {{ normalized: r.normalized, p: r.probs.positive, label: r.label,
           words: Object.fromEntries(r.wordContributions) }};
}});
process.stdout.write(JSON.stringify(out));
""",
        encoding="utf-8",
    )
    inputs = tmp_path / "texts.json"
    inputs.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [NODE, str(runner), str(inputs)], capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("word_ngram_max", [1, 2])
def test_js_matches_python(tmp_path, word_ngram_max):
    pipe = _toy_pipeline(word_ngram_max)
    model_path = write_export(pipe, tmp_path / "model.json")
    js = _run_node(model_path, TRICKY, tmp_path)

    normalized = [normalize_text(t) for t in TRICKY]
    assert [r["normalized"] for r in js] == normalized

    p_py = pipe.predict_proba(normalized)[:, list(pipe.classes_).index("positive")]
    p_js = np.array([r["p"] for r in js])
    np.testing.assert_allclose(p_js, p_py, atol=1e-5)
    assert [r["label"] for r in js] == list(pipe.predict(normalized))


def test_js_word_contributions_match_explain(tmp_path):
    from sentiment.models.classical import ClassicalPredictor

    pipe = _toy_pipeline()
    model_path = write_export(pipe, tmp_path / "model.json")
    text = "film je odličan ali loš kraj"
    js = _run_node(model_path, [text], tmp_path)[0]["words"]
    py = dict(ClassicalPredictor(pipe).explain(text, k=50))
    # explain() reports contributions towards the predicted class with a 2x weight gap
    # (binary weights are expanded to [-w, w]); sentiment.js reports pull towards positive.
    sign = 1.0 if pipe.predict([text])[0] == "positive" else -1.0
    assert set(js) == set(py)
    for word, value in js.items():
        assert py[word] == pytest.approx(2 * sign * value, abs=1e-5)


def test_export_rejects_unsupported(tmp_path):
    pipe = _toy_pipeline()
    pipe.named_steps["features"].transformer_list[1][1].analyzer = "char"
    with pytest.raises(ValueError, match="Unsupported analyzer"):
        export_pipeline(pipe)


def test_export_contains_normalization_tables():
    data = export_pipeline(_toy_pipeline())
    assert data["classes"] == ["negative", "positive"]
    assert data["normalization"]["cyr_to_lat"]["љ"] == "lj"
    assert data["normalization"]["cyr_upper_digraphs"]["Љ"] == ["Lj", "LJ"]
    assert sum(len(b["terms"]) for b in data["blocks"]) == len(
        _toy_pipeline().named_steps["clf"].coef_[0]
    )
