"""Export a fitted TF-IDF + logistic regression pipeline to JSON for the browser demo.

The static demo (``app/static/``) reimplements inference in plain JavaScript, so the
model must be described by data only: vocabularies, IDF weights, coefficients and the
text-normalization tables. Exporting the normalization tables from the Python
constants keeps a single source of truth for them.

A unit test checks that the JavaScript implementation reproduces
``pipeline.predict_proba`` on the same inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment.data import cleaning

EXPORT_VERSION = 1


def _vectorizer_spec(vec: TfidfVectorizer, coef: np.ndarray, digits: int) -> dict[str, Any]:
    if vec.analyzer not in {"word", "char_wb"}:
        raise ValueError(f"Unsupported analyzer {vec.analyzer!r}")
    if vec.token_pattern != r"(?u)\b\w\w+\b" and vec.analyzer == "word":
        raise ValueError("Only the default token_pattern is supported")
    if vec.norm != "l2" or not vec.use_idf or not vec.lowercase or vec.binary:
        raise ValueError("Only lower-cased, l2-normalized, IDF-weighted TF-IDF is supported")
    if vec.strip_accents or vec.preprocessor or vec.tokenizer:
        raise ValueError("Custom preprocessing is not supported")
    terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    return {
        "analyzer": vec.analyzer,
        "ngram_range": list(vec.ngram_range),
        "sublinear_tf": bool(vec.sublinear_tf),
        "terms": terms,
        "idf": [round(float(x), digits) for x in vec.idf_],
        "coef": [round(float(x), digits) for x in coef],
    }


def export_pipeline(pipeline: Pipeline, digits: int = 7) -> dict[str, Any]:
    """Describe a fitted ``features`` (FeatureUnion of TF-IDF) + ``clf`` pipeline as JSON data.

    Args:
        pipeline: Fitted pipeline with a binary :class:`LogisticRegression` as ``clf``.
        digits: Decimal places kept for IDF values and coefficients.

    Returns:
        JSON-serializable dict consumed by ``app/static/sentiment.js``.

    Raises:
        ValueError: If the pipeline uses features the browser implementation lacks.
    """
    clf = pipeline.named_steps["clf"]
    if not isinstance(clf, LogisticRegression) or len(clf.classes_) != 2:
        raise ValueError("Only binary LogisticRegression is supported")
    union = pipeline.named_steps["features"]
    if union.transformer_weights:
        raise ValueError("FeatureUnion transformer_weights are not supported")
    coef = clf.coef_[0]
    blocks, offset = [], 0
    for _, vec in union.transformer_list:
        size = len(vec.vocabulary_)
        blocks.append(_vectorizer_spec(vec, coef[offset : offset + size], digits))
        offset += size
    return {
        "version": EXPORT_VERSION,
        "classes": [str(c) for c in clf.classes_],
        "intercept": round(float(clf.intercept_[0]), digits),
        "blocks": blocks,
        "normalization": {
            "cyr_to_lat": cleaning._CYR_TO_LAT,
            "cyr_upper_digraphs": {k: list(v) for k, v in cleaning._CYR_UPPER_DIGRAPHS.items()},
            "diacritic_fixes": cleaning._DIACRITIC_FIXES,
            "punct_fixes": cleaning._PUNCT_FIXES,
        },
    }


def write_export(
    pipeline: Pipeline, path: str | Path, metadata: dict[str, Any] | None = None
) -> Path:
    """Export ``pipeline`` to a compact JSON file, optionally with display metadata."""
    data = export_pipeline(pipeline)
    if metadata:
        data["metadata"] = metadata
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path
