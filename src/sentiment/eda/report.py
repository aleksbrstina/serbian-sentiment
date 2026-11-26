"""Run the full EDA and render a Markdown report with numerically backed insights.

Word-level analysis uses the training split only, so nothing learned from the
validation/test data can leak into modelling decisions. Validation data is used
only to measure out-of-vocabulary rates, which is exactly what a model will face.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sentiment.eda import plots
from sentiment.eda.stats import (
    LengthComparison,
    VocabularyStats,
    class_distribution,
    cue_rates,
    informative_words,
    labels_in,
    length_stats,
    share_longer_than,
    top_frequent_words,
    vocabulary_stats,
    word_forms,
)

#: Inflected forms of "film" used to illustrate Serbian morphology.
FILM_FORMS = ("film", "filma", "filmu", "filmom", "filmovi", "filmova", "filmove", "filmovima")
TRUNCATION_THRESHOLDS = (128, 256, 512)
SIGNIFICANCE = 0.05


@dataclass
class EdaResult:
    """All EDA outputs; every number in the report comes from here."""

    distribution: pd.DataFrame
    lengths: LengthComparison
    truncation: dict[int, float]
    vocab: VocabularyStats
    forms: dict[str, int]
    frequent: dict[str, list[tuple[str, int]]]
    informative: pd.DataFrame
    cues: pd.DataFrame
    figures: dict[str, Path] = field(default_factory=dict)


def run_eda(splits: Mapping[str, pd.DataFrame], top_n: int = 20) -> EdaResult:
    """Compute all EDA statistics.

    Args:
        splits: Mapping with at least ``"train"`` and ``"val"``.
        top_n: Number of words per class in the word lists.

    Returns:
        Collected statistics.
    """
    train = splits["train"]
    return EdaResult(
        distribution=class_distribution(splits),
        lengths=length_stats(train),
        truncation=share_longer_than(train, TRUNCATION_THRESHOLDS),
        vocab=vocabulary_stats(train, splits["val"]),
        forms=word_forms(train, FILM_FORMS),
        frequent=top_frequent_words(train, n=top_n),
        informative=informative_words(train, n=top_n),
        cues=cue_rates(train),
    )


def save_figures(result: EdaResult, train: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    """Render and save all EDA figures; also stores the paths on ``result``."""
    out_dir = Path(out_dir)
    result.figures = {
        "class_distribution": plots.save_figure(
            plots.plot_class_distribution(result.distribution), out_dir / "class_distribution.png"
        ),
        "length_distribution": plots.save_figure(
            plots.plot_length_distribution(train), out_dir / "length_distribution.png"
        ),
        "informative_words": plots.save_figure(
            plots.plot_informative_words(result.informative), out_dir / "informative_words.png"
        ),
        "cue_rates": plots.save_figure(
            plots.plot_cue_rates(result.cues), out_dir / "cue_rates.png"
        ),
    }
    return result.figures


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #


def _pct(x: float) -> str:
    return f"{x:.1%}"


def insights(result: EdaResult) -> list[str]:
    """Generate data-driven insights. Wording adapts to the numbers.

    Args:
        result: Output of :func:`run_eda`.

    Returns:
        List of Markdown bullet texts, each backed by at least one number.
    """
    out: list[str] = []
    dist = result.distribution
    labels = [c for c in dist.columns if c != "total"]
    total = dist.loc[:, labels].sum()
    shares = total / total.sum()
    balance = ", ".join(f"{lab} {_pct(shares[lab])}" for lab in labels)
    if shares.max() - shares.min() < 0.05:
        out.append(
            f"**Classes are balanced** ({balance} of {int(total.sum())} reviews), and the "
            "stratified split preserves this in every split. Accuracy and macro F1 will "
            "therefore tell the same story, and no class re-weighting is needed."
        )
    else:
        out.append(
            f"**Classes are imbalanced** ({balance}). Macro F1 is the primary metric and "
            "class weighting should be considered."
        )

    lens = result.lengths
    medians = lens.summary["50%"]
    median_txt = " vs. ".join(f"{int(medians[lab])} ({lab})" for lab in medians.index)
    if lens.p_value >= SIGNIFICANCE:
        out.append(
            f"**Review length carries no sentiment signal.** Median length is {median_txt} "
            f"words; a {lens.test_name} test finds no significant difference "
            f"(p = {lens.p_value:.2f}). Models cannot shortcut through length and must rely "
            "on content."
        )
    else:
        out.append(
            f"**Review length differs between classes**: median {median_txt} words "
            f"({lens.test_name}, p = {lens.p_value:.3g}). Length is a weak but real signal."
        )

    trunc = result.truncation
    overall_median = int(lens.summary["50%"].mean())
    out.append(
        f"**Reviews are long, so transformer truncation matters.** The median review has "
        f"~{overall_median} words, and {_pct(trunc[256])} exceed 256 words "
        f"({_pct(trunc[512])} exceed 512). Subword tokenization only increases the token "
        "count, so a 256-token model sees just the opening part of nearly every review. "
        "Head+tail truncation or chunking should be evaluated for BERTić."
    )

    v = result.vocab
    forms = ", ".join(f"*{w}* {c}" for w, c in result.forms.items() if c)
    out.append(
        f"**Rich morphology produces a long-tailed vocabulary.** The training split has "
        f"{v.n_types:,} distinct word forms in {v.n_tokens:,} tokens; {_pct(v.hapax_share)} "
        f"of them occur only once. {_pct(v.oov_type_rate)} of word types "
        f"({_pct(v.oov_token_rate)} of tokens) in the validation split never appear in "
        f"training. One lemma alone appears in many forms ({forms}). This motivates "
        "character n-grams for TF-IDF and subword-aware fastText embeddings for the BiLSTM."
    )

    info = result.informative
    parts = []
    for lab in labels_in(info):
        others = [c for c in info.columns if c.startswith("doc_share_") and c != f"doc_share_{lab}"]
        sub = info[info["label"] == lab].head(3)
        words = []
        for _, row in sub.iterrows():
            other_share = max(row[c] for c in others)
            words.append(
                f"*{row['word']}* ({_pct(row[f'doc_share_{lab}'])} of {lab} vs. "
                f"{_pct(other_share)} of other reviews)"
            )
        parts.append(f"{lab}: " + ", ".join(words))
    out.append(
        "**A few evaluative words are strongly class-specific** (top χ² words, share of "
        "reviews containing them): " + "; ".join(parts) + ". Even the strongest cues "
        "occur in a minority of reviews, so no single keyword solves the task."
    )

    cues = result.cues
    if len(labels) >= 2:
        first, last = labels[0], labels[-1]
        neg_name = next(n for n in cues.index if n.startswith("negation"))
        q_name = "question mark"
        score_name = next(n for n in cues.index if n.startswith("explicit score"))
        neg, q = cues.loc[neg_name], cues.loc[q_name]
        score_docs = max(cues.loc[score_name, f"docs_{lab}"] for lab in labels)
        out.append(
            f"**Negation and rhetorical questions lean {first}, but are everywhere.** "
            f"Negation words occur {neg[f'per100_{first}']:.2f} times per 100 words in "
            f"{first} vs. {neg[f'per100_{last}']:.2f} in {last} reviews "
            f"(×{neg['ratio']:.2f}), and appear in {_pct(neg[f'docs_{last}'])} of {last} "
            f"reviews. Question marks are ×{q['ratio']:.2f} as frequent in {first} reviews. "
            "Unigram features cannot tell *dobar* from *nije dobar*, and bigrams only catch "
            "negation next to the word it modifies, so negation is expected to be a major "
            'error category for the classical models. Explicit scores such as "7/10" appear '
            f"in at most {_pct(score_docs)} of reviews per class, so label leakage through "
            "ratings is negligible."
        )
    return out


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def render_report(result: EdaResult, figure_dir: Path | None = None) -> str:
    """Render the EDA report as Markdown.

    Args:
        result: Output of :func:`run_eda` (with figures saved, if they should be linked).
        figure_dir: Directory the report lives in; figure links are made relative to it.

    Returns:
        Markdown document.
    """

    def fig(name: str, alt: str) -> str:
        if name not in result.figures:
            return ""
        path = result.figures[name]
        rel = path.relative_to(figure_dir) if figure_dir else path
        return f"![{alt}]({rel.as_posix()})"

    dist = result.distribution.reset_index(names="split")
    lens = result.lengths.summary.reset_index()[["label", "count", "mean", "50%", "min", "max"]]
    lens = lens.rename(columns={"50%": "median"})
    lens["count"] = lens["count"].astype(int)
    trunc = pd.DataFrame(
        {"Max words": list(result.truncation), "Reviews longer": list(result.truncation.values())}
    )
    trunc["Reviews longer"] = trunc["Reviews longer"].map(_pct)
    cues = result.cues.reset_index(names="cue")
    info = result.informative.copy()
    info["p_value"] = info["p_value"].map(lambda p: f"{p:.1e}")
    for c in [c for c in info.columns if c.startswith("doc_share_")]:
        info[c] = info[c].map(_pct)
    info["chi2"] = info["chi2"].map(lambda x: f"{x:.1f}")

    freq_rows = []
    max_len = max(len(v) for v in result.frequent.values())
    for i in range(max_len):
        freq_rows.append(
            {
                lab: f"{words[i][0]} ({words[i][1]})" if i < len(words) else ""
                for lab, words in result.frequent.items()
            }
        )
    v = result.vocab

    sections = [
        "# Exploratory data analysis",
        "_Generated by `scripts/run_eda.py` — do not edit by hand. Word-level statistics "
        "use the training split only._",
        "## Key insights",
        "\n".join(f"{i}. {text}" for i, text in enumerate(insights(result), start=1)),
        "## Class distribution",
        _table(dist),
        fig("class_distribution", "Class distribution"),
        "## Review length (training split, words)",
        _table(lens, ".1f"),
        f"{result.lengths.test_name} test for a length difference between classes: "
        f"p = {result.lengths.p_value:.3f}.",
        _table(trunc),
        fig("length_distribution", "Length distribution"),
        "## Vocabulary",
        _table(
            pd.DataFrame(
                {
                    "Metric": [
                        "Tokens (train)",
                        "Word types (train)",
                        "Hapax share (train)",
                        "OOV token rate (val)",
                        "OOV type rate (val)",
                    ],
                    "Value": [
                        f"{v.n_tokens:,}",
                        f"{v.n_types:,}",
                        _pct(v.hapax_share),
                        _pct(v.oov_token_rate),
                        _pct(v.oov_type_rate),
                    ],
                }
            )
        ),
        "## Most informative words (χ² on word presence)",
        _table(info),
        fig("informative_words", "Most informative words"),
        "## Most frequent words (stopwords removed)",
        _table(pd.DataFrame(freq_rows)),
        "## Lexical cues (occurrences per 100 words; share of reviews containing the cue)",
        _table(cues),
        fig("cue_rates", "Lexical cues"),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def write_report(
    result: EdaResult, splits: Mapping[str, pd.DataFrame], out_dir: str | Path
) -> dict[str, Path]:
    """Save figures, CSV word lists and the Markdown report to ``out_dir``.

    Returns:
        Mapping of output name to path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(save_figures(result, splits["train"], out_dir))
    paths["informative_words_csv"] = out_dir / "informative_words.csv"
    result.informative.to_csv(paths["informative_words_csv"], index=False)
    paths["frequent_words_csv"] = out_dir / "frequent_words.csv"
    pd.DataFrame(
        [(lab, w, c) for lab, words in result.frequent.items() for w, c in words],
        columns=["label", "word", "count"],
    ).to_csv(paths["frequent_words_csv"], index=False)
    paths["report"] = out_dir / "eda_report.md"
    paths["report"].write_text(render_report(result, out_dir), encoding="utf-8")
    return paths
