"""Plots for exploratory data analysis. Each function returns a matplotlib Figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from sentiment.eda.stats import add_length_columns, labels_in

#: Consistent class colors across all figures.
LABEL_COLORS: dict[str, str] = {
    "negative": "#c0392b",
    "neutral": "#7f8c8d",
    "positive": "#2471a3",
}


def _palette(labels: list[str]) -> list[str]:
    return [LABEL_COLORS.get(lab, "#555555") for lab in labels]


def plot_class_distribution(distribution: pd.DataFrame) -> Figure:
    """Grouped bar chart of class counts per split.

    Args:
        distribution: Output of :func:`sentiment.eda.stats.class_distribution`.
    """
    labels = [c for c in distribution.columns if c != "total"]
    long = (
        distribution[labels]
        .reset_index(names="split")
        .melt(id_vars="split", var_name="label", value_name="count")
    )
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(data=long, x="split", y="count", hue="label", palette=_palette(labels), ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    ax.set(title="Class distribution per split", xlabel="", ylabel="Reviews")
    ax.legend(title="")
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_length_distribution(df: pd.DataFrame, truncation: int | None = 256) -> Figure:
    """Histogram of review length in words per class, with an optional cut-off line.

    Args:
        df: Data with ``text`` and ``label``.
        truncation: If given, draw a vertical line at this word count (transformer
            max sequence length) to show how many reviews would be truncated.
    """
    data = add_length_columns(df)
    labels = labels_in(data)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.histplot(
        data=data,
        x="n_words",
        hue="label",
        hue_order=labels,
        palette=_palette(labels),
        bins=50,
        element="step",
        alpha=0.3,
        ax=ax,
    )
    for lab, color in zip(labels, _palette(labels), strict=True):
        median = data.loc[data["label"] == lab, "n_words"].median()
        ax.axvline(median, color=color, linestyle="--", linewidth=1)
    if truncation:
        ax.axvline(truncation, color="black", linewidth=1.2)
        ax.annotate(
            f"{truncation} words",
            xy=(truncation, ax.get_ylim()[1] * 0.9),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(title="Review length (dashed: class medians)", xlabel="Words", ylabel="Reviews")
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_informative_words(table: pd.DataFrame, n: int = 15) -> Figure:
    """Horizontal bar charts of the top chi-square words, one panel per class.

    Args:
        table: Output of :func:`sentiment.eda.stats.informative_words`.
        n: Words per class to show.
    """
    labels = list(dict.fromkeys(table["label"]))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.5 * len(labels), 0.32 * n + 1.2))
    axes = [axes] if len(labels) == 1 else list(axes)
    for ax, lab in zip(axes, labels, strict=True):
        sub = table[table["label"] == lab].head(n).iloc[::-1]
        ax.barh(sub["word"], sub["chi2"], color=LABEL_COLORS.get(lab, "#555555"))
        ax.set(title=f"Most informative: {lab}", xlabel="χ² score")
        ax.tick_params(axis="y", labelsize=9)
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_cue_rates(cues: pd.DataFrame) -> Figure:
    """Grouped bar chart of lexical cue frequency (per 100 words) by class.

    Args:
        cues: Output of :func:`sentiment.eda.stats.cue_rates`.
    """
    cols = [c for c in cues.columns if c.startswith("per100_")]
    labels = [c.removeprefix("per100_") for c in cols]
    long = (
        cues[cols]
        .rename(columns=dict(zip(cols, labels, strict=True)))
        .reset_index(names="cue")
        .melt(id_vars="cue", var_name="label", value_name="per100")
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=long, y="cue", x="per100", hue="label", palette=_palette(labels), ax=ax)
    ax.set(title="Lexical cues by class", xlabel="Occurrences per 100 words", ylabel="")
    ax.legend(title="")
    sns.despine(fig)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure to ``path`` (creating parent directories) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
