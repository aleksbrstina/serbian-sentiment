"""Gradio demo: one Serbian review, three models side by side (FR-13, FR-14, FR-16).

Models are loaded through ``sentiment.predict.load_predictor``. Each model's location
comes from an environment variable, falling back to the local ``artifacts/`` directory
and then to the Hugging Face Hub, so the same file runs locally and on HF Spaces.

Run locally:
    python app/app.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").exists():  # running from the repository without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from sentiment.data.cleaning import normalize_text  # noqa: E402
from sentiment.predict import Predictor, load_predictor  # noqa: E402

BASELINES_REPO = "aleksbrstina/serbian-sentiment-baselines"
BERTIC_REPO = "aleksbrstina/bertic-serbmr-sentiment"
LABELS_SR = {"positive": "pozitivno", "negative": "negativno", "neutral": "neutralno"}

EXAMPLES = [
    "Odličan film! Gluma je sjajna, a priča me je potpuno osvojila od prvog do poslednjeg kadra.",
    "Nažalost, film je dosadan, predugačak i traljavo režiran. Šteta vremena.",
    "Film nije loš.",
    "Specijalni efekti su impresivni, ali scenario je toliko slab da ni sjajna gluma ne može "
    "da spase ovaj film.",
    "Bravo za scenaristu: uspeo je da od zanimljive knjige napravi dva sata savršenog dosađivanja.",
]


@dataclass
class ModelSlot:
    """A model shown in the demo and where it came from."""

    key: str
    title: str
    predictor: Predictor
    source: str


def _resolve(env_var: str, local: Path, hub_subfolder: str | None, hub_repo: str) -> str:
    """Pick the model location: env var, then local artifacts, then the Hub."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    if local.exists():
        return str(local)
    from huggingface_hub import snapshot_download

    pattern = [f"{hub_subfolder}/*"] if hub_subfolder else None
    path = Path(snapshot_download(hub_repo, allow_patterns=pattern))
    return str(path / hub_subfolder) if hub_subfolder else str(path)


def load_models() -> list[ModelSlot]:
    """Load the three demo models (classical, BiLSTM, BERTić)."""
    specs = [
        ("classical", "TF-IDF + logistička regresija", "SENTIMENT_CLASSICAL",
         ROOT / "artifacts/classical/logreg", "logreg", BASELINES_REPO),
        ("bilstm", "BiLSTM (fastText)", "SENTIMENT_BILSTM",
         ROOT / "artifacts/bilstm", "bilstm", BASELINES_REPO),
        ("bertic", "BERTić (fino podešen)", "SENTIMENT_BERTIC",
         ROOT / "artifacts/bertic/model", None, BERTIC_REPO),
    ]  # fmt: skip
    slots = []
    for key, title, env_var, local, subfolder, repo in specs:
        source = _resolve(env_var, local, subfolder, repo)
        if key == "bertic" and not Path(source).exists():
            from sentiment.models.transformer import TransformerPredictor

            predictor: Predictor = TransformerPredictor.load(source, name=title)
        else:
            predictor = load_predictor(source)
        slots.append(ModelSlot(key, title, predictor, source))
    return slots


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #

_SEGMENT_RE = regex.compile(r"\p{L}+|[^\p{L}]+")


def classical_highlights(predictor: Any, text: str, k: int = 12) -> list[tuple[str, float | None]]:
    """Split ``text`` into segments, scoring words by their pull towards *negative*.

    Scores are in [-1, 1]. Gradio's HighlightedText draws positive scores red and
    negative scores blue, so the sign is flipped: red = pushes towards negative,
    blue = pushes towards positive (bigram contributions are split over their words).
    """
    contributions = predictor.explain(text, k=k)
    pred = predictor.predict([text])[0]
    sign = -1.0 if pred == "positive" else 1.0  # >0 means "towards negative" (drawn red)
    word_scores: dict[str, float] = {}
    for feature, value in contributions:
        parts = feature.split()
        for part in parts:
            word_scores[part] = word_scores.get(part, 0.0) + sign * value / len(parts)
    top = max((abs(v) for v in word_scores.values()), default=0.0) or 1.0
    out: list[tuple[str, float | None]] = []
    for segment in _SEGMENT_RE.findall(text):
        score = word_scores.get(segment.lower())
        out.append((segment, round(score / top, 3) if score is not None else None))
    return out


def attention_highlights(predictor: Any, text: str) -> list[tuple[str, float | None]]:
    """BERTić ``[CLS]`` attention per word, formatted for ``gr.HighlightedText``."""
    out: list[tuple[str, float | None]] = []
    for word, weight in predictor.word_attention(text):
        if weight is None:  # truncation marker
            out.append((f" {word} ", None))
        else:
            out.append((f"{word} ", round(weight, 3)))
    return out


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def analyze(text: str, models: list[ModelSlot]) -> dict[str, Any]:
    """Run every model on one text.

    Returns:
        Dict with per-model ``probs`` (Serbian label -> probability), ``label``,
        ``ms`` (latency), the classical highlights and BERTić attention.
    """
    clean = normalize_text(text or "")
    if not clean:
        return {"empty": True}
    result: dict[str, Any] = {"empty": False, "normalized": clean, "models": {}}
    for slot in models:
        start = time.perf_counter()
        proba = slot.predictor.predict_proba([clean])[0]
        ms = 1000 * (time.perf_counter() - start)
        labels = slot.predictor.labels
        probs = {LABELS_SR.get(lab, lab): float(p) for lab, p in zip(labels, proba, strict=True)}
        result["models"][slot.key] = {
            "title": slot.title,
            "probs": probs,
            "label": max(probs, key=probs.get),
            "ms": ms,
        }
    by_key = {s.key: s.predictor for s in models}
    if "classical" in by_key:
        result["classical_highlights"] = classical_highlights(by_key["classical"], clean)
    if "bertic" in by_key:
        result["attention"] = attention_highlights(by_key["bertic"], clean)
    return result


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

DESCRIPTION = """
# Analiza sentimenta recenzija na srpskom

Unesite recenziju (latinica ili ćirilica) i uporedite tri pristupa istreniranih na
[SerbMR](https://github.com/vukbatanovic/SerbMR) recenzijama filmova: klasični TF-IDF model,
neuronsku mrežu napisanu od nule (BiLSTM) i fino podešeni transformer
[BERTić](https://huggingface.co/aleksbrstina/bertic-serbmr-sentiment).
Kod, rezultati i analiza grešaka: [GitHub](https://github.com/aleksbrstina/serbian-sentiment).
"""

FOOTNOTE = """
**Kako čitati objašnjenja.** Kod TF-IDF modela boja reči pokazuje koliko je ona povukla
odluku ka *pozitivnom* (plavo) ili *negativnom* (crveno). Kod BERTić-a je prikazana pažnja
(attention) poslednjeg sloja sa `[CLS]` tokena — pokazuje gde model „gleda", ali nije
direktno objašnjenje odluke. Duge recenzije BERTić vidi skraćene: prvih i poslednjih 127
tokena (oznaka […]).
"""


def build_demo(models: list[ModelSlot]) -> Any:
    """Build the Gradio Blocks interface (without launching it)."""
    import gradio as gr

    titles = {s.key: s.title for s in models}

    def run(text: str) -> tuple[Any, ...]:
        res = analyze(text, models)
        if res.get("empty"):
            empty = [None] * len(models)
            return (*empty, "", [], [])
        labels_out = [res["models"][s.key]["probs"] for s in models]
        timing = " · ".join(f"{titles[s.key]}: {res['models'][s.key]['ms']:.0f} ms" for s in models)
        return (
            *labels_out,
            f"Vreme odgovora — {timing}",
            res.get("classical_highlights", []),
            res.get("attention", []),
        )

    with gr.Blocks(title="Sentiment recenzija na srpskom") as demo:
        gr.Markdown(DESCRIPTION)
        text = gr.Textbox(label="Recenzija", lines=6, placeholder="Unesite tekst recenzije…")
        button = gr.Button("Analiziraj", variant="primary")
        with gr.Row():
            label_outputs = [gr.Label(label=s.title, num_top_classes=2) for s in models]
        timing = gr.Markdown()
        with gr.Accordion("Zašto? Reči koje su najviše uticale na TF-IDF model", open=True):
            explain = gr.HighlightedText(
                label="Doprinos reči (plavo = ka pozitivnom, crveno = ka negativnom)",
                show_legend=False,
                combine_adjacent=False,
            )
        with gr.Accordion("BERTić: pažnja poslednjeg sloja", open=False):
            attention = gr.HighlightedText(
                label="Pažnja [CLS] tokena po rečima", show_legend=False, combine_adjacent=False
            )
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=text)
        gr.Markdown(FOOTNOTE)
        outputs = [*label_outputs, timing, explain, attention]
        button.click(run, inputs=text, outputs=outputs)
        text.submit(run, inputs=text, outputs=outputs)
    return demo


def main() -> None:
    """Load the models and launch the app."""
    models = load_models()
    demo = build_demo(models)
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))


if __name__ == "__main__":
    main()
