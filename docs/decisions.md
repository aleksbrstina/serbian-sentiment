# Design decisions

Short rationale for non-trivial choices. Newest phase at the bottom.

## Phase 1: Project setup

**uv + `pyproject.toml`, pinned to Python 3.11.** uv resolves and installs much faster than pip, produces a lockfile (`uv.lock`) for exact reproducibility, and can fetch the right Python interpreter itself, so `make install` works on a clean machine.

**Heavy dependencies as optional extras (`neural`, `transformer`, `app`).** torch, transformers and gradio add gigabytes. The data pipeline, classical models and unit tests don't need them, so CI and `make install` stay fast; `make install-all` pulls everything.

**`src/` layout.** Tests run against the installed package rather than whatever happens to be on the working directory path, which catches packaging mistakes early.

**`Predictor` as an abstract base class with `predict` derived from `predict_proba`.** Each model implements only `predict_proba`; deriving `predict` via argmax guarantees labels and probabilities never disagree, and evaluation and the demo depend on a single interface.

**pydantic schema for configs with `extra="forbid"`.** A typo in a YAML key fails loudly instead of being silently ignored, which matters for reproducibility (NFR-01). Model-specific fields live under `params` for now and get strict schemas when each model is implemented.

## Phase 2: Data and cleaning

**SerbMR-2C as the default, 3C behind a config flag.** 2C is perfectly balanced (841/841), so accuracy and macro F1 tell the same story and class weighting is unnecessary. 3C (FR-15) is a superset with an added neutral class and is selected with `variant: 3C`; the rest of the pipeline is label-agnostic.

**Normalization order: HTML → invisible chars → Cyrillic → diacritics → punctuation → whitespace.** HTML entities can decode into Cyrillic letters or non-breaking spaces, so they go first. Whitespace is collapsed last because earlier steps can introduce or remove spaces. The pipeline is idempotent (tested), so running it twice is safe.

**Custom Cyrillic→Latin map instead of `cyrtranslit`.** It is 30 letters and one dependency less. Handling the digraph letters Љ/Њ/Џ needs context: `Љубав` → `Ljubav` but `ЉУБАВ` → `LJUBAV`. The map is tested on the full alphabet in both cases.

**Diacritics are unified, not stripped.** NFC composes decomposed sequences (`c` + combining caron → `č`), and look-alikes (`ð` → `đ`) are replaced. `dj` is *not* turned into `đ` because it is ambiguous (`nadjačati`, `podjednako`). Stripping diacritics entirely loses information (`kuća`/`kuca`), so it is an opt-in experiment flag (`ascii_diacritics`) rather than the default. The models themselves can still lower-case.

**Typographic punctuation mapped to ASCII.** In SerbMR 52% of reviews contain „“ ” – … variants. Unifying them means `"` and `“` share one token in TF-IDF and in the BiLSTM vocabulary instead of splitting counts.

**Deduplication on a normalized key, before splitting.** Duplicates are detected after normalization and with a case/punctuation-insensitive key, so `Dobar film!` and `dobar film` count as one. Doing this before the split prevents the same review from landing in both train and test, which would inflate test scores. Copies of the same text with *different* labels are all dropped, because there is no way to know which label is right. On SerbMR-2C every rule removes 0 records. The rules stay in place for other datasets, and the report shows the zeros explicitly rather than hiding them.

**Stratified 80/10/10 with seed 42; test carved out first.** Stratifying keeps the class ratio identical across splits, which matters with only ~170 examples in val/test. Rescaling the second split (`val / (1 - test)`) makes both fractions refer to the full dataset.

**Raw file SHA-256 in the quality report.** It ties every processed split to an exact input file, so a silently changed download is detectable (NFR-01).

**`id` column kept.** The row number from the raw file travels with every record into the splits, so misclassified examples in error analysis (FR-11) can be traced back to the original.

## Phase 3: Exploratory data analysis

**Word-level EDA on the training split only.** χ² rankings, frequency lists and cue rates shape modelling decisions (features, truncation, error categories). Computing them on val/test would leak held-out information into those decisions. The validation split is used only for the OOV rate, because that is exactly the situation a trained model faces.

**χ² on binary word presence, not counts.** With reviews of 21–1835 words, raw counts let a single long review dominate a word's score. Presence answers a cleaner question: "does a review containing this word tend to be positive?" Each word is assigned to the class where its document share is highest.

**Mann-Whitney U instead of a t-test for length.** The length distribution is strongly right-skewed (mean > median, long tail up to ~1800 words). A rank test makes no normality assumption.

**Insights are generated from code with conditional wording.** E.g. "length carries no signal" is only emitted if p ≥ 0.05; otherwise the text says the opposite. If the data or split changes, the report cannot silently go stale (spec: numbers are never typed by hand).

**Findings that change later phases:**
- *Truncation.* ~90% of reviews exceed 256 words, and subword tokens outnumber words. With the spec's `max_length=256`, BERTić sees only the opening of almost every review. In Phase 6, compare head-only with head+tail truncation (e.g. first 128 + last 128 tokens). Reviews often end with a verdict, as the samples in the notebook show.
- *Morphology.* 27% of validation word types are unseen in training. This supports character n-grams in TF-IDF (Phase 4) and fastText initialization (fastText builds vectors from subwords) for the BiLSTM (Phase 5), and it argues for a frequency cut-off on the BiLSTM vocabulary, since 55% of training types occur only once.
- *Negation.* Negation words appear in ~98% of reviews of both classes, so a bag of words cannot use "nije" as a sentiment cue. It is expected to be a top error category (FR-11).
- *Mixed sentiment.* Strong negative cues appear in positive reviews (*nažalost* occurs in 39 positive training reviews). Long reviews weigh pros and cons, so mixed sentiment is a second expected error category.
- *No leakage via explicit scores.* "x/10" appears in <1% of reviews, so it is not worth masking.

**Notebook generated from the same functions as the report, and executed by `make notebooks`.** The notebook is a view on `sentiment.eda`, not a second implementation, so its numbers are always identical to `reports/eda/eda_report.md`. It is committed with outputs so the plots are visible on GitHub without running anything.

## Phase 4: Classical models

**Features: word TF-IDF + character n-grams (`char_wb`) in one `FeatureUnion`.** Character n-grams inside word boundaries share evidence across inflected forms (`odličan`/`odlična`/`odlično` all contain `odlič`), which the EDA showed is critical: 27% of validation word types are unseen in training. The search picked char n-grams for **all three** classifiers.

**Search space and what it found.** Word n-grams {1, 1–2}, char n-grams {none, 2–5, 3–5}, `min_df` {1, 2, 3}, sublinear TF {on, off}, plus a log-uniform `C` / `alpha`. Every best configuration used **word unigrams only**, char 3–5-grams, `min_df=3` and sublinear TF. Bigrams did not help: with ~1,300 training reviews most bigrams are too rare to estimate, and char n-grams already capture local context. Sublinear TF (1 + log tf) won everywhere, which makes sense for long reviews where a word repeated 10 times is not 10× more informative.

**Optuna (TPE) on the fixed validation split, as the spec requires (FR-06).** TPE is seeded, so the search is reproducible (tested). Vectorizer outputs are cached per setting, with word and char blocks cached separately and stacked, so 90 trials take ~3 minutes instead of ~15.

**Key lesson: the validation set is too small to rank close configurations.** With 169 validation reviews, one review = 0.6 F1 points. In the SVM search, 10 of 30 trials score within one review of the best, and `C` from 0.5 to 95 gives the same F1. The first run's best `C=9.4` landed at the old upper bound (10). After widening the range to 100 the search found the same top score with `C=0.52`, which confirms a flat plateau, not an optimum beyond the edge. The report now flags any regularization parameter at the edge of its range automatically. Consequences:
- The choice of "best" classical model (LR 0.870 vs. SVM 0.864 vs. NB 0.852) is within noise for LR vs. SVM. Test-set comparison in Phase 7 should report bootstrap confidence intervals, not just point estimates.
- More trials would mostly overfit the validation set (the search would pick lucky configurations), so 30 per classifier is deliberate. A more robust alternative is k-fold CV on train+val for selection. It is not used here because the spec fixes the validation-split protocol, and the same protocol is needed for the BiLSTM and BERTić for a fair comparison.

**Final models are refitted on the training split only, not train+val.** The saved model is then exactly the one whose validation score is reported. Refitting on train+val would gain ~12% more data but leave no honest validation number for it.

**Linear SVM probabilities via Platt scaling (`CalibratedClassifierCV`, sigmoid, 5 folds).** `LinearSVC` has no `predict_proba`, but the `Predictor` interface and the demo need probabilities. Calibration fits five SVMs on 4/5 of the data and averages, so its validation F1 differs slightly from the uncalibrated search score; this is visible in the report. The search itself runs without calibration (5× faster, and it doesn't change which settings separate the classes best).

**Complement NB instead of Multinomial NB.** Complement NB estimates each class's parameters from the *other* classes' data, which is more stable on text and works well with TF-IDF inputs. It is the fastest and smallest model, but the weakest (0.852): its feature-independence assumption is badly violated by overlapping char n-grams.

**Training time = fresh fit of the full pipeline.** The vectorizer, classifier and calibration are all inside the timed fit, so the number is comparable with the neural models (FR-10). Inference is reported two ways: per example in a batch (ms/example) and single-text latency, which is what the demo experiences (NFR-06: < 2 s; the classical models need ~5–10 ms).

**Interpretability (FR-14 groundwork).** Per-class weights come from `coef_` (binary models expanded to `[-w, w]`), averaged over calibration folds for the SVM, and from centered log-probabilities for NB. `explain()` multiplies a text's TF-IDF values by the weight difference to show which words drove the decision. Example: *"Film nije loš."* ("The film is not bad.") is classified **negative** (p = 0.95) because `loš` dominates and `nije` adds negative weight; unigram features cannot represent negation scope. This is exactly the error type the EDA predicted.

**Influential-word lists contain function words** (`ni`, `bi`, `bilo`, `ja`, `vas`). These are real signals, not noise. Conditional `bi`/`bilo` ("would have been") expresses unmet expectations, and `vas` ("you") reflects reviews addressing the reader with recommendations. No stopword removal is applied for modelling for this reason; the stopword list is only used for raw frequency counts in the EDA.
