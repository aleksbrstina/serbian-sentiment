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
