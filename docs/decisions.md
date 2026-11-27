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

## Phase 5: BiLSTM

All numbers below come from `reports/bilstm/training.md` and `reports/bilstm/experiments.md` (generated from MLflow). Final model: **0.763 validation macro F1**, about 10 points below TF-IDF (0.870).

**Why the BiLSTM loses to TF-IDF here, and why that's expected.** 1,344 training reviews is tiny for a sequence model. The BiLSTM has to learn composition from scratch, while TF-IDF + a linear model needs only one weight per feature. The learning curves show it: train F1 reaches 0.98 while validation peaks at ~0.79. The network memorizes rather than generalizes. With pretrained contextual representations (BERTić, Phase 6) this gap should close. The BiLSTM is the "neural network from scratch" baseline the spec asks for, not a model expected to win.

**Tokenizer: lower-cased regex, words + numbers + `!`/`?`.** It matches how fastText vectors are keyed (lower-cased surface forms), so coverage stays high. `!` and `?` are kept because the EDA showed question marks are 1.5× more frequent in negative reviews.

**Vocabulary: `min_freq=2` (34,179 types).** 55% of training types are hapaxes. They would get embedding rows with one training example each, and almost none appear in validation. Rare words map to `<unk>`, and since fastText covers them anyway, a cut-off loses little: 88.1% of validation tokens are in the vocabulary.

**fastText transliteration: Cyrillic entries mapped to Latin.** 74% of the 200k most frequent `cc.sr.300` words are Cyrillic (Serbian Wikipedia and web text are largely Cyrillic), while our data is Latin. Lower-casing and transliterating the fastText vocabulary before lookup raises type coverage from **78.0% to 95.2%**. Token coverage moves less (85.7% → 87.8%), because frequent words exist in both scripts; the gain is in rarer, content-bearing words. When several entries map to one key, the most frequent wins (the file is frequency-sorted).

**Frozen embeddings.** Unfreezing let the model fit the training set within 3 epochs (train loss 0.05) while validation loss rose to 1.07: 10.2M embedding parameters vs. 1,344 reviews. Frozen embeddings leave 188k trainable parameters (LSTM + output). This also means the 41.8 MB model file is 98% fixed embedding rows.

**Embeddings rescaled to unit standard deviation.** Raw fastText values have std ≈ 0.047, so the LSTM's gates saw near-zero inputs and learned very slowly with frozen embeddings (train loss barely moved in 4 epochs). Dividing by a single global constant keeps all relative geometry (cosine similarities unchanged) and puts inputs on the scale LSTM initialization assumes.

**Pooling: max + mean over time instead of the last hidden state.** For ~250-token inputs, the final state is dominated by the last sentences. Max pooling lets a strong cue anywhere reach the classifier, and mean pooling adds an overall tone. Max-only reached the same F1 (0.763) but needed 18 epochs instead of 5.

**Truncation: head + tail, max 256 tokens.** This is the clearest experimental result in the project: keeping the first 128 + last 128 tokens gives **0.763 vs. 0.685** for keeping only the first 256 (same settings otherwise). It confirms the EDA hypothesis that reviews end with the verdict. The same choice will be tested for BERTić. 256 instead of 512 tokens: 2.6× faster per epoch (15 vs. 41 s), needed to meet the 15-minute CPU budget (NFR-02).

**Optimization: AdamW, lr 2e-3, batch 16, gradient clipping at 1.0, ReduceLROnPlateau (×0.5 after 1 flat epoch).** Batch 32 gave only 42 updates per epoch and learned too slowly. Batch 16 doubles the updates at almost the same epoch time. Clipping prevents the exploding gradients LSTMs are prone to on long sequences. The scheduler damped the epoch-to-epoch swings seen without it (val F1 jumping 0.44 ↔ 0.76).

**Early stopping on validation loss (patience 5), checkpoint = best epoch.** Validation loss is smoother than validation F1 on 169 examples (one review = 0.6 F1 points), and it also penalizes over-confident wrong predictions, which matters for the probabilities shown in the demo. The best epoch's weights are kept in memory and restored after training, and only those are saved.

**Small, logged experiment series instead of a large search.** Four configurations, each a full run logged to MLflow (`exp-*` runs, `scripts/train.py --set ...`). With a validation set this small, a large search would mostly fit noise (see Phase 4). Each experiment tested one hypothesis: learning rate/dropout, pooling, truncation. Total cost ~15 minutes.

**Reproducibility verified.** The official run reproduces the experiment run with identical loss and F1 in every epoch: seeded Python/NumPy/PyTorch RNGs, a seeded `DataLoader` generator, and a deterministic vocabulary order (ties broken alphabetically).

**Plain PyTorch training loop.** `run_epoch` is the whole loop: forward, loss, `zero_grad`, `backward`, clip, `step`, with `torch.set_grad_enabled` switching between training and evaluation. `pack_padded_sequence` makes the LSTM skip padding, and a test checks that extra padding does not change the output.

## Phase 6: Transformer (BERTić)

**Model verified on the Hub: `classla/bcms-bertic`.** It is an ELECTRA-base discriminator (110M parameters, 12 layers, 768 hidden, 32k WordPiece vocabulary, cased, Apache 2.0) pretrained on 8 billion tokens of Bosnian, Croatian, Montenegrin and Serbian. Loading it for classification drops the pretraining head (`discriminator_predictions.*`) and initializes a new `classifier.*` head; the load report shows exactly that, as expected. Fallback: `FacebookAI/xlm-roberta-base` (multilingual, `--set params.model_name=...`).

**Library versions checked before implementation (transformers 5.24, datasets 5.2, accelerate 1.16).** Four 5.x API changes would have broken code written from older tutorials, each found by a smoke test and fixed:
1. `warmup_ratio` is gone; `warmup_steps` accepts a float in [0, 1) as a ratio.
2. `tokenizer.build_inputs_with_special_tokens` is gone. `special_token_frame()` derives the special tokens by tokenizing a probe word with and without them, so it works for any architecture.
3. Tokenizers take `vocab=` (dict); `vocab_file=` is silently ignored and yields an all-`[UNK]` tokenizer (caught by a unit test).
4. The default SDPA attention kernel returns no attention weights; attention visualization switches to the eager implementation for that call only.

**Truncation is the central problem: 97.8% of reviews exceed 256 subword tokens.** Median review: 669 subwords (1.51 per word; Serbian morphology splits words, e.g. `traljavo` → `tra ##lja ##vo`). 74% exceed even BERT's hard limit of 512. With the spec's `max_length=256`, the model sees about 38% of a median review. Default: **head+tail** (first 127 + last 127 subwords), because the BiLSTM experiment showed head+tail beats head-only by 7.8 F1 points and reviews end with a verdict. Hugging Face tokenizers only truncate one side, so the full text is tokenized and truncated manually. The Colab notebook runs a head-only comparison.

**Truncation settings stored in the model's `config.json`** (`sentiment_max_length`, `sentiment_truncation`). A model downloaded from the Hub then carries its own preprocessing; `TransformerPredictor` reads it, so the demo and evaluation can't silently use different truncation than training did.

**Hyperparameters: lr 2e-5, batch 16, 5 epochs, linear warmup over 10% then linear decay, weight decay 0.01.** These are the standard BERT fine-tuning settings (Devlin et al. recommend lr {2, 3, 5}e-5, batch {16, 32}, 2–4 epochs). Warmup keeps the large early gradients of the randomly initialized head from wrecking pretrained weights. With only 84 steps per epoch, 5 epochs gives the schedule room, and early stopping (patience 2) cuts it short if validation loss rises.

**Model selection on validation loss, not F1.** Same reasoning as the BiLSTM: on 169 validation reviews F1 moves in 0.6-point jumps, while loss is continuous and also penalizes over-confident errors, which matters for the probabilities in the demo. The Trainer reloads the best checkpoint at the end (`load_best_model_at_end`); intermediate checkpoints are deleted after saving the final model.

**Hugging Face Trainer, as the spec requires, with metrics from the shared module.** `compute_metrics` reuses `sentiment.evaluation.metrics`, so the macro F1 the Trainer optimizes is computed identically to the other models. Mixed precision (`fp16`) is enabled automatically on GPU. Dynamic padding (`DataCollatorWithPadding`) pads each batch only to its longest sequence.

**Colab reproducibility: the notebook verifies the data, it doesn't trust it.** Processed data is not in git, so Colab rebuilds it from the raw file. The notebook asserts that the raw file's SHA-256 *and* a split fingerprint (hash of which review ids are in train/val/test) match the committed `reports/data_quality.md`. A different scikit-learn version could reshuffle the split, and then BERTić would be compared against the other models on a different test set without anyone noticing.

**`requires-python` relaxed to `>=3.11`.** Colab runs a newer Python than 3.11, and `pip install -e .` would refuse to install with the old `<3.12` cap. Local development stays pinned to 3.11 via `.python-version` and `uv sync --python 3.11`.

**Publishing: `HfApi.upload_folder` with a generated model card.** One upload contains the weights (safetensors), tokenizer, `metadata.json` and a README with metrics, preprocessing, usage, and the CC BY-NC-SA data license. Authentication uses `huggingface_hub.login()` in the notebook, so no token ever touches the repository or config files.

**Colab runs are imported into local MLflow** (`scripts/import_run.py`), so all three models end up in one MLflow experiment even though the transformer trains on another machine.

**Offline tests with a tiny random ELECTRA.** A 2-layer, 16-dimensional ELECTRA with a 17-word vocabulary is built in the test fixture, so the full `train_transformer` pipeline (Trainer, checkpoint selection, saving, MLflow, reloading) runs in seconds without network access, in CI too.

### Phase 6 results (CPU verification run)

Numbers from `reports/bertic/training.md`. The run used the official config on this project's 4-core CPU as an end-to-end check before Colab: **55 minutes** (≈17 min per epoch; a T4 GPU is expected to be roughly 10× faster).

- **Validation macro F1 0.875 at the selected epoch 2**, ahead of the BiLSTM (0.763) and level with TF-IDF + LR (0.870; one review apart). Inference: ~100 ms per review on CPU, well under the 2 s demo budget (NFR-06).
- **Model selection on loss picked epoch 2, while epoch 3 had higher F1 (0.899 vs 0.875).** Validation loss at epochs 2 and 3 is almost identical (0.316 vs 0.323), so loss slightly preferred the less confident model, while F1 counted four more correct reviews at epoch 3. The criterion is **not** switched after seeing this. Choosing the selection metric after looking at the results is a form of overfitting to the validation set, and the 0.024 F1 gap is about four reviews out of 169, within noise. The test set in Phase 7, with confidence intervals, is the right place to compare the models.
- **Overfitting starts after epoch 2:** training loss keeps falling (0.36 → 0.11) while validation loss rises (0.32 → 0.39). Early stopping (patience 2) ended training after epoch 4 of 5, and the epoch-2 checkpoint was restored.
- **97.8% of training reviews were truncated** at 256 tokens. That BERTić matches TF-IDF while seeing roughly a third of each review suggests the transformer extracts more per token; a longer context (512) is the most obvious next experiment.
- **Published to the Hub as [`aleksbrstina/bertic-serbmr-sentiment`](https://huggingface.co/aleksbrstina/bertic-serbmr-sentiment)** (public, safetensors, generated model card). Downloading it into an empty cache and loading it through `TransformerPredictor` gives predictions identical to the local model and the same validation macro F1, which confirms the published artifact carries its preprocessing settings correctly.
- **First sign that context helps with negation:** *"Film nije loš."* ("The film is not bad.") is classified positive by BERTić (p = 0.79), while TF-IDF + LR (0.82 negative) and the BiLSTM (0.94 negative) get it wrong. One example proves nothing, but negation is a planned error category, and Phase 7 will measure it over the test set.

## Phase 7: Evaluation, error analysis and demo

Numbers below are from `reports/comparison.md` and `reports/evaluation/*.csv`, generated by `scripts/evaluate.py`.

**One evaluation path for every model.** `scripts/evaluate.py` loads all five models with `load_predictor`, scores them on the same 169 test reviews with the shared metrics module, and times them on the same CPU (one warm-up call, then batch time and the median of 50 single-review calls). The test set is used here for the first time: no threshold, hyperparameter or tag cut-off was ever chosen on it.

**Bootstrap intervals and paired tests instead of point estimates.** With 169 reviews, a 95% interval for macro F1 is about ±0.05 wide, wider than most differences between the classical models. Each of the 2,000 bootstrap resamples scores *all* models on the same reviews, so differences between models are estimated with their correlation intact (paired). McNemar's exact test complements it by looking only at reviews where exactly one of two models is right. BERTić vs. TF-IDF + LR: +0.083 macro F1, interval [+0.015, +0.149], McNemar p = 0.024. LR vs. SVM: no difference at all.

**Validation optimism confirmed.** The classical models scored 0.870 on validation and 0.834 on test; BERTić went the other way (0.875 → 0.917). Phase 4 predicted this: with 30 Optuna trials on 169 validation reviews, the selected configuration is partly chosen for luck. BERTić had no hyperparameter search, so its validation score carried no selection bias.

**Heuristic tags with training-split thresholds.** Each tag is a measurable property (negation density, contrastive markers, question/exclamation marks, length, and *contradicting cues*: more training-χ² cue words of the opposite class than of the review's own). "Dense" means the top quartile of the training split. Comparing the error rate on tagged vs. untagged reviews shows *where* a model fails, not just how often.

**Findings from the tags, including one that contradicted the EDA:**
- *Contradicting cues* are the clearest weakness of bag-of-words: 42% error for every TF-IDF model vs. 17% overall; BERTić 14%.
- *Long reviews* hurt the BiLSTM most (42% error), consistent with it seeing 256 of ~470 tokens.
- *Negation-dense reviews were not harder for TF-IDF* (6% error vs. 17% overall). The EDA expected negation to be a major error source. In long reviews, many negations cancel out in a bag of words, and char n-grams partly capture adjacent negation. Negation does show up in the manual review, but as a contributing factor in hard reviews rather than a cause of errors by itself. The report keeps this tag even though it disproves the hypothesis.

**Manual review: 40 misclassified reviews, hardest first.** The candidates are ranked by how many models got them wrong, then by the strongest model's confidence. Each was read (opening and closing ~350 characters, where reviews state their verdict) and given one or more categories from a fixed list; one category, `rare_vocabulary`, was added because a clear pattern didn't fit the spec's list. Result: **mixed sentiment** dominates (21 of 40). TF-IDF + LR fails on 15 of those, BERTić on 5. On the 9 ironic reviews, TF-IDF + LR fails on 7, BERTić on 1. The review is marked as a first pass by Claude Code in the CSV, so the author should verify it before citing it. `short_text` stayed empty: SerbMR's shortest review has 21 words.

**BERTić's errors are one-sided.** 12 of its 14 test errors are positive reviews predicted negative (negative recall 98%, positive recall 86%). A plausible cause: selection on validation loss picked epoch 2, a less confident checkpoint, and critics' positive reviews often contain explicit criticism, which a cautious model reads as negative. Because of the selection-bias argument above, this is not "fixed" by picking another epoch; with a larger validation set, checking calibration or a decision threshold would be the next step.

**Demo design (FR-13/14/16).**
- The demo applies the training normalization (`normalize_text`) before prediction, so Cyrillic input works and the models see text in the form they were trained on.
- *TF-IDF explanation*: each word's contribution (TF-IDF value × weight difference between the classes) is drawn on the text. Gradio draws positive scores red, so the sign is flipped to make red mean "towards negative" and blue "towards positive"; a screenshot check caught the original mismatch with the legend.
- *BERTić attention*: last-layer `[CLS]` attention, averaged over heads and merged from subwords into words, with a `[…]` marker where head+tail truncation removed the middle. The UI and the notebook both state the caveat that attention is a diagnostic, not an explanation.
- *Latency*: all three models together answer in 0.1–0.4 s on CPU, well within NFR-06 (< 2 s).

**Deployment: models on the Hub, Space built by a script.** The TF-IDF and BiLSTM weights go to a separate model repository (`serbian-sentiment-baselines`, CC BY-NC-SA because of SerbMR and the fastText-derived embedding matrix); BERTić is already on the Hub. `scripts/deploy_space.py` assembles the Space from the repository (app, `sentiment` package, Space card) and pins the runtime libraries to the exact versions the models were trained with. This matters most for scikit-learn: the pipeline is a pickle and is only guaranteed to load with the same version. The app resolves each model from an environment variable, then the local `artifacts/`, then the Hub, so the same file runs locally, in tests and on Spaces.

**README numbers are generated.** The results table sits between `<!-- RESULTS:START/END -->` markers and is rewritten by `scripts/evaluate.py`; the prose findings link to the generated reports instead of repeating numbers that would go stale.

**Public demo as a static, in-browser Space.** Hugging Face no longer hosts Gradio Spaces on free CPU without a PRO subscription; static Spaces stay free. Options considered:
- *Gradio-Lite (Python in the browser via Pyodide)*: rejected. Pyodide ships its own scikit-learn version, and unpickling a pipeline across versions is unsafe; it would also make visitors download ~30 MB before the first prediction.
- *Export the model as data and reimplement inference in JavaScript*: chosen. `sentiment.export_static` writes the vocabularies, IDF weights, coefficients and the normalization tables (taken from the Python constants, so there is one source of truth) to a 3.6 MB JSON file (1.0 MB gzipped). `app/static/sentiment.js` reimplements scikit-learn's `word` and `char_wb` analyzers, sublinear TF-IDF with per-block L2 normalization, and the logistic function.
- *Parity is tested, not assumed*: `tests/test_static_demo.py` runs the JavaScript in Node on tricky inputs (Cyrillic all-caps digraphs, HTML entities, combining diacritics, BOM and exotic whitespace, word bigrams) and compares normalization, probabilities and word contributions with Python. On the real model the largest probability difference over all 169 test reviews is below 1e-6. Two details that needed care: whitespace must follow Python's `str.isspace()` (JavaScript's `\s` includes the BOM, Python's doesn't), and ties at p = 0.5 must go to the first class, like numpy's `argmax`.
- *Trade-off*: the public demo shows only the TF-IDF model; BERTić (443 MB) can't run in a static page. The page says so, links the published BERTić model, and shows the generated comparison table so visitors see what they are missing. The three-model Gradio app is still in the repository (`make app`, or `deploy_space.py --mode gradio` with a paid plan).
