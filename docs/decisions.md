# Design decisions

Short rationale for non-trivial choices. Newest phase at the bottom.

## Phase 1: Project setup

**uv + `pyproject.toml`, pinned to Python 3.11.** uv resolves and installs much faster than pip, produces a lockfile (`uv.lock`) for exact reproducibility, and can fetch the right Python interpreter itself, so `make install` works on a clean machine.

**Heavy dependencies as optional extras (`neural`, `transformer`, `app`).** torch, transformers and gradio add gigabytes. The data pipeline, classical models and unit tests don't need them, so CI and `make install` stay fast; `make install-all` pulls everything.

**`src/` layout.** Tests run against the installed package rather than whatever happens to be on the working directory path, which catches packaging mistakes early.

**`Predictor` as an abstract base class with `predict` derived from `predict_proba`.** Each model implements only `predict_proba`; deriving `predict` via argmax guarantees labels and probabilities never disagree, and evaluation and the demo depend on a single interface.

**pydantic schema for configs with `extra="forbid"`.** A typo in a YAML key fails loudly instead of being silently ignored, which matters for reproducibility (NFR-01). Model-specific fields live under `params` for now and get strict schemas when each model is implemented.
