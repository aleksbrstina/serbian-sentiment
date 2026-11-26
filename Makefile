.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run

.PHONY: help install install-all lint format test test-cov pre-commit prepare-data mlflow-ui clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv (Python 3.11) with core + dev dependencies
	$(UV) sync --python 3.11

install-all: ## Also install optional extras (torch, transformers, gradio)
	$(UV) sync --python 3.11 --all-extras

lint: ## Check code style with ruff
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Auto-format and fix lint issues with ruff
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test: ## Run unit tests
	$(RUN) pytest

test-cov: ## Run unit tests with a coverage report
	$(RUN) pytest --cov=sentiment --cov-report=term-missing

pre-commit: ## Install git pre-commit hooks
	$(RUN) pre-commit install

prepare-data: ## Normalize, clean and split raw data (configs/data.yaml)
	$(RUN) python scripts/prepare_data.py --config configs/data.yaml

mlflow-ui: ## Launch the local MLflow UI on http://127.0.0.1:5000
	$(RUN) mlflow ui --backend-store-uri ./mlruns

clean: ## Remove caches and build outputs (keeps data, mlruns and artifacts)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
