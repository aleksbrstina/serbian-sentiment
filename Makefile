.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run

.PHONY: help install install-all lint format test test-cov pre-commit prepare-data eda train-classical train-bilstm train-bertic notebooks mlflow-ui clean

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

eda: ## Generate the EDA report and figures in reports/eda
	$(RUN) python scripts/run_eda.py --data-dir data/processed --out-dir reports/eda

train-classical: ## Tune, train and log the TF-IDF models (configs/classical.yaml)
	$(RUN) python scripts/train.py --config configs/classical.yaml

train-bilstm: ## Train the BiLSTM (configs/bilstm.yaml; needs `make install-all` and fastText vectors)
	$(RUN) python scripts/train.py --config configs/bilstm.yaml

train-bertic: ## Fine-tune BERTić (configs/bertic.yaml); use a GPU, see notebooks/colab_train_bertic.ipynb
	$(RUN) python scripts/train.py --config configs/bertic.yaml

notebooks: ## Execute notebooks in place (requires processed data)
	$(RUN) python scripts/execute_notebooks.py notebooks/01_eda.ipynb

mlflow-ui: ## Launch the local MLflow UI on http://127.0.0.1:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db

clean: ## Remove caches and build outputs (keeps data, mlruns and artifacts)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
