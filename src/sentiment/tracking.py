"""Thin helpers around local MLflow tracking (NFR-04)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Silence MLflow's start-up hint about bundled assistant skills; it is noise in CLI logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from sentiment.utils import RunConfig  # noqa: E402


def setup_mlflow(config: RunConfig) -> str:
    """Point MLflow at the configured local store and return the experiment id.

    The experiment is created on first use with its artifacts under
    ``config.mlflow_artifact_root`` so everything stays inside the repository folder.

    Args:
        config: Run configuration with ``tracking_uri`` and ``experiment_name``.

    Returns:
        MLflow experiment id.
    """
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        artifact_root = Path(config.mlflow_artifact_root).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            config.experiment_name, artifact_location=artifact_root.as_uri()
        )
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts into ``{"a.b": "value"}`` for ``mlflow.log_params``."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)
    return flat
