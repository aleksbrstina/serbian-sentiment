"""Shared utilities: seeding, logging and configuration loading."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def set_seed(seed: int) -> None:
    """Seed every random number generator used in the project.

    Covers Python's ``random``, NumPy and, when installed, PyTorch (CPU and CUDA).

    Args:
        seed: Non-negative integer seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes formatted records to stderr.

    Calling this repeatedly with the same name does not add duplicate handlers.

    Args:
        name: Logger name, usually ``__name__``.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


class DataPaths(BaseModel):
    """Locations of the processed data splits."""

    model_config = ConfigDict(extra="forbid")

    train: Path = Path("data/processed/train.parquet")
    val: Path = Path("data/processed/val.parquet")
    test: Path = Path("data/processed/test.parquet")


class RunConfig(BaseModel):
    """Common configuration shared by every training run.

    Model-specific settings live in ``params`` for now; each model module will
    define a stricter schema when it is implemented.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model identifier, e.g. 'classical', 'bilstm', 'bertic'.")
    seed: int = Field(default=42, ge=0)
    experiment_name: str = "serbian-sentiment"
    output_dir: Path = Path("artifacts")
    tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_root: Path = Path("mlruns")
    data: DataPaths = Field(default_factory=DataPaths)
    params: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML run configuration.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the content does not match the schema.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RunConfig.model_validate(raw)


def save_config(config: RunConfig, path: str | Path) -> Path:
    """Write a configuration to YAML so it can be stored next to run results.

    Args:
        config: Configuration to save.
        path: Destination file; parent directories are created.

    Returns:
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
    return path


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Measure wall-clock time of a block.

    Example:
        >>> with timer() as t:
        ...     do_work()
        >>> t["seconds"]

    Yields:
        Dict whose ``"seconds"`` key is filled in when the block exits.
    """
    result: dict[str, float] = {"seconds": float("nan")}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def path_size_mb(path: str | Path) -> float:
    """Return the size of a file, or the total size of a directory, in megabytes."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
