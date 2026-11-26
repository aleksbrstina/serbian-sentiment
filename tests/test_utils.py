import random
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from sentiment.utils import RunConfig, get_logger, load_config, save_config, set_seed

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_set_seed_is_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand(3).tolist())
    set_seed(123)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second


def test_get_logger_does_not_duplicate_handlers():
    a = get_logger("sentiment.test")
    b = get_logger("sentiment.test")
    assert a is b
    assert len(a.handlers) == 1


@pytest.mark.parametrize("name", ["classical", "bilstm", "bertic"])
def test_shipped_configs_are_valid(name):
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert config.model == name
    assert config.seed == 42


def test_config_roundtrip(tmp_path):
    config = RunConfig(model="classical", seed=7, params={"C": [0.1, 1.0]})
    path = save_config(config, tmp_path / "run" / "config.yaml")
    assert load_config(path) == config


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model: classical\nunknown_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_config_rejects_negative_seed():
    with pytest.raises(ValidationError):
        RunConfig(model="classical", seed=-1)
