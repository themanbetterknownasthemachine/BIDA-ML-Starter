"""Tests for data_loader module."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


def test_load_config():
    """Config should load without errors."""
    from src.config import load_config
    cfg = load_config()
    assert "project" in cfg
    assert "snowflake" in cfg
    assert "timeseries" in cfg
    assert cfg["project"]["random_state"] == 42


def test_get_model_config():
    """Should return config for known models."""
    from src.config import get_model_config
    lgb_cfg = get_model_config("lightgbm")
    assert "n_estimators" in lgb_cfg
    assert "learning_rate" in lgb_cfg


def test_get_model_config_unknown():
    """Should raise KeyError for unknown model."""
    from src.config import get_model_config
    with pytest.raises(KeyError):
        get_model_config("nonexistent_model")
