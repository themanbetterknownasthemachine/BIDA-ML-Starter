"""Tests for the config module."""

import pytest


def test_load_config_resolves_env(monkeypatch):
    """{env} placeholders resolve to the environment from ML_ENV."""
    from src.config import load_config

    monkeypatch.setenv("ML_ENV", "DEV")
    cfg = load_config()
    assert cfg["env"] == "DEV"
    assert cfg["snowflake"]["database"] == "DEV_ML"
    assert "{env}" not in str(cfg)
    assert cfg["project"]["random_state"] == 42


def test_env_override_and_validation(monkeypatch):
    """Explicit env wins over ML_ENV; unknown environments are rejected."""
    from src.config import get_env, load_config

    monkeypatch.setenv("ML_ENV", "DEV")
    assert load_config(env="PROD")["snowflake"]["database"] == "PROD_ML"

    monkeypatch.setenv("ML_ENV", "TEST")
    with pytest.raises(ValueError):
        get_env()


def test_random_state():
    """RANDOM_STATE should be 42."""
    from src.config import RANDOM_STATE

    assert RANDOM_STATE == 42


def test_snowflake_config(monkeypatch):
    """Snowflake section carries role, warehouse and the three target schemas."""
    from src.config import get_snowflake_config

    monkeypatch.setenv("ML_ENV", "DEV")
    sf = get_snowflake_config()
    assert sf["role"] == "ML_DEVELOPER"
    assert sf["warehouse"] == "CONSUMER"
    assert set(sf["schemas"]) == {"inference", "registry", "monitoring"}
