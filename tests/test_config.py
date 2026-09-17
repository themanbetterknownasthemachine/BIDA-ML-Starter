"""Tests for the config module."""


def test_load_config():
    """Config should load and contain the main sections."""
    from src.config import load_config

    cfg = load_config()
    assert "project" in cfg
    assert "snowflake" in cfg
    assert "tables" in cfg
    assert cfg["project"]["random_state"] == 42


def test_random_state():
    """RANDOM_STATE should be 42."""
    from src.config import RANDOM_STATE

    assert RANDOM_STATE == 42


def test_snowflake_config():
    """Snowflake config should point to the ML database with the three target schemas."""
    from src.config import get_snowflake_config

    sf = get_snowflake_config()
    assert sf["database"] in ("DEV_ML", "PROD_ML")
    assert sf["role"] == "ML_DEVELOPER"
    assert sf["warehouse"] == "CONSUMER"
    assert set(sf["schemas"]) == {"inference", "registry", "monitoring"}
