"""Tests for config module."""

import pytest


def test_load_config():
    """Config should load without errors."""
    from src.config import load_config
    cfg = load_config()
    assert "project" in cfg
    assert "snowflake" in cfg
    assert cfg["project"]["random_state"] == 42


def test_random_state():
    """RANDOM_STATE should be 42."""
    from src.config import RANDOM_STATE
    assert RANDOM_STATE == 42


def test_snowflake_config():
    """Snowflake config should have required fields."""
    from src.config import get_snowflake_config
    sf = get_snowflake_config()
    assert "database" in sf
    assert "warehouse" in sf
