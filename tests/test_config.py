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


def test_source_env_defaults_to_ml_env(tmp_path, monkeypatch):
    """Without SOURCE_ENV, {source_env} resolves to the same value as ML_ENV."""
    from src.config import load_config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('db: "{env}_ML"\nsrc: "{source_env}_DATALAKE.S.V"\n', encoding="utf-8")
    monkeypatch.setenv("ML_ENV", "DEV")
    monkeypatch.setenv("SOURCE_ENV", "")
    cfg = load_config(cfg_file)
    assert cfg["db"] == "DEV_ML"
    assert cfg["src"] == "DEV_DATALAKE.S.V"
    assert cfg["source_env"] == "DEV"


def test_source_env_separate_from_target(tmp_path, monkeypatch):
    """SOURCE_ENV=PROD reads from PROD while the target stays DEV_ML."""
    from src.config import get_source_env, load_config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('db: "{env}_ML"\nsrc: "{source_env}_DATALAKE.S.V"\n', encoding="utf-8")
    monkeypatch.setenv("ML_ENV", "DEV")
    monkeypatch.setenv("SOURCE_ENV", "prod")
    cfg = load_config(cfg_file)
    assert cfg["db"] == "DEV_ML"
    assert cfg["src"] == "PROD_DATALAKE.S.V"
    assert load_config(cfg_file, source_env="INT")["src"] == "INT_DATALAKE.S.V"

    monkeypatch.setenv("SOURCE_ENV", "TEST")
    with pytest.raises(ValueError):
        get_source_env()


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
