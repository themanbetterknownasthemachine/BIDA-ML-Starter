"""Configuration loader for BIDA ML Starter."""

import os
from pathlib import Path
from typing import Any

import yaml


def _find_config_path() -> Path:
    """Walk up from CWD or this file to find configs/config.yaml."""
    anchors = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for anchor in anchors:
        for parent in [anchor] + list(anchor.parents):
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                return candidate
    raise FileNotFoundError("configs/config.yaml not found")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and return as dict.

    Args:
        path: Optional explicit path. Auto-detected if None.

    Returns:
        Parsed config dictionary.
    """
    config_path = Path(path) if path else _find_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_snowflake_config(cfg: dict | None = None) -> dict[str, str]:
    """Extract Snowflake connection params from config."""
    cfg = cfg or load_config()
    return cfg.get("snowflake", {})


def get_timeseries_config(cfg: dict | None = None) -> dict[str, Any]:
    """Extract time-series specific config."""
    cfg = cfg or load_config()
    return cfg.get("timeseries", {})


def get_model_config(model_name: str, cfg: dict | None = None) -> dict[str, Any]:
    """Get config for a specific model (lightgbm, nhits, baseline)."""
    cfg = cfg or load_config()
    models = cfg.get("models", {})
    if model_name not in models:
        raise KeyError(f"Model '{model_name}' not found in config. Available: {list(models.keys())}")
    return models[model_name]


# Convenience: project-wide random state
RANDOM_STATE = 42
