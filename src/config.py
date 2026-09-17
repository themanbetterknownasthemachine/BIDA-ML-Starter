"""Configuration loader for BIDA ML Starter.

The target environment (DEV / INT / PROD) is a single switch: ML_ENV in .env.
config.yaml contains templates such as "{env}_ML" or "{env}_DATALAKE.<SCHEMA>.<VIEW>";
load_config() replaces "{env}" with the current environment so notebooks and skills
never hardcode DEV_ML / PROD_ML.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ENVIRONMENTS = ("DEV", "INT", "PROD")


def get_env() -> str:
    """Return the current environment from ML_ENV in .env (default: DEV)."""
    load_dotenv()
    env = os.getenv("ML_ENV", "DEV").upper()
    if env not in ENVIRONMENTS:
        raise ValueError(f"ML_ENV must be one of {ENVIRONMENTS}, got '{env}'")
    return env


def _find_config_path() -> Path:
    """Walk up from CWD or this file to find configs/config.yaml."""
    anchors = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for anchor in anchors:
        for parent in [anchor] + list(anchor.parents):
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                return candidate
    raise FileNotFoundError("configs/config.yaml not found")


def _resolve(value: Any, env: str) -> Any:
    """Replace the {env} placeholder in all strings of a nested config structure."""
    if isinstance(value, str):
        return value.replace("{env}", env)
    if isinstance(value, dict):
        return {k: _resolve(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, env) for v in value]
    return value


def load_config(path: str | Path | None = None, env: str | None = None) -> dict[str, Any]:
    """Load the YAML config with {env} placeholders resolved.

    Args:
        path: Optional explicit path. Auto-detected if None.
        env: Environment override (DEV / INT / PROD). Default: ML_ENV from .env.

    Returns:
        Parsed config dictionary; cfg["env"] holds the resolved environment.
    """
    config_path = Path(path) if path else _find_config_path()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env = (env or get_env()).upper()
    cfg = _resolve(cfg, env)
    cfg["env"] = env
    return cfg


def get_snowflake_config(cfg: dict | None = None) -> dict[str, Any]:
    """Extract the Snowflake section (database, schemas, role, warehouse) from config."""
    cfg = cfg or load_config()
    return cfg.get("snowflake", {})


# Convenience: project-wide random state
RANDOM_STATE = 42
