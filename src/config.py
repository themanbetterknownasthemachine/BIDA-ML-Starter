"""Configuration loader for BIDA ML Starter.

Two environment switches in .env (values DEV / INT / PROD):

- ML_ENV: target environment. "{env}" in config.yaml resolves to it, e.g. "{env}_ML".
  Notebooks only ever write to this ML database.
- SOURCE_ENV (optional): source environment for reading. "{source_env}" in config.yaml
  resolves to it, e.g. "{source_env}_DATALAKE.<SCHEMA>.<VIEW>". Default: same as ML_ENV.
  Typical laptop setup: ML_ENV=DEV, SOURCE_ENV=PROD (read production data, write to DEV_ML).

load_config() replaces both placeholders so notebooks and skills never hardcode
DEV_ML / PROD_ML or DEV_DATALAKE / PROD_DATALAKE.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ENVIRONMENTS = ("DEV", "INT", "PROD")


def _validate(name: str, value: str) -> str:
    value = value.upper()
    if value not in ENVIRONMENTS:
        raise ValueError(f"{name} must be one of {ENVIRONMENTS}, got '{value}'")
    return value


def get_env() -> str:
    """Return the target environment from ML_ENV in .env (default: DEV)."""
    load_dotenv()
    return _validate("ML_ENV", os.getenv("ML_ENV", "DEV"))


def get_source_env() -> str:
    """Return the source environment from SOURCE_ENV in .env (default: ML_ENV)."""
    load_dotenv()
    value = os.getenv("SOURCE_ENV", "").strip()
    return _validate("SOURCE_ENV", value) if value else get_env()


def _find_config_path() -> Path:
    """Walk up from CWD or this file to find configs/config.yaml."""
    anchors = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for anchor in anchors:
        for parent in [anchor] + list(anchor.parents):
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                return candidate
    raise FileNotFoundError("configs/config.yaml not found")


def _resolve(value: Any, env: str, source_env: str) -> Any:
    """Replace {source_env} and {env} in all strings of a nested config structure."""
    if isinstance(value, str):
        return value.replace("{source_env}", source_env).replace("{env}", env)
    if isinstance(value, dict):
        return {k: _resolve(v, env, source_env) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, env, source_env) for v in value]
    return value


def load_config(
    path: str | Path | None = None,
    env: str | None = None,
    source_env: str | None = None,
) -> dict[str, Any]:
    """Load the YAML config with {env} and {source_env} placeholders resolved.

    Args:
        path: Optional explicit path. Auto-detected if None.
        env: Target environment override (DEV / INT / PROD). Default: ML_ENV from .env.
        source_env: Source environment override. Default: SOURCE_ENV from .env,
            falling back to the target environment.

    Returns:
        Parsed config dictionary; cfg["env"] and cfg["source_env"] hold the resolved values.
    """
    config_path = Path(path) if path else _find_config_path()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env = _validate("env", env) if env else get_env()
    if source_env:
        source_env = _validate("source_env", source_env)
    elif os.getenv("SOURCE_ENV", "").strip():
        source_env = get_source_env()
    else:
        source_env = env
    cfg = _resolve(cfg, env, source_env)
    cfg["env"] = env
    cfg["source_env"] = source_env
    return cfg


def get_snowflake_config(cfg: dict | None = None) -> dict[str, Any]:
    """Extract the Snowflake section (database, schemas, role, warehouse) from config."""
    cfg = cfg or load_config()
    return cfg.get("snowflake", {})


# Convenience: project-wide random state
RANDOM_STATE = 42
