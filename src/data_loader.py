"""Snowflake data loader.

Reads credentials from .env and opens one Snowpark session per Python process.
Set exactly one auth method in .env:
    1. Key-Pair:         SF_PRIVATE_KEY_PATH  (recommended, no MFA prompt)
    2. External Browser: SF_AUTHENTICATOR=externalbrowser  (SSO)
    3. Password:         SF_PASSWORD  (may trigger an MFA prompt)
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_SESSION = None


def _load_private_key(path: str) -> bytes:
    """Read a PEM private key and return it as DER bytes for the connector."""
    from cryptography.hazmat.primitives import serialization

    key_path = Path(path).expanduser()
    with open(key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _env(name: str, default: str | None = None) -> str | None:
    """Read SF_<name> from the environment, falling back to SNOWFLAKE_<name>.

    SF_* is the convention of this repo; SNOWFLAKE_* is what the
    Agentic-Engineering-Starter-Template uses in its .env.example.
    """
    return os.getenv(f"SF_{name}") or os.getenv(f"SNOWFLAKE_{name}") or default


def _connection_params() -> dict:
    """Build Snowpark connection parameters from .env."""
    load_dotenv()

    account = _env("ACCOUNT")
    user = _env("USER")
    if not account or not user:
        raise ValueError("SF_ACCOUNT and SF_USER must be set in .env (see .env.example)")

    params = {
        "account": account,
        "user": user,
        "role": _env("ROLE", "ML_DEVELOPER"),
        "warehouse": _env("WAREHOUSE", "CONSUMER"),
        "database": _env("DATABASE", "PROD_ML"),
        "schema": _env("SCHEMA", "INFERENCE"),
    }

    private_key_path = _env("PRIVATE_KEY_PATH")
    authenticator = (_env("AUTHENTICATOR") or "").lower()
    password = _env("PASSWORD")

    if private_key_path:
        params["private_key"] = _load_private_key(private_key_path)
    elif authenticator == "externalbrowser":
        params["authenticator"] = "externalbrowser"
    elif password:
        params["password"] = password
    else:
        raise ValueError(
            "No auth method configured in .env. Set one of:\n"
            "  SF_PRIVATE_KEY_PATH=~/.snowflake/rsa_key.p8  (recommended)\n"
            "  SF_AUTHENTICATOR=externalbrowser             (SSO)\n"
            "  SF_PASSWORD=...                              (may require MFA)"
        )
    return params


def get_session():
    """Return the shared Snowpark session, creating it on first call.

    Returns:
        snowflake.snowpark.Session
    """
    global _SESSION
    if _SESSION is None:
        from snowflake.snowpark import Session

        _SESSION = Session.builder.configs(_connection_params()).create()
    return _SESSION


def load_query(query: str) -> pd.DataFrame:
    """Run a SQL query and return the result as a DataFrame."""
    return get_session().sql(query).to_pandas()


def load_table(table_name: str, limit: int | None = None) -> pd.DataFrame:
    """Load a table or view as DataFrame.

    Args:
        table_name: Fully qualified name (e.g. 'PROD_DATALAKE.MSACCESS.MY_VIEW').
        limit: Optional row limit.
    """
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"
    return load_query(query)


def load_timeseries(table_name: str | None = None, unique_id: str | None = None) -> pd.DataFrame:
    """Load time series data in [unique_id, ds, y] format (columns lowercased).

    Args:
        table_name: Table or view to load. If None, uses tables.training_data from config.yaml.
        unique_id: Optional filter to a single series.
    """
    if table_name is None:
        from src.config import load_config

        table_name = load_config().get("tables", {}).get("training_data")
        if not table_name:
            raise ValueError("No table configured. Set tables.training_data in configs/config.yaml")

    query = f"SELECT * FROM {table_name}"
    if unique_id:
        query += f" WHERE unique_id = '{unique_id}'"
    query += " ORDER BY unique_id, ds"

    df = load_query(query)
    df.columns = [c.lower() for c in df.columns]
    return df


def write_to_snowflake(
    df: pd.DataFrame,
    table_name: str,
    schema: str | None = None,
    database: str | None = None,
    overwrite: bool = False,
) -> None:
    """Write a DataFrame to PROD_ML (default schema: INFERENCE).

    Column names are written unquoted, i.e. Snowflake stores them in UPPER CASE.

    Args:
        df: DataFrame to write.
        table_name: Target table name (created automatically if missing).
        schema: Target schema. Default: snowflake.schemas.inference from config.yaml.
        database: Target database. Default: snowflake.database from config.yaml.
        overwrite: If True, replace the table. If False, append rows.
    """
    from src.config import load_config

    sf_cfg = load_config().get("snowflake", {})
    database = database or sf_cfg.get("database", "PROD_ML")
    schema = schema or sf_cfg.get("schemas", {}).get("inference", "INFERENCE")

    get_session().write_pandas(
        df,
        table_name=table_name,
        database=database,
        schema=schema,
        auto_create_table=True,
        overwrite=overwrite,
        quote_identifiers=False,
    )
    mode = "replaced" if overwrite else "appended"
    print(f"{len(df)} rows written to {database}.{schema}.{table_name} ({mode}).")
