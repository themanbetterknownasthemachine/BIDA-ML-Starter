"""Data loader — connects to Snowflake automatically.

Detects environment and chooses the right connection method:
- Snowflake Container Runtime: get_active_session() (automatic)
- VM/lokal: reads .env and connects via one of three methods:
    1. External Browser (SSO) — easiest for teams, set SF_AUTHENTICATOR=externalbrowser
    2. Key-Pair — best for automation/VMs, set SF_PRIVATE_KEY_PATH
    3. Password + MFA — fallback, set SF_PASSWORD (will prompt for MFA if required)
"""

import os
import pandas as pd


def get_session():
    """Get a Snowflake session. Auto-detects environment.

    In Snowflake Container Runtime: uses get_active_session().
    In VM/lokal: reads credentials from .env file.

    Returns:
        snowflake.snowpark.Session
    """
    # Try Snowflake Container Runtime first
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        return session
    except Exception:
        pass

    # Fall back to local connection via .env
    from dotenv import load_dotenv
    load_dotenv()

    account = os.getenv("SF_ACCOUNT")
    user = os.getenv("SF_USER")
    role = os.getenv("SF_ROLE", "ROLE_DS")
    warehouse = os.getenv("SF_WAREHOUSE", "ML_WH")
    database = os.getenv("SF_DATABASE", "ML_DB")
    schema = os.getenv("SF_SCHEMA", "DATA")
    authenticator = os.getenv("SF_AUTHENTICATOR", "").lower()
    private_key_path = os.getenv("SF_PRIVATE_KEY_PATH", "")

    if not account or not user:
        raise ValueError("SF_ACCOUNT and SF_USER must be set in .env")

    from snowflake.snowpark import Session

    # Method 1: External Browser (SSO) — easiest for teams
    if authenticator == "externalbrowser":
        session = Session.builder.configs({
            "account": account,
            "user": user,
            "authenticator": "externalbrowser",
            "role": role,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
        }).create()
        return session

    # Method 2: Key-Pair — best for automation
    if private_key_path:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        key_path = os.path.expanduser(private_key_path)
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        session = Session.builder.configs({
            "account": account,
            "user": user,
            "private_key": private_key_bytes,
            "role": role,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
        }).create()
        return session

    # Method 3: Password (may trigger MFA prompt)
    password = os.getenv("SF_PASSWORD")
    if not password:
        raise ValueError(
            "No auth method configured in .env. Set one of:\n"
            "  SF_AUTHENTICATOR=externalbrowser  (easiest)\n"
            "  SF_PRIVATE_KEY_PATH=~/.snowflake/snowflake_key.p8  (automation)\n"
            "  SF_PASSWORD=...  (may require MFA)"
        )

    session = Session.builder.configs({
        "account": account,
        "user": user,
        "password": password,
        "role": role,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
    }).create()
    return session


def load_query(query: str) -> pd.DataFrame:
    """Run a SQL query and return results as DataFrame.

    Args:
        query: SQL query string.

    Returns:
        pandas DataFrame with query results.
    """
    session = get_session()
    return session.sql(query).to_pandas()


def load_table(table_name: str, limit: int = None) -> pd.DataFrame:
    """Load a full table as DataFrame.

    Args:
        table_name: Fully qualified table name (e.g. 'ML_DB.DATA.MY_TABLE').
        limit: Optional row limit.

    Returns:
        pandas DataFrame.
    """
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"
    return load_query(query)


def load_timeseries(table_name: str = None, unique_id: str = None) -> pd.DataFrame:
    """Load time series data in [unique_id, ds, y] format.

    Args:
        table_name: Table to load. If None, reads from config.yaml.
        unique_id: Filter to a specific series (e.g. 'kategorie_a').

    Returns:
        pandas DataFrame with columns unique_id, ds, y.
    """
    if table_name is None:
        from src.config import load_config
        cfg = load_config()
        table_name = cfg.get("tables", {}).get("training_data")
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
    database: str = None,
    schema: str = None,
    overwrite: bool = False,
) -> None:
    """Write a DataFrame to Snowflake.

    Args:
        df: pandas DataFrame to write.
        table_name: Target table name.
        database: Database (default: from config).
        schema: Schema (default: results_schema from config).
        overwrite: If True, replace table. If False, append.
    """
    if database is None or schema is None:
        from src.config import load_config
        cfg = load_config()
        sf_cfg = cfg.get("snowflake", {})
        database = database or sf_cfg.get("database", "ML_DB")
        schema = schema or sf_cfg.get("results_schema", "INFERENCE")

    session = get_session()
    session.write_pandas(
        df,
        table_name=table_name,
        database=database,
        schema=schema,
        auto_create_table=True,
        overwrite=overwrite,
    )
    mode = "überschrieben" if overwrite else "angehängt"
    print(f"{len(df)} Zeilen nach {database}.{schema}.{table_name} {mode}.")
