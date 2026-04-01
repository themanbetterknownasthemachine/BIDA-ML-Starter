"""Data loading utilities — works in Snowflake Notebooks and on VM/local."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


# ---------------------------------------------------------------------------
# Snowflake session helper
# ---------------------------------------------------------------------------

def get_session():
    """Get a Snowflake session.

    In a Snowflake Notebook the session is provided automatically.
    On a VM/local machine we build one from environment variables.
    """
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        pass

    # Fallback: build session from .env (VM / local)
    import os
    from dotenv import load_dotenv
    from snowflake.snowpark import Session

    load_dotenv()
    connection_params = {
        "account": os.getenv("SF_ACCOUNT"),
        "user": os.getenv("SF_USER"),
        "password": os.getenv("SF_PASSWORD", ""),
        "private_key_file": os.getenv("SF_PRIVATE_KEY_PATH", ""),
        "role": os.getenv("SF_ROLE", "ROLE_DS"),
        "warehouse": os.getenv("SF_WAREHOUSE", "ML_WH"),
        "database": os.getenv("SF_DATABASE", "ML_DB"),
        "schema": os.getenv("SF_SCHEMA", "DATA"),
    }
    # Remove empty values
    connection_params = {k: v for k, v in connection_params.items() if v}
    return Session.builder.configs(connection_params).create()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_table(table_name: str | None = None, session=None) -> pd.DataFrame:
    """Load a full table from Snowflake as pandas DataFrame.

    Args:
        table_name: Fully qualified table name. If None, reads from config.
        session: Optional Snowpark session. Auto-detected if None.

    Returns:
        pandas DataFrame with all rows.
    """
    if table_name is None:
        cfg = load_config()
        table_name = cfg["tables"]["training_data"]

    session = session or get_session()
    df = session.table(table_name).to_pandas()

    # Snowflake returns UPPERCASE columns — normalize to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df


def load_query(sql: str, session=None) -> pd.DataFrame:
    """Execute a SQL query and return result as pandas DataFrame.

    Args:
        sql: SQL query string.
        session: Optional Snowpark session.

    Returns:
        pandas DataFrame with query results.
    """
    session = session or get_session()
    df = session.sql(sql).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    return df


def load_timeseries(
    table_name: str | None = None,
    unique_id: str | None = None,
    min_date: str | None = None,
    session=None,
) -> pd.DataFrame:
    """Load time-series data with optional filtering.

    Args:
        table_name: Source table. Reads from config if None.
        unique_id: Filter to a specific series (e.g. 'kategorie_a').
        min_date: Only load data from this date onwards (e.g. '2022-01-01').
        session: Optional Snowpark session.

    Returns:
        pandas DataFrame with columns [unique_id, ds, y, ...].
    """
    cfg = load_config()
    ts = cfg["timeseries"]

    if table_name is None:
        table_name = cfg["tables"]["training_data"]

    where_clauses = []
    if unique_id:
        where_clauses.append(f"{ts['id_column']} = '{unique_id}'")
    if min_date:
        where_clauses.append(f"{ts['time_column']} >= '{min_date}'")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"""
        SELECT {ts['id_column']}, {ts['time_column']}, {ts['target_column']}
        FROM {table_name}
        {where_sql}
        ORDER BY {ts['id_column']}, {ts['time_column']}
    """
    return load_query(sql, session=session)
