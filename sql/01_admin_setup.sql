-- =============================================================================
-- BIDA ML Starter — Snowflake Admin Setup
-- =============================================================================
-- Ausführen als ACCOUNTADMIN (einmalig)
-- Erstellt: Datenbank, Schemas, Rollen, Warehouse, External Access
-- =============================================================================

USE ROLE ACCOUNTADMIN;

-- 1. Datenbank & Schemas
CREATE DATABASE IF NOT EXISTS ML_DB;
CREATE SCHEMA IF NOT EXISTS ML_DB.DATA;
CREATE SCHEMA IF NOT EXISTS ML_DB.INFERENCE;
CREATE SCHEMA IF NOT EXISTS ML_DB.NOTEBOOKS;

-- 2. Warehouse für ML-Arbeit
CREATE WAREHOUSE IF NOT EXISTS ML_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  COMMENT = 'Warehouse für ML Notebooks und Queries';

-- 3. Rolle für Data Science Team
CREATE ROLE IF NOT EXISTS ROLE_DS;
COMMENT ON ROLE ROLE_DS IS 'Data Science Team — ML Entwicklung';

-- Berechtigungen vergeben
GRANT USAGE ON DATABASE ML_DB TO ROLE ROLE_DS;
GRANT ALL ON SCHEMA ML_DB.DATA TO ROLE ROLE_DS;
GRANT ALL ON SCHEMA ML_DB.INFERENCE TO ROLE ROLE_DS;
GRANT ALL ON SCHEMA ML_DB.NOTEBOOKS TO ROLE ROLE_DS;
GRANT CREATE NOTEBOOK ON SCHEMA ML_DB.NOTEBOOKS TO ROLE ROLE_DS;
GRANT CREATE SERVICE ON SCHEMA ML_DB.NOTEBOOKS TO ROLE ROLE_DS;
GRANT ALL ON WAREHOUSE ML_WH TO ROLE ROLE_DS;

-- Rolle an User zuweisen (anpassen!)
-- GRANT ROLE ROLE_DS TO USER toni_buehlmann;
-- GRANT ROLE ROLE_DS TO USER max_mustermann;

-- 4. External Access für pip install (PyPI)
CREATE OR REPLACE NETWORK RULE pypi_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'pypi.org',
    'files.pythonhosted.org',
    'conda.anaconda.org',
    'repo.anaconda.com',
    'github.com'
  );

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION pypi_access
  ALLOWED_NETWORK_RULES = (pypi_network_rule)
  ENABLED = TRUE;

GRANT USAGE ON INTEGRATION pypi_access TO ROLE ROLE_DS;

-- 5. Compute Pool Zugriff (System-Pools)
GRANT USAGE ON COMPUTE POOL SYSTEM_COMPUTE_POOL_CPU TO ROLE ROLE_DS;
-- Optional GPU:
-- GRANT USAGE ON COMPUTE POOL SYSTEM_COMPUTE_POOL_GPU TO ROLE ROLE_DS;

-- 6. Snowflake ML Functions Berechtigung
GRANT CREATE SNOWFLAKE.ML.FORECAST ON SCHEMA ML_DB.DATA TO ROLE ROLE_DS;

-- =============================================================================
-- Fertig! Nächster Schritt: 02_create_sample_data.sql
-- =============================================================================
