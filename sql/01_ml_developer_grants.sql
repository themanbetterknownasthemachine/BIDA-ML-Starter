-- =============================================================================
-- BIDA ML Starter - Berechtigungen für die Rolle ML_DEVELOPER (Referenz für Admins)
-- =============================================================================
-- Grundlage: ADR "Snowflake-Anbindung aus lokalen Python-Notebooks".
-- Nur von einem Admin ausführen, wenn die Rolle eingerichtet wird oder ein neuer
-- User oder ein neues Quellschema dazukommt. Claude Code führt dieses Script nie aus.
--
-- Grundsätze:
--   - Lesen:     DEV_DATALAKE und PROD_DATALAKE (freigegebene Quellschemas, nur SELECT).
--   - Schreiben: nur DEV_ML (Schemas INFERENCE, REGISTRY, MONITORING).
--   - INT_ML und PROD_ML: KEINE Rechte von Laptops aus. Objekte dort werden im ITB
--     modelliert und über das offizielle Deployment angelegt.
--   - Kein Schreiben nach *_DATALAKE, *_LANDING oder *_CONSUMPTION.
--
-- Passend dazu in .env: ML_ENV=DEV (Ziel) und optional SOURCE_ENV=PROD (Quelle).
-- =============================================================================

USE ROLE SECURITYADMIN;

-- -----------------------------------------------------------------------------
-- 1. Warehouse
-- -----------------------------------------------------------------------------
GRANT USAGE ON WAREHOUSE CONSUMER TO ROLE ML_DEVELOPER;

-- -----------------------------------------------------------------------------
-- 2. Leserechte auf Quellen (optional über eine eigene Leserolle ML_READER)
--    Pro freigegebenem Quellschema wiederholen (MSACCESS, WAMAS4, ...).
-- -----------------------------------------------------------------------------
CREATE ROLE IF NOT EXISTS ML_READER;

GRANT USAGE ON WAREHOUSE CONSUMER TO ROLE ML_READER;

-- PROD_DATALAKE (produktive Daten, nur lesen)
GRANT USAGE ON DATABASE PROD_DATALAKE TO ROLE ML_READER;
GRANT USAGE ON SCHEMA PROD_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON ALL TABLES    IN SCHEMA PROD_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON ALL VIEWS     IN SCHEMA PROD_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON FUTURE TABLES IN SCHEMA PROD_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON FUTURE VIEWS  IN SCHEMA PROD_DATALAKE.MSACCESS TO ROLE ML_READER;

-- DEV_DATALAKE (Entwicklungsdaten, nur lesen)
GRANT USAGE ON DATABASE DEV_DATALAKE TO ROLE ML_READER;
GRANT USAGE ON SCHEMA DEV_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON ALL TABLES    IN SCHEMA DEV_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON ALL VIEWS     IN SCHEMA DEV_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON FUTURE TABLES IN SCHEMA DEV_DATALAKE.MSACCESS TO ROLE ML_READER;
GRANT SELECT ON FUTURE VIEWS  IN SCHEMA DEV_DATALAKE.MSACCESS TO ROLE ML_READER;

-- Optional, falls freigegeben: *_CONSUMPTION analog nur mit SELECT.

GRANT ROLE ML_READER TO ROLE ML_DEVELOPER;

-- -----------------------------------------------------------------------------
-- 3. Volle Arbeitsrechte in DEV_ML (Outputs der Notebooks)
--    CREATE STAGE braucht write_pandas(auto_create_table=True) für den Upload.
--    CREATE VIEW erlaubt eigene Hilfs-Views in DEV (z.B. Forecast vs. Actual).
-- -----------------------------------------------------------------------------
GRANT USAGE ON DATABASE DEV_ML TO ROLE ML_DEVELOPER;

GRANT USAGE, CREATE TABLE, CREATE VIEW, CREATE STAGE
  ON SCHEMA DEV_ML.INFERENCE  TO ROLE ML_DEVELOPER;
GRANT USAGE, CREATE TABLE, CREATE VIEW, CREATE STAGE
  ON SCHEMA DEV_ML.REGISTRY   TO ROLE ML_DEVELOPER;
GRANT USAGE, CREATE TABLE, CREATE VIEW, CREATE STAGE
  ON SCHEMA DEV_ML.MONITORING TO ROLE ML_DEVELOPER;

GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES    IN SCHEMA DEV_ML.INFERENCE  TO ROLE ML_DEVELOPER;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON FUTURE TABLES IN SCHEMA DEV_ML.INFERENCE  TO ROLE ML_DEVELOPER;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES    IN SCHEMA DEV_ML.REGISTRY   TO ROLE ML_DEVELOPER;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON FUTURE TABLES IN SCHEMA DEV_ML.REGISTRY   TO ROLE ML_DEVELOPER;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES    IN SCHEMA DEV_ML.MONITORING TO ROLE ML_DEVELOPER;
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON FUTURE TABLES IN SCHEMA DEV_ML.MONITORING TO ROLE ML_DEVELOPER;

GRANT SELECT ON ALL VIEWS    IN DATABASE DEV_ML TO ROLE ML_DEVELOPER;
GRANT SELECT ON FUTURE VIEWS IN DATABASE DEV_ML TO ROLE ML_DEVELOPER;

-- -----------------------------------------------------------------------------
-- 4. INT_ML / PROD_ML: bewusst keine Grants für ML_DEVELOPER.
--    Schreiben dort nur über das Deployment (Service-User, eigene Rolle).
-- -----------------------------------------------------------------------------

-- -----------------------------------------------------------------------------
-- 5. Rolle an User zuweisen (pro User, anpassen)
-- -----------------------------------------------------------------------------
-- GRANT ROLE ML_DEVELOPER TO USER <user>;

-- -----------------------------------------------------------------------------
-- 6. Key-Pair-Auth pro User (Public Key aus rsa_key.pub, ohne BEGIN/END-Zeilen)
--    Passwort-Login nur als Notlösung.
-- -----------------------------------------------------------------------------
-- ALTER USER <user> SET RSA_PUBLIC_KEY='MIIBIjANBgkqhk...';

-- -----------------------------------------------------------------------------
-- 7. Prüfen
-- -----------------------------------------------------------------------------
-- SHOW GRANTS TO ROLE ML_DEVELOPER;
-- SHOW GRANTS TO ROLE ML_READER;
