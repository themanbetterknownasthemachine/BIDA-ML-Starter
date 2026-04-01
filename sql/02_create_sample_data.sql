-- =============================================================================
-- BIDA ML Starter — Synthetische Testdaten
-- =============================================================================
-- Erzeugt realistische Zeitreihen-Daten zum Testen des Templates.
-- Kann auf jedem Snowflake-Account ausgeführt werden.
-- =============================================================================

USE DATABASE ML_DB;
USE SCHEMA DATA;
USE WAREHOUSE ML_WH;

-- Tägliche Daten mit Saisonalität (2.5 Jahre, 4 Serien als Beispiel)
CREATE OR REPLACE TABLE fct_daily_sales AS
WITH date_series AS (
    SELECT 
        DATEADD('day', SEQ4(), '2022-01-01')::DATE AS ds,
        SEQ4() AS day_num
    FROM TABLE(GENERATOR(ROWCOUNT => 900))
),
products AS (
    SELECT column1 AS unique_id, column2 AS base_value, column3 AS noise_factor
    FROM VALUES 
        ('kategorie_a', 150, 20),
        ('kategorie_b', 80, 10),
        ('kategorie_c', 200, 25),
        ('kategorie_d', 120, 15)
)
SELECT 
    p.unique_id,
    d.ds,
    GREATEST(0, ROUND(
        p.base_value
        + 30 * SIN(2 * PI() * d.day_num / 365.25)        -- Jahressaisonalität
        + 15 * SIN(2 * PI() * d.day_num / 7)              -- Wochenmuster
        + CASE DAYOFWEEK(d.ds) WHEN 0 THEN -40 ELSE 0 END -- Sonntag tief
        + CASE DAYOFWEEK(d.ds) WHEN 1 THEN 20 ELSE 0 END  -- Montag Spitze
        + CASE MONTH(d.ds) WHEN 12 THEN 30 ELSE 0 END     -- Dezember-Peak
        + UNIFORM(-1 * p.noise_factor, p.noise_factor, RANDOM())
    , 1)) AS y
FROM date_series d
CROSS JOIN products p
ORDER BY p.unique_id, d.ds;

-- Prüfung
SELECT 
    unique_id,
    COUNT(*) AS tage,
    MIN(ds) AS von,
    MAX(ds) AS bis,
    ROUND(AVG(y), 1) AS avg_menge,
    ROUND(STDDEV(y), 1) AS std_menge
FROM fct_daily_sales
GROUP BY unique_id
ORDER BY unique_id;
