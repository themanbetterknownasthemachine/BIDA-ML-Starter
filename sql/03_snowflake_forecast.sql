-- =============================================================================
-- BIDA ML Starter — Snowflake ML FORECAST Baseline
-- =============================================================================
-- Nutzt Snowflakes eingebaute ML-Funktion für einen Quick-Win-Forecast.
-- Kein Python nötig — reines SQL.
-- =============================================================================

USE DATABASE ML_DB;
USE SCHEMA DATA;
USE WAREHOUSE ML_WH;

-- 1. Univariates Modell (eine Zeitreihe)
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST baseline_single_model(
    INPUT_DATA => TABLE(
        SELECT ds, y FROM fct_daily_sales WHERE unique_id = 'kategorie_a'
    ),
    TIMESTAMP_COLNAME => 'DS',
    TARGET_COLNAME => 'Y'
);

-- Forecast erzeugen
SELECT * FROM TABLE(baseline_single_model!FORECAST(FORECASTING_PERIODS => 14));

-- Feature Importance anschauen
SELECT * FROM TABLE(baseline_single_model!EXPLAIN_FEATURE_IMPORTANCE());

-- Modellqualität prüfen
SELECT * FROM TABLE(baseline_single_model!SHOW_EVALUATION_METRICS());


-- 2. Multi-Series Modell (alle Artikelgruppen gleichzeitig)
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST baseline_multi_model(
    INPUT_DATA => TABLE(fct_daily_sales),
    SERIES_COLNAME => 'UNIQUE_ID',
    TIMESTAMP_COLNAME => 'DS',
    TARGET_COLNAME => 'Y'
);

-- Forecast für alle Serien
SELECT * FROM TABLE(baseline_multi_model!FORECAST(FORECASTING_PERIODS => 14))
ORDER BY series, ts;

-- Evaluation
SELECT * FROM TABLE(baseline_multi_model!SHOW_EVALUATION_METRICS());
