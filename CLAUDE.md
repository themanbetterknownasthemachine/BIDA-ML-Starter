# CLAUDE.md — BIDA ML Starter Projekt

## Projekt-Überblick

Pistor BIDA ML Starter Template für reproduzierbare Machine-Learning-Projekte.
Läuft in **drei Umgebungen** ohne Code-Änderungen: Snowflake Container Runtime, VM (Remote-SSH) und lokaler Laptop.
Einsetzbar für Forecasting, Klassifikation, Regression und weitere ML-Aufgaben in allen Geschäftsbereichen.

## Umgebungen

| Umgebung | Setup | Snowflake-Verbindung | AI-Assistent |
|----------|-------|---------------------|-------------|
| **Snowflake Container Runtime** | `pip install -r requirements-snowflake.txt` | Automatisch (`get_active_session()`) | Cortex Code |
| **VM (Remote-SSH)** | `uv sync` | Via `.env` Datei | Claude Code |
| **Lokaler Laptop** | `uv sync` | Via `.env` Datei | Claude Code |

## Projektstruktur

```
BIDA-ML-Starter/
├── CLAUDE.md                          → Diese Datei (Instruktionen für Claude / Cortex Code)
├── README.md                          → Setup-Anleitung für das Team
├── pyproject.toml                     → Dependencies für VM/lokal (uv sync)
├── requirements-snowflake.txt         → Dependencies für Snowflake Container Runtime
├── .env.example                       → Snowflake Credentials Template (nur VM/lokal)
├── configs/
│   └── config.yaml                    → Zentrale Konfiguration (Tabellen, Params)
├── skills/
│   ├── FORECASTING_SKILL.md           → Coding-Instruktionen für Forecasting (20 Schritte)
│   ├── CLASSIFICATION_SKILL.md        → Coding-Instruktionen für Klassifikation (11 Schritte)
│   └── REGRESSION_SKILL.md            → Coding-Instruktionen für Regression (16 Schritte)
├── sql/
│   ├── 01_admin_setup.sql             → Snowflake Admin: Rollen, Datenbank, EAI
│   ├── 02_create_sample_data.sql      → Synthetische Testdaten erzeugen
│   └── 03_snowflake_forecast.sql      → ML FORECAST Baseline (SQL)
├── notebooks/
│   ├── 00_environment_check.ipynb     → Smoke-Test (mit Code)
│   ├── 01_data_exploration.ipynb      → EDA (nur Anleitung, 8 Schritte)
│   ├── 02_forecasting.ipynb           → Zeitreihen-Forecasting (nur Anleitung, 20 Schritte)
│   ├── 03_classification.ipynb        → Klassifikation (nur Anleitung, 11 Schritte)
│   └── 04_regression.ipynb            → Regression (nur Anleitung, 16 Schritte)
├── src/
│   ├── __init__.py
│   ├── config.py                      → Config laden (YAML)
│   └── data_loader.py                 → Snowflake-Daten laden (erkennt Umgebung automatisch)
├── tests/
│   └── test_config.py
├── data/                              → Lokale Daten (NICHT in Git)
├── models/                            → Trainierte Modelle
├── reports/figures/                   → Plots, Metriken
└── docs/                              → Zusätzliche Dokumentation
```

## Konzept

### Notebooks = Anleitungen (nur Text)
- Notebooks enthalten **keine Code-Zellen** (Ausnahme: `00_environment_check`)
- Jedes Notebook beschreibt Schritt für Schritt, was zu tun ist und warum
- Der Data Scientist liest die Anleitung und fragt **Claude oder Cortex Code** für den Code
- Claude/Cortex Code kennt über die **Skill-Dateien** den Projektkontext und generiert passenden Code

### Skills = Code-Referenz für Claude/Cortex Code
- Jeder Skill enthält den kompletten Code für alle Schritte eines Notebooks
- Skills sind Markdown-Dateien, keine Python-Module
- Sie funktionieren in Claude Code (VS Code) und Cortex Code (Snowflake)

### src/ = Minimale Infrastruktur
- Nur `config.py` und `data_loader.py` — das Minimum für die Snowflake-Verbindung
- Alles andere (Features, Evaluation, Plots, Modeling) schreibt der Data Scientist mit Hilfe von Claude direkt im Notebook
- Wenn eine Funktion wiederverwendbar wird, kann sie später in src/ ausgelagert werden

## Sprach-Konventionen

- **Notebooks**: Deutsch (Markdown-Zellen)
- **Python-Code**: Englisch (Funktionsnamen, Variablen, Docstrings)
- **SQL**: Englisch (Snowflake-Standard)
- **Commit Messages**: Englisch

## Workflow-Regeln

### Daten laden
- **In Snowflake:** `get_active_session()` → `session.sql("SELECT ...").to_pandas()`
- **Auf VM/lokal:** Liest Credentials aus `.env` → `snowpark.Session`
- Kein hardcodierter SQL — Tabellennamen kommen aus `configs/config.yaml`

### Daten
- Rohdaten kommen aus Snowflake (keine lokalen CSVs in Produktion)
- `data/raw/` nur für lokale Tests oder Exports
- Ergebnisse werden nach Snowflake zurückgeschrieben

### Modelle
- Trainierte Modelle: `models/` mit Versionierung (z.B. `model_v1/`)
- In Snowflake: Modell auf Stage speichern oder Model Registry nutzen
- Metriken und Plots: `reports/figures/`

## ML-Workflow

### Template-Auswahl
| Problem | Template | Modelle |
|---------|----------|---------|
| Zeitreihen-Prognosen | `02_forecasting.ipynb` | Statistisch, ML, Neural (volles Spektrum) |
| Binäre/Multi-Class Klassifikation | `03_classification.ipynb` | LightGBM, XGBoost, sklearn |
| Kontinuierliche Zielvariable | `04_regression.ipynb` | LightGBM, XGBoost, Optuna |
| Quick-Win SQL Baseline (Zeitreihen) | `sql/03_snowflake_forecast.sql` | Snowflake ML FORECAST |

### Pflicht-Schritte (immer durchführen)
1. Daten aus Snowflake laden und prüfen
2. EDA (Missing Values, Verteilungen, Zusammenhänge)
3. Train/Test Split (zeitlich bei Zeitreihen, stratified bei Klassifikation)
4. Baseline Model (Pflicht! Jedes ML-Modell muss die Baseline schlagen)
5. Cross-Validation
6. Evaluation auf Test-Set mit Visualisierung
7. Ergebnisse nach Snowflake schreiben

### Metriken
| Forecasting | Klassifikation | Regression |
|-------------|----------------|------------|
| MAE, RMSE | Accuracy, F1 | RMSE, MAE |
| MAPE, sMAPE | ROC-AUC, PR-AUC | R², MAPE |
| Forecast-Plot | Confusion Matrix | Residuen-Analyse |

## Verfügbare Skills

### `skills/FORECASTING_SKILL.md`
Zeitreihen-Forecasting mit dem vollen Modellspektrum: Statistische Modelle (SARIMAX, AutoARIMA, ETS, Theta, OLS), Klassische ML (LightGBM, XGBoost via MLForecast), Neurale Modelle (N-HiTS, N-BEATS, PatchTST, TFT, TSMixerx, TiDE, NeuralProphet). Baseline ist Pflicht. Enthält den Code für alle 20 Schritte aus `02_forecasting.ipynb`.

### `skills/CLASSIFICATION_SKILL.md`
Klassifikation mit LightGBM/XGBoost und sklearn Pipelines. Enthält den Code für alle 11 Schritte aus `03_classification.ipynb`.

### `skills/REGRESSION_SKILL.md`
Regression mit LightGBM/XGBoost und Optuna. Enthält den Code für alle 16 Schritte aus `04_regression.ipynb`.

## Code-Standards

### Python
- Funktionen: `snake_case`
- Klassen: `PascalCase`
- Konstanten: `UPPER_SNAKE_CASE` (`RANDOM_STATE = 42`)

### Reproduzierbarkeit
- Immer `random_state=42` setzen
- Snowflake: `pip install -r requirements-snowflake.txt`
- VM/lokal: `uv sync`
- Keine hardcodierten Pfade oder Tabellennamen — Config nutzen

## Häufige Befehle

### In Snowflake (Terminal im Notebook)
```bash
pip install -r requirements-snowflake.txt
pip list | grep -i "lightgbm\|neuralforecast\|shap"
```

### Auf VM (Remote-SSH) oder lokal
```bash
uv sync
jupyter lab
pytest
```

### Git (alle Umgebungen)
```bash
git pull
git add . && git commit -m "..."
git push
```

## Hinweise für Claude / Cortex Code

- Der User folgt einer Text-Anleitung im Notebook und braucht Code für einzelne Schritte
- Lies den passenden Skill um den Kontext zu verstehen
- Frage zuerst, in welcher Umgebung der User arbeitet (Snowflake, VM, lokal)
- Bei Snowflake: `get_active_session()` nutzen, sys.path Zelle einbauen
- Bei VM/lokal: `.env` basierte Verbindung, `uv sync` für Packages
- Bei Zeitreihen: Immer **zeitlichen** Train/Test Split, nie random
- Bei Klassifikation: Immer **stratified** splitten
- Tabellennamen aus `configs/config.yaml` lesen, nicht hardcoden
- Code direkt im Notebook schreiben (nicht in src/ Module auslagern)
- Ergebnisse immer nach Snowflake zurückschreiben
- Sprache: Notebook-Text auf Deutsch, Code auf Englisch
