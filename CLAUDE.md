# CLAUDE.md - BIDA ML Starter

## Projekt-Überblick

Pistor BIDA ML Starter: Template für reproduzierbare Machine-Learning-Projekte
(Forecasting, Klassifikation, Regression) im BI & Data Analytics Team.

- Gearbeitet wird **lokal** (Windows-Notebook, optional NVIDIA-GPU) oder auf einer Linux-VM via Remote-SSH,
  in VS Code mit Claude Code.
- **Datenquelle ist frei wählbar:** Snowflake (häufig Views in `PROD_DATALAKE`), Dateien (CSV, Excel, Parquet
  in `data/raw/`) oder andere Systeme. **Ziel ist immer Snowflake:** die ML-Datenbank `PROD_ML`.
- Environment mit `uv` (empfohlen) oder `conda`; beide installieren dieselben Versionen.

## Projektstruktur

```
BIDA-ML-Starter/
├── CLAUDE.md                     -> Diese Datei
├── README.md                     -> Setup-Anleitung für das Team
├── pyproject.toml / uv.lock      -> Dependencies (uv); Extras: cpu, gpu
├── environment.yml / requirements.txt -> Conda-Variante (requirements.txt aus uv.lock exportiert)
├── .python-version               -> 3.12
├── .pre-commit-config.yaml       -> ruff + nbstripout
├── .env.example                  -> Snowflake Credentials Template
├── .claude/
│   ├── settings.json             -> Team-Settings (Permissions, Hooks)
│   ├── skills/<name>/SKILL.md    -> forecasting, classification, regression (Code-Referenz)
│   ├── rules/                    -> security (immer), notebooks, python (pfadbezogen)
│   ├── hooks/                    -> protect-files.sh: blockt Credential-Zugriff und destruktives SQL
│   └── agents/                   -> code-reviewer, security-reviewer (read-only)
├── configs/config.yaml           -> Zentrale Konfiguration (Snowflake-Ziel, Quelltabellen, Seeds)
├── sql/01_ml_developer_grants.sql -> Berechtigungen der Rolle ML_DEVELOPER (Referenz für Admins)
├── notebooks/
│   ├── 00_environment_check.ipynb -> Smoke-Test (mit Code)
│   ├── 01_data_exploration.ipynb  -> EDA (nur Anleitung, 8 Schritte)
│   ├── 02_forecasting.ipynb       -> Zeitreihen-Forecasting (nur Anleitung, 20 Schritte)
│   ├── 03_classification.ipynb    -> Klassifikation (nur Anleitung, 11 Schritte)
│   └── 04_regression.ipynb        -> Regression (nur Anleitung, 16 Schritte)
├── src/
│   ├── config.py                 -> Config laden (YAML), RANDOM_STATE
│   └── data_loader.py            -> Snowflake-Session (.env), load_query/load_table/load_timeseries, write_to_snowflake
├── tests/test_config.py
├── data/                         -> Lokale Daten (nicht in Git)
├── models/                       -> Trainierte Modelle (nicht in Git)
├── reports/figures/              -> Plots, Metriken
└── docs/                         -> Kurzanleitung für das Team (PDF und HTML, gleicher Stil wie die Report-Anleitungen)
```

## Snowflake-Struktur

| Bereich | Objekt | Zweck |
|---------|--------|-------|
| Quelle (optional, nur lesen) | z.B. `PROD_DATALAKE.<QUELLE>.PSA_*` und `*_90` Views | Häufigster Fall: historisierte Rohdaten, `_90` = rollierendes 90-Tage-Fenster. Andere Quellen sind zulässig |
| Ziel | `PROD_ML.INFERENCE` | Forecasts / Predictions je Lauf (z.B. `FORECAST_SNAPSHOT`) |
| Ziel | `PROD_ML.REGISTRY` | Modell-Läufe, Parameter, Deployment (`MODEL_RUNS`, `MODEL_COEFFICIENTS`, `MODEL_DEPLOYMENT`) |
| Ziel | `PROD_ML.MONITORING` | Metriken und DQ-Checks (`MODEL_EVALUATION_LOG`, `DQ_CHECK_LOG`, `V_FORECAST_VS_ACTUAL`) |

- Rolle `ML_DEVELOPER`, Warehouse `CONSUMER` (Defaults in `configs/config.yaml` und `.env.example`).
- **Umgebungen (DEV / INT / PROD):** Einziger Schalter ist `ML_ENV` in `.env`. `configs/config.yaml`
  enthält Vorlagen mit `{env}` (`{env}_ML`, `{env}_DATALAKE...`), die `load_config()` auflöst.
  Auf Laptops immer `DEV` (heute existiert nur `DEV_ML`; `PROD_ML` kommt mit dem Deployment).
  Nie `DEV_ML` oder `PROD_ML` fest in Code, Config oder Notebooks schreiben; `PROD_ML` in dieser
  Datei meint "die ML-Datenbank der jeweiligen Umgebung".
- `PROD_ML` ist bewusst vom Data-Vault-Modell getrennt. Aus dem Notebook wird **nur nach PROD_ML** geschrieben.
  Der Rückfluss ins DWH (`PROD_LANDING.ML` -> `PROD_DATALAKE.ML` -> `PROD_CONSUMPTION`) und Power BI liegen beim DWH-Team.
- Kein Schreiben nach `PROD_DATALAKE`, `PROD_LANDING` oder `PROD_CONSUMPTION`.

## Konzept

### Notebooks = Anleitungen (nur Text)
- `01_` bis `04_` enthalten keine Code-Zellen, nur eine Schritt-für-Schritt-Anleitung (Deutsch).
- Der Data Scientist legt ein eigenes Arbeits-Notebook an, liest die Anleitung und fragt Claude Code
  nach dem Code für den jeweiligen Schritt.
- Ausnahme: `00_environment_check.ipynb` enthält fertigen Code.

### Skills = Code-Referenz für Claude
- `.claude/skills/forecasting`, `classification`, `regression` enthalten den Code für alle Schritte
  des jeweiligen Notebooks. Claude lädt sie automatisch, wenn das Thema passt (`/forecasting` erzwingt es).

### src/ = Minimale Infrastruktur
- Nur `config.py` und `data_loader.py`. Features, Evaluation, Plots und Modeling entstehen im Notebook.
- Wiederverwendbare Funktionen können später nach `src/` wandern (mit Test in `tests/`).

## Schnittstelle zum Agentic Engineering Starter Template

Dieses Repo ist die ML-Methoden-Bibliothek; das Projekt-Skelett (Spec, Verifier, Hooks, Security)
liefert das Agentic-Engineering-Starter-Template. Dessen `scripts/new-ml-project.sh` kopiert von hier:
`.claude/skills/<name>/SKILL.md`, `notebooks/0X_<name>.ipynb` (Paar), `src/config.py`,
`src/data_loader.py`, `configs/config.yaml`. Diese Pfade nicht umbenennen oder verschieben.
Skills müssen ohne die hiesigen Rules und Settings funktionieren.

## Umgebung & Befehle

```bash
# uv (empfohlen)
uv sync --extra cpu          # Laptop ohne GPU, VM
uv sync --extra gpu          # Lenovo Notebook mit NVIDIA GPU (CUDA 13.0)
uv run python -m pytest      # Launcher-EXEs (pytest.exe, pre-commit.exe) sind auf Pistor-Notebooks blockiert
uv run ruff check --fix src tests && uv run ruff format src tests
uv run python -m pre_commit install    # einmalig
uv lock && uv export --no-hashes --group dev -o requirements.txt   # nach Dependency-Änderung (hält conda synchron)

# conda
conda env create -f environment.yml && conda activate bida-ml
```

- Firmen-Proxy: `pyproject.toml` setzt `system-certs = true`, damit `uv` die Windows-Zertifikate nutzt.
- Dependencies nur in `pyproject.toml` ändern, nie direkt in `requirements.txt`.

## Sprach-Konventionen

- **Notebooks**: Deutsch (Markdown-Zellen)
- **Python-Code**: Englisch (Funktionsnamen, Variablen, Docstrings, Kommentare)
- **SQL**: Englisch, Objektnamen UPPER_CASE
- **Commit Messages**: Englisch
- Keine Emojis im Projekt (Doku, Notebooks, Code, Commits).
- Deutsche Texte mit echten Umlauten (ä, ö, ü), Schweizer Schreibweise ohne Eszett (ss), keine Gedankenstriche.

## Workflow-Regeln

- Daten laden über `src.data_loader` (`load_query`, `load_table`, `load_timeseries`); Tabellennamen aus `configs/config.yaml`
  (mit `{env}`-Platzhalter, z.B. `"{env}_DATALAKE.MSACCESS.PSA_..._90"`).
- Ergebnisse mit `write_to_snowflake(df, "TABELLE", schema=...)` nach `PROD_ML` schreiben (append, nicht overwrite),
  immer mit `run_id`, Modellname, Version und Zeitstempel.
- Trainierte Modelle unter `models/<name>_v<n>_<datum>/` speichern (nicht in Git).
- Metriken und Plots nach `reports/figures/`.
- Reproduzierbarkeit: `random_state=42` (`RANDOM_STATE` aus `src.config`), keine hardcodierten Pfade oder Tabellennamen.

## ML-Workflow

| Problem | Template | Skill | Modelle |
|---------|----------|-------|---------|
| Zeitreihen-Prognosen | `02_forecasting.ipynb` | forecasting | StatsForecast, MLForecast (LightGBM/XGBoost), NeuralForecast (N-HiTS, TFT, ...) |
| Binäre/Multi-Class Klassifikation | `03_classification.ipynb` | classification | LightGBM, XGBoost, sklearn |
| Kontinuierliche Zielvariable | `04_regression.ipynb` | regression | LightGBM, XGBoost, Optuna |

### Pflicht-Schritte
1. Daten laden (Snowflake, Datei oder andere Quelle) und prüfen
2. EDA (Missing Values, Verteilungen, Zusammenhänge)
3. Train/Test Split (zeitlich bei Zeitreihen, stratified bei Klassifikation)
4. Baseline-Modell (jedes ML-Modell muss die Baseline schlagen)
5. Cross-Validation
6. Evaluation auf dem Test-Set mit Visualisierung
7. Ergebnisse nach PROD_ML schreiben

### Metriken
| Forecasting | Klassifikation | Regression |
|-------------|----------------|------------|
| MAE, RMSE, MAPE, sMAPE | Accuracy, F1, ROC-AUC, PR-AUC | RMSE, MAE, R2, MAPE |
| Forecast-Plot | Confusion Matrix | Residuen-Analyse |

## Code-Standards

- Funktionen `snake_case`, Klassen `PascalCase`, Konstanten `UPPER_SNAKE_CASE`.
- ruff (Zeilenlänge 100, Regeln E/F/I/W/B/UP), Type Hints.
- Notebook-Outputs werden vor dem Commit von nbstripout entfernt.

## Hinweise für Claude

- Der User folgt einer Text-Anleitung im Notebook und braucht Code für einzelne Schritte.
  Code gehört in das Arbeits-Notebook des Users, nicht in die Template-Notebooks und nicht in `src/`.
- Passenden Skill nutzen (forecasting / classification / regression) und die Schritt-Nummer des Notebooks referenzieren.
- Zuerst klären: Welche Datenquelle (Snowflake-Tabelle, Datei, anderes System), welche Zielvariable, welcher Horizont.
  Bei Snowflake-Quelle: fehlt `tables.training_data` in der Config, darauf hinweisen. Bei Dateien: mit pandas aus `data/raw/` laden.
- Bei Zeitreihen immer zeitlich splitten, bei Klassifikation immer stratified.
- Baseline zuerst, dann komplexere Modelle.
- GPU: NeuralForecast/PyTorch nutzen CUDA automatisch; mit `torch.cuda.is_available()` prüfen.
- `.env` und Private Keys nie lesen oder ausgeben.
- Ergebnisse immer nach PROD_ML schreiben; nichts nach PROD_DATALAKE / PROD_CONSUMPTION.
- Notebook-Text auf Deutsch, Code auf Englisch, keine Emojis.
