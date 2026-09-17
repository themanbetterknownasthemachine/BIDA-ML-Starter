# BIDA ML Starter - Pistor

Standardisiertes ML-Template für das Pistor BI & Data Analytics Team.
Einsetzbar für Forecasting, Klassifikation, Regression und weitere ML-Aufgaben.

Das Projekt wird lokal geklont (Windows-Notebook, optional mit NVIDIA-GPU) oder auf einer Linux-VM
via Remote-SSH genutzt. Die Datenquelle ist frei wählbar (Snowflake, CSV, Excel, Parquet, andere Systeme);
Ziel für alle Ergebnisse ist die ML-Datenbank in Snowflake (`PROD_ML`).
Der Code entsteht in VS Code mit Claude Code, geleitet durch die Template-Notebooks und Skills.

## Voraussetzungen

- Git, VS Code mit den Extensions Python, Jupyter und Claude Code
- Python-Umgebung: `uv` (empfohlen) oder `conda`
- Snowflake-User mit Rolle `ML_DEVELOPER` (Berechtigungen siehe `sql/01_ml_developer_grants.sql`)

## Setup mit uv (empfohlen)

1. uv installieren (einmalig): `winget install astral-sh.uv` oder über die IT.
2. Repository klonen und Umgebung erstellen:
   ```bash
   git clone <repo-url>
   cd BIDA-ML-Starter
   uv sync --extra cpu      # Laptop ohne GPU, VM
   uv sync --extra gpu      # Lenovo Notebook mit NVIDIA GPU (CUDA 13.0)
   ```
   `uv` lädt Python 3.12 automatisch und legt `.venv/` an. Der Firmen-Proxy ist bereits berücksichtigt
   (`system-certs = true` in `pyproject.toml`).
3. Credentials anlegen (siehe Abschnitt Snowflake-Zugang):
   ```bash
   cp .env.example .env
   ```
4. Git-Hooks aktivieren (ruff, nbstripout):
   ```bash
   uv run python -m pre_commit install
   ```
   Hinweis: Auf den Pistor-Notebooks werden die Launcher-EXEs in `.venv/Scripts` (z.B. `pytest.exe`,
   `pre-commit.exe`) blockiert. Tools deshalb immer als Modul starten: `uv run python -m pytest`,
   `uv run python -m pre_commit ...`. `uv run ruff` funktioniert direkt.
5. In VS Code den Kernel `.venv` wählen und `notebooks/00_environment_check.ipynb` ausführen.

## Setup mit conda

```bash
git clone <repo-url>
cd BIDA-ML-Starter
conda env create -f environment.yml
conda activate bida-ml
cp .env.example .env
python -m pre_commit install
```

`environment.yml` installiert über pip exakt die Versionen aus `requirements.txt`, die aus `uv.lock` exportiert
wird. uv- und conda-Umgebungen sind damit identisch.

GPU mit conda:
```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

## GPU

- Die neuen Lenovo-Notebooks mit NVIDIA-GPU brauchen einen aktuellen NVIDIA-Treiber (>= 580 für CUDA 13.0).
  Bei älteren Treibern in `pyproject.toml` den Index auf `cu126` umstellen und `uv lock` ausführen.
- Prüfen: `00_environment_check.ipynb` zeigt `CUDA: True` und den GPU-Namen.
- NeuralForecast (N-HiTS, TFT, ...) nutzt die GPU automatisch. LightGBM und XGBoost laufen auf CPU (für
  tabellarische Daten in unserer Grössenordnung ausreichend).

## Snowflake-Zugang

Credentials stehen in `.env` (nicht in Git). Vorlage: `.env.example`.

```
SF_ACCOUNT=pistor.eu-central-1
SF_USER=dein_user
ML_ENV=DEV
SF_PRIVATE_KEY_PATH=~/.snowflake/rsa_key.p8
```

Rolle (`ML_DEVELOPER`), Warehouse (`CONSUMER`), Datenbank (`{env}_ML`) und Schema kommen aus
`configs/config.yaml`; `SF_ROLE`, `SF_WAREHOUSE`, `SF_SCHEMA` in `.env` sind nur
für Ausnahmen. Die Datenbank ist bewusst nicht überschreibbar, `ML_ENV` bleibt der einzige Schalter.

Empfohlen ist Key-Pair-Authentifizierung (kein MFA-Prompt bei jeder Verbindung):

```bash
mkdir ~/.snowflake
openssl genrsa 2048 | openssl pkcs8 -topk8 -nocrypt -out ~/.snowflake/rsa_key.p8
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout -out ~/.snowflake/rsa_key.pub
```

Den Public Key (Inhalt von `rsa_key.pub` ohne BEGIN/END-Zeilen) an den Snowflake-Admin geben:
`ALTER USER dein_user SET RSA_PUBLIC_KEY='...';`. Alternativen: `SF_AUTHENTICATOR=externalbrowser` (SSO)
oder `SF_PASSWORD` (mit MFA).

## Snowflake-Struktur

| Bereich | Objekt | Zweck |
|---------|--------|-------|
| Quelle (optional, lesen) | z.B. `PROD_DATALAKE.<QUELLE>.PSA_*`, `*_90` Views | Häufigster Fall: historisierte Rohdaten, rollierendes 90-Tage-Fenster. Andere Quellen (Dateien, andere Systeme) sind genauso möglich |
| Ziel | `PROD_ML.INFERENCE` | Forecasts und Predictions je Lauf (z.B. `FORECAST_SNAPSHOT`) |
| Ziel | `PROD_ML.REGISTRY` | Modell-Läufe, Parameter, Deployment |
| Ziel | `PROD_ML.MONITORING` | Metriken, DQ-Checks, Forecast vs. Actual |

Aus den Notebooks wird ausschliesslich in die ML-Datenbank geschrieben. Den Rückfluss ins DWH und nach
Power BI übernimmt das DWH-Team.

Umgebungen: `ML_ENV` in `.env` (`DEV`, `INT` oder `PROD`) ist der einzige Schalter. `configs/config.yaml`
verwendet Vorlagen wie `"{env}_ML"` und `"{env}_DATALAKE.MSACCESS.PSA_..._90"`, die beim Laden aufgelöst
werden. Auf Laptops steht `ML_ENV=DEV`; bis zum Deployment existiert ohnehin nur `DEV_ML`. Code, Config
und Skills bleiben in allen Umgebungen identisch.

## Arbeiten mit dem Template

1. Datenquelle festlegen. Bei Snowflake: Tabelle oder View in `configs/config.yaml` unter `tables.training_data`
   eintragen, voll qualifiziert und mit
   `{env}` statt `DEV_`/`PROD_`, z.B. `"{env}_DATALAKE.MSACCESS.PSA_CSV_LOGISTIK_RUESTMENGEN_90"`.
   Bei Dateien: nach `data/raw/` legen (nicht in Git) und mit pandas laden.
2. Eigenes Arbeits-Notebook anlegen, z.B. `notebooks/work_<projekt>.ipynb`.
3. Passendes Template-Notebook öffnen (`01_` bis `04_`) und Schritt für Schritt durchgehen.
4. Für jeden Schritt Claude Code nach dem Code fragen, z.B. "Schritt 8 aus 02_forecasting für meine Daten".
   Claude nutzt dafür automatisch die Skills in `.claude/skills/` (forecasting, classification, regression).
5. Ergebnisse mit `write_to_snowflake()` nach `PROD_ML` schreiben.

| Notebook | Use Case | Inhalt |
|----------|----------|--------|
| `00_environment_check` | Smoke-Test | Libraries, GPU und Snowflake-Verbindung prüfen (mit Code) |
| `01_data_exploration` | EDA | Datenqualität, Verteilungen, Muster (8 Schritte) |
| `02_forecasting` | Zeitreihen | Baseline, statistische Modelle, ML, Neural (20 Schritte) |
| `03_classification` | Klassifikation | LightGBM, XGBoost, sklearn Pipelines (11 Schritte) |
| `04_regression` | Regression | LightGBM, XGBoost, Optuna (16 Schritte) |

Die Template-Notebooks enthalten nur Text (Anleitung), keinen Code. Einzige Ausnahme ist `00_environment_check`.

## Projektstruktur

```
BIDA-ML-Starter/
├── CLAUDE.md                 # Instruktionen für Claude Code
├── README.md                 # Diese Datei
├── pyproject.toml, uv.lock   # Dependencies (uv); Extras cpu / gpu
├── environment.yml, requirements.txt  # Dependencies (conda, aus uv.lock exportiert)
├── .python-version           # 3.12
├── .pre-commit-config.yaml   # ruff + nbstripout
├── .env.example              # Snowflake Credentials Vorlage (.env liegt daneben, nicht in Git)
├── .claude/
│   ├── settings.json         # Permissions und Hooks (Team, committet)
│   ├── skills/               # forecasting, classification, regression (Code-Referenz)
│   ├── rules/                # security (immer), notebooks, python (pfadbezogen)
│   ├── hooks/                # protect-files.sh: blockt Credential-Zugriff und destruktives SQL
│   └── agents/               # code-reviewer, security-reviewer (read-only)
├── configs/config.yaml       # Zentrale Konfiguration ({env}-Vorlagen, Quelltabellen)
├── sql/                      # Berechtigungen der Rolle ML_DEVELOPER (Referenz für Admins)
├── notebooks/                # Template-Notebooks (Anleitungen), 00 = Smoke-Test
├── src/                      # config.py + data_loader.py
├── tests/                    # Unit Tests
├── docs/                     # Zusätzliche Dokumentation
├── data/                     # Lokale Daten (nicht in Git)
├── models/                   # Trainierte Modelle (nicht in Git)
└── reports/figures/          # Plots, Metriken
```

## Verwendung im Agentic Engineering Starter Template

Das [Agentic-Engineering-Starter-Template](https://github.com/themanbetterknownasthemachine/Agentic-Engineering-Starter-Template)
ist das Projekt-Skelett für jedes Claude-Code-Projekt (Spec, Verifier, Hooks, Security-Rules).
BIDA-ML-Starter ist die ML-Methoden-Bibliothek dazu. Bei einem neuen ML-Projekt kopiert
`scripts/new-ml-project.sh` aus diesem Repo ins Projekt:

| Datei in BIDA-ML-Starter | Ziel im Projekt | Zweck |
|--------------------------|-----------------|-------|
| `.claude/skills/<name>/SKILL.md` | `.claude/skills/<name>/SKILL.md` | Methoden-Skill (Frontmatter bereits enthalten) |
| `notebooks/0X_<name>.ipynb` | `notebooks/` | Anleitungs-Notebook, gehört zum Skill (immer als Paar kopieren) |
| `src/config.py`, `src/data_loader.py` | `src/` | Config und Snowflake-Verbindung |
| `configs/config.yaml` | `configs/` | Ziel-Schemas in PROD_ML, Quelltabelle |

Diese Pfade sind die Schnittstelle zum Template und bleiben stabil. Alles andere in diesem Repo
(`.claude/settings.json`, `.claude/rules/`, `environment.yml`, README) dient nur der Standalone-Nutzung
und wird nicht kopiert; Hooks, Security-Rules und Verifier kommen aus dem Template.

## Qualität und Reproduzierbarkeit

- `uv run python -m pytest` führt die Tests aus, `uv run ruff check --fix src tests` und `uv run ruff format src tests` formatieren.
- pre-commit entfernt Notebook-Outputs (nbstripout) und prüft Python-Code (ruff) vor jedem Commit.
- Dependencies nur in `pyproject.toml` ändern, danach `uv lock` und
  `uv export --no-hashes --group dev -o requirements.txt` (hält conda synchron). Beides committen.
- `random_state=42` als Standard, Konfiguration in `configs/config.yaml`, keine hardcodierten Pfade oder Tabellennamen.

## Konventionen

- Notebooks: Deutsch (Markdown). Python-Code, SQL und Commits: Englisch.
- Keine Emojis im Projekt. Deutsche Texte mit ä, ö, ü, ohne Eszett und ohne Gedankenstriche.

## Team

Erstellt von Toni Bühlmann, Pistor BI & Data Analytics
