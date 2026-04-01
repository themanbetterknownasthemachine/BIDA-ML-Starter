# 🧠 BIDA ML Starter Projekt — Pistor AG

Standardisiertes ML-Template für das Pistor AG BI & Data Analytics Team.
Einsetzbar für Forecasting, Klassifikation, Regression und weitere ML-Aufgaben.

## 🌐 Ein Projekt, drei Umgebungen

Dieses Projekt läuft ohne Anpassungen in drei Umgebungen. Derselbe Code, dieselben Module, dieselbe Config.

| | Snowflake Container Runtime | VM (Remote-SSH) | Lokaler Laptop |
|---|---|---|---|
| **Setup** | `pip install -r requirements-snowflake.txt` | `uv sync` | `uv sync` |
| **Snowflake-Verbindung** | Automatisch (`get_active_session()`) | Via `.env` Datei | Via `.env` Datei |
| **AI-Assistent** | Cortex Code (Claude) | Claude Code | Claude Code |
| **GPU** | Compute Pool wechseln | GPU-VM bestellen | Nein |
| **Daten bleiben in Snowflake** | Ja | Nein | Nein |
| **24/7 Automatisierung** | ML Jobs + Tasks | Cron / Airflow | Nein |
| **Was brauchst du?** | Browser + Snowflake Account | VS Code + SSH | VS Code + Python/uv |
| **Wann nutzen?** | Produktion, Governance, GPU | Entwicklung, Refactoring | Offline, schnelle Tests |

Der Trick: `src/data_loader.py` erkennt automatisch, wo es läuft. In Snowflake nutzt es die eingebaute Session, lokal liest es die Credentials aus `.env`. Der restliche Code (Features, Training, Evaluation) ist identisch.

**Was nur in bestimmten Umgebungen relevant ist:**

| Datei | Snowflake | VM / Lokal |
|---|---|---|
| `requirements-snowflake.txt` | ✅ Verwenden | Ignorieren |
| `pyproject.toml` + `uv.lock` | Ignorieren | ✅ Verwenden |
| `sql/` Ordner | ✅ Einmalig ausführen | Ignorieren (Daten bereits in SF) |
| `.env.example` | Nicht nötig | ✅ Kopieren und ausfüllen |

## 🏗️ Projektstruktur

```
BIDA-ML-Starter/
├── CLAUDE.md                      # Instruktionen für Claude / Cortex Code
├── README.md                      # Diese Datei
├── pyproject.toml                 # Dependencies (VM/lokal via uv)
├── requirements-snowflake.txt     # Dependencies (Snowflake Container Runtime)
├── configs/config.yaml            # Zentrale Konfiguration
├── skills/                        # Coding-Skills für Claude/Cortex Code
├── sql/                           # Snowflake Setup & Baseline SQL
├── notebooks/                     # Jupyter Notebooks (Schritt-für-Schritt Anleitungen)
├── src/                           # config.py + data_loader.py (Minimum)
├── tests/                         # Unit Tests
├── data/                          # Lokale Daten (nicht in Git)
├── models/                        # Trainierte Modelle
└── reports/                       # Metriken, Plots, Reports
```

## ⚡ Quickstart

### Option A: Snowflake Container Runtime

1. Admin führt `sql/01_admin_setup.sql` in Snowflake aus
2. Workspace erstellen → Notebook anlegen (Container Runtime)
3. Im Terminal:
   ```bash
   pip install -r requirements-snowflake.txt
   ```
4. Notebook `00_environment_check.ipynb` ausführen
5. Loslegen mit dem passenden Template-Notebook

### Option B: VM (Remote-SSH mit VS Code)

1. IT stellt eine Ubuntu-VM bereit (SSH-Zugang, PyPI-Zugriff)
2. VS Code mit Remote-SSH Extension verbinden
3. Repository klonen und Environment erstellen:
   ```bash
   git clone <repo-url>
   cd BIDA-ML-Starter
   uv sync
   ```
4. `.env` anlegen:
   ```bash
   cp .env.example .env
   # Snowflake-Credentials eintragen
   ```
5. Smoke-Test: `00_environment_check.ipynb` in VS Code ausführen

### Option C: Lokaler Laptop (wenn IT pip/uv erlaubt)

1. Repository klonen:
   ```bash
   git clone <repo-url>
   cd BIDA-ML-Starter
   ```
2. Environment erstellen:
   ```bash
   uv sync
   ```
3. `.env` anlegen und Credentials eintragen
4. Smoke-Test in VS Code oder JupyterLab ausführen

## 📊 Verfügbare Templates

| Notebook | Use Case | Modelle / Ansatz |
|----------|----------|-----------------|
| `00_environment_check` | Smoke-Test | Libraries + Snowflake prüfen (mit Code) |
| `01_data_exploration` | EDA | Datenqualität, Verteilungen, Muster (8 Schritte) |
| `02_forecasting` | Zeitreihen | Statistisch → ML → Neural (volles Spektrum, 20 Schritte) |
| `03_classification` | Klassifikation | LightGBM, XGBoost, sklearn Pipelines (11 Schritte) |
| `04_regression` | Regression | LightGBM, XGBoost, Optuna (16 Schritte) |

Jedes Template-Notebook enthält eine Schritt-für-Schritt Anleitung (nur Text, kein Code). Der Code wird mit Hilfe von Claude oder Cortex Code erstellt, basierend auf den Skill-Dateien im `skills/` Ordner. Einzige Ausnahme: `00_environment_check` enthält fertigen Code für den Smoke-Test.

## 🔐 Credentials

### Snowflake Container Runtime
Kein Setup nötig — `get_active_session()` verbindet automatisch.

### VM / Lokal
Credentials in `.env` (nicht in Git):
```
SF_ACCOUNT=pistor.eu-central-1
SF_USER=dein_user
SF_ROLE=ROLE_DS
SF_WAREHOUSE=WH_ML
SF_DATABASE=ML_DB
SF_SCHEMA=DATA
```

## 🔁 Reproduzierbarkeit

- **Snowflake:** `pip install -r requirements-snowflake.txt` (Versionen gepinnt)
- **VM/lokal:** `uv sync` (Lockfile für exakte Reproduktion)
- **Config:** Alle Parameter in `configs/config.yaml` (keine Hardcodes)
- **Seeds:** `random_state=42` als Standard

## 📝 Konventionen

- **Notebooks:** Deutsch (Markdown, Kommentare)
- **Python-Code:** Englisch (Funktionen, Variablen, Docstrings)
- **SQL:** Englisch
- **Commits:** Englisch

## 👥 Team

Erstellt von **Toni Bühlmann** — Pistor BI & Data Analytics
