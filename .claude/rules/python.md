---
paths:
  - "src/**"
  - "tests/**"
---

# Regeln fuer Python-Code in src/ und tests/

- `src/` bleibt minimal: `config.py` (Config laden) und `data_loader.py` (Snowflake-Verbindung).
  Neue Module nur, wenn eine Funktion in mehreren Notebooks wiederverwendet wird.
- Code, Docstrings und Kommentare auf Englisch. Type Hints verwenden.
- Formatierung und Linting mit ruff (`uv run ruff format`, `uv run ruff check --fix`), Zeilenlaenge 100.
- Keine hardcodierten Tabellennamen, Schemas oder Pfade: alles aus `configs/config.yaml` bzw. `.env`.
- Keine Credentials im Code. `.env` und Private Keys nie lesen oder ausgeben.
- Jede Aenderung an `src/` braucht einen Test in `tests/` (`uv run python -m pytest`; Launcher-EXEs wie
  `pytest.exe` sind auf den Firmen-Notebooks blockiert, deshalb immer `python -m`).
