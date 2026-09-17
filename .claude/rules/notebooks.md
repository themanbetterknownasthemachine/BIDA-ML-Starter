---
paths:
  - "notebooks/**"
---

# Regeln für Notebooks

- `01_` bis `04_` sind reine Text-Anleitungen (Markdown, Deutsch) ohne Code-Zellen. Sie bleiben so.
- Nur `00_environment_check.ipynb` enthält Code (Smoke-Test).
- Code für einen Schritt wird auf Anfrage generiert und vom User in ein eigenes Arbeits-Notebook
  übernommen (z.B. `notebooks/work_ruestmengen.ipynb`). Template-Notebooks nicht mit Code füllen.
- Markdown-Zellen auf Deutsch, Code-Zellen auf Englisch.
- Notebook-Outputs werden vor dem Commit entfernt (nbstripout via pre-commit).
