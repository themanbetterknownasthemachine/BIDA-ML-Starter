---
name: classification
description: Code fuer Klassifikation im BIDA ML Starter (sklearn Pipelines, LightGBM/XGBoost, stratified CV, Tuning, Evaluation mit ROC/PR, SHAP, Schreiben nach PROD_ML). Nutzen, wenn der User einen Schritt aus notebooks/03_classification.ipynb umsetzt.
---

# Classification Skill

Der User folgt der Anleitung in `notebooks/03_classification.ipynb` (11 Schritte) und braucht Code
fuer einzelne Schritte. Code wird in ein eigenes Arbeits-Notebook des Users geschrieben.

## Kontext

- **Daten:** Tabelle oder View aus `PROD_DATALAKE` (voll qualifizierter Name in `configs/config.yaml`).
- **Ziel:** Predictions nach `PROD_ML.INFERENCE`, Metriken nach `PROD_ML.MONITORING`.
- **src/ enthaelt nur** `config.py` und `data_loader.py`; alles andere wird im Notebook geschrieben.

## Code-Referenz nach Schritt

### Schritt 1: Imports & Daten laden
```python
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay, accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from src.config import RANDOM_STATE, load_config
from src.data_loader import load_query, load_table, write_to_snowflake

cfg = load_config()
df = load_table(cfg["tables"]["training_data"])
df.columns = [c.lower() for c in df.columns]
TARGET = "zielvariable"   # ANPASSEN
```

### Schritt 2: EDA
```python
print(df.shape)
print(df.isna().mean().sort_values(ascending=False).head(10))
print(df[TARGET].value_counts(normalize=True))   # Klassenbalance
sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", center=0)
```

### Schritt 3: Feature Engineering & Split
```python
feature_cols = [c for c in df.columns if c != TARGET]
X, y = df[feature_cols], df[TARGET]
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Stratified Split!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

### Schritt 4: Pipeline mit Preprocessing & Baseline
```python
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", cat_pipeline, categorical_features),
])

pipe_lgb = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight="balanced",
                                      random_state=RANDOM_STATE, verbosity=-1)),
])

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pred_base = baseline.predict(X_test)
print(f"Baseline Accuracy: {accuracy_score(y_test, pred_base):.4f}  F1: {f1_score(y_test, pred_base, average='weighted'):.4f}")
```

### Schritt 5: Cross-Validation
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "f1_weighted", "roc_auc", "average_precision"]
cv_results = cross_validate(pipe_lgb, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
for s in scoring:
    print(f"{s:18s} test {cv_results[f'test_{s}'].mean():.4f} +/- {cv_results[f'test_{s}'].std():.4f}"
          f"   train {cv_results[f'train_{s}'].mean():.4f}")
```

### Schritt 6: Hyperparameter-Tuning
```python
param_grid = {
    "classifier__n_estimators": [300, 500, 800],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__num_leaves": [15, 31, 63],
}
grid_search = GridSearchCV(pipe_lgb, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_, round(grid_search.best_score_, 4))
best_model = grid_search.best_estimator_
```

### Schritt 8: Evaluation
```python
pred_test = best_model.predict(X_test)
pred_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred_test))
print(f"ROC-AUC: {roc_auc_score(y_test, pred_proba):.4f}   PR-AUC: {average_precision_score(y_test, pred_proba):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(confusion_matrix(y_test, pred_test), annot=True, fmt="d", cmap="Blues", ax=axes[0]); axes[0].set_title("Confusion Matrix")
RocCurveDisplay.from_predictions(y_test, pred_proba, ax=axes[1])
PrecisionRecallDisplay.from_predictions(y_test, pred_proba, ax=axes[2])
plt.tight_layout(); plt.show()
```

### Schritt 9: Feature Importance & SHAP
```python
import shap

classifier = best_model.named_steps["classifier"]
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)
importance.head(20).plot(kind="barh", figsize=(8, 6), title="Feature Importance")

X_transformed = best_model.named_steps["preprocessor"].transform(X_test)
shap_values = shap.TreeExplainer(classifier).shap_values(X_transformed)
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
```

### Schritt 10: Modell speichern & Ergebnisse nach Snowflake
```python
from pathlib import Path
import joblib

MODEL_DIR = Path("models") / f"clf_v1_{pd.Timestamp.today():%Y%m%d}"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_DIR / "pipeline.joblib")   # ganze Pipeline inkl. Preprocessing

RUN_ID = f"clf_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
predictions = X_test.assign(y_true=y_test.values, y_pred=pred_test, y_proba=pred_proba, run_id=RUN_ID)
write_to_snowflake(predictions, "CLASSIFICATION_PREDICTIONS")

metrics = pd.DataFrame([{"run_id": RUN_ID, "model": "LightGBM", "accuracy": accuracy_score(y_test, pred_test),
                         "f1_weighted": f1_score(y_test, pred_test, average="weighted"),
                         "roc_auc": roc_auc_score(y_test, pred_proba), "pr_auc": average_precision_score(y_test, pred_proba),
                         "created_at": pd.Timestamp.now()}])
write_to_snowflake(metrics, "MODEL_EVALUATION_LOG", schema=cfg["snowflake"]["schemas"]["monitoring"])
```

## Regeln

- IMMER stratified splitten (`stratify=y`) und `StratifiedKFold` nutzen.
- Klassenbalance pruefen; bei Imbalance `class_weight="balanced"` oder SMOTE (`imbalanced-learn`) vorschlagen.
- Pipeline nutzen (Preprocessing + Modell zusammen), nie getrennt fitten.
- ROC-AUC und PR-AUC immer beide zeigen.
- Baseline (`DummyClassifier`) ist Pflicht.
- `random_state` immer 42 (`RANDOM_STATE`).
- Ergebnisse nach PROD_ML schreiben, Code direkt im Arbeits-Notebook.
- Notebook-Text auf Deutsch, Code auf Englisch.
