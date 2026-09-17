---
name: regression
description: Code für Regression im BIDA ML Starter (sklearn Pipelines, LightGBM/XGBoost, Optuna, Residuen-Analyse, SHAP, Schreiben nach PROD_ML). Nutzen, wenn der User einen Schritt aus notebooks/04_regression.ipynb umsetzt.
---

# Regression Skill

Der User folgt der Anleitung in `notebooks/04_regression.ipynb` (16 Schritte) und braucht Code
für einzelne Schritte. Code wird in ein eigenes Arbeits-Notebook des Users geschrieben.

## Kontext

- **Daten:** Quelle ist frei. Snowflake-Tabelle oder View (voll qualifizierter Name in `configs/config.yaml`,
  laden mit `load_table`) oder Datei aus `data/raw/` (`pd.read_csv`, `pd.read_excel`, `pd.read_parquet`).
- **Ziel:** Predictions nach `PROD_ML.INFERENCE`, Metriken nach `PROD_ML.MONITORING`.
- **src/ enthält nur** `config.py` und `data_loader.py`; alles andere wird im Notebook geschrieben.

## Code-Referenz nach Schritt

### Schritt 1: Imports & Konfiguration
```python
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from src.config import RANDOM_STATE, load_config
from src.data_loader import load_query, load_table, write_to_snowflake

cfg = load_config()
df = load_table(cfg["tables"]["training_data"])
df.columns = [c.lower() for c in df.columns]
TARGET = "zielvariable"  # ANPASSEN
```

### Schritt 3/4: EDA & Features
```python
print(df.describe().T)
print(df.isna().mean().sort_values(ascending=False).head(10))
print(df.select_dtypes("number").corr()[TARGET].sort_values(ascending=False).head(10))

feature_cols = [c for c in df.columns if c != TARGET]
X, y = df[feature_cols], df[TARGET]
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
```

### Schritt 5: Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(
    f"Train {len(X_train)}  Test {len(X_test)}   y mean train {y_train.mean():.2f} / test {y_test.mean():.2f}"
)
```

### Schritt 6: Outlier-Analyse Zielvariable
```python
skewness = y_train.skew()
q1, q3 = y_train.quantile([0.25, 0.75])
iqr = q3 - q1
outliers = ((y_train < q1 - 1.5 * iqr) | (y_train > q3 + 1.5 * iqr)).sum()
print(
    f"Schiefe: {skewness:.3f}   Outlier: {outliers} von {len(y_train)} ({outliers / len(y_train) * 100:.1f}%)"
)

if abs(skewness) > 0.5:
    pt = PowerTransformer(method="yeo-johnson")
    y_yj = pt.fit_transform(y_train.to_frame()).ravel()
    print(f"Schiefe nach Yeo-Johnson: {pd.Series(y_yj).skew():.3f}")
    # Bei starker Schiefe: TransformedTargetRegressor(regressor=..., transformer=PowerTransformer()) nutzen
```

### Schritt 7: Pipeline mit Preprocessing
```python
numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features),
    ]
)

pipe_lgb = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1
            ),
        ),
    ]
)
```

### Schritt 8: Baseline
```python
def calc_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred),
    }


baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
results = {"Baseline (mean)": calc_metrics(y_test, baseline.predict(X_test))}
print(pd.DataFrame(results).T.round(4))
```

### Schritt 9: Cross-Validation
```python
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
cv_results = cross_validate(
    pipe_lgb, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
)
print(
    f"MAE:  {-cv_results['test_neg_mean_absolute_error'].mean():.4f} +/- {cv_results['test_neg_mean_absolute_error'].std():.4f}"
)
print(f"RMSE: {-cv_results['test_neg_root_mean_squared_error'].mean():.4f}")
print(f"R2:   {cv_results['test_r2'].mean():.4f}  (train {cv_results['train_r2'].mean():.4f})")
```

### Schritt 10: Hyperparameter-Tuning (Optuna)
```python
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    params = {
        "regressor__n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "regressor__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "regressor__max_depth": trial.suggest_int("max_depth", 3, 12),
        "regressor__num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "regressor__min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "regressor__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "regressor__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    pipe_lgb.set_params(**params)
    return -cross_val_score(
        pipe_lgb, X_train, y_train, cv=3, scoring="neg_mean_absolute_error"
    ).mean()


study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"Bester CV-MAE: {study.best_value:.4f}")
print(study.best_params)
```

### Schritt 11/12: Finales Training & Evaluation
```python
best_model = pipe_lgb.set_params(
    **{f"regressor__{k}": v for k, v in study.best_params.items()}
).fit(X_train, y_train)
pred_test = best_model.predict(X_test)
results["LightGBM (tuned)"] = calc_metrics(y_test, pred_test)
comparison = pd.DataFrame(results).T.sort_values("MAE").round(4)
print(comparison)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, pred_test, alpha=0.3, s=15)
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs. Predicted")
plt.tight_layout()
plt.show()
```

### Schritt 13: Residuen-Analyse
```python
residuals = y_test - pred_test

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(pred_test, residuals, alpha=0.3, s=15)
axes[0].axhline(0, color="r", linestyle="--")
axes[0].set_title("Residuen vs. Predicted")
axes[1].hist(residuals, bins=40, edgecolor="white")
axes[1].set_title("Verteilung Residuen")
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot")
plt.tight_layout()
plt.show()

print(
    f"Residuen Mean: {residuals.mean():.4f} (sollte ~0 sein)   Std: {residuals.std():.4f}   Schiefe: {residuals.skew():.3f}"
)
```

### Schritt 14: Feature Importance & SHAP
```python
import shap

regressor = best_model.named_steps["regressor"]
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(
    ascending=False
)
importance.head(20).plot(kind="barh", figsize=(8, 6), title="Feature Importance")

X_transformed = best_model.named_steps["preprocessor"].transform(X_test)
shap_values = shap.TreeExplainer(regressor).shap_values(X_transformed)
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
```

### Schritt 15: Modell speichern & Ergebnisse nach Snowflake
```python
from pathlib import Path
import joblib

MODEL_DIR = Path("models") / f"reg_v1_{pd.Timestamp.today():%Y%m%d}"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_DIR / "pipeline.joblib")

RUN_ID = f"reg_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
predictions = X_test.assign(y_true=y_test.values, y_pred=pred_test, run_id=RUN_ID)
write_to_snowflake(predictions, "REGRESSION_PREDICTIONS")

metrics = (
    comparison.reset_index()
    .rename(columns={"index": "model"})
    .assign(run_id=RUN_ID, created_at=pd.Timestamp.now())
)
write_to_snowflake(
    metrics, "MODEL_EVALUATION_LOG", schema=cfg["snowflake"]["schemas"]["monitoring"]
)
```

## Regeln

- `random_state` immer 42 (`RANDOM_STATE`).
- Pipeline nutzen (Preprocessing + Modell zusammen), nie getrennt fitten.
- Baseline (`DummyRegressor`) ist Pflicht.
- Outlier-Analyse der Zielvariable VOR dem Training.
- Residuen-Analyse und Actual-vs-Predicted Plot immer zeigen.
- Ergebnisse nach PROD_ML schreiben, Code direkt im Arbeits-Notebook.
- Notebook-Text auf Deutsch, Code auf Englisch.
