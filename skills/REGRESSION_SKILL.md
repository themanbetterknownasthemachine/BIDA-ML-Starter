# REGRESSION_SKILL.md

Du hilfst bei Regressions-Projekten im BIDA ML Starter Projekt. Der User folgt der Anleitung in `notebooks/04_regression.ipynb` (16 Schritte) und braucht Code für einzelne Schritte.

## Kontext

- **Umgebung:** Snowflake Container Runtime, VM oder lokaler Laptop
- **src/ enthält nur:** `config.py` (Config laden) und `data_loader.py` (Snowflake-Verbindung)
- **Alles andere** wird direkt im Notebook geschrieben

## Pflicht-Zellen in Snowflake Notebooks

**SQL-Zelle:**
```sql
USE DATABASE ML_DB;
USE SCHEMA DATA;
USE WAREHOUSE ML_WH;
```

**Python-Zelle:**
```python
import sys
from pathlib import Path
for _p in Path("/filesystem").rglob("BIDA-ML-Starter"):
    if (_p / "src").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
```

## Code-Referenz nach Schritt

### Schritt 1: Imports & Konfiguration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from src.config import load_config, RANDOM_STATE
from src.data_loader import load_query

cfg = load_config()

# ANPASSEN:
# df = load_query("SELECT * FROM ML_DB.DATA.DEIN_DATENSATZ")
# TARGET = 'deine_zielvariable'
```

### Schritt 5: Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
```

### Schritt 6: Outlier-Analyse Zielvariable
```python
skewness = y_train.skew()
print(f"Schiefe: {skewness:.3f}")

# IQR-Methode
Q1, Q3 = y_train.quantile(0.25), y_train.quantile(0.75)
IQR = Q3 - Q1
outliers = ((y_train < Q1 - 1.5*IQR) | (y_train > Q3 + 1.5*IQR)).sum()
print(f"Outlier: {outliers} von {len(y_train)} ({outliers/len(y_train)*100:.1f}%)")

# Transformationen vergleichen (bei Schiefe > 0.5)
if abs(skewness) > 0.5:
    pt = PowerTransformer(method='yeo-johnson')
    y_yj = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    print(f"Yeo-Johnson Schiefe: {pd.Series(y_yj).skew():.3f}")
```

### Schritt 7: Pipeline mit Preprocessing
```python
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_pipeline, numeric_features)],
    remainder='drop'
)

if categorical_features:
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features),
    ], remainder='drop')

pipe_lgb = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1))
])
```

### Schritt 8: Baseline
```python
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)

mae_baseline = mean_absolute_error(y_test, pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, pred_baseline))
r2_baseline = r2_score(y_test, pred_baseline)
print(f"Baseline MAE:  {mae_baseline:.4f}")
print(f"Baseline RMSE: {rmse_baseline:.4f}")
print(f"Baseline R²:   {r2_baseline:.4f}")
```

### Schritt 9: Cross-Validation
```python
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
cv_results = cross_validate(pipe_lgb, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

print(f"MAE:  {-cv_results['test_neg_mean_absolute_error'].mean():.4f} ± {cv_results['test_neg_mean_absolute_error'].std():.4f}")
print(f"RMSE: {-cv_results['test_neg_root_mean_squared_error'].mean():.4f}")
print(f"R²:   {cv_results['test_r2'].mean():.4f} ± {cv_results['test_r2'].std():.4f}")
```

### Schritt 10: Hyperparameter-Tuning (Optuna)
```python
import optuna
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'regressor__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'regressor__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'regressor__max_depth': trial.suggest_int('max_depth', 3, 12),
        'regressor__num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'regressor__min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'regressor__reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'regressor__reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    pipe_lgb.set_params(**params)
    score = cross_val_score(pipe_lgb, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    return -score.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"Bester MAE: {study.best_value:.4f}")
```

### Schritt 12: Evaluation
```python
pred_test = best_model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, pred_test, alpha=0.3, s=15, color='#2997ff')
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
ax.plot(lims, lims, 'k--', linewidth=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs. Predicted')
```

### Schritt 13: Residuen-Analyse
```python
from scipy import stats

residuals = y_test - pred_test

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].scatter(pred_test, residuals, alpha=0.3, s=15)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title('Residuen vs. Predicted')

axes[1].hist(residuals, bins=40, color='#2997ff', alpha=0.8, edgecolor='white')
axes[1].set_title('Verteilung Residuen')

stats.probplot(residuals, dist='norm', plot=axes[2])
axes[2].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

print(f"Residuen Mean: {residuals.mean():.4f} (sollte ~0 sein)")
print(f"Residuen Std:  {residuals.std():.4f}")
print(f"Schiefe:       {residuals.skew():.3f}")
```

### Schritt 14: Feature Importance
```python
regressor = best_model.named_steps['regressor']
try:
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out().tolist()
except:
    feature_names = feature_cols

importance = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(ascending=False)
importance.head(20).plot(kind='barh', figsize=(8, 6), title='Feature Importance')

# SHAP (optional)
import shap
X_transformed = best_model.named_steps['preprocessor'].transform(X_test)
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_transformed)
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
```

## Regeln

- Random State immer 42
- Pipeline nutzen (Preprocessing + Modell zusammen), nie getrennt fitten
- Residuen-Analyse ist Pflicht: Muster in Fehlern erkennen
- Outlier-Analyse der Zielvariable VOR dem Training durchführen
- Actual vs. Predicted Plot immer zeigen
- Code direkt im Notebook schreiben
- Sprache: Notebook-Text auf Deutsch, Code auf Englisch
