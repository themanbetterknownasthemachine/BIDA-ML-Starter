# CLASSIFICATION_SKILL.md

Du hilfst bei Klassifikations-Projekten im BIDA ML Starter Projekt. Der User folgt der Anleitung in `notebooks/03_classification.ipynb` (11 Schritte) und braucht Code für einzelne Schritte.

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

### Schritt 1: Imports & Daten laden
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from src.config import load_config, RANDOM_STATE
from src.data_loader import load_query

cfg = load_config()

# ANPASSEN:
# df = load_query("SELECT * FROM ML_DB.DATA.DEIN_DATENSATZ")
# TARGET = 'deine_zielvariable'
```

### Schritt 3: Feature Engineering & Vorbereitung
```python
feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols]
y = df[TARGET]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Stratified Split!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

### Schritt 4: Pipeline mit Preprocessing
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
    ('classifier', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1))
])

# Baseline
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"Baseline Accuracy: {baseline_acc:.4f}")
```

### Schritt 5: Cross-Validation
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ['accuracy', 'f1_weighted', 'roc_auc']
cv_results = cross_validate(pipe_lgb, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"F1:       {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
print(f"ROC-AUC:  {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
```

### Schritt 6: Hyperparameter-Tuning
```python
param_grid = {
    'classifier__n_estimators': [300, 500, 800],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7, -1],
    'classifier__num_leaves': [15, 31, 63],
}
grid_search = GridSearchCV(pipe_lgb, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)
print(f"Beste Parameter: {grid_search.best_params_}")
print(f"Bester CV F1:    {grid_search.best_score_:.4f}")
```

### Schritt 8: Evaluation
```python
best_model = grid_search.best_estimator_
pred_test = best_model.predict(X_test)
pred_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred_test))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, pred_test), annot=True, fmt='d', cmap='Blues', ax=ax)

# ROC + PR Kurve
fpr, tpr, _ = roc_curve(y_test, pred_proba)
auc_score = roc_auc_score(y_test, pred_proba)
```

### Schritt 9: Feature Importance
```python
classifier = best_model.named_steps['classifier']
try:
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out().tolist()
except:
    feature_names = feature_cols

importance = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)
importance.head(20).plot(kind='barh', figsize=(8, 6), title='Feature Importance')

# SHAP (optional)
import shap
X_transformed = best_model.named_steps['preprocessor'].transform(X_test)
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_transformed)
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
```

## Regeln

- Bei Klassifikation IMMER stratified splitten (stratify=y)
- Klassenbalance prüfen, bei Imbalance SMOTE oder class_weight='balanced' vorschlagen
- Pipeline nutzen (Preprocessing + Modell zusammen), nie getrennt fitten
- ROC-AUC und PR-AUC immer beide zeigen
- Random State immer 42
- Code direkt im Notebook schreiben
- Sprache: Notebook-Text auf Deutsch, Code auf Englisch
