# FORECASTING_SKILL.md

Du hilfst beim Zeitreihen-Forecasting im BIDA ML Starter Projekt. Der User folgt der Anleitung in `notebooks/02_forecasting.ipynb` (20 Schritte) und braucht Code für einzelne Schritte.

## Kontext

- **Umgebung:** Snowflake Container Runtime, VM oder lokaler Laptop
- **Daten:** Zeitreihen aus Snowflake (Tabelle in `configs/config.yaml`)
- **Ansatz:** Von einfach zu komplex. Baseline ist Pflicht.
- **src/ enthält nur:** `config.py` (Config laden) und `data_loader.py` (Snowflake-Verbindung)
- **Alles andere** (Features, Evaluation, Plots, Modeling) wird direkt im Notebook geschrieben

## Verfügbare Modelle

### Statistische Modelle
| Modell | Library | Exogene Features |
|--------|---------|-----------------|
| SARIMAX | statsmodels | Ja |
| AutoARIMA | StatsForecast (Nixtla) | Ja |
| AutoETS | StatsForecast | Nein |
| AutoTheta | StatsForecast | Nein |
| CES | StatsForecast | Nein |
| OLS Regression | sklearn | Ja |

### Klassische ML
| Modell | Library | Exogene Features |
|--------|---------|-----------------|
| LightGBM | MLForecast (Nixtla) oder direkt | Ja (als Features) |
| XGBoost | MLForecast oder direkt | Ja (als Features) |

### Neurale Modelle
| Modell | Library | Exogene Features (FUTR_EXOG) |
|--------|---------|------------------------------|
| N-HiTS | NeuralForecast (Nixtla) | Ja |
| N-BEATS | NeuralForecast | Nein (rein univariat) |
| PatchTST | NeuralForecast | Nein |
| TFT | NeuralForecast | Ja |
| TSMixerx | NeuralForecast | Ja |
| TiDE | NeuralForecast | Ja |
| NeuralProphet | neuralprophet | Ja |

## Pflicht-Zellen in Snowflake Notebooks

**SQL-Zelle (erste Zelle):**
```sql
USE DATABASE ML_DB;
USE SCHEMA DATA;
USE WAREHOUSE ML_WH;
```

**Python-Zelle (zweite Zelle):**
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

from src.config import load_config, RANDOM_STATE
from src.data_loader import load_timeseries, load_query

cfg = load_config()

# Horizont und Input Size direkt definieren (oder in config.yaml ergänzen)
HORIZON = 14
INPUT_SIZE = 28
```

### Schritt 2: Daten laden & bereinigen
```python
df = load_timeseries()
df['ds'] = pd.to_datetime(df['ds'])

# Bereinigung (anpassen an Use Case)
# df = df[df['ds'].dt.dayofweek < 5]  # nur Mo-Fr
# df = df[df['y'] > 0]                # Nullwerte entfernen

print(f"Shape: {df.shape}")
print(f"Zeitraum: {df['ds'].min()} bis {df['ds'].max()}")
print(f"Serien: {df['unique_id'].nunique()}")
```

### Schritt 5: STL Decomposition
```python
from statsmodels.tsa.seasonal import STL
series = df[df['unique_id'] == SERIE].set_index('ds')['y']
stl = STL(series, period=5)  # 5=Geschäftswoche, 7=Kalenderwoche
result = stl.fit()
result.plot()
plt.tight_layout()
plt.show()
```

### Schritt 8: Baseline
```python
# Seasonal Naive: gleicher Wochentag letzte Woche
baseline_pred = test['y_lag_7'].values

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_baseline = mean_absolute_error(y_test, baseline_pred)
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"Baseline MAE:  {mae_baseline:.2f}")
print(f"Baseline RMSE: {rmse_baseline:.2f}")
```

### Schritt 9: Statistische Modelle
```python
# --- SARIMAX ---
from statsmodels.tsa.statespace.sarimax import SARIMAX
model_sarima = SARIMAX(train_y, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,5))
result_sarima = model_sarima.fit(disp=False)
pred_sarima = result_sarima.forecast(steps=len(test), exog=test_exog)

# --- StatsForecast ---
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
sf = StatsForecast(
    models=[AutoARIMA(season_length=5), AutoETS(season_length=5), AutoTheta(season_length=5)],
    freq='B', n_jobs=-1
)
sf.fit(df=train_sf)
pred_stats = sf.predict(h=HORIZON)

# --- OLS ---
from sklearn.linear_model import LinearRegression
model_ols = LinearRegression()
model_ols.fit(X_train, y_train)
pred_ols = model_ols.predict(X_test)
```

### Schritt 10: Klassische ML
```python
# --- LightGBM via MLForecast ---
from mlforecast import MLForecast
import lightgbm as lgb

mlf = MLForecast(
    models=[lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbosity=-1)],
    freq='B',
    lags=[1, 2, 3, 5, 10, 20],
)
mlf.fit(df=train_mlf)
pred_lgb = mlf.predict(h=HORIZON)
```

### Schritt 11: Neurale Modelle
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, PatchTST, TFT, TSMixerx, TiDE

# Modelle mit exogenen Features
models_exog = [
    NHITS(h=HORIZON, input_size=INPUT_SIZE, max_steps=500, futr_exog_list=exog_cols),
    TFT(h=HORIZON, input_size=INPUT_SIZE, max_steps=500, futr_exog_list=exog_cols),
    TSMixerx(h=HORIZON, input_size=INPUT_SIZE, max_steps=500, futr_exog_list=exog_cols, n_series=1),
]

# Modelle ohne exogene Features
models_univar = [
    NBEATS(h=HORIZON, input_size=INPUT_SIZE, max_steps=500),
    PatchTST(h=HORIZON, input_size=INPUT_SIZE, max_steps=500),
]

nf = NeuralForecast(models=models_exog, freq='B')
nf.fit(df=train_nf)
pred_nf = nf.predict(futr_df=future_exog_df)
```

### Schritt 13: Modellvergleich
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calc_metrics(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'R2': round(r2_score(y_true, y_pred), 4),
    }

results = {
    'Baseline': calc_metrics(y_test, baseline_pred),
    'SARIMAX': calc_metrics(y_test, pred_sarima),
    'AutoARIMA': calc_metrics(y_test, pred_autoarima),
    'LightGBM': calc_metrics(y_test, pred_lgb),
    'N-HiTS': calc_metrics(y_test, pred_nhits),
}
comparison = pd.DataFrame(results).T.sort_values('MAE')
print(comparison)
```

### Schritt 16: Hyperparameter-Tuning (Beispiel N-HiTS)
```python
import optuna

def objective(trial):
    config = {
        'h': HORIZON,
        'input_size': trial.suggest_int('input_size', 60, 180),
        'max_steps': trial.suggest_int('max_steps', 300, 1000),
        'learning_rate': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'n_pool_kernel_size': [trial.suggest_categorical(f'pool_{i}', [2, 4, 8]) for i in range(3)],
        'scaler_type': trial.suggest_categorical('scaler', ['standard', 'robust', 'minmax']),
        'futr_exog_list': exog_cols,
        'random_seed': 42,
    }
    model = NHITS(**config)
    nf = NeuralForecast(models=[model], freq='B')
    cv = nf.cross_validation(df=train_nf, n_windows=3, step_size=HORIZON)
    return abs(cv['y'] - cv['NHITS']).mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## Regeln

- Zeitreihen IMMER zeitlich splitten, nie zufällig
- Baseline ist PFLICHT. Jedes weitere Modell muss die Baseline schlagen.
- Von einfach zu komplex: Statistisch → ML → Neural
- Exogene Features: Nur verwenden was in der Zukunft bekannt ist
- Dokumentieren welche Modelle mit/ohne exogene Features laufen
- Random State immer 42
- Code direkt im Notebook schreiben, nicht in src/ Module auslagern
- Sprache: Notebook-Text auf Deutsch, Code auf Englisch
- Der User entscheidet welche Modelle er testen will
