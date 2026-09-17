---
name: forecasting
description: Code fuer Zeitreihen-Forecasting im BIDA ML Starter (Baseline, StatsForecast, MLForecast/LightGBM, NeuralForecast, Evaluation, Schreiben nach PROD_ML). Nutzen, wenn der User einen Schritt aus notebooks/02_forecasting.ipynb umsetzt oder nach Forecast-Code fragt.
---

# Forecasting Skill

Der User folgt der Anleitung in `notebooks/02_forecasting.ipynb` (20 Schritte) und braucht Code
fuer einzelne Schritte. Code wird in ein eigenes Arbeits-Notebook des Users geschrieben.

## Kontext

- **Daten:** Zeitreihen aus Snowflake im Format `[unique_id, ds, y]` (+ optionale exogene Spalten).
  Quelle steht in `configs/config.yaml` unter `tables.training_data` (View in `PROD_DATALAKE`).
- **Ziel:** Forecasts nach `PROD_ML.INFERENCE`, Laufinfos nach `PROD_ML.REGISTRY`,
  Metriken nach `PROD_ML.MONITORING` (siehe `write_to_snowflake`).
- **Ansatz:** Von einfach zu komplex. Baseline ist Pflicht. Der User entscheidet, welche Modelle er testet.
- **src/ enthaelt nur** `config.py` und `data_loader.py`; alles andere wird im Notebook geschrieben.
- **GPU:** NeuralForecast nutzt automatisch CUDA, wenn `torch.cuda.is_available()`.

## Verfuegbare Modelle

| Familie | Modell | Library | Exogene Features |
|---------|--------|---------|------------------|
| Statistisch | SARIMAX | statsmodels | Ja |
| Statistisch | AutoARIMA | StatsForecast | Ja |
| Statistisch | AutoETS, AutoTheta, AutoCES | StatsForecast | Nein |
| Statistisch | OLS Regression | sklearn | Ja |
| Klassisches ML | LightGBM, XGBoost | MLForecast | Ja (als Features) |
| Neural | N-HiTS, TFT, TSMixerx, TiDE | NeuralForecast | Ja (`futr_exog_list`) |
| Neural | N-BEATS, PatchTST | NeuralForecast | Nein (univariat) |

## Code-Referenz nach Schritt

### Schritt 1: Imports & Konfiguration
```python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from src.config import RANDOM_STATE, load_config
from src.data_loader import load_query, load_timeseries, write_to_snowflake

cfg = load_config()

HORIZON = 14        # Forecast-Horizont in Perioden
INPUT_SIZE = 28     # Lookback fuer neurale Modelle
FREQ = "D"          # "D" Kalendertage, "B" Geschaeftstage
SEASON = 7          # 7 Kalenderwoche, 5 Geschaeftswoche, 12 Monate
```

### Schritt 2: Daten laden & bereinigen
```python
df = load_timeseries()
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

# Bereinigung (an Use Case anpassen)
# df = df[df["ds"].dt.dayofweek < 5]   # nur Mo-Fr
# df = df[df["y"] > 0]                  # Nullwerte entfernen

print(f"Shape: {df.shape}")
print(f"Zeitraum: {df['ds'].min().date()} bis {df['ds'].max().date()}")
print(f"Serien: {df['unique_id'].nunique()}")

# Luecken pruefen
full_idx = df.groupby("unique_id")["ds"].agg(["min", "max", "count"])
full_idx["expected"] = (full_idx["max"] - full_idx["min"]).dt.days + 1
print(full_idx)
```

### Schritt 3: EDA
```python
SERIE = df["unique_id"].iloc[0]
s = df[df["unique_id"] == SERIE]

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
axes[0].plot(s["ds"], s["y"]); axes[0].set_title(f"Zeitreihe {SERIE}")
sns.boxplot(x=s["ds"].dt.dayofweek, y=s["y"], ax=axes[1]); axes[1].set_title("Wochentag")
sns.barplot(x=s["ds"].dt.month, y=s["y"], ax=axes[2], errorbar=None); axes[2].set_title("Monat")
plt.tight_layout()
plt.show()
```

### Schritt 4: Feature Engineering & Feiertags-Kalender
```python
import holidays

ch_holidays = holidays.country_holidays("CH", subdiv="LU", years=range(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 2))

def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and holiday features that are known in the future."""
    out = data.copy()
    out["dow"] = out["ds"].dt.dayofweek
    out["month"] = out["ds"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_holiday"] = out["ds"].isin(pd.to_datetime(list(ch_holidays.keys()))).astype(int)
    out["day_before_holiday"] = out.groupby("unique_id")["is_holiday"].shift(-1).fillna(0).astype(int)
    out["is_month_start"] = out["ds"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["ds"].dt.is_month_end.astype(int)
    return out

df = add_calendar_features(df)
exog_cols = ["dow", "month", "is_weekend", "is_holiday", "day_before_holiday", "is_month_start", "is_month_end"]

# Lag- und Rolling-Features (nur fuer Modelle mit eigener Feature-Tabelle, z.B. OLS)
for lag in [1, SEASON, 2 * SEASON]:
    df[f"y_lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)
df["y_roll_mean_7"] = df.groupby("unique_id")["y"].transform(lambda x: x.shift(1).rolling(7).mean())
```

### Schritt 5: STL Decomposition
```python
from statsmodels.tsa.seasonal import STL

series = df[df["unique_id"] == SERIE].set_index("ds")["y"].asfreq(FREQ).interpolate()
result = STL(series, period=SEASON).fit()
result.plot()
plt.tight_layout()
plt.show()
```

### Schritt 6: Ausreisser
```python
q1, q3 = df["y"].quantile([0.25, 0.75])
iqr = q3 - q1
mask = (df["y"] < q1 - 1.5 * iqr) | (df["y"] > q3 + 1.5 * iqr)
print(f"Ausreisser: {mask.sum()} von {len(df)} ({mask.mean() * 100:.1f}%)")
print(df.loc[mask, ["unique_id", "ds", "y"]].head(20))
```

### Schritt 7: Train/Test Split & Datenformate
```python
# Zeitlicher Split: die letzten HORIZON Perioden pro Serie sind Test
cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
train = df[df["ds"] <= cutoff].copy()
test = df[df["ds"] > cutoff].copy()
print(f"Train: {train['ds'].min().date()} bis {train['ds'].max().date()} ({len(train)} Zeilen)")
print(f"Test:  {test['ds'].min().date()} bis {test['ds'].max().date()} ({len(test)} Zeilen)")

# Nixtla-Format (StatsForecast, MLForecast, NeuralForecast)
train_nf = train[["unique_id", "ds", "y"] + exog_cols]
future_exog = test[["unique_id", "ds"] + exog_cols]     # bekannte Zukunfts-Features
y_test = test.set_index(["unique_id", "ds"])["y"]
```

### Schritt 8: Baseline
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calc_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100,
        "sMAPE": np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100,
    }

# Seasonal Naive: Wert von vor SEASON Perioden
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

sf_base = StatsForecast(models=[SeasonalNaive(season_length=SEASON)], freq=FREQ, n_jobs=-1)
pred_base = sf_base.forecast(df=train_nf[["unique_id", "ds", "y"]], h=HORIZON)
merged = test.merge(pred_base, on=["unique_id", "ds"])
results = {"Baseline (SeasonalNaive)": calc_metrics(merged["y"], merged["SeasonalNaive"])}
print(pd.DataFrame(results).T.round(2))
```

### Schritt 9: Statistische Modelle
```python
from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta

sf = StatsForecast(
    models=[
        AutoARIMA(season_length=SEASON),
        AutoETS(season_length=SEASON),
        AutoTheta(season_length=SEASON),
        AutoCES(season_length=SEASON),
    ],
    freq=FREQ, n_jobs=-1,
)
pred_stats = sf.forecast(df=train_nf[["unique_id", "ds", "y"]], h=HORIZON)
merged = test.merge(pred_stats, on=["unique_id", "ds"])
for m in ["AutoARIMA", "AutoETS", "AutoTheta", "CES"]:
    results[m] = calc_metrics(merged["y"], merged[m])

# SARIMAX mit exogenen Features (eine Serie)
from statsmodels.tsa.statespace.sarimax import SARIMAX

tr, te = train[train["unique_id"] == SERIE], test[test["unique_id"] == SERIE]
sarimax = SARIMAX(tr["y"].values, exog=tr[exog_cols], order=(1, 1, 1), seasonal_order=(1, 1, 1, SEASON)).fit(disp=False)
pred_sarimax = sarimax.forecast(steps=len(te), exog=te[exog_cols])
results[f"SARIMAX ({SERIE})"] = calc_metrics(te["y"], pred_sarimax)
```

### Schritt 10: Klassische ML-Modelle
```python
import lightgbm as lgb
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

mlf = MLForecast(
    models={
        "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE),
    },
    freq=FREQ,
    lags=[1, 2, 3, SEASON, 2 * SEASON],
    lag_transforms={1: [RollingMean(window_size=7)]},
    date_features=["dayofweek", "month"],
)
mlf.fit(train_nf, static_features=[])
pred_ml = mlf.predict(h=HORIZON, X_df=future_exog)
merged = test.merge(pred_ml, on=["unique_id", "ds"])
for m in ["LightGBM", "XGBoost"]:
    results[m] = calc_metrics(merged["y"], merged[m])
```

### Schritt 11: Neurale Modelle
```python
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, TFT, PatchTST, TiDE, TSMixerx

print(f"GPU: {torch.cuda.is_available()}")
n_series = train_nf["unique_id"].nunique()
common = dict(h=HORIZON, input_size=INPUT_SIZE, max_steps=500, random_seed=RANDOM_STATE)

models = [
    NHITS(**common, futr_exog_list=exog_cols),
    TFT(**common, futr_exog_list=exog_cols),
    TSMixerx(**common, futr_exog_list=exog_cols, n_series=n_series),
    TiDE(**common, futr_exog_list=exog_cols),
    NBEATS(**common),        # univariat
    PatchTST(**common),      # univariat
]
nf = NeuralForecast(models=models, freq=FREQ)
nf.fit(df=train_nf)
pred_nf = nf.predict(futr_df=future_exog)
merged = test.merge(pred_nf, on=["unique_id", "ds"])
for m in ["NHITS", "TFT", "TSMixerx", "TiDE", "NBEATS", "PatchTST"]:
    results[m] = calc_metrics(merged["y"], merged[m])
```

### Schritt 12: Cross-Validation
```python
# Rolling-Origin CV, gleiche Logik fuer StatsForecast, MLForecast und NeuralForecast
cv_df = sf.cross_validation(df=train_nf[["unique_id", "ds", "y"]], h=HORIZON, n_windows=3, step_size=HORIZON)
cv_summary = cv_df.groupby("cutoff").apply(lambda g: pd.Series({m: calc_metrics(g["y"], g[m])["MAE"] for m in ["AutoARIMA", "AutoETS", "AutoTheta"]}))
print(cv_summary.round(2))
print("Mittel:", cv_summary.mean().round(2).to_dict())
print("Std:   ", cv_summary.std().round(2).to_dict())
```

### Schritt 13: Evaluation & Modellvergleich
```python
comparison = pd.DataFrame(results).T.sort_values("MAE").round(2)
print(comparison)
BEST = comparison.index[0]

fig, ax = plt.subplots(figsize=(14, 5))
s_test = test[test["unique_id"] == SERIE]
ax.plot(s_test["ds"], s_test["y"], "k-", label="Actual", linewidth=2)
for m in comparison.index[:4]:
    if m in merged.columns:
        mm = merged[merged["unique_id"] == SERIE]
        ax.plot(mm["ds"], mm[m], "--", label=m)
ax.legend(); ax.set_title(f"Forecast vs. Actual ({SERIE})")
plt.tight_layout(); plt.show()
```

### Schritt 14: Fehleranalyse nach Wochentag
```python
merged["abs_err"] = (merged["y"] - merged[BEST]).abs()
merged["dow"] = merged["ds"].dt.day_name()
print(merged.groupby("dow")["abs_err"].mean().round(2))
print(merged.groupby("is_holiday")["abs_err"].mean().round(2))
```

### Schritt 15: SHAP (fuer LightGBM via MLForecast)
```python
import shap

lgb_model = mlf.models_["LightGBM"]
X_feat = mlf.preprocess(train_nf, static_features=[]).drop(columns=["unique_id", "ds", "y"])
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_feat)
shap.summary_plot(shap_values, X_feat)
```

### Schritt 16: Hyperparameter-Tuning (Beispiel N-HiTS mit Optuna)
```python
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    model = NHITS(
        h=HORIZON,
        input_size=trial.suggest_int("input_size", HORIZON, 6 * HORIZON, step=HORIZON),
        max_steps=trial.suggest_int("max_steps", 300, 1000, step=100),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        scaler_type=trial.suggest_categorical("scaler_type", ["standard", "robust", "minmax"]),
        futr_exog_list=exog_cols,
        random_seed=RANDOM_STATE,
    )
    cv = NeuralForecast(models=[model], freq=FREQ).cross_validation(df=train_nf, n_windows=3, step_size=HORIZON)
    return (cv["y"] - cv["NHITS"]).abs().mean()

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=30)
print(study.best_params, study.best_value)
```

### Schritt 17/18: Finales Training & Speichern
```python
from pathlib import Path

full_nf = df[["unique_id", "ds", "y"] + exog_cols]
nf_final = NeuralForecast(models=[NHITS(h=HORIZON, input_size=INPUT_SIZE, max_steps=500, futr_exog_list=exog_cols, random_seed=RANDOM_STATE, **study.best_params)], freq=FREQ)
nf_final.fit(df=full_nf)

MODEL_DIR = Path("models") / f"nhits_v1_{pd.Timestamp.today():%Y%m%d}"
nf_final.save(path=str(MODEL_DIR), overwrite=True)
# Laden: NeuralForecast.load(path=str(MODEL_DIR))
# sklearn/LightGBM: joblib.dump(model, MODEL_DIR / "model.joblib")
```

### Schritt 19: Ergebnisse nach Snowflake schreiben
```python
RUN_ID = f"{cfg['project']['name'].lower().replace(' ', '_')}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
MODEL_NAME, MODEL_VERSION = "NHITS", "v1"

# Zukunfts-Features fuer den echten Forecast bereitstellen (Kalender ist bekannt)
future_dates = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=HORIZON, freq=FREQ)
future_df = pd.MultiIndex.from_product([df["unique_id"].unique(), future_dates], names=["unique_id", "ds"]).to_frame(index=False)
future_df = add_calendar_features(future_df)[["unique_id", "ds"] + exog_cols]
forecast = nf_final.predict(futr_df=future_df).rename(columns={"NHITS": "forecast"})

# 1. Forecast -> PROD_ML.INFERENCE
snapshot = forecast.assign(run_id=RUN_ID, model_name=MODEL_NAME, model_version=MODEL_VERSION, created_at=pd.Timestamp.now())
write_to_snowflake(snapshot, "FORECAST_SNAPSHOT")

# 2. Lauf -> PROD_ML.REGISTRY
run_info = pd.DataFrame([{"run_id": RUN_ID, "model_name": MODEL_NAME, "model_version": MODEL_VERSION,
                          "params": str(study.best_params), "train_start": df["ds"].min(), "train_end": df["ds"].max(),
                          "horizon": HORIZON, "created_at": pd.Timestamp.now()}])
write_to_snowflake(run_info, "MODEL_RUNS", schema=cfg["snowflake"]["schemas"]["registry"])

# 3. Metriken -> PROD_ML.MONITORING
metrics = comparison.reset_index().rename(columns={"index": "model"}).assign(run_id=RUN_ID, created_at=pd.Timestamp.now())
write_to_snowflake(metrics, "MODEL_EVALUATION_LOG", schema=cfg["snowflake"]["schemas"]["monitoring"])
```

## Regeln

- Zeitreihen IMMER zeitlich splitten, nie zufaellig.
- Baseline ist Pflicht. Jedes weitere Modell muss die Baseline schlagen.
- Von einfach zu komplex: Statistisch, dann ML, dann Neural.
- Exogene Features: nur verwenden, was in der Zukunft bekannt ist (Kalender, Feiertage, geplante Aktionen).
- Lag-Features nie aus der Zukunft (kein Data Leakage).
- `random_state` / `random_seed` immer 42 (`RANDOM_STATE`).
- Ergebnisse immer nach PROD_ML schreiben (append, nicht overwrite), mit `run_id`, Modellname und Version.
- Code direkt im Arbeits-Notebook, nicht in `src/`.
- Notebook-Text auf Deutsch, Code auf Englisch.
