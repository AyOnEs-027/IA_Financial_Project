#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inversiones_Hibrido.py (BUILD estable)
--------------------------------------
Versión híbrida **interactiva** con hardening para evitar errores comunes:
- Prophet (sin regresores) + LSTM robusto
- Ensemble 60% LSTM / 40% Prophet con suavizado
- Escenarios macro (presets o personalizados) aplicados vía **betas OLS**
- Gráficas guardadas SIEMPRE en `./plots` **junto al archivo** (no al CWD)
- Walk-forward opcional (MAPE, RMSE, precisión direccional)
- **HARDENING**:
  * `ensure_close_smoothed(df)` normaliza/crea `Close_smoothed` en todas las rutas
  * Fallback a `Adj Close` si falta `Close`
  * Normalización de ticker a UPPERCASE para salida

NOTA: Uso educativo/informativo. No es recomendación de inversión.
"""

# ==================
# Silencioso / imports
# ==================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True
logging.getLogger("fbprophet").disabled = True
logging.basicConfig(level=logging.ERROR)

import sys
from contextlib import redirect_stdout
import math
from typing import Tuple, List, Any, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # backend no interactivo
import matplotlib.pyplot as plt

try:
    import ta
except Exception:
    ta = None

try:
    from prophet import Prophet
    _PROPHET_OK = True
except Exception:
    _PROPHET_OK = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    _TF_OK = True
except Exception:
    _TF_OK = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ==================
# Config
# ==================
SEED = 42
np.random.seed(SEED)
if _TF_OK:
    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass

LOOKBACK = 80              # mayor contexto para precisión
LSTM_EPOCHS = 80           # prioriza exactitud sobre rapidez
LSTM_BATCH = 32
ENSEMBLE_W_LSTM = 0.60
ENSEMBLE_W_PROP = 0.40
SMOOTH_WINDOW = 7          # un poco más de suavizado
PRED_HORIZONS = [30, 90, 180]

# ==================
# Helpers de archivo/paths
# ==================

def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

# ==================
# HARDENING: asegurar Close_smoothed
# ==================

def ensure_close_smoothed(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza/crea la columna 'Close_smoothed':
    - Renombra variantes (CLose_Smoothed, Close_Smoothed, close_smoothed, etc.) → 'Close_smoothed'
    - Si no existe, la crea a partir de 'Close' (o 'Adj Close') con EWMA(7)
    """
    variants = ['Close_smoothed', 'CLose_Smoothed', 'Close_Smoothed', 'close_smoothed', 'CLOSE_SMOOTHED']
    # Renombrar si encuentra variante
    for v in variants:
        if v in df.columns:
            if v != 'Close_smoothed':
                df = df.rename(columns={v: 'Close_smoothed'})
            break
    # Crear si no existe
    if 'Close_smoothed' not in df.columns:
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                raise KeyError("No se encontró 'Close' ni 'Adj Close' para derivar 'Close_smoothed'.")
        c = df['Close'] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[:, 0]
        df['Close_smoothed'] = c.ewm(span=7, adjust=False).mean()
    return df

# ==================
# Utils
# ==================

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze()


def get_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Fallback por si faltan columnas críticas
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fallback Close
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Close' not in df.columns:
        raise KeyError("No se encontró 'Close' ni 'Adj Close' en el DataFrame.")

    c = _safe_series(df, 'Close')
    h = _safe_series(df, 'High') if 'High' in df.columns else c
    l = _safe_series(df, 'Low') if 'Low' in df.columns else c

    # Base para modelos/indicadores
    df['Close_smoothed'] = c.ewm(span=7, adjust=False).mean()

    if ta is not None:
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['Close_smoothed'], window=14).rsi().bfill()
        except Exception:
            df['rsi'] = np.nan
        try:
            macd = ta.trend.MACD(df['Close_smoothed'])
            df['macd'] = macd.macd().bfill()
            df['macd_signal'] = macd.macd_signal().bfill()
        except Exception:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
        try:
            df['adx'] = ta.trend.ADXIndicator(h, l, df['Close_smoothed'], window=14).adx().bfill()
        except Exception:
            df['adx'] = np.nan
    else:
        df['rsi'] = np.nan
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['adx'] = np.nan

    df['ma50'] = df['Close_smoothed'].rolling(50, min_periods=1).mean().bfill()
    df['ma200'] = df['Close_smoothed'].rolling(200, min_periods=1).mean().bfill()

    # Hardening final de nombre
    df = ensure_close_smoothed(df)
    return df

# ==================
# Macro data
# ==================
MACRO_TICKERS = {
    'irx': '^IRX',
    'sp500': '^GSPC',
    'oil': 'CL=F'
}
_DXY_ALIASES = ['DX-Y.NYB', 'DXY', 'DX-Y.NYB']


def _download_single(ticker: str, period: str = '5y') -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def download_macro_series(period: str = '5y') -> pd.DataFrame:
    dfs = {}
    for k, tk in MACRO_TICKERS.items():
        dfs[k] = _download_single(tk, period)
    dxy_df = None
    for alias in _DXY_ALIASES:
        dxy_df = _download_single(alias, period)
        if dxy_df is not None and not dxy_df.empty:
            break
    dfs['dxy'] = dxy_df

    frames = []
    for k, d in dfs.items():
        if d is not None and not d.empty:
            frames.append(d[['Close']].rename(columns={'Close': k}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1, join='outer').sort_index().ffill()
    return out


def estimate_macro_betas(stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> Dict[str, float]:
    if macro_df is None or macro_df.empty:
        return {k: 0.0 for k in ['irx','dxy','oil','sp500']}
    stock_df = ensure_close_smoothed(stock_df)
    s = stock_df[['Close_smoothed']].pct_change().dropna()
    m = macro_df[['irx','dxy','oil','sp500']].pct_change().dropna()
    df = s.join(m, how='inner').replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return {k: 0.0 for k in ['irx','dxy','oil','sp500']}
    y = df['Close_smoothed'].values.reshape(-1,1)
    X = df[['irx','dxy','oil','sp500']].values
    Xi = np.hstack([np.ones((X.shape[0],1)), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    return {'intercept': float(beta[0,0]), 'irx': float(beta[1,0]), 'dxy': float(beta[2,0]), 'oil': float(beta[3,0]), 'sp500': float(beta[4,0])}

# ==================
# Modelos y pronóstico
# ==================

def prophet_forecast(df: pd.DataFrame, days_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = ensure_close_smoothed(df)
    last_date = df.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')

    if not _PROPHET_OK:
        base = float(df['Close_smoothed'].iloc[-1])
        yhat = pd.Series(np.repeat(base, days_ahead), index=future_idx)
        return yhat, yhat*0, yhat*0

    tmp = df[['Close_smoothed']].reset_index().rename(columns={'index': 'ds', 'Close_smoothed': 'y'})
    tmp = tmp.rename(columns={tmp.columns[0]: 'ds'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    with redirect_stdout(sys.stderr):
        m.fit(tmp)
    future = m.make_future_dataframe(periods=days_ahead)
    fcst = m.predict(future)
    tail = fcst.tail(days_ahead).copy()
    tail.index = pd.to_datetime(tail['ds'])
    return tail['yhat'], tail['yhat_lower'], tail['yhat_upper']


def lstm_forecast(df: pd.DataFrame, days_ahead: int, epochs: int = LSTM_EPOCHS) -> pd.Series:
    df = ensure_close_smoothed(df)
    last_date = df.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')

    if not _TF_OK:
        return pd.Series(np.repeat(float(df['Close_smoothed'].iloc[-1]), days_ahead), index=future_idx)

    series = df[['Close_smoothed']].values.astype('float32')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    if len(scaled) <= LOOKBACK + 1:
        return pd.Series(np.repeat(float(df['Close_smoothed'].iloc[-1]), days_ahead), index=future_idx)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    callbacks = [
        EarlyStopping(monitor='loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, verbose=0)
    ]
    with redirect_stdout(sys.stderr):
        model.fit(X, y, epochs=epochs, batch_size=LSTM_BATCH, verbose=0, callbacks=callbacks)

    last_window = scaled[-LOOKBACK:].reshape((1, LOOKBACK, 1))
    preds_scaled = []
    for _ in range(days_ahead):
        p = float(model.predict(last_window, verbose=0)[0, 0])
        preds_scaled.append(p)
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0, -1, 0] = p

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return pd.Series(preds, index=future_idx)


def ensemble_forecast(df: pd.DataFrame, days_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = ensure_close_smoothed(df)
    pf_y, pf_lo, pf_hi = prophet_forecast(df, days_ahead)
    lf_y = lstm_forecast(df, days_ahead)
    idx = pf_y.index.intersection(lf_y.index)
    base = ENSEMBLE_W_LSTM * lf_y.loc[idx].values + ENSEMBLE_W_PROP * pf_y.loc[idx].values
    ens = pd.Series(base, index=idx).rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    # bandas: Prophet como proxy + pequeño spread del LSTM
    lo = ENSEMBLE_W_PROP * pf_lo.loc[idx].values + ENSEMBLE_W_LSTM * (lf_y.loc[idx].values * 0.98)
    hi = ENSEMBLE_W_PROP * pf_hi.loc[idx].values + ENSEMBLE_W_LSTM * (lf_y.loc[idx].values * 1.02)
    return ens, pd.Series(lo, index=idx), pd.Series(hi, index=idx), lf_y.loc[idx]

# ==================
# Macro escenarios
# ==================
PRESET_SCENARIOS = {
    'neutral': {'irx_bps': 0,   'dxy_pct': 0.0,  'oil_pct': 0.0,  'sp500_pct': 0.0},
    'risk_on': {'irx_bps': -25, 'dxy_pct': -1.0, 'oil_pct': 5.0,  'sp500_pct': 3.0},
    'risk_off':{'irx_bps': 25,  'dxy_pct': 1.5,  'oil_pct': -4.0, 'sp500_pct': -3.0},
    'hawkish': {'irx_bps': 50,  'dxy_pct': 2.0,  'oil_pct': -2.0, 'sp500_pct': -4.0},
    'dovish':  {'irx_bps': -50, 'dxy_pct': -2.0, 'oil_pct': 2.0,  'sp500_pct': 4.0}
}


def choose_macro_scenario_interactive() -> Tuple[str, Dict[str, float]]:
    print("\nEscenarios macro (base 90 días):")
    for i, k in enumerate(PRESET_SCENARIOS.keys(), 1):
        print(f" {i}) {k}")
    print(" 6) personalizado")
    try:
        op = int(input("Elige una opción [1-6]: ").strip() or '1')
    except Exception:
        op = 1
    if op in range(1,6):
        name = list(PRESET_SCENARIOS.keys())[op-1]
        return name, PRESET_SCENARIOS[name]
    # personalizado
    def _float(prompt, default):
        try:
            v = input(f"{prompt} (default {default}): ").strip()
            return float(v) if v != '' else default
        except Exception:
            return default
    s = {
        'irx_bps': _float('Cambio en tasas IRX (bps) para ~90d', 0),
        'dxy_pct': _float('Cambio en DXY (%) para ~90d', 0.0),
        'oil_pct': _float('Cambio en WTI (%) para ~90d', 0.0),
        'sp500_pct': _float('Cambio en S&P500 (%) para ~90d', 0.0)
    }
    return 'custom', s


def apply_macro_scenario(ens_path: pd.Series, betas: Dict[str,float], scenario: Dict[str,float], horizon_days: int) -> pd.Series:
    """Ajusta la trayectoria del ensemble según betas y el escenario para el horizonte dado.
    Escala linealmente cambios especificados para ~90d a horizon_days.
    """
    scale = horizon_days / 90.0
    d_irx = (scenario.get('irx_bps', 0.0) / 100.0) * scale   # bps->nivel aproximado
    d_dxy = (scenario.get('dxy_pct', 0.0) / 100.0) * scale
    d_oil = (scenario.get('oil_pct', 0.0) / 100.0) * scale
    d_spx = (scenario.get('sp500_pct', 0.0) / 100.0) * scale

    adj_ret_total = betas.get('irx',0.0)*d_irx + betas.get('dxy',0.0)*d_dxy + \
                     betas.get('oil',0.0)*d_oil + betas.get('sp500',0.0)*d_spx

    steps = len(ens_path)
    if steps == 0:
        return ens_path
    per_step = adj_ret_total / steps
    cumulative = np.cumsum(np.repeat(per_step, steps))
    adj = ens_path.values * (1.0 + cumulative)
    return pd.Series(adj, index=ens_path.index)

# ==================
# Validación walk-forward (one-step)
# ==================

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"MAPE": np.nan, "RMSE": np.nan, "DirAcc": np.nan}
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100) if len(y_true) > 1 else np.nan
    return {"MAPE": mape, "RMSE": rmse, "DirAcc": dir_acc}


def walk_forward_eval(df: pd.DataFrame, steps: int = 60, retrain_epochs_lstm: int = 10) -> pd.DataFrame:
    df = ensure_close_smoothed(df)
    close_s = df['Close_smoothed']
    n = len(close_s)
    if n <= LOOKBACK + 2:
        return pd.DataFrame([])

    y_true_p, y_pred_p = [], []
    y_true_l, y_pred_l = [], []
    y_true_e, y_pred_e = [], []

    start = max(LOOKBACK + 1, n - steps)
    for i in range(start, n):
        train = df.iloc[:i]
        train = ensure_close_smoothed(train)
        real = float(close_s.iloc[i])
        pf,_,_ = prophet_forecast(train, 1)
        pred_p = float(pf.iloc[-1])
        lf = lstm_forecast(train, 1, epochs=retrain_epochs_lstm)
        pred_l = float(lf.iloc[-1])
        pred_e = ENSEMBLE_W_LSTM * pred_l + ENSEMBLE_W_PROP * pred_p
        y_true_p.append(real); y_pred_p.append(pred_p)
        y_true_l.append(real); y_pred_l.append(pred_l)
        y_true_e.append(real); y_pred_e.append(pred_e)

    rows = []
    rows.append({"Modelo": "Prophet", **_metrics(np.array(y_true_p), np.array(y_pred_p))})
    rows.append({"Modelo": "LSTM", **_metrics(np.array(y_true_l), np.array(y_pred_l))})
    rows.append({"Modelo": "Ensemble", **_metrics(np.array(y_true_e), np.array(y_pred_e))})
    return pd.DataFrame(rows)

# ==================
# Señal (informativa)
# ==================

def compute_signal(ma50: float, ma200: float, preds: Dict[int,float], precio_actual: float, rsi: float, adx: float) -> str:
    trend = 1.0 if ma50 > ma200 else -1.0
    proj_list = []
    for d in (30, 90, 180):
        p = preds.get(d, np.nan)
        if not (p is None or (isinstance(p, float) and math.isnan(p))) and precio_actual != 0:
            proj_list.append((p - precio_actual) / precio_actual)
    proj = float(np.mean(proj_list)) if proj_list else 0.0
    proj = max(min(proj, 1.0), -1.0)
    if math.isnan(rsi): rsi_score = 0.0
    else:
        if rsi < 30: rsi_score = 0.6
        elif rsi < 50: rsi_score = 0.2
        elif rsi < 70: rsi_score = -0.1
        else: rsi_score = -0.5
    if math.isnan(adx): adx_score = 0.0
    else:
        if adx > 25: adx_score = 0.3
        elif adx > 20: adx_score = 0.15
        else: adx_score = 0.0
    score = 0.25*trend + 0.45*proj + 0.15*rsi_score + 0.15*adx_score
    score = max(min(score, 1.0), -1.0)
    if score >= 0.4: return "Fuerte Alcista"
    if score >= 0.15: return "Alcista"
    if score > -0.15: return "Neutral"
    if score > -0.4: return "Bajista"
    return "Fuerte Bajista"

# ==================
# FX USD->MXN
# ==================

def usd_to_mxn_ratio() -> float:
    try:
        fx = yf.download("USDMXN=X", period="1y", progress=False, auto_adjust=True)
        if fx is None or fx.empty:
            return 1.0
        return float(fx['Close'].iloc[-1])
    except Exception:
        return 1.0

# ==================
# Gráficas (plots junto al archivo)
# ==================

def plot_forecasts(ticker: str, df: pd.DataFrame, pf_y: pd.Series, pf_lo: pd.Series, pf_hi: pd.Series,
                   lf_y: pd.Series, ens_y: pd.Series, ens_lo: pd.Series, ens_hi: pd.Series,
                   ens_adj: Optional[pd.Series], scenario_name: str, out_dir: str = 'plots') -> str:
    base_dir = _script_dir()
    out_path = os.path.join(base_dir, out_dir)
    os.makedirs(out_path, exist_ok=True)

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Close', color='#999', linewidth=1.0)
    df = ensure_close_smoothed(df)
    plt.plot(df.index, df['Close_smoothed'], label='Close_smoothed', color='#333', linewidth=1.2)
    # Prophet
    plt.plot(pf_y.index, pf_y.values, label='Prophet yhat', color='#1f77b4', alpha=0.9)
    if len(pf_lo)==len(pf_y)==len(pf_hi):
        plt.fill_between(pf_y.index, pf_lo.values, pf_hi.values, color='#1f77b4', alpha=0.15, label='Prophet interval')
    # LSTM
    plt.plot(lf_y.index, lf_y.values, label='LSTM', color='#ff7f0e', alpha=0.9)
    # Ensemble
    plt.plot(ens_y.index, ens_y.values, label='Ensemble', color='#2ca02c', linewidth=2)
    if len(ens_lo)==len(ens_y)==len(ens_hi):
        plt.fill_between(ens_y.index, ens_lo.values, ens_hi.values, color='#2ca02c', alpha=0.10, label='Ensemble band')
    # Ajuste por escenario
    if ens_adj is not None:
        plt.plot(ens_adj.index, ens_adj.values, label=f'Ensemble (escenario: {scenario_name})', color='#d62728', linewidth=2.0, linestyle='--')

    plt.title(f"{ticker} – Histórico y Pronósticos")
    plt.legend(loc='best')
    plt.grid(alpha=0.25)
    fname = os.path.join(out_path, f"{ticker.replace(':','_').replace('/','_')}_forecast.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()
    return fname

# ==================
# Tabla
# ==================

def print_table(rows: List[Tuple[Any, ...]]):
    headers = ["Ticker", "Tendencia", "Precio Actual", "30d", "90d", "180d", "Señal", "Escenario"]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "+" + "+".join(["-"*(w+2) for w in widths]) + "+"
    def fmt_row(vals):
        return "| " + " | ".join([str(vals[i]).ljust(widths[i]) for i in range(len(vals))]) + " |"
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)

# ==================
# Flujo por ticker
# ==================

def prophet_lstm_ensemble(df: pd.DataFrame, horizon: int):
    pf_y, pf_lo, pf_hi = prophet_forecast(df, horizon)
    lf_y = lstm_forecast(df, horizon)
    ens_y, ens_lo, ens_hi, _ = ensemble_forecast(df, horizon)
    return pf_y, pf_lo, pf_hi, lf_y, ens_y, ens_lo, ens_hi


def analyze_ticker(ticker: str, convert_mxn: bool, do_backtest: bool,
                   macro_df: pd.DataFrame, scenario_name: str, scenario: Dict[str,float]) -> Tuple[Any, ...]:
    yf_ticker = ticker.strip().upper()
    if yf_ticker.startswith("BMV:"):
        yf_ticker = yf_ticker.replace("BMV:", "") + ".MX"

    print(f"\nAnalizando {yf_ticker} ...")
    df = get_stock_data(yf_ticker, period="5y")
    df = add_technical_indicators(df)
    df = ensure_close_smoothed(df)

    precio_actual = float(df['Close'].iloc[-1])
    fx = usd_to_mxn_ratio() if convert_mxn else 1.0

    betas = estimate_macro_betas(df, macro_df)

    # Pronósticos 180d y luego recortes 30/90/180 con escenario
    pf_y, pf_lo, pf_hi, lf_y, ens_y, ens_lo, ens_hi = prophet_lstm_ensemble(df, 180)

    ens_adj_30  = apply_macro_scenario(ens_y.iloc[:30],  betas, scenario, 30)
    ens_adj_90  = apply_macro_scenario(ens_y.iloc[:90],  betas, scenario, 90)
    ens_adj_180 = apply_macro_scenario(ens_y.iloc[:180], betas, scenario, 180)

    preds = {
        30: float(ens_adj_30.iloc[-1])  if len(ens_adj_30)>0  else float(ens_y.iloc[29]),
        90: float(ens_adj_90.iloc[-1])  if len(ens_adj_90)>0  else float(ens_y.iloc[89]),
        180: float(ens_adj_180.iloc[-1]) if len(ens_adj_180)>0 else float(ens_y.iloc[179])
    }

    last = df.iloc[-1]
    tendencia = "alcista" if last['ma50'] > last['ma200'] else "bajista"
    señal = compute_signal(float(last['ma50']), float(last['ma200']), preds, precio_actual, float(last['rsi']), float(last['adx']))

    if do_backtest:
        print("\nValidación walk-forward (one-step). Esto puede tardar ...")
        valdf = walk_forward_eval(df, steps=60, retrain_epochs_lstm=12)
        if not valdf.empty:
            print(valdf.to_string(index=False, formatters={
                'MAPE': lambda x: f"{x:,.2f}%",
                'RMSE': lambda x: f"{x:,.2f}",
                'DirAcc': lambda x: f"{x:,.2f}%"
            }))
        else:
            print("No hay suficientes datos para validar.")

    # Gráfica completa con trayectoria ajustada 180d
    ens_adj_full = apply_macro_scenario(ens_y, betas, scenario, 180)
    plot_path = plot_forecasts(yf_ticker, df, pf_y, pf_lo, pf_hi, lf_y, ens_y, ens_lo, ens_hi, ens_adj_full, scenario_name)
    print(f"Gráfica guardada en: {plot_path}")

    row = (
        yf_ticker,  # en mayúsculas en la tabla
        tendencia,
        f"{precio_actual*fx:,.2f}",
        f"{preds[30]*fx:,.2f}",
        f"{preds[90]*fx:,.2f}",
        f"{preds[180]*fx:,.2f}",
        señal,
        scenario_name
    )
    return row

# ==================
# Main interactivo
# ==================

def print_intro():
    print("Predicción Híbrida v2 (fix) – Prophet + LSTM + Ensemble + Macro Escenarios + Gráficas\n")


def main():
    print_intro()
    entrada = input("Introduce tickers (comas o espacios) (ej. AAPL MSFT SPY o BMV:AMXL): ").strip()
    if not entrada:
        print("No ingresaste tickers. Saliendo.")
        return
    import re
    tickers = [t.strip() for t in re.split(r'[\s,]+', entrada) if t.strip()]

    conv = input("¿Convertir precios a MXN? (s/N): ").strip().lower() == 's'
    backtest = input("¿Ejecutar validación walk-forward (one-step)? (s/N): ").strip().lower() == 's'

    print("\nDescargando series macro...")
    macro_df = download_macro_series('5y')

    print("Selecciona escenario macro:")
    scen_name, scen = choose_macro_scenario_interactive()

    rows = []
    for t in tickers:
        try:
            rows.append(analyze_ticker(t, convert_mxn=conv, do_backtest=backtest,
                                       macro_df=macro_df, scenario_name=scen_name, scenario=scen))
        except Exception as e:
            print(f"⚠️ Error con {t}: {e}")
            rows.append((t.upper(), "ERROR", "N/D", "N/D", "N/D", "N/D", "N/D", scen_name))

    print("\n📊 Resumen final")
    print_table(rows)
    print("\nListo. Carpeta 'plots' creada en el MISMO directorio de este archivo.")


if __name__ == "__main__":
    main()
