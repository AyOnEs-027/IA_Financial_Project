#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inversiones_Hibrido_FINAL.py
----------------------------
Build estable y robusto (con corrección de MultiIndex de yfinance):
- Prophet (sin regresores) + LSTM robusto
- Ensemble 60% LSTM / 40% Prophet con suavizado (ventana 7)
- Escenarios macro (presets/personalizado) aplicados vía betas OLS
- Walk-forward opcional (MAPE, RMSE, direccional)
- Gráficas guardadas SIEMPRE en ./plots **junto al archivo**
- HARDENING clave:
  * Aplanado de columnas OHLCV cuando yfinance entrega **MultiIndex**
  * `ensure_close_smoothed(df)` en **todas** las rutas que usan la serie objetivo
  * Fallback a 'Adj Close' si falta 'Close'
  * Manejo opcional de `ta` (si no está instalado, indicadores se van a NaN sin romper)

Uso educativo. No es recomendación de inversión.
"""

import os, sys, math, warnings
from contextlib import redirect_stdout
from typing import Tuple, List, Any, Dict, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
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

LOOKBACK = 80
LSTM_EPOCHS = 80
LSTM_BATCH = 32
ENSEMBLE_W_LSTM = 0.60
ENSEMBLE_W_PROP = 0.40
SMOOTH_WINDOW = 7

# ==================
# Paths
# ==================

def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

# ==================
# Aplanado de columnas OHLCV (MultiIndex -> plano)
# ==================

def flatten_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas de yfinance cuando vienen como MultiIndex.
    ('Close','AAPL') -> 'Close'. Asegura Series 1D para OHLCV.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        std = {'Open','High','Low','Close','Adj Close','Volume'}
        for col in df.columns:
            if isinstance(col, tuple):
                parts = list(col)
                pick = parts[0] if parts[0] in std else (parts[1] if len(parts)>1 and parts[1] in std else parts[0])
                new_cols.append(pick)
            else:
                new_cols.append(col)
        df.columns = new_cols
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Drop duplicadas (si hay)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    # Asegurar columnas como Series 1D
    for col in ['Open','High','Low','Close','Adj Close','Volume']:
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                df[col] = s.iloc[:,0]
    return df

# ==================
# Hardening Close_smoothed
# ==================

def ensure_close_smoothed(df: pd.DataFrame) -> pd.DataFrame:
    variants = ['Close_smoothed','CLose_Smoothed','Close_Smoothed','close_smoothed','CLOSE_SMOOTHED']
    found = None
    for v in variants:
        if v in df.columns:
            found = v
            break
    if found and found != 'Close_smoothed':
        df = df.rename(columns={found: 'Close_smoothed'})
    if 'Close_smoothed' not in df.columns:
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        if 'Close' not in df.columns:
            raise KeyError("No se encontró 'Close' ni 'Adj Close' para derivar 'Close_smoothed'.")
        c = df['Close'] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[:,0]
        df['Close_smoothed'] = pd.Series(c).ewm(span=7, adjust=False).mean()
    return df

# ==================
# Utils
# ==================

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:,0]
    return s.squeeze()

# ==================
# Datos
# ==================

def get_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True,
        group_by='column'  # fuerza columnas planas
    )
    if df is None or df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")
    df = flatten_ohlcv_columns(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = flatten_ohlcv_columns(df)

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Close' not in df.columns:
        raise KeyError("No se encontró 'Close' ni 'Adj Close' en el DataFrame.")

    c = _safe_series(df, 'Close')
    h = _safe_series(df, 'High') if 'High' in df.columns else c
    l = _safe_series(df, 'Low')  if 'Low'  in df.columns else c

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
            df['adx'] = ta.trend.ADXIndicator(high=h, low=l, close=df['Close_smoothed'], window=14).adx().bfill()
        except Exception:
            df['adx'] = np.nan
    else:
        df['rsi'] = df['macd'] = df['macd_signal'] = df['adx'] = np.nan

    df['ma50']  = df['Close_smoothed'].rolling(50, min_periods=1).mean().bfill()
    df['ma200'] = df['Close_smoothed'].rolling(200, min_periods=1).mean().bfill()

    df = ensure_close_smoothed(df)
    return df

# ==================
# Macro
# ==================
MACRO_TICKERS = {'irx': '^IRX', 'sp500': '^GSPC', 'oil': 'CL=F'}
_DXY_ALIASES = ['DX-Y.NYB', 'DXY', 'DX-Y.NYB']


def _download_single(tk: str, period='5y') -> Optional[pd.DataFrame]:
    try:
        df = yf.download(tk, period=period, progress=False, auto_adjust=True, group_by='column')
        if df is None or df.empty:
            return None
        df = flatten_ohlcv_columns(df)
        return df
    except Exception:
        return None


def download_macro_series(period: str = '5y') -> pd.DataFrame:
    dfs = {}
    for k, tk in MACRO_TICKERS.items():
        dfs[k] = _download_single(tk, period)
    dxy = None
    for alias in _DXY_ALIASES:
        dxy = _download_single(alias, period)
        if dxy is not None:
            break
    dfs['dxy'] = dxy

    frames = []
    for k, d in dfs.items():
        if d is not None and not d.empty:
            frames.append(d[['Close']].rename(columns={'Close': k}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1, join='outer').sort_index().ffill()
    out = flatten_ohlcv_columns(out)
    return out


def estimate_macro_betas(stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> Dict[str,float]:
    stock_df = flatten_ohlcv_columns(stock_df)
    stock_df = ensure_close_smoothed(stock_df)
    if macro_df is None or macro_df.empty:
        return {k: 0.0 for k in ['irx','dxy','oil','sp500']}
    s = stock_df[['Close_smoothed']].pct_change().dropna()
    m = macro_df[['irx','dxy','oil','sp500']].pct_change().dropna()
    df = s.join(m, how='inner').replace([np.inf,-np.inf], np.nan).dropna()
    if df.empty:
        return {k: 0.0 for k in ['irx','dxy','oil','sp500']}
    y = df['Close_smoothed'].values.reshape(-1,1)
    X = df[['irx','dxy','oil','sp500']].values
    Xi = np.hstack([np.ones((X.shape[0],1)), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    return {'intercept': float(beta[0,0]), 'irx': float(beta[1,0]), 'dxy': float(beta[2,0]), 'oil': float(beta[3,0]), 'sp500': float(beta[4,0])}

# ==================
# Modelos
# ==================

def prophet_forecast(df: pd.DataFrame, days_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = flatten_ohlcv_columns(df)
    df = ensure_close_smoothed(df)
    last_date = df.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')
    if not _PROPHET_OK:
        base = float(df['Close_smoothed'].iloc[-1])
        yhat = pd.Series(np.repeat(base, days_ahead), index=future_idx)
        return yhat, yhat*0, yhat*0
    tmp = df[['Close_smoothed']].reset_index().rename(columns={'index':'ds','Close_smoothed':'y'})
    tmp = tmp.rename(columns={tmp.columns[0]:'ds'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    with redirect_stdout(sys.stderr):
        m.fit(tmp)
    future = m.make_future_dataframe(periods=days_ahead)
    fcst = m.predict(future)
    tail = fcst.tail(days_ahead).copy(); tail.index = pd.to_datetime(tail['ds'])
    return tail['yhat'], tail['yhat_lower'], tail['yhat_upper']


def lstm_forecast(df: pd.DataFrame, days_ahead: int, epochs: int = LSTM_EPOCHS) -> pd.Series:
    df = flatten_ohlcv_columns(df)
    df = ensure_close_smoothed(df)
    last_date = df.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')
    if not _TF_OK:
        return pd.Series(np.repeat(float(df['Close_smoothed'].iloc[-1]), days_ahead), index=future_idx)
    series = df[['Close_smoothed']].values.astype('float32')
    scaler = MinMaxScaler(); scaled = scaler.fit_transform(series)
    if len(scaled) <= LOOKBACK + 1:
        return pd.Series(np.repeat(float(df['Close_smoothed'].iloc[-1]), days_ahead), index=future_idx)
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0]); y.append(scaled[i, 0])
    X = np.array(X).reshape((-1, LOOKBACK, 1)); y = np.array(y)
    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(128, return_sequences=True), Dropout(0.2),
        LSTM(64), Dense(32, activation='relu'), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    callbacks = [EarlyStopping(monitor='loss', patience=8, restore_best_weights=True), ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, verbose=0)]
    with redirect_stdout(sys.stderr):
        model.fit(X, y, epochs=epochs, batch_size=LSTM_BATCH, verbose=0, callbacks=callbacks)
    last_window = scaled[-LOOKBACK:].reshape((1, LOOKBACK, 1))
    preds_scaled = []
    for _ in range(days_ahead):
        p = float(model.predict(last_window, verbose=0)[0, 0])
        preds_scaled.append(p)
        last_window = np.roll(last_window, -1, axis=1); last_window[0, -1, 0] = p
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return pd.Series(preds, index=future_idx)


def ensemble_forecast(df: pd.DataFrame, days_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = flatten_ohlcv_columns(df)
    df = ensure_close_smoothed(df)
    pf_y, pf_lo, pf_hi = prophet_forecast(df, days_ahead)
    lf_y = lstm_forecast(df, days_ahead)
    idx = pf_y.index.intersection(lf_y.index)
    base = ENSEMBLE_W_LSTM * lf_y.loc[idx].values + ENSEMBLE_W_PROP * pf_y.loc[idx].values
    ens = pd.Series(base, index=idx).rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
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
    scale = horizon_days / 90.0
    d_irx = (scenario.get('irx_bps', 0.0) / 100.0) * scale
    d_dxy = (scenario.get('dxy_pct', 0.0) / 100.0) * scale
    d_oil = (scenario.get('oil_pct', 0.0) / 100.0) * scale
    d_spx = (scenario.get('sp500_pct', 0.0) / 100.0) * scale
    adj_ret_total = betas.get('irx',0.0)*d_irx + betas.get('dxy',0.0)*d_dxy + betas.get('oil',0.0)*d_oil + betas.get('sp500',0.0)*d_spx
    steps = len(ens_path)
    if steps == 0: return ens_path
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
    df = flatten_ohlcv_columns(df)
    df = ensure_close_smoothed(df)
    close_s = df['Close_smoothed']
    n = len(close_s)
    if n <= LOOKBACK + 2:
        return pd.DataFrame([])
    y_true_p, y_pred_p, y_true_l, y_pred_l, y_true_e, y_pred_e = [], [], [], [], [], []
    start = max(LOOKBACK + 1, n - steps)
    for i in range(start, n):
        train = flatten_ohlcv_columns(df.iloc[:i])
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
    rows = [
        {"Modelo":"Prophet", **_metrics(np.array(y_true_p), np.array(y_pred_p))},
        {"Modelo":"LSTM", **_metrics(np.array(y_true_l), np.array(y_pred_l))},
        {"Modelo":"Ensemble", **_metrics(np.array(y_true_e), np.array(y_pred_e))},
    ]
    return pd.DataFrame(rows)

# ==================
# Gráficas (plots junto al archivo)
# ==================

def plot_forecasts(ticker: str, df: pd.DataFrame, pf_y: pd.Series, pf_lo: pd.Series, pf_hi: pd.Series,
                   lf_y: pd.Series, ens_y: pd.Series, ens_lo: pd.Series, ens_hi: pd.Series,
                   ens_adj: Optional[pd.Series], scenario_name: str, out_dir: str = 'plots') -> str:
    base_dir = _script_dir(); out_path = os.path.join(base_dir, out_dir)
    os.makedirs(out_path, exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Close', color='#999', linewidth=1.0)
    df = ensure_close_smoothed(df)
    plt.plot(df.index, df['Close_smoothed'], label='Close_smoothed', color='#333', linewidth=1.2)
    plt.plot(pf_y.index, pf_y.values, label='Prophet yhat', color='#1f77b4', alpha=0.9)
    if len(pf_lo)==len(pf_y)==len(pf_hi):
        plt.fill_between(pf_y.index, pf_lo.values, pf_hi.values, color='#1f77b4', alpha=0.15, label='Prophet interval')
    plt.plot(lf_y.index, lf_y.values, label='LSTM', color='#ff7f0e', alpha=0.9)
    plt.plot(ens_y.index, ens_y.values, label='Ensemble', color='#2ca02c', linewidth=2)
    if len(ens_lo)==len(ens_y)==len(ens_hi):
        plt.fill_between(ens_y.index, ens_lo.values, ens_hi.values, color='#2ca02c', alpha=0.10, label='Ensemble band')
    if ens_adj is not None:
        plt.plot(ens_adj.index, ens_adj.values, label=f'Ensemble (escenario: {scenario_name})', color='#d62728', linewidth=2.0, linestyle='--')
    plt.title(f"{ticker} – Histórico y Pronósticos")
    plt.legend(loc='best'); plt.grid(alpha=0.25)
    fname = os.path.join(out_path, f"{ticker.replace(':','_').replace('/','_')}_forecast.png")
    plt.tight_layout(); plt.savefig(fname, dpi=140); plt.close()
    return fname

# ==================
# Tabla
# ==================

def print_table(rows: List[Tuple[Any, ...]]):
    headers = ["Ticker","Tendencia","Precio Actual","30d","90d","180d","Señal","Escenario"]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "+" + "+".join(["-"*(w+2) for w in widths]) + "+"
    def fmt_row(vals):
        return "| " + " | ".join([str(vals[i]).ljust(widths[i]) for i in range(len(vals))]) + " |"
    print(sep); print(fmt_row(headers)); print(sep)
    for r in rows: print(fmt_row(r))
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
    if yf_ticker.startswith('BMV:'):
        yf_ticker = yf_ticker.replace('BMV:','') + '.MX'

    print(f"\nAnalizando {yf_ticker} ...")
    df = get_stock_data(yf_ticker, period='5y')
    df = add_technical_indicators(df)
    df = ensure_close_smoothed(df)

    precio_actual = float(df['Close'].iloc[-1])
    fx = 1.0
    if convert_mxn:
        try:
            fx_df = yf.download('USDMXN=X', period='1y', progress=False, auto_adjust=True, group_by='column')
            fx_df = flatten_ohlcv_columns(fx_df) if fx_df is not None and not fx_df.empty else None
            fx = float(fx_df['Close'].iloc[-1]) if fx_df is not None and not fx_df.empty else 1.0
        except Exception:
            fx = 1.0

    betas = estimate_macro_betas(df, macro_df)

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
    tendencia = 'alcista' if last['ma50'] > last['ma200'] else 'bajista'

    # señal informativa
    rsi = float(last['rsi']) if 'rsi' in df.columns else float('nan')
    adx = float(last['adx']) if 'adx' in df.columns else float('nan')
    proj_list = []
    for d in (30,90,180):
        p = preds.get(d, float('nan'))
        if not (isinstance(p, float) and (math.isnan(p) or precio_actual==0)):
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
    score = 0.25*(1.0 if tendencia=='alcista' else -1.0) + 0.45*proj + 0.15*rsi_score + 0.15*adx_score
    señal = 'Fuerte Alcista' if score>=0.4 else ('Alcista' if score>=0.15 else ('Neutral' if score>-0.15 else ('Bajista' if score>-0.4 else 'Fuerte Bajista')))

    if do_backtest:
        valdf = walk_forward_eval(df, steps=60, retrain_epochs_lstm=12)
        if not valdf.empty:
            print(valdf.to_string(index=False, formatters={'MAPE':lambda x: f"{x:,.2f}%", 'RMSE':lambda x: f"{x:,.2f}", 'DirAcc':lambda x: f"{x:,.2f}%"}))
        else:
            print("No hay suficientes datos para validar.")

    ens_adj_full = apply_macro_scenario(ens_y, betas, scenario, 180)
    plot_path = plot_forecasts(yf_ticker, df, pf_y, pf_lo, pf_hi, lf_y, ens_y, ens_lo, ens_hi, ens_adj_full, scenario_name)
    print(f"Gráfica guardada en: {plot_path}")

    return (
        yf_ticker,
        tendencia,
        f"{precio_actual*fx:,.2f}",
        f"{preds[30]*fx:,.2f}",
        f"{preds[90]*fx:,.2f}",
        f"{preds[180]*fx:,.2f}",
        señal,
        scenario_name
    )

# ==================
# Main
# ==================

def print_intro():
    print("Predicción Híbrida FINAL – Prophet + LSTM + Ensemble + Macro Escenarios + Gráficas\n")


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


if __name__ == '__main__':
    main()
