#!/usr/bin/env python3
"""
Prediccion_Hibrida_Interactiva.py
---------------------------------
Script interactivo para pronóstico de precios con un enfoque híbrido y robusto:
- Modelos: Prophet (sin regresores para evitar fuga de información) + LSTM
- Ensemble: 60% LSTM + 40% Prophet con suavizado ligero
- Validación: Walk-forward (one-step) opcional con MAPE, RMSE y precisión direccional
- Interfaz interactiva por consola (tickers, conversión a MXN, opción de backtesting)
- Manejo de logs/avisos y fallbacks si TensorFlow no está disponible

NOTA IMPORTANTE: Este código es para fines educativos/informativos. No constituye
recomendación de inversión. Valide resultados antes de cualquier uso real.
"""

# -----------------
# Silenciar logs
# -----------------
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

# -----------------
# Imports
# -----------------
import sys
from contextlib import redirect_stdout
import math
from typing import Tuple, List, Any
import numpy as np
import pandas as pd
import yfinance as yf
import ta

try:
    from prophet import Prophet
    _PROPHET_OK = True
except Exception:
    _PROPHET_OK = False

# TensorFlow opcional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    _TF_OK = True
except Exception:
    _TF_OK = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# -----------------
# Configuración
# -----------------
SEED = 42
np.random.seed(SEED)
if _TF_OK:
    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass

LOOKBACK = 60
LSTM_EPOCHS = 20
LSTM_BATCH = 32
ENSEMBLE_W_LSTM = 0.60
ENSEMBLE_W_PROP = 0.40
SMOOTH_WINDOW = 5  # para suavizar secuencia futura del ensemble

PRED_HORIZONS = [30, 90, 180]

# -----------------
# Utilidades
# -----------------

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze()


def get_stock_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Descarga datos de Yahoo Finance con auto_adjust para evitar splits/dividendos.
    Devuelve DataFrame con índice datetime y columnas OHLCV.
    """
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores básicos sólo para contexto/tabla (no para Prophet)."""
    df = df.copy()
    c = _safe_series(df, 'Close')
    h = _safe_series(df, 'High')
    l = _safe_series(df, 'Low')
    v = _safe_series(df, 'Volume') if 'Volume' in df.columns else pd.Series(index=df.index, dtype=float)

    # Suavizado para el objetivo y para entrenar LSTM
    df['Close_smoothed'] = c.ewm(span=5, adjust=False).mean()

    # Indicadores para contexto
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
    df['ma50'] = df['Close_smoothed'].rolling(50, min_periods=1).mean().bfill()
    df['ma200'] = df['Close_smoothed'].rolling(200, min_periods=1).mean().bfill()
    try:
        df['adx'] = ta.trend.ADXIndicator(h, l, df['Close_smoothed'], window=14).adx().bfill()
    except Exception:
        df['adx'] = np.nan
    return df


# -----------------
# Modelos
# -----------------

def prophet_forecast(df: pd.DataFrame, days_ahead: int) -> pd.Series:
    """Pronóstico con Prophet sobre Close_smoothed. Sin regresores para evitar fuga.
    Devuelve Serie con índice de fechas futuras (longitud days_ahead).
    """
    if not _PROPHET_OK:
        # Si no hay Prophet, usar naive: último valor extendido
        last_date = df.index[-1]
        future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')
        return pd.Series(np.repeat(float(df['Close_smoothed'].iloc[-1]), days_ahead), index=future_idx)

    tmp = df[['Close_smoothed']].reset_index().rename(columns={'index': 'ds', 'Close_smoothed': 'y'})
    tmp = tmp.rename(columns={tmp.columns[0]: 'ds'})  # asegurar nombre de la primera col

    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    with redirect_stdout(sys.stderr):
        m.fit(tmp)
    future = m.make_future_dataframe(periods=days_ahead)
    fcst = m.predict(future)
    tail = fcst[['ds', 'yhat']].tail(days_ahead)
    tail.index = pd.to_datetime(tail['ds'])
    return tail['yhat']


def lstm_forecast(df: pd.DataFrame, days_ahead: int, epochs: int = LSTM_EPOCHS) -> pd.Series:
    """Pronóstico LSTM iterativo sobre Close_smoothed. Si TF no está disponible, naive."""
    last_date = df.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')

    if not _TF_OK:
        return pd.Series(np.repeat(float(df['Close_smoothed'].iloc[-1]), days_ahead), index=future_idx)

    series = df[['Close_smoothed']].values.astype('float32')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    if len(scaled) <= LOOKBACK + 1:
        # datos insuficientes, devolver naive
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
        LSTM(64, return_sequences=True),
        Dropout(0.1),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    callbacks = [EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)] if _TF_OK else []
    with redirect_stdout(sys.stderr):
        model.fit(X, y, epochs=epochs, batch_size=LSTM_BATCH, verbose=0, callbacks=callbacks)

    # Predicción iterativa
    last_window = scaled[-LOOKBACK:].reshape((1, LOOKBACK, 1))
    preds_scaled = []
    for _ in range(days_ahead):
        p = float(model.predict(last_window, verbose=0)[0, 0])
        preds_scaled.append(p)
        # desplazar ventana e insertar predicción
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0, -1, 0] = p

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return pd.Series(preds, index=future_idx)


def ensemble_forecast(df: pd.DataFrame, days_ahead: int) -> pd.Series:
    pf = prophet_forecast(df, days_ahead)
    lf = lstm_forecast(df, days_ahead)
    # Alinear por índice
    common_idx = pf.index.intersection(lf.index)
    arr = ENSEMBLE_W_LSTM * lf.loc[common_idx].values + ENSEMBLE_W_PROP * pf.loc[common_idx].values
    # Suavizado ligero sobre el futuro
    sm = pd.Series(arr, index=common_idx).rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    return sm


# -----------------
# Validación walk-forward (one-step)
# -----------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"MAPE": np.nan, "RMSE": np.nan, "DirAcc": np.nan}
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100) if len(y_true) > 1 else np.nan
    return {"MAPE": mape, "RMSE": rmse, "DirAcc": dir_acc}


def walk_forward_eval(df: pd.DataFrame, steps: int = 30, retrain_epochs_lstm: int = 5) -> pd.DataFrame:
    """Evalúa Prophet, LSTM y Ensemble con predicción a 1 paso hacia adelante.
    Retorna DataFrame con métricas por modelo.
    """
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
        real = float(close_s.iloc[i])

        # Prophet 1-step
        pf = prophet_forecast(train, 1)
        pred_p = float(pf.iloc[-1])

        # LSTM 1-step (re-entrena ligero)
        lf = lstm_forecast(train, 1, epochs=retrain_epochs_lstm)
        pred_l = float(lf.iloc[-1])

        # Ensemble
        pred_e = ENSEMBLE_W_LSTM * pred_l + ENSEMBLE_W_PROP * pred_p

        y_true_p.append(real); y_pred_p.append(pred_p)
        y_true_l.append(real); y_pred_l.append(pred_l)
        y_true_e.append(real); y_pred_e.append(pred_e)

    rows = []
    rows.append({"Modelo": "Prophet", **_metrics(np.array(y_true_p), np.array(y_pred_p))})
    rows.append({"Modelo": "LSTM", **_metrics(np.array(y_true_l), np.array(y_pred_l))})
    rows.append({"Modelo": "Ensemble", **_metrics(np.array(y_true_e), np.array(y_pred_e))})
    return pd.DataFrame(rows)


# -----------------
# Señal (no es recomendación de inversión)
# -----------------

def compute_signal(ma50: float, ma200: float, preds: dict, precio_actual: float, rsi: float, adx: float) -> str:
    """Devuelve una etiqueta descriptiva (Fuerte Alcista/Alcista/Neutral/Bajista/Fuerte Bajista)."""
    # Tendencia por medias
    trend = 1.0 if ma50 > ma200 else -1.0
    # Proyección relativa promedio (cap -1..1)
    proj_list = []
    for d in (30, 90, 180):
        p = preds.get(d, np.nan)
        if not (p is None or (isinstance(p, float) and math.isnan(p))) and precio_actual != 0:
            proj_list.append((p - precio_actual) / precio_actual)
    proj = float(np.mean(proj_list)) if proj_list else 0.0
    proj = max(min(proj, 1.0), -1.0)
    # RSI
    if math.isnan(rsi):
        rsi_score = 0.0
    else:
        if rsi < 30: rsi_score = 0.6
        elif rsi < 50: rsi_score = 0.2
        elif rsi < 70: rsi_score = -0.1
        else: rsi_score = -0.5
    # ADX (fuerza)
    if math.isnan(adx):
        adx_score = 0.0
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


# -----------------
# Conversión USD->MXN (opcional, vía yfinance USDMXN=X)
# -----------------

def usd_to_mxn_ratio() -> float:
    try:
        fx = yf.download("USDMXN=X", period="1y", progress=False, auto_adjust=True)
        if fx is None or fx.empty:
            return 1.0
        return float(fx['Close'].iloc[-1])
    except Exception:
        return 1.0


# -----------------
# Tabla de resultados
# -----------------

def print_table(rows: List[Tuple[Any, ...]]):
    headers = ["Ticker", "Tendencia", "Precio Actual", "30d", "90d", "180d", "Señal"]
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


# -----------------
# Flujo principal por ticker
# -----------------

def analyze_ticker(ticker: str, convert_mxn: bool = False, do_backtest: bool = False) -> Tuple[Any, ...]:
    yf_ticker = ticker.strip().upper()
    # Compatibilidad básica con BMV: permitir prefijo BMV:
    if yf_ticker.startswith("BMV:"):
        yf_ticker = yf_ticker.replace("BMV:", "") + ".MX"

    print(f"\nAnalizando {yf_ticker} ...")
    df = get_stock_data(yf_ticker, period="3y")
    df = add_technical_indicators(df)

    precio_actual = float(df['Close'].iloc[-1])
    fx = 1.0
    if convert_mxn:
        fx = usd_to_mxn_ratio()

    # Pronósticos
    preds = {}
    for d in PRED_HORIZONS:
        ens = ensemble_forecast(df, d)
        val = float(ens.iloc[-1])
        preds[d] = val * fx

    # Señal / tendencia
    last = df.iloc[-1]
    tendencia = "alcista" if last['ma50'] > last['ma200'] else "bajista"
    señal = compute_signal(float(last['ma50']), float(last['ma200']), preds, precio_actual, float(last['rsi']), float(last['adx']))

    # Backtesting opcional
    if do_backtest:
        print("\nValidación walk-forward (one-step). Esto puede tardar un poco...")
        valdf = walk_forward_eval(df, steps=30, retrain_epochs_lstm=5)
        if not valdf.empty:
            print(valdf.to_string(index=False, formatters={
                'MAPE': lambda x: f"{x:,.2f}%",
                'RMSE': lambda x: f"{x:,.2f}",
                'DirAcc': lambda x: f"{x:,.2f}%"
            }))
        else:
            print("No hay suficientes datos para validar.")

    return (
        yf_ticker,
        f"{tendencia}",
        f"{precio_actual*fx:,.2f}",
        f"{preds[30]:,.2f}",
        f"{preds[90]:,.2f}",
        f"{preds[180]:,.2f}",
        señal
    )


# -----------------
# Main interactivo
# -----------------

def main():
    print("Predicción Híbrida (Prophet + LSTM + Ensemble)\n")
    entrada = input("Introduce tickers (comas o espacios) (ej. AAPL MSFT SPY o BMV:AMXL): ").strip()
    if not entrada:
        print("No ingresaste tickers. Saliendo.")
        return
    # split por coma o espacio
    import re
    tickers = [t.strip() for t in re.split(r'[\s,]+', entrada) if t.strip()]

    conv = input("¿Convertir precios a MXN? (s/N): ").strip().lower() == 's'
    backtest = input("¿Ejecutar validación walk-forward (one-step)? (s/N): ").strip().lower() == 's'

    rows = []
    for t in tickers:
        try:
            rows.append(analyze_ticker(t, convert_mxn=conv, do_backtest=backtest))
        except Exception as e:
            print(f"⚠️ Error con {t}: {e}")
            rows.append((t, "ERROR", "N/D", "N/D", "N/D", "N/D", "N/D"))

    print("\n📊 Resumen final")
    print_table(rows)


if __name__ == "__main__":
    main()
