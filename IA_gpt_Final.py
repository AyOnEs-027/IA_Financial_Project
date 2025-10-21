#!/usr/bin/env python3
"""
Invest_Predictions_TEST2.py
Versión final y extendida (todo en un solo archivo).
Características principales:
 - Interactivo: pide tickers (separador coma o espacio)
 - Predicciones: Prophet (con regresores técnicos y macro) + LSTM (ensemble 70/30)
 - Regresores macro: ^IRX (tasa), DXY (dólar), CL=F (petróleo), ^GSPC (S&P500) vía yfinance
 - Sentimiento: NewsAPI + VADER (API key incluida)
 - Conversión opcional USD->MXN usando exchangerate.host
 - Tabla final alineada con símbolos ▲ (verde) y ▼ (rojo) y columnas en el orden solicitado
 - Spinner "Analizando ..." por ticker
 - Supresión de logs y warnings molestos
 - Comentarios claros en todo el código para facilitar lectura y modificaciones
"""

# -------------------------------------------------------
# 1) SILENCIAR LOGS / WARNINGS (debe ir antes de importar TF/Prophet)
# -------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # oculta logs informativos de TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # evita mensajes about oneDNN

import warnings
warnings.filterwarnings("ignore")  # silenciar warnings globalmente (ajustable)

import logging
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True
logging.getLogger("fbprophet").disabled = True
logging.basicConfig(level=logging.ERROR)

# -------------------------------------------------------
# 2) IMPORTS PRINCIPALES
# -------------------------------------------------------
import re
import sys
import time
import math
import random
import threading
from typing import List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import ta
from prophet import Prophet

# tensorflow + keras (opcional: si no está, se cae a Prophet únicamente)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    EarlyStopping = None

from sklearn.preprocessing import MinMaxScaler

# colorama para colores ANSI en Windows/VSCode
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
except Exception:
    class _Fake:
        RESET_ALL = ""
    Fore = type("F", (), {"GREEN":"", "RED":"", "YELLOW":""})()
    Style = _Fake()

# -------------------------------------------------------
# 3) CONFIG: NewsAPI key y símbolos
# -------------------------------------------------------
NEWS_API_KEY = "6f2ff15ba4164b4981ab6c83c1c45aee"  # tu key, tal como pediste
SYMBOL_UP = "▲"
SYMBOL_DOWN = "▼"

# intentamos inicializar NewsAPI & VADER (sin romper si falla)
SENTIMIENTO_ACTIVO = False
try:
    from newsapi import NewsApiClient
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    analyzer = SentimentIntensityAnalyzer()
    SENTIMIENTO_ACTIVO = True
except Exception:
    SENTIMIENTO_ACTIVO = False

# -------------------------------------------------------
# 4) SEMILLAS PARA ESTABILIDAD (reduce saltos drásticos)
# -------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
try:
    if tf is not None:
        tf.random.set_seed(SEED)
except Exception:
    pass

# -------------------------------------------------------
# 5) UTILIDADES: ANSI, ancho columnas, spinner
# -------------------------------------------------------
_ansi_re = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

def strip_ansi(s: str) -> str:
    """Quita códigos ANSI para medir longitud real del texto mostrado."""
    return _ansi_re.sub('', str(s))

def col_display_width(s: str) -> int:
    return len(strip_ansi(s))

def format_cell_with_ansi(text: str, width: int) -> str:
    """Rellena o trunca teniendo en cuenta códigos ANSI (para mantener la tabla alineada)."""
    s = str(text)
    raw = strip_ansi(s)
    if len(raw) <= width:
        return s + " " * (width - len(raw))
    truncated = raw[:max(0, width-3)] + "..."
    return truncated + " " * (width - len(truncated))

# spinner (animación) para la consola -- compatible con Windows/VSCode
def spinner_start(message: str, stop_event: threading.Event):
    """Imprime message y va añadiendo puntos hasta que stop_event esté seteado."""
    sys.stdout.write(message)
    sys.stdout.flush()
    i = 0
    while not stop_event.is_set():
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(0.6)
        i += 1
        if i % 12 == 0:
            # limpia la línea de puntos cada x iteraciones
            sys.stdout.write('\b' * 12 + ' ' * 12 + '\b' * 12)
            sys.stdout.flush()
    sys.stdout.write("\n")

# -------------------------------------------------------
# 6) TIPO DE CAMBIO (opcional): USD -> MXN usando exchangerate.host
# -------------------------------------------------------
def get_usd_to_mxn(timeout: int = 6) -> Optional[float]:
    """Obtiene tipo de cambio USD->MXN desde exchangerate.host (sin API key)."""
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=MXN", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return float(data["rates"]["MXN"])
    except Exception:
        return None

# -------------------------------------------------------
# 7) DESCARGA DE DATOS (preprocesamiento)
# -------------------------------------------------------
def download_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    """
    Descarga precio histórico de Yahoo Finance.
    - Usa auto_adjust=True para obtener precios ajustados y evitar warnings.
    - Periodo por defecto 3 años para balance entre reciente y contexto.
    """
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def safe_squeeze_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Asegura que la columna devuelta sea una pd.Series 1D (evita errores en 'ta')."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade indicadores técnicos clave:
     - Close_smoothed (EWMA)
     - RSI, MACD, MACD signal
     - MA50, MA200
     - ADX, OBV, Bollinger Bands
    """
    df = df.copy()
    # columnas 1D seguras
    close = safe_squeeze_col(df, 'Close')
    high = safe_squeeze_col(df, 'High')
    low  = safe_squeeze_col(df, 'Low')
    vol  = safe_squeeze_col(df, 'Volume')

    # Suavizado para reducir ruido
    df['Close_smoothed'] = close.ewm(span=5, adjust=False).mean()

    c = df['Close_smoothed']
    df['rsi'] = ta.momentum.RSIIndicator(c, window=14).rsi().bfill()
    macd = ta.trend.MACD(c)
    df['macd'] = macd.macd().bfill()
    df['macd_signal'] = macd.macd_signal().bfill()

    df['ma50'] = c.rolling(50, min_periods=1).mean().bfill()
    df['ma200'] = c.rolling(200, min_periods=1).mean().bfill()

    try:
        df['adx'] = ta.trend.ADXIndicator(high, low, c, window=14).adx().bfill()
    except Exception:
        df['adx'] = np.nan
    try:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=c, volume=vol).on_balance_volume().bfill()
    except Exception:
        df['obv'] = np.nan

    df['bb_upper'] = c.rolling(20).mean() + 2*c.rolling(20).std()
    df['bb_lower'] = c.rolling(20).mean() - 2*c.rolling(20).std()

    return df

# -------------------------------------------------------
# 8) FACTORES MACRO (via yfinance): descarga y resumen
# -------------------------------------------------------
MACRO_TICKERS = {
    'irx': '^IRX',      # proxy de tasas cortas (puede no estar disponible en todos los mercados)
    'dxy': 'DX-Y.NYB',  # DXY (algunas instalaciones usan 'DXY' o 'DX-Y.NYB' - probamos uno y si falla lo omitimos)
    'oil': 'CL=F',      # Petróleo WTI
    'sp500': '^GSPC'    # S&P500
}

def download_macro(timeout: int = 8) -> dict:
    """
    Descarga datos macro de los tickers definidos. Devuelve dict con últimos movimientos % (30d).
    Si algún ticker no se obtiene, lo omite y devuelve NaN para ese factor.
    """
    macro = {}
    for key, tk in MACRO_TICKERS.items():
        try:
            df = yf.download(tk, period="2y", progress=False, auto_adjust=True)
            if df is None or df.empty:
                macro[key] = np.nan
                continue
            # % cambio último mes (aprox 22 días hábiles)
            last = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-22] if len(df) > 22 else df['Close'].iloc[0]
            pct = (last - prev) / prev * 100.0
            macro[key] = float(pct)
        except Exception:
            macro[key] = np.nan
    return macro

def interpret_macro(macro: dict) -> List[str]:
    """
    Genera líneas interpretativas sencillas a partir de los % cambios macro.
    """
    lines = []
    # IRX (tasas)
    irx = macro.get('irx', np.nan)
    if not math.isnan(irx):
        lines.append(f"- Tasa (IRX) cambio ~ {irx:+.2f}% " + ("📈" if irx > 0 else "📉"))
    else:
        lines.append("- Tasa (IRX): N/D")
    # DXY
    dxy = macro.get('dxy', np.nan)
    if not math.isnan(dxy):
        lines.append(f"- Dólar (DXY) cambio ~ {dxy:+.2f}% " + ("📈" if dxy > 0 else "📉"))
    else:
        lines.append("- Dólar (DXY): N/D")
    # Oil
    oil = macro.get('oil', np.nan)
    if not math.isnan(oil):
        lines.append(f"- Petróleo (WTI) cambio ~ {oil:+.2f}% " + ("📈" if oil > 0 else "📉"))
    else:
        lines.append("- Petróleo (CL=F): N/D")
    # S&P500
    sp = macro.get('sp500', np.nan)
    if not math.isnan(sp):
        lines.append(f"- S&P500 cambio ~ {sp:+.2f}% " + ("📈" if sp > 0 else "📉"))
    else:
        lines.append("- S&P500 (^GSPC): N/D")
    return lines

# -------------------------------------------------------
# 9) PROPHET con regressors técnicos y macroeconómicos
# -------------------------------------------------------
def prophet_forecast_with_regressors(df: pd.DataFrame, macro_current: dict, days_ahead: int, interval_width: float = 0.8) -> pd.DataFrame:
    """
    Construye un dataframe para Prophet que incluye:
     - y = Close_smoothed
     - regressors: rsi, macd, obv, Volume
     - regressors macro: irx, dxy, oil, sp500 (constantes, últimos valores)
    Retorna últimos 'days_ahead' pronosticados.
    """
    tmp = df.reset_index().copy()
    tmp = tmp.rename(columns={'index': 'Date'}) if 'index' in tmp.columns else tmp
    if 'Date' not in tmp.columns:
        tmp['Date'] = tmp.iloc[:,0]
    prophet_df = tmp[['Date', 'Close_smoothed', 'rsi', 'macd', 'obv', 'Volume']].copy()
    prophet_df.columns = ['ds', 'y', 'rsi', 'macd', 'obv', 'Volume']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Agregar columnas macro con valores constantes del dict macro_current
    for macro_name in ['irx', 'dxy', 'oil', 'sp500']:
        val = macro_current.get(macro_name, 0.0)
        prophet_df[macro_name] = float(val) if val is not None else 0.0

    # Construir modelo Prophet
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, interval_width=interval_width)

    # Añadir regressors técnicos
    for reg in ['rsi', 'macd', 'obv', 'Volume']:
        m.add_regressor(reg, standardize=True, mode='additive')

    # Añadir regressors macroeconómicos
    for macro_name in ['irx', 'dxy', 'oil', 'sp500']:
        m.add_regressor(macro_name, standardize=True, mode='additive')

    # Entrenamiento
    m.fit(prophet_df)

    # Dataframe futuro
    future = m.make_future_dataframe(periods=days_ahead)
    last_vals = prophet_df[['rsi', 'macd', 'obv', 'Volume']].iloc[-1].to_dict()
    for r in ['rsi', 'macd', 'obv', 'Volume']:
        future[r] = last_vals.get(r, 0)

    # Macros constantes en el futuro
    for macro_name in ['irx', 'dxy', 'oil', 'sp500']:
        val = macro_current.get(macro_name, 0.0)
        future[macro_name] = float(val) if val is not None else 0.0

    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds').tail(days_ahead)


# -------------------------------------------------------
# 10) LSTM robusto (entrena sobre Close_smoothed)
# -------------------------------------------------------
def lstm_forecast(df: pd.DataFrame, days_ahead: int, epochs: int = 60) -> pd.DataFrame:
    """
    Entrena LSTM sobre Close_smoothed y genera 'days_ahead' predicciones iterativas.
    Si TF no está disponible devuelve NaNs (Prophet hace fallback).
    """
    if Sequential is None:
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
        return pd.DataFrame({'ds': future_dates, 'yhat': [np.nan]*len(future_dates)}).set_index('ds')

    series = df[['Close_smoothed']].values.astype('float32')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X); y = np.array(y)
    if X.size == 0:
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
        return pd.DataFrame({'ds': future_dates, 'yhat': [np.nan]*len(future_dates)}).set_index('ds')

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Modelo con Input(...) para evitar warnings de Keras
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    callbacks = []
    if EarlyStopping is not None:
        callbacks.append(EarlyStopping(monitor='loss', patience=6, restore_best_weights=True))

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=callbacks)

    last_window = scaled[-look_back:].reshape((1, look_back, 1))
    preds_scaled = []
    for _ in range(days_ahead):
        p = model.predict(last_window, verbose=0)[0,0]
        preds_scaled.append(p)
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0, -1, 0] = p

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
    return pd.DataFrame({'ds': future_dates, 'yhat': preds}).set_index('ds')

# -------------------------------------------------------
# 11) SENTIMIENTO (NewsAPI + VADER)
# -------------------------------------------------------
def sentimiento_mercado(ticker: str) -> float:
    """
    Devuelve compound score medio [-1,1] usando NewsAPI y VADER.
    Si no está disponible devuelve 0.0 (neutral).
    """
    if not SENTIMIENTO_ACTIVO:
        return 0.0
    try:
        news = newsapi.get_everything(q=ticker, language='en', page_size=25)
        scores = []
        for art in news.get('articles', []):
            txt = (art.get('title') or '') + ". " + (art.get('description') or '')
            scores.append(analyzer.polarity_scores(txt)['compound'])
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

# -------------------------------------------------------
# 12) RECOMENDACIÓN: función cuantificada (explicación incluida)
# -------------------------------------------------------
def compute_recommendation_score(ma50: float, ma200: float, preds: dict, precio_actual: float,
                                 rsi: float, adx: float, sentiment_score: float,
                                 macro: dict) -> Tuple[float, str]:
    """
    Calcula una puntuación -1..1 y una etiqueta textual según:
     - Tendencia: ma50 vs ma200
     - Proyección: cambios relativos medios en 30/90/180 (ponderados)
     - RSI: sobreventa/compra
     - ADX: fuerza de tendencia
     - Sentimiento: NewsAPI
     - Macro: efecto agregado (tasas, dxy, oil, sp500)
    Se devuelven (score, label).
    """

    # (A) Tendencia técnica (MA50 vs MA200)
    trend_score = 1.0 if ma50 > ma200 else -1.0

    # (B) Proyección IA: promedio de cambios relativos (30/90/180)
    proj_changes = []
    for d in (30,90,180):
        p = preds.get(d, np.nan)
        if not (p is None) and not math.isnan(p) and precio_actual != 0:
            proj_changes.append((p - precio_actual) / precio_actual)
    proj_score = np.mean(proj_changes) if proj_changes else 0.0
    proj_score = max(min(proj_score, 1.0), -1.0)

    # (C) RSI -> prefer low RSI (oversold positive), high RSI negative
    if math.isnan(rsi):
        rsi_score = 0.0
    else:
        if rsi < 30:
            rsi_score = 0.8
        elif rsi < 50:
            rsi_score = 0.3
        elif rsi < 70:
            rsi_score = -0.1
        else:
            rsi_score = -0.6

    # (D) ADX -> fuerza de tendencia
    if math.isnan(adx):
        adx_score = 0.0
    else:
        if adx > 25:
            adx_score = 0.4
        elif adx > 20:
            adx_score = 0.2
        else:
            adx_score = 0.0

    # (E) Sentiment
    s = float(sentiment_score)
    if s > 0.2:
        sent_s = 0.6
    elif s > 0.05:
        sent_s = 0.2
    elif s < -0.2:
        sent_s = -0.6
    elif s < -0.05:
        sent_s = -0.2
    else:
        sent_s = 0.0

    # (F) Macro aggregated effect: transform each macro % change into -1..1 small signals
    # We scale roughly: +5% in oil or sp500 -> +1 signal, -5% -> -1 (cap)
    def scale_macro(val, cap=5.0):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0
        return max(min(val / cap, 1.0), -1.0)

    macro_irx = scale_macro(macro.get('irx', 0.0), cap=1.0) * -1.0  # tasa al alza es negativo para equities
    macro_dxy = scale_macro(macro.get('dxy', 0.0), cap=2.0) * -0.3  # dólar fuerte puede ser negativo
    macro_oil = scale_macro(macro.get('oil', 0.0), cap=5.0) * 0.2
    macro_sp = scale_macro(macro.get('sp500', 0.0), cap=5.0) * 0.9
    macro_score = np.mean([macro_irx, macro_dxy, macro_oil, macro_sp])

    # Ponderaciones finales (ajustables)
    w_trend = 0.20
    w_proj = 0.30
    w_rsi = 0.10
    w_adx = 0.10
    w_sent = 0.15
    w_macro = 0.15

    score = (w_trend*trend_score +
             w_proj*proj_score +
             w_rsi*rsi_score +
             w_adx*adx_score +
             w_sent*sent_s +
             w_macro*macro_score)

    # normalizar
    score = max(min(score, 1.0), -1.0)

    # etiquetas
    if score >= 0.4:
        label = "🟢 Comprar"
    elif score >= 0.15:
        label = "🟡 Mantener"
    elif score >= -0.15:
        label = "⚪ Neutral"
    elif score >= -0.4:
        label = "🟠 Vender Parcial"
    else:
        label = "🔴 Vender"

    return score, label

# -------------------------------------------------------
# 13) IMPRESIÓN DE LA TABLA FINAL (orden y formato)
# -------------------------------------------------------
def mostrar_tabla_final(rows: List[Tuple[Any, ...]]):
    """
    rows: lista de tuplas:
     (Ticker, Tendencia, PrecioActual, P30, P90, P180, SentimientoText, RecomendacionText)
    Orden y formato según especificado por el usuario.
    """
    headers = ["Ticker","Tendencia","Precio Actual","30d Precio","90d Precio","180d Precio","Sentimiento","Recomendación"]
    col_widths = [len(h) for h in headers]
    # calcular ancho real por columna (ignorando ANSI)
    for r in rows:
        for i, cell in enumerate(r):
            w = col_display_width(str(cell))
            if w > col_widths[i]:
                col_widths[i] = w

    sep = '+' + '+'.join(['-'*(w+2) for w in col_widths]) + '+'
    print(sep)
    print('| ' + ' | '.join([headers[i].ljust(col_widths[i]) for i in range(len(headers))]) + ' |')
    print(sep)
    for r in rows:
        ticker, trend, precio_actual, p30, p90, p180, sent_txt, rec_txt = r

        def price_with_symbol(pred):
            try:
                p = float(pred)
                base = float(precio_actual)
                sym = SYMBOL_UP if p > base else (SYMBOL_DOWN if p < base else "")
                if sym == SYMBOL_UP:
                    return f"{Fore.GREEN}{p:.2f} {sym}{Style.RESET_ALL}"
                elif sym == SYMBOL_DOWN:
                    return f"{Fore.RED}{p:.2f} {sym}{Style.RESET_ALL}"
                else:
                    return f"{p:.2f}"
            except Exception:
                return str(pred)

        cells = [
            str(ticker),
            str(trend),
            f"{float(precio_actual):.2f}",
            price_with_symbol(p30),
            price_with_symbol(p90),
            price_with_symbol(p180),
            str(sent_txt),
            str(rec_txt)
        ]
        formatted = '| ' + ' | '.join([format_cell_with_ansi(cells[i], col_widths[i]) for i in range(len(cells))]) + ' |'
        print(formatted)
    print(sep)

# -------------------------------------------------------
# 14) FLUJO PRINCIPAL: analizar un ticker completo y mostrar macro/interps
# -------------------------------------------------------
def analyze_ticker(ticker: str, convert_to_mxn: bool = False, usd_to_mxn: Optional[float] = None) -> Tuple[Any, ...]:
    """
    Analiza un ticker y devuelve la tupla para la fila de la tabla final.
    También imprime el resumen macro y la interpretación antes de la predicción individual.
    """
    ticker_clean = ticker.strip().upper()
    yf_ticker = ticker_clean.replace("BMV:", "") + ".MX" if ticker_clean.upper().startswith("BMV:") else ticker_clean

    # spinner
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner_start, args=(f"Analizando {yf_ticker}", stop_event))
    spinner_thread.daemon = True
    spinner_thread.start()

    try:
        # descargar macro y mostrar interpretación (macro por ticker, se puede reutilizar)
        macro = download_macro()
        # download data for ticker
        df = download_data(yf_ticker, period="3y")
        df = add_technical_indicators(df)
        last = df.tail(1)
        if last.empty:
            raise ValueError("No hay datos recientes para este ticker")

        # valores base
        precio_actual = float(last['Close'].iloc[0])
        if convert_to_mxn and usd_to_mxn:
            precio_actual = precio_actual * usd_to_mxn

        ma50 = float(last['ma50'].iloc[0]) if 'ma50' in last.columns else np.nan
        ma200 = float(last['ma200'].iloc[0]) if 'ma200' in last.columns else np.nan
        tendencia = "alcista 📈" if ma50 > ma200 else "bajista 📉"
        rsi = float(last['rsi'].iloc[0]) if 'rsi' in last.columns else np.nan
        adx = float(last['adx'].iloc[0]) if 'adx' in last.columns else np.nan

        # Detener spinner temporalmente para imprimir macro
        stop_event.set()
        spinner_thread.join(timeout=0.2)

        # IMPRIMIR resumen macro e interpretación
        print("\n📊 Factores macroeconómicos recientes (cambios % últimos ~30d):")
        for line in interpret_macro(macro):
            print(line)
        # breve interpretación heurística
        print("\n🧠 Interpretación macroeconómica:")
        # ejemplo simple de interpretación basada en macro
        irx = macro.get('irx', np.nan)
        if not math.isnan(irx) and irx > 0.2:
            print("  • Tasa en aumento rápido → presión bajista en renta variable.")
        elif not math.isnan(irx) and irx < -0.2:
            print("  • Tasa cayendo → ambiente favorable para acciones.")
        else:
            print("  • Tasa estable/moderada → efecto neutral.")

        dxy = macro.get('dxy', np.nan)
        if not math.isnan(dxy) and dxy > 1.0:
            print("  • Dólar subiendo → presión sobre empresas exportadoras.")
        elif not math.isnan(dxy) and dxy < -1.0:
            print("  • Dólar debilitado → favorable para exportadoras.")
        else:
            print("  • Dólar estable/moderado → efecto neutral.")

        oil = macro.get('oil', np.nan)
        if not math.isnan(oil) and oil > 3.0:
            print("  • Petróleo al alza → indica actividad económica; positivo para sectores energía/materiales.")
        else:
            print("  • Petróleo estable/moderado → efecto mixto.")

        sp = macro.get('sp500', np.nan)
        if not math.isnan(sp) and sp > 1.0:
            print("  • S&P500 al alza → apetito por riesgo general.")
        else:
            print("  • S&P500 estable/bajo → mercado cauto.")

        # Reiniciar spinner para pronósticos (visual)
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner_start, args=(f"Generando pronósticos para {yf_ticker}", stop_event))
        spinner_thread.daemon = True
        spinner_thread.start()

        # Sentimiento
        sent_score = sentimiento_mercado(yf_ticker)
        if sent_score > 0.2:
            sent_txt = "👍 Positivo"
        elif sent_score < -0.2:
            sent_txt = "👎 Negativo"
        else:
            sent_txt = "😐 Neutral"

        # Forecast: Prophet (con regressors) y LSTM, ensemble 0.7/0.3 con suavizado
        preds = {}
        for days in (30,90,180):
            pf = prophet_forecast_with_regressors(df, macro_current=macro, days_ahead=days, interval_width=0.8)
            lf = lstm_forecast(df, days_ahead=days, epochs=60)
            pf_y = pf['yhat'].values if 'yhat' in pf.columns else np.array([np.nan]*days)
            lf_y = lf['yhat'].values if 'yhat' in lf.columns else np.array([np.nan]*days)

            # si LSTM no funciona (NaNs) usar solo Prophet
            if np.all(np.isnan(lf_y)):
                ensemble_val = pf_y[-1]
            else:
                # alinear tamaños (tomar últimas n)
                n = min(len(pf_y), len(lf_y))
                arr = (0.7 * pf_y[-n:] + 0.3 * lf_y[-n:])
                # suavizar con media móvil (reduce saltos)
                ensemble_smoothed = pd.Series(arr).rolling(window=5, min_periods=1).mean().values
                ensemble_val = ensemble_smoothed[-1]
            val = float(ensemble_val)
            if convert_to_mxn and usd_to_mxn:
                val = val * usd_to_mxn
            preds[days] = val

        # detener spinner
        stop_event.set()
        spinner_thread.join(timeout=0.2)

        # calcular recommendation score
        score, rec_label = compute_recommendation_score(ma50, ma200, preds, precio_actual, rsi, adx, sent_score, macro)

        # we return the tuple used for table
        return (yf_ticker, tendencia, precio_actual, preds[30], preds[90], preds[180], sent_txt, rec_label)

    except Exception as e:
        # asegurar que spinner pare
        try:
            stop_event.set()
            spinner_thread.join(timeout=0.2)
        except Exception:
            pass
        print(f"\n  ⚠️ Error analizando {yf_ticker}: {e}")
        return (yf_ticker, "ERROR", 0.0, np.nan, np.nan, np.nan, "N/D", "Error")

# -------------------------------------------------------
# 15) PUNTO DE ENTRADA: interacción con usuario
# -------------------------------------------------------
def main():
    print("Invest_Predictions_TEST2 - Analizador avanzado (Prophet + LSTM + Macro + Sentiment)\n")
    entrada = input("Introduce tickers (comas o espacios) (ej. AAPL MSFT SPY): ").strip()
    if not entrada:
        print("No ingresaste tickers. Saliendo.")
        return
    # aceptar comas o espacios como separadores
    tickers = [t.strip().upper() for t in re.split(r'[,\s]+', entrada) if t.strip()]

    conv = input("Convertir precios a MXN? (s/N): ").strip().lower() == 's'
    usd_to_mxn = None
    if conv:
        print("Obteniendo tipo de cambio USD->MXN ...")
        usd_to_mxn = get_usd_to_mxn()
        if usd_to_mxn is None:
            print("  ⚠️ No se pudo obtener tipo de cambio. Se continuará en USD.")
            conv = False

    results = []
    # analizamos cada ticker
    for t in tickers:
        res = analyze_ticker(t, convert_to_mxn=conv, usd_to_mxn=usd_to_mxn)
        results.append(res)

    # mostrar tabla final
    if results:
        print("\n\n📊 Resumen final de todas las acciones / ETFs")
        mostrar_tabla_final(results)
    else:
        print("No se generaron resultados.")

if __name__ == "__main__":
    main()
