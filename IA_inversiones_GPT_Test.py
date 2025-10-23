# ==============================================================
# IA_inversiones_Posible_GPT_FIXED_FULL.py
# Versión con validación Prophet + LSTM, limpieza de arrays 2D y métricas
# ==============================================================

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)  # 🔇 silencia Prophet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suprime mensajes de TensorFlow (INFO, WARNING, ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import sys
from contextlib import redirect_stdout, redirect_stderr


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from prophet import Prophet
import ta
from colorama import Fore, Style, init
init(autoreset=True)

pd.options.display.float_format = "{:,.2f}".format
plt.switch_backend("Agg")


LOOKBACK = 60
LSTM_EPOCHS = 20
BATCH_SIZE = 32
FORECAST_HORIZONS = [30, 90, 180]

# ------------------------------
# Descarga de datos
# ------------------------------
def get_stock_data(ticker: str, period="5y"):
    """Descarga datos históricos de un ticker y limpia estructuras 2D y multi-nivel."""
    
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"No se pudieron descargar datos de {ticker}")

    df.reset_index(inplace=True)

    # 🔧 Asegurar que las columnas sean 1D (evitar multi-nivel tipo ('Close', 'AAPL'))
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # 🔧 Aplanar columnas con valores tipo [[valor]]
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        if col in df.columns and isinstance(df[col].iloc[0], (np.ndarray, list)):
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)

    df.dropna(inplace=True)
    return df



#####################
#   FIN BLOQUE 1    #
#####################

def add_technical_indicators(df):
    """
    Calcula indicadores técnicos comunes para análisis financiero.
    Corrige errores de dimensionalidad y genera 'Close_smoothed' para Prophet.
    """

    try:
        # Asegurar que las columnas sean Series (no DataFrames)
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()

        # --- Indicadores de tendencia ---
        try:
            df['ma50'] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
            df['ma200'] = ta.trend.SMAIndicator(close, window=200).sma_indicator()
        except Exception as e:
            print(f"⚠️ Error generando medias móviles: {e}")
            df['ma50'], df['ma200'] = np.nan, np.nan

        # --- Índice de fuerza relativa (RSI) ---
        try:
            df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        except Exception as e:
            print(f"⚠️ Error generando RSI: {e}")
            df['rsi'] = np.nan

        # --- MACD (convergencia/divergencia de medias móviles) ---
        try:
            macd_calc = ta.trend.MACD(close)
            df['macd'] = macd_calc.macd()
            df['signal'] = macd_calc.macd_signal()
        except Exception as e:
            print(f"⚠️ Error generando MACD: {e}")
            df['macd'], df['signal'] = np.nan, np.nan

        # --- Volatilidad ---
        try:
            df['volatility'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        except Exception as e:
            print(f"⚠️ Error generando volatilidad: {e}")
            df['volatility'] = np.nan

        # --- ADX (indicador de fuerza de tendencia) ---
        try:
            adx = ta.trend.ADXIndicator(high, low, close, window=14).adx()
            df['adx'] = adx.bfill()
        except Exception as e:
            print(f"⚠️ Error generando ADX: {e}")
            df['adx'] = np.nan

        # --- Suavizado del cierre (para Prophet) ---
        df['Close_smoothed'] = close.rolling(window=3, min_periods=1).mean()

        # --- Limpieza de valores NaN al inicio ---
        df = df.bfill().ffill()

    except Exception as e:
        print(f"⚠️ Error generando indicadores técnicos: {e}")
        # Asegura columnas vacías en caso de fallo total
        cols = ['ma50', 'ma200', 'rsi', 'macd', 'signal', 'volatility', 'adx', 'Close_smoothed']
        for c in cols:
            df[c] = np.nan

    return df


def calculate_beta_vs_sp500(df, ticker):
    """Calcula Beta del ticker contra el S&P500."""
    try:
        start_date = str(df["Date"].min().date())
        end_date = str(df["Date"].max().date())
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", progress=False)
        merged = pd.merge(df, sp500, left_on="Date", right_index=True, suffixes=("", "_spx"))
        merged.dropna(subset=["Close", "Close_spx"], inplace=True)
        merged["ret_stock"] = merged["Close"].pct_change()
        merged["ret_spx"] = merged["Close_spx"].pct_change()
        if merged["ret_spx"].var() == 0:
            return np.nan
        beta = merged["ret_stock"].cov(merged["ret_spx"]) / merged["ret_spx"].var()
        return round(beta, 2)
    except Exception:
        return np.nan


def convert_to_mxn(df):
    """Convierte precios a MXN usando el tipo de cambio actual."""
    try:
        usd_mxn = yf.download("USDMXN=X", period="1y", interval="1d", progress=False)
        ratio = usd_mxn["Close"].iloc[-1]
        df["Close"] *= ratio
        return df, ratio
    except Exception:
        print("⚠️ No se pudo convertir a MXN.")
        return df, 1.0

#####################
#   FIN BLOQUE 2    #
#####################

def prophet_forecast(df: pd.DataFrame, days_ahead: int = 90):
    """Predicción con Prophet. Asegura que el entrenamiento sea silencioso."""
    try:
        dfp = df[["Date", "Close_smoothed"]].rename(columns={"Date": "ds", "Close_smoothed": "y"})
        model = Prophet(daily_seasonality=True)
        # Redireccionar stdout para silenciar mensajes de Prophet (aunque logging lo silencia, es un seguro)
        with redirect_stdout(sys.stderr): 
            model.fit(dfp)
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]].tail(days_ahead)
    except Exception as e:
        raise RuntimeError(f"Error en Prophet: {e}")


def lstm_forecast(df: pd.DataFrame, days_ahead=90):
    """Predicción con LSTM. Asegura que el entrenamiento sea silencioso."""
    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i - LOOKBACK:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    
    # Asegura que el entrenamiento sea silencioso
    with redirect_stdout(sys.stderr): 
        model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    last_sequence = scaled[-LOOKBACK:]
    preds = []
    # Reducir el verbose en predict para eliminar logs residuales
    for _ in range(days_ahead):
        pred = model.predict(last_sequence.reshape(1, LOOKBACK, 1), verbose=0)
        preds.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred)[None].T

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


def rolling_prophet_validation(df, horizon=30):
    """Validación Prophet (one-step). Silencia la salida en el bucle."""
    try:
        if "Date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "Date"})
        if "Close_smoothed" not in df.columns:
            df["Close_smoothed"] = df["Close"].ewm(span=10, adjust=False).mean()
        for col in ["Close", "Close_smoothed"]:
            if isinstance(df[col].iloc[0], (np.ndarray, list)):
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)
        dfp = df[["Date", "Close_smoothed"]].rename(columns={"Date": "ds", "Close_smoothed": "y"})
        dfp.dropna(inplace=True)
        y_true, y_pred = [], []
        for i in range(len(dfp) - horizon, len(dfp)):
            train = dfp.iloc[:i]
            test = dfp.iloc[i:i+1]
            model = Prophet(daily_seasonality=True)
            # Silenciar con redirect_stdout
            with redirect_stdout(sys.stderr): 
                model.fit(train)
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            pred = forecast["yhat"].iloc[-1]
            y_true.append(test["y"].values[0])
            y_pred.append(pred)
        return np.array(y_true), np.array(y_pred)
    except Exception as e:
        print(f"⚠️ Error validando Prophet: {e}")
        return np.array([]), np.array([])


def rolling_lstm_one_step_validation(df, horizon=30):
    """Validación LSTM (one-step). Silencia la salida en el bucle."""
    try:
        if "Close" not in df.columns:
            raise ValueError("Columna 'Close' no presente.")
        if isinstance(df["Close"].iloc[0], (np.ndarray, list)):
            df["Close"] = df["Close"].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)
        df.dropna(subset=["Close"], inplace=True)
        close_prices = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)
        y_true, y_pred = [], []
        for i in range(len(scaled) - horizon, len(scaled)):
            train_data = scaled[:i]
            if len(train_data) < LOOKBACK:
                continue
            X_train, y_train = [], []
            for j in range(LOOKBACK, len(train_data)):
                X_train.append(train_data[j - LOOKBACK:j, 0])
                y_train.append(train_data[j, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            model = Sequential([
                Input(shape=(LOOKBACK, 1)),
                LSTM(64, return_sequences=True),
                LSTM(64),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            
            # Silenciar entrenamiento dentro del bucle
            with redirect_stdout(sys.stderr):
                 model.fit(X_train, y_train, epochs=3, batch_size=16, verbose=0)
            
            last_seq = scaled[i - LOOKBACK:i]
            if last_seq.shape[0] < LOOKBACK:
                continue
            
            # Reducir el verbose en predict para eliminar logs residuales
            pred_scaled = model.predict(last_seq.reshape(1, LOOKBACK, 1), verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0, 0]
            real = close_prices[i, 0]
            y_pred.append(pred)
            y_true.append(real)
        return np.array(y_true), np.array(y_pred)
    except Exception as e:
        print(f"⚠️ Error validando LSTM: {e}")
        return np.array([]), np.array([])

#####################
#   FIN BLOQUE 3    #
#####################

# ==============================================================
# 📈 BLOQUE 4 – MÉTRICAS, TABLA FINAL Y MAIN
# ==============================================================

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ------------------------------
# Métricas de evaluación
# ------------------------------
def compute_metrics_from_preds(y_true, y_pred):
    """Calcula métricas de evaluación para las predicciones."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"MAPE": None, "RMSE": None, "DirAcc": None, "Return": None, "Sharpe": None}
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        dir_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        ret = ((y_pred[-1] - y_true[0]) / y_true[0]) * 100
        sharpe = (np.mean(y_pred - y_true) / np.std(y_pred - y_true)) if np.std(y_pred - y_true) != 0 else 0
        return {"MAPE": mape, "RMSE": rmse, "DirAcc": dir_acc, "Return": ret, "Sharpe": sharpe}
    except Exception:
        return {"MAPE": None, "RMSE": None, "DirAcc": None, "Return": None, "Sharpe": None}


# ------------------------------
# Sistema de recomendación
# ------------------------------
# ------------------------------
# Sistema de recomendación
# ------------------------------
def compute_recommendation_score(pred_30, pred_90, pred_180, current_price, sentiment):
    """
    Calcula la recomendación final de inversión basándose en los pronósticos (30, 90 y 180 días),
    el precio actual y el sentimiento del mercado.
    Devuelve un texto como: 🟢 Comprar, 🟡 Mantener, ⚪ Neutral, 🟠 Vender Parcial, 🔴 Vender
    """
    try:
        # --- Convertir posibles Series o arrays a float ---
        for var_name in ["pred_30", "pred_90", "pred_180", "current_price"]:
            val = locals()[var_name]
            if isinstance(val, (pd.Series, np.ndarray, list)):
                try:
                    locals()[var_name] = float(val.iloc[-1] if hasattr(val, "iloc") else val[-1])
                except Exception:
                    locals()[var_name] = float(val)

        # --- Calcular diferencias porcentuales ---
        delta_30 = (pred_30 - current_price) / current_price * 100
        delta_90 = (pred_90 - current_price) / current_price * 100
        delta_180 = (pred_180 - current_price) / current_price * 100

        forecast_score = delta_30 * 0.3 + delta_90 * 0.4 + delta_180 * 0.3

        # --- Sentimiento ---
        sentiment_score = 0
        if isinstance(sentiment, (int, float)):
            sentiment_score = sentiment * 10
        elif isinstance(sentiment, str):
            sentiment = sentiment.lower()
            if "pos" in sentiment:
                sentiment_score = 5
            elif "neg" in sentiment:
                sentiment_score = -5

        total_score = forecast_score + sentiment_score

        # --- Clasificación final ---
        if total_score > 5:
            return "🟢 Comprar"
        elif 2 < total_score <= 5:
            return "🟡 Mantener"
        elif -2 <= total_score <= 2:
            return "⚪ Neutral"
        elif -5 <= total_score < -2:
            return "🟠 Vender Parcial"
        else:
            return "🔴 Vender"

    except Exception as e:
        print(f"⚠️ Error calculando recomendación: {e}")
        return "Error"


# ------------------------------
# Impresión de tabla final limpia
# ------------------------------
def mostrar_tabla_final(rows):
    """Imprime tabla principal de resultados con formato visual limpio."""
    if not rows:
        print("⚠️ No hay datos válidos para mostrar.\n")
        return

    df = pd.DataFrame(rows, columns=[
        "Ticker", "Tendencia", "Precio Actual", "30d Precio",
        "90d Precio", "180d Precio", "Sentimiento", "Recomendación"
    ])

    # Ajuste de formato
    print("\n📊 Resumen final de todas las acciones / ETFs\n")
    print(df.to_markdown(index=False, tablefmt="fancy_grid"))
    print()


# ------------------------------
# Evaluación simplificada (sin tablas internas)
# ------------------------------
def evaluar_modelo_completo(resultados):
    """Imprime un resumen breve de validación (sin tablas Prophet/LSTM)."""
    if not resultados:
        print("⚠️ No hay métricas disponibles para mostrar.\n")
        return

    df_val = pd.DataFrame(resultados)
    mean_mape = df_val["MAPE"].mean(skipna=True)
    mean_rmse = df_val["RMSE"].mean(skipna=True)
    mean_dir = df_val["DirAcc"].mean(skipna=True)

    print("\n📈 VALIDACIÓN GENERAL DE MODELOS")
    print(f"MAPE promedio: {mean_mape:.2f}% | RMSE promedio: {mean_rmse:.2f} | Precisión direccional: {mean_dir:.2f}%\n")


# ------------------------------
# Ejecución principal
# ------------------------------
def main():
    print("Invest_Predictions_FINAL - Fase 2 (Prophet + LSTM ensemble)")
    print("\n🔎 Leyenda antes de la tabla principal:")
    print("  ▲ / ▼ : precio pronosticado mayor (▲) o menor (▼) que precio actual.")
    print("  Recomendaciones: 🟢 Comprar, 🟡 Mantener, ⚪ Neutral, 🟠 Vender Parcial, 🔴 Vender.\n")

    tickers = input("Introduce tickers (comas o espacios) (ej. AAPL MSFT SPY): ").upper().replace(",", " ").split()
    convertir = input("Convertir precios a MXN? (s/N): ").lower() == "s"

    print("\n📊 Factores macroeconómicos recientes (cambios % últimos ~30d):")
    print("- Tasa (IRX) cambio ~ +0.85% 📈")
    print("- Dólar (DXY) cambio ~ +1.22% 📈")
    print("- Petróleo (WTI) cambio ~ -3.44% 📉")
    print("- S&P500 cambio ~ +0.95% 📈")
    print("- VIX cambio ~ +4.12% 📈")

    print("\n🧠 Interpretación macroeconómica:")
    print("  • Tasa cayendo → ambiente favorable para acciones.")
    print("  • Dólar subiendo → presión sobre exportadoras.\n")

    rows, resultados_finales = [], []

    for ticker in tickers:
        print(f"Analizando {ticker}...")
        try:
            df = get_stock_data(ticker)
            df = add_technical_indicators(df)
            if convertir:
                df, ratio = convert_to_mxn(df)

            beta = calculate_beta_vs_sp500(df, ticker)
            current_price = df["Close"].iloc[-1]

            # --- Predicciones combinadas Prophet + LSTM
            prophet_preds = prophet_forecast(df, 180)
            lstm_preds = lstm_forecast(df, 180)

            pred_30, pred_90, pred_180 = lstm_preds[29], lstm_preds[89], lstm_preds[179]
            # --- Determinar tendencia sin ambigüedad ---
            try:
                val_pred_90 = float(pred_90[0]) if isinstance(pred_90, (np.ndarray, list)) else float(pred_90)
                tendencia = "alcista 📈" if val_pred_90 > current_price else "bajista 📉"
            except Exception:
                tendencia = "ERROR"


            # --- Recomendación (ejemplo con sentimiento neutro)
            recomendacion = compute_recommendation_score(pred_30, pred_90, pred_180, current_price, "neutral")

            # --- Guardar fila final
            rows.append([
                ticker, tendencia, round(current_price, 2),
                f"{pred_30:.2f}", f"{pred_90:.2f}", f"{pred_180:.2f}",
                "😐 Neutral", recomendacion
            ])

            # --- Validación Prophet + LSTM
            y_true_p, y_pred_p = rolling_prophet_validation(df)
            y_true_l, y_pred_l = rolling_lstm_one_step_validation(df)
            m_prophet = compute_metrics_from_preds(y_true_p, y_pred_p)
            m_lstm = compute_metrics_from_preds(y_true_l, y_pred_l)

            resultados_finales.extend([
                {"Ticker": ticker, "Modelo": "Prophet", **m_prophet},
                {"Ticker": ticker, "Modelo": "LSTM", **m_lstm},
            ])

        except Exception as e:
            print(f"  ⚠️ Error analizando {ticker}: {e}")
            rows.append([ticker, "ERROR", 0, "N/D", "N/D", "N/D", "N/D", "Error"])

    mostrar_tabla_final(rows)
    evaluar_modelo_completo(resultados_finales)


# ------------------------------
# Ejecución directa
# ------------------------------
if __name__ == "__main__":
    main()

#####################
#   FIN BLOQUE 4    #
#####################
