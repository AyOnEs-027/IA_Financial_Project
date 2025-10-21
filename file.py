# -----------------------------
# AÑADIR / REEMPLAZAR: Nivel 1 mejoras (VIX + walk-forward + tuning)
# -----------------------------
from sklearn.metrics import mean_absolute_percentage_error

# 1) Incluir VIX en los macros (reemplaza MACRO_TICKERS existente o actualiza)
MACRO_TICKERS.update({
    'vix': '^VIX'   # índice de volatilidad (miedo) - muy útil como regresor / señal de riesgo
})

# 2) walk-forward validation para LSTM
def walk_forward_validation_lstm(df: pd.DataFrame, n_splits: int = 5, look_back: int = 60, epochs: int = 30, verbose: bool = False):
    """
    Realiza walk-forward validation en la serie Close_smoothed para evaluar el LSTM.
    - df: DataFrame con Close_smoothed (índice datetime)
    - n_splits: cuántas ventanas de validación consecutivas
    - look_back: ventana histórica para cada entrenamiento
    - epochs: epochs para entrenar en cada paso
    Devuelve: promedio MAPE sobre las splits.
    Nota: es relativamente costoso. Si TF no está disponible devuelve np.nan.
    """
    if Sequential is None:
        if verbose: print("TensorFlow no disponible -> no se puede validar LSTM.")
        return np.nan

    series = df[['Close_smoothed']].values.astype('float32').flatten()
    N = len(series)
    if N < (look_back + n_splits + 1):
        if verbose: print("Serie muy corta para walk-forward con los parámetros dados.")
        return np.nan

    # definir tamaño de cada test (aproximadamente)
    test_size = max(1, (N - look_back) // (n_splits + 1))
    mape_scores = []

    for i in range(n_splits):
        # entrenamiento hasta index: train_end
        train_end = look_back + i * test_size
        val_start = train_end
        val_end = train_end + test_size

        train_series = series[:train_end]
        val_series = series[val_start:val_end]

        # preparar X/y
        def create_xy(arr):
            X, y = [], []
            for j in range(look_back, len(arr)):
                X.append(arr[j-look_back:j])
                y.append(arr[j])
            X = np.array(X).reshape(-1, look_back, 1)
            y = np.array(y)
            return X, y

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_series.reshape(-1,1)).flatten()
        val_scaled = scaler.transform(val_series.reshape(-1,1)).flatten()

        # construir ventanas para train
        X_train, y_train = create_xy(train_scaled)
        if X_train.size == 0:
            if verbose: print("No hay suficientes datos para crear X_train en split", i)
            continue

        # modelo pequeño para validar
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],1)))
        model.add(LSTM(60, return_sequences=True))
        model.add(Dropout(0.15))
        model.add(LSTM(30))
        model.add(Dropout(0.15))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        cb = []
        if EarlyStopping is not None:
            cb = [EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)]

        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=cb)

        # predecir val con recorrido iterativo (usar última ventana del train y luego predecir test_size pasos)
        last_window = train_scaled[-look_back:].reshape((1, look_back, 1))
        preds_scaled = []
        for _ in range(len(val_scaled)):
            p = model.predict(last_window, verbose=0)[0,0]
            preds_scaled.append(p)
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, 0] = p

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
        # comparar preds con val_series (ya en original scale)
        if len(preds) != len(val_series):
            # fallback: cortar/ajustar
            L = min(len(preds), len(val_series))
            preds = preds[:L]
            true = val_series[:L]
        else:
            true = val_series

        # MAPE (evitar división por 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = mean_absolute_percentage_error(true, preds) if len(true)>0 else np.nan
        if verbose:
            print(f"split {i+1}/{n_splits}: MAPE = {mape:.4f}")
        mape_scores.append(mape)

    if len(mape_scores) == 0:
        return np.nan
    return float(np.nanmean(mape_scores))

# 3) Grid search simple para look_back / epochs (pequeño por defecto)
def tune_lstm_hyperparams(df: pd.DataFrame, look_back_list=[30,60], epochs_list=[20,40], n_splits=3, verbose=False):
    """
    Recorre combinaciones (look_back, epochs) y devuelve el mejor par según walk-forward (MAPE).
    - restringido por defecto para no explotar tiempo CPU.
    """
    best = None
    best_score = np.inf
    results = []
    for lb in look_back_list:
        for ep in epochs_list:
            if verbose: print(f"Evaluando look_back={lb}, epochs={ep} ...")
            try:
                mape = walk_forward_validation_lstm(df, n_splits=n_splits, look_back=lb, epochs=ep, verbose=verbose)
            except Exception as e:
                mape = np.nan
            results.append((lb, ep, mape))
            if not math.isnan(mape) and mape < best_score:
                best_score = mape
                best = (lb, ep)
    return best, best_score, results

# 4) Versión de lstm_forecast que acepta params (look_back) y permite reusar entrenamiento rápido
def lstm_forecast_with_params(df: pd.DataFrame, days_ahead: int, look_back: int = 60, epochs: int = 60):
    """
    Misma lógica de lstm_forecast pero parametrizable por look_back y epochs.
    (Recomendado para usar después de hallar mejores params con tune_lstm_hyperparams()).
    """
    # Reusar la implementación base (se asume que Sequential está definido)
    if Sequential is None:
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
        return pd.DataFrame({'ds': future_dates, 'yhat': [np.nan]*len(future_dates)}).set_index('ds')

    series = df[['Close_smoothed']].values.astype('float32')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    lb = int(look_back)
    X, y = [], []
    for i in range(lb, len(scaled)):
        X.append(scaled[i-lb:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X); y = np.array(y)
    if X.size == 0:
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
        return pd.DataFrame({'ds': future_dates, 'yhat': [np.nan]*len(future_dates)}).set_index('ds')

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Modelo ajustado
    model = Sequential()
    model.add(Input(shape=(X.shape[1],1)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    callbacks = []
    if EarlyStopping is not None:
        callbacks.append(EarlyStopping(monitor='loss', patience=6, restore_best_weights=True))

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=callbacks)

    last_window = scaled[-lb:].reshape((1, lb, 1))
    preds_scaled = []
    for _ in range(days_ahead):
        p = model.predict(last_window, verbose=0)[0,0]
        preds_scaled.append(p)
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0, -1, 0] = p

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
    return pd.DataFrame({'ds': future_dates, 'yhat': preds}).set_index('ds')

# 5) Integración sugerida en tu flujo principal:
#    - Define un flag arriba del script para habilitar tuning (por defecto False)
TUNE_LSTM = False   # cambiar a True solo si quieres que el script intente optimizar (tarda bastante)

# Ejemplo de uso en analyze_ticker (sustituir la llamada a lstm_forecast por esta lógica):
#
#    # antes de generar pronósticos, si TUNE_LSTM True, buscar mejores params (con costo)
#    if TUNE_LSTM:
#        best, best_score, allres = tune_lstm_hyperparams(df, look_back_list=[30,60], epochs_list=[20,40], n_splits=2, verbose=False)
#        if best is not None:
#            use_look_back, use_epochs = best
#        else:
#            use_look_back, use_epochs = 60, 60
#    else:
#        use_look_back, use_epochs = 60, 60
#
#    # dentro del loop de pronósticos:
#    lf = lstm_forecast_with_params(df, days_ahead=days, look_back=use_look_back, epochs=use_epochs)
#
# Nota: activa TUNE_LSTM = True solo si entiendes que el tiempo de ejecución aumentará.
# -----------------------------
