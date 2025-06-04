import logging
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Sequential:
    """Entrena un modelo LSTM con Keras.

    Args:
        X_train: Datos de entrenamiento (features) en formato 3D para LSTM.
        y_train: Variable objetivo.

    Returns:
        Modelo Keras entrenado.
    """
    model = Sequential([
    LSTM(60, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train,
          epochs=60, batch_size=32,
          validation_split=0.1, verbose=0)
    model.save("data/06_models/modelo.keras")

    return model


def evaluate_model(modelo: Sequential, scaler: MinMaxScaler, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Calcula métricas para evaluación de modelos de series de tiempo.

    Args:
        regressor: Modelo Keras entrenado.
        X_test: Datos de prueba (features), forma (samples, timesteps, features).
        y_test: Valores reales para la variable objetivo.

    Returns:
        Diccionario con métricas: MAE, RMSE, MAPE, SMAPE y R2.
    """
    y_pred_test_scaled = modelo.predict(X_test).flatten()
    min_v, max_v = scaler.data_min_[0], scaler.data_max_[0]
    y_test_inv      = y_test * (max_v - min_v) + min_v
    y_pred_test_inv = y_pred_test_scaled * (max_v - min_v) + min_v

    mae  = mean_absolute_error(y_test_inv, y_pred_test_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_test_inv)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Model evaluation metrics -- MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

