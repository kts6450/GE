import argparse
import json
import os
import random

# Metal(GPU) / XLA 컴파일 hang 방지: CPU 전용 + Eager 모드 강제
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import tensorflow as tf

# Apple Silicon Metal GPU 비활성화 + Eager 실행 강제 (XLA / tf.data hang 방지)
# keras import 전에 설정해야 적용됨
tf.config.set_visible_devices([], "GPU")
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

import joblib
import matplotlib
matplotlib.use("Agg")  # macOS GUI 충돌 방지: 비대화형 백엔드 강제
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    DENSE_MODEL_PATH,
    EPOCHS,
    FEATURE_COLUMNS_PATH,
    FIGURES_DIR,
    LSTM_LOOKBACK_DAYS,
    LSTM_MODEL_PATH,
    METRICS_PATH,
    RANDOM_SEED,
    TARGET_SCALER_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_directories,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_feature_columns() -> list[str]:
    return json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))


def build_dense_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ],
        name="ge_dense_regression",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
        run_eagerly=True,
    )
    return model


def build_lstm_model(lookback_days: int, feature_count: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(lookback_days, feature_count)),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ],
        name="ge_lstm_regression",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
        run_eagerly=True,
    )
    return model


def create_lstm_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for i in range(lookback_days, len(features)):
        x_values.append(features[i - lookback_days : i])
        y_values.append(targets[i])
    return np.array(x_values), np.array(y_values)


def inverse_target(values: np.ndarray) -> np.ndarray:
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    return target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def plot_training_history(history: keras.callbacks.History, output_path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_predictions(
    dates: pd.Series,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual Close", linewidth=2)
    for label, values in predictions.items():
        plt.plot(dates.iloc[-len(values) :], values, label=label, alpha=0.8)
    plt.title("GE Actual vs Predicted Close")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_dense_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[keras.Model, keras.callbacks.History, np.ndarray]:
    x_train = train_data[feature_columns].to_numpy(dtype="float32")
    y_train = train_data[target_column].to_numpy(dtype="float32")
    x_test = test_data[feature_columns].to_numpy(dtype="float32")

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )

    model = build_dense_model(input_dim=len(feature_columns))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, factor=0.5),
    ]

    history = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(DENSE_MODEL_PATH)
    predictions_scaled = model.predict(x_test, verbose=0).reshape(-1)
    predictions = inverse_target(predictions_scaled)
    return model, history, predictions


def train_lstm_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback_days: int = LSTM_LOOKBACK_DAYS,
) -> tuple[keras.Model, keras.callbacks.History, np.ndarray]:
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    features = combined_data[feature_columns].to_numpy(dtype="float32")
    targets = combined_data[target_column].to_numpy(dtype="float32")

    x_all, y_all = create_lstm_sequences(features, targets, lookback_days)
    train_sequence_count = max(len(train_data) - lookback_days, 0)

    x_train = x_all[:train_sequence_count]
    y_train = y_all[:train_sequence_count]
    x_test = x_all[train_sequence_count:]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )

    model = build_lstm_model(lookback_days=lookback_days, feature_count=len(feature_columns))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, factor=0.5),
    ]

    history = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(LSTM_MODEL_PATH)
    predictions_scaled = model.predict(x_test, verbose=0).reshape(-1)
    predictions = inverse_target(predictions_scaled)
    return model, history, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TensorFlow regression models for GE stock data.")
    parser.add_argument("--skip-lstm", action="store_true", help="Train only the Dense regression model.")
    args = parser.parse_args()

    ensure_directories()
    set_seed()

    feature_columns = load_feature_columns()
    train_data = pd.read_csv(TRAIN_DATA_PATH, parse_dates=["Date"])
    test_data = pd.read_csv(TEST_DATA_PATH, parse_dates=["Date"])
    target_column = "target_return_next_day_scaled"

    print("\n=== Dense Regression 학습 (TensorFlow) ===")
    _, dense_history, dense_predictions = train_dense_model(
        train_data=train_data,
        test_data=test_data,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    actual = test_data["target_close_next_day"].to_numpy()
    current_close = test_data["current_close"].to_numpy()
    dense_predictions_close = current_close * (1 + dense_predictions)

    metrics = {
        "baseline_today_equals_tomorrow": calculate_metrics(actual, current_close),
        "dense_regression": calculate_metrics(actual, dense_predictions_close),
    }

    plot_training_history(dense_history, FIGURES_DIR / "dense_training_loss.png")

    prediction_series: dict[str, np.ndarray] = {
        "Baseline": current_close,
        "Dense Regression": dense_predictions_close,
    }

    if not args.skip_lstm:
        print("\n=== LSTM Regression 학습 (TensorFlow) ===")
        _, lstm_history, lstm_predictions = train_lstm_model(
            train_data=train_data,
            test_data=test_data,
            feature_columns=feature_columns,
            target_column=target_column,
        )
        lstm_actual = actual[-len(lstm_predictions) :]
        lstm_current_close = current_close[-len(lstm_predictions) :]
        lstm_predictions_close = lstm_current_close * (1 + lstm_predictions)
        metrics["lstm_regression"] = calculate_metrics(lstm_actual, lstm_predictions_close)
        plot_training_history(lstm_history, FIGURES_DIR / "lstm_training_loss.png")
        prediction_series["LSTM Regression"] = lstm_predictions_close

    plot_predictions(
        dates=test_data["Date"],
        actual=actual,
        predictions=prediction_series,
        output_path=FIGURES_DIR / "actual_vs_predicted.png",
    )

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
