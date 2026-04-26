import argparse
import csv
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from collect_data import collect_stock_data
from config import (
    DENSE_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_SCALER_PATH,
    LSTM_LOOKBACK_DAYS,
    LSTM_MODEL_PATH,
    LATEST_PREDICTION_PATH,
    PREDICTION_HISTORY_PATH,
    START_DATE,
    TARGET_SCALER_PATH,
    TICKER,
    ensure_directories,
)
from indicators import add_technical_indicators


def load_feature_columns() -> list[str]:
    return json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))


def prepare_latest_features(raw_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data = add_technical_indicators(raw_data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if data.empty:
        raise RuntimeError("No usable rows after feature engineering.")

    return data


def predict_dense(data: pd.DataFrame, feature_columns: list[str]) -> float:
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    model = tf.keras.models.load_model(DENSE_MODEL_PATH)

    latest_features = data[feature_columns].tail(1)
    latest_scaled = feature_scaler.transform(latest_features)
    prediction_scaled = model.predict(latest_scaled, verbose=0)
    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

    return float(prediction)


def predict_lstm(data: pd.DataFrame, feature_columns: list[str]) -> float:
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    model = tf.keras.models.load_model(LSTM_MODEL_PATH)

    if len(data) < LSTM_LOOKBACK_DAYS:
        raise RuntimeError(f"LSTM prediction requires at least {LSTM_LOOKBACK_DAYS} processed rows.")

    latest_features = data[feature_columns].tail(LSTM_LOOKBACK_DAYS)
    latest_scaled = feature_scaler.transform(latest_features)
    sequence = np.expand_dims(latest_scaled, axis=0)
    prediction_scaled = model.predict(sequence, verbose=0)
    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

    return float(prediction)


def build_prediction_payload(
    data: pd.DataFrame,
    predicted_return: float,
    model_name: str,
) -> dict:
    latest = data.iloc[-1]
    current_close = float(latest["Close"])
    predicted_close = current_close * (1 + predicted_return)

    return {
        "ticker": TICKER,
        "model": model_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_market_date": str(pd.to_datetime(latest["Date"]).date()),
        "current_close": round(current_close, 4),
        "predicted_next_close": round(predicted_close, 4),
        "predicted_next_return_pct": round(predicted_return * 100, 4),
        "rsi_14": round(float(latest["rsi_14"]), 4),
        "macd": round(float(latest["macd"]), 4),
        "volatility_20_pct": round(float(latest["volatility_20"]) * 100, 4),
        "ma_20": round(float(latest["ma_20"]), 4),
        "ma_60": round(float(latest["ma_60"]), 4),
        "disclaimer": "For educational use only. This is not investment advice.",
    }


def save_prediction(payload: dict) -> None:
    LATEST_PREDICTION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(payload.keys())
    file_exists = PREDICTION_HISTORY_PATH.exists()
    with PREDICTION_HISTORY_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)


def format_message(payload: dict) -> str:
    direction = "상승" if payload["predicted_next_return_pct"] >= 0 else "하락"
    return (
        "[GE Daily Prediction]\n"
        f"기준일: {payload['last_market_date']}\n"
        f"현재 종가: ${payload['current_close']}\n"
        f"예측 다음 거래일 종가: ${payload['predicted_next_close']}\n"
        f"예상 수익률: {payload['predicted_next_return_pct']}% ({direction})\n"
        f"모델: {payload['model']}\n"
        f"RSI(14): {payload['rsi_14']}\n"
        f"최근 20일 변동성: {payload['volatility_20_pct']}%\n"
        "주의: 본 결과는 학습용 예측이며 투자 조언이 아닙니다."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next trading day close for GE.")
    parser.add_argument("--model", choices=["dense", "lstm"], default="dense", help="Model to use.")
    parser.add_argument("--start", default=START_DATE, help="Download start date.")
    args = parser.parse_args()

    ensure_directories()
    feature_columns = load_feature_columns()
    raw_data = collect_stock_data(ticker=TICKER, start_date=args.start)
    prepared_data = prepare_latest_features(raw_data, feature_columns)

    if args.model == "lstm":
        predicted_return = predict_lstm(prepared_data, feature_columns)
        model_name = "TensorFlow LSTM Regression"
    else:
        predicted_return = predict_dense(prepared_data, feature_columns)
        model_name = "TensorFlow Dense Regression"

    payload = build_prediction_payload(prepared_data, predicted_return, model_name)
    save_prediction(payload)

    print(json.dumps(payload, indent=2))
    print()
    print(format_message(payload))


if __name__ == "__main__":
    main()
