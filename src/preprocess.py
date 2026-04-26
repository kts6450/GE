import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import (
    FEATURE_COLUMNS_PATH,
    FEATURE_SCALER_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_SCALER_PATH,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
    ensure_directories,
)
from indicators import add_technical_indicators


BASE_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "daily_return",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_60",
    "volume_ma_5",
    "volume_ma_10",
    "volume_ma_20",
    "volume_ma_60",
    "ma_20_gap",
    "ma_60_gap",
    "volume_change",
    "high_low_range",
    "open_gap",
    "volatility_5",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_position",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"])
    data = data.sort_values("Date").drop_duplicates(subset="Date")
    return data


def preprocess_data(
    raw_data: pd.DataFrame,
    target_column: str = "target_return_next_day",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    data = add_technical_indicators(raw_data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_columns = [column for column in BASE_FEATURE_COLUMNS if column in data.columns]
    missing_columns = sorted(set(BASE_FEATURE_COLUMNS) - set(feature_columns))
    if missing_columns:
        raise RuntimeError(f"Missing feature columns: {missing_columns}")

    train_data, test_data = train_test_split(data, test_size=TEST_SIZE, shuffle=False)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features_scaled = feature_scaler.fit_transform(train_data[feature_columns])
    test_features_scaled = feature_scaler.transform(test_data[feature_columns])

    train_target_scaled = target_scaler.fit_transform(train_data[[target_column]])
    test_target_scaled = target_scaler.transform(test_data[[target_column]])

    train_scaled = train_data[["Date", "target_close_next_day", "target_return_next_day"]].copy()
    test_scaled = test_data[["Date", "target_close_next_day", "target_return_next_day"]].copy()
    train_scaled["current_close"] = train_data["Close"].to_numpy()
    test_scaled["current_close"] = test_data["Close"].to_numpy()

    for index, column in enumerate(feature_columns):
        train_scaled[column] = train_features_scaled[:, index]
        test_scaled[column] = test_features_scaled[:, index]

    train_scaled[f"{target_column}_scaled"] = train_target_scaled.reshape(-1)
    test_scaled[f"{target_column}_scaled"] = test_target_scaled.reshape(-1)

    processed = pd.concat([train_scaled, test_scaled], ignore_index=True)

    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    return processed, train_scaled, test_scaled, feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess GE stock data for TensorFlow models.")
    parser.add_argument("--input", default=RAW_DATA_PATH, help="Raw CSV input path.")
    parser.add_argument("--target", default="target_return_next_day", help="Regression target column.")
    args = parser.parse_args()

    ensure_directories()
    raw_data = load_raw_data(args.input)
    processed, train_data, test_data, feature_columns = preprocess_data(raw_data, args.target)

    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    print(f"Saved processed data to {PROCESSED_DATA_PATH}")
    print(f"Train rows: {len(train_data)}, Test rows: {len(test_data)}")
    print(f"Feature count: {len(feature_columns)}")


if __name__ == "__main__":
    main()
