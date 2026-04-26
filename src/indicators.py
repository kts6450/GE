import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["previous_close"] = data["Close"].shift(1)
    data["daily_return"] = data["Close"].pct_change()
    data["target_close_next_day"] = data["Close"].shift(-1)
    data["target_return_next_day"] = data["Close"].shift(-1) / data["Close"] - 1

    for window in [5, 10, 20, 60]:
        data[f"ma_{window}"] = data["Close"].rolling(window=window).mean()
        data[f"volume_ma_{window}"] = data["Volume"].rolling(window=window).mean()

    data["ma_20_gap"] = data["Close"] / data["ma_20"] - 1
    data["ma_60_gap"] = data["Close"] / data["ma_60"] - 1
    data["volume_change"] = data["Volume"].pct_change()
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["open_gap"] = (data["Open"] - data["previous_close"]) / data["previous_close"]

    data["volatility_5"] = data["daily_return"].rolling(window=5).std()
    data["volatility_20"] = data["daily_return"].rolling(window=20).std()

    data["rsi_14"] = calculate_rsi(data["Close"], window=14)

    macd, macd_signal, macd_hist = calculate_macd(data["Close"])
    data["macd"] = macd
    data["macd_signal"] = macd_signal
    data["macd_hist"] = macd_hist

    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(data["Close"], window=20)
    data["bb_middle"] = bb_middle
    data["bb_upper"] = bb_upper
    data["bb_lower"] = bb_lower
    data["bb_width"] = (bb_upper - bb_lower) / bb_middle
    data["bb_position"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower)

    return data


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: int = 2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std

    return middle, upper, lower
