import argparse

import pandas as pd
import yfinance as yf

from config import INTERVAL, RAW_DATA_PATH, START_DATE, TICKER, ensure_directories


def collect_stock_data(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    data = yf.download(
        tickers=ticker,
        start=start_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(f"No data downloaded for ticker {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    data = data.sort_values("Date").drop_duplicates(subset="Date")

    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_columns = [column for column in expected_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(f"Downloaded data is missing columns: {missing_columns}")

    return data[expected_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GE stock data from Yahoo Finance.")
    parser.add_argument("--ticker", default=TICKER, help="Ticker symbol to download.")
    parser.add_argument("--start", default=START_DATE, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--interval", default=INTERVAL, help="Yahoo Finance interval.")
    parser.add_argument("--output", default=RAW_DATA_PATH, help="Output CSV path.")
    args = parser.parse_args()

    ensure_directories()
    data = collect_stock_data(args.ticker, args.start, args.interval)
    data.to_csv(args.output, index=False)

    print(f"Saved {len(data)} rows to {args.output}")
    print(f"Date range: {data['Date'].min()} -> {data['Date'].max()}")


if __name__ == "__main__":
    main()
