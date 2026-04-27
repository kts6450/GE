"""
GE 및 관련 테마주 비교 분석 모듈
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

THEME_TICKERS: dict[str, str] = {
    "GE": "GE Aerospace",
    "GEV": "GE Vernova (에너지)",
    "GEHC": "GE HealthCare",
    "RTX": "RTX (라이시온)",
    "HON": "Honeywell",
    "BA": "Boeing",
}

PERIOD_DAYS: dict[str, int] = {
    "1개월": 21,
    "3개월": 63,
    "6개월": 126,
    "1년": 252,
    "YTD": -1,
}


def fetch_close_prices(
    tickers: list[str],
    start_date: str,
) -> pd.DataFrame:
    """여러 티커의 종가를 한 번에 내려받아 Date 인덱스 DataFrame 반환."""
    raw = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close = close.dropna(how="all")
    return close


def normalize(prices: pd.DataFrame) -> pd.DataFrame:
    """첫 날 종가를 100으로 정규화."""
    first = prices.iloc[0].replace(0, np.nan)
    return (prices / first * 100).round(4)


def calc_period_return(prices: pd.DataFrame, days: int, ytd: bool = False) -> pd.Series:
    """지정 기간 수익률(%) 계산."""
    if ytd:
        year_start = datetime(datetime.today().year, 1, 1).strftime("%Y-%m-%d")
        subset = prices[prices.index >= year_start]
        if len(subset) < 2:
            return pd.Series(np.nan, index=prices.columns)
        return ((subset.iloc[-1] / subset.iloc[0] - 1) * 100).round(2)

    if len(prices) < days + 1:
        return pd.Series(np.nan, index=prices.columns)
    return ((prices.iloc[-1] / prices.iloc[-days] - 1) * 100).round(2)


def build_returns_table(prices: pd.DataFrame) -> pd.DataFrame:
    """기간별 수익률 비교 테이블 생성."""
    rows = []
    for label, days in PERIOD_DAYS.items():
        ytd = days == -1
        ret = calc_period_return(prices, days if not ytd else 0, ytd=ytd)
        rows.append(ret.rename(label))
    df = pd.DataFrame(rows).T
    df.index.name = "Ticker"
    return df


def calc_correlation(prices: pd.DataFrame, days: int = 126) -> pd.DataFrame:
    """최근 N 거래일 수익률 기반 상관관계 행렬."""
    subset = prices.tail(days).pct_change().dropna()
    return subset.corr().round(3)


def calc_volatility(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """연율화 변동성(%) - 최근 window 거래일 기준."""
    daily_ret = prices.pct_change().dropna()
    vol = daily_ret.tail(window).std() * np.sqrt(252) * 100
    return vol.round(2)


def get_comparison_data(
    start_date: str = "2023-01-01",
) -> dict:
    """비교 분석에 필요한 모든 데이터를 반환."""
    tickers = list(THEME_TICKERS.keys())
    prices = fetch_close_prices(tickers, start_date=start_date)

    # 다운로드 실패한 티커 제거
    prices = prices.dropna(axis=1, how="all")
    available = [t for t in tickers if t in prices.columns]

    normalized = normalize(prices[available])
    returns_table = build_returns_table(prices[available])
    correlation = calc_correlation(prices[available])
    volatility = calc_volatility(prices[available])

    latest_prices = prices[available].iloc[-1].round(2)
    latest_date = prices.index[-1].strftime("%Y-%m-%d")

    return {
        "prices": prices[available],
        "normalized": normalized,
        "returns_table": returns_table,
        "correlation": correlation,
        "volatility": volatility,
        "latest_prices": latest_prices,
        "latest_date": latest_date,
        "available_tickers": available,
        "ticker_labels": {t: THEME_TICKERS[t] for t in available},
    }
