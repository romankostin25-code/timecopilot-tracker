"""Fetches historical price data from Yahoo Finance."""

import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()


def fetch_historical_data(ticker: str | None = None, years: int = 2) -> pd.DataFrame:
    ticker = ticker or os.getenv("ASSET_TICKER", "SPY")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    print(f"Fetching {years}y of daily closes for {ticker}...")

    raw = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                      end=end_date.strftime("%Y-%m-%d"), progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2 with single ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    closes = raw["Close"].dropna()

    df = pd.DataFrame({
        "ds": closes.index.normalize().tz_localize(None),  # strip timezone
        "y": closes.values.astype(float),
    })
    df = df.sort_values("ds").reset_index(drop=True)

    print(f"  Loaded {len(df)} data points  |  latest date: {df['ds'].max().date()}")
    return df


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data = fetch_historical_data(ticker_arg)
    print(data.tail())
