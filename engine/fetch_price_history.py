"""Fetch and store 90-day historical closing prices for all tickers."""

import os
import sys
import io
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "data/price_history.csv"
DAYS = 90


def _download(ticker, start, end):
    stderr_save = sys.stderr
    sys.stderr = io.StringIO()
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    finally:
        sys.stderr = stderr_save
    return raw


def fetch_price_history(days=DAYS):
    from engine.universe import ALL_TICKERS

    end   = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=days)
    dfs   = []

    for ticker in ALL_TICKERS:
        try:
            raw = _download(ticker, start, end)
            if raw.empty:
                print(f"[price_history] {ticker}: no data")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            closes = raw[["Close"]].copy()
            closes.index = pd.to_datetime(closes.index)
            closes["date"]   = closes.index.strftime("%Y-%m-%d")
            closes["ticker"] = ticker
            closes["price"]  = closes["Close"].round(6)
            dfs.append(closes[["date", "ticker", "price"]].reset_index(drop=True))
            print(f"[price_history] {ticker}: {len(closes)} rows")
        except Exception as e:
            print(f"[price_history] {ticker}: {e}")

    if not dfs:
        print("[price_history] No data fetched.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs("data", exist_ok=True)

    # Merge with existing to preserve older history beyond 90d window
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        existing = existing[~existing["date"].isin(combined["date"].unique())]
        combined = pd.concat([existing, combined], ignore_index=True)
        combined = combined.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])

    combined.to_csv(CSV_PATH, index=False)
    tickers_fetched = combined["ticker"].nunique()
    print(f"[price_history] Saved {len(combined)} rows ({tickers_fetched} tickers) to {CSV_PATH}")


if __name__ == "__main__":
    fetch_price_history()
