"""Fills in actual closing prices for past forecast target dates."""

import os
from datetime import date

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FORECASTS_CSV = "forecasts.csv"


def update_actuals(csv_path: str = FORECASTS_CSV) -> int:
    if not os.path.exists(csv_path):
        print("forecasts.csv not found — nothing to update.")
        return 0

    df = pd.read_csv(csv_path, parse_dates=["forecast_date", "target_date"])

    today = date.today()
    mask = df["actual"].isna() & (df["target_date"].dt.date <= today)
    pending = df[mask]

    if pending.empty:
        print("No rows to update — all past target dates already have actuals.")
        return 0

    updated = 0
    for ticker, group in pending.groupby("ticker"):
        dates_needed = group["target_date"].dt.date.unique()
        min_date = min(dates_needed) - pd.Timedelta(days=5)
        max_date = max(dates_needed) + pd.Timedelta(days=2)

        raw = yf.download(
            ticker,
            start=str(min_date),
            end=str(max_date + pd.Timedelta(days=1)),
            progress=False,
        )
        if raw.empty:
            print(f"  Warning: no data returned for {ticker}")
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        closes = raw["Close"].dropna()
        closes.index = closes.index.normalize().tz_localize(None).date

        for idx in group.index:
            target = df.at[idx, "target_date"].date()
            if target in closes.index:
                df.at[idx, "actual"] = round(float(closes[target]), 4)
                updated += 1
            else:
                # Try nearest available date (market holiday / weekend)
                available = sorted(closes.index)
                nearest = min(available, key=lambda d: abs((d - target).days), default=None)
                if nearest and abs((nearest - target).days) <= 3:
                    df.at[idx, "actual"] = round(float(closes[nearest]), 4)
                    updated += 1

    df.to_csv(csv_path, index=False)
    print(f"Updated {updated} rows with actual closing prices.")
    return updated


if __name__ == "__main__":
    update_actuals()
