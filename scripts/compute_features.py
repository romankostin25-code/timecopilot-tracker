"""Compute daily technical + macro features for all tickers.

Outputs data/features.parquet with one row per (ticker, date) containing:
  Price features:     log_ret_1d, 5d, 20d, 60d
  Volatility:         vol_20d, vol_60d
  Technical:          rsi_14, macd_signal, bb_pos (Bollinger position)
  Volume:             vol_ratio_20d
  Macro (shared):     vix, yield_10y, yield_2y, yield_spread, dxy
  Targets (fwd):      fwd_ret_5d, fwd_ret_30d, fwd_ret_90d
                      actual_bullish_5d, _30d, _90d

Usage:
    python scripts/compute_features.py
    python scripts/compute_features.py --days 365
"""

import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = "data/features.parquet"
MACRO_TICKERS = {"^VIX": "vix", "^TNX": "yield_10y", "^IRX": "yield_3m",
                 "DX-Y.NYB": "dxy"}


def _compute_features_for_ticker(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
    from engine.feature_builder import compute_price_features
    df = compute_price_features(price_df, ticker)

    # Forward returns (targets)
    close = df["close"].astype(float)
    df["fwd_ret_5d"]  = np.log(close.shift(-5)  / close)
    df["fwd_ret_30d"] = np.log(close.shift(-30) / close)
    df["fwd_ret_90d"] = np.log(close.shift(-90) / close)
    df["actual_bullish_5d"]  = (df["fwd_ret_5d"]  > 0).astype(float)
    df["actual_bullish_30d"] = (df["fwd_ret_30d"] > 0).astype(float)
    df["actual_bullish_90d"] = (df["fwd_ret_90d"] > 0).astype(float)

    # Mark unknowable future targets as NaN
    today = date.today()
    df.loc[pd.to_datetime(df["date"]).dt.date >= today - timedelta(days=5),
           ["fwd_ret_5d",  "actual_bullish_5d"]]  = np.nan
    df.loc[pd.to_datetime(df["date"]).dt.date >= today - timedelta(days=30),
           ["fwd_ret_30d", "actual_bullish_30d"]] = np.nan
    df.loc[pd.to_datetime(df["date"]).dt.date >= today - timedelta(days=90),
           ["fwd_ret_90d", "actual_bullish_90d"]] = np.nan
    return df


def pull_macro_series(days_back: int = 1095) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=days_back + 30)
    macro_frames = []
    for ytick, col in MACRO_TICKERS.items():
        try:
            raw = yf.download(ytick, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                continue
            s = raw["Close"].squeeze().reset_index()
            s.columns = ["date", col]
            s["date"] = pd.to_datetime(s["date"]).dt.date
            macro_frames.append(s.set_index("date"))
        except Exception as e:
            print(f"  [macro] {ytick}: {e}")

    if not macro_frames:
        return pd.DataFrame()
    macro = macro_frames[0].join(macro_frames[1:], how="outer").ffill().reset_index()
    # Derived macro features
    if "yield_10y" in macro.columns and "yield_3m" in macro.columns:
        macro["yield_spread"] = macro["yield_10y"] - macro["yield_3m"]
    return macro


def compute_all_features(days_back: int = 1095, tickers: list[str] = None):
    from engine.universe import ALL_TICKERS
    if tickers is None:
        tickers = ALL_TICKERS

    end = datetime.today()
    start = end - timedelta(days=days_back + 120)  # extra lookback for rolling windows

    print("Fetching macro series...")
    macro = pull_macro_series(days_back + 120)

    all_frames = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                print("no data")
                continue
            price_df = raw[["Close", "Volume"]].reset_index()
            price_df.columns = ["date", "close", "volume"]
            price_df["date"] = pd.to_datetime(price_df["date"]).dt.date

            feat = _compute_features_for_ticker(ticker, price_df)

            # Merge macro
            if not macro.empty:
                macro["date"] = pd.to_datetime(macro["date"]).dt.date
                feat = feat.merge(macro, on="date", how="left")
                feat[list(macro.columns.drop("date"))] = feat[list(macro.columns.drop("date"))].ffill()

            # Keep only rows within requested window
            cutoff = (end - timedelta(days=days_back)).date()
            feat = feat[pd.to_datetime(feat["date"]).dt.date >= cutoff]
            all_frames.append(feat)
            print(f"{len(feat)} rows")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_frames:
        print("No data collected.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows × {len(combined.columns)} cols to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=1095)
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    tickers = [args.ticker] if args.ticker else None
    compute_all_features(days_back=args.days, tickers=tickers)
