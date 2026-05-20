"""Merge price features + news sentiment into the TFT training dataset.

Reads:
    data/features.parquet          — technical + macro features (from compute_features.py)
    data/news_sentiment.parquet    — daily sentiment per ticker (from pull_historical_news.py)

Outputs:
    data/training_dataset.parquet  — merged, cleaned, ready for TFT training

Adds sentiment momentum features:
    sentiment_7d_ma     — 7-day rolling mean of daily sentiment
    sentiment_momentum  — today's sentiment minus 7d MA (surprise direction)
    article_count_7d    — rolling 7d article count (news volume)
"""

import os
import numpy as np
import pandas as pd

FEATURES_PATH   = "data/features.parquet"
NEWS_PATH       = "data/news_sentiment.parquet"
OUTPUT_PATH     = "data/training_dataset.parquet"

FEATURE_COLS = [
    "ticker", "date",
    # Price features
    "log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d",
    "vol_20d", "vol_60d",
    # Technical
    "rsi_14", "macd_signal", "bb_pos", "vol_ratio_20d",
    # Macro
    "vix", "yield_10y", "yield_3m", "yield_spread", "dxy",
    # News (filled after merge)
    "sentiment_score", "article_count", "sentiment_7d_ma", "sentiment_momentum",
    # Targets
    "fwd_ret_5d", "fwd_ret_30d", "fwd_ret_90d",
    "actual_bullish_5d", "actual_bullish_30d", "actual_bullish_90d",
]


def assemble():
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: {FEATURES_PATH} not found. Run scripts/compute_features.py first.")
        return

    print("Loading features...")
    feat = pd.read_parquet(FEATURES_PATH)
    feat["date"] = pd.to_datetime(feat["date"]).dt.date

    news = pd.DataFrame()
    if os.path.exists(NEWS_PATH):
        print("Loading news sentiment...")
        news = pd.read_parquet(NEWS_PATH)
        news["date"] = pd.to_datetime(news["date"]).dt.date

        # Add 7d rolling features per ticker
        news = news.sort_values(["ticker", "date"])
        news["sentiment_7d_ma"] = (
            news.groupby("ticker")["sentiment_score"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
        news["sentiment_momentum"] = news["sentiment_score"] - news["sentiment_7d_ma"]
        news["article_count_7d"] = (
            news.groupby("ticker")["article_count"]
            .transform(lambda x: x.rolling(7, min_periods=1).sum())
        )
    else:
        print("No news sentiment file — training without news features.")

    # Merge
    if not news.empty:
        merged = feat.merge(
            news[["ticker", "date", "sentiment_score", "article_count",
                  "sentiment_7d_ma", "sentiment_momentum", "article_count_7d"]],
            on=["ticker", "date"], how="left",
        )
        # Fill missing sentiment with 0 (neutral) and zero count
        for col in ["sentiment_score", "article_count", "sentiment_7d_ma",
                    "sentiment_momentum", "article_count_7d"]:
            merged[col] = merged[col].fillna(0.0)
    else:
        merged = feat.copy()
        for col in ["sentiment_score", "article_count", "sentiment_7d_ma",
                    "sentiment_momentum", "article_count_7d"]:
            merged[col] = 0.0

    # Clip extreme values to reduce outlier impact
    for col in ["log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d"]:
        if col in merged.columns:
            merged[col] = merged[col].clip(-0.5, 0.5)
    for col in ["rsi_14", "macd_signal", "bb_pos"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())

    # Keep only rows where key features are not NaN
    key_features = ["log_ret_1d", "vol_20d", "rsi_14"]
    merged = merged.dropna(subset=key_features)

    # Build time_idx AFTER dropna — must be consecutive integers per ticker (no gaps)
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged["time_idx"] = merged.groupby("ticker").cumcount()

    os.makedirs("data", exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    n_tickers = merged["ticker"].nunique()
    n_rows    = len(merged)
    n_with_5d = merged["actual_bullish_5d"].notna().sum()
    print(f"Saved {OUTPUT_PATH}: {n_rows} rows, {n_tickers} tickers, {n_with_5d} labeled 5d rows")


if __name__ == "__main__":
    assemble()
