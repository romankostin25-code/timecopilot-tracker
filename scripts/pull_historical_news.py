"""Pull 3 years of per-ticker news from Polygon.io and aggregate daily sentiment.

Polygon Starter news endpoint returns articles with per-ticker sentiment scores
(positive/negative/neutral) in the `insights` array. We aggregate to a daily
sentiment score in [-1, 1] per ticker and save to data/news_sentiment.parquet.

Usage:
    python scripts/pull_historical_news.py            # all tickers, 3yr
    python scripts/pull_historical_news.py --days 90  # recent 90 days only
    python scripts/pull_historical_news.py --ticker SPY --days 365
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
SENTIMENT_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
OUTPUT_PATH = "data/news_sentiment.parquet"
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 0.1  # 10 req/s to stay within Starter limits


def _fetch_news_page(ticker: str, start: str, end: str, next_url: str = None) -> dict:
    if next_url:
        url = next_url
        if "apiKey" not in url:
            url += f"&apiKey={POLYGON_API_KEY}"
    else:
        url = (
            f"https://api.polygon.io/v2/reference/news"
            f"?ticker={ticker}&limit=1000"
            f"&published_utc.gte={start}&published_utc.lte={end}"
            f"&sort=published_utc&order=asc"
            f"&apiKey={POLYGON_API_KEY}"
        )
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 429:
                time.sleep(12)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def pull_ticker_news(ticker: str, start: date, end: date) -> list[dict]:
    """Returns list of {date, ticker, sentiment_score, article_count} dicts."""
    rows = []
    next_url = None
    start_str = start.isoformat()
    end_str = end.isoformat()
    page = 0

    while True:
        data = _fetch_news_page(ticker, start_str, end_str, next_url)
        articles = data.get("results", [])
        for art in articles:
            pub = art.get("published_utc", "")[:10]  # YYYY-MM-DD
            insights = art.get("insights", [])
            for ins in insights:
                if ins.get("ticker", "").upper() == ticker.upper():
                    rows.append({
                        "date": pub,
                        "ticker": ticker,
                        "raw_sentiment": SENTIMENT_MAP.get(ins.get("sentiment", "neutral"), 0.0),
                    })
        page += 1
        next_url = data.get("next_url")
        if not next_url or not articles:
            break
        time.sleep(RATE_LIMIT_SLEEP)

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    daily = (
        df.groupby(["date", "ticker"])["raw_sentiment"]
        .agg(sentiment_score="mean", article_count="count")
        .reset_index()
    )
    return daily.to_dict("records")


def pull_all_news(days_back: int = 1095, tickers: list[str] = None):
    if not POLYGON_API_KEY:
        print("ERROR: POLYGON_API_KEY not set in environment")
        sys.exit(1)

    if tickers is None:
        from engine.universe import ALL_TICKERS
        tickers = [t for t in ALL_TICKERS if not t.startswith("^")]  # skip index tickers

    end = date.today()
    start = end - timedelta(days=days_back)

    # Load existing to avoid re-fetching
    existing = pd.DataFrame()
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_parquet(OUTPUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date

    all_rows = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker} ... ", end="", flush=True)
        try:
            # Only fetch dates not already in existing
            if not existing.empty and ticker in existing["ticker"].values:
                last_date = existing[existing["ticker"] == ticker]["date"].max()
                ticker_start = last_date + timedelta(days=1)
                if ticker_start > end:
                    print("up-to-date")
                    continue
            else:
                ticker_start = start

            rows = pull_ticker_news(ticker, ticker_start, end)
            all_rows.extend(rows)
            print(f"{len(rows)} day-rows")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        print("No new rows fetched.")
        return

    new_df = pd.DataFrame(all_rows)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["ticker", "date"])
    os.makedirs("data", exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} total rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=1095, help="Days of history to pull")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker override")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else None
    pull_all_news(days_back=args.days, tickers=tickers)
