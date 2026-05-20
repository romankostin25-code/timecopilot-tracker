"""Vercel cron function — runs every 15 min on weekdays, refreshes live prices + news."""

import os
import json
import yfinance as yf
from datetime import datetime, timezone
from pathlib import Path

PRICES_PATH  = Path("data/prices_live.json")
CRON_SECRET  = os.getenv("VERCEL_CRON_SECRET", "")


def handler(request):
    # Authenticate cron calls in production
    auth = ""
    if hasattr(request, "headers"):
        auth = request.headers.get("authorization", "")
    if CRON_SECRET and os.getenv("VERCEL_ENV") == "production" and auth != f"Bearer {CRON_SECRET}":
        return {"statusCode": 401, "body": "Unauthorized"}

    from engine.universe import ALL_TICKERS, UNIVERSE

    prices = {}
    for ticker in ALL_TICKERS:
        try:
            raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if not raw.empty:
                closes = raw["Close"].squeeze().dropna()
                current = float(closes.iloc[-1])
                prev    = float(closes.iloc[-2]) if len(closes) >= 2 else current
                prices[ticker] = {
                    "price":       round(current, 6),
                    "prev_close":  round(prev, 6),
                    "change_pct":  round((current - prev) / prev * 100, 4),
                    "name":        UNIVERSE[ticker]["name"],
                    "class":       UNIVERSE[ticker]["class"],
                    "fetched_at":  datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            prices[ticker] = {"error": str(e)}

    PRICES_PATH.parent.mkdir(exist_ok=True)
    PRICES_PATH.write_text(json.dumps(prices, indent=2, default=str))

    # Trigger news ingestion + NLP processing
    try:
        from intelligence.news_poller import fetch_all_feeds, update_feed
        added = update_feed(fetch_all_feeds())
        print(f"[poll] +{len(added)} news articles")
    except Exception as e:
        print(f"[poll] News error: {e}")

    try:
        from intelligence.nlp_pipeline import process_feed
        process_feed(max_batch=25)
    except Exception as e:
        print(f"[poll] NLP error: {e}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "updated":   len(prices),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }),
    }


if __name__ == "__main__":
    handler(None)
