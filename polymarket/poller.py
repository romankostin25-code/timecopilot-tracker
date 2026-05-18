"""Polymarket snapshot poller — saves market snapshots every run for momentum tracking."""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

POLY_API       = "https://gamma-api.polymarket.com/markets"
SNAPSHOTS_PATH = Path("data/poly_snapshots.csv")
MIN_VOLUME     = float(os.getenv("POLYMARKET_MIN_VOLUME", 25000))


def fetch_all_snapshots(pages=4):
    markets = []
    seen = set()
    for page in range(pages):
        try:
            resp = requests.get(
                POLY_API,
                params={"active": "true", "limit": 50, "offset": page * 50,
                        "order": "volume", "ascending": "false"},
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for m in batch:
                mid = m.get("id", "")
                if mid and mid not in seen:
                    seen.add(mid)
                    markets.append(m)
        except Exception as e:
            print(f"[poller] Page {page}: {e}")
            break

    rows = []
    now  = datetime.now(timezone.utc).isoformat()
    for m in markets:
        try:
            volume = float(m.get("volume", 0) or 0)
            if volume < MIN_VOLUME:
                continue
            prices = m.get("outcomePrices", [])
            if len(prices) < 2:
                continue
            rows.append({
                "timestamp":   now,
                "market_id":   m.get("id", ""),
                "question":    m.get("question", ""),
                "prob_yes":    float(prices[0]),
                "prob_no":     float(prices[1]),
                "volume":      volume,
                "end_date":    m.get("endDate", ""),
            })
        except Exception:
            continue

    if rows:
        new_df = pd.DataFrame(rows)
        if SNAPSHOTS_PATH.exists():
            existing = pd.read_csv(SNAPSHOTS_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Keep only last 7 days of snapshots
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
            combined = combined[combined["timestamp"] >= cutoff]
            combined["timestamp"] = combined["timestamp"].astype(str)
        else:
            combined = new_df
        SNAPSHOTS_PATH.parent.mkdir(exist_ok=True)
        combined.to_csv(SNAPSHOTS_PATH, index=False)
        print(f"[poller] Saved {len(rows)} snapshots. Total: {len(combined)}")

    return rows


if __name__ == "__main__":
    fetch_all_snapshots()
