"""
Polymarket Poller — fetches active markets and maps them to macro events.

IMPORTANT: Polymarket's gamma-api `search` param does NOT filter markets —
it always returns the same top markets sorted by activity. We must:
  1. Fetch all active markets in batches (up to 1000)
  2. Filter client-side using word-boundary regex matching
  3. Map each matched market to its event_type via the keyword table
"""

import re
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Support both direct execution and module import
try:
    from .event_map import all_keywords
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from event_map import all_keywords

SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "poly_snapshots.csv"
SNAPSHOT_PATH.parent.mkdir(exist_ok=True)

POLYMARKET_API = "https://gamma-api.polymarket.com/markets"
MIN_VOLUME_USD = 10_000


def _fetch_page(offset: int = 0, limit: int = 50) -> list[dict]:
    try:
        resp = requests.get(
            POLYMARKET_API,
            params={"active": "true", "limit": limit, "offset": offset},
            timeout=15,
        )
        resp.raise_for_status()
        out = []
        for m in resp.json():
            try:
                raw_prices = m.get("outcomePrices", [])
                if isinstance(raw_prices, str):
                    raw_prices = json.loads(raw_prices)
                if len(raw_prices) < 2:
                    continue
                volume = float(m.get("volume", 0) or 0)
                if volume < MIN_VOLUME_USD:
                    continue
                out.append({
                    "market_id":    m.get("id", ""),
                    "slug":         m.get("slug", ""),
                    "question":     m.get("question", ""),
                    "prob_yes":     float(raw_prices[0]),
                    "prob_no":      float(raw_prices[1]),
                    "volume_usd":   volume,
                    "liquidity_usd": float(m.get("liquidity", 0) or 0),
                    "end_date":     m.get("endDate", ""),
                })
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"  [poller] Page error offset={offset}: {e}")
        return []


def _load_all_markets(pages: int = 20) -> list[dict]:
    markets, seen = [], set()
    for i in range(pages):
        batch = _fetch_page(offset=i * 50, limit=50)
        if not batch:
            break
        for m in batch:
            if m["market_id"] not in seen:
                seen.add(m["market_id"])
                markets.append(m)
        if len(batch) < 50:
            break
        time.sleep(0.25)
    print(f"  [poller] loaded {len(markets)} active markets (min volume ${MIN_VOLUME_USD:,})")
    return markets


def _match_markets(all_markets: list[dict]) -> list[dict]:
    """
    Match each market to event types using word-boundary regex.
    A market can match multiple event types — we emit one row per match.
    """
    kw_list = all_keywords()  # [(keyword_lower, event_type), ...]
    patterns = [
        (re.compile(r'\b' + re.escape(kw) + r'\b'), event_type)
        for kw, event_type in kw_list
    ]

    matched = []
    for m in all_markets:
        q = m["question"].lower()
        matched_types = set()
        for pat, event_type in patterns:
            if event_type in matched_types:
                continue
            if pat.search(q):
                matched_types.add(event_type)
                matched.append({**m, "event_type": event_type})

    return matched


def fetch_all_snapshots() -> list[dict]:
    """
    Poll all markets, match to event types, store timestamped snapshot.
    Called by the scheduler every 15 minutes (or by GitHub Actions hourly).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    all_markets = _load_all_markets(pages=20)
    matched = _match_markets(all_markets)

    if not matched:
        print(f"[poller] No macro markets matched at {timestamp}")
        # Still write a heartbeat row so we know the poller ran
        matched = [{
            "timestamp": timestamp,
            "market_id": "__heartbeat__",
            "event_type": "__heartbeat__",
            "keyword_matched": "",
            "question": "__heartbeat__",
            "prob_yes": 0,
            "prob_no": 0,
            "volume_usd": 0,
            "liquidity_usd": 0,
            "end_date": "",
        }]
    else:
        snapshot_rows = [{
            "timestamp":       timestamp,
            "market_id":       m["market_id"],
            "event_type":      m["event_type"],
            "keyword_matched": "",
            "question":        m["question"],
            "prob_yes":        round(m["prob_yes"], 4),
            "prob_no":         round(m["prob_no"], 4),
            "volume_usd":      m["volume_usd"],
            "liquidity_usd":   m["liquidity_usd"],
            "end_date":        m["end_date"],
        } for m in matched]

        new_df = pd.DataFrame(snapshot_rows)
        if SNAPSHOT_PATH.exists():
            existing = pd.read_csv(SNAPSHOT_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Keep only the last 30 days of snapshots to avoid unbounded growth
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
            combined = combined[combined["timestamp"] >= cutoff]
        else:
            combined = new_df

        combined.to_csv(SNAPSHOT_PATH, index=False)
        print(f"[poller] Snapshot: {len(snapshot_rows)} matched markets at {timestamp}")

        by_event = {}
        for r in snapshot_rows:
            by_event.setdefault(r["event_type"], []).append(r["question"][:60])
        for et, qs in by_event.items():
            print(f"  [{et}] {len(qs)} markets — e.g. {qs[0]}")

        return snapshot_rows

    return []


if __name__ == "__main__":
    rows = fetch_all_snapshots()
    print(f"\nTotal matched: {len(rows)}")
