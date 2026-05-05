"""Trading Co-Pilot — Polymarket public API integration."""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

MIN_VOLUME = float(os.getenv("POLYMARKET_MIN_VOLUME", 50000))
CSV_PATH   = "polymarket_data.csv"

POLYMARKET_KEYWORDS = {
    "GC=F":      ["gold"],
    "SI=F":      ["silver"],
    "CL=F":      ["oil", "crude", "wti", "brent"],
    "NG=F":      ["natural gas", "gas price"],
    "HG=F":      ["copper"],
    "EURUSD=X":  ["euro", "eur/usd", "eurusd"],
    "GBPUSD=X":  ["pound", "gbp", "sterling"],
    "USDJPY=X":  ["yen", "usdjpy", "usd/jpy"],
    "AUDUSD=X":  ["australian dollar", "aud"],
    "USDCHF=X":  ["swiss franc", "chf"],
    "DX-Y.NYB":  ["dollar index", "dxy"],
    "^TNX":      ["federal reserve", "fed rate", "interest rate", "fomc", "rate cut", "treasury"],
    "^GSPC":     ["s&p", "sp500", "stock market", "recession", "nasdaq"],
}

# Fetch markets in small pages to avoid rate limits
def _fetch_page(offset: int = 0, limit: int = 20) -> list[dict]:
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "limit": limit, "offset": offset},
            timeout=15,
        )
        resp.raise_for_status()
        out = []
        for m in resp.json():
            try:
                raw_prices = m.get("outcomePrices", [])
                # API returns outcomePrices as a JSON string, not a list
                if isinstance(raw_prices, str):
                    raw_prices = json.loads(raw_prices)
                if len(raw_prices) < 2:
                    continue
                out.append({
                    "question":   m.get("question", ""),
                    "prob_yes":   float(raw_prices[0]),
                    "prob_no":    float(raw_prices[1]),
                    "volume_usd": float(m.get("volume", 0) or 0),
                    "end_date":   m.get("endDate", ""),
                    "market_id":  m.get("id", ""),
                    "url":        f"https://polymarket.com/event/{m.get('slug', '')}",
                })
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"  Polymarket page error (offset={offset}): {e}")
        return []


def _load_all_markets(pages: int = 5) -> list[dict]:
    """Fetch up to pages*20 markets in small batches."""
    markets, seen = [], set()
    for i in range(pages):
        batch = _fetch_page(offset=i * 20, limit=20)
        if not batch:
            break
        for m in batch:
            if m["market_id"] not in seen:
                seen.add(m["market_id"])
                markets.append(m)
        if len(batch) < 20:
            break
        time.sleep(0.3)
    print(f"  [polymarket] loaded {len(markets)} active markets")
    return markets


def _search_markets(keywords: list[str], all_markets: list[dict]) -> list[dict]:
    import re
    matched, seen = [], set()
    # Word-boundary patterns prevent "gold" matching "golden", "aud" matching "saudi"
    patterns = [re.compile(r'\b' + re.escape(kw.lower()) + r'\b') for kw in keywords]
    for m in all_markets:
        if m["market_id"] in seen:
            continue
        q = m["question"].lower()
        if any(p.search(q) for p in patterns):
            seen.add(m["market_id"])
            matched.append(m)
    return matched


def _derive_directional(markets: list[dict]) -> dict | None:
    bull, bear = [], []
    for m in markets:
        if m["volume_usd"] < MIN_VOLUME:
            continue
        q    = m["question"].lower()
        prob = m["prob_yes"]
        vol  = m["volume_usd"]
        if any(w in q for w in ["above", "exceed", "higher", "rise", "reach", "over", "up", "break", "bull"]):
            bull.append((prob, vol))
        elif any(w in q for w in ["below", "under", "drop", "fall", "lower", "decline", "crash", "cut", "bear"]):
            bear.append((1 - prob, vol))

    all_sigs = bull + bear
    if not all_sigs:
        return None
    total    = sum(v for _, v in all_sigs)
    weighted = sum(p * v for p, v in all_sigs) / total
    return {
        "poly_prob_bullish":     round(weighted, 4),
        "poly_prob_bearish":     round(1 - weighted, 4),
        "poly_signal":           "BULLISH" if weighted > 0.55 else "BEARISH" if weighted < 0.45 else "NEUTRAL",
        "poly_market_count":     len(all_sigs),
        "poly_total_volume_usd": round(total, 2),
    }


def fetch_all_polymarket():
    today   = str(date.today())
    rows    = []
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]

    all_markets = _load_all_markets(pages=10)  # ~200 markets max

    for ticker in tickers:
        keywords = POLYMARKET_KEYWORDS.get(ticker, [ticker])
        print(f"[{ticker}] Fetching Polymarket…")
        matched = _search_markets(keywords, all_markets)

        if not matched:
            rows.append({
                "date": today, "ticker": ticker,
                "poly_prob_bullish": None, "poly_prob_bearish": None,
                "poly_signal": "NO_DATA", "poly_market_count": 0,
                "poly_total_volume_usd": 0,
                "top_question": "", "top_question_prob_yes": None,
                "top_question_url": "", "all_markets_json": "[]",
            })
            continue

        matched.sort(key=lambda x: x["volume_usd"], reverse=True)
        top     = matched[0]
        derived = _derive_directional(matched)

        rows.append({
            "date":                   today,
            "ticker":                 ticker,
            "poly_prob_bullish":      derived["poly_prob_bullish"] if derived else None,
            "poly_prob_bearish":      derived["poly_prob_bearish"] if derived else None,
            "poly_signal":            derived["poly_signal"] if derived else "INSUFFICIENT_VOLUME",
            "poly_market_count":      derived["poly_market_count"] if derived else 0,
            "poly_total_volume_usd":  derived["poly_total_volume_usd"] if derived else 0,
            "top_question":           top["question"],
            "top_question_prob_yes":  top["prob_yes"],
            "top_question_url":       top["url"],
            "all_markets_json":       json.dumps(matched[:5]),
        })
        sig = derived["poly_signal"] if derived else "n/a"
        vol = derived["poly_total_volume_usd"] if derived else 0
        print(f"[{ticker}] ✓ {sig} | ${vol:,.0f} | {len(matched)} markets matched")

    new_df = pd.DataFrame(rows)
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        existing = existing[existing["date"] != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(CSV_PATH, index=False)
    print(f"\n✓ polymarket_data.csv updated ({len(rows)} rows).")


if __name__ == "__main__":
    fetch_all_polymarket()
