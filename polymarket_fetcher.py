"""Trading Co-Pilot — Polymarket public API integration."""

import os
import json
import requests
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

MIN_VOLUME = float(os.getenv("POLYMARKET_MIN_VOLUME", 50000))
CSV_PATH   = "polymarket_data.csv"

POLYMARKET_KEYWORDS = {
    "GC=F":      ["gold price", "gold"],
    "SI=F":      ["silver price", "silver"],
    "CL=F":      ["oil price", "crude oil", "WTI"],
    "NG=F":      ["natural gas"],
    "HG=F":      ["copper"],
    "EURUSD=X":  ["euro dollar", "EUR USD", "euro"],
    "GBPUSD=X":  ["pound dollar", "GBP"],
    "USDJPY=X":  ["yen", "USD JPY"],
    "AUDUSD=X":  ["australian dollar"],
    "USDCHF=X":  ["swiss franc"],
    "DX-Y.NYB":  ["dollar index", "DXY"],
    "^TNX":      ["federal reserve", "fed rate", "interest rate cut", "FOMC"],
    "^GSPC":     ["S&P 500", "stock market", "recession"],
}


def _fetch_markets(keyword: str) -> list[dict]:
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"search": keyword, "active": "true", "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        out = []
        for m in resp.json():
            try:
                prices = m.get("outcomePrices", [])
                if len(prices) < 2:
                    continue
                out.append({
                    "question":          m.get("question", ""),
                    "prob_yes":          float(prices[0]),
                    "prob_no":           float(prices[1]),
                    "volume_usd":        float(m.get("volume", 0) or 0),
                    "end_date":          m.get("endDate", ""),
                    "market_id":         m.get("id", ""),
                    "url":               f"https://polymarket.com/event/{m.get('slug', '')}",
                })
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"  Polymarket API error for '{keyword}': {e}")
        return []


def _derive_directional(markets: list[dict]) -> dict | None:
    bull, bear = [], []
    for m in markets:
        if m["volume_usd"] < MIN_VOLUME:
            continue
        q    = m["question"].lower()
        prob = m["prob_yes"]
        vol  = m["volume_usd"]
        if any(w in q for w in ["above", "exceed", "higher", "rise", "reach", "over", "up"]):
            bull.append((prob, vol))
        elif any(w in q for w in ["below", "under", "drop", "fall", "lower", "decline", "crash"]):
            bear.append((1 - prob, vol))

    all_sigs = bull + bear
    if not all_sigs:
        return None
    total = sum(v for _, v in all_sigs)
    weighted = sum(p * v for p, v in all_sigs) / total
    return {
        "poly_prob_bullish":      round(weighted, 4),
        "poly_prob_bearish":      round(1 - weighted, 4),
        "poly_signal":            "BULLISH" if weighted > 0.55 else "BEARISH" if weighted < 0.45 else "NEUTRAL",
        "poly_market_count":      len(all_sigs),
        "poly_total_volume_usd":  round(total, 2),
    }


def fetch_all_polymarket():
    today = str(date.today())
    rows  = []
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]

    for ticker in tickers:
        keywords = POLYMARKET_KEYWORDS.get(ticker, [ticker])
        print(f"[{ticker}] Fetching Polymarket…")

        all_markets, seen = [], set()
        for kw in keywords:
            for m in _fetch_markets(kw):
                if m["market_id"] not in seen:
                    seen.add(m["market_id"])
                    all_markets.append(m)

        if not all_markets:
            rows.append({
                "date": today, "ticker": ticker,
                "poly_prob_bullish": None, "poly_prob_bearish": None,
                "poly_signal": "NO_DATA", "poly_market_count": 0,
                "poly_total_volume_usd": 0,
                "top_question": "", "top_question_prob_yes": None,
                "top_question_url": "", "all_markets_json": "[]",
            })
            continue

        all_markets.sort(key=lambda x: x["volume_usd"], reverse=True)
        top    = all_markets[0]
        derived = _derive_directional(all_markets)

        rows.append({
            "date":                    today,
            "ticker":                  ticker,
            "poly_prob_bullish":       derived["poly_prob_bullish"] if derived else None,
            "poly_prob_bearish":       derived["poly_prob_bearish"] if derived else None,
            "poly_signal":             derived["poly_signal"] if derived else "INSUFFICIENT_VOLUME",
            "poly_market_count":       derived["poly_market_count"] if derived else 0,
            "poly_total_volume_usd":   derived["poly_total_volume_usd"] if derived else 0,
            "top_question":            top["question"],
            "top_question_prob_yes":   top["prob_yes"],
            "top_question_url":        top["url"],
            "all_markets_json":        json.dumps(all_markets[:5]),
        })
        sig = derived["poly_signal"] if derived else "n/a"
        vol = derived["poly_total_volume_usd"] if derived else 0
        print(f"[{ticker}] ✓ {sig} | ${vol:,.0f} | {len(all_markets)} markets")

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
