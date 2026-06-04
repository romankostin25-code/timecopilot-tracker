"""Claude NLP pipeline — parses news articles into structured asset signals."""

import os
import json
import math
import anthropic
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FEED_PATH = Path("data/intelligence_feed.json")

_FEED_CACHE: list | None = None
_FEED_CACHE_MTIME: float = 0.0


def _get_cached_articles() -> list:
    global _FEED_CACHE, _FEED_CACHE_MTIME
    if not FEED_PATH.exists():
        return []
    try:
        mtime = FEED_PATH.stat().st_mtime
        if _FEED_CACHE is None or mtime != _FEED_CACHE_MTIME:
            _FEED_CACHE = json.loads(FEED_PATH.read_text())
            _FEED_CACHE_MTIME = mtime
        return _FEED_CACHE
    except Exception:
        return []


def _get_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


def _system_prompt():
    from engine.universe import ALL_TICKERS
    tickers_str = ", ".join(ALL_TICKERS)
    return f"""You are a financial news analyst for a macro trading fund.
Read news headlines and summaries and extract structured trading signals.

Return ONLY valid JSON with this exact structure:
{{
  "assets_affected": ["TICKER1", "TICKER2"],
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "sentiment_confidence": 0.0 to 1.0,
  "event_type": "fed_rate_cut" | "fed_rate_hike" | "fed_rate_hold" | "cpi_above_target" |
                "cpi_below_target" | "us_recession" | "middle_east_conflict" | "russia_ukraine" |
                "ecb_rate_cut" | "boj_rate_hike" | "trump_tariffs" | "us_debt_ceiling" |
                "earnings_beat" | "earnings_miss" | "crypto_regulation" | "crypto_rally" |
                "oil_supply_disruption" | "energy_transition" | "usda_crop_report" |
                "macro_other" | "irrelevant",
  "impact_horizon": "SHORT" | "MEDIUM" | "LONG",
  "summary": "One sentence, max 15 words, market impact",
  "fed_hawkishness": null or -1.0 to 1.0,
  "signal_direction": {{"TICKER": 1 or -1}},
  "signal_strength": {{"TICKER": 0.0 to 1.0}}
}}

Available tickers: {tickers_str}
Rules:
- Only include tickers clearly affected by this specific news
- signal_direction: +1 bullish, -1 bearish
- impact_horizon: SHORT=1-5d, MEDIUM=5-30d, LONG=30d+
- fed_hawkishness: ONLY set for Fed/CB speech or minutes. +1.0=very hawkish (rate hike language,
  "inflation concern", "not cutting soon"), -1.0=very dovish ("cutting rates", "easing", "below target").
  null for non-Fed news.
- If irrelevant to tracked assets, return empty arrays and event_type "irrelevant"
- Return ONLY the JSON object"""


def process_article(article, client):
    prompt = f"Headline: {article['headline']}\nSummary: {article.get('summary_raw', '')}"
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        article.update({
            "assets_affected":      parsed.get("assets_affected", []),
            "sentiment":            parsed.get("sentiment"),
            "sentiment_confidence": parsed.get("sentiment_confidence"),
            "event_type":           parsed.get("event_type"),
            "impact_horizon":       parsed.get("impact_horizon"),
            "summary":              parsed.get("summary"),
            "fed_hawkishness":      parsed.get("fed_hawkishness"),
            "signal_direction":     parsed.get("signal_direction", {}),
            "signal_strength":      parsed.get("signal_strength", {}),
            "nlp_processed":        True,
            "nlp_processed_at":     datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        print(f"[nlp] Error on '{article['headline'][:50]}': {e}")
        article["nlp_processed"] = True
        article["event_type"] = "irrelevant"
    return article


def process_feed(max_batch=20):
    if not FEED_PATH.exists():
        print("[nlp_pipeline] No feed file found.")
        return
    articles = _get_cached_articles()
    unprocessed = [a for a in articles if not a.get("nlp_processed")]
    batch = unprocessed[:max_batch]
    if not batch:
        print("[nlp_pipeline] All articles processed.")
        return

    client = _get_client()
    print(f"[nlp_pipeline] Processing {len(batch)} articles…")
    batch_ids = {a["id"] for a in batch}

    for i, article in enumerate(batch):
        print(f"  [{i+1}/{len(batch)}] {article['source']}: {article['headline'][:60]}")
        article = process_article(article, client)

    for i, a in enumerate(articles):
        if a["id"] in batch_ids:
            match = next((b for b in batch if b["id"] == a["id"]), None)
            if match:
                articles[i] = match

    FEED_PATH.write_text(json.dumps(articles, indent=2, default=str))
    global _FEED_CACHE, _FEED_CACHE_MTIME
    _FEED_CACHE = None  # invalidate cache after write
    _FEED_CACHE_MTIME = 0.0
    print(f"[nlp_pipeline] ✓ {len(batch)} articles processed.")


def get_asset_signals_from_news(ticker, max_age_hours=24):
    articles = _get_cached_articles()
    if not articles:
        return {"signal": "NEUTRAL", "confidence": "NO_DATA", "items": []}
    cutoff_str = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    relevant = [
        a for a in articles
        if a.get("nlp_processed") and
        ticker in a.get("assets_affected", []) and
        a.get("event_type") != "irrelevant" and
        a.get("fetched_at", "") >= cutoff_str
    ]
    if not relevant:
        return {"signal": "NEUTRAL", "confidence": "NO_DATA", "items": []}

    now = datetime.now(timezone.utc)
    weighted_dir = total_w = 0.0
    for a in relevant:
        direction = a.get("signal_direction", {}).get(ticker, 0)
        strength  = a.get("signal_strength", {}).get(ticker, 0.5)
        conf      = a.get("sentiment_confidence", 0.5) or 0.5
        fetched   = datetime.fromisoformat(a["fetched_at"].replace("Z", "+00:00"))
        age_h     = (now - fetched).total_seconds() / 3600
        decay     = math.exp(-0.115 * age_h)
        w = strength * conf * decay
        weighted_dir += direction * w
        total_w += w

    if total_w == 0:
        return {"signal": "NEUTRAL", "confidence": "LOW", "items": relevant[:3]}

    net = weighted_dir / total_w
    signal = "BULLISH" if net > 0.2 else "BEARISH" if net < -0.2 else "NEUTRAL"
    confidence = "HIGH" if abs(net) > 0.5 and len(relevant) >= 3 else "MEDIUM" if abs(net) > 0.2 else "LOW"
    return {
        "signal":        signal,
        "confidence":    confidence,
        "net_score":     round(net, 4),
        "article_count": len(relevant),
        "top_headline":  relevant[0]["headline"] if relevant else "",
        "items":         relevant[:5],
    }


if __name__ == "__main__":
    process_feed()
