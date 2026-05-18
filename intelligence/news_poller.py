"""RSS news ingestion — 17 feeds, 48h window."""

import feedparser
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

FEED_PATH = Path("data/intelligence_feed.json")
FEED_PATH.parent.mkdir(exist_ok=True)
MAX_AGE_HOURS = 48

FEEDS = [
    {"url": "https://feeds.reuters.com/reuters/businessNews",        "source": "Reuters",          "type": "macro"},
    {"url": "https://www.cnbc.com/id/100727362/device/rss/rss.html", "source": "CNBC",             "type": "macro"},
    {"url": "https://finance.yahoo.com/news/rssindex",               "source": "Yahoo Finance",    "type": "macro"},
    {"url": "https://feeds.marketwatch.com/marketwatch/topstories",  "source": "MarketWatch",      "type": "equities"},
    {"url": "https://www.federalreserve.gov/feeds/press_all.xml",    "source": "Federal Reserve",  "type": "central_bank"},
    {"url": "https://www.ecb.europa.eu/rss/press.html",              "source": "ECB",              "type": "central_bank"},
    {"url": "https://www.eia.gov/rss/news.xml",                      "source": "EIA",              "type": "energy"},
    {"url": "https://feeds.reuters.com/reuters/commoditiesNews",     "source": "Reuters Commodities", "type": "commodities"},
    {"url": "https://oilprice.com/rss/main",                         "source": "OilPrice",         "type": "energy"},
    {"url": "https://cointelegraph.com/rss",                         "source": "CoinTelegraph",    "type": "crypto"},
    {"url": "https://decrypt.co/feed",                               "source": "Decrypt",          "type": "crypto"},
    {"url": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines", "source": "MarketWatch RT", "type": "macro"},
    {"url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom",
                                                                     "source": "SEC EDGAR 8-K",    "type": "sec"},
    {"url": "https://feeds.reuters.com/Reuters/worldNews",           "source": "Reuters World",    "type": "geopolitics"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml","source": "NYT World",        "type": "geopolitics"},
    {"url": "https://feeds.washingtonpost.com/rss/world",            "source": "WashPost World",   "type": "geopolitics"},
    {"url": "https://www.ft.com/markets?format=rss",                 "source": "FT",               "type": "macro"},
]


def fetch_all_feeds():
    articles = []
    seen = set()
    for feed_cfg in FEEDS:
        try:
            feed = feedparser.parse(feed_cfg["url"])
            for entry in feed.entries[:20]:
                url = entry.get("link", "")
                if not url:
                    continue
                aid = hashlib.sha256(url.encode()).hexdigest()[:16]
                if aid in seen:
                    continue
                seen.add(aid)
                articles.append({
                    "id":          aid,
                    "source":      feed_cfg["source"],
                    "source_type": feed_cfg["type"],
                    "headline":    entry.get("title", ""),
                    "url":         url,
                    "summary_raw": entry.get("summary", "")[:500],
                    "published_at": entry.get("published", entry.get("updated", "")),
                    "fetched_at":  datetime.now(timezone.utc).isoformat(),
                    "assets_affected": [],
                    "sentiment": None,
                    "sentiment_confidence": None,
                    "event_type": None,
                    "impact_horizon": None,
                    "summary": None,
                    "signal_direction": {},
                    "signal_strength": {},
                    "decay_weight": 1.0,
                    "nlp_processed": False,
                })
        except Exception as e:
            print(f"[news_poller] {feed_cfg['source']}: {e}")
    return articles


def update_feed(new_articles):
    existing = json.loads(FEED_PATH.read_text()) if FEED_PATH.exists() else []
    existing_ids = {a["id"] for a in existing}
    added = [a for a in new_articles if a["id"] not in existing_ids]

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
    filtered = [
        a for a in existing
        if a.get("fetched_at") and
        datetime.fromisoformat(a["fetched_at"].replace("Z", "+00:00")) > cutoff
    ]
    combined = sorted(filtered + added, key=lambda x: x.get("published_at", ""), reverse=True)
    FEED_PATH.write_text(json.dumps(combined, indent=2, default=str))
    print(f"[news_poller] +{len(added)} articles. Feed: {len(combined)} total.")
    return added


if __name__ == "__main__":
    update_feed(fetch_all_feeds())
