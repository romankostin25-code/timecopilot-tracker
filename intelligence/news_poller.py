"""RSS news ingestion — expanded feeds across crypto, energy, equities, CB."""

import feedparser
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

FEED_PATH = Path("data/intelligence_feed.json")
FEED_PATH.parent.mkdir(exist_ok=True)
MAX_AGE_HOURS = 48

FEEDS = [
    # ── MACRO / GENERAL ──────────────────────────────────────────────────────
    {"url": "https://feeds.reuters.com/reuters/businessNews",           "source": "Reuters Business",     "type": "macro"},
    {"url": "https://www.cnbc.com/id/100727362/device/rss/rss.html",   "source": "CNBC Markets",          "type": "macro"},
    {"url": "https://finance.yahoo.com/news/rssindex",                  "source": "Yahoo Finance",         "type": "macro"},
    {"url": "https://feeds.marketwatch.com/marketwatch/topstories",     "source": "MarketWatch",           "type": "macro"},
    {"url": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines","source":"MarketWatch RT",       "type": "macro"},
    {"url": "https://www.ft.com/markets?format=rss",                    "source": "Financial Times",       "type": "macro"},
    {"url": "https://feeds.reuters.com/Reuters/worldNews",              "source": "Reuters World",         "type": "geopolitics"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",  "source": "NYT World",             "type": "geopolitics"},
    {"url": "https://feeds.washingtonpost.com/rss/world",              "source": "WashPost World",        "type": "geopolitics"},

    # ── EQUITIES ─────────────────────────────────────────────────────────────
    {"url": "https://feeds.reuters.com/reuters/companyNews",            "source": "Reuters Companies",     "type": "equities"},
    {"url": "https://www.investing.com/rss/news_25.rss",               "source": "Investing.com Stocks",  "type": "equities"},
    {"url": "https://www.investors.com/feed/",                         "source": "IBD",                   "type": "equities"},
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",          "source": "WSJ Markets",           "type": "equities"},
    {"url": "https://www.barrons.com/xml/rss/3_7520.xml",             "source": "Barron's",              "type": "equities"},
    {"url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom",
                                                                        "source": "SEC EDGAR 8-K",         "type": "sec"},

    # ── CENTRAL BANKS ────────────────────────────────────────────────────────
    {"url": "https://www.federalreserve.gov/feeds/press_all.xml",      "source": "Federal Reserve",       "type": "central_bank"},
    {"url": "https://www.ecb.europa.eu/rss/press.html",                "source": "ECB",                   "type": "central_bank"},
    {"url": "https://www.bankofengland.co.uk/rss/news",                "source": "Bank of England",       "type": "central_bank"},
    {"url": "https://www.bis.org/doclist/speeches.rss",                "source": "BIS Speeches",          "type": "central_bank"},
    {"url": "https://www.imf.org/en/News/RSS",                         "source": "IMF",                   "type": "central_bank"},
    {"url": "https://www.boj.or.jp/en/announcements/release_2024/rss.xml","source":"Bank of Japan",       "type": "central_bank"},

    # ── ENERGY ───────────────────────────────────────────────────────────────
    {"url": "https://www.eia.gov/rss/news.xml",                        "source": "EIA",                   "type": "energy"},
    {"url": "https://feeds.reuters.com/reuters/commoditiesNews",       "source": "Reuters Commodities",   "type": "energy"},
    {"url": "https://oilprice.com/rss/main",                           "source": "OilPrice",              "type": "energy"},
    {"url": "https://www.rigzone.com/news/rss/rigzone_latest.aspx",    "source": "Rigzone",               "type": "energy"},
    {"url": "https://www.naturalgasintelligence.com/feed/",            "source": "NGI",                   "type": "energy"},
    {"url": "https://www.spglobal.com/commodityinsights/en/rss-feed/oil", "source":"S&P Global Oil",      "type": "energy"},
    {"url": "https://www.energymonitor.ai/feed/",                      "source": "Energy Monitor",        "type": "energy"},

    # ── CRYPTO ───────────────────────────────────────────────────────────────
    {"url": "https://cointelegraph.com/rss",                           "source": "CoinTelegraph",         "type": "crypto"},
    {"url": "https://decrypt.co/feed",                                 "source": "Decrypt",               "type": "crypto"},
    {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/",        "source": "CoinDesk",              "type": "crypto"},
    {"url": "https://bitcoinmagazine.com/.rss/full/",                  "source": "Bitcoin Magazine",      "type": "crypto"},
    {"url": "https://theblock.co/rss.xml",                            "source": "The Block",             "type": "crypto"},
    {"url": "https://cryptoslate.com/feed/",                          "source": "CryptoSlate",           "type": "crypto"},
    {"url": "https://cryptobriefing.com/feed/",                       "source": "Crypto Briefing",       "type": "crypto"},
]


def fetch_all_feeds():
    articles = []
    seen = set()
    for feed_cfg in FEEDS:
        try:
            feed = feedparser.parse(feed_cfg["url"])
            for entry in feed.entries[:15]:
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
