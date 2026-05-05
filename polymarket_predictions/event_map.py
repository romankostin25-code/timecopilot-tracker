"""
Causal event map: Polymarket event type → affected assets → expected direction
Direction conventions:
  +1 = event being more likely pushes asset price UP
  -1 = event being more likely pushes asset price DOWN
   0 = ambiguous / context-dependent
"""

EVENT_MAP = {

    # ── FEDERAL RESERVE ──────────────────────────────────────────────────────
    "fed_rate_cut": {
        "keywords": ["fed rate cut", "federal reserve cut", "fomc cut", "rate cut",
                     "fed funds rate decrease", "fed pivot", "fed lower",
                     "cut rates", "cutting rates", "50 basis", "25 basis point cut"],
        "assets": {
            "GC=F":      +1,
            "SI=F":      +1,
            "DX-Y.NYB":  -1,
            "^TNX":      -1,
            "EURUSD=X":  +1,
            "GBPUSD=X":  +1,
            "AUDUSD=X":  +1,
            "USDJPY=X":  -1,
            "^GSPC":     +1,
            "CL=F":      +1,
            "HG=F":      +1,
        },
        "lag_days": [1, 3, 5],
        "typical_impact_pct": {
            "GC=F": 0.8, "DX-Y.NYB": -0.6, "^GSPC": 1.2, "CL=F": 0.5
        }
    },

    "fed_rate_hike": {
        "keywords": ["fed rate hike", "federal reserve hike", "fomc hike",
                     "rate increase", "fed tightening", "raise rates", "hiking rates",
                     "25 basis point hike", "50 basis point hike"],
        "assets": {
            "GC=F":      -1,
            "SI=F":      -1,
            "DX-Y.NYB":  +1,
            "^TNX":      +1,
            "EURUSD=X":  -1,
            "GBPUSD=X":  -1,
            "AUDUSD=X":  -1,
            "USDJPY=X":  +1,
            "^GSPC":     -1,
            "CL=F":      -1,
            "HG=F":      -1,
        },
        "lag_days": [1, 3, 5],
        "typical_impact_pct": {
            "GC=F": -0.8, "DX-Y.NYB": 0.6, "^GSPC": -1.2
        }
    },

    "fed_rate_hold": {
        "keywords": ["fed hold", "fomc pause", "no rate change", "rates unchanged",
                     "fed on hold", "hold rates", "pause rate"],
        "assets": {
            "GC=F":      0,
            "DX-Y.NYB":  0,
            "^GSPC":     0.3,
        },
        "lag_days": [1],
        "typical_impact_pct": {}
    },

    # ── INFLATION / CPI ──────────────────────────────────────────────────────
    "cpi_above_target": {
        "keywords": ["cpi above", "inflation above", "cpi higher than expected",
                     "hot inflation", "inflation exceeds", "inflation rise",
                     "cpi beat", "inflation surges"],
        "assets": {
            "GC=F":      +1,
            "SI=F":      +1,
            "^TNX":      +1,
            "DX-Y.NYB":  +1,
            "EURUSD=X":  -1,
            "^GSPC":     -1,
            "CL=F":      +0.5,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {
            "GC=F": 0.5, "^TNX": 0.15, "^GSPC": -0.8
        }
    },

    "cpi_below_target": {
        "keywords": ["cpi below", "inflation below", "cpi cooler than expected",
                     "cooling inflation", "disinflation", "cpi miss", "inflation falls",
                     "deflation", "inflation drops"],
        "assets": {
            "GC=F":      -0.5,
            "^TNX":      -1,
            "DX-Y.NYB":  -1,
            "EURUSD=X":  +1,
            "^GSPC":     +1,
            "CL=F":      +0.5,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {
            "^TNX": -0.12, "^GSPC": 0.9, "DX-Y.NYB": -0.4
        }
    },

    # ── RECESSION / GROWTH ───────────────────────────────────────────────────
    "us_recession": {
        "keywords": ["us recession", "recession probability", "gdp contraction",
                     "economic recession", "recession 2025", "recession 2026",
                     "us enter recession", "technical recession", "negative gdp"],
        "assets": {
            "GC=F":      +1,
            "SI=F":      -0.5,
            "HG=F":      -1,
            "CL=F":      -1,
            "NG=F":      -0.5,
            "^GSPC":     -1,
            "DX-Y.NYB":  +0.5,
            "USDJPY=X":  -1,
            "AUDUSD=X":  -1,
        },
        "lag_days": [1, 3, 5, 10],
        "typical_impact_pct": {
            "GC=F": 1.5, "HG=F": -2.0, "CL=F": -3.0, "^GSPC": -2.5
        }
    },

    # ── GEOPOLITICAL / OIL ───────────────────────────────────────────────────
    "middle_east_conflict": {
        "keywords": ["israel", "iran", "middle east war", "strait of hormuz",
                     "oil supply disruption", "opec conflict", "gaza",
                     "iran attack", "israel attack"],
        "assets": {
            "CL=F":      +1,
            "GC=F":      +1,
            "NG=F":      +0.5,
            "^GSPC":     -1,
            "DX-Y.NYB":  +0.5,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {
            "CL=F": 2.0, "GC=F": 1.0, "^GSPC": -1.5
        }
    },

    "russia_ukraine": {
        "keywords": ["russia ukraine", "ukraine war", "nato russia", "russian sanctions",
                     "ukraine ceasefire", "ukraine peace", "russia invade",
                     "ukraine invasion"],
        "assets": {
            "NG=F":      +1,
            "GC=F":      +1,
            "CL=F":      +0.5,
            "EURUSD=X":  -1,
            "GBPUSD=X":  -0.5,
            "^GSPC":     -0.5,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {
            "NG=F": 3.0, "EURUSD=X": -0.8
        }
    },

    # ── CENTRAL BANKS (NON-FED) ──────────────────────────────────────────────
    "ecb_rate_cut": {
        "keywords": ["ecb cut", "ecb rate cut", "european central bank cut",
                     "ecb dovish", "ecb lower", "lagarde cut"],
        "assets": {
            "EURUSD=X":  -1,
            "GBPUSD=X":  -0.3,
            "^GSPC":     +0.3,
            "GC=F":      +0.3,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {
            "EURUSD=X": -0.7
        }
    },

    "boj_rate_hike": {
        "keywords": ["boj hike", "bank of japan hike", "japan rate increase",
                     "boj tightening", "boj raise", "japan raise rates"],
        "assets": {
            "USDJPY=X":  -1,
            "GC=F":      -0.3,
            "^GSPC":     -0.5,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {
            "USDJPY=X": -1.5
        }
    },

    # ── COMMODITY-SPECIFIC ───────────────────────────────────────────────────
    "oil_above_threshold": {
        "keywords": ["oil above", "crude above", "wti above", "oil price above",
                     "brent above", "oil hits", "crude oil reaches"],
        "assets": {
            "CL=F":      +1,
            "GC=F":      +0.3,
            "^GSPC":     -0.5,
            "EURUSD=X":  -0.3,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {"CL=F": 1.5}
    },

    "oil_below_threshold": {
        "keywords": ["oil below", "crude below", "wti below", "oil crash",
                     "oil price falls", "crude oil drops"],
        "assets": {
            "CL=F":      -1,
            "^GSPC":     +0.3,
            "DX-Y.NYB":  +0.3,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {"CL=F": -1.5}
    },

    "gold_above_threshold": {
        "keywords": ["gold above", "gold price above", "gold hits", "gold reaches",
                     "gold price hits", "gold surges", "gold rally"],
        "assets": {
            "GC=F":      +1,
            "SI=F":      +0.7,
            "DX-Y.NYB":  -0.3,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {"GC=F": 1.0}
    },

    # ── US POLITICS / POLICY ─────────────────────────────────────────────────
    "trump_tariffs": {
        "keywords": ["tariffs", "trade war", "trump tariff", "china tariffs",
                     "import duties", "trade deficit", "25% tariff", "tariff on",
                     "new tariffs", "tariff increase", "tariff announcement"],
        "assets": {
            "HG=F":      -1,
            "^GSPC":     -1,
            "DX-Y.NYB":  +0.5,
            "AUDUSD=X":  -1,
            "CL=F":      -0.5,
            "GC=F":      +0.5,
        },
        "lag_days": [1, 3, 5],
        "typical_impact_pct": {
            "HG=F": -2.0, "^GSPC": -1.5, "AUDUSD=X": -0.8
        }
    },

    "us_debt_ceiling": {
        "keywords": ["debt ceiling", "us default", "treasury default",
                     "government shutdown", "debt limit", "hit debt ceiling"],
        "assets": {
            "GC=F":      +1,
            "DX-Y.NYB":  -1,
            "^TNX":      +1,
            "^GSPC":     -1,
            "USDJPY=X":  -1,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {
            "GC=F": 1.2, "^GSPC": -2.0
        }
    },

    "china_taiwan": {
        "keywords": ["china invade taiwan", "taiwan invasion", "china taiwan war",
                     "taiwan strait", "china military taiwan", "china blockade taiwan"],
        "assets": {
            "GC=F":      +1,
            "HG=F":      -1,
            "^GSPC":     -1,
            "AUDUSD=X":  -1,
            "DX-Y.NYB":  +0.5,
            "CL=F":      +0.5,
        },
        "lag_days": [1, 3, 5],
        "typical_impact_pct": {
            "GC=F": 2.0, "^GSPC": -3.0, "HG=F": -3.0
        }
    },

    "bitcoin_rally": {
        "keywords": ["bitcoin above", "bitcoin hits", "bitcoin reaches", "btc above",
                     "bitcoin price", "crypto rally", "bitcoin ath"],
        "assets": {
            "GC=F":      +0.3,
            "^GSPC":     +0.3,
        },
        "lag_days": [1, 3],
        "typical_impact_pct": {}
    },
}


def get_events_for_asset(ticker: str) -> dict:
    return {
        event_type: data
        for event_type, data in EVENT_MAP.items()
        if ticker in data["assets"]
    }


def get_direction_multiplier(event_type: str, ticker: str) -> float:
    if event_type not in EVENT_MAP:
        return 0
    return EVENT_MAP[event_type]["assets"].get(ticker, 0)


def all_keywords() -> list[tuple[str, str]]:
    """Return list of (keyword, event_type) pairs for all events."""
    result = []
    for event_type, data in EVENT_MAP.items():
        for kw in data["keywords"]:
            result.append((kw.lower(), event_type))
    return result
