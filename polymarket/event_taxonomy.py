"""Canonical macro event types — regime definitions and asset priors."""

MACRO_REGIMES = {
    "REGIME_EASING": {
        "description": "Fed or major CB cutting rates / dovish pivot",
        "polymarket_themes": ["fed rate cut", "fomc cut", "rate cut", "fed pivot", "rate decrease"],
        "regime_effect": {"risk_sentiment": +1, "dollar_direction": -1, "yield_direction": -1, "gold_bias": +1, "equity_bias": +1, "crypto_bias": +1},
        "asset_priors": {
            "SPY": +1, "QQQ": +1, "IWM": +1, "GLD": +1, "GC=F": +1,
            "TLT": +1, "IEF": +1, "HYG": +1, "LQD": +1,
            "BTC-USD": +1, "ETH-USD": +1, "SOL-USD": +0.5,
            "DX-Y.NYB": -1, "^TNX": -1, "CL=F": +0.5, "HG=F": +0.5,
        },
    },
    "REGIME_TIGHTENING": {
        "description": "Fed hiking or holding hawkish",
        "polymarket_themes": ["fed rate hike", "rate increase", "fomc hike", "hawkish fed", "rates higher"],
        "regime_effect": {"risk_sentiment": -1, "dollar_direction": +1, "yield_direction": +1, "gold_bias": -1, "equity_bias": -1, "crypto_bias": -1},
        "asset_priors": {
            "SPY": -1, "QQQ": -1, "IWM": -1, "GLD": -1, "GC=F": -1,
            "TLT": -1, "IEF": -0.5, "HYG": -1, "LQD": -0.5,
            "BTC-USD": -1, "ETH-USD": -1, "SOL-USD": -1,
            "DX-Y.NYB": +1, "^TNX": +1, "BIL": +0.5,
            "CL=F": -0.5, "HG=F": -0.5,
        },
    },
    "REGIME_RECESSION": {
        "description": "US or global recession probability elevated",
        "polymarket_themes": ["recession", "gdp contraction", "economic downturn", "recession 2025", "recession 2026"],
        "regime_effect": {"risk_sentiment": -1, "dollar_direction": +0.5, "yield_direction": -1, "gold_bias": +1, "equity_bias": -1, "crypto_bias": -1},
        "asset_priors": {
            "SPY": -1, "QQQ": -1, "IWM": -1, "XLE": -1, "XLF": -1,
            "GLD": +1, "GC=F": +1, "TLT": +1, "IEF": +0.5, "BIL": +1,
            "HYG": -1, "LQD": -0.5,
            "BTC-USD": -1, "ETH-USD": -1, "SOL-USD": -1,
            "CL=F": -1, "HG=F": -1, "NG=F": -0.5,
            "DX-Y.NYB": +0.5, "^VIX": +1, "UVXY": +1,
            "EEM": -1, "EFA": -0.5, "FXI": -1,
        },
    },
    "REGIME_INFLATION": {
        "description": "Inflation running above target / sticky",
        "polymarket_themes": ["inflation above", "cpi above", "inflation exceeds", "hot cpi", "stagflation"],
        "regime_effect": {"risk_sentiment": -0.5, "dollar_direction": +1, "yield_direction": +1, "gold_bias": +1, "equity_bias": -0.5},
        "asset_priors": {
            "GLD": +1, "GC=F": +1, "SI=F": +0.5, "SLV": +0.5,
            "TLT": -1, "IEF": -0.5, "^TNX": +1,
            "SPY": -0.5, "QQQ": -0.5,
            "DX-Y.NYB": +0.5, "CL=F": +0.5, "ZW=F": +0.5, "ZC=F": +0.5,
        },
    },
    "REGIME_DISINFLATION": {
        "description": "Inflation cooling toward target",
        "polymarket_themes": ["inflation cooling", "cpi below", "disinflation", "inflation falling"],
        "regime_effect": {"risk_sentiment": +0.5, "dollar_direction": -0.5, "yield_direction": -1, "gold_bias": -0.5, "equity_bias": +1},
        "asset_priors": {
            "SPY": +1, "QQQ": +1, "TLT": +1, "IEF": +0.5,
            "HYG": +0.5, "LQD": +0.5, "^TNX": -1, "DX-Y.NYB": -0.5,
            "BTC-USD": +0.5, "ETH-USD": +0.5,
        },
    },
    "REGIME_GEOPOLITICAL": {
        "description": "Active geopolitical conflict / supply disruption",
        "polymarket_themes": ["war", "conflict", "military", "iran", "russia", "middle east", "taiwan", "ukraine"],
        "regime_effect": {"risk_sentiment": -1, "dollar_direction": +0.5, "gold_bias": +1, "equity_bias": -0.5},
        "asset_priors": {
            "GLD": +1, "GC=F": +1, "CL=F": +1, "NG=F": +1,
            "SPY": -0.5, "QQQ": -0.5,
            "DX-Y.NYB": +0.5, "^VIX": +1, "UVXY": +1,
            "EEM": -1, "EFA": -0.5,
        },
    },
    "REGIME_CRYPTO_RISK": {
        "description": "Crypto-specific risk — regulation, exchange failure",
        "polymarket_themes": ["crypto regulation", "sec bitcoin", "crypto ban"],
        "regime_effect": {"crypto_bias": -1},
        "asset_priors": {"BTC-USD": -1, "ETH-USD": -1, "SOL-USD": -1},
    },
    "REGIME_CRYPTO_BULL": {
        "description": "Crypto-positive catalyst — ETF, adoption, halving",
        "polymarket_themes": ["bitcoin above", "btc above", "bitcoin etf approved", "crypto rally"],
        "regime_effect": {"crypto_bias": +1},
        "asset_priors": {"BTC-USD": +1, "ETH-USD": +1, "SOL-USD": +1},
    },
    "REGIME_ELECTION": {
        "description": "US election or major political transition",
        "polymarket_themes": ["president", "election", "trump", "harris", "democrat", "republican"],
        "regime_effect": {},
        "asset_priors": {},
    },
}
