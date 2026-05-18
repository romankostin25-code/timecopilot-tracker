UNIVERSE = {
    # ── EQUITY ETFs ──────────────────────────────────────────────────────────
    "SPY":  {"name": "S&P 500 ETF",              "class": "equity_etf",   "curve": False},
    "QQQ":  {"name": "Nasdaq 100 ETF",            "class": "equity_etf",   "curve": False},
    "IWM":  {"name": "Russell 2000 ETF",          "class": "equity_etf",   "curve": False},
    "XLE":  {"name": "Energy Select ETF",         "class": "equity_etf",   "curve": False},
    "XLF":  {"name": "Financials Select ETF",     "class": "equity_etf",   "curve": False},
    "XLK":  {"name": "Technology Select ETF",     "class": "equity_etf",   "curve": False},
    "GLD":  {"name": "Gold ETF (GLD)",            "class": "equity_etf",   "curve": False},
    "SLV":  {"name": "Silver ETF (SLV)",          "class": "equity_etf",   "curve": False},

    # ── FIXED INCOME ETFs ────────────────────────────────────────────────────
    "TLT":  {"name": "20Y+ Treasury ETF",         "class": "fixed_income", "curve": False},
    "IEF":  {"name": "7-10Y Treasury ETF",        "class": "fixed_income", "curve": False},
    "HYG":  {"name": "High Yield Bond ETF",       "class": "fixed_income", "curve": False},
    "LQD":  {"name": "Investment Grade Bond ETF", "class": "fixed_income", "curve": False},
    "BIL":  {"name": "1-3 Month T-Bill ETF",      "class": "fixed_income", "curve": False},
    "^TNX": {"name": "US 10Y Treasury Yield",     "class": "rates",        "curve": False},

    # ── COMMODITY FUTURES ────────────────────────────────────────────────────
    "GC=F": {"name": "Gold Futures",              "class": "commodity",    "curve": True},
    "SI=F": {"name": "Silver Futures",            "class": "commodity",    "curve": True},
    "CL=F": {"name": "WTI Crude Oil",             "class": "commodity",    "curve": True},
    "NG=F": {"name": "Natural Gas",               "class": "commodity",    "curve": True},
    "HG=F": {"name": "Copper Futures",            "class": "commodity",    "curve": True},
    "ZW=F": {"name": "Wheat Futures",             "class": "commodity",    "curve": True},
    "ZC=F": {"name": "Corn Futures",              "class": "commodity",    "curve": True},

    # ── VOLATILITY ───────────────────────────────────────────────────────────
    "^VIX": {"name": "CBOE Volatility Index",     "class": "volatility",   "curve": False},
    "UVXY": {"name": "ProShares Ultra VIX ETF",   "class": "volatility",   "curve": False},

    # ── MACRO / RATES ────────────────────────────────────────────────────────
    "DX-Y.NYB": {"name": "US Dollar Index",       "class": "macro",        "curve": False},
    "^IRX":     {"name": "13-Week T-Bill Yield",  "class": "rates",        "curve": False},

    # ── CRYPTO ───────────────────────────────────────────────────────────────
    "BTC-USD": {"name": "Bitcoin",                "class": "crypto",       "curve": False},
    "ETH-USD": {"name": "Ethereum",               "class": "crypto",       "curve": False},
    "SOL-USD": {"name": "Solana",                 "class": "crypto",       "curve": False},

    # ── INTERNATIONAL EQUITY ETFs ────────────────────────────────────────────
    "EEM":  {"name": "Emerging Markets ETF",      "class": "equity_etf",   "curve": False},
    "EFA":  {"name": "EAFE Developed Mkts ETF",   "class": "equity_etf",   "curve": False},
    "FXI":  {"name": "China Large-Cap ETF",       "class": "equity_etf",   "curve": False},
}

ALL_TICKERS = list(UNIVERSE.keys())

ASSET_GROUPS = {
    "Equity ETFs":   [t for t, d in UNIVERSE.items() if d["class"] == "equity_etf"],
    "Fixed Income":  [t for t, d in UNIVERSE.items() if d["class"] in ("fixed_income", "rates")],
    "Commodities":   [t for t, d in UNIVERSE.items() if d["class"] == "commodity"],
    "Volatility":    [t for t, d in UNIVERSE.items() if d["class"] == "volatility"],
    "Crypto":        [t for t, d in UNIVERSE.items() if d["class"] == "crypto"],
    "Macro":         [t for t, d in UNIVERSE.items() if d["class"] == "macro"],
    "International": [t for t in ("EEM", "EFA", "FXI")],
}

CURVE_ASSETS  = [t for t, d in UNIVERSE.items() if d["curve"]]
CRYPTO_TICKERS = [t for t, d in UNIVERSE.items() if d["class"] == "crypto"]
