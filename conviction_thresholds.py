"""ATR-based direction thresholds and conviction gates per forecastability tier."""

TIER_CONFIG = {
    "TIER_1": {
        "tickers":      ["^VIX", "GC=F", "^TNX", "^IRX", "DX-Y.NYB"],
        "atr_mult":     0.30,
        "conviction_gate": 0.54,
    },
    "TIER_2": {
        "tickers":      ["CL=F", "HG=F", "EURUSD=X", "GBPUSD=X", "USDJPY=X",
                         "AUDUSD=X", "^GSPC", "HYG"],
        "atr_mult":     0.35,
        "conviction_gate": 0.56,
    },
    "TIER_3": {
        "tickers":      ["NG=F", "SI=F", "PA=F", "HO=F"],
        "atr_mult":     0.40,
        "conviction_gate": 0.58,
    },
}

_TICKER_TIER: dict[str, str] = {}
for _tier, _cfg in TIER_CONFIG.items():
    for _t in _cfg["tickers"]:
        _TICKER_TIER[_t] = _tier


def get_ticker_tier(ticker: str) -> str:
    return _TICKER_TIER.get(ticker, "TIER_2")


def atr_direction_threshold(ticker: str, atr_14_pct: float) -> float:
    """Minimum absolute expected return to issue a directional call."""
    mult = TIER_CONFIG[get_ticker_tier(ticker)]["atr_mult"]
    return mult * max(atr_14_pct, 0.002)


def is_directional(ticker: str, expected_return: float, atr_14_pct: float) -> str:
    thresh = atr_direction_threshold(ticker, atr_14_pct)
    if expected_return > thresh:
        return "BULLISH"
    if expected_return < -thresh:
        return "BEARISH"
    return "NEUTRAL"


def passes_conviction_gate(ticker: str, combined_prob: float) -> bool:
    """True when probability is far enough from 0.5 to warrant a call."""
    gate = TIER_CONFIG[get_ticker_tier(ticker)]["conviction_gate"]
    return combined_prob > gate or combined_prob < (1.0 - gate)
