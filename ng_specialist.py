"""Natural Gas specialist: seasonal adjustments + EIA storage signal."""

import os
import warnings
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# Monthly seasonal multipliers: >1.0 = bullish season, <1.0 = bearish
_SEASONAL = {
    1: 1.15, 2: 1.10, 3: 0.92, 4: 0.82, 5: 0.88,
    6: 0.98, 7: 1.04, 8: 1.04, 9: 0.93, 10: 1.00,
    11: 1.10, 12: 1.20,
}


def get_seasonal_factor(target_date: date) -> float:
    return _SEASONAL.get(target_date.month, 1.0)


def get_eia_storage_signal(fred_key: str = None) -> dict:
    """Fetch EIA weekly natural gas storage and compute supply/demand signal."""
    base = {"eia_storage_bcf": None, "eia_5yr_avg": None,
            "eia_deviation_pct": None, "eia_signal": "NEUTRAL", "eia_signal_enc": 0}

    fred_key = fred_key or os.getenv("FRED_API_KEY", "")
    if not fred_key or fred_key == "your_fred_key_here":
        return base

    try:
        from fredapi import Fred
        import pandas as pd
        fred    = Fred(api_key=fred_key)
        start   = str(date.today() - timedelta(days=365 * 6))
        storage = fred.get_series("NGWIUS", observation_start=start)
        if storage.empty:
            return base

        current      = float(storage.iloc[-1])
        week_num     = storage.index[-1].isocalendar()[1]
        hist_same_wk = storage[storage.index.isocalendar().week == week_num]
        avg_5yr      = float(hist_same_wk.tail(5).mean())
        dev_pct      = (current - avg_5yr) / avg_5yr * 100

        # High storage → oversupply → bearish; low storage → bullish
        if dev_pct > 5:
            signal = "BEARISH"
        elif dev_pct < -5:
            signal = "BULLISH"
        else:
            signal = "NEUTRAL"

        return {
            "eia_storage_bcf":   round(current, 1),
            "eia_5yr_avg":       round(avg_5yr, 1),
            "eia_deviation_pct": round(dev_pct, 2),
            "eia_signal":        signal,
            "eia_signal_enc":    1 if signal == "BULLISH" else -1 if signal == "BEARISH" else 0,
        }
    except Exception as e:
        print(f"  [ng_specialist] EIA error: {e}")
        return base


def ng_adjust_features(base_features: dict, target_date: date = None) -> dict:
    """Inject NG-specific seasonal + EIA features into a feature dict."""
    features = base_features.copy()
    td = target_date or date.today()
    features["ng_seasonal"]  = get_seasonal_factor(td)
    features.update(get_eia_storage_signal())
    return features


def ng_adjust_signal(combined_result: dict, target_date: date = None) -> dict:
    """Apply seasonal and EIA pressure to an already-computed signal dict."""
    td       = target_date or date.today()
    seasonal = get_seasonal_factor(td)
    eia      = get_eia_storage_signal()

    result = combined_result.copy()

    # Nudge combined_prob_up by seasonal and EIA
    prob = result.get("combined_prob_up", 0.5)
    seasonal_nudge = (seasonal - 1.0) * 0.10   # ±10% seasonal influence
    eia_nudge      = eia["eia_signal_enc"] * 0.05

    prob_adj = max(0.05, min(0.95, prob + seasonal_nudge + eia_nudge))
    result["combined_prob_up"] = round(prob_adj, 4)

    # Recompute direction from adjusted probability
    from conviction_thresholds import passes_conviction_gate
    if not passes_conviction_gate("NG=F", prob_adj):
        result["direction"] = "NEUTRAL"
    elif prob_adj > 0.5:
        result["direction"] = "BULLISH"
    else:
        result["direction"] = "BEARISH"

    result["conviction_score"] = round(abs(prob_adj - 0.5) * 2.0, 4)
    result["ng_seasonal"]      = seasonal
    result["eia_signal"]       = eia["eia_signal"]
    return result
