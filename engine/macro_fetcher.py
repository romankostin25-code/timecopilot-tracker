"""Macro context and regime classification."""

import os
import json
import math
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

CSV_PATH = "data/macro_context.csv"


def _fed_hawkishness_signal(max_age_days: int = 7) -> float:
    """Return a time-decayed average fed_hawkishness from recent NLP-processed articles.

    Positive = hawkish (rate hike language), negative = dovish (easing language).
    Returns 0.0 when no Fed speech data is available.
    """
    feed_path = "data/intelligence_feed.json"
    if not os.path.exists(feed_path):
        return 0.0
    try:
        articles = json.loads(open(feed_path).read())
    except Exception:
        return 0.0

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_str = cutoff.isoformat()

    weighted_sum = 0.0
    total_weight = 0.0
    for a in articles:
        hawk = a.get("fed_hawkishness")
        if hawk is None:
            continue
        if not a.get("nlp_processed"):
            continue
        fetched = a.get("fetched_at", "")
        if fetched < cutoff_str:
            continue
        try:
            age_h = (datetime.now(timezone.utc) -
                     datetime.fromisoformat(fetched.replace("Z", "+00:00"))).total_seconds() / 3600
        except Exception:
            continue
        decay = math.exp(-0.05 * age_h)  # slower decay than news (Fed speeches stay relevant longer)
        weighted_sum += float(hawk) * decay
        total_weight += decay

    if total_weight < 0.5:
        return 0.0
    return weighted_sum / total_weight


def fetch_macro_context() -> dict:
    today = str(date.today())
    macro: dict = {"date": today}

    yf_targets = {
        "^VIX": "vix", "^VIX9D": "vix9d", "^VIX3M": "vix3m",
        "^TNX": "us10y", "DX-Y.NYB": "dxy",
        "^GSPC": "sp500", "GC=F": "gold", "CL=F": "oil",
    }
    for yticker, key in yf_targets.items():
        try:
            raw = yf.download(yticker, period="30d", auto_adjust=True, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                closes = raw["Close"].dropna()
                macro[key] = round(float(closes.iloc[-1]), 4)
                if len(closes) >= 6:
                    macro[f"{key}_5d_chg_pct"] = round(
                        (closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6] * 100, 4
                    )
        except Exception as e:
            print(f"[macro] {yticker}: {e}")

    # VIX term structure slope: positive = normal contango, negative = backwardation (panic)
    vix_spot = macro.get("vix", 20)
    vix_3m   = macro.get("vix3m", vix_spot)
    vix_9d   = macro.get("vix9d", vix_spot)
    if vix_spot > 0:
        macro["vix_term_slope"] = round((vix_3m - vix_9d) / vix_spot, 4)
    else:
        macro["vix_term_slope"] = 0.0

    fred_key = os.getenv("FRED_API_KEY", "")
    if fred_key and fred_key not in ("", "your_fred_key_here"):
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            macro["fed_funds_rate"]    = round(float(fred.get_series("FEDFUNDS").iloc[-1]), 4)
            cpi = fred.get_series("CPIAUCSL")
            macro["cpi_yoy_pct"]       = round(float(cpi.pct_change(12).iloc[-1] * 100), 4)
            macro["unemployment_rate"] = round(float(fred.get_series("UNRATE").iloc[-1]), 4)
            print("[macro] ✓ FRED data loaded")
        except Exception as e:
            print(f"[macro] FRED error: {e}")

    vix        = macro.get("vix", 20)
    dxy_chg    = macro.get("dxy_5d_chg_pct", 0) or 0
    us10y      = macro.get("us10y", 4.0)
    term_slope = macro.get("vix_term_slope", 0.0) or 0.0

    # Backwardation (short-term fear > long-term) lowers RISK_ON threshold
    if vix > 22 or term_slope < -0.10:
        risk_regime = "RISK_OFF"
    elif vix < 15 and term_slope >= -0.05:
        risk_regime = "RISK_ON"
    else:
        risk_regime = "NEUTRAL"

    macro["risk_regime"]   = risk_regime
    macro["dollar_regime"] = "DOLLAR_STRENGTH" if dxy_chg > 0.5 else "DOLLAR_WEAKNESS" if dxy_chg < -0.5 else "DOLLAR_NEUTRAL"

    # Base rate regime from yield level
    rate_regime = "HAWKISH" if us10y > 4.5 else "DOVISH" if us10y < 3.5 else "NEUTRAL_RATES"

    # Override with Fed speech hawkishness signal when strong enough
    fed_hawk = _fed_hawkishness_signal()
    us10y_5d = macro.get("us10y_5d_chg_pct", 0.0) or 0.0
    macro["fed_hawkishness_avg"] = round(fed_hawk, 4)
    if fed_hawk > 0.35 and us10y_5d > -1.5:
        rate_regime = "HAWKISH"
    elif fed_hawk < -0.35:
        rate_regime = "EASING"
    elif fed_hawk > 0.20 and rate_regime == "NEUTRAL_RATES" and us10y_5d > -1.5:
        rate_regime = "HAWKISH"
    elif fed_hawk < -0.20 and rate_regime == "NEUTRAL_RATES":
        rate_regime = "DOVISH"

    macro["rate_regime"]  = rate_regime
    macro["generated_at"] = datetime.now().isoformat()

    os.makedirs("data", exist_ok=True)
    new_df = pd.DataFrame([macro])
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        existing = existing[existing["date"].astype(str) != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(CSV_PATH, index=False)

    print(f"✓ macro_context.csv | VIX={vix} | {macro['risk_regime']} | {macro['dollar_regime']} | {macro['rate_regime']} | fed_hawk={macro['fed_hawkishness_avg']:+.3f}")
    return macro


if __name__ == "__main__":
    fetch_macro_context()
