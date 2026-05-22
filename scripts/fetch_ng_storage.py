"""Fetch EIA weekly natural gas storage and generate a NG=F directional signal.

The EIA weekly storage report (Thursdays) is the single most predictive signal
for natural gas prices. A storage surplus vs. year-ago = bearish (supply glut);
a deficit vs. year-ago = bullish (supply squeeze).

Signal: YoY storage change normalized to [-1, 1].
  > +0.5  = significant surplus  → bearish NG  (signal negative)
  < -0.5  = significant deficit  → bullish NG  (signal positive)

Data sources (tried in order):
  1. EIA API v1 — requires free EIA_API_KEY from eia.gov/opendata
  2. FRED NGSC series — requires FRED_API_KEY
  3. Returns 0.0 signal if neither key is available
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH   = Path("data/ng_storage.json")
EIA_SERIES_ID = "NG.NW2_EPG0_SWO_R48_BCF.W"   # Lower 48 weekly working gas in storage (Bcf)
FRED_SERIES   = "NGSC"                          # FRED: natural gas in underground storage


def _fetch_eia(api_key: str) -> pd.Series | None:
    url = "https://api.eia.gov/series/"
    try:
        r = requests.get(url, params={
            "api_key":   api_key,
            "series_id": EIA_SERIES_ID,
            "num":       60,
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = data.get("series", [{}])[0].get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["period", "value"])
        df["date"]  = pd.to_datetime(df["period"], format="%Y%m%d", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date").sort_index()
        print(f"[NG storage] EIA: {len(df)} weeks loaded")
        return df["value"]
    except Exception as e:
        print(f"[NG storage] EIA fetch failed: {e}")
        return None


def _fetch_fred() -> pd.Series | None:
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key or fred_key in ("", "your_fred_key_here"):
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        s = fred.get_series(FRED_SERIES)
        print(f"[NG storage] FRED: {len(s)} weeks loaded")
        return s
    except Exception as e:
        print(f"[NG storage] FRED fetch failed: {e}")
        return None


def _compute_signal(series: pd.Series) -> dict:
    """YoY storage change → contrarian signal for NG=F.

    Returns signal in [-1, 1]:
      Surplus (storage above YoY) → negative (bearish NG)
      Deficit (storage below YoY) → positive (bullish NG)
    """
    s = series.dropna().sort_index()
    if len(s) < 54:
        return {"signal": 0.0, "yoy_pct": None, "bcf_current": None, "bcf_yoy": None}

    current     = float(s.iloc[-1])
    yoy_approx  = s.iloc[-54:-52].mean() if len(s) >= 54 else current
    yoy_pct     = (current - yoy_approx) / yoy_approx * 100 if yoy_approx != 0 else 0.0

    # Normalize: ±10% YoY maps to ±1.0 signal; inverted (surplus=bearish)
    raw_signal  = float(np.clip(-yoy_pct / 10.0, -1.0, 1.0))

    if raw_signal > 0.5:
        label = "SUPPLY_DEFICIT_BULLISH"
    elif raw_signal < -0.5:
        label = "SUPPLY_SURPLUS_BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "signal":      round(raw_signal, 4),
        "signal_label": label,
        "yoy_pct":     round(yoy_pct, 2),
        "bcf_current": round(current, 1),
        "bcf_yoy":     round(yoy_approx, 1),
    }


def fetch_ng_storage() -> dict:
    print("[NG storage] Fetching EIA weekly natural gas storage...")

    series = None
    source = "none"

    eia_key = os.getenv("EIA_API_KEY", "")
    if eia_key:
        series = _fetch_eia(eia_key)
        if series is not None:
            source = "EIA"

    if series is None:
        series = _fetch_fred()
        if series is not None:
            source = "FRED"

    if series is None:
        print("[NG storage] No data source available — signal=0.0")
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source":       "none",
            "signal":       0.0,
            "signal_label": "NO_DATA",
        }
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2))
        return result

    sig = _compute_signal(series)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source":       source,
        **sig,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[NG storage] YoY={sig.get('yoy_pct', 'N/A'):+.1f}%  "
          f"current={sig.get('bcf_current', 'N/A')} Bcf  "
          f"signal={sig.get('signal', 0.0):+.3f}  → {sig.get('signal_label', '')}")
    return result


if __name__ == "__main__":
    fetch_ng_storage()
