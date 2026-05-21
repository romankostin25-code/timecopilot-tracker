"""Fetch CFTC Commitments of Traders (Disaggregated Futures-Only) weekly report.

Extracts managed-money net positioning for key contracts and saves a normalized
positioning signal (-1 to +1) to data/cot_positioning.json.

Signal interpretation:
  > +0.6  = speculators heavily long → overbought, watch for reversal
  < -0.6  = speculators heavily short → oversold, watch for reversal
  Contrarian use: extreme longs are bearish, extreme shorts are bullish.
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

COT_URL     = "https://www.cftc.gov/dea/newcot/fut_disagg.txt"
OUTPUT_PATH = Path("data/cot_positioning.json")

# Maps CFTC contract name fragments → our ticker symbols
CONTRACT_MAP = {
    "GOLD - COMMODITY EXCHANGE":          "GC=F",
    "SILVER - COMMODITY EXCHANGE":        "SI=F",
    "CRUDE OIL, LIGHT SWEET":             "CL=F",
    "NATURAL GAS (NYMEX":                 "NG=F",
    "COPPER- #1":                         "HG=F",
    "WHEAT-SRW - CHICAGO BOARD OF TRADE": "ZW=F",
    "CORN - CHICAGO BOARD OF TRADE":      "ZC=F",
    "U.S. DOLLAR INDEX":                  "DX-Y.NYB",
    "E-MINI S&P 500":                     "SPY",
}

LONG_COL  = "M_Money_Positions_Long_All"
SHORT_COL = "M_Money_Positions_Short_All"
OI_COL    = "Open_Interest_All"


def fetch_cot() -> dict:
    print("[COT] Downloading CFTC disaggregated report...")
    try:
        r = requests.get(COT_URL, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"[COT] Download failed: {e}")
        return {}

    from io import StringIO
    df = pd.read_csv(StringIO(r.text), low_memory=False)

    results = {}
    for fragment, ticker in CONTRACT_MAP.items():
        match = df[df["Market_and_Exchange_Names"].str.contains(fragment, case=False, na=False)]
        if match.empty:
            continue
        row = match.iloc[0]
        try:
            longs  = float(row[LONG_COL])
            shorts = float(row[SHORT_COL])
            oi     = float(row[OI_COL])
            if oi <= 0:
                continue
            net_pct = (longs - shorts) / oi          # -1 to +1 raw
            results[ticker] = {
                "net_pct":       round(net_pct, 4),
                "longs":         int(longs),
                "shorts":        int(shorts),
                "open_interest": int(oi),
            }
        except Exception as e:
            print(f"[COT] {ticker} parse error: {e}")

    # Load history to compute z-score vs 52-week range
    history = {}
    if OUTPUT_PATH.exists():
        try:
            saved = json.loads(OUTPUT_PATH.read_text())
            history = saved.get("history", {})
        except Exception:
            pass

    positioning = {}
    for ticker, vals in results.items():
        net = vals["net_pct"]
        hist = history.get(ticker, [])
        hist.append(net)
        hist = hist[-52:]                           # keep 52 weeks
        history[ticker] = hist

        if len(hist) >= 4:
            arr  = np.array(hist)
            mean = float(np.mean(arr))
            std  = float(np.std(arr)) or 0.01
            z    = (net - mean) / std
            # Clip to [-2, 2] and normalize to [-1, 1] for signal
            signal = float(np.clip(z / 2.0, -1.0, 1.0))
        else:
            signal = float(np.clip(net * 2, -1.0, 1.0))  # raw normalized before history builds

        positioning[ticker] = {
            **vals,
            "z_score":        round(z if len(hist) >= 4 else 0.0, 3),
            "signal":         round(signal, 4),
            "signal_label":   "EXTREME_LONG" if signal > 0.6 else
                              "EXTREME_SHORT" if signal < -0.6 else
                              "LONG" if signal > 0.25 else
                              "SHORT" if signal < -0.25 else "NEUTRAL",
        }
        print(f"[COT] {ticker:12s}  net={net:+.3f}  z={positioning[ticker]['z_score']:+.2f}  → {positioning[ticker]['signal_label']}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "positioning":  positioning,
        "history":      history,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"[COT] Saved {len(positioning)} contracts → {OUTPUT_PATH}")
    return positioning


if __name__ == "__main__":
    fetch_cot()
