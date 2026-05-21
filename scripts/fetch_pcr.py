"""Fetch CBOE equity put/call ratio (daily) as a contrarian signal for equities.

Signal interpretation (contrarian):
  PCR > 0.80  = excessive put buying = fear = contrarian BULLISH
  PCR < 0.45  = excessive call buying = complacency = contrarian BEARISH
  Normal range: ~0.55–0.70

The z-score normalizes PCR vs its rolling 52-week history so the signal is
comparable across different vol regimes.
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

OUTPUT_PATH = Path("data/pcr_signal.json")

# CBOE publishes equity put/call history as a public CSV
_URLS = [
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/PCR_EQUITY.csv",
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/CBOE_EQUITY_ONLY_PCR.csv",
]

# Contrarian thresholds — based on historical equity P/C distribution
PCR_HIGH = 0.80   # fear threshold (contrarian bullish)
PCR_LOW  = 0.45   # complacency threshold (contrarian bearish)
PCR_NORM = 0.625  # historical mid-point


def _fetch_raw() -> pd.DataFrame | None:
    for url in _URLS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), skiprows=0)
            # normalize column names
            df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
            date_col = next((c for c in df.columns if "DATE" in c), None)
            pcr_col  = next((c for c in df.columns if "PUT" in c or "RATIO" in c or "P/C" in c.upper()
                             or "PCR" in c), None)
            if date_col and pcr_col:
                df = df[[date_col, pcr_col]].rename(columns={date_col: "date", pcr_col: "pcr"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["pcr"]  = pd.to_numeric(df["pcr"], errors="coerce")
                df = df.dropna().sort_values("date").reset_index(drop=True)
                if not df.empty:
                    print(f"[PCR] Fetched {len(df)} rows from {url.split('/')[-1]}")
                    return df
        except Exception as e:
            print(f"[PCR] {url.split('/')[-1]}: {e}")
    return None


def fetch_pcr() -> dict:
    print("[PCR] Fetching CBOE equity put/call ratio...")

    # Load history for z-score continuity
    history: list[float] = []
    if OUTPUT_PATH.exists():
        try:
            saved = json.loads(OUTPUT_PATH.read_text())
            history = saved.get("history", [])
        except Exception:
            pass

    raw = _fetch_raw()
    if raw is None:
        print("[PCR] All sources failed — using cached signal if available")
        if OUTPUT_PATH.exists():
            return json.loads(OUTPUT_PATH.read_text()).get("signal", {})
        return {}

    latest_pcr = float(raw["pcr"].iloc[-1])
    latest_date = str(raw["date"].iloc[-1].date())

    # Build/extend history from fetched data (up to 52 weekly samples)
    # Use last ~52 trading-week rows (≈260 trading days) sampled weekly for robustness
    hist_src = raw["pcr"].dropna().values
    if len(hist_src) >= 5:
        # Downsample to ~weekly to match COT history granularity
        step = max(1, len(hist_src) // 52)
        history = [float(v) for v in hist_src[::step]][-52:]
    else:
        history.append(latest_pcr)
        history = history[-52:]

    # Z-score normalization
    if len(history) >= 4:
        arr  = np.array(history)
        mean = float(np.mean(arr))
        std  = float(np.std(arr)) or 0.01
        z    = (latest_pcr - mean) / std
        # Contrarian: high PCR (panic) → bullish → invert z-score
        signal = float(np.clip(-z / 2.0, -1.0, 1.0))
    else:
        # Raw contrarian mapping before history builds
        mid    = PCR_NORM
        z      = 0.0
        signal = float(np.clip(-(latest_pcr - mid) / 0.20, -1.0, 1.0))

    if signal > 0.6:
        label = "CONTRARIAN_BULL"
    elif signal < -0.6:
        label = "CONTRARIAN_BEAR"
    elif signal > 0.25:
        label = "MILDLY_BULL"
    elif signal < -0.25:
        label = "MILDLY_BEAR"
    else:
        label = "NEUTRAL"

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_date":  latest_date,
        "pcr":          round(latest_pcr, 4),
        "z_score":      round(z, 3),
        "signal":       round(signal, 4),
        "signal_label": label,
        "history":      history,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"[PCR] {latest_date}  pcr={latest_pcr:.3f}  z={z:+.2f}  → {label}  (signal={signal:+.3f})")
    return output


if __name__ == "__main__":
    fetch_pcr()
