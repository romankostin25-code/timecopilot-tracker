"""Macro context and regime classification."""

import os
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

CSV_PATH = "data/macro_context.csv"


def fetch_macro_context() -> dict:
    today = str(date.today())
    macro: dict = {"date": today}

    yf_targets = {
        "^VIX": "vix", "^TNX": "us10y", "DX-Y.NYB": "dxy",
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

    vix     = macro.get("vix", 20)
    dxy_chg = macro.get("dxy_5d_chg_pct", 0) or 0
    us10y   = macro.get("us10y", 4.0)

    macro["risk_regime"]   = "RISK_OFF" if vix > 25 else "RISK_ON" if vix < 15 else "NEUTRAL"
    macro["dollar_regime"] = "DOLLAR_STRENGTH" if dxy_chg > 0.5 else "DOLLAR_WEAKNESS" if dxy_chg < -0.5 else "DOLLAR_NEUTRAL"
    macro["rate_regime"]   = "HAWKISH" if us10y > 4.5 else "DOVISH" if us10y < 3.5 else "NEUTRAL_RATES"
    macro["generated_at"]  = datetime.now().isoformat()

    os.makedirs("data", exist_ok=True)
    new_df = pd.DataFrame([macro])
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        existing = existing[existing["date"].astype(str) != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(CSV_PATH, index=False)

    print(f"✓ macro_context.csv | VIX={vix} | {macro['risk_regime']} | {macro['dollar_regime']} | {macro['rate_regime']}")
    return macro


if __name__ == "__main__":
    fetch_macro_context()
