"""Futures term structure engine."""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "data/futures_curve.csv"
_MONTH_CODES = "FGHJKMNQUVXZ"


def _contract_codes(base_code, exchange_suffix, n=5):
    today = date.today()
    codes = []
    month, year = today.month, today.year
    for _ in range(n):
        month += 1
        if month > 12:
            month, year = 1, year + 1
        yr2 = str(year)[-2:]
        codes.append(f"{base_code}{_MONTH_CODES[month - 1]}{yr2}.{exchange_suffix}")
    return codes


def _build_curve_tickers():
    return {
        "CL=F": ["CL=F"] + _contract_codes("CL", "NYM", 5),
        "GC=F": ["GC=F"] + _contract_codes("GC", "CMX", 4),
        "NG=F": ["NG=F"] + _contract_codes("NG", "NYM", 4),
        "SI=F": ["SI=F"] + _contract_codes("SI", "CMX", 3),
        "HG=F": ["HG=F"] + _contract_codes("HG", "CMX", 3),
        "ZW=F": ["ZW=F"] + _contract_codes("ZW", "CBT", 3),
        "ZC=F": ["ZC=F"] + _contract_codes("ZC", "CBT", 3),
    }


def _price(ticker):
    try:
        raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return round(float(raw["Close"].dropna().iloc[-1]), 6)
    except Exception:
        return None


def fetch_all_curves():
    today = str(date.today())
    rows = []
    for base, contracts in _build_curve_tickers().items():
        print(f"[{base}] Fetching curve…")
        prices = [_price(c) for c in contracts]
        spot = prices[0]
        if spot is None:
            continue
        m1, m2, m3 = (prices[i] if i < len(prices) else None for i in range(1, 4))
        ref = m1 or spot
        c1 = round((ref - spot) / spot * 100, 4) if m1 else None
        c2 = round((m2 - spot) / spot * 100, 4) if m2 else None
        c3 = round((m3 - spot) / spot * 100, 4) if m3 else None
        roll = round(-c1 * 12, 4) if c1 is not None else None
        slope = round((m3 - m1) / m1 * 100, 4) if (m3 and m1) else None
        regime = ("UNKNOWN" if c1 is None else
                  "BACKWARDATION" if c1 < -0.3 else
                  "CONTANGO" if c1 > 0.3 else "FLAT")
        rows.append({
            "date": today, "ticker": base, "spot": spot,
            "m1": m1, "m2": m2, "m3": m3,
            "regime": regime,
            "contango_m1_pct": c1, "contango_m2_pct": c2, "contango_m3_pct": c3,
            "roll_yield_annualized_pct": roll, "curve_slope": slope,
        })
        print(f"[{base}] ✓ {regime}")

    os.makedirs("data", exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        existing = existing[existing["date"].astype(str) != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(CSV_PATH, index=False)
    print(f"✓ futures_curve.csv updated ({len(rows)} rows).")


if __name__ == "__main__":
    fetch_all_curves()
