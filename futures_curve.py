"""Trading Co-Pilot — Futures term structure engine."""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "futures_curve.csv"

_MONTH_CODES = "FGHJKMNQUVXZ"


def _contract_codes(base_code: str, exchange_suffix: str, n: int = 5) -> list:
    """Generate next n front-month contract codes dynamically from today's date."""
    today = date.today()
    codes = []
    month, year = today.month, today.year
    for _ in range(n):
        month += 1
        if month > 12:
            month = 1
            year += 1
        yr2 = str(year)[-2:]
        codes.append(f"{base_code}{_MONTH_CODES[month - 1]}{yr2}.{exchange_suffix}")
    return codes


def _build_curve_tickers() -> dict:
    return {
        "CL=F": ["CL=F"] + _contract_codes("CL", "NYM", 5),
        "GC=F": ["GC=F"] + _contract_codes("GC", "CMX", 4),
        "NG=F": ["NG=F"] + _contract_codes("NG", "NYM", 4),
        "SI=F": ["SI=F"] + _contract_codes("SI", "CMX", 3),
        "HG=F": ["HG=F"] + _contract_codes("HG", "CMX", 3),
    }


def _price(ticker: str) -> float | None:
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
    curve_tickers = _build_curve_tickers()

    for base, contracts in curve_tickers.items():
        print(f"[{base}] Fetching futures curve… contracts: {contracts[1:]}")
        prices = [_price(c) for c in contracts]
        spot = prices[0]
        if spot is None:
            print(f"[{base}] No spot price — skipping.")
            continue

        m1, m2, m3, m4, m5 = (prices[i] if i < len(prices) else None for i in range(1, 6))
        ref = m1 or spot

        contango_m1 = round((ref - spot) / spot * 100, 4) if m1 else None
        contango_m2 = round((m2 - spot) / spot * 100, 4) if m2 else None
        contango_m3 = round((m3 - spot) / spot * 100, 4) if m3 else None
        roll_yield = round(-contango_m1 * 12, 4) if contango_m1 is not None else None
        curve_slope = round((m3 - m1) / m1 * 100, 4) if (m3 and m1) else None

        if contango_m1 is None:
            regime = "UNKNOWN"
        elif contango_m1 < -0.3:
            regime = "BACKWARDATION"
        elif contango_m1 > 0.3:
            regime = "CONTANGO"
        else:
            regime = "FLAT"

        rows.append({
            "date": today, "ticker": base,
            "spot": spot, "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5,
            "regime": regime,
            "contango_m1_pct": contango_m1,
            "contango_m2_pct": contango_m2,
            "contango_m3_pct": contango_m3,
            "roll_yield_annualized_pct": roll_yield,
            "curve_slope": curve_slope,
        })
        print(f"[{base}] ✓ {regime} | roll yield: {roll_yield}%/yr")

    new_df = pd.DataFrame(rows)
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        existing = existing[existing["date"] != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(CSV_PATH, index=False)
    print(f"\n✓ futures_curve.csv updated ({len(rows)} rows).")


if __name__ == "__main__":
    fetch_all_curves()
