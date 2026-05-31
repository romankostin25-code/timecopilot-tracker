"""Compute daily technical + macro features for all tickers.

Outputs data/features.parquet with one row per (ticker, date) containing:
  Price features:     log_ret_1d, 5d, 20d, 60d
  Volatility:         vol_20d, vol_60d
  Technical:          rsi_14, macd_signal, bb_pos (Bollinger position)
  Volume:             vol_ratio_20d
  Macro (shared):     vix, yield_10y, yield_2y, yield_spread, dxy
  Targets (fwd):      fwd_ret_5d, fwd_ret_30d, fwd_ret_90d
                      actual_bullish_5d, _30d, _90d

Usage:
    python scripts/compute_features.py
    python scripts/compute_features.py --days 365
"""

import io
import os
import time
import zipfile
import argparse
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = "data/features.parquet"
MACRO_TICKERS = {"^VIX": "vix", "^TNX": "yield_10y", "^IRX": "yield_3m",
                 "DX-Y.NYB": "dxy"}

# Fama-French daily factor ZIP files from Ken French's Data Library
_FF3_URL  = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
_MOM_URL  = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
# CBOE equity put/call ratio (public CSV)
_PCR_URL  = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PCR_EQUITY.csv"

# FRED series IDs for credit/yield curve/macro surprise
_FRED_SERIES = {
    "DGS2":         "yield_2y",
    "DGS5":         "yield_5y",
    "DGS30":        "yield_30y",
    "BAMLC0A0CM":   "ig_spread",
    "BAMLH0A0HYM2": "hy_spread",
    "NAPM":         "ism_mfg",      # ISM Manufacturing PMI (monthly → ffill)
    "IC4WSA":       "claims_4wk",   # 4-week avg initial claims (weekly → ffill)
}


def _parse_ff_csv(raw_text: str) -> pd.DataFrame:
    """Parse Ken French CSV format: skip header lines, date=YYYYMMDD, values in %."""
    lines = raw_text.splitlines()
    # Skip leading copyright/blank lines until we hit the data header
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("Copyright") and "," in stripped:
            start = i
            break
    data_text = "\n".join(lines[start:])
    try:
        df = pd.read_csv(io.StringIO(data_text), header=0, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        date_col = df.columns[0]
        df = df[pd.to_numeric(df[date_col], errors="coerce").notna()].copy()
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"})
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0  # % → decimal
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  [FF] parse error: {e}")
        return pd.DataFrame()


def pull_ff_factors(days_back: int = 1095) -> pd.DataFrame:
    """Download Fama-French 3-factor + momentum daily data.

    Returns DataFrame with columns: date, ff_mkt_rf, ff_smb, ff_hml, ff_mom
    """
    cutoff = date.today() - timedelta(days=days_back)
    frames = {}

    for name, url in [("FF3", _FF3_URL), ("MOM", _MOM_URL)]:
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                zdata = resp.read()
            with zipfile.ZipFile(io.BytesIO(zdata)) as z:
                csv_name = next(n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
                raw = z.read(csv_name).decode("latin-1")
            df = _parse_ff_csv(raw)
            if not df.empty:
                df = df[df["date"].dt.date >= cutoff].copy()
            frames[name] = df
            print(f"  [FF] {name}: {len(df)} rows")
        except Exception as e:
            print(f"  [FF] {name} download failed: {e}")
            frames[name] = pd.DataFrame()

    ff3 = frames.get("FF3", pd.DataFrame())
    mom = frames.get("MOM", pd.DataFrame())

    if ff3.empty:
        return pd.DataFrame()

    # Rename FF3 columns
    rename_ff3 = {}
    for col in ff3.columns:
        c = col.upper().replace("-", "_").replace(" ", "_")
        if "MKT" in c or "MKT_RF" in c:
            rename_ff3[col] = "ff_mkt_rf"
        elif col.upper() == "SMB":
            rename_ff3[col] = "ff_smb"
        elif col.upper() == "HML":
            rename_ff3[col] = "ff_hml"
        elif col.upper() == "RF":
            rename_ff3[col] = "ff_rf"
    ff3 = ff3.rename(columns=rename_ff3)

    if not mom.empty:
        rename_mom = {col: "ff_mom" for col in mom.columns if col.upper() in ("UMD", "MOM", "WML")}
        mom = mom.rename(columns=rename_mom)[["date", "ff_mom"]] if "ff_mom" in mom.rename(columns=rename_mom).columns else pd.DataFrame()

    result = ff3[["date"] + [c for c in ["ff_mkt_rf", "ff_smb", "ff_hml"] if c in ff3.columns]]
    if not mom.empty and "ff_mom" in mom.columns:
        result = result.merge(mom[["date", "ff_mom"]], on="date", how="left")
    else:
        result["ff_mom"] = 0.0

    result["date"] = result["date"].dt.date
    return result


def pull_pcr_history(days_back: int = 1095) -> pd.DataFrame:
    """Fetch CBOE equity put/call ratio daily history."""
    cutoff = date.today() - timedelta(days=days_back)
    try:
        import requests
        r = requests.get(_PCR_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        date_col = next((c for c in df.columns if "DATE" in c), None)
        pcr_col  = next((c for c in df.columns if any(k in c for k in ("PUT", "RATIO", "PCR"))), None)
        if not date_col or not pcr_col:
            return pd.DataFrame()
        df = df[[date_col, pcr_col]].rename(columns={date_col: "date", pcr_col: "pcr"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["pcr"]  = pd.to_numeric(df["pcr"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        df = df[df["date"] >= cutoff]
        print(f"  [PCR] {len(df)} rows")
        return df
    except Exception as e:
        print(f"  [PCR] fetch failed: {e}")
        return pd.DataFrame()


def pull_fred_series(days_back: int = 1095) -> pd.DataFrame:
    """Fetch credit spreads, yield curve, and macro surprise series from FRED."""
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        print("  [FRED] FRED_API_KEY not set — skipping extended macro series")
        return pd.DataFrame()

    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
    except ImportError:
        print("  [FRED] fredapi not installed")
        return pd.DataFrame()

    cutoff = date.today() - timedelta(days=days_back + 60)
    frames = {}
    for series_id, col_name in _FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=cutoff.isoformat())
            if s is None or s.empty:
                continue
            s.name = col_name
            frames[col_name] = s
            print(f"  [FRED] {series_id} → {col_name}: {len(s)} obs")
        except Exception as e:
            print(f"  [FRED] {series_id}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index().ffill()

    # Derived features
    if "yield_10y" not in combined.columns:
        # yield_10y comes from yfinance macro pull; use DGS10 from FRED if missing
        try:
            s = fred.get_series("DGS10", observation_start=cutoff.isoformat())
            combined["yield_10y_fred"] = s
        except Exception:
            pass

    if "yield_2y" in combined.columns and "yield_10y_fred" in combined.columns:
        combined["yield_2s10s"] = combined["yield_10y_fred"] - combined["yield_2y"]
    elif "yield_2y" in combined.columns:
        combined["yield_2s10s_partial"] = -combined["yield_2y"]  # will be merged later

    if "yield_5y" in combined.columns and "yield_30y" in combined.columns:
        combined["yield_5s30s"] = combined["yield_30y"] - combined["yield_5y"]

    # ISM macro surprise: z-score vs rolling 12-month window
    if "ism_mfg" in combined.columns:
        ism = combined["ism_mfg"].ffill()
        ism_mean = ism.rolling(window=252, min_periods=30).mean()
        ism_std  = ism.rolling(window=252, min_periods=30).std()
        combined["macro_surprise_ism"] = ((ism - ism_mean) / ism_std.replace(0, 1.0)).clip(-3, 3)

    combined = combined.reset_index().rename(columns={"index": "date"})
    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    cutoff_use = date.today() - timedelta(days=days_back)
    combined = combined[combined["date"] >= cutoff_use]
    return combined


def _compute_features_for_ticker(ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
    from engine.feature_builder import compute_price_features
    df = compute_price_features(price_df, ticker)

    # Forward returns (targets)
    close = df["close"].astype(float)
    df["fwd_ret_5d"]  = np.log(close.shift(-5)  / close)
    df["fwd_ret_30d"] = np.log(close.shift(-30) / close)
    df["fwd_ret_90d"] = np.log(close.shift(-90) / close)
    df["actual_bullish_5d"]  = (df["fwd_ret_5d"]  > 0).astype(float)
    df["actual_bullish_30d"] = (df["fwd_ret_30d"] > 0).astype(float)
    df["actual_bullish_90d"] = (df["fwd_ret_90d"] > 0).astype(float)

    # Mark unknowable future targets as NaN
    today = date.today()
    df.loc[pd.to_datetime(df["date"]).dt.date >= today - timedelta(days=5),
           ["fwd_ret_5d",  "actual_bullish_5d"]]  = np.nan
    df.loc[pd.to_datetime(df["date"]).dt.date >= today - timedelta(days=30),
           ["fwd_ret_30d", "actual_bullish_30d"]] = np.nan
    df.loc[pd.to_datetime(df["date"]).dt.date >= today - timedelta(days=90),
           ["fwd_ret_90d", "actual_bullish_90d"]] = np.nan
    return df


def pull_macro_series(days_back: int = 1095) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=days_back + 30)
    macro_frames = []
    for ytick, col in MACRO_TICKERS.items():
        try:
            raw = yf.download(ytick, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                continue
            s = raw["Close"].squeeze().reset_index()
            s.columns = ["date", col]
            s["date"] = pd.to_datetime(s["date"]).dt.date
            macro_frames.append(s.set_index("date"))
        except Exception as e:
            print(f"  [macro] {ytick}: {e}")

    if not macro_frames:
        return pd.DataFrame()
    macro = macro_frames[0].join(macro_frames[1:], how="outer").ffill().reset_index()
    if "yield_10y" in macro.columns and "yield_3m" in macro.columns:
        macro["yield_spread"] = macro["yield_10y"] - macro["yield_3m"]
    return macro


def compute_all_features(days_back: int = 1095, tickers: list[str] = None):
    from engine.universe import ALL_TICKERS
    if tickers is None:
        tickers = ALL_TICKERS

    end = datetime.today()
    start = end - timedelta(days=days_back + 120)  # extra lookback for rolling windows

    print("Fetching yfinance macro series (VIX, yields, DXY)...")
    macro = pull_macro_series(days_back + 120)

    print("Fetching FRED credit spreads + macro surprise series...")
    fred_df = pull_fred_series(days_back + 120)

    print("Fetching Fama-French factors...")
    ff_df = pull_ff_factors(days_back + 120)

    print("Fetching CBOE put/call ratio history...")
    pcr_df = pull_pcr_history(days_back + 120)

    # Build a unified daily macro frame (all market-wide, same for every ticker)
    macro_unified = macro.copy() if not macro.empty else pd.DataFrame()

    if not fred_df.empty:
        fred_df["date"] = pd.to_datetime(fred_df["date"]).dt.date
        if macro_unified.empty:
            macro_unified = fred_df
        else:
            macro_unified["date"] = pd.to_datetime(macro_unified["date"]).dt.date
            macro_unified = macro_unified.merge(fred_df, on="date", how="outer")

    if not ff_df.empty:
        ff_df["date"] = pd.to_datetime(ff_df["date"]).dt.date
        if macro_unified.empty:
            macro_unified = ff_df
        else:
            macro_unified["date"] = pd.to_datetime(macro_unified["date"]).dt.date
            macro_unified = macro_unified.merge(ff_df, on="date", how="outer")

    if not pcr_df.empty:
        pcr_df["date"] = pd.to_datetime(pcr_df["date"]).dt.date
        if macro_unified.empty:
            macro_unified = pcr_df
        else:
            macro_unified["date"] = pd.to_datetime(macro_unified["date"]).dt.date
            macro_unified = macro_unified.merge(pcr_df, on="date", how="outer")

    if not macro_unified.empty:
        macro_unified = macro_unified.sort_values("date").ffill().reset_index(drop=True)

        # Derive yield curve spreads from FRED yields (more complete than yfinance)
        if "yield_2y" in macro_unified.columns and "yield_10y" in macro_unified.columns:
            macro_unified["yield_2s10s"] = macro_unified["yield_10y"] - macro_unified["yield_2y"]
        elif "yield_2s10s" not in macro_unified.columns:
            macro_unified["yield_2s10s"] = 0.5  # reasonable default
        if "yield_5y" in macro_unified.columns and "yield_30y" in macro_unified.columns:
            macro_unified["yield_5s30s"] = macro_unified["yield_30y"] - macro_unified["yield_5y"]
        elif "yield_5s30s" not in macro_unified.columns:
            macro_unified["yield_5s30s"] = 0.3

    all_frames = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                print("no data")
                continue
            price_df = raw[["Close", "Volume"]].reset_index()
            price_df.columns = ["date", "close", "volume"]
            price_df["date"] = pd.to_datetime(price_df["date"]).dt.date

            feat = _compute_features_for_ticker(ticker, price_df)

            # Merge unified macro (VIX, yields, DXY, FF factors, credit spreads, PCR)
            if not macro_unified.empty:
                macro_unified["date"] = pd.to_datetime(macro_unified["date"]).dt.date
                feat = feat.merge(macro_unified, on="date", how="left")
                macro_cols = [c for c in macro_unified.columns if c != "date"]
                feat[macro_cols] = feat[macro_cols].ffill()

            # Keep only rows within requested window
            cutoff = (end - timedelta(days=days_back)).date()
            feat = feat[pd.to_datetime(feat["date"]).dt.date >= cutoff]
            all_frames.append(feat)
            print(f"{len(feat)} rows")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_frames:
        print("No data collected.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows × {len(combined.columns)} cols to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=1095)
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    tickers = [args.ticker] if args.ticker else None
    compute_all_features(days_back=args.days, tickers=tickers)
