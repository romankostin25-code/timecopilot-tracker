"""Feature engineering pipeline for directional classifier training and inference."""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from typing import Optional

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_20d", "atr_14_pct",
    "rsi_14", "macd_hist", "bb_pos",
    "close_vs_ma20", "close_vs_ma50",
    "vix", "us10y", "dxy", "vix_5d_chg_pct", "dxy_5d_chg_pct",
    "poly_prob_bullish",
    "regime_id",
]


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


_macro_history_cache: pd.DataFrame | None = None


def _fetch_macro_history(years: int = 3) -> pd.DataFrame:
    """Fetch historical VIX, 10Y, DXY from yfinance for feature computation during training."""
    global _macro_history_cache
    if _macro_history_cache is not None and not _macro_history_cache.empty:
        return _macro_history_cache

    end   = datetime.today()
    start = end - timedelta(days=365 * years + 30)
    parts = {}
    for yticker, col in [("^VIX", "vix"), ("^TNX", "us10y"), ("DX-Y.NYB", "dxy")]:
        try:
            raw = yf.download(yticker, start=start.strftime("%Y-%m-%d"),
                              end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            parts[col] = raw["Close"].dropna().rename(col)
        except Exception:
            pass

    if not parts:
        return pd.DataFrame()

    macro = pd.DataFrame(parts)
    macro.index = pd.to_datetime(macro.index).tz_localize(None).normalize()
    macro = macro.resample("B").last().ffill()
    macro.index.name = "ds"
    macro = macro.reset_index()
    macro["ds"] = pd.to_datetime(macro["ds"])
    if "vix" in macro.columns:
        macro["vix_5d_chg_pct"] = macro["vix"].pct_change(5) * 100
    if "dxy" in macro.columns:
        macro["dxy_5d_chg_pct"] = macro["dxy"].pct_change(5) * 100

    _macro_history_cache = macro
    return macro


def fetch_ohlcv(ticker: str, years: int = 3) -> pd.DataFrame:
    """Fetch OHLCV data as a clean DataFrame with columns: ds, open, high, low, close, volume."""
    end   = datetime.today()
    start = end - timedelta(days=365 * years + 30)
    raw   = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if raw.empty:
        raw = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No OHLCV data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [c.lower() if c != "Date" else "ds" for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "ds"})

    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("ds").reset_index(drop=True)

    # Ensure standard column names
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0.0
            elif col in ["open", "high", "low"]:
                df[col] = df.get("close", df.get("adj close", np.nan))

    df = df[["ds", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df = df.resample("B", on="ds").last().ffill().reset_index()
    return df


def compute_features(
    price_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    poly_df:  Optional[pd.DataFrame] = None,
    ticker:   str = "",
) -> pd.DataFrame:
    """
    Compute feature matrix from OHLCV data + macro + poly context.

    price_df must have: ds, close (or y), and optionally high/low/volume/open.
    Returns DataFrame with FEATURE_COLS + target column (1=up next day, 0=down).
    """
    df = price_df.copy().sort_values("ds").reset_index(drop=True)

    # Normalise close column name
    if "close" not in df.columns and "y" in df.columns:
        df["close"] = df["y"]
    close = df["close"]

    high   = df["high"]   if "high"   in df.columns else close
    low    = df["low"]    if "low"    in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

    # ── Price returns ──────────────────────────────────────────────────────────
    for w in [1, 3, 5, 10, 20]:
        df[f"ret_{w}d"] = close.pct_change(w)

    # ── Volatility ─────────────────────────────────────────────────────────────
    df["vol_5d"]  = close.pct_change().rolling(5).std()
    df["vol_20d"] = close.pct_change().rolling(20).std()

    # ── ATR (normalised by price) ───────────────────────────────────────────────
    atr = _compute_atr(high, low, close)
    df["atr_14_pct"] = atr / close.replace(0, np.nan)

    # ── RSI ────────────────────────────────────────────────────────────────────
    df["rsi_14"] = _compute_rsi(close) / 100.0

    # ── MACD histogram (normalised) ─────────────────────────────────────────────
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig9  = macd.ewm(span=9).mean()
    df["macd_hist"] = (macd - sig9) / close.replace(0, np.nan)

    # ── Bollinger band position ─────────────────────────────────────────────────
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    df["bb_pos"] = (close - sma20) / (2 * std20.replace(0, np.nan))

    # ── Price vs moving averages ────────────────────────────────────────────────
    df["close_vs_ma20"] = (close - close.rolling(20).mean()) / close.replace(0, np.nan)
    df["close_vs_ma50"] = (close - close.rolling(50).mean()) / close.replace(0, np.nan)

    # ── Macro context ───────────────────────────────────────────────────────────
    macro_needed = ["vix", "us10y", "dxy", "vix_5d_chg_pct", "dxy_5d_chg_pct"]

    # Determine effective macro source: prefer passed macro_df, but supplement with
    # yfinance historical data when training window exceeds available macro_context.csv dates
    effective_macro = macro_df
    if macro_df is None or macro_df.empty:
        effective_macro = None
    else:
        # Check coverage: if price_df spans dates not in macro_df, fetch historical
        price_min = pd.to_datetime(df["ds"]).min()
        macro_min = pd.to_datetime(macro_df["date"]).min()
        if price_min < macro_min - pd.Timedelta(days=30):
            effective_macro = None  # not enough coverage, use yfinance history

    if effective_macro is None:
        hist = _fetch_macro_history(years=3)
        if not hist.empty:
            df["ds_dt"] = pd.to_datetime(df["ds"])
            hist["ds"]  = pd.to_datetime(hist["ds"])
            hist_cols   = [c for c in macro_needed if c in hist.columns]
            df = df.merge(hist[["ds"] + hist_cols], left_on="ds_dt", right_on="ds",
                          how="left", suffixes=("", "_macro"))
            df = df.drop(columns=["ds_macro", "ds_dt"], errors="ignore")
            df[macro_needed] = df[[c for c in macro_needed if c in df.columns]].ffill()
            for col in macro_needed:
                if col not in df.columns:
                    df[col] = np.nan
        else:
            for col in macro_needed:
                df[col] = np.nan
    else:
        mcols = ["date"] + [c for c in macro_needed if c in effective_macro.columns]
        msub  = effective_macro[mcols].copy()
        msub["ds_date"] = pd.to_datetime(msub["date"]).dt.date
        df["ds_date"]   = pd.to_datetime(df["ds"]).dt.date
        df = df.merge(msub.drop(columns="date"), on="ds_date", how="left")
        df[macro_needed] = df[macro_needed].ffill()
        for col in macro_needed:
            if col not in df.columns:
                df[col] = np.nan
        df = df.drop(columns=["ds_date"], errors="ignore")

    # ── Polymarket signal ───────────────────────────────────────────────────────
    if poly_df is not None and not poly_df.empty and "poly_prob_bullish" in poly_df.columns:
        pticker = poly_df[poly_df["ticker"] == ticker] if "ticker" in poly_df.columns else poly_df
        if not pticker.empty:
            psub = pticker[["date", "poly_prob_bullish"]].copy()
            psub["ds_date"] = pd.to_datetime(psub["date"]).dt.date
            df["ds_date"]   = pd.to_datetime(df["ds"]).dt.date
            df = df.merge(psub[["ds_date", "poly_prob_bullish"]], on="ds_date", how="left")
            df["poly_prob_bullish"] = df["poly_prob_bullish"].ffill().fillna(0.5)
            df = df.drop(columns=["ds_date"], errors="ignore")
        else:
            df["poly_prob_bullish"] = 0.5
    else:
        df["poly_prob_bullish"] = 0.5

    # ── Regime ──────────────────────────────────────────────────────────────────
    from regime_classifier import classify_regime
    df["regime_id"] = df.apply(
        lambda r: classify_regime(
            float(r.get("vix", 20) or 20),
            float(r.get("us10y", 4.0) or 4.0),
            float(r.get("dxy_5d_chg_pct", 0) or 0),
        ),
        axis=1,
    )

    # ── Target: 1 if next close > current ──────────────────────────────────────
    df["target"] = (close.shift(-1) > close).astype(float)
    df.loc[df.index[-1], "target"] = np.nan  # last row has no future

    return df.reset_index(drop=True)


def get_current_features(ticker: str, macro_df: Optional[pd.DataFrame] = None,
                          poly_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Fetch latest OHLCV and return current feature values for live inference.
    Returns a dict mapping feature names → values.
    """
    df      = fetch_ohlcv(ticker, years=1)
    feat_df = compute_features(df, macro_df, poly_df, ticker=ticker)
    if feat_df.empty:
        return {}
    latest = feat_df.iloc[-1]
    return {col: (None if pd.isna(latest.get(col)) else float(latest[col]))
            for col in FEATURE_COLS if col in feat_df.columns}
