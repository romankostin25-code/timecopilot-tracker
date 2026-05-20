"""Shared feature computation used by both training (compute_features.py) and inference.

Outputs a DataFrame with all technical + macro + sentiment columns expected by the TFT.
Keeping this in one place ensures training and inference features are always identical.
"""

import numpy as np
import pandas as pd

# Columns the TFT expects as time-varying inputs
TFT_FEATURE_COLS = [
    "log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d",
    "vol_20d", "vol_60d",
    "rsi_14", "macd_signal", "bb_pos", "vol_ratio_20d",
    "vix", "yield_10y", "yield_spread", "dxy",
    "sentiment_score", "sentiment_7d_ma", "sentiment_momentum",
]


def compute_price_features(price_df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Compute all technical features from a close-price DataFrame.

    price_df must have columns: date, close, [volume]
    Returns the same DataFrame with feature columns appended.
    """
    df = price_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=df.index)

    # Log returns
    log_ret = np.log(close / close.shift(1))
    df["log_ret_1d"]  = log_ret
    df["log_ret_5d"]  = np.log(close / close.shift(5))
    df["log_ret_20d"] = np.log(close / close.shift(20))
    df["log_ret_60d"] = np.log(close / close.shift(60))

    # Realised volatility (annualised)
    df["vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["vol_60d"] = log_ret.rolling(60).std() * np.sqrt(252)

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))

    # MACD signal line (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    # Bollinger Band position (±1 at ±2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_pos"] = (close - bb_mid) / (2 * bb_std.replace(0, np.nan))

    # Volume ratio vs 20d MA
    if not volume.isna().all():
        vol_ma = volume.rolling(20).mean()
        df["vol_ratio_20d"] = volume / vol_ma.replace(0, np.nan)
    else:
        df["vol_ratio_20d"] = np.nan

    if ticker:
        df["ticker"] = ticker

    return df


def attach_macro(feat_df: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """Fill macro columns from the macro context dict."""
    feat_df = feat_df.copy()
    feat_df["vix"]          = float(macro.get("vix", 20.0))
    feat_df["yield_10y"]    = float(macro.get("yield_10y", 4.0))
    feat_df["yield_spread"] = float(macro.get("yield_spread", 0.5))
    feat_df["dxy"]          = float(macro.get("dxy", 104.0))
    return feat_df


def attach_news(feat_df: pd.DataFrame, ticker: str,
                news_df: pd.DataFrame | None) -> pd.DataFrame:
    """Merge daily sentiment scores onto feature rows by date."""
    feat_df = feat_df.copy()
    for col in ["sentiment_score", "sentiment_7d_ma", "sentiment_momentum"]:
        feat_df[col] = 0.0

    if news_df is None or news_df.empty:
        return feat_df

    t_news = news_df[news_df["ticker"] == ticker].copy()
    if t_news.empty:
        return feat_df

    t_news["date"] = pd.to_datetime(t_news["date"]).dt.date
    feat_df["date_key"] = pd.to_datetime(feat_df["date"]).dt.date

    for col in ["sentiment_score", "sentiment_7d_ma", "sentiment_momentum"]:
        if col in t_news.columns:
            mapping = t_news.set_index("date")[col]
            feat_df[col] = feat_df["date_key"].map(mapping).fillna(0.0)

    feat_df = feat_df.drop(columns=["date_key"])
    return feat_df


def build_inference_features(ticker: str, price_arr: np.ndarray,
                              macro: dict | None = None,
                              news_df: pd.DataFrame | None = None,
                              encoder_len: int = 60) -> pd.DataFrame | None:
    """Build a feature DataFrame for live TFT inference from a raw price array.

    price_arr: numpy array of closing prices, oldest first, recent last.
    Returns a DataFrame with the last encoder_len+1 complete rows, or None
    if there isn't enough data.
    """
    if len(price_arr) < encoder_len + 20:
        return None

    # Build date index (business days ending today)
    import pandas as pd
    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=today, periods=len(price_arr))
    price_df = pd.DataFrame({"date": dates.date, "close": price_arr.astype(float)})

    feat = compute_price_features(price_df, ticker)
    feat = attach_macro(feat, macro or {})
    feat = attach_news(feat, ticker, news_df)

    # Fill remaining macro columns with defaults
    for col in TFT_FEATURE_COLS:
        if col not in feat.columns:
            feat[col] = 0.0
        feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0.0)

    # Drop the initial NaN-heavy rows from rolling windows
    feat = feat.dropna(subset=["log_ret_1d", "vol_20d", "rsi_14"])
    feat = feat.tail(encoder_len + 1).copy()

    if len(feat) < 30:
        return None

    feat = feat.reset_index(drop=True)
    feat["time_idx"] = feat.index
    feat["ticker"]   = ticker
    return feat
