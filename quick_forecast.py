"""
Quick forecast generator — runs today's forecasts without statsforecast.
Uses EWM trend + ATR bands for the TC arm, CLF models for direction.
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

TICKERS  = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]
HORIZON  = int(os.getenv("FORECAST_HORIZON", 10))
FREQ     = "B"
CSV_PATH = "forecasts.csv"

CSV_COLUMNS = [
    "forecast_date", "target_date", "ticker",
    "p10", "p50", "p90",
    "actual", "model_used",
    "direction", "signal_strength", "conviction_score",
    "error_abs", "error_pct", "hit", "direction_correct",
    "graded_at", "notes",
    "arm_tc_prob", "arm_clf_prob", "arm_poly_prob",
]


def fetch_price_data(ticker: str, years: int = 2) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=365 * years)
    raw   = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if raw.empty:
        raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [c.lower() if c.lower() != "date" else "ds" for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "ds"})

    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    df["unique_id"] = ticker
    df = df.rename(columns={"close": "y"}) if "close" in df.columns else df
    if "y" not in df.columns:
        raise ValueError(f"No close column for {ticker}")

    df = df.set_index("ds")
    df = df.resample("B").last().ffill()
    df = df.reset_index()
    df["unique_id"] = ticker
    return df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)


def _compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def generate_ewm_forecast(price_df: pd.DataFrame, horizon: int, clf_prob: float = None):
    """
    Generate P10/P50/P90 using EWM trend + ATR-based bands.
    When CLF has strong conviction, steer P50 toward CLF direction.
    """
    close  = price_df["y"]
    last   = float(close.iloc[-1])

    # EWM trend: smoothed daily return
    daily_ret  = close.pct_change().dropna()
    ewm_return = float(daily_ret.ewm(span=10).mean().iloc[-1])

    # Blend EWM trend with CLF direction for P50 path:
    # Strong CLF bullish/bearish overrides EWM downtrend/uptrend for a coherent forecast
    if clf_prob is not None:
        clf_dir = clf_prob - 0.5  # range -0.5 to +0.5, 0 = neutral
        long_run_ret = float(daily_ret.ewm(span=60).mean().iloc[-1])  # 3-month EWM
        # Weight: CLF conviction steers forecast toward expected direction
        # At clf_prob=0.8: clf_dir=0.3, which should nudge P50 upward even if EWM is negative
        clf_weight = min(abs(clf_dir) * 2, 0.8)  # max 80% weight to CLF direction
        ewm_weight = 1.0 - clf_weight
        # Use long-run base return + CLF direction nudge
        base_daily = ewm_weight * ewm_return + clf_weight * (abs(long_run_ret) * np.sign(clf_dir))
        ewm_return = max(-0.005, min(0.005, base_daily))  # cap daily drift at ±0.5%

    # ATR for band width
    if "high" in price_df.columns and "low" in price_df.columns:
        atr = _compute_atr(price_df["high"], price_df["low"], close)
        last_atr = float(atr.dropna().iloc[-1])
        atr_pct  = last_atr / max(abs(last), 1e-9)
    else:
        atr_pct = abs(daily_ret).ewm(span=14).mean().iloc[-1] * 1.5

    p50_vals, p10_vals, p90_vals = [], [], []
    for i in range(1, horizon + 1):
        p50 = last * (1 + ewm_return * i)
        # Bands widen with sqrt(time) — typical for price forecasts
        band = atr_pct * np.sqrt(i) * 1.5
        p10_vals.append(round(p50 * (1 - band), 6))
        p50_vals.append(round(p50, 6))
        p90_vals.append(round(p50 * (1 + band), 6))

    return p10_vals, p50_vals, p90_vals, atr_pct


def forecast_ticker(ticker: str, existing: pd.DataFrame,
                    macro_df: pd.DataFrame, poly_df: pd.DataFrame) -> list[dict]:
    from signal_combiner import combine_signals

    today_str = str(date.today())
    if not existing.empty:
        dupe = existing[
            (existing["forecast_date"].astype(str) == today_str) &
            (existing["ticker"] == ticker)
        ]
        if not dupe.empty:
            print(f"  [{ticker}] already forecasted today — skipping.")
            return []

    try:
        price_df = fetch_price_data(ticker)
    except Exception as e:
        print(f"  [{ticker}] price fetch failed: {e}")
        return []

    last_price = float(price_df["y"].iloc[-1])
    print(f"  [{ticker}] {len(price_df)} pts, last={last_price:.4f}")

    # Polymarket arm
    poly_prob = None
    if not poly_df.empty and "poly_prob_bullish" in poly_df.columns:
        tp = poly_df[poly_df["ticker"] == ticker]
        if not tp.empty:
            raw = tp.sort_values("date").iloc[-1]["poly_prob_bullish"]
            if not pd.isna(raw):
                try:
                    from poly_calibration import calibrate_prob
                    poly_prob = calibrate_prob(float(raw))
                except Exception:
                    poly_prob = float(raw)

    # CLF arm — compute FIRST so we can steer P50 direction
    clf_prob = None
    try:
        from directional_classifier import predict_prob_up
        from feature_pipeline import compute_features, FEATURE_COLS
        cdf = price_df.rename(columns={"y": "close"})
        feat_df = compute_features(cdf, macro_df, poly_df, ticker=ticker)
        if not feat_df.empty:
            latest  = feat_df.iloc[-1]
            features = {col: (None if pd.isna(latest.get(col)) else float(latest[col]))
                        for col in FEATURE_COLS if col in feat_df.columns}
            if features:
                clf_prob = predict_prob_up(ticker, features)
    except Exception as e:
        print(f"  [{ticker}] CLF warning: {e}")

    # EWM forecast — steered by CLF direction for coherent chart
    p10_vals, p50_vals, p90_vals, atr_14_pct = generate_ewm_forecast(price_df, HORIZON, clf_prob)

    from pandas.tseries.offsets import BDay
    last_date    = pd.Timestamp(price_df["ds"].max())
    target_dates = [last_date + BDay(i + 1) for i in range(HORIZON)]

    rows = []
    for i, tgt in enumerate(target_dates):
        p10 = p10_vals[i]
        p50 = p50_vals[i]
        p90 = p90_vals[i]

        sig = combine_signals(
            ticker=ticker,
            p10=p10, p50=p50, p90=p90,
            last_price=last_price, atr_14_pct=atr_14_pct,
            clf_prob_up=clf_prob,
            poly_prob_bullish=poly_prob,
        )

        rows.append({
            "forecast_date":    today_str,
            "target_date":      tgt.date().isoformat(),
            "ticker":           ticker,
            "p10":              p10,
            "p50":              p50,
            "p90":              p90,
            "actual":           "",
            "model_used":       "EWM_CLF_Ensemble",
            "direction":        sig["direction"],
            "signal_strength":  sig["signal_strength"],
            "conviction_score": sig["conviction_score"],
            "error_abs":        "",
            "error_pct":        "",
            "hit":              "",
            "direction_correct": "",
            "graded_at":        "",
            "notes":            "",
            "arm_tc_prob":      sig["arm_tc_prob"],
            "arm_clf_prob":     sig["arm_clf_prob"],
            "arm_poly_prob":    sig["arm_poly_prob"],
        })

    direction  = rows[0]["direction"]  if rows else "?"
    conviction = rows[0]["conviction_score"] if rows else 0
    print(f"  [{ticker}] {direction} conv={conviction:.3f}  "
          f"clf={'n/a' if clf_prob is None else f'{clf_prob:.2f}'}  "
          f"poly={'n/a' if poly_prob is None else f'{poly_prob:.2f}'}")
    return rows


def run_all():
    print("=== Quick Forecast Run ===")
    existing = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    macro_df = pd.read_csv("macro_context.csv")   if os.path.exists("macro_context.csv")   else pd.DataFrame()
    poly_df  = pd.read_csv("polymarket_data.csv") if os.path.exists("polymarket_data.csv") else pd.DataFrame()

    all_rows = []
    for ticker in TICKERS:
        try:
            rows = forecast_ticker(ticker, existing, macro_df, poly_df)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [{ticker}] FAILED: {e}")

    if all_rows:
        new_df   = pd.DataFrame(all_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        for col in CSV_COLUMNS:
            if col not in combined.columns:
                combined[col] = ""
        combined = combined[CSV_COLUMNS + [c for c in combined.columns if c not in CSV_COLUMNS]]
        combined.to_csv(CSV_PATH, index=False)
        print(f"\n✓ {len(all_rows)} new rows written to {CSV_PATH}")
    else:
        print("\nNo new rows added.")

    return len(all_rows)


if __name__ == "__main__":
    run_all()
