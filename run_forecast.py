"""Trading Co-Pilot — statsforecast ensemble forecasting engine with 3-arm signal combiner."""

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
FREQ     = os.getenv("FORECAST_FREQUENCY", "B")
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


def _build_models():
    from statsforecast.models import AutoARIMA, AutoETS
    print("  Models: AutoARIMA + AutoETS")
    return [AutoARIMA(), AutoETS()]


def fetch_price_data(ticker: str, years: int = 2) -> pd.DataFrame:
    """Fetch OHLCV data; returns DataFrame with ds, y, high, low, volume, unique_id."""
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
    # normalise column names
    df.columns = [c.lower() if c.lower() != "date" else "ds" for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "ds"})

    df["ds"]        = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    df["unique_id"] = ticker

    # statsforecast needs 'y' column
    df = df.rename(columns={"close": "y"}) if "close" in df.columns else df
    if "y" not in df.columns:
        raise ValueError(f"No close/y column for {ticker}")

    df = df.set_index("ds")
    df = df.resample("B").last().ffill()
    df = df.reset_index()
    df["unique_id"] = ticker
    return df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)


def _ensemble_quantiles(fcst: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt_cols = [c for c in fcst.columns if c not in ("unique_id", "ds")
               and "-lo-" not in c and "-hi-" not in c]
    lo_cols = [c for c in fcst.columns if "-lo-80" in c]
    hi_cols = [c for c in fcst.columns if "-hi-80" in c]
    p50 = fcst[pt_cols].median(axis=1).values
    p10 = fcst[lo_cols].median(axis=1).values if lo_cols else p50 * 0.98
    p90 = fcst[hi_cols].median(axis=1).values if hi_cols else p50 * 1.02
    return p10, p50, p90


def _get_poly_signal(ticker: str, poly_df: pd.DataFrame) -> float | None:
    """Return calibrated Polymarket bullish probability for ticker, or None."""
    if poly_df.empty or "poly_prob_bullish" not in poly_df.columns:
        return None
    t = poly_df[poly_df["ticker"] == ticker]
    if t.empty:
        return None
    raw_prob = t.sort_values("date").iloc[-1]["poly_prob_bullish"]
    if pd.isna(raw_prob):
        return None
    from poly_calibration import calibrate_prob
    return calibrate_prob(float(raw_prob))


def _get_current_features(ticker: str, price_df: pd.DataFrame,
                           macro_df: pd.DataFrame, poly_df: pd.DataFrame) -> dict:
    """Compute latest technical + macro features from existing price_df."""
    try:
        from feature_pipeline import compute_features, FEATURE_COLS
        # Rename y → close for compute_features
        cdf = price_df.rename(columns={"y": "close"})
        feat_df = compute_features(cdf, macro_df, poly_df, ticker=ticker)
        if feat_df.empty:
            return {}
        latest = feat_df.iloc[-1]
        return {col: (None if pd.isna(latest.get(col)) else float(latest[col]))
                for col in FEATURE_COLS if col in feat_df.columns}
    except Exception as e:
        print(f"  [{ticker}] feature extraction warning: {e}")
        return {}


def forecast_ticker(ticker: str, models, existing: pd.DataFrame,
                    macro_df: pd.DataFrame = None,
                    poly_df:  pd.DataFrame = None) -> list[dict]:
    from statsforecast import StatsForecast
    from pandas.tseries.offsets import BDay

    if macro_df is None:
        macro_df = pd.DataFrame()
    if poly_df is None:
        poly_df = pd.DataFrame()

    today_str = str(date.today())
    if not existing.empty:
        dupe = existing[
            (existing["forecast_date"].astype(str) == today_str) &
            (existing["ticker"] == ticker)
        ]
        if not dupe.empty:
            print(f"  [{ticker}] already forecasted today — skipping.")
            return []

    price_df   = fetch_price_data(ticker)
    last_price = float(price_df["y"].iloc[-1])
    print(f"  [{ticker}] {len(price_df)} pts, last={last_price:.4f} — running statsforecast…")

    # ── TimeCopilot ARM: statsforecast ──────────────────────────────────────────
    sf_df = price_df[["ds", "y", "unique_id"]].copy()
    sf    = StatsForecast(models=models, freq=FREQ, n_jobs=-1)
    fcst  = sf.forecast(df=sf_df, h=HORIZON, level=[80])
    p10_vals, p50_vals, p90_vals = _ensemble_quantiles(fcst)

    # ── ATR for thresholding ────────────────────────────────────────────────────
    atr_14_pct = 0.005  # sensible default
    try:
        if "high" in price_df.columns and "low" in price_df.columns:
            from feature_pipeline import _compute_atr
            atr_series = _compute_atr(price_df["high"], price_df["low"], price_df["y"])
            last_atr   = float(atr_series.dropna().iloc[-1])
            atr_14_pct = last_atr / max(abs(last_price), 1e-9)
    except Exception:
        pass

    # ── Polymarket ARM ──────────────────────────────────────────────────────────
    poly_prob = _get_poly_signal(ticker, poly_df)

    # ── DirectionalCLF ARM ──────────────────────────────────────────────────────
    clf_prob: float | None = None
    try:
        from directional_classifier import predict_prob_up
        features = _get_current_features(ticker, price_df, macro_df, poly_df)
        if features:
            clf_prob = predict_prob_up(ticker, features)
    except Exception as e:
        print(f"  [{ticker}] CLF inference warning: {e}")

    # ── Build forecast rows ──────────────────────────────────────────────────────
    from signal_combiner import combine_signals

    last_date    = pd.Timestamp(price_df["ds"].max())
    target_dates = [last_date + BDay(i + 1) for i in range(HORIZON)]

    rows = []
    for i, tgt in enumerate(target_dates):
        p10 = round(float(p10_vals[i]), 6)
        p50 = round(float(p50_vals[i]), 6)
        p90 = round(float(p90_vals[i]), 6)

        sig = combine_signals(
            ticker=ticker,
            p10=p10, p50=p50, p90=p90,
            last_price=last_price, atr_14_pct=atr_14_pct,
            clf_prob_up=clf_prob,
            poly_prob_bullish=poly_prob,
        )

        # Natural Gas specialist adjustment on day-1
        if ticker == "NG=F" and i == 0:
            try:
                from ng_specialist import ng_adjust_signal
                sig = ng_adjust_signal(sig, tgt.date())
            except Exception:
                pass

        rows.append({
            "forecast_date":    today_str,
            "target_date":      tgt.date().isoformat(),
            "ticker":           ticker,
            "p10":              p10,
            "p50":              p50,
            "p90":              p90,
            "actual":           "",
            "model_used":       "3Arm_Ensemble",
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
    print(f"  [{ticker}] ✓ {direction} conv={conviction}  "
          f"tc={sig['arm_tc_prob']:.2f} clf={'n/a' if clf_prob is None else f'{clf_prob:.2f}'}  "
          f"poly={'n/a' if poly_prob is None else f'{poly_prob:.2f}'}")
    return rows


def run_all_forecasts():
    print("=== Trading Co-Pilot: Forecast Run ===")
    models   = _build_models()
    existing = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

    # Load context DataFrames once
    macro_df = pd.read_csv("macro_context.csv")   if os.path.exists("macro_context.csv")   else pd.DataFrame()
    poly_df  = pd.read_csv("polymarket_data.csv") if os.path.exists("polymarket_data.csv") else pd.DataFrame()

    all_rows, skipped = [], []
    for ticker in TICKERS:
        try:
            rows = forecast_ticker(ticker, models, existing, macro_df, poly_df)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [{ticker}] ✗ FAILED: {e}")
            skipped.append(ticker)

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

    if skipped:
        print(f"Skipped: {skipped}")
    return len(all_rows)


if __name__ == "__main__":
    run_all_forecasts()
