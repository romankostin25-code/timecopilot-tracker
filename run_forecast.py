"""Trading Co-Pilot — TimeCopilot ensemble forecasting engine."""

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
]


def _build_models():
    """Assemble available TimeCopilot models; skip any that fail to import."""
    from timecopilot.models.stats import AutoARIMA, AutoETS
    models = [AutoARIMA(), AutoETS()]
    try:
        from timecopilot.models.ml import AutoLGBM
        models.append(AutoLGBM())
        print("  + AutoLGBM")
    except Exception as e:
        print(f"  - AutoLGBM skipped: {e}")
    try:
        from timecopilot.models.foundation.chronos import Chronos
        models.append(Chronos())
        print("  + Chronos")
    except Exception as e:
        print(f"  - Chronos skipped: {e}")
    try:
        from timecopilot.models.foundation.toto import Toto
        models.append(Toto())
        print("  + Toto")
    except Exception as e:
        print(f"  - Toto skipped: {e}")
    return models


def fetch_price_data(ticker: str, years: int = 2) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=365 * years)
    raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if raw.empty:
        raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df["unique_id"] = ticker
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    # Resample to business-day frequency so TimeCopilot can infer freq cleanly
    df = df.set_index("ds")
    df = df.resample("B").last().ffill()
    df = df.reset_index()
    df["unique_id"] = ticker
    return df


def _ensemble_quantiles(fcst: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive P10/P50/P90 by taking median across all model outputs."""
    # Point forecast columns (no '-lo-' or '-hi-')
    pt_cols  = [c for c in fcst.columns if c not in ("unique_id", "ds")
                and "-lo-" not in c and "-hi-" not in c]
    lo_cols  = [c for c in fcst.columns if "-lo-80" in c]
    hi_cols  = [c for c in fcst.columns if "-hi-80" in c]

    p50 = fcst[pt_cols].median(axis=1).values
    p10 = fcst[lo_cols].median(axis=1).values if lo_cols else p50 * 0.98
    p90 = fcst[hi_cols].median(axis=1).values if hi_cols else p50 * 1.02
    return p10, p50, p90


def compute_signals(p10_d1, p50_d1, p50_d5, p90_d1, last_price):
    ret_d1 = (p50_d1 - last_price) / last_price
    ret_d5 = (p50_d5 - last_price) / last_price
    direction = (
        "BULLISH" if ret_d1 > 0.003 else
        "BEARISH" if ret_d1 < -0.003 else
        "NEUTRAL"
    )
    signal_strength  = round(abs(ret_d5) * 100, 4)
    band_width       = (p90_d1 - p10_d1) / abs(last_price)
    conviction_score = round(max(0.0, 1.0 - band_width / 0.05), 4)
    return direction, signal_strength, conviction_score


def forecast_ticker(ticker: str, models, existing: pd.DataFrame) -> list[dict]:
    from timecopilot import TimeCopilotForecaster
    from pandas.tseries.offsets import BDay

    today_str = str(date.today())
    if not existing.empty:
        dupe = existing[
            (existing["forecast_date"].astype(str) == today_str) &
            (existing["ticker"] == ticker)
        ]
        if not dupe.empty:
            print(f"  [{ticker}] already forecasted today — skipping.")
            return []

    df = fetch_price_data(ticker)
    last_price = float(df["y"].iloc[-1])
    print(f"  [{ticker}] {len(df)} pts, last={last_price:.4f} — running TimeCopilot…")

    tcf  = TimeCopilotForecaster(models=models)
    fcst = tcf.forecast(df=df, h=HORIZON, level=[80])

    p10_vals, p50_vals, p90_vals = _ensemble_quantiles(fcst)

    last_date    = pd.Timestamp(df["ds"].max())
    target_dates = [last_date + BDay(i + 1) for i in range(HORIZON)]

    rows = []
    for i, tgt in enumerate(target_dates):
        p10 = round(float(p10_vals[i]), 6)
        p50 = round(float(p50_vals[i]), 6)
        p90 = round(float(p90_vals[i]), 6)
        p50_d5 = float(p50_vals[min(4, len(p50_vals) - 1)])
        direction, sig_str, conviction = compute_signals(p10, p50, p50_d5, p90, last_price)
        rows.append({
            "forecast_date":    today_str,
            "target_date":      tgt.date().isoformat(),
            "ticker":           ticker,
            "p10":              p10,
            "p50":              p50,
            "p90":              p90,
            "actual":           "",
            "model_used":       "TimeCopilot_MedianEnsemble",
            "direction":        direction,
            "signal_strength":  sig_str,
            "conviction_score": conviction,
            "error_abs":        "",
            "error_pct":        "",
            "hit":              "",
            "direction_correct": "",
            "graded_at":        "",
            "notes":            "",
        })

    print(f"  [{ticker}] ✓ direction={direction} conviction={conviction}")
    return rows


def run_all_forecasts():
    print("=== Trading Co-Pilot: Forecast Run ===")
    print("Building model stack…")
    models  = _build_models()
    existing = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

    all_rows, skipped = [], []
    for ticker in TICKERS:
        try:
            rows = forecast_ticker(ticker, models, existing)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [{ticker}] ✗ FAILED: {e}")
            skipped.append(ticker)

    if all_rows:
        new_df   = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Ensure all columns exist (migrate old CSV)
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
