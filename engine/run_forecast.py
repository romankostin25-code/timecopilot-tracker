"""Multi-horizon forecasting engine (5d / 30d / 90d)."""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

HORIZONS = [int(h) for h in os.getenv("FORECAST_HORIZONS", "5,30,90").split(",")]
FREQ = os.getenv("FORECAST_FREQUENCY", "B")
MAX_HORIZON = max(HORIZONS)
CSV_PATH = "data/forecasts.csv"


def fetch_price_data(ticker, years=3):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    period_map = {"DX-Y.NYB": "3y", "^TNX": "3y", "^IRX": "3y", "^VIX": "3y"}
    if ticker in period_map:
        raw = yf.download(ticker, period=period_map[ticker], auto_adjust=True, progress=False)
    else:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    df = raw[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df["unique_id"] = ticker
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows")
    return df


def _build_forecaster():
    try:
        from timecopilot import TimeCopilotForecaster
        from timecopilot.models.foundation.chronos import Chronos
        from timecopilot.models.foundation.toto import Toto
        from timecopilot.models.stats import AutoARIMA, AutoETS
        from timecopilot.models.ml import AutoLGBM
        from timecopilot.models.ensembles import MedianEnsemble
        return TimeCopilotForecaster(
            models=[Chronos(), Toto(), AutoARIMA(), AutoETS(), AutoLGBM(), MedianEnsemble()]
        ), "TimeCopilot_MedianEnsemble"
    except ImportError:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS
        return StatsForecast(models=[AutoARIMA(), AutoETS()], freq=FREQ, n_jobs=-1), "StatsForecast_Ensemble"


def compute_signals(p10_d1, p50_d1, p50_target, p90_d1, last_price, horizon):
    thresholds = {5: 0.003, 30: 0.008, 90: 0.015}
    threshold = thresholds.get(horizon, 0.005)
    forecast_return = (p50_target - last_price) / last_price
    direction = (
        "BULLISH" if forecast_return > threshold else
        "BEARISH" if forecast_return < -threshold else
        "NEUTRAL"
    )
    signal_strength = round(abs(forecast_return) * 100, 4)
    band_width = (p90_d1 - p10_d1) / last_price
    conviction_score = round(max(0, 1 - (band_width / 0.05)), 4)
    return direction, signal_strength, conviction_score


def run_all_forecasts():
    from engine.universe import ALL_TICKERS
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", ",".join(ALL_TICKERS)).split(",") if t.strip()]

    forecaster, model_name = _build_forecaster()
    forecast_date = datetime.today().date()
    new_rows, skipped = [], []

    for ticker in tickers:
        print(f"\n[{ticker}] Fetching data...")
        try:
            df = fetch_price_data(ticker)
            last_price = df["y"].iloc[-1]
            print(f"[{ticker}] Forecasting {MAX_HORIZON}d ({len(df)} pts)...")

            # Single forecast call — extract all horizons from result
            try:
                fcst = forecaster.forecast(df=df, h=MAX_HORIZON)
            except TypeError:
                # statsforecast path
                fcst = forecaster.forecast(df=df, h=MAX_HORIZON, level=[80])

            fcst_rows = fcst.reset_index(drop=True)
            all_cols = [c for c in fcst_rows.columns if c not in ["unique_id", "ds"]]
            p10_col = next((c for c in all_cols if any(k in c.lower() for k in ["lo", "q10", "p10", "10"])), None)
            p50_col = next((c for c in all_cols if any(k in c.lower() for k in ["median", "q50", "p50", "mean", "50"])), all_cols[0])
            p90_col = next((c for c in all_cols if any(k in c.lower() for k in ["hi", "q90", "p90", "90"])), None)

            p50_vals = fcst_rows[p50_col].values
            p10_vals = fcst_rows[p10_col].values if p10_col else p50_vals * 0.99
            p90_vals = fcst_rows[p90_col].values if p90_col else p50_vals * 1.01

            for horizon in HORIZONS:
                h_idx = min(horizon - 1, len(p50_vals) - 1)
                p50_h = round(float(p50_vals[h_idx]), 6)
                p10_h = round(float(p10_vals[h_idx]), 6)
                p90_h = round(float(p90_vals[h_idx]), 6)
                p50_d1 = float(p50_vals[0])
                p10_d1 = float(p10_vals[0])
                p90_d1 = float(p90_vals[0])

                target_date = (datetime.today() + timedelta(days=horizon)).date()
                direction, signal_strength, conviction = compute_signals(
                    p10_d1, p50_d1, p50_h, p90_d1, last_price, horizon
                )
                new_rows.append({
                    "forecast_date": str(forecast_date),
                    "target_date":   str(target_date),
                    "ticker":        ticker,
                    "horizon":       horizon,
                    "p10": p10_h, "p50": p50_h, "p90": p90_h,
                    "actual": "",
                    "model_used":       model_name,
                    "direction":        direction,
                    "signal_strength":  signal_strength,
                    "conviction_score": conviction,
                    "poly_signal": "", "poly_regime": "", "poly_confidence": "",
                    "poly_alignment": "", "poly_band_adj_pct": "",
                    "news_signal": "", "news_confidence": "", "news_top_headline": "",
                    "error_abs": "", "error_pct": "", "hit": "",
                    "direction_correct": "", "graded_at": "", "notes": "",
                })

            print(f"[{ticker}] ✓ {len(HORIZONS)} horizons | last={last_price:.4f}")

        except Exception as e:
            print(f"[{ticker}] ✗ {e}")
            skipped.append(ticker)

    os.makedirs("data", exist_ok=True)
    existing = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    new_df = pd.DataFrame(new_rows)

    if not existing.empty and not new_df.empty:
        for col in ["forecast_date", "target_date"]:
            existing[col] = existing[col].astype(str)
            new_df[col] = new_df[col].astype(str)
        key_cols = ["ticker", "forecast_date", "target_date", "horizon"]
        # Only keep new rows whose keys don't exist yet
        if all(c in existing.columns for c in key_cols) and all(c in new_df.columns for c in key_cols):
            existing_keys = set(
                zip(existing["ticker"], existing["forecast_date"].astype(str),
                    existing["target_date"].astype(str), existing["horizon"].astype(str))
            )
            new_df = new_df[new_df.apply(
                lambda r: (r["ticker"], str(r["forecast_date"]), str(r["target_date"]), str(r["horizon"]))
                not in existing_keys, axis=1
            )]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"\n✓ {len(new_df)} new rows. {len(skipped)} skipped: {skipped}")


if __name__ == "__main__":
    run_all_forecasts()
