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
    close = raw["Close"].squeeze()
    df = close.reset_index()
    df.columns = ["ds", "y"]
    df["unique_id"] = ticker
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows")
    return df


CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}

def _freq_for(ticker):
    return "D" if ticker in CRYPTO_TICKERS else FREQ


def _ensure_regular_freq(df, freq):
    """Reindex to a gapless frequency so TimeCopilot can infer the interval.

    yfinance data has holiday gaps on business-day series which cause
    TimeCopilot's frequency detector to fail. Forward-filling to a complete
    date range removes those gaps without distorting the series.
    """
    ts = df[["ds", "y", "unique_id"]].copy()
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.set_index("ds").sort_index()
    if freq == "B":
        idx = pd.bdate_range(ts.index.min(), ts.index.max())
    elif freq == "D":
        idx = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    else:
        return df
    ts = ts.reindex(idx).ffill().dropna()
    ts.index.name = "ds"
    out = ts.reset_index()[["ds", "y", "unique_id"]]
    out["ds"] = pd.to_datetime(out["ds"]).dt.tz_localize(None).dt.normalize()
    return out


def _build_tc_forecaster():
    """Try TimeCopilot. Returns (forecaster, name) or (None, None)."""
    try:
        from timecopilot import TimeCopilotForecaster
        from timecopilot.models.stats import AutoARIMA, AutoETS

        # AutoLGBM excluded: known LightGBM "feature index -1" bug in this env
        models = [AutoARIMA(), AutoETS()]

        for cls_path, label in [
            ("timecopilot.models.foundation.chronos.Chronos", "Chronos"),
            ("timecopilot.models.foundation.toto.Toto",       "Toto"),
        ]:
            try:
                mod_name, cls_name = cls_path.rsplit(".", 1)
                import importlib
                models.insert(0, getattr(importlib.import_module(mod_name), cls_name)())
                print(f"[forecaster] {label} loaded")
            except Exception as e:
                print(f"[forecaster] {label} unavailable: {e}")

        loaded = [type(m).__name__ for m in models]
        print(f"[forecaster] TimeCopilot ready with: {loaded}")
        return TimeCopilotForecaster(models=models), "TimeCopilot_Ensemble"
    except Exception as e:
        print(f"[forecaster] TimeCopilot unavailable: {e}")
        return None, None


def _sf_forecast(df, freq, h):
    """StatsForecast ensemble: median of AutoARIMA + AutoETS + AutoTheta.
    Returns DataFrame with columns: unique_id, ds, p10, p50, p90."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    sf = StatsForecast(
        models=[AutoARIMA(), AutoETS(), AutoTheta()],
        freq=freq,
        n_jobs=1,
    )
    raw = sf.forecast(df=df, h=h, level=[80]).reset_index(drop=True)
    lo_cols = [c for c in raw.columns if "-lo-80" in c]
    hi_cols = [c for c in raw.columns if "-hi-80" in c]
    pt_cols = [c for c in raw.columns if c not in ("unique_id", "ds")
               and "-lo-" not in c and "-hi-" not in c]
    base = raw[["unique_id", "ds"]] if "unique_id" in raw.columns else raw[["ds"]]
    out = base.copy()
    out["p50"] = raw[pt_cols].median(axis=1) if pt_cols else raw.iloc[:, 2]
    out["p10"] = raw[lo_cols].median(axis=1) if lo_cols else out["p50"] * 0.99
    out["p90"] = raw[hi_cols].median(axis=1) if hi_cols else out["p50"] * 1.01
    return out


def _extract_quantiles(fcst_rows):
    """Extract (p50_vals, p10_vals, p90_vals) arrays from any forecaster output DataFrame."""
    cols = [c for c in fcst_rows.columns if c not in ("unique_id", "ds")]
    # Clean p10/p50/p90 columns already present (from _sf_forecast or TC with quantiles)
    if "p50" in cols:
        p50 = fcst_rows["p50"].values
        p10 = fcst_rows["p10"].values if "p10" in cols else p50 * 0.99
        p90 = fcst_rows["p90"].values if "p90" in cols else p50 * 1.01
        return p50, p10, p90
    # Generic column detection for TimeCopilotForecaster output
    p10_col = next((c for c in cols if any(k in c.lower() for k in ["lo", "q10", "p10", "0.1"])), None)
    p50_col = next((c for c in cols if any(k in c.lower() for k in ["median", "q50", "p50", "mean", "0.5"])), cols[0])
    p90_col = next((c for c in cols if any(k in c.lower() for k in ["hi", "q90", "p90", "0.9"])), None)
    print(f"[forecaster] TC columns detected — p10:{p10_col}  p50:{p50_col}  p90:{p90_col}")
    p50 = fcst_rows[p50_col].values
    p10 = fcst_rows[p10_col].values if p10_col else p50 * 0.99
    p90 = fcst_rows[p90_col].values if p90_col else p50 * 1.01
    return p50, p10, p90


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

    tc_forecaster, tc_name = _build_tc_forecaster()
    forecast_date = datetime.today().date()
    new_rows, skipped = [], []

    for ticker in tickers:
        print(f"\n[{ticker}] Fetching data...")
        try:
            df = fetch_price_data(ticker)
            last_price = float(df["y"].iloc[-1])
            freq = _freq_for(ticker)
            print(f"[{ticker}] Forecasting {MAX_HORIZON}d ({len(df)} pts, freq={freq})...")

            if tc_forecaster is not None:
                try:
                    df_tc = _ensure_regular_freq(df, freq)
                    fcst = tc_forecaster.forecast(df=df_tc, h=MAX_HORIZON)
                    model_name = tc_name
                    print(f"[{ticker}] TimeCopilot OK — cols: {list(fcst.columns)}")
                except Exception as e:
                    print(f"[{ticker}] TimeCopilot failed: {e} — falling back to StatsForecast")
                    fcst = _sf_forecast(df, freq, MAX_HORIZON)
                    model_name = f"StatsForecast_{freq}"
            else:
                fcst = _sf_forecast(df, freq, MAX_HORIZON)
                model_name = f"StatsForecast_{freq}"

            p50_vals, p10_vals, p90_vals = _extract_quantiles(fcst.reset_index(drop=True))

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

            print(f"[{ticker}] ✓ {len(HORIZONS)} horizons | last={last_price:.4f} | {model_name}")

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
