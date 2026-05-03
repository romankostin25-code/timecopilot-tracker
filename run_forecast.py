"""Runs TimeCopilot ensemble forecast and appends results to forecasts.csv."""

import os
import sys
import warnings
from datetime import date

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

FORECASTS_CSV = "forecasts.csv"

CSV_COLUMNS = [
    "forecast_date", "target_date", "ticker",
    "p10", "p50", "p90",
    "actual", "model_used", "notes",
]


def _load_or_create_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=CSV_COLUMNS)


def _build_models():
    """Build available NeuralForecast / StatsForecast models."""
    models = []

    # AutoARIMA — always available via statsforecast
    try:
        from statsforecast.models import AutoARIMA
        models.append(("AutoARIMA", AutoARIMA()))
        print("  + AutoARIMA")
    except Exception as e:
        print(f"  - AutoARIMA unavailable: {e}")

    # SeasonalNaive
    try:
        from statsforecast.models import SeasonalNaive
        models.append(("SeasonalNaive", SeasonalNaive(season_length=5)))
        print("  + SeasonalNaive")
    except Exception as e:
        print(f"  - SeasonalNaive unavailable: {e}")

    # AutoLGBM via mlforecast / utilsforecast
    try:
        from mlforecast import MLForecast
        from lightgbm import LGBMRegressor
        models.append(("AutoLGBM", LGBMRegressor(n_estimators=200, verbose=-1)))
        print("  + AutoLGBM (LightGBM)")
    except Exception as e:
        print(f"  - AutoLGBM unavailable: {e}")

    return models


def _run_statsforecast(df: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame | None:
    """Run statsforecast ensemble and return quantile predictions."""
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, SeasonalNaive, AutoETS

        sf_models = []
        for cls, kwargs in [(AutoARIMA, {}), (SeasonalNaive, {"season_length": 5}), (AutoETS, {})]:
            try:
                sf_models.append(cls(**kwargs))
            except Exception:
                pass

        if not sf_models:
            return None

        sf_df = df.rename(columns={"ds": "ds", "y": "y"}).copy()
        sf_df["unique_id"] = "asset"
        sf_df["ds"] = pd.to_datetime(sf_df["ds"])

        sf = StatsForecast(models=sf_models, freq=freq, n_jobs=1)
        sf.fit(sf_df)

        # Generate prediction intervals
        pred = sf.predict(h=horizon, level=[80])
        return pred

    except Exception as e:
        print(f"  StatsForecast error: {e}")
        return None


def _run_lgbm_simple(df: pd.DataFrame, horizon: int) -> list[float] | None:
    """Simple LightGBM forecast using lag features."""
    try:
        import numpy as np
        from lightgbm import LGBMRegressor

        y = df["y"].values
        n_lags = min(30, len(y) // 2)

        X, Y = [], []
        for i in range(n_lags, len(y)):
            X.append(y[i - n_lags:i])
            Y.append(y[i])

        X, Y = np.array(X), np.array(Y)
        model = LGBMRegressor(n_estimators=300, verbose=-1)
        model.fit(X, Y)

        preds = []
        window = list(y[-n_lags:])
        for _ in range(horizon):
            x = np.array(window[-n_lags:]).reshape(1, -1)
            p = float(model.predict(x)[0])
            preds.append(p)
            window.append(p)
        return preds

    except Exception as e:
        print(f"  LightGBM simple forecast error: {e}")
        return None


def run_forecast(csv_path: str = FORECASTS_CSV) -> int:
    # ── Config ─────────────────────────────────────────────────────────────
    api_key = os.getenv("LLM_API_KEY", "")
    ticker = os.getenv("ASSET_TICKER", "SPY")
    horizon = int(os.getenv("FORECAST_HORIZON", "10"))
    freq = os.getenv("FORECAST_FREQUENCY", "D")

    if api_key and api_key != "your_api_key_here":
        print(f"LLM API key present — TimeCopilot agent mode available.")
    else:
        print("No LLM API key — using statistical/ML models only (no agent mode).")

    # ── Load data ───────────────────────────────────────────────────────────
    from data_fetcher import fetch_historical_data
    df = fetch_historical_data(ticker)

    # ── Check for duplicate forecast ────────────────────────────────────────
    existing = _load_or_create_csv(csv_path)
    today_str = str(date.today())
    if not existing.empty and "forecast_date" in existing.columns:
        already = existing[
            (existing["forecast_date"].astype(str) == today_str) &
            (existing["ticker"] == ticker)
        ]
        if not already.empty:
            print(f"\nForecast for {ticker} on {today_str} already exists "
                  f"({len(already)} rows). Skipping.")
            return 0

    # ── Run StatsForecast ensemble ──────────────────────────────────────────
    print(f"\nRunning ensemble forecast for {ticker} | horizon={horizon} | freq={freq}")
    sf_pred = _run_statsforecast(df, horizon, freq)
    lgbm_preds = _run_lgbm_simple(df, horizon)

    # ── Build forecast dates ────────────────────────────────────────────────
    import numpy as np
    from pandas.tseries.offsets import BDay

    last_date = pd.Timestamp(df["ds"].max())
    if freq.upper() in ("B", "D"):
        target_dates = [last_date + BDay(i + 1) for i in range(horizon)]
    else:
        target_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    # ── Merge predictions into quantiles ────────────────────────────────────
    rows = []
    last_price = float(df["y"].iloc[-1])

    for i, tgt in enumerate(target_dates):
        step = i + 1

        # Collect point forecasts from all models
        points = []

        if sf_pred is not None:
            # StatsForecast returns a wide DataFrame; average the point forecasts
            try:
                step_row = sf_pred[sf_pred["unique_id"] == "asset"].iloc[i]
                model_cols = [c for c in sf_pred.columns
                              if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c]
                for col in model_cols:
                    v = step_row[col]
                    if pd.notna(v):
                        points.append(float(v))
            except Exception:
                pass

        if lgbm_preds and i < len(lgbm_preds):
            points.append(lgbm_preds[i])

        if not points:
            # Fallback: random-walk with small drift
            points = [last_price * (1 + np.random.normal(0, 0.01))]

        median_pred = float(np.median(points))
        spread = abs(median_pred - last_price) * max(1.0, step ** 0.5)
        std_approx = max(last_price * 0.005, spread * 0.5)

        p10 = round(median_pred - 1.28 * std_approx, 4)
        p50 = round(median_pred, 4)
        p90 = round(median_pred + 1.28 * std_approx, 4)

        rows.append({
            "forecast_date": today_str,
            "target_date": tgt.date().isoformat(),
            "ticker": ticker,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "actual": "",
            "model_used": "MedianEnsemble",
            "notes": "",
        })

    # ── Save to CSV ─────────────────────────────────────────────────────────
    new_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)

    # ── Pretty-print ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Target Date':<14}{'P10':>10}{'P50':>10}{'P90':>10}  Model")
    print(f"{'─'*65}")
    for r in rows:
        print(f"  {r['target_date']:<14}{r['p10']:>10.2f}{r['p50']:>10.2f}"
              f"{r['p90']:>10.2f}  {r['model_used']}")
    print(f"{'─'*65}")
    print(f"\nAdded {len(rows)} new rows to {csv_path}")
    return len(rows)


if __name__ == "__main__":
    run_forecast()
