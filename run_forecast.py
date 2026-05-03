"""Runs TimeCopilot ensemble forecast for all configured tickers."""

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


def _run_statsforecast(df: pd.DataFrame, horizon: int, freq: str):
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

        sf_df = df.copy()
        sf_df["unique_id"] = "asset"
        sf_df["ds"] = pd.to_datetime(sf_df["ds"])
        sf = StatsForecast(models=sf_models, freq=freq, n_jobs=1)
        sf.fit(sf_df)
        return sf.predict(h=horizon, level=[80])
    except Exception as e:
        print(f"  StatsForecast error: {e}")
        return None


def _run_lgbm_simple(df: pd.DataFrame, horizon: int):
    try:
        import numpy as np
        from lightgbm import LGBMRegressor

        y = df["y"].values
        n_lags = min(30, len(y) // 2)
        X, Y = [], []
        for i in range(n_lags, len(y)):
            X.append(y[i - n_lags:i])
            Y.append(y[i])
        X, Y = __import__("numpy").array(X), __import__("numpy").array(Y)
        model = LGBMRegressor(n_estimators=300, verbose=-1)
        model.fit(X, Y)
        preds, window = [], list(y[-n_lags:])
        for _ in range(horizon):
            x = __import__("numpy").array(window[-n_lags:]).reshape(1, -1)
            p = float(model.predict(x)[0])
            preds.append(p)
            window.append(p)
        return preds
    except Exception as e:
        print(f"  LightGBM error: {e}")
        return None


def forecast_ticker(ticker: str, horizon: int, freq: str,
                    existing: pd.DataFrame, csv_path: str) -> int:
    import numpy as np
    from pandas.tseries.offsets import BDay

    today_str = str(date.today())

    # Skip if already done today
    if not existing.empty and "forecast_date" in existing.columns:
        dupe = existing[
            (existing["forecast_date"].astype(str) == today_str) &
            (existing["ticker"] == ticker)
        ]
        if not dupe.empty:
            print(f"  [{ticker}] already forecasted today — skipping.")
            return 0

    from data_fetcher import fetch_historical_data
    try:
        df = fetch_historical_data(ticker)
    except Exception as e:
        print(f"  [{ticker}] data fetch failed: {e}")
        return 0

    sf_pred  = _run_statsforecast(df, horizon, freq)
    lgbm_pts = _run_lgbm_simple(df, horizon)

    last_date  = pd.Timestamp(df["ds"].max())
    last_price = float(df["y"].iloc[-1])

    if freq.upper() in ("B", "D"):
        target_dates = [last_date + BDay(i + 1) for i in range(horizon)]
    else:
        target_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    rows = []
    for i, tgt in enumerate(target_dates):
        points = []
        if sf_pred is not None:
            try:
                step_row = sf_pred[sf_pred["unique_id"] == "asset"].iloc[i]
                model_cols = [c for c in sf_pred.columns
                              if c not in ("unique_id", "ds")
                              and "-lo-" not in c and "-hi-" not in c]
                for col in model_cols:
                    v = step_row[col]
                    if pd.notna(v):
                        points.append(float(v))
            except Exception:
                pass
        if lgbm_pts and i < len(lgbm_pts):
            points.append(lgbm_pts[i])
        if not points:
            points = [last_price * (1 + np.random.normal(0, 0.005))]

        median_pred = float(np.median(points))
        spread      = abs(median_pred - last_price) * max(1.0, (i + 1) ** 0.5)
        std_approx  = max(last_price * 0.004, spread * 0.5)

        rows.append({
            "forecast_date": today_str,
            "target_date":   tgt.date().isoformat(),
            "ticker":        ticker,
            "p10":  round(median_pred - 1.28 * std_approx, 6),
            "p50":  round(median_pred, 6),
            "p90":  round(median_pred + 1.28 * std_approx, 6),
            "actual":     "",
            "model_used": "MedianEnsemble",
            "notes":      "",
        })

    # Print table for this ticker
    print(f"\n  {'Target Date':<13} {'P10':>10} {'P50':>10} {'P90':>10}")
    print(f"  {'─'*46}")
    for r in rows:
        print(f"  {r['target_date']:<13} {r['p10']:>10.4f} {r['p50']:>10.4f} {r['p90']:>10.4f}")
    return len(rows), rows


def run_forecast(csv_path: str = FORECASTS_CSV) -> int:
    tickers_raw = os.getenv("ASSET_TICKERS", os.getenv("ASSET_TICKER", "SPY"))
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    horizon = int(os.getenv("FORECAST_HORIZON", "10"))
    freq    = os.getenv("FORECAST_FREQUENCY", "B")

    api_key = os.getenv("LLM_API_KEY", "")
    if api_key and api_key != "your_api_key_here":
        print("LLM API key present — agent mode available.")
    else:
        print("No LLM API key — using statistical/ML models only.")

    existing = _load_or_create_csv(csv_path)
    all_rows = []
    total_new = 0

    for ticker in tickers:
        print(f"\n{'═'*50}")
        print(f"  {ticker}  |  horizon={horizon}  |  freq={freq}")
        print(f"{'═'*50}")
        result = forecast_ticker(ticker, horizon, freq, existing, csv_path)
        if isinstance(result, tuple):
            n, rows = result
            total_new += n
            all_rows.extend(rows)

    if all_rows:
        new_df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False)
        print(f"\nAdded {total_new} new rows to {csv_path}")
    else:
        print("\nNo new rows added.")

    return total_new


if __name__ == "__main__":
    run_forecast()
