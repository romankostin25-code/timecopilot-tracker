"""
Learning Aggregator — LightGBM meta-model trained on TimeCopilot + Polymarket
outputs and macro context. Generates its own price direction predictions.

Requires ≥30 graded rows per ticker before generating predictions.
Stores predictions in aggregator_forecasts.csv for accuracy tracking.
"""

import os
import warnings
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

FORECAST_CSV      = "forecasts.csv"
MACRO_CSV         = "macro_context.csv"
POLY_CSV          = "polymarket_data.csv"
AGG_FORECAST_CSV  = "aggregator_forecasts.csv"
AGG_MODEL_DIR     = "data/agg_models"
MIN_TRAIN_ROWS    = 30

AGG_COLUMNS = [
    "forecast_date", "target_date", "ticker",
    "agg_p50", "agg_direction", "agg_confidence",
    "actual", "agg_hit", "agg_direction_correct",
    "model_version", "graded_at",
]

os.makedirs(AGG_MODEL_DIR, exist_ok=True)


# ── Feature engineering ──────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame, macro_latest: dict, poly_latest: dict) -> pd.DataFrame:
    """Build feature matrix from forecasts + macro + poly for a single ticker."""
    rows = []
    for _, r in df.iterrows():
        try:
            p10 = float(r["p10"])
            p50 = float(r["p50"])
            p90 = float(r["p90"])
            band_width = (p90 - p10) / max(abs(p50), 1e-9)
            feat = {
                "p50":             p50,
                "p10":             p10,
                "p90":             p90,
                "band_width":      band_width,
                "conviction_score": float(r.get("conviction_score") or 0.5),
                "signal_strength":  float(r.get("signal_strength") or 0),
                "direction_enc":    1 if r.get("direction") == "BULLISH" else -1 if r.get("direction") == "BEARISH" else 0,
                "vix":              float(macro_latest.get("vix", 20)),
                "us10y":            float(macro_latest.get("us10y", 4.5)),
                "dxy":              float(macro_latest.get("dxy", 100)),
                "risk_regime_enc":  1 if macro_latest.get("risk_regime") == "RISK-ON" else -1 if macro_latest.get("risk_regime") == "RISK-OFF" else 0,
                "poly_signal_enc":  1 if poly_latest.get("poly_signal") == "BULLISH" else -1 if poly_latest.get("poly_signal") == "BEARISH" else 0,
                "poly_volume_log":  np.log1p(float(poly_latest.get("poly_total_volume_usd") or 0)),
            }
            rows.append(feat)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _build_target(df: pd.DataFrame) -> np.ndarray | None:
    """Binary target: 1 = price went up, 0 = went down. Only for graded rows."""
    y = []
    for _, r in df.iterrows():
        actual = r.get("actual")
        p50    = r.get("p50")
        if actual == "" or actual is None or pd.isna(actual):
            return None
        try:
            direction_correct = int(float(r.get("direction_correct", 0)))
            y.append(direction_correct)
        except Exception:
            return None
    return np.array(y, dtype=int)


# ── Training ──────────────────────────────────────────────────────────────────

def train_ticker(ticker: str, forecasts: pd.DataFrame, macro_df: pd.DataFrame, poly_df: pd.DataFrame):
    try:
        import lightgbm as lgb
    except ImportError:
        return None, None

    ticker_rows = forecasts[
        (forecasts["ticker"] == ticker) &
        (forecasts["actual"] != "") &
        (forecasts["actual"].notna()) &
        (forecasts["direction_correct"] != "") &
        (forecasts["direction_correct"].notna())
    ].copy()

    if len(ticker_rows) < MIN_TRAIN_ROWS:
        return None, len(ticker_rows)

    # Merge macro (use latest available per date)
    macro_by_date = {}
    for _, row in macro_df.iterrows():
        macro_by_date[str(row["date"])] = row.to_dict()

    # Merge poly (latest per ticker/date)
    poly_by_date = {}
    poly_ticker = poly_df[poly_df["ticker"] == ticker]
    for _, row in poly_ticker.iterrows():
        poly_by_date[str(row["date"])] = row.to_dict()

    X_rows, y_rows = [], []
    for _, r in ticker_rows.iterrows():
        fdate = str(r["forecast_date"])
        macro_ctx = macro_by_date.get(fdate, {})
        poly_ctx  = poly_by_date.get(fdate, {})

        try:
            band_width = (float(r["p90"]) - float(r["p10"])) / max(abs(float(r["p50"])), 1e-9)
            feat = [
                float(r["p50"]),
                float(r["p10"]),
                float(r["p90"]),
                band_width,
                float(r.get("conviction_score") or 0.5),
                float(r.get("signal_strength") or 0),
                1 if r.get("direction") == "BULLISH" else -1 if r.get("direction") == "BEARISH" else 0,
                float(macro_ctx.get("vix", 20)),
                float(macro_ctx.get("us10y", 4.5)),
                float(macro_ctx.get("dxy", 100)),
                1 if macro_ctx.get("risk_regime") == "RISK-ON" else -1 if macro_ctx.get("risk_regime") == "RISK-OFF" else 0,
                1 if poly_ctx.get("poly_signal") == "BULLISH" else -1 if poly_ctx.get("poly_signal") == "BEARISH" else 0,
                np.log1p(float(poly_ctx.get("poly_total_volume_usd") or 0)),
            ]
            X_rows.append(feat)
            y_rows.append(int(float(r["direction_correct"])))
        except Exception:
            continue

    if len(X_rows) < MIN_TRAIN_ROWS:
        return None, len(X_rows)

    X = np.array(X_rows)
    y = np.array(y_rows)

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        verbose=-1,
        random_state=42,
    )
    model.fit(X, y)

    model_path = Path(AGG_MODEL_DIR) / f"{ticker.replace('=', '_').replace('^', '_')}.pkl"
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, len(X_rows)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_ticker(ticker: str, latest_forecast_row: pd.Series,
                   macro_latest: dict, poly_latest: dict) -> dict | None:
    model_path = Path(AGG_MODEL_DIR) / f"{ticker.replace('=', '_').replace('^', '_')}.pkl"
    if not model_path.exists():
        return None

    try:
        import pickle, lightgbm as lgb
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        r = latest_forecast_row
        band_width = (float(r["p90"]) - float(r["p10"])) / max(abs(float(r["p50"])), 1e-9)
        feat = np.array([[
            float(r["p50"]),
            float(r["p10"]),
            float(r["p90"]),
            band_width,
            float(r.get("conviction_score") or 0.5),
            float(r.get("signal_strength") or 0),
            1 if r.get("direction") == "BULLISH" else -1 if r.get("direction") == "BEARISH" else 0,
            float(macro_latest.get("vix", 20)),
            float(macro_latest.get("us10y", 4.5)),
            float(macro_latest.get("dxy", 100)),
            1 if macro_latest.get("risk_regime") == "RISK-ON" else -1 if macro_latest.get("risk_regime") == "RISK-OFF" else 0,
            1 if poly_latest.get("poly_signal") == "BULLISH" else -1 if poly_latest.get("poly_signal") == "BEARISH" else 0,
            np.log1p(float(poly_latest.get("poly_total_volume_usd") or 0)),
        ]])

        proba = model.predict_proba(feat)[0]
        p_up  = float(proba[1]) if len(proba) > 1 else 0.5
        direction = "BULLISH" if p_up > 0.55 else "BEARISH" if p_up < 0.45 else "NEUTRAL"
        confidence = abs(p_up - 0.5) * 2  # scale to [0, 1]

        return {
            "agg_p50":       float(r["p50"]),
            "agg_direction": direction,
            "agg_confidence": round(confidence, 4),
            "model_version": "lgbm_v1",
        }
    except Exception as e:
        print(f"  [aggregator] predict error for {ticker}: {e}")
        return None


# ── Grading ───────────────────────────────────────────────────────────────────

def grade_aggregator_forecasts():
    if not Path(AGG_FORECAST_CSV).exists():
        return

    df = pd.read_csv(AGG_FORECAST_CSV)
    ungraded = df[(df["actual"] == "") | (df["actual"].isna())]

    if ungraded.empty:
        return

    today = date.today()
    count = 0

    for idx, row in ungraded.iterrows():
        target = pd.to_datetime(row["target_date"]).date()
        if target > today:
            continue

        ticker = row["ticker"]
        try:
            raw = yf.download(ticker, start=str(target - timedelta(days=2)),
                              end=str(target + timedelta(days=3)),
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            prices = raw["Close"].dropna()
            price_dates = [d.date() for d in prices.index]

            actual = None
            for off in range(4):
                chk = target + timedelta(days=off)
                if chk in price_dates:
                    actual = float(prices.iloc[price_dates.index(chk)])
                    break
            if actual is None:
                continue

            # Fetch the price at forecast_date to determine direction
            fdate = pd.to_datetime(row["forecast_date"]).date()
            base_raw = yf.download(ticker, start=str(fdate - timedelta(days=2)),
                                   end=str(fdate + timedelta(days=3)),
                                   auto_adjust=True, progress=False)
            if not base_raw.empty:
                if isinstance(base_raw.columns, pd.MultiIndex):
                    base_raw.columns = base_raw.columns.get_level_values(0)
                base_prices = base_raw["Close"].dropna()
                base_dates  = [d.date() for d in base_prices.index]
                base_price  = None
                for off in range(4):
                    chk = fdate + timedelta(days=off)
                    if chk in base_dates:
                        base_price = float(base_prices.iloc[base_dates.index(chk)])
                        break
                if base_price:
                    actual_up = actual > base_price
                    pred_up   = row["agg_direction"] == "BULLISH"
                    dir_correct = int(actual_up == pred_up) if row["agg_direction"] != "NEUTRAL" else None
                else:
                    dir_correct = None
            else:
                dir_correct = None

            df.loc[idx, "actual"]                = round(actual, 6)
            df.loc[idx, "agg_hit"]               = ""  # not applicable for direction model
            df.loc[idx, "agg_direction_correct"]  = dir_correct if dir_correct is not None else ""
            df.loc[idx, "graded_at"]             = str(date.today())
            count += 1

        except Exception as e:
            print(f"  [aggregator] grading error {ticker}: {e}")
            continue

    df.to_csv(AGG_FORECAST_CSV, index=False)
    if count:
        print(f"[aggregator] Graded {count} aggregator forecasts.")


# ── Main run ──────────────────────────────────────────────────────────────────

def run_aggregator():
    print("=== Aggregator: Training + Prediction ===")

    if not Path(FORECAST_CSV).exists():
        print("[aggregator] forecasts.csv not found — skipping.")
        return

    forecasts  = pd.read_csv(FORECAST_CSV)
    macro_df   = pd.read_csv(MACRO_CSV) if Path(MACRO_CSV).exists() else pd.DataFrame()
    poly_df    = pd.read_csv(POLY_CSV)  if Path(POLY_CSV).exists()  else pd.DataFrame()

    macro_latest = {}
    if not macro_df.empty:
        macro_latest = macro_df.sort_values("date").iloc[-1].to_dict()

    poly_latest_by_ticker = {}
    if not poly_df.empty:
        for _, row in poly_df.sort_values("date").groupby("ticker").last().reset_index().iterrows():
            poly_latest_by_ticker[row["ticker"]] = row.to_dict()

    today_str = str(date.today())
    latest_date = forecasts["forecast_date"].astype(str).max()
    today_forecasts = forecasts[forecasts["forecast_date"].astype(str) == latest_date]

    tickers = today_forecasts["ticker"].unique().tolist()
    all_rows = []

    existing_agg = pd.read_csv(AGG_FORECAST_CSV) if Path(AGG_FORECAST_CSV).exists() else pd.DataFrame()

    for ticker in tickers:
        poly_ctx = poly_latest_by_ticker.get(ticker, {})

        # Train / retrain model
        model, n_train = train_ticker(ticker, forecasts, macro_df, poly_df)
        if model is None:
            rows_so_far = n_train if n_train is not None else 0
            print(f"  [{ticker}] skipping — {rows_so_far}/{MIN_TRAIN_ROWS} graded rows (need {MIN_TRAIN_ROWS})")
            continue

        print(f"  [{ticker}] model trained on {n_train} rows")

        # Check if already predicted today
        if not existing_agg.empty:
            dupe = existing_agg[
                (existing_agg["forecast_date"].astype(str) == today_str) &
                (existing_agg["ticker"] == ticker)
            ]
            if not dupe.empty:
                print(f"  [{ticker}] already predicted today — skipping.")
                continue

        # Get today's day-1 forecast row
        row_today = today_forecasts[today_forecasts["ticker"] == ticker]
        if row_today.empty:
            continue
        latest_row = row_today.sort_values("target_date").iloc[0]

        pred = predict_ticker(ticker, latest_row, macro_latest, poly_ctx)
        if pred is None:
            continue

        target_date = latest_row["target_date"]
        all_rows.append({
            "forecast_date":       today_str,
            "target_date":         target_date,
            "ticker":              ticker,
            "agg_p50":             pred["agg_p50"],
            "agg_direction":       pred["agg_direction"],
            "agg_confidence":      pred["agg_confidence"],
            "actual":              "",
            "agg_hit":             "",
            "agg_direction_correct": "",
            "model_version":       pred["model_version"],
            "graded_at":           "",
        })
        print(f"  [{ticker}] ✓ {pred['agg_direction']} (confidence={pred['agg_confidence']:.2f})")

    if all_rows:
        new_df   = pd.DataFrame(all_rows, columns=AGG_COLUMNS)
        combined = pd.concat([existing_agg, new_df], ignore_index=True) if not existing_agg.empty else new_df
        for col in AGG_COLUMNS:
            if col not in combined.columns:
                combined[col] = ""
        combined[AGG_COLUMNS].to_csv(AGG_FORECAST_CSV, index=False)
        print(f"\n[aggregator] {len(all_rows)} new prediction rows written.")
    else:
        print("\n[aggregator] No new predictions (insufficient training data or all skipped).")

    # Grade historical predictions
    grade_aggregator_forecasts()


if __name__ == "__main__":
    run_aggregator()
