"""Daily grading engine — fills actuals and scores forecasts at all horizons."""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "data/forecasts.csv"
SCORECARD_PATH = "data/scorecard.json"


def fill_actuals_and_grade():
    if not os.path.exists(CSV_PATH):
        print("No forecasts.csv found — skipping grading.")
        return
    df = pd.read_csv(CSV_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    today = date.today()
    updated = 0

    needs = df[
        (df["actual"].isna() | (df["actual"].astype(str) == "")) &
        (df["target_date"] < today)
    ]

    if needs.empty:
        print("No rows need actuals.")
    else:
        for ticker, group in needs.groupby("ticker"):
            target_dates  = group["target_date"].tolist()
            fc_dates_raw  = group["forecast_date"].tolist()
            fc_dates      = [pd.to_datetime(d).date() for d in fc_dates_raw]
            # Download range must cover both forecast dates (for direction baseline)
            # and target dates (for actual grading)
            min_d = min(min(target_dates), min(fc_dates)) - timedelta(days=5)
            max_d = max(target_dates) + timedelta(days=5)
            try:
                raw = yf.download(ticker.strip(), start=min_d,
                                  end=max_d + timedelta(days=1),
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    continue
                prices = raw["Close"].squeeze().reset_index()
                prices.columns = ["date", "price"]
                prices["date"] = pd.to_datetime(prices["date"]).dt.date

                for idx in group.index:
                    target = df.loc[idx, "target_date"]
                    actual = None
                    for offset in range(4):
                        m = prices[prices["date"] == target + timedelta(days=offset)]
                        if not m.empty:
                            actual = round(float(m["price"].iloc[0]), 6)
                            break
                    if actual is None:
                        continue

                    p10 = float(df.loc[idx, "p10"])
                    p50 = float(df.loc[idx, "p50"])
                    p90 = float(df.loc[idx, "p90"])

                    df.loc[idx, "actual"]    = actual
                    df.loc[idx, "error_abs"] = round(abs(actual - p50), 6)
                    df.loc[idx, "error_pct"] = round(abs(actual - p50) / p50 * 100, 4)
                    df.loc[idx, "hit"]       = 1 if p10 <= actual <= p90 else 0

                    # Direction baseline = closing price on forecast_date
                    # (stored as last_price if available, otherwise look up from prices)
                    stored_lp = df.loc[idx, "last_price"] if "last_price" in df.columns else None
                    baseline = None
                    try:
                        _lp = float(stored_lp) if stored_lp is not None else None
                        if _lp is not None and not pd.isna(_lp):
                            baseline = _lp
                    except (ValueError, TypeError):
                        pass
                    if baseline is None:
                        fc_date = pd.to_datetime(df.loc[idx, "forecast_date"]).date()
                        for offset in range(4):
                            m_fc = prices[prices["date"] == fc_date + timedelta(days=offset)]
                            if not m_fc.empty:
                                baseline = float(m_fc["price"].iloc[0])
                                break
                    if baseline is None:
                        baseline = p50  # last resort

                    actual_dir = "BULLISH" if actual > baseline else "BEARISH"
                    # NEUTRAL was never BULLISH/BEARISH — infer direction from p50 vs baseline
                    pred_dir = str(df.loc[idx, "direction"])
                    if pred_dir == "NEUTRAL":
                        pred_dir = "BULLISH" if p50 >= baseline else "BEARISH"
                    df.loc[idx, "direction_correct"] = 1 if pred_dir == actual_dir else 0
                    df.loc[idx, "graded_at"] = datetime.now().isoformat()
                    updated += 1

            except Exception as e:
                print(f"[{ticker}] Error: {e}")

    df.to_csv(CSV_PATH, index=False)
    print(f"✓ Graded {updated} forecasts.")
    _regenerate_scorecard(df)
    if updated > 0:
        from engine.train_direction_model import train_direction_model
        train_direction_model()


def _regenerate_scorecard(df):
    df["hit"]              = pd.to_numeric(df["hit"], errors="coerce")
    df["direction_correct"] = pd.to_numeric(df["direction_correct"], errors="coerce")
    df["horizon"]          = pd.to_numeric(df["horizon"], errors="coerce")
    graded = df[df["hit"].notna()]
    today  = date.today()

    def rolling(subset, col, days):
        cutoff = today - timedelta(days=days)
        recent = subset[pd.to_datetime(subset["target_date"]).dt.date >= cutoff]
        if recent.empty or recent[col].isna().all():
            return None
        return round(recent[col].mean(), 4)

    by_horizon = {}
    for h in [5, 30, 90]:
        hg = graded[graded["horizon"] == h]
        by_horizon[str(h)] = {
            "calibration_7d":              rolling(hg, "hit", 7),
            "calibration_14d":             rolling(hg, "hit", 14),
            "calibration_30d":             rolling(hg, "hit", 30),
            "calibration_alltime":         round(hg["hit"].mean(), 4) if not hg.empty else None,
            "directional_accuracy_7d":     rolling(hg, "direction_correct", 7),
            "directional_accuracy_14d":    rolling(hg, "direction_correct", 14),
            "directional_accuracy_30d":    rolling(hg, "direction_correct", 30),
            "directional_accuracy_alltime": round(hg["direction_correct"].mean(), 4) if not hg.empty else None,
            "forecasts_graded":            len(hg),
        }

    by_asset = {}
    for ticker in graded["ticker"].unique():
        t = graded[(graded["ticker"] == ticker) & (graded["horizon"] == 5)]
        consecutive = 0
        for val in t.sort_values("target_date", ascending=False)["hit"].values:
            if val == 1:
                consecutive += 1
            else:
                break
        by_asset[ticker] = {
            "calibration_7d":            rolling(t, "hit", 7),
            "calibration_14d":           rolling(t, "hit", 14),
            "calibration_30d":           rolling(t, "hit", 30),
            "directional_accuracy_7d":   rolling(t, "direction_correct", 7),
            "directional_accuracy_14d":  rolling(t, "direction_correct", 14),
            "directional_accuracy_30d":  rolling(t, "direction_correct", 30),
            "consecutive_hits":          consecutive,
        }

    h5 = by_horizon.get("5", {})
    trend = "IMPROVING" if (h5.get("calibration_7d") or 0) > (h5.get("calibration_30d") or 0) else "DECLINING"

    scorecard = {
        "generated_at": datetime.now().isoformat(),
        "global": {
            "forecasts_graded_total": len(graded),
            "trend": trend,
        },
        "by_horizon": by_horizon,
        "by_asset":   by_asset,
    }

    os.makedirs("data", exist_ok=True)
    with open(SCORECARD_PATH, "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    print("✓ scorecard.json regenerated.")


def regrade_direction_correct():
    """Re-grade direction_correct for all rows that already have actuals.

    Run once after fixing the direction grading baseline bug. Uses the same
    corrected logic as fill_actuals_and_grade() (forecast_date price as baseline,
    NEUTRAL mapped to p50 vs baseline).
    """
    if not os.path.exists(CSV_PATH):
        print("No forecasts.csv — nothing to re-grade.")
        return
    df = pd.read_csv(CSV_PATH)
    df["target_date"]   = pd.to_datetime(df["target_date"]).dt.date
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    today = date.today()

    has_actual = (
        df["actual"].notna() &
        (~df["actual"].astype(str).str.strip().isin(["", "nan", "None"]))
    )
    to_regrade = df[has_actual & (df["target_date"] < today)]

    if to_regrade.empty:
        print("No rows to re-grade.")
        _regenerate_scorecard(df)
        return

    updated = 0
    for ticker, group in to_regrade.groupby("ticker"):
        fc_dates     = [pd.to_datetime(d).date() for d in group["forecast_date"].tolist()]
        target_dates = group["target_date"].tolist()
        min_d = min(min(target_dates), min(fc_dates)) - timedelta(days=5)
        max_d = max(target_dates) + timedelta(days=5)
        try:
            raw = yf.download(ticker.strip(), start=min_d,
                              end=max_d + timedelta(days=1),
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            prices = raw["Close"].squeeze().reset_index()
            prices.columns = ["date", "price"]
            prices["date"] = pd.to_datetime(prices["date"]).dt.date

            for idx in group.index:
                actual = float(df.loc[idx, "actual"])
                p50    = float(df.loc[idx, "p50"])

                stored_lp = df.loc[idx, "last_price"] if "last_price" in df.columns else None
                baseline = None
                if stored_lp not in (None, "", "nan", float("nan")):
                    try:
                        baseline = float(stored_lp)
                    except (ValueError, TypeError):
                        pass
                if baseline is None:
                    fc_date = pd.to_datetime(df.loc[idx, "forecast_date"]).date()
                    for offset in range(4):
                        m_fc = prices[prices["date"] == fc_date + timedelta(days=offset)]
                        if not m_fc.empty:
                            baseline = float(m_fc["price"].iloc[0])
                            break
                if baseline is None:
                    baseline = p50

                actual_dir = "BULLISH" if actual > baseline else "BEARISH"
                pred_dir   = str(df.loc[idx, "direction"])
                if pred_dir == "NEUTRAL":
                    pred_dir = "BULLISH" if p50 >= baseline else "BEARISH"
                df.loc[idx, "direction_correct"] = 1 if pred_dir == actual_dir else 0
                updated += 1

        except Exception as e:
            print(f"[{ticker}] Error: {e}")

    df.to_csv(CSV_PATH, index=False)
    print(f"✓ Re-graded direction_correct for {updated} rows.")
    _regenerate_scorecard(df)


if __name__ == "__main__":
    fill_actuals_and_grade()
