"""Trading Co-Pilot — Daily grading engine."""

import os
import json
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

CSV_PATH = "forecasts.csv"


def fill_actuals_and_grade():
    if not os.path.exists(CSV_PATH):
        print("No forecasts.csv found.")
        return 0

    df = pd.read_csv(CSV_PATH)
    df["target_date"]   = pd.to_datetime(df["target_date"]).dt.date
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    today   = date.today()
    updated = 0

    needs = df[
        (df["actual"].isna() | (df["actual"].astype(str).str.strip() == "")) &
        (df["target_date"] < today)
    ]

    if needs.empty:
        print("No rows need actuals filled.")
    else:
        for ticker, group in needs.groupby("ticker"):
            dates_needed = group["target_date"].tolist()
            min_d = min(dates_needed) - timedelta(days=5)
            max_d = max(dates_needed) + timedelta(days=5)
            try:
                raw = yf.download(
                    ticker.strip(),
                    start=str(min_d),
                    end=str(max_d + timedelta(days=1)),
                    auto_adjust=True, progress=False,
                )
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                prices = raw["Close"].reset_index()
                prices.columns = ["dt", "price"]
                prices["dt"] = pd.to_datetime(prices["dt"]).dt.date

                for idx in group.index:
                    target = df.loc[idx, "target_date"]
                    actual = None
                    for offset in range(4):
                        match = prices[prices["dt"] == target + timedelta(days=offset)]
                        if not match.empty:
                            actual = round(float(match["price"].iloc[0]), 6)
                            break
                    if actual is None:
                        continue

                    df.loc[idx, "actual"] = actual
                    p10 = float(df.loc[idx, "p10"])
                    p50 = float(df.loc[idx, "p50"])
                    p90 = float(df.loc[idx, "p90"])

                    df.loc[idx, "error_abs"] = round(abs(actual - p50), 6)
                    df.loc[idx, "error_pct"] = round(abs(actual - p50) / abs(p50) * 100, 4)
                    df.loc[idx, "hit"]        = 1 if p10 <= actual <= p90 else 0

                    # Direction correctness vs prev available actual
                    prev_rows = df[
                        (df["ticker"] == ticker) &
                        (df["target_date"] < target) &
                        (~df["actual"].isna()) &
                        (df["actual"].astype(str).str.strip() != "")
                    ]
                    if not prev_rows.empty:
                        prev_close = float(prev_rows.sort_values("target_date").iloc[-1]["actual"])
                    else:
                        prev_close = p50
                    actual_dir = "BULLISH" if actual > prev_close else "BEARISH"
                    called_dir = str(df.loc[idx, "direction"]) if "direction" in df.columns else "NEUTRAL"
                    df.loc[idx, "direction_correct"] = 1 if called_dir == actual_dir else 0
                    df.loc[idx, "graded_at"] = datetime.now().isoformat()
                    updated += 1

            except Exception as e:
                print(f"[{ticker}] Error: {e}")

    df.to_csv(CSV_PATH, index=False)
    print(f"✓ Graded {updated} forecasts.")
    _regenerate_scorecard(df)
    return updated


def _regenerate_scorecard(df: pd.DataFrame):
    df["hit"]               = pd.to_numeric(df["hit"], errors="coerce")
    df["direction_correct"] = pd.to_numeric(df["direction_correct"], errors="coerce")
    graded  = df[df["hit"].notna()].copy()
    today   = date.today()
    yesterday = today - timedelta(days=1)

    def rolling_rate(subset, col, days):
        cutoff = today - timedelta(days=days)
        recent = subset[pd.to_datetime(subset["target_date"]).dt.date >= cutoff]
        return round(float(recent[col].mean()), 4) if not recent.empty else None

    cal_7d   = rolling_rate(graded, "hit", 7)
    cal_30d  = rolling_rate(graded, "hit", 30)
    cal_all  = round(float(graded["hit"].mean()), 4) if not graded.empty else None
    dir_7d   = rolling_rate(graded, "direction_correct", 7)
    dir_30d  = rolling_rate(graded, "direction_correct", 30)
    dir_all  = round(float(graded["direction_correct"].mean()), 4) if not graded.empty else None
    trend    = "IMPROVING" if (cal_7d or 0) > (cal_30d or 0) else "DECLINING"

    yest_graded = graded[pd.to_datetime(graded["target_date"]).dt.date == yesterday]

    by_asset = {}
    for ticker in graded["ticker"].unique():
        t    = graded[graded["ticker"] == ticker]
        yt   = yest_graded[yest_graded["ticker"] == ticker]
        consec = 0
        for val in t.sort_values("target_date", ascending=False)["hit"].values:
            if val == 1:
                consec += 1
            else:
                break
        by_asset[str(ticker)] = {
            "calibration_7d":            rolling_rate(t, "hit", 7),
            "calibration_30d":           rolling_rate(t, "hit", 30),
            "directional_accuracy_7d":   rolling_rate(t, "direction_correct", 7),
            "directional_accuracy_30d":  rolling_rate(t, "direction_correct", 30),
            "consecutive_hits":          consec,
            "yesterday_hit":             int(yt["hit"].iloc[0]) if not yt.empty else None,
            "yesterday_direction_correct": int(yt["direction_correct"].iloc[0]) if not yt.empty else None,
            "yesterday_error_pct":       round(float(yt["error_pct"].iloc[0]), 4) if not yt.empty else None,
            "yesterday_p50":             round(float(yt["p50"].iloc[0]), 4) if not yt.empty else None,
            "yesterday_actual":          round(float(yt["actual"].iloc[0]), 4) if not yt.empty else None,
        }

    misses = yest_graded[yest_graded["hit"] == 0][
        ["ticker", "p50", "actual", "error_pct", "direction_correct"]
    ].to_dict(orient="records")

    scorecard = {
        "generated_at": datetime.now().isoformat(),
        "yesterday": str(yesterday),
        "global": {
            "forecasts_graded_total":      len(graded),
            "forecasts_graded_yesterday":  len(yest_graded),
            "calibration_7d":              cal_7d,
            "calibration_30d":             cal_30d,
            "calibration_alltime":         cal_all,
            "directional_accuracy_7d":     dir_7d,
            "directional_accuracy_30d":    dir_30d,
            "directional_accuracy_alltime": dir_all,
            "trend":                       trend,
        },
        "by_asset":        by_asset,
        "yesterday_misses": misses,
    }

    with open("scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    print("✓ scorecard.json regenerated.")


if __name__ == "__main__":
    fill_actuals_and_grade()
