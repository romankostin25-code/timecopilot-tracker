"""Polymarket probability calibration and accuracy statistics."""

import numpy as np
import pandas as pd
from pathlib import Path


def calibrate_prob(raw_prob: float, shrinkage: float = 0.12) -> float:
    """Platt-style shrinkage toward 0.5 to reduce overconfidence."""
    if raw_prob is None:
        return 0.5
    return round(float(raw_prob) * (1 - shrinkage) + 0.5 * shrinkage, 4)


def compute_poly_calibration_stats(
    forecasts_csv: str = "forecasts.csv",
    poly_csv:      str = "polymarket_data.csv",
) -> dict:
    """
    Compute per-ticker directional accuracy of Polymarket signals vs actuals.
    Returns {ticker: {poly_directional_accuracy, n}}.
    """
    results: dict = {}
    try:
        fdf = pd.read_csv(forecasts_csv)
        pdf = pd.read_csv(poly_csv)
    except Exception:
        return results

    graded = fdf[
        fdf["direction_correct"].notna() &
        (fdf["direction_correct"].astype(str).str.strip() != "")
    ].copy()
    graded["direction_correct"] = pd.to_numeric(graded["direction_correct"], errors="coerce")
    graded = graded.dropna(subset=["direction_correct"])

    for ticker in graded["ticker"].unique():
        t_graded = graded[graded["ticker"] == ticker]
        t_poly   = pdf[pdf["ticker"] == ticker]

        if t_poly.empty or "poly_prob_bullish" not in t_poly.columns:
            continue

        merged = t_graded.merge(
            t_poly[["date", "poly_prob_bullish"]].rename(columns={"date": "forecast_date"}),
            on="forecast_date",
            how="left",
        )
        valid = merged.dropna(subset=["poly_prob_bullish", "direction_correct"])
        if len(valid) < 10:
            continue

        poly_pred = (valid["poly_prob_bullish"] > 0.5).astype(int)
        actual    = valid["direction_correct"].astype(int)
        accuracy  = float((poly_pred == actual).mean())

        results[ticker] = {
            "poly_directional_accuracy": round(accuracy, 4),
            "n": len(valid),
        }

    return results


def save_calibration_stats(
    forecasts_csv: str = "forecasts.csv",
    poly_csv:      str = "polymarket_data.csv",
    out_path:      str = "data/poly_calibration.json",
) -> None:
    import json
    stats = compute_poly_calibration_stats(forecasts_csv, poly_csv)
    Path(out_path).parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[poly_calibration] saved {len(stats)} tickers → {out_path}")
