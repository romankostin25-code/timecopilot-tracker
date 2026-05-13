"""
Integrator — injects Polymarket spike signals into TimeCopilot forecasts.
Adjusts P50, widens/narrows confidence bands, and stamps alignment flag.
"""

import json
import pandas as pd
from pathlib import Path

try:
    from .signal_generator import load_signals
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from signal_generator import load_signals

FORECAST_CSV = Path(__file__).parent.parent / "forecasts.csv"


def get_poly_adjustment(ticker: str, base_p50: float, base_p10: float, base_p90: float) -> dict:
    signals = load_signals()

    if ticker not in signals or signals[ticker]["direction"] == "NEUTRAL":
        sig = signals.get(ticker, {})
        return {
            "adjusted_p50":             base_p50,
            "adjusted_p10":             base_p10,
            "adjusted_p90":             base_p90,
            "poly_signal":              sig.get("signal", "NO_SIGNAL"),
            "poly_confidence":          sig.get("confidence_label", "NO_DATA"),
            "alignment":                "POLY_NEUTRAL" if sig.get("direction") == "NEUTRAL" else "UNKNOWN",
            "poly_composite_score":     sig.get("composite_score", 0),
            "poly_contributing_events": sig.get("contributing_events", 0),
            "poly_top_question":        sig.get("top_question", ""),
            "adjustment_pct":           0,
        }

    sig = signals[ticker]
    direction        = sig["direction"]
    composite_score  = float(sig["composite_score"])
    confidence       = float(sig["confidence"])
    confidence_label = sig["confidence_label"]

    # Determine model implied direction from forecast midpoint
    model_bullish = base_p50 > (base_p10 + base_p90) / 2

    if (direction == "BULLISH" and model_bullish) or (direction == "BEARISH" and not model_bullish):
        alignment = "ALIGNED"
    else:
        alignment = "DIVERGENT"

    # Shift P50 by up to 2% depending on composite_score and confidence
    max_adj = 0.02
    adjustment_pct = composite_score * confidence * max_adj
    adjusted_p50 = base_p50 * (1 + adjustment_pct)

    # Widen bands on divergence, narrow on high-confidence alignment
    if alignment == "ALIGNED" and confidence_label == "HIGH":
        band_factor = 0.92
    elif alignment == "DIVERGENT":
        band_factor = 1.15
    else:
        band_factor = 1.0

    band_half    = (base_p90 - base_p10) / 2
    adjusted_p10 = adjusted_p50 - band_half * band_factor
    adjusted_p90 = adjusted_p50 + band_half * band_factor

    return {
        "adjusted_p50":             round(adjusted_p50, 6),
        "adjusted_p10":             round(adjusted_p10, 6),
        "adjusted_p90":             round(adjusted_p90, 6),
        "poly_signal":              sig["signal"],
        "poly_confidence":          confidence_label,
        "alignment":                alignment,
        "poly_composite_score":     composite_score,
        "poly_contributing_events": sig.get("contributing_events", 0),
        "poly_top_question":        sig.get("top_question", ""),
        "adjustment_pct":           round(adjustment_pct * 100, 4),
    }


def apply_to_forecast_csv(forecast_csv_path: str | None = None):
    path = Path(forecast_csv_path) if forecast_csv_path else FORECAST_CSV
    if not path.exists():
        print("[integrator] forecasts.csv not found.")
        return

    df = pd.read_csv(path)

    # Ensure string columns have object dtype so empty-string assignment doesn't
    # fail when the column was previously read as float64 (all-NaN).
    for col in ["poly_signal", "poly_confidence", "poly_alignment", "poly_top_question"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(object)

    mask = df["actual"].isna() | (df["actual"] == "")

    count = 0
    for idx in df[mask].index:
        ticker   = df.loc[idx, "ticker"]
        base_p50 = float(df.loc[idx, "p50"])
        base_p10 = float(df.loc[idx, "p10"])
        base_p90 = float(df.loc[idx, "p90"])

        adj = get_poly_adjustment(ticker, base_p50, base_p10, base_p90)

        df.loc[idx, "poly_adjusted_p50"]      = adj["adjusted_p50"]
        df.loc[idx, "poly_adjusted_p10"]      = adj["adjusted_p10"]
        df.loc[idx, "poly_adjusted_p90"]      = adj["adjusted_p90"]
        df.loc[idx, "poly_signal"]            = adj["poly_signal"]
        df.loc[idx, "poly_confidence"]        = adj["poly_confidence"]
        df.loc[idx, "poly_alignment"]         = adj["alignment"]
        df.loc[idx, "poly_adjustment_pct"]    = adj["adjustment_pct"]
        df.loc[idx, "poly_top_question"]      = adj["poly_top_question"]
        count += 1

    df.to_csv(path, index=False)
    print(f"[integrator] Applied Polymarket adjustments to {count} forecast rows.")


if __name__ == "__main__":
    apply_to_forecast_csv()
