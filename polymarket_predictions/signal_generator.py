"""
Signal Generator — combines active spikes + historical correlations into a
final directional signal with confidence score for each tracked asset.
"""

import json
import math
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

try:
    from .spike_detector import get_active_spikes
    from .correlation_engine import query_correlations
    from .event_map import EVENT_MAP, get_direction_multiplier, get_events_for_asset
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from spike_detector import get_active_spikes
    from correlation_engine import query_correlations
    from event_map import EVENT_MAP, get_direction_multiplier, get_events_for_asset

SIGNALS_PATH = Path(__file__).parent.parent / "data" / "poly_signals.json"
SIGNALS_PATH.parent.mkdir(exist_ok=True)

ALL_ASSETS = [
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X",
    "DX-Y.NYB", "^TNX", "^GSPC",
]


def _compute_asset_signal(ticker: str, active_spikes: list[dict], primary_horizon: int = 5) -> dict:
    relevant_events = get_events_for_asset(ticker)
    contributing = []

    for spike in active_spikes:
        event_type = spike.get("event_type", "")
        if event_type not in relevant_events:
            continue

        direction_mult = get_direction_multiplier(event_type, ticker)
        if direction_mult == 0:
            continue

        prob_change = float(spike["prob_change"])
        spike_sign  = +1 if prob_change > 0 else -1
        net_direction = spike_sign * direction_mult

        # Log-normalize volume to avoid massive markets dominating
        log_vol = math.log(max(float(spike.get("volume_usd", 1)), 1)) / 20
        spike_strength = abs(prob_change) * float(spike.get("conviction_score", 0.5)) * log_vol

        correlation = query_correlations(event_type, ticker, primary_horizon)
        if correlation:
            stat_weight   = 1.5 if correlation["statistically_significant"] else 1.0
            dir_acc = correlation.get("hc_direction_accuracy") or correlation["direction_accuracy"]
            hist_confidence = dir_acc * stat_weight
            data_quality = "HISTORICAL" if correlation["sample_count"] >= 10 else "LIMITED_HISTORY"
            hist_avg_return = correlation["avg_return_pct"]
        else:
            hist_confidence  = 0.5
            data_quality     = "NO_HISTORY"
            hist_avg_return  = None

        contributing.append({
            "event_type":               event_type,
            "question":                 spike.get("question", ""),
            "spike_direction":          "BULLISH" if net_direction > 0 else "BEARISH",
            "net_direction":            net_direction,
            "spike_strength":           spike_strength,
            "historical_confidence":    hist_confidence,
            "historical_avg_return_pct": hist_avg_return,
            "data_quality":             data_quality,
            "conviction_label":         spike.get("conviction_label", "LOW"),
            "prob_change":              prob_change,
            "volume_usd":               float(spike.get("volume_usd", 0)),
        })

    if not contributing:
        return {
            "ticker":            ticker,
            "signal":            "NO_SIGNAL",
            "direction":         "NEUTRAL",
            "composite_score":   0,
            "confidence":        0,
            "confidence_label":  "NO_DATA",
            "contributing_events": 0,
            "top_contributing_event": None,
            "top_question":      None,
            "alignment_with_model": "UNKNOWN",
            "generated_at":      datetime.now(timezone.utc).isoformat(),
        }

    total_weight = sum(s["spike_strength"] for s in contributing) or 1.0
    weighted_dir = sum(s["net_direction"] * s["spike_strength"] for s in contributing) / total_weight
    avg_hist_conf = sum(s["historical_confidence"] * s["spike_strength"] for s in contributing) / total_weight

    composite_score = weighted_dir * avg_hist_conf

    if composite_score > 0.1:
        direction = "BULLISH"
    elif composite_score < -0.1:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    abs_score = abs(composite_score)
    if abs_score > 0.5 and avg_hist_conf > 0.65:
        confidence_label = "HIGH"
    elif abs_score > 0.25 and avg_hist_conf > 0.55:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    contributing.sort(key=lambda x: -x["spike_strength"])

    return {
        "ticker":                 ticker,
        "signal":                 f"POLY_{direction}",
        "direction":              direction,
        "composite_score":        round(float(composite_score), 4),
        "confidence":             round(float(avg_hist_conf), 4),
        "confidence_label":       confidence_label,
        "contributing_events":    len(contributing),
        "top_contributing_event": contributing[0]["event_type"],
        "top_question":           contributing[0]["question"],
        "signals_detail":         contributing,
        "alignment_with_model":   "PENDING",
        "generated_at":           datetime.now(timezone.utc).isoformat(),
    }


def generate_all_signals() -> dict:
    active_spikes = get_active_spikes(max_age_hours=6)
    print(f"[signal_generator] Active spikes: {len(active_spikes)}")

    all_signals = {}
    for ticker in ALL_ASSETS:
        sig = _compute_asset_signal(ticker, active_spikes)
        all_signals[ticker] = sig
        if sig["direction"] != "NEUTRAL":
            print(
                f"[signal_generator] {ticker}: {sig['direction']} | "
                f"score={sig['composite_score']:+.3f} | {sig['confidence_label']} | "
                f"events={sig['contributing_events']}"
            )

    # Serialize (drop signals_detail for brevity in the JSON)
    export = {}
    for ticker, sig in all_signals.items():
        export[ticker] = {k: v for k, v in sig.items() if k != "signals_detail"}

    with open(SIGNALS_PATH, "w") as f:
        json.dump(export, f, indent=2, default=str)

    print(f"[signal_generator] poly_signals.json saved ({len(all_signals)} assets).")
    return all_signals


def load_signals() -> dict:
    if not SIGNALS_PATH.exists():
        return {}
    with open(SIGNALS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    generate_all_signals()
