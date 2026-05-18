"""ARM 3 — Event-driven band widener: adjusts P10/P90 for uncertainty."""

import json
from pathlib import Path

REGIMES_PATH = Path("data/poly_regimes.json")


def get_band_adjustment(ticker, base_p10, base_p50, base_p90):
    from intelligence.calendar_engine import get_band_widen_factor
    calendar_factor = get_band_widen_factor(ticker)
    poly_factor = 1.0
    reasoning = []

    if REGIMES_PATH.exists():
        regime_data = json.loads(REGIMES_PATH.read_text())
        active = regime_data.get("active_regimes", {})

        high_prob = [(k, v) for k, v in active.items() if v.get("probability", 0) > 0.4]
        if len(high_prob) >= 2:
            poly_factor += 0.10
            reasoning.append(f"{len(high_prob)} active regimes → uncertainty elevated")

        recession_prob = regime_data.get("regime_probabilities", {}).get("REGIME_RECESSION", {}).get("probability", 0)
        if recession_prob and recession_prob > 0.4:
            poly_factor += 0.15
            reasoning.append(f"Recession regime {recession_prob:.0%} → tail risk elevated")

        dominant = [(k, v) for k, v in active.items() if v.get("probability", 0) > 0.70]
        if dominant:
            regime_key = dominant[0][0]
            score = abs(regime_data.get("asset_priors", {}).get(ticker, {}).get("score", 0))
            if score > 0.4:
                poly_factor -= 0.05
                reasoning.append(f"Dominant regime {regime_key} → slight band narrowing")

    total = max(calendar_factor * poly_factor, 1.0)
    half  = (base_p90 - base_p10) / 2
    return {
        "adjusted_p10":     round(base_p50 - half * total, 6),
        "adjusted_p90":     round(base_p50 + half * total, 6),
        "band_widen_factor": round(total, 4),
        "band_adj_pct":      round((total - 1.0) * 100, 2),
        "reasoning":         reasoning,
    }
