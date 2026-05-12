"""
3-arm signal combiner:
  ARM 1 (35%): TimeCopilot probabilistic forecast
  ARM 2 (45%): DirectionalCLF ML ensemble
  ARM 3 (20%): Polymarket crowd signal
"""

from typing import Optional
from conviction_thresholds import is_directional, passes_conviction_gate

# Arm weights (sum to 1.0)
_W_TC   = 0.35
_W_CLF  = 0.45
_W_POLY = 0.20


def combine_signals(
    ticker: str,
    # TimeCopilot arm
    p10: float, p50: float, p90: float, last_price: float, atr_14_pct: float,
    # DirectionalCLF arm (None = no model yet)
    clf_prob_up: Optional[float],
    # Polymarket arm (None = no data)
    poly_prob_bullish: Optional[float],
) -> dict:
    """
    Combine 3 arms into a direction call + conviction score.

    Returns dict with:
      direction, combined_prob_up, conviction_score, signal_strength,
      arm_tc_prob, arm_clf_prob, arm_poly_prob, arm_tc_dir
    """
    safe_last = max(abs(last_price), 1e-9)

    # ── ARM 1: TimeCopilot ──────────────────────────────────────────────────────
    tc_return = (p50 - last_price) / safe_last
    tc_dir    = is_directional(ticker, tc_return, atr_14_pct)

    # Convert direction to probability; temper by band certainty
    band_ratio    = (p90 - p10) / max(abs(p50), 1e-9)
    tc_confidence = max(0.0, 1.0 - band_ratio / 0.10)
    if tc_dir == "BULLISH":
        tc_prob_raw = 0.65
    elif tc_dir == "BEARISH":
        tc_prob_raw = 0.35
    else:
        tc_prob_raw = 0.50
    tc_prob = 0.5 + (tc_prob_raw - 0.5) * tc_confidence

    # ── ARM 2: DirectionalCLF ───────────────────────────────────────────────────
    w_clf      = _W_CLF if clf_prob_up is not None else 0.0
    clf_prob   = clf_prob_up if clf_prob_up is not None else 0.5

    # ── ARM 3: Polymarket ────────────────────────────────────────────────────────
    w_poly     = _W_POLY if poly_prob_bullish is not None else 0.0
    poly_prob  = poly_prob_bullish if poly_prob_bullish is not None else 0.5

    # ── Weighted combination ────────────────────────────────────────────────────
    total_w = _W_TC + w_clf + w_poly
    combined_prob = (
        _W_TC  * tc_prob +
        w_clf  * clf_prob +
        w_poly * poly_prob
    ) / total_w

    # ── Conviction gate ──────────────────────────────────────────────────────────
    if not passes_conviction_gate(ticker, combined_prob):
        direction = "NEUTRAL"
    elif combined_prob > 0.5:
        direction = "BULLISH"
    else:
        direction = "BEARISH"

    conviction_score = round(abs(combined_prob - 0.5) * 2.0, 4)

    return {
        "direction":        direction,
        "combined_prob_up": round(combined_prob, 4),
        "conviction_score": conviction_score,
        "signal_strength":  round(abs(tc_return) * 100, 4),
        "arm_tc_prob":      round(tc_prob, 4),
        "arm_clf_prob":     round(clf_prob, 4),
        "arm_poly_prob":    round(poly_prob, 4),
        "arm_tc_dir":       tc_dir,
    }
