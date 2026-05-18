"""Combine ARM 1 (regime) + ARM 2 (narrative) into per-asset poly signals."""

import json
from pathlib import Path

REGIMES_PATH    = Path("data/poly_regimes.json")
NARRATIVES_PATH = Path("data/poly_narratives.json")
SIGNALS_PATH    = Path("data/poly_signals.json")


def combine_signals():
    from engine.universe import ALL_TICKERS
    from polymarket.event_taxonomy import MACRO_REGIMES

    regime_data    = json.loads(REGIMES_PATH.read_text()) if REGIMES_PATH.exists() else {}
    narrative_data = json.loads(NARRATIVES_PATH.read_text()) if NARRATIVES_PATH.exists() else {}

    # Build narrative boost per ticker
    narrative_boosts = {t: 0.0 for t in ALL_TICKERS}
    for narr in narrative_data.get("narratives", []):
        regime_key = narr.get("regime")
        direction  = narr.get("direction", "NEUTRAL")
        momentum   = min(narr.get("total_momentum", 0) / 100, 1.0)
        asset_prs  = MACRO_REGIMES.get(regime_key, {}).get("asset_priors", {})
        dir_mult   = 1 if direction == "BULLISH" else -1 if direction == "BEARISH" else 0
        for ticker, prior in asset_prs.items():
            if ticker in narrative_boosts:
                narrative_boosts[ticker] += prior * dir_mult * momentum * 0.3

    signals = {}
    for ticker in ALL_TICKERS:
        regime_prior = regime_data.get("asset_priors", {}).get(ticker, {})
        arm1 = regime_prior.get("score", 0) * 0.5
        arm2 = narrative_boosts.get(ticker, 0)
        combined = arm1 + arm2
        direction = "BULLISH" if combined > 0.1 else "BEARISH" if combined < -0.1 else "NEUTRAL"
        signals[ticker] = {
            "direction":          direction,
            "signal":             f"POLY_{direction}",
            "composite_score":    round(combined, 4),
            "confidence":         "HIGH" if abs(combined) > 0.4 else "MEDIUM" if abs(combined) > 0.2 else "LOW",
            "arm1_regime_score":  round(arm1, 4),
            "arm1_top_regime":    regime_prior.get("top_regime"),
            "arm2_narrative_boost": round(arm2, 4),
            "arm1_confidence":    regime_prior.get("confidence", "NO_DATA"),
        }

    SIGNALS_PATH.parent.mkdir(exist_ok=True)
    SIGNALS_PATH.write_text(json.dumps(signals, indent=2))
    active = [t for t, s in signals.items() if s["direction"] != "NEUTRAL"]
    print(f"[signal_combiner] Active signals: {active}")
    return signals


if __name__ == "__main__":
    combine_signals()
