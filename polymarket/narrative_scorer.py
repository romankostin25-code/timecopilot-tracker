"""ARM 2 — Narrative momentum scorer: cluster markets by theme, measure probability velocity."""

import json
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOTS_PATH  = Path("data/poly_snapshots.csv")
NARRATIVES_PATH = Path("data/poly_narratives.json")
POLY_API = "https://gamma-api.polymarket.com/markets"


def compute_momentum(snaps_df, market_id, hours=24):
    sub = snaps_df[snaps_df["market_id"] == market_id].sort_values("timestamp")
    if len(sub) < 2:
        return None, None, None
    now_row = sub.iloc[-1]
    now_ts  = pd.to_datetime(now_row["timestamp"], utc=True)
    cutoff  = now_ts - pd.Timedelta(hours=hours)
    old = sub[pd.to_datetime(sub["timestamp"], utc=True) <= cutoff]
    current = float(now_row["prob_yes"])
    if old.empty:
        return current, None, None
    past    = float(old.iloc[-1]["prob_yes"])
    change  = current - past
    elapsed = (now_ts - pd.to_datetime(old.iloc[-1]["timestamp"], utc=True)).total_seconds() / 3600
    return current, round(change, 4), round(change / max(elapsed, 1), 6)


def score_narratives():
    from polymarket.event_taxonomy import MACRO_REGIMES
    from polymarket.regime_engine import classify_market_to_regime

    snaps = pd.read_csv(SNAPSHOTS_PATH) if SNAPSHOTS_PATH.exists() else pd.DataFrame()

    try:
        markets = requests.get(
            POLY_API, params={"active": "true", "limit": 200, "order": "volume"}, timeout=15
        ).json()
    except Exception as e:
        print(f"[narrative_scorer] API error: {e}")
        return []

    narratives: dict = {}
    for m in markets:
        try:
            volume   = float(m.get("volume", 0) or 0)
            if volume < 25000:
                continue
            market_id = m.get("id", "")
            question  = m.get("question", "")
            prob_yes  = float(m.get("outcomePrices", [0.5])[0])
            matches   = classify_market_to_regime(question)
            if not matches:
                continue
            top_regime = matches[0][0]

            current, change_24h, velocity = (
                compute_momentum(snaps, market_id) if not snaps.empty else (prob_yes, None, None)
            )
            momentum = abs(change_24h or 0) * min(volume / 500_000, 2.0)

            if top_regime not in narratives:
                narratives[top_regime] = {
                    "regime": top_regime,
                    "description": MACRO_REGIMES[top_regime]["description"],
                    "markets": [], "total_momentum": 0, "total_volume": 0,
                }
            narratives[top_regime]["markets"].append({
                "market_id": market_id, "question": question,
                "prob_yes": prob_yes, "volume": volume,
                "change_24h": change_24h, "velocity": velocity,
                "momentum_score": momentum,
            })
            narratives[top_regime]["total_momentum"] += momentum
            narratives[top_regime]["total_volume"]   += volume
        except Exception:
            continue

    for key, data in narratives.items():
        if not data["markets"]:
            continue
        probs = [m["prob_yes"] for m in data["markets"]]
        data["avg_probability"] = round(sum(probs) / len(probs), 4)
        changes = [m["change_24h"] for m in data["markets"] if m["change_24h"] is not None]
        if changes:
            avg_chg = sum(changes) / len(changes)
            bias = MACRO_REGIMES.get(key, {}).get("regime_effect", {}).get("equity_bias", 0)
            net  = avg_chg * bias
            data["direction"]     = "BULLISH" if net > 0.01 else "BEARISH" if net < -0.01 else "NEUTRAL"
            data["avg_change_24h"] = round(avg_chg, 4)
        else:
            data["direction"] = "NEUTRAL"
        data["markets"] = sorted(data["markets"], key=lambda x: x["momentum_score"], reverse=True)[:5]

    result = sorted(narratives.values(), key=lambda x: x["total_momentum"], reverse=True)
    NARRATIVES_PATH.parent.mkdir(exist_ok=True)
    NARRATIVES_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "narratives":   result,
    }, indent=2, default=str))
    return result


if __name__ == "__main__":
    for n in score_narratives()[:5]:
        print(f"{n['regime']}: momentum={n['total_momentum']:.2f} dir={n['direction']}")
