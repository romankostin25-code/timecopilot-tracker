"""ARM 1 — Macro regime engine: classify markets into regimes, derive asset priors."""

import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

POLY_API    = "https://gamma-api.polymarket.com/markets"
REGIME_PATH = Path("data/poly_regimes.json")
MIN_VOLUME  = float(os.getenv("POLYMARKET_MIN_VOLUME", 25000))


def fetch_markets(limit=200):
    try:
        resp = requests.get(
            POLY_API,
            params={"active": "true", "limit": limit, "order": "volume", "ascending": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[regime_engine] API error: {e}")
        return []


def classify_market_to_regime(question, description=""):
    from polymarket.event_taxonomy import MACRO_REGIMES
    text = (question + " " + description).lower()
    matches = []
    for key, data in MACRO_REGIMES.items():
        score = sum(1 for theme in data["polymarket_themes"] if theme.lower() in text)
        if score > 0:
            matches.append((key, score))
    return sorted(matches, key=lambda x: x[1], reverse=True)


def compute_regime_probabilities(markets):
    from polymarket.event_taxonomy import MACRO_REGIMES
    scores = {k: {"prob_sum": 0, "weight_sum": 0, "markets": []} for k in MACRO_REGIMES}

    for m in markets:
        try:
            prices = m.get("outcomePrices", [])
            if len(prices) < 2:
                continue
            prob_yes = float(prices[0])
            volume   = float(m.get("volume", 0) or 0)
            if volume < MIN_VOLUME:
                continue
            question = m.get("question", "")
            for regime_key, match_score in classify_market_to_regime(question)[:2]:
                w = volume * match_score
                scores[regime_key]["prob_sum"]    += prob_yes * w
                scores[regime_key]["weight_sum"]  += w
                scores[regime_key]["markets"].append({
                    "question": question, "prob_yes": prob_yes,
                    "volume": volume, "match_score": match_score,
                })
        except Exception:
            continue

    result = {}
    for k, data in scores.items():
        if data["weight_sum"] > 0:
            result[k] = {
                "probability":   round(data["prob_sum"] / data["weight_sum"], 4),
                "market_count":  len(data["markets"]),
                "total_volume":  round(data["weight_sum"], 0),
                "top_markets":   sorted(data["markets"], key=lambda x: x["volume"], reverse=True)[:3],
            }
        else:
            result[k] = {"probability": None, "market_count": 0, "total_volume": 0, "top_markets": []}
    return result


def compute_asset_priors(regime_probabilities):
    from polymarket.event_taxonomy import MACRO_REGIMES
    from engine.universe import ALL_TICKERS

    asset_scores  = {t: 0.0 for t in ALL_TICKERS}
    asset_weights = {t: 0.0 for t in ALL_TICKERS}

    for regime_key, rdata in regime_probabilities.items():
        prob = rdata.get("probability")
        if prob is None or prob < 0.1:
            continue
        vol = rdata.get("total_volume", 0)
        weight = prob * min(vol / 1_000_000, 2.0)
        for ticker, direction in MACRO_REGIMES.get(regime_key, {}).get("asset_priors", {}).items():
            if ticker in asset_scores:
                asset_scores[ticker]  += direction * weight
                asset_weights[ticker] += abs(weight)

    result = {}
    for ticker in ALL_TICKERS:
        w = asset_weights[ticker]
        if w == 0:
            result[ticker] = {"score": 0, "direction": "NEUTRAL", "confidence": "NO_DATA", "top_regime": None}
            continue
        score = asset_scores[ticker] / w
        result[ticker] = {
            "score":      round(score, 4),
            "direction":  "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL",
            "confidence": "HIGH" if abs(score) > 0.5 else "MEDIUM" if abs(score) > 0.25 else "LOW",
            "top_regime": max(
                ((rk, MACRO_REGIMES[rk]["asset_priors"].get(ticker, 0) * (rv.get("probability") or 0))
                 for rk, rv in regime_probabilities.items()
                 if MACRO_REGIMES.get(rk, {}).get("asset_priors", {}).get(ticker)),
                key=lambda x: abs(x[1]), default=(None, 0)
            )[0],
        }
    return result


def run_regime_engine():
    markets = fetch_markets(200)
    print(f"[regime_engine] {len(markets)} markets fetched")
    regime_probs  = compute_regime_probabilities(markets)
    asset_priors  = compute_asset_priors(regime_probs)
    active = {k: v for k, v in regime_probs.items() if v.get("probability") and v["probability"] > 0.3}
    print(f"[regime_engine] Active regimes (>30%): {list(active.keys())}")

    REGIME_PATH.parent.mkdir(exist_ok=True)
    REGIME_PATH.write_text(json.dumps({
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "regime_probabilities": regime_probs,
        "asset_priors":         asset_priors,
        "active_regimes":       active,
    }, indent=2, default=str))
    return {"regime_probabilities": regime_probs, "asset_priors": asset_priors, "active_regimes": active}


if __name__ == "__main__":
    run_regime_engine()
