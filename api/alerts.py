"""Signal alerts — webhook dispatch on high-conviction signals and accuracy drops."""

import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

WEBHOOK_URL        = os.getenv("ALERT_WEBHOOK_URL", "")
MIN_CONVICTION     = os.getenv("ALERT_MIN_CONVICTION", "HIGH")
ACCURACY_THRESHOLD = float(os.getenv("ALERT_ACCURACY_THRESHOLD", 0.55))

ALERT_STATE_PATH = Path("data/alert_state.json")

# Per-ticker alert fires when 14d accuracy falls below this
TICKER_ACCURACY_FLOOR  = float(os.getenv("TICKER_ACCURACY_FLOOR", 0.35))
# Minimum graded 14d samples before a per-ticker alert is actionable
TICKER_MIN_GRADED_14D  = int(os.getenv("TICKER_MIN_GRADED", 5))
# Re-fire cooldown for the same alert key (hours)
ALERT_COOLDOWN_HOURS   = int(os.getenv("ALERT_COOLDOWN_HOURS", 20))


def _load_state() -> dict:
    try:
        return json.loads(ALERT_STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(state: dict):
    ALERT_STATE_PATH.parent.mkdir(exist_ok=True)
    ALERT_STATE_PATH.write_text(json.dumps(state, indent=2))


def _cooldown_ok(state: dict, key: str) -> bool:
    """Returns True if enough time has passed since this alert last fired."""
    last = state.get(key)
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last_dt).total_seconds() > ALERT_COOLDOWN_HOURS * 3600
    except Exception:
        return True


def _mark_fired(state: dict, key: str):
    state[key] = datetime.now(timezone.utc).isoformat()


def _send_webhook(message: str, ticker=None, signal=None, conviction=None):
    if not WEBHOOK_URL:
        return
    payload = {
        "text": message,
        "embeds": [{
            "title":       "Trading Co-Pilot Alert",
            "description": message,
            "fields": [
                {"name": "Ticker",     "value": ticker     or "N/A", "inline": True},
                {"name": "Signal",     "value": signal     or "N/A", "inline": True},
                {"name": "Conviction", "value": conviction or "N/A", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "color": 0x00d4ff,
        }],
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"[alerts] Webhook error: {e}")


def check_and_dispatch_alerts():
    fired = []
    state = _load_state()

    try:
        signals = json.loads(open("data/signals.json").read())
    except Exception:
        signals = {}

    try:
        scorecard = json.loads(open("data/scorecard.json").read())
    except Exception:
        scorecard = {}

    # ── High-conviction signal alerts ─────────────────────────────────────────
    for ticker, sig in signals.items():
        analysis = sig.get("analysis", {})
        direction = sig.get("direction") or (
            analysis.get("scenarios", {}).get("base", {}).get("narrative", "")[:10]
        )
        poly_alignment = analysis.get("poly_alignment", "UNKNOWN")
        drivers = analysis.get("key_drivers", [])
        high_weight_drivers = [d for d in drivers if d.get("weight") == "HIGH"]
        alert_key = f"high_conviction_{ticker}"
        if (len(high_weight_drivers) >= 2 and direction
                and "BULLISH" in str(direction).upper()
                and _cooldown_ok(state, alert_key)):
            msg = (f"HIGH CONVICTION SIGNAL: {ticker}\n"
                   f"Poly alignment: {poly_alignment}\n"
                   f"Key driver: {high_weight_drivers[0].get('driver', '')}")
            _send_webhook(msg, ticker=ticker, signal=direction, conviction="HIGH")
            _mark_fired(state, alert_key)
            fired.append({"type": "high_conviction", "ticker": ticker})

    # ── Aggregate accuracy drop alert ─────────────────────────────────────────
    h5 = scorecard.get("by_horizon", {}).get("5", {})
    dir_acc  = h5.get("directional_accuracy_14d")
    n_graded = h5.get("forecasts_graded", 0)
    MIN_GRADED = 30
    agg_key = "accuracy_drop_aggregate"
    if (dir_acc is not None and n_graded >= MIN_GRADED
            and dir_acc < ACCURACY_THRESHOLD
            and _cooldown_ok(state, agg_key)):
        msg = (f"ACCURACY ALERT: 14d directional accuracy = {dir_acc:.1%} "
               f"(threshold: {ACCURACY_THRESHOLD:.0%}, n={n_graded})")
        _send_webhook(msg, signal="ACCURACY_DROP", conviction="SYSTEM")
        _mark_fired(state, agg_key)
        fired.append({"type": "accuracy_drop", "value": dir_acc})

    # ── Per-ticker underperformer digest ──────────────────────────────────────
    by_asset = scorecard.get("by_asset", {})
    n_tickers = max(len(by_asset), 1)
    per_ticker_est = n_graded / n_tickers  # proxy for per-ticker graded sample depth
    underperformers = []
    cold_streaks    = []

    if per_ticker_est >= TICKER_MIN_GRADED_14D:
        for ticker, stats in by_asset.items():
            acc14  = stats.get("directional_accuracy_14d")
            acc7   = stats.get("directional_accuracy_7d")
            streak = stats.get("consecutive_hits", None)
            if acc14 is not None and acc14 < TICKER_ACCURACY_FLOOR:
                underperformers.append((ticker, acc14, acc7))
            if (streak is not None and streak == 0
                    and acc14 is not None and acc14 < 0.40):
                cold_streaks.append((ticker, acc14))

    # Send a single digest for underperformers (group to avoid spam)
    ticker_digest_key = "per_ticker_underperformer_digest"
    if underperformers and _cooldown_ok(state, ticker_digest_key):
        lines = ["PER-TICKER ACCURACY ALERT\n"]
        for tk, a14, a7 in sorted(underperformers, key=lambda x: x[1]):
            a7_str = f"  7d={a7:.0%}" if a7 is not None else ""
            lines.append(f"  {tk:<12}  14d={a14:.0%}{a7_str}")
        lines.append(f"\n(floor={TICKER_ACCURACY_FLOOR:.0%}, n≈{per_ticker_est:.0f}/ticker)")
        msg = "\n".join(lines)
        _send_webhook(msg, signal="PER_TICKER_UNDERPERFORM", conviction="SYSTEM")
        _mark_fired(state, ticker_digest_key)
        fired.append({"type": "per_ticker_underperform", "tickers": [t for t, *_ in underperformers]})
        print(f"[alerts] Per-ticker underperformers: {[t for t, *_ in underperformers]}")

    # Cold-streak digest (separate, higher urgency)
    cold_key = "cold_streak_digest"
    if cold_streaks and _cooldown_ok(state, cold_key):
        streak_list = ", ".join(f"{t}({a:.0%})" for t, a in cold_streaks)
        msg = f"COLD STREAK ALERT: {streak_list} — consecutive_hits=0 with sub-40% 14d accuracy"
        _send_webhook(msg, signal="COLD_STREAK", conviction="HIGH")
        _mark_fired(state, cold_key)
        fired.append({"type": "cold_streak", "tickers": [t for t, _ in cold_streaks]})
        print(f"[alerts] Cold streaks: {[t for t, _ in cold_streaks]}")

    _save_state(state)
    print(f"[alerts] Fired {len(fired)} alerts.")
    return fired


def handler(request):
    fired = check_and_dispatch_alerts()
    return {
        "statusCode": 200,
        "body": json.dumps({"alerts_fired": len(fired)}),
    }


if __name__ == "__main__":
    check_and_dispatch_alerts()
