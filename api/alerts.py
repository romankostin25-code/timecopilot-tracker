"""Signal alerts — webhook dispatch on high-conviction signals and accuracy drops."""

import os
import json
import requests
from datetime import datetime

WEBHOOK_URL        = os.getenv("ALERT_WEBHOOK_URL", "")
MIN_CONVICTION     = os.getenv("ALERT_MIN_CONVICTION", "HIGH")
ACCURACY_THRESHOLD = float(os.getenv("ALERT_ACCURACY_THRESHOLD", 0.55))


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

    try:
        signals = json.loads(open("data/signals.json").read())
    except Exception:
        signals = {}

    try:
        scorecard = json.loads(open("data/scorecard.json").read())
    except Exception:
        scorecard = {}

    # High-conviction signal alert
    for ticker, sig in signals.items():
        analysis = sig.get("analysis", {})
        direction = sig.get("direction") or (
            analysis.get("scenarios", {}).get("base", {}).get("narrative", "")[:10]
        )
        poly_alignment = analysis.get("poly_alignment", "UNKNOWN")
        # Check conviction from analysis drivers
        drivers = analysis.get("key_drivers", [])
        high_weight_drivers = [d for d in drivers if d.get("weight") == "HIGH"]
        if len(high_weight_drivers) >= 2 and direction and "BULLISH" in str(direction).upper():
            msg = (f"HIGH CONVICTION SIGNAL: {ticker}\n"
                   f"Poly alignment: {poly_alignment}\n"
                   f"Key driver: {high_weight_drivers[0].get('driver', '')}")
            _send_webhook(msg, ticker=ticker, signal=direction, conviction="HIGH")
            fired.append({"type": "high_conviction", "ticker": ticker})

    # Accuracy drop alert — use 14d metric (more stable than 7d) and require
    # enough graded forecasts to avoid noise from sparse/freshly-regraded data
    h5 = scorecard.get("by_horizon", {}).get("5", {})
    dir_acc   = h5.get("directional_accuracy_14d")
    n_graded  = h5.get("forecasts_graded", 0)
    MIN_GRADED = 30
    if dir_acc is not None and n_graded >= MIN_GRADED and dir_acc < ACCURACY_THRESHOLD:
        msg = (f"ACCURACY ALERT: 14d directional accuracy = {dir_acc:.1%} "
               f"(threshold: {ACCURACY_THRESHOLD:.0%}, n={n_graded})")
        _send_webhook(msg, signal="ACCURACY_DROP", conviction="SYSTEM")
        fired.append({"type": "accuracy_drop", "value": dir_acc})

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
