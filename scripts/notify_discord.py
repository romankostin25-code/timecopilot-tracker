"""Post a run-completion summary to Discord via webhook.

Usage:
    python scripts/notify_discord.py --event evening
    python scripts/notify_discord.py --event morning
    python scripts/notify_discord.py --event retrain
    python scripts/notify_discord.py --event backfill
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime, timezone
from pathlib import Path

WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
SCORECARD_PATH = Path("data/scorecard.json")
TFT_LOG_PATH   = Path("data/tft_training_log.json")
DIR_MODEL_PATH = Path("data/direction_model.pkl")

EVENT_LABELS = {
    "evening":  ("📊 Evening Forecast",  "Evening run complete — new 5d/30d/90d predictions generated"),
    "morning":  ("🌅 Morning Update",    "Morning grades + signals refreshed"),
    "retrain":  ("🧠 ML Retrain",        "TFT + LR models retrained on fresh data"),
    "backfill": ("🔄 Backfill",          "Historical forecasts regenerated with updated signal rules"),
}

THRESHOLD = float(os.getenv("ALERT_ACCURACY_THRESHOLD", 0.55))


def _pct(v):
    return f"{v:.1%}" if v is not None else "—"


def _load_scorecard():
    try:
        return json.loads(SCORECARD_PATH.read_text())
    except Exception:
        return {}


def _build_accuracy_fields(sc):
    g    = sc.get("global", {})
    h5   = sc.get("by_horizon", {}).get("5", {})
    h30  = sc.get("by_horizon", {}).get("30", {})

    dir7_global  = g.get("directional_accuracy_7d")
    dir14_global = g.get("directional_accuracy_14d")
    dir7_5d      = h5.get("directional_accuracy_7d")
    dir14_5d     = h5.get("directional_accuracy_14d")
    dir7_30d     = h30.get("directional_accuracy_7d")
    dir14_30d    = h30.get("directional_accuracy_14d")
    cal14_5d     = h5.get("calibration_14d")
    n5           = h5.get("forecasts_graded", 0)
    n30          = h30.get("forecasts_graded", 0)
    trend        = g.get("trend", "—")

    on_target = dir7_global is not None and dir7_global >= THRESHOLD
    status    = "✅ On target" if on_target else "⚠️ Below target"
    color     = 0x00d084 if on_target else 0xff9900

    fields = [
        {"name": "Global 7d",      "value": _pct(dir7_global),  "inline": True},
        {"name": "Global 14d",     "value": _pct(dir14_global), "inline": True},
        {"name": "Trend",          "value": trend,              "inline": True},
        {"name": "5D dir 7d",      "value": _pct(dir7_5d),      "inline": True},
        {"name": "5D dir 14d",     "value": _pct(dir14_5d),     "inline": True},
        {"name": "5D band 14d",    "value": _pct(cal14_5d),     "inline": True},
        {"name": "30D dir 7d",     "value": _pct(dir7_30d),     "inline": True},
        {"name": "30D dir 14d",    "value": _pct(dir14_30d),    "inline": True},
        {"name": "Graded (5d/30d)","value": f"{n5} / {n30}",   "inline": True},
    ]
    return fields, status, color


def _top_and_bottom(sc, n=3):
    assets = sc.get("by_asset", {})
    scored = [
        (t, v["directional_accuracy_14d"])
        for t, v in assets.items()
        if v.get("directional_accuracy_14d") is not None
    ]
    scored.sort(key=lambda x: x[1])
    bottom = ", ".join(f"{t} {_pct(a)}" for t, a in scored[:n])
    top    = ", ".join(f"{t} {_pct(a)}" for t, a in scored[-n:][::-1])
    return top, bottom


def _retrain_summary():
    lines = []
    try:
        log = json.loads(TFT_LOG_PATH.read_text())
        for h, info in sorted(log.items()):
            key = "finetuned_at" if "finetuned_at" in info else "trained_at"
            ts  = info.get(key, "?")[:19].replace("T", " ")
            lines.append(f"h{h}: {key.replace('_at','')} {ts}")
    except Exception:
        lines.append("TFT log unavailable")
    try:
        import joblib
        payload = joblib.load(DIR_MODEL_PATH)
        summary = payload.get("summary", {})
        for h, s in sorted(summary.items()):
            if s.get("trained"):
                lines.append(f"LR h{h}: n={s['n']} acc={s.get('train_acc','?')}")
    except Exception:
        lines.append("LR model log unavailable")
    return "\n".join(lines)


def notify(event: str):
    if not WEBHOOK_URL:
        print("[notify] No ALERT_WEBHOOK_URL — skipping.")
        return

    sc              = _load_scorecard()
    fields, status, color = _build_accuracy_fields(sc)
    top, bottom     = _top_and_bottom(sc)
    title, desc     = EVENT_LABELS.get(event, ("📢 Run complete", ""))

    if event == "retrain":
        extra_text = _retrain_summary()
        fields.append({"name": "Models", "value": extra_text or "—", "inline": False})

    if top:
        fields.append({"name": f"🏆 Top 3 (14d)",    "value": top,    "inline": True})
        fields.append({"name": f"🔻 Bottom 3 (14d)", "value": bottom, "inline": True})

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "embeds": [{
            "title":       f"{title} — {status}",
            "description": desc,
            "fields":      fields,
            "color":       color,
            "timestamp":   now,
            "footer":      {"text": f"Trading Co-Pilot · {event}"},
        }]
    }

    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        print(f"[notify] Discord {event}: HTTP {r.status_code}")
    except Exception as e:
        print(f"[notify] Discord error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True,
                        choices=list(EVENT_LABELS.keys()),
                        help="Which run type to report")
    args = parser.parse_args()
    notify(args.event)
