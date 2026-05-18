"""Economic calendar — tracks high-impact events and computes band widening factors."""

import json
from datetime import datetime, date, timedelta
from pathlib import Path

CALENDAR_PATH = Path("data/economic_calendar.json")

EVENT_SENSITIVITY = {
    "NFP":            {"assets": ["SPY", "QQQ", "TLT", "DX-Y.NYB", "^TNX"], "band_widen_pct": 0.25},
    "CPI":            {"assets": ["SPY", "TLT", "GC=F", "^TNX", "DX-Y.NYB"], "band_widen_pct": 0.30},
    "FOMC":           {"assets": ["SPY", "QQQ", "TLT", "GC=F", "BTC-USD", "^TNX"], "band_widen_pct": 0.40},
    "GDP":            {"assets": ["SPY", "QQQ", "DX-Y.NYB", "CL=F"], "band_widen_pct": 0.20},
    "PCE":            {"assets": ["SPY", "TLT", "^TNX", "GC=F"], "band_widen_pct": 0.20},
    "JOBLESS_CLAIMS": {"assets": ["SPY", "TLT"], "band_widen_pct": 0.10},
    "ISM_MANU":       {"assets": ["SPY", "HG=F", "CL=F"], "band_widen_pct": 0.15},
    "RETAIL_SALES":   {"assets": ["SPY", "XLK", "XLF"], "band_widen_pct": 0.15},
    "EIA_OIL":        {"assets": ["CL=F", "NG=F", "XLE"], "band_widen_pct": 0.15},
    "EIA_GAS":        {"assets": ["NG=F", "XLE"], "band_widen_pct": 0.20},
}

# Update monthly
MANUAL_CALENDAR = [
    {"event": "FOMC",    "date": "2025-06-11", "time_utc": "18:00"},
    {"event": "CPI",     "date": "2025-06-11", "time_utc": "12:30"},
    {"event": "NFP",     "date": "2025-07-03", "time_utc": "12:30"},
    {"event": "EIA_OIL", "date": "2025-06-04", "time_utc": "14:30"},
    {"event": "EIA_GAS", "date": "2025-06-05", "time_utc": "14:30"},
    {"event": "PCE",     "date": "2025-05-30", "time_utc": "12:30"},
    {"event": "GDP",     "date": "2025-05-29", "time_utc": "12:30"},
    {"event": "FOMC",    "date": "2025-07-30", "time_utc": "18:00"},
    {"event": "CPI",     "date": "2025-07-15", "time_utc": "12:30"},
    {"event": "NFP",     "date": "2025-08-01", "time_utc": "12:30"},
]


def get_upcoming_events(days_ahead=10):
    today = date.today()
    result = []
    for ev in MANUAL_CALENDAR:
        ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        days_until = (ev_date - today).days
        if 0 <= days_until <= days_ahead:
            ev_copy = ev.copy()
            ev_copy["days_until"] = days_until
            ev_copy["sensitivity"] = EVENT_SENSITIVITY.get(ev["event"], {})
            result.append(ev_copy)
    return sorted(result, key=lambda x: x["days_until"])


def get_band_widen_factor(ticker, days_ahead=5):
    max_widen = 0.0
    for ev in get_upcoming_events(days_ahead):
        sens = EVENT_SENSITIVITY.get(ev["event"], {})
        if ticker in sens.get("assets", []):
            prox = 1.0 - (ev["days_until"] / (days_ahead + 1))
            widen = sens.get("band_widen_pct", 0.10) * prox
            max_widen = max(max_widen, widen)
    return round(1.0 + max_widen, 4)


def save_calendar():
    CALENDAR_PATH.parent.mkdir(exist_ok=True)
    CALENDAR_PATH.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "events":       MANUAL_CALENDAR,
        "upcoming_7d":  get_upcoming_events(7),
    }, indent=2))
    print(f"[calendar] Saved {len(MANUAL_CALENDAR)} events.")


if __name__ == "__main__":
    save_calendar()
    for ev in get_upcoming_events(10):
        print(f"  {ev['date']} | {ev['event']} | {ev['days_until']}d away")
