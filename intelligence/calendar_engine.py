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
    "USDA_WASDE":     {"assets": ["ZC=F", "ZW=F", "SI=F"], "band_widen_pct": 0.25},
    "USDA_CROP":      {"assets": ["ZC=F", "ZW=F"], "band_widen_pct": 0.15},
}

MANUAL_CALENDAR = [
    # ── FOMC 2026 ────────────────────────────────────────────────────────────
    {"event": "FOMC",         "date": "2026-06-17", "time_utc": "18:00"},
    {"event": "FOMC",         "date": "2026-07-29", "time_utc": "18:00"},
    {"event": "FOMC",         "date": "2026-09-16", "time_utc": "18:00"},
    {"event": "FOMC",         "date": "2026-10-28", "time_utc": "18:00"},
    {"event": "FOMC",         "date": "2026-12-09", "time_utc": "18:00"},
    # ── CPI 2026 ─────────────────────────────────────────────────────────────
    {"event": "CPI",          "date": "2026-06-10", "time_utc": "12:30"},
    {"event": "CPI",          "date": "2026-07-14", "time_utc": "12:30"},
    {"event": "CPI",          "date": "2026-08-12", "time_utc": "12:30"},
    {"event": "CPI",          "date": "2026-09-11", "time_utc": "12:30"},
    {"event": "CPI",          "date": "2026-10-13", "time_utc": "12:30"},
    {"event": "CPI",          "date": "2026-11-12", "time_utc": "12:30"},
    {"event": "CPI",          "date": "2026-12-10", "time_utc": "12:30"},
    # ── NFP 2026 ─────────────────────────────────────────────────────────────
    {"event": "NFP",          "date": "2026-06-05", "time_utc": "12:30"},
    {"event": "NFP",          "date": "2026-07-02", "time_utc": "12:30"},
    {"event": "NFP",          "date": "2026-08-07", "time_utc": "12:30"},
    {"event": "NFP",          "date": "2026-09-04", "time_utc": "12:30"},
    {"event": "NFP",          "date": "2026-10-02", "time_utc": "12:30"},
    {"event": "NFP",          "date": "2026-11-06", "time_utc": "12:30"},
    {"event": "NFP",          "date": "2026-12-04", "time_utc": "12:30"},
    # ── PCE / GDP 2026 ───────────────────────────────────────────────────────
    {"event": "PCE",          "date": "2026-05-29", "time_utc": "12:30"},
    {"event": "GDP",          "date": "2026-05-28", "time_utc": "12:30"},
    {"event": "PCE",          "date": "2026-06-26", "time_utc": "12:30"},
    {"event": "GDP",          "date": "2026-06-25", "time_utc": "12:30"},
    {"event": "PCE",          "date": "2026-07-31", "time_utc": "12:30"},
    {"event": "GDP",          "date": "2026-07-30", "time_utc": "12:30"},
    # ── EIA weekly (Wednesday for oil, Thursday for gas) ─────────────────────
    {"event": "EIA_OIL",      "date": "2026-05-27", "time_utc": "14:30"},
    {"event": "EIA_GAS",      "date": "2026-05-28", "time_utc": "14:30"},
    {"event": "EIA_OIL",      "date": "2026-06-03", "time_utc": "14:30"},
    {"event": "EIA_GAS",      "date": "2026-06-04", "time_utc": "14:30"},
    {"event": "EIA_OIL",      "date": "2026-06-10", "time_utc": "14:30"},
    {"event": "EIA_GAS",      "date": "2026-06-11", "time_utc": "14:30"},
    # ── USDA WASDE (monthly crop supply/demand) ──────────────────────────────
    {"event": "USDA_WASDE",   "date": "2026-06-11", "time_utc": "16:00"},
    {"event": "USDA_WASDE",   "date": "2026-07-11", "time_utc": "16:00"},
    {"event": "USDA_WASDE",   "date": "2026-08-12", "time_utc": "16:00"},
    {"event": "USDA_WASDE",   "date": "2026-09-11", "time_utc": "16:00"},
    {"event": "USDA_WASDE",   "date": "2026-10-09", "time_utc": "16:00"},
    # ── USDA Crop Progress (Monday, growing season May–Nov) ──────────────────
    {"event": "USDA_CROP",    "date": "2026-05-26", "time_utc": "20:00"},
    {"event": "USDA_CROP",    "date": "2026-06-02", "time_utc": "20:00"},
    {"event": "USDA_CROP",    "date": "2026-06-09", "time_utc": "20:00"},
    {"event": "USDA_CROP",    "date": "2026-06-16", "time_utc": "20:00"},
    {"event": "USDA_CROP",    "date": "2026-06-23", "time_utc": "20:00"},
    {"event": "USDA_CROP",    "date": "2026-06-30", "time_utc": "20:00"},
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
