"""
Correlation Engine — builds and queries the historical spike→price database.
Runs daily after markets close to backfill outcomes for historical spikes.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from scipy import stats

try:
    from .event_map import EVENT_MAP, get_direction_multiplier
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from event_map import EVENT_MAP, get_direction_multiplier

SPIKES_PATH       = Path(__file__).parent.parent / "data" / "poly_spikes.csv"
CORRELATIONS_PATH = Path(__file__).parent.parent / "data" / "poly_correlations.csv"

HORIZONS = [1, 3, 5, 10]


def _price_return(ticker: str, spike_date: date, horizon_days: int) -> float | None:
    try:
        start = spike_date - timedelta(days=3)
        end   = spike_date + timedelta(days=horizon_days + 7)
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        prices = raw["Close"].dropna()
        price_dates = [d.date() for d in prices.index]

        base_price = None
        for offset in range(5):
            check = spike_date + timedelta(days=offset)
            if check in price_dates:
                base_price = float(prices.iloc[price_dates.index(check)])
                break
        if base_price is None:
            return None

        future_price = None
        target_date = spike_date + timedelta(days=horizon_days)
        for offset in range(5):
            check = target_date + timedelta(days=offset)
            if check in price_dates:
                future_price = float(prices.iloc[price_dates.index(check)])
                break
        if future_price is None:
            return None

        return (future_price - base_price) / base_price * 100
    except Exception:
        return None


def build_all_correlations():
    if not SPIKES_PATH.exists():
        print("[correlation] No spikes file yet.")
        return

    spikes_df = pd.read_csv(SPIKES_PATH)
    spikes_df["detected_at"] = pd.to_datetime(spikes_df["detected_at"], utc=True)

    min_horizon = max(HORIZONS)
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=min_horizon)).date()
    processable = spikes_df[spikes_df["detected_at"].dt.date <= cutoff_date]

    if processable.empty:
        print("[correlation] No spikes old enough to have outcome data yet.")
        return

    existing = pd.DataFrame()
    existing_keys = set()
    if CORRELATIONS_PATH.exists():
        existing = pd.read_csv(CORRELATIONS_PATH)
        existing_keys = set(
            zip(existing["market_id"], existing["ticker"], existing["horizon_days"])
        )

    new_records = []
    for _, spike in processable.iterrows():
        spike_dict = spike.to_dict()
        event_type = spike["event_type"]
        if event_type not in EVENT_MAP:
            continue

        spike_date = pd.to_datetime(spike["detected_at"]).date()
        affected_assets = EVENT_MAP[event_type]["assets"]

        for ticker, direction_mult in affected_assets.items():
            if direction_mult == 0:
                continue
            for horizon in HORIZONS:
                key = (spike["market_id"], ticker, horizon)
                if key in existing_keys:
                    continue

                price_return = _price_return(ticker, spike_date, horizon)
                if price_return is None:
                    continue

                prob_change = float(spike["prob_change"])
                spike_direction = +1 if prob_change > 0 else -1
                net_direction = spike_direction * direction_mult

                expected_up = net_direction > 0
                actual_up   = price_return > 0
                direction_correct = expected_up == actual_up

                new_records.append({
                    "spike_detected_at":   spike["detected_at"],
                    "spike_date":          spike_date,
                    "event_type":          event_type,
                    "market_id":           spike["market_id"],
                    "question":            spike["question"],
                    "prob_change":         prob_change,
                    "conviction_label":    spike["conviction_label"],
                    "conviction_score":    spike["conviction_score"],
                    "ticker":              ticker,
                    "horizon_days":        horizon,
                    "direction_multiplier": direction_mult,
                    "expected_direction":  "UP" if expected_up else "DOWN",
                    "price_return_pct":    round(price_return, 4),
                    "actual_direction":    "UP" if actual_up else "DOWN",
                    "direction_correct":   direction_correct,
                    "abs_return":          abs(price_return),
                })
                existing_keys.add(key)

    if new_records:
        new_df = pd.DataFrame(new_records)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined.to_csv(CORRELATIONS_PATH, index=False)
        print(f"[correlation] Added {len(new_records)} new records.")
    else:
        print("[correlation] No new records to add.")


def query_correlations(event_type: str, ticker: str, horizon_days: int, min_samples: int = 5) -> dict | None:
    if not CORRELATIONS_PATH.exists():
        return None

    df = pd.read_csv(CORRELATIONS_PATH)
    subset = df[
        (df["event_type"] == event_type) &
        (df["ticker"] == ticker) &
        (df["horizon_days"] == horizon_days)
    ]

    if len(subset) < min_samples:
        return None

    returns = subset["price_return_pct"].values
    direction_accuracy = subset["direction_correct"].mean()

    t_stat, p_value = stats.ttest_1samp(returns, 0)
    significant = p_value < 0.1

    hc = subset[subset["conviction_label"] == "HIGH"]
    hc_dir_accuracy = hc["direction_correct"].mean() if len(hc) >= 3 else None

    return {
        "event_type":            event_type,
        "ticker":                ticker,
        "horizon_days":          horizon_days,
        "sample_count":          len(subset),
        "direction_accuracy":    round(float(direction_accuracy), 4),
        "hc_direction_accuracy": round(float(hc_dir_accuracy), 4) if hc_dir_accuracy is not None else None,
        "avg_return_pct":        round(float(returns.mean()), 4),
        "median_return_pct":     round(float(np.median(returns)), 4),
        "std_return_pct":        round(float(returns.std()), 4),
        "t_stat":                round(float(t_stat), 4),
        "p_value":               round(float(p_value), 4),
        "statistically_significant": significant,
        "confidence_label": (
            "HIGH"   if significant and len(subset) >= 20 else
            "MEDIUM" if significant and len(subset) >= 10 else
            "LOW"
        ),
    }


def get_all_correlations_summary() -> list[dict]:
    """Return all correlation stats for the dashboard table."""
    if not CORRELATIONS_PATH.exists():
        return []
    df = pd.read_csv(CORRELATIONS_PATH)
    if df.empty:
        return []

    rows = []
    for (event_type, ticker, horizon), grp in df.groupby(["event_type", "ticker", "horizon_days"]):
        if len(grp) < 3:
            continue
        returns = grp["price_return_pct"].values
        dir_acc = grp["direction_correct"].mean()
        rows.append({
            "event_type":         event_type,
            "ticker":             ticker,
            "horizon_days":       int(horizon),
            "sample_count":       len(grp),
            "direction_accuracy": round(float(dir_acc), 4),
            "avg_return_pct":     round(float(returns.mean()), 4),
        })

    rows.sort(key=lambda x: -x["direction_accuracy"])
    return rows


if __name__ == "__main__":
    build_all_correlations()
