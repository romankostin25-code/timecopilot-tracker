"""
Spike Detector — detects sudden probability moves in Polymarket markets.
Compares the latest snapshot to a snapshot from LOOKBACK_HOURS ago.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "poly_snapshots.csv"
SPIKES_PATH   = Path(__file__).parent.parent / "data" / "poly_spikes.csv"

MIN_PROB_CHANGE      = 0.08   # 8% absolute move minimum
MIN_VOLUME_USD       = 25_000
MIN_VELOCITY_PER_HR  = 0.015  # 1.5% per hour
LOOKBACK_HOURS       = 24


def load_snapshots() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SNAPSHOT_PATH)
    df = df[df["market_id"] != "__heartbeat__"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def detect_spikes(lookback_hours: int = LOOKBACK_HOURS) -> list[dict]:
    df = load_snapshots()
    if df.empty:
        print("[spike_detector] No snapshot data yet — run poller first.")
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)

    latest = (
        df.sort_values("timestamp")
        .groupby("market_id")
        .last()
        .reset_index()
    )

    historical = (
        df[df["timestamp"] <= cutoff]
        .sort_values("timestamp")
        .groupby("market_id")
        .last()
        .reset_index()
        .rename(columns={
            "prob_yes":   "prob_yes_historical",
            "timestamp":  "timestamp_historical",
            "volume_usd": "volume_usd_historical",
        })
    )

    if historical.empty:
        print(f"[spike_detector] Need {lookback_hours}h of history — collecting data.")
        return []

    merged = latest.merge(
        historical[["market_id", "prob_yes_historical",
                    "timestamp_historical", "volume_usd_historical"]],
        on="market_id",
        how="inner",
    )

    spikes = []
    for _, row in merged.iterrows():
        prob_change = row["prob_yes"] - row["prob_yes_historical"]
        abs_change = abs(prob_change)

        if abs_change < MIN_PROB_CHANGE:
            continue
        if row["volume_usd"] < MIN_VOLUME_USD:
            continue

        elapsed_hours = max(
            (row["timestamp"] - row["timestamp_historical"]).total_seconds() / 3600,
            0.25,
        )
        velocity = abs_change / elapsed_hours

        if velocity < MIN_VELOCITY_PER_HR:
            continue

        vol_change = row["volume_usd"] - row["volume_usd_historical"]
        vol_accel = vol_change / max(row["volume_usd_historical"], 1)

        conviction_score = (
            (abs_change * 0.5) +
            (min(velocity / 0.1, 1.0) * 0.3) +
            (min(vol_accel, 1.0) * 0.2)
        )

        conviction_label = (
            "HIGH" if conviction_score > 0.6 else
            "MEDIUM" if conviction_score > 0.35 else
            "LOW"
        )

        spikes.append({
            "detected_at":        now.isoformat(),
            "market_id":          row["market_id"],
            "event_type":         row["event_type"],
            "question":           row["question"],
            "prob_yes_now":       round(float(row["prob_yes"]), 4),
            "prob_yes_before":    round(float(row["prob_yes_historical"]), 4),
            "prob_change":        round(float(prob_change), 4),
            "abs_change":         round(float(abs_change), 4),
            "velocity_per_hour":  round(float(velocity), 4),
            "direction":          "BULLISH_SHIFT" if prob_change > 0 else "BEARISH_SHIFT",
            "volume_usd":         float(row["volume_usd"]),
            "volume_acceleration": round(float(vol_accel), 4),
            "conviction_score":   round(float(conviction_score), 4),
            "conviction_label":   conviction_label,
            "lookback_hours":     round(float(elapsed_hours), 2),
            "end_date":           row.get("end_date", ""),
        })
        print(
            f"[spike_detector] SPIKE: {row['event_type']} | "
            f"{prob_change:+.1%} in {elapsed_hours:.1f}h | "
            f"{conviction_label} | Vol: ${row['volume_usd']:,.0f}"
        )

    if spikes:
        new_df = pd.DataFrame(spikes)
        if SPIKES_PATH.exists():
            existing = pd.read_csv(SPIKES_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(SPIKES_PATH, index=False)
        print(f"[spike_detector] {len(spikes)} spikes saved.")
    else:
        print(f"[spike_detector] No spikes detected (threshold: abs_change≥{MIN_PROB_CHANGE:.0%})")

    return spikes


def get_active_spikes(max_age_hours: int = 6) -> list[dict]:
    """Return spikes detected within the last max_age_hours, deduplicated by event_type."""
    if not SPIKES_PATH.exists():
        return []

    df = pd.read_csv(SPIKES_PATH)
    df["detected_at"] = pd.to_datetime(df["detected_at"], utc=True)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    active = df[df["detected_at"] >= cutoff]

    if not active.empty:
        active = (
            active.sort_values("conviction_score", ascending=False)
            .groupby("event_type")
            .first()
            .reset_index()
        )

    return active.to_dict(orient="records")


if __name__ == "__main__":
    spikes = detect_spikes()
    print(f"\n{len(spikes)} spikes detected.")
