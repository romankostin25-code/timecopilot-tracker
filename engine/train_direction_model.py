"""Train per-horizon logistic regression on graded forecast history.

Core features (always used):
  forecast_return = (p50 - last_price) / last_price  — model's predicted move
  band_width      = (p90 - p10) / last_price          — model uncertainty

Extended features (used when present in ≥50% of graded rows):
  macro_sc        — asset-class macro composite signal [-1, 1]
  tft_score_raw   — TFT up-probability [0, 1]
  cot_signal      — COT managed-money net positioning signal [-1, 1]
  pcr_signal      — CBOE put/call contrarian signal [-1, 1]

Target: actual_bullish — reconstructed from (direction, direction_correct):
  BULLISH+correct=1 or BEARISH+correct=0  → actual price rose  → 1
  BULLISH+correct=0 or BEARISH+correct=1  → actual price fell  → 0

One Pipeline (StandardScaler → LogisticRegression) is trained per horizon.
Saved to data/direction_model.pkl. compute_signals loads it at runtime and
falls back to hardcoded weights if the file is missing or a horizon lacks data.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CSV_PATH = "data/forecasts.csv"
MODEL_PATH = "data/direction_model.pkl"
MIN_SAMPLES = 30


def train_direction_model():
    if not os.path.exists(CSV_PATH):
        print("[train] No forecasts.csv — skipping.")
        return None

    df = pd.read_csv(CSV_PATH)
    for col in ["direction_correct", "p50", "p10", "p90", "last_price", "horizon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    graded = df[df["direction_correct"].notna()].copy()
    graded = graded[graded["last_price"].notna() & (graded["last_price"] > 0)]
    if graded.empty:
        print("[train] No graded rows with last_price — skipping.")
        return None

    graded["actual_bullish"] = (
        (graded["direction"] == "BULLISH") == (graded["direction_correct"] == 1)
    ).astype(int)

    graded["forecast_return"] = (graded["p50"] - graded["last_price"]) / graded["last_price"]
    graded["band_width"] = (graded["p90"] - graded["p10"]) / graded["last_price"]
    graded = graded.dropna(subset=["forecast_return", "band_width", "actual_bullish"])

    # Extended signal features — include when present in ≥50% of graded rows
    EXTENDED_FEATURES = ["macro_sc", "tft_score_raw", "cot_signal", "pcr_signal"]
    base_features = ["forecast_return", "band_width"]
    extra_features = []
    for col in EXTENDED_FEATURES:
        if col in graded.columns:
            coverage = pd.to_numeric(graded[col], errors="coerce").notna().mean()
            if coverage >= 0.50:
                graded[col] = pd.to_numeric(graded[col], errors="coerce").fillna(0.0)
                extra_features.append(col)
    feature_cols = base_features + extra_features
    if extra_features:
        print(f"[train] Extended features: {extra_features}")

    models = {}
    summary = {}
    for h in [5, 30, 90]:
        sub = graded[graded["horizon"] == h].copy()
        sub = sub.dropna(subset=feature_cols)
        n = len(sub)
        if n < MIN_SAMPLES:
            print(f"[train] Horizon {h}d: {n} samples (need {MIN_SAMPLES}) — skipped.")
            models[str(h)] = None
            summary[str(h)] = {"n": n, "trained": False}
            continue

        X = sub[feature_cols].values
        y = sub["actual_bullish"].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ])
        pipe.fit(X, y)
        train_acc = float((pipe.predict(X) == y).mean())
        bull_frac = float(y.mean())
        print(f"[train] Horizon {h}d: n={n}  features={feature_cols}  train_acc={train_acc:.3f}  bull_frac={bull_frac:.3f}")
        models[str(h)] = pipe
        # Store feature list so compute_signals can pass the right columns
        models[f"{h}_features"] = feature_cols
        summary[str(h)] = {
            "n": n, "trained": True, "features": feature_cols,
            "train_acc": round(train_acc, 4), "bull_frac": round(bull_frac, 4),
        }

    payload = {
        "models": models,
        "trained_at": datetime.now().isoformat(),
        "summary": summary,
    }
    os.makedirs("data", exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    print(f"[train] Saved {MODEL_PATH} — {json.dumps(summary)}")
    return payload


if __name__ == "__main__":
    train_direction_model()
