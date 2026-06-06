"""Train per-horizon logistic regression on graded forecast history.

Core features (always used):
  forecast_return = (p50 - last_price) / last_price  — model's predicted move
  band_width      = (p90 - p10) / last_price          — model uncertainty

Extended features (used when present in ≥50% of graded rows):
  macro_sc        — asset-class macro composite signal [-1, 1]
  tft_score_raw   — TFT up-probability [0, 1]
  cot_signal      — COT managed-money net positioning signal [-1, 1]
  pcr_signal      — CBOE put/call contrarian signal [-1, 1]
  news_sc         — news sentiment score [-1, 1]

Ticker dummies (always included when ticker has ≥ MIN_SAMPLES rows):
  ticker_<SYMBOL> — one-hot per ticker so the model learns per-asset biases.
  This lets the LR learn "SI=F with positive forecast → actually bearish" without
  overriding the signal for other tickers.

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
MIN_TICKER_ROWS = 15  # minimum rows per ticker to include its dummy


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
    EXTENDED_FEATURES = ["macro_sc", "tft_score_raw", "cot_signal", "pcr_signal", "news_sc"]
    base_features = ["forecast_return", "band_width"]
    extra_features = []
    for col in EXTENDED_FEATURES:
        if col in graded.columns:
            coverage = pd.to_numeric(graded[col], errors="coerce").notna().mean()
            if coverage >= 0.50:
                graded[col] = pd.to_numeric(graded[col], errors="coerce").fillna(0.0)
                extra_features.append(col)
    if extra_features:
        print(f"[train] Extended features: {extra_features}")

    # Ticker one-hot dummies — let LR learn per-ticker biases
    ticker_counts = graded["ticker"].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= MIN_TICKER_ROWS].index.tolist()
    ticker_dummies = pd.get_dummies(
        graded["ticker"].where(graded["ticker"].isin(valid_tickers), other="OTHER"),
        prefix="ticker",
    )
    # Remove the "OTHER" column (reference level) to avoid multicollinearity
    other_col = "ticker_OTHER"
    if other_col in ticker_dummies.columns:
        ticker_dummies = ticker_dummies.drop(columns=[other_col])
    ticker_dummy_cols = list(ticker_dummies.columns)
    if ticker_dummy_cols:
        graded = pd.concat([graded.reset_index(drop=True), ticker_dummies.reset_index(drop=True)], axis=1)
        print(f"[train] Ticker dummies: {len(ticker_dummy_cols)} tickers ({valid_tickers[:5]}...)")

    feature_cols = base_features + extra_features + ticker_dummy_cols

    models = {}
    summary = {}
    for h in [5, 30, 90]:
        sub = graded[graded["horizon"] == h].copy()
        sub = sub.dropna(subset=base_features + extra_features)
        # Fill ticker dummies with 0 for rows where they might be NaN after concat
        for col in ticker_dummy_cols:
            if col in sub.columns:
                sub[col] = sub[col].fillna(0).astype(float)
            else:
                sub[col] = 0.0
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
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42,
                                      C=0.5)),  # regularise more to avoid overfitting ticker dummies
        ])
        pipe.fit(X, y)
        train_acc = float((pipe.predict(X) == y).mean())
        bull_frac = float(y.mean())
        print(f"[train] Horizon {h}d: n={n}  features={len(feature_cols)}  "
              f"train_acc={train_acc:.3f}  bull_frac={bull_frac:.3f}")
        models[str(h)] = pipe
        models[f"{h}_features"] = feature_cols
        summary[str(h)] = {
            "n": n, "trained": True, "features": feature_cols,
            "train_acc": round(train_acc, 4), "bull_frac": round(bull_frac, 4),
        }

    payload = {
        "models": models,
        "trained_at": datetime.now().isoformat(),
        "summary": summary,
        "ticker_dummies": ticker_dummy_cols,
    }
    os.makedirs("data", exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    print(f"[train] Saved {MODEL_PATH} — {json.dumps({h: s.get('trained') for h, s in summary.items()})}")
    return payload


if __name__ == "__main__":
    train_direction_model()
