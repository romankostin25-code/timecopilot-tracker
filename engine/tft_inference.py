"""TFT inference — run trained checkpoints to get P(bullish) per ticker/horizon.

Called from compute_signals when data/tft_checkpoint/ exists and contains
trained checkpoints. Falls back gracefully if pytorch-forecasting unavailable.

Returns a dict: {ticker: {5: p_bullish, 30: p_bullish, 90: p_bullish}}
Cached in memory for the duration of a forecast run.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import date, timedelta

CHECKPOINT_DIR = "data/tft_checkpoint"
DATASET_PATH   = "data/training_dataset.parquet"

_TFT_CACHE: dict = {}          # horizon → (model, dataset_config)
_TFT_LOADED: dict = {}         # horizon → bool


def _load_tft(horizon: int):
    if _TFT_LOADED.get(horizon):
        return _TFT_CACHE.get(horizon)
    _TFT_LOADED[horizon] = True

    ckpt = os.path.join(CHECKPOINT_DIR, f"tft_h{horizon}.ckpt")
    if not os.path.exists(ckpt):
        return None
    try:
        from pytorch_forecasting import TemporalFusionTransformer
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
        model.eval()
        _TFT_CACHE[horizon] = model
        print(f"[tft] Loaded h{horizon} checkpoint")
        return model
    except Exception as e:
        print(f"[tft] Failed to load h{horizon}: {e}")
        return None


def get_tft_probability(ticker: str, horizon: int,
                        recent_features: pd.DataFrame | None = None) -> float | None:
    """Return P(bullish) for ticker/horizon from TFT, or None if unavailable.

    recent_features: DataFrame with the last 60+ rows of features for this ticker.
    If None, loads from data/training_dataset.parquet (slower but works for daily runs).
    """
    model = _load_tft(horizon)
    if model is None:
        return None

    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from torch.utils.data import DataLoader

        if recent_features is None:
            if not os.path.exists(DATASET_PATH):
                return None
            full = pd.read_parquet(DATASET_PATH)
            recent_features = full[full["ticker"] == ticker].tail(120).copy()

        if len(recent_features) < 30:
            return None

        # Prepare inference dataset
        df = recent_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Add dummy target for inference
        target_col = f"actual_bullish_{horizon}d"
        if target_col not in df.columns:
            df[target_col] = 0

        # Fill NaN features
        from engine.train_tft import TIME_VARYING_UNKNOWN, _fill_missing_features
        df = _fill_missing_features(df)

        # Use last row as prediction point
        predict_df = df.tail(61).copy()  # 60 encoder + 1 decoder

        dataset = TimeSeriesDataSet(
            data=predict_df,
            time_idx="time_idx",
            target=target_col,
            group_ids=["ticker"],
            min_encoder_length=30,
            max_encoder_length=60,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=["ticker"],
            time_varying_unknown_reals=TIME_VARYING_UNKNOWN,
            target_normalizer=None,
            predict_mode=True,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        import torch
        with torch.no_grad():
            preds = model.predict(loader, return_predictions=True)
        if preds is not None and len(preds) > 0:
            probs = preds[0]
            if hasattr(probs, "numpy"):
                probs = probs.numpy()
            p_bullish = float(probs[-1, 1]) if probs.ndim == 2 else float(probs[-1])
            return p_bullish
    except Exception as e:
        print(f"[tft] Inference error {ticker} h{horizon}: {e}")
    return None


def batch_tft_predictions(tickers: list[str], horizons: tuple = (5, 30, 90)) -> dict:
    """Return {ticker: {5: p, 30: p, 90: p}} for all tickers with loaded checkpoints."""
    if not os.path.exists(DATASET_PATH):
        return {}

    try:
        full = pd.read_parquet(DATASET_PATH)
    except Exception:
        return {}

    results = {}
    for ticker in tickers:
        ticker_df = full[full["ticker"] == ticker]
        if ticker_df.empty:
            continue
        results[ticker] = {}
        for h in horizons:
            p = get_tft_probability(ticker, h, recent_features=ticker_df.tail(120).copy())
            if p is not None:
                results[ticker][h] = round(p, 4)
    return results
