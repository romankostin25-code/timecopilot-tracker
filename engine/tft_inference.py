"""TFT batch inference — called once per forecast run, returns P(bullish) for all tickers.

Flow:
  1. `run_all_forecasts` calls `precompute_tft_scores(tickers, price_data, macro, news_df)`
  2. For each horizon, builds feature DataFrames from live price arrays, runs one
     forward pass through the loaded TFT checkpoint.
  3. Returns {ticker: {horizon: P(bullish)}} dict consumed by compute_signals.

Falls back gracefully when checkpoints or pytorch-forecasting are unavailable.
Dataset templates (saved by train_tft.py) are required for proper normalisation.
"""

import os
import pickle
import numpy as np
import pandas as pd

CHECKPOINT_DIR = "data/tft_checkpoint"

_MODEL_CACHE: dict = {}
_TEMPLATE_CACHE: dict = {}
_CACHE_LOADED: dict = {}


def _load_model_and_template(horizon: int):
    if _CACHE_LOADED.get(horizon):
        return _MODEL_CACHE.get(horizon), _TEMPLATE_CACHE.get(horizon)
    _CACHE_LOADED[horizon] = True

    ckpt     = os.path.join(CHECKPOINT_DIR, f"tft_h{horizon}.ckpt")
    template = os.path.join(CHECKPOINT_DIR, f"dataset_h{horizon}.pkl")

    if not os.path.exists(ckpt) or not os.path.exists(template):
        return None, None

    try:
        from pytorch_forecasting import TemporalFusionTransformer
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
        model.eval()
        _MODEL_CACHE[horizon] = model
        print(f"[tft] h{horizon} checkpoint loaded")
    except Exception as e:
        print(f"[tft] Failed to load h{horizon} checkpoint: {e}")
        return None, None

    try:
        with open(template, "rb") as f:
            tmpl = pickle.load(f)
        _TEMPLATE_CACHE[horizon] = tmpl
    except Exception as e:
        print(f"[tft] Failed to load h{horizon} dataset template: {e}")
        return None, None

    return _MODEL_CACHE[horizon], _TEMPLATE_CACHE[horizon]


def precompute_tft_scores(
    tickers: list[str],
    price_data: dict,          # {ticker: np.ndarray of prices, oldest first}
    horizons: tuple = (5, 30, 90),
    macro: dict | None = None,
    news_df: pd.DataFrame | None = None,
) -> dict:
    """Return {ticker: {horizon: P(bullish)}} for all available tickers/horizons.

    Only horizons with trained checkpoints produce output; others are omitted
    so compute_signals knows to fall back to the rule-based weights.
    """
    from engine.feature_builder import build_inference_features, TFT_FEATURE_COLS

    results: dict = {t: {} for t in tickers}
    any_checkpoint = any(
        os.path.exists(os.path.join(CHECKPOINT_DIR, f"tft_h{h}.ckpt"))
        for h in horizons
    )
    if not any_checkpoint:
        return results

    try:
        import torch
        from pytorch_forecasting import TimeSeriesDataSet
        from torch.utils.data import DataLoader
    except ImportError:
        print("[tft] pytorch-forecasting not installed — skipping TFT inference")
        return results

    for horizon in horizons:
        model, template = _load_model_and_template(horizon)
        if model is None or template is None:
            continue

        target_col = f"actual_bullish_{horizon}d"
        frames = []
        valid_tickers = []

        for ticker in tickers:
            prices = price_data.get(ticker)
            if prices is None or len(prices) < 80:
                continue
            feat = build_inference_features(ticker, prices, macro=macro, news_df=news_df)
            if feat is None:
                continue
            feat[target_col] = 0  # dummy — not used at inference
            frames.append(feat)
            valid_tickers.append(ticker)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)

        # Fill any missing feature columns
        for col in TFT_FEATURE_COLS:
            if col not in combined.columns:
                combined[col] = 0.0

        try:
            predict_ds = TimeSeriesDataSet.from_dataset(
                template, combined, predict=True, stop_randomization=True
            )
            loader = DataLoader(predict_ds, batch_size=64, shuffle=False, num_workers=0)

            with torch.no_grad():
                raw_preds = model.predict(loader, return_predictions=True)

            # raw_preds shape depends on pytorch-forecasting version
            # Each element is the prediction for one sample
            pred_list = []
            for p in raw_preds:
                if hasattr(p, "numpy"):
                    p = p.numpy()
                if np.ndim(p) == 2:
                    pred_list.append(float(p[-1, 1]))   # P(bullish) from last step
                else:
                    pred_list.append(float(p[-1]))

            for ticker, p_bullish in zip(valid_tickers, pred_list):
                results[ticker][horizon] = round(p_bullish, 4)

            print(f"[tft] h{horizon}: scored {len(valid_tickers)} tickers")

        except Exception as e:
            print(f"[tft] Batch inference h{horizon} failed: {e}")

    return results
