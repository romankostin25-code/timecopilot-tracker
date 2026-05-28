"""TFT batch inference — called once per forecast run, returns P(bullish) for all tickers.

Flow:
  1. `run_all_forecasts` calls `precompute_tft_scores(tickers, price_data, macro, news_df)`
  2. For each horizon, builds feature DataFrames from live price arrays, runs one
     forward pass through the loaded TFT checkpoint.
  3. Returns {ticker: {horizon: P(bullish)}} dict consumed by compute_signals.

Falls back gracefully when checkpoints or pytorch-forecasting are unavailable:
  - If pytorch-forecasting is missing, loads data/tft_scores_cache.json (written by
    the tft_scores.yml workflow which runs with ML deps before the evening forecast).
  - Cache is accepted if written today or yesterday (2-day window).

Dataset templates (saved by train_tft.py) are required for proper normalisation.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import date, timedelta

CACHE_PATH = "data/tft_scores_cache.json"

CHECKPOINT_DIR = "data/tft_checkpoint"

_MODEL_CACHE: dict = {}
_TEMPLATE_CACHE: dict = {}
_CACHE_LOADED: dict = {}


def _load_scores_cache(max_age_days: int = 2) -> dict | None:
    """Load tft_scores_cache.json if it exists and is fresh enough."""
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH) as f:
            payload = json.load(f)
        cache_date = date.fromisoformat(payload["date"])
        if (date.today() - cache_date).days > max_age_days:
            print(f"[tft] cache is {(date.today() - cache_date).days}d old — too stale, ignoring")
            return None
        print(f"[tft] loaded scores cache from {cache_date} ({len(payload['scores'])} tickers)")
        return payload["scores"]
    except Exception as e:
        print(f"[tft] cache load failed: {e}")
        return None


def _save_scores_cache(scores: dict) -> None:
    os.makedirs("data", exist_ok=True)
    payload = {"date": date.today().isoformat(), "scores": scores}
    with open(CACHE_PATH, "w") as f:
        json.dump(payload, f)
    print(f"[tft] saved scores cache — {len(scores)} tickers")


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
        print("[tft] pytorch-forecasting not installed — trying scores cache")
        cached = _load_scores_cache()
        if cached:
            # Merge cache into results dict (only tickers we're forecasting)
            for t in tickers:
                if t in cached:
                    results[t] = {int(k): v for k, v in cached[t].items()}
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
            print(f"[tft] h{horizon}: dataset size={len(predict_ds)}, "
                  f"index_len={len(predict_ds.index) if hasattr(predict_ds, 'index') else 'n/a'}")

            # pytorch-forecasting's own collate does key renaming
            # (e.g. decoder_length → decoder_lengths) that default_collate skips.
            # Extract it from a temp DataLoader so we can wrap it to patch None weights.
            try:
                _pf_collate = predict_ds.to_dataloader(
                    train=False, batch_size=64, num_workers=0
                ).collate_fn
            except Exception:
                _pf_collate = torch.utils.data.dataloader.default_collate

            def collate_fix_none_weights(batch):
                """Replace None sample weights before PF's collate sees them."""
                batch = [x for x in batch if x is not None]
                if not batch:
                    return None
                fixed = []
                for item in batch:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        x, y = item[0], item[1]
                        if isinstance(y, (tuple, list)) and len(y) >= 2 and y[1] is None:
                            n = int(y[0].shape[0]) if hasattr(y[0], "shape") and y[0].shape else 1
                            y = (y[0], torch.ones(n, dtype=torch.float32))
                        fixed.append((x, y))
                    else:
                        fixed.append(item)
                return _pf_collate(fixed)

            loader = DataLoader(predict_ds, batch_size=64, shuffle=False,
                                num_workers=0, collate_fn=collate_fix_none_weights)

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

    # Persist scores so lightweight runs (no pytorch) can use them
    non_empty = {t: v for t, v in results.items() if v}
    if non_empty:
        _save_scores_cache(non_empty)

    return results
