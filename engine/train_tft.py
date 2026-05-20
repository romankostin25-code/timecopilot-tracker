"""Train Temporal Fusion Transformer on the assembled training dataset.

Architecture:
  - Encoder lookback: 60 trading days
  - Decoder horizon: 90 steps
  - At inference we read positions 5, 30, 90 for respective horizon predictions
  - Target: actual_bullish_5d (binary classification, extended to multi-target via separate models)
  - Features: log returns, volatility, RSI, MACD, Bollinger, macro, sentiment

Usage:
    python engine/train_tft.py                          # train all horizons
    python engine/train_tft.py --horizon 5              # single horizon
    python engine/train_tft.py --horizon 5 --epochs 30  # custom epochs
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

DATASET_PATH = "data/training_dataset.parquet"
CHECKPOINT_DIR = "data/tft_checkpoint"
MIN_ROWS_PER_TICKER = 60


TIME_VARYING_KNOWN = []  # no future-known inputs in our setup

TIME_VARYING_UNKNOWN = [
    "log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d",
    "vol_20d", "vol_60d",
    "rsi_14", "macd_signal", "bb_pos", "vol_ratio_20d",
    "vix", "yield_10y", "yield_spread", "dxy",
    "sentiment_score", "sentiment_7d_ma", "sentiment_momentum",
]

STATIC_CATEGORICALS = ["ticker"]


def _fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in TIME_VARYING_UNKNOWN:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median()).fillna(0.0)
    return df


def train_horizon(horizon: int, df: pd.DataFrame, epochs: int = 50, batch_size: int = 64):
    target_col = f"actual_bullish_{horizon}d"
    if target_col not in df.columns:
        print(f"[TFT] No target column {target_col} in dataset.")
        return None

    try:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import CrossEntropy
        import lightning as L
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"[TFT] pytorch-forecasting not installed: {e}")
        print("  Install with: pip install pytorch-forecasting lightning")
        return None

    labeled = df[df[target_col].notna()].copy()
    if len(labeled) < MIN_ROWS_PER_TICKER * 5:
        print(f"[TFT] Horizon {horizon}d: only {len(labeled)} labeled rows — skipping.")
        return None

    labeled[target_col] = labeled[target_col].astype(int)
    labeled = _fill_missing_features(labeled)

    # Encode ticker as integer for static categorical
    ticker_map = {t: i for i, t in enumerate(sorted(labeled["ticker"].unique()))}
    labeled["ticker_idx"] = labeled["ticker"].map(ticker_map)

    # Filter tickers with enough rows
    counts = labeled.groupby("ticker")["time_idx"].count()
    valid_tickers = counts[counts >= MIN_ROWS_PER_TICKER].index
    labeled = labeled[labeled["ticker"].isin(valid_tickers)]
    print(f"[TFT] Horizon {horizon}d: {len(labeled)} rows, {labeled['ticker'].nunique()} tickers")

    # Train/val split: last 10% of time as validation
    max_idx = labeled.groupby("ticker")["time_idx"].transform("max")
    cutoff = max_idx * 0.90
    train_df = labeled[labeled["time_idx"] < cutoff].copy()
    val_df   = labeled[labeled["time_idx"] >= cutoff].copy()

    encoder_length = 60

    def make_dataset(data: pd.DataFrame, predict: bool = False):
        return TimeSeriesDataSet(
            data=data,
            time_idx="time_idx",
            target=target_col,
            group_ids=["ticker"],
            min_encoder_length=encoder_length,
            max_encoder_length=encoder_length,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=["ticker"],
            static_reals=[],
            time_varying_known_reals=[],
            time_varying_unknown_reals=TIME_VARYING_UNKNOWN,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
            predict_mode=predict,
        )

    train_dataset = make_dataset(train_df)

    # Pre-scan to filter indices where __getitem__ returns None (pytorch-forecasting bug)
    def _has_none(obj):
        if obj is None:
            return True
        if isinstance(obj, dict):
            return any(_has_none(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(_has_none(v) for v in obj)
        return False

    print(f"[TFT] Scanning {len(train_dataset)} sequences for invalid indices...")
    valid_idx = [i for i in range(len(train_dataset))
                 if not _has_none(train_dataset[i])]
    n_dropped = len(train_dataset) - len(valid_idx)
    if n_dropped:
        print(f"[TFT] Dropped {n_dropped} invalid sequences, {len(valid_idx)} remaining")
        from torch.utils.data import Subset
        train_dataset_use = Subset(train_dataset, valid_idx)
    else:
        train_dataset_use = train_dataset

    train_loader = DataLoader(train_dataset_use, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=32,
        output_size=2,  # binary: [P(bearish), P(bullish)]
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print(f"[TFT] Parameters: {sum(p.numel() for p in tft.parameters()):,}")

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        enable_model_summary=False,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )
    # No val_dataloaders — avoids None-tensor collate error from predict=True sequences
    trainer.fit(tft, train_dataloaders=train_loader)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path     = os.path.join(CHECKPOINT_DIR, f"tft_h{horizon}.ckpt")
    template_path = os.path.join(CHECKPOINT_DIR, f"dataset_h{horizon}.pkl")

    trainer.save_checkpoint(ckpt_path)

    with open(template_path, "wb") as f:
        pickle.dump(train_dataset, f)
    print(f"[TFT] Horizon {horizon}d trained → {ckpt_path}")
    return ckpt_path


def train_all(horizons=(5, 30, 90), epochs=50):
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found. Run scripts/assemble_training_data.py first.")
        return

    print(f"[TFT] Loading {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # Rebuild time_idx to guarantee no gaps (dropna in assembly may create gaps)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("ticker").cumcount()

    results = {}
    for h in horizons:
        print(f"\n{'='*50}\n[TFT] Training horizon {h}d\n{'='*50}")
        ckpt = train_horizon(h, df, epochs=epochs)
        results[str(h)] = {"checkpoint": ckpt, "trained_at": datetime.now().isoformat()}

    import json
    os.makedirs("data", exist_ok=True)
    with open("data/tft_training_log.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[TFT] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=None, help="Single horizon (5/30/90)")
    parser.add_argument("--epochs",  type=int, default=50)
    args = parser.parse_args()

    if args.horizon:
        if not os.path.exists(DATASET_PATH):
            print(f"ERROR: {DATASET_PATH} not found.")
        else:
            df = pd.read_parquet(DATASET_PATH)
            df["date"] = pd.to_datetime(df["date"])
            train_horizon(args.horizon, df, epochs=args.epochs)
    else:
        train_all(epochs=args.epochs)
