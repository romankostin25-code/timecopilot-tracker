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
    # Core macro
    "vix", "yield_10y", "yield_spread", "dxy",
    # Credit market
    "ig_spread", "hy_spread",
    # Yield curve
    "yield_2s10s", "yield_5s30s",
    # Fama-French factors
    "ff_mkt_rf", "ff_smb", "ff_hml", "ff_mom",
    # Options market
    "pcr",
    # Macro surprise
    "macro_surprise_ism",
    # News sentiment
    "sentiment_score", "sentiment_7d_ma", "sentiment_momentum",
]

STATIC_CATEGORICALS = ["ticker"]

FINETUNE_DAYS = 90  # days of recent data used for Phase 2 fine-tuning


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
        from pytorch_forecasting.data import NaNLabelEncoder
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
            target_normalizer=NaNLabelEncoder(add_nan=False),
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
            predict_mode=predict,
        )

    # Only pass columns pytorch-forecasting needs — extra cols can corrupt internal conversion
    required_cols = (["ticker", "time_idx", target_col] + TIME_VARYING_UNKNOWN)
    train_df_clean = train_df[[c for c in required_cols if c in train_df.columns]].reset_index(drop=True)

    train_dataset = make_dataset(train_df_clean)
    # Use pytorch-forecasting's own to_dataloader — it has a custom collate_fn
    # that handles the (target, None_weight) tuple that default_collate can't handle
    train_loader  = train_dataset.to_dataloader(batch_size=batch_size, train=True, num_workers=0)

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

    # Build validation DataLoader from the last 20% of time (predict=False for standard sliding window)
    val_loader = None
    if len(val_df) >= encoder_length + 5:
        val_cols = [c for c in required_cols if c in val_df.columns]
        val_df_clean = val_df[val_cols].reset_index(drop=True)
        try:
            val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df_clean, predict=False, stop_randomization=True)
            val_loader  = val_dataset.to_dataloader(batch_size=batch_size, train=False, num_workers=0)
            print(f"[TFT] h{horizon}: val_size={len(val_dataset)}")
        except Exception as e:
            print(f"[TFT] h{horizon}: val_loader failed ({e}) — training without validation")

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path     = os.path.join(CHECKPOINT_DIR, f"tft_h{horizon}.ckpt")
    template_path = os.path.join(CHECKPOINT_DIR, f"dataset_h{horizon}.pkl")
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"tft_h{horizon}_best.ckpt")

    callbacks = []
    if val_loader is not None:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True))
        callbacks.append(ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename=f"tft_h{horizon}_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ))

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        enable_model_summary=False,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        callbacks=callbacks if callbacks else None,
    )

    if val_loader is not None:
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(tft, train_dataloaders=train_loader)

    # Use best val checkpoint if available, otherwise save last
    best_ckpt_file = os.path.join(CHECKPOINT_DIR, f"tft_h{horizon}_best.ckpt")
    if os.path.exists(best_ckpt_file):
        import shutil
        shutil.copy(best_ckpt_file, ckpt_path)
        print(f"[TFT] h{horizon}: using best val_loss checkpoint → {ckpt_path}")
    else:
        trainer.save_checkpoint(ckpt_path)

    with open(template_path, "wb") as f:
        pickle.dump(train_dataset, f)
    print(f"[TFT] Horizon {horizon}d trained → {ckpt_path}")
    return ckpt_path


def finetune_horizon(horizon: int, df: pd.DataFrame, finetune_days: int = FINETUNE_DAYS,
                     lr: float = 1e-4, epochs: int = 20):
    """Phase 2: fine-tune an existing checkpoint on the most recent `finetune_days` of data.

    Loads the Phase 1 checkpoint, resets the optimizer to a lower LR, and continues
    training on only the most recent window. Saves back to the same checkpoint path.
    """
    ckpt_path     = os.path.join(CHECKPOINT_DIR, f"tft_h{horizon}.ckpt")
    template_path = os.path.join(CHECKPOINT_DIR, f"dataset_h{horizon}.pkl")

    if not os.path.exists(ckpt_path) or not os.path.exists(template_path):
        print(f"[TFT] h{horizon}: no checkpoint to fine-tune — run full training first")
        return

    try:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import CrossEntropy
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping
    except ImportError as e:
        print(f"[TFT] pytorch-forecasting not installed: {e}")
        return

    target_col = f"actual_bullish_{horizon}d"
    if target_col not in df.columns:
        print(f"[TFT] h{horizon}: no target column")
        return

    labeled = df[df[target_col].notna()].copy()
    labeled[target_col] = labeled[target_col].astype(int)
    labeled = _fill_missing_features(labeled)

    # Use only the most recent finetune_days window
    max_date = labeled["date"].max()
    cutoff   = max_date - pd.Timedelta(days=finetune_days + 90)  # extra for encoder lookback
    recent   = labeled[labeled["date"] >= cutoff].copy()

    ticker_map = {t: i for i, t in enumerate(sorted(recent["ticker"].unique()))}
    recent["ticker_idx"] = recent["ticker"].map(ticker_map)
    recent = recent.sort_values(["ticker", "date"]).reset_index(drop=True)
    recent["time_idx"] = recent.groupby("ticker").cumcount()

    encoder_length = 60
    if recent.groupby("ticker")["time_idx"].max().min() < encoder_length + 5:
        print(f"[TFT] h{horizon}: insufficient recent data for fine-tune ({len(recent)} rows)")
        return

    try:
        with open(template_path, "rb") as f:
            train_dataset_template = pickle.load(f)
    except Exception as e:
        print(f"[TFT] h{horizon}: template load failed: {e}")
        return

    required_cols = (["ticker", "time_idx", target_col] + TIME_VARYING_UNKNOWN)
    recent_clean  = recent[[c for c in required_cols if c in recent.columns]].reset_index(drop=True)

    try:
        ft_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset_template, recent_clean, predict=False, stop_randomization=True
        )
        ft_loader  = ft_dataset.to_dataloader(batch_size=32, train=True, num_workers=0)
    except Exception as e:
        print(f"[TFT] h{horizon}: fine-tune dataset creation failed: {e}")
        return

    try:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        model.hparams.learning_rate = lr
        # Reset optimizer by modifying configure_optimizers
        model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=lr)
    except Exception as e:
        print(f"[TFT] h{horizon}: checkpoint load for fine-tune failed: {e}")
        return

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        enable_model_summary=False,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        callbacks=[EarlyStopping(monitor="train_loss", patience=3, mode="min")],
    )

    print(f"[TFT] Fine-tuning h{horizon} on last {finetune_days}d ({len(recent)} rows), lr={lr}")
    trainer.fit(model, train_dataloaders=ft_loader)
    trainer.save_checkpoint(ckpt_path)
    print(f"[TFT] h{horizon} fine-tuned → {ckpt_path}")


def train_all(horizons=(5, 30, 90), epochs=30, finetune: bool = False,
              finetune_days: int = FINETUNE_DAYS):
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
    if finetune:
        # Phase 2 only — fine-tune existing checkpoints
        for h in horizons:
            print(f"\n{'='*50}\n[TFT] Fine-tuning horizon {h}d (last {finetune_days}d)\n{'='*50}")
            finetune_horizon(h, df, finetune_days=finetune_days)
            results[str(h)] = {"finetuned_at": datetime.now().isoformat(), "finetune_days": finetune_days}
    else:
        # Phase 1 — full training from scratch
        for h in horizons:
            print(f"\n{'='*50}\n[TFT] Training horizon {h}d\n{'='*50}")
            ckpt = train_horizon(h, df, epochs=epochs)
            results[str(h)] = {"checkpoint": ckpt, "trained_at": datetime.now().isoformat()}

    import json
    os.makedirs("data", exist_ok=True)
    log_path = "data/tft_training_log.json"
    existing = {}
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(results)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    print("\n[TFT] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",       type=int,  default=None,  help="Single horizon (5/30/90)")
    parser.add_argument("--epochs",        type=int,  default=30,    help="Max training epochs (Phase 1)")
    parser.add_argument("--finetune",      action="store_true",       help="Phase 2: fine-tune on recent data")
    parser.add_argument("--finetune-days", type=int,  default=FINETUNE_DAYS, help="Days of recent data for Phase 2")
    args = parser.parse_args()

    if args.horizon:
        if not os.path.exists(DATASET_PATH):
            print(f"ERROR: {DATASET_PATH} not found.")
        else:
            df = pd.read_parquet(DATASET_PATH)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
            df["time_idx"] = df.groupby("ticker").cumcount()
            if args.finetune:
                finetune_horizon(args.horizon, df, finetune_days=args.finetune_days)
            else:
                train_horizon(args.horizon, df, epochs=args.epochs)
    else:
        train_all(epochs=args.epochs, finetune=args.finetune, finetune_days=args.finetune_days)
