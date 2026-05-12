"""
Walk-forward directional classifier: XGBoost + LightGBM + Logistic Regression ensemble.

Training:  python main.py --retrain
Inference: predict_prob_up(ticker, feature_dict) → P(price up next trading day)
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from feature_pipeline import FEATURE_COLS, compute_features, fetch_ohlcv

warnings.filterwarnings("ignore")

MODEL_DIR    = Path("data/clf_models")
TRAIN_WINDOW = 252   # trading days in training window
RETRAIN_STEP = 21    # advance step for walk-forward
MIN_TRAIN    = 100   # minimum rows before fitting


def _fill(X: pd.DataFrame) -> np.ndarray:
    arr = X.values.astype(float)
    for j in range(arr.shape[1]):
        col_vals = arr[:, j]
        valid = col_vals[~np.isnan(col_vals)]
        fill_val = float(np.median(valid)) if len(valid) > 0 else 0.0
        arr[np.isnan(arr[:, j]), j] = fill_val
    return arr


def _build_ensemble(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model    import LogisticRegression
    from sklearn.preprocessing   import StandardScaler
    import xgboost  as xgb
    import lightgbm as lgb

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb_m = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    xgb_m.fit(X, y)

    lgb_m = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=5, verbose=-1, random_state=42,
    )
    lgb_m.fit(X, y)

    lr_m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_m.fit(X_scaled, y)

    return {"xgb": xgb_m, "lgb": lgb_m, "lr": lr_m, "scaler": scaler}


def _ensemble_prob(models: dict, row: np.ndarray) -> float:
    p_xgb = float(models["xgb"].predict_proba([row])[0][1])
    p_lgb = float(models["lgb"].predict_proba([row])[0][1])
    row_s = models["scaler"].transform([row])
    p_lr  = float(models["lr"].predict_proba(row_s)[0][1])
    return (p_xgb + p_lgb + p_lr) / 3.0


def _model_path(ticker: str) -> Path:
    safe = ticker.replace("=", "_").replace("^", "_").replace("-", "_").replace(".", "_")
    return MODEL_DIR / f"{safe}_clf.pkl"


def train_and_save(ticker: str, feature_df: pd.DataFrame) -> Optional[dict]:
    """
    Walk-forward cross-validate and save final model for live inference.
    Returns OOS accuracy metrics or None on failure.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feat_cols = [c for c in FEATURE_COLS if c in feature_df.columns]
    df = feature_df.dropna(subset=["target"]).reset_index(drop=True)

    if len(df) < TRAIN_WINDOW + 20:
        print(f"  [{ticker}] classifier: only {len(df)} rows, need {TRAIN_WINDOW + 20}")
        return None

    oos_probs, oos_targets = [], []
    last_models = None

    for t_end in range(TRAIN_WINDOW, len(df), RETRAIN_STEP):
        train = df.iloc[:t_end]
        test  = df.iloc[t_end:t_end + RETRAIN_STEP]
        if test.empty:
            break

        X_tr = _fill(train[feat_cols])
        y_tr = train["target"].values.astype(int)

        if len(X_tr) < MIN_TRAIN:
            continue
        pos, neg = y_tr.sum(), (len(y_tr) - y_tr.sum())
        if pos < 10 or neg < 10:
            continue

        try:
            models      = _build_ensemble(X_tr, y_tr)
            last_models = models
            X_te        = _fill(test[feat_cols])
            for row in X_te:
                oos_probs.append(_ensemble_prob(models, row))
            oos_targets.extend(test["target"].values.astype(int).tolist())
        except Exception as e:
            print(f"  [{ticker}] walk-forward step {t_end}: {e}")
            continue

    if not oos_probs or last_models is None:
        return None

    oos_p = np.array(oos_probs)
    oos_t = np.array(oos_targets[: len(oos_p)])

    confident = (oos_p > 0.55) | (oos_p < 0.45)
    if confident.sum() >= 10:
        dir_acc = float(((oos_p[confident] > 0.5) == oos_t[confident]).mean())
    else:
        dir_acc = float(((oos_p > 0.5) == oos_t).mean())

    print(f"  [{ticker}] CLF OOS dir-acc={dir_acc:.3f}  "
          f"(confident={confident.sum()}/{len(oos_p)})")

    # Retrain on all data for live inference
    X_all = _fill(df[feat_cols])
    y_all = df["target"].values.astype(int)
    try:
        final = _build_ensemble(X_all, y_all)
        with open(_model_path(ticker), "wb") as f:
            pickle.dump({"models": final, "feat_cols": feat_cols, "ticker": ticker,
                         "oos_accuracy": dir_acc}, f)
    except Exception as e:
        print(f"  [{ticker}] final model save: {e}")
        return None

    return {
        "ticker":       ticker,
        "oos_accuracy": round(dir_acc, 4),
        "oos_n":        len(oos_p),
        "confident_n":  int(confident.sum()),
    }


def retrain_ticker(ticker: str, macro_df: pd.DataFrame = None,
                   poly_df: pd.DataFrame = None) -> Optional[dict]:
    """Fetch data, compute features, train and save model."""
    try:
        price_df = fetch_ohlcv(ticker, years=3)
        feat_df  = compute_features(price_df, macro_df, poly_df, ticker=ticker)
        return train_and_save(ticker, feat_df)
    except Exception as e:
        print(f"  [{ticker}] retrain error: {e}")
        return None


def predict_prob_up(ticker: str, feature_dict: dict) -> Optional[float]:
    """
    Load saved model and return P(price up next trading day).
    Returns None if no model exists for this ticker.
    """
    path = _model_path(ticker)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        models    = bundle["models"]
        feat_cols = bundle["feat_cols"]
        row       = np.array([feature_dict.get(c, 0.0) or 0.0 for c in feat_cols])
        return _ensemble_prob(models, row)
    except Exception as e:
        print(f"  [clf] predict error {ticker}: {e}")
        return None
