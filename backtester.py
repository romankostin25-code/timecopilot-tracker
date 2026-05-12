"""
Walk-forward backtester: trains directional classifiers for all tickers
and prints OOS accuracy by asset and tier.

Usage:  python main.py --backtest
"""

import warnings
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

RESULTS_PATH = Path("data/backtest_results.csv")


def run_all_backtests(tickers: list[str],
                      macro_csv: str = "macro_context.csv",
                      poly_csv:  str = "polymarket_data.csv") -> pd.DataFrame:
    from feature_pipeline        import fetch_ohlcv, compute_features
    from directional_classifier  import train_and_save
    from conviction_thresholds   import get_ticker_tier

    macro_df = pd.read_csv(macro_csv) if Path(macro_csv).exists() else pd.DataFrame()
    poly_df  = pd.read_csv(poly_csv)  if Path(poly_csv).exists()  else pd.DataFrame()

    results = []
    for ticker in tickers:
        print(f"[backtest] {ticker}…")
        try:
            price_df = fetch_ohlcv(ticker, years=3)
            feat_df  = compute_features(price_df, macro_df, poly_df, ticker=ticker)
            metrics  = train_and_save(ticker, feat_df)
            if metrics:
                metrics["tier"] = get_ticker_tier(ticker)
                results.append(metrics)
            else:
                results.append({"ticker": ticker, "tier": get_ticker_tier(ticker),
                                 "oos_accuracy": None, "error": "insufficient data"})
        except Exception as e:
            results.append({"ticker": ticker, "tier": get_ticker_tier(ticker),
                             "oos_accuracy": None, "error": str(e)})
            print(f"  [{ticker}] backtest error: {e}")

    df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n=== Backtest Summary ===")
    print(df[["ticker", "tier", "oos_accuracy", "confident_n"]].to_string(index=False))

    for tier in ["TIER_1", "TIER_2", "TIER_3"]:
        subset = df[(df["tier"] == tier) & df["oos_accuracy"].notna()]
        if not subset.empty:
            avg = subset["oos_accuracy"].mean()
            print(f"  {tier} avg directional accuracy: {avg:.3f}")

    print(f"\n[backtest] results → {RESULTS_PATH}")
    return df
