"""Trading Co-Pilot — Main orchestrator."""

import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 17-ticker universe (PRD v1)
DEFAULT_TICKERS = (
    "NG=F,GC=F,SI=F,CL=F,HG=F,HO=F,PA=F,"
    "EURUSD=X,GBPUSD=X,USDJPY=X,AUDUSD=X,"
    "DX-Y.NYB,^TNX,^IRX,^GSPC,^VIX,HYG"
)


def main():
    parser = argparse.ArgumentParser(description="Trading Co-Pilot")
    parser.add_argument("--forecast",     action="store_true", help="Run forecasts")
    parser.add_argument("--update",       action="store_true", help="Grade forecasts against real prices")
    parser.add_argument("--auto",         action="store_true", help="Full run: macro + poly + forecast + grade + explain + aggregator")
    parser.add_argument("--retrain",      action="store_true", help="Retrain all directional classifiers (walk-forward)")
    parser.add_argument("--backtest",     action="store_true", help="Run walk-forward backtest for all tickers")
    parser.add_argument("--curve",        action="store_true", help="Fetch futures curves only")
    parser.add_argument("--polymarket",   action="store_true", help="Fetch Polymarket odds")
    parser.add_argument("--poly-signals", action="store_true", help="Polymarket spike detection + signal generation")
    parser.add_argument("--macro",        action="store_true", help="Fetch macro context only")
    parser.add_argument("--explain",      action="store_true", help="Generate LLM signal explanations")
    parser.add_argument("--aggregator",   action="store_true", help="Train and run the learning aggregator")
    args = parser.parse_args()

    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", DEFAULT_TICKERS).split(",") if t.strip()]

    if args.auto:
        print("=== Trading Co-Pilot: Full Auto Run ===")
        from macro_fetcher      import fetch_macro_context
        from polymarket_fetcher import fetch_all_polymarket
        from run_forecast       import run_all_forecasts
        from futures_curve      import fetch_all_curves
        from update_actuals     import fill_actuals_and_grade
        from explain_forecast   import generate_all_explanations
        from aggregator         import run_aggregator

        fetch_macro_context()
        fetch_all_polymarket()
        run_all_forecasts()
        fetch_all_curves()
        fill_actuals_and_grade()

        try:
            from polymarket_predictions.poller           import fetch_all_snapshots
            from polymarket_predictions.spike_detector   import detect_spikes
            from polymarket_predictions.signal_generator import generate_all_signals
            from polymarket_predictions.correlation_engine import build_all_correlations
            from polymarket_predictions.integrator       import apply_to_forecast_csv
            fetch_all_snapshots()
            detect_spikes()
            generate_all_signals()
            build_all_correlations()
            apply_to_forecast_csv()
        except Exception as e:
            print(f"  [poly-signals] {e}")

        generate_all_explanations()
        run_aggregator()

        # Save Polymarket calibration stats
        try:
            from poly_calibration import save_calibration_stats
            save_calibration_stats()
        except Exception as e:
            print(f"  [poly_calibration] {e}")

        print("\n=== Auto run complete ===")

    elif args.retrain:
        print("=== Retraining directional classifiers ===")
        import pandas as pd
        from pathlib import Path
        from directional_classifier import retrain_ticker

        macro_df = pd.read_csv("macro_context.csv")   if Path("macro_context.csv").exists()   else pd.DataFrame()
        poly_df  = pd.read_csv("polymarket_data.csv") if Path("polymarket_data.csv").exists() else pd.DataFrame()

        results = []
        for ticker in tickers:
            print(f"\n[retrain] {ticker}")
            r = retrain_ticker(ticker, macro_df, poly_df)
            if r:
                results.append(r)
                print(f"  ✓ OOS accuracy: {r['oos_accuracy']:.3f}  "
                      f"(confident {r['confident_n']}/{r['oos_n']})")
            else:
                print(f"  ✗ insufficient data")

        if results:
            import json
            from pathlib import Path
            Path("data").mkdir(exist_ok=True)
            with open("data/retrain_results.json", "w") as f:
                json.dump(results, f, indent=2)
            avg = sum(r["oos_accuracy"] for r in results) / len(results)
            print(f"\n=== Retrain complete: {len(results)}/{len(tickers)} models, avg OOS acc={avg:.3f} ===")

    elif args.backtest:
        from backtester import run_all_backtests
        run_all_backtests(tickers)

    elif args.forecast:
        from macro_fetcher      import fetch_macro_context
        from polymarket_fetcher import fetch_all_polymarket
        from run_forecast       import run_all_forecasts
        from futures_curve      import fetch_all_curves
        fetch_macro_context()
        fetch_all_polymarket()
        run_all_forecasts()
        fetch_all_curves()

    elif args.update:
        from update_actuals import fill_actuals_and_grade
        fill_actuals_and_grade()

    elif args.curve:
        from futures_curve import fetch_all_curves
        fetch_all_curves()

    elif args.polymarket:
        from polymarket_fetcher import fetch_all_polymarket
        fetch_all_polymarket()

    elif args.poly_signals:
        from polymarket_predictions.poller           import fetch_all_snapshots
        from polymarket_predictions.spike_detector   import detect_spikes
        from polymarket_predictions.signal_generator import generate_all_signals
        from polymarket_predictions.correlation_engine import build_all_correlations
        from polymarket_predictions.integrator       import apply_to_forecast_csv
        rows   = fetch_all_snapshots()
        spikes = detect_spikes()
        generate_all_signals()
        build_all_correlations()
        apply_to_forecast_csv()
        print(f"\n[poly-signals] {len(rows)} markets polled, {len(spikes)} spikes detected.")

    elif args.macro:
        from macro_fetcher import fetch_macro_context
        fetch_macro_context()

    elif args.explain:
        from explain_forecast import generate_all_explanations
        generate_all_explanations()

    elif args.aggregator:
        from aggregator import run_aggregator
        run_aggregator()

    else:
        print("""
Trading Co-Pilot — Command Reference
─────────────────────────────────────
python main.py --auto          Full run: macro + poly + forecast + curve + grade + explain + aggregator
python main.py --forecast      Run forecasts (includes macro + polymarket + curve)
python main.py --retrain       Retrain directional classifiers (walk-forward, ~15 min)
python main.py --backtest      Walk-forward backtest + accuracy report
python main.py --update        Grade forecasts against real prices, regenerate scorecard
python main.py --curve         Fetch futures curves only
python main.py --polymarket    Fetch Polymarket odds
python main.py --poly-signals  Polymarket spike detection + signal generation
python main.py --macro         Fetch macro context only
python main.py --explain       Generate LLM signal explanations (requires OPENAI_API_KEY)
python main.py --aggregator    Train and run the learning aggregator
        """)


if __name__ == "__main__":
    main()
