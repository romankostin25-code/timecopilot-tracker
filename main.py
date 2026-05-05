"""Trading Co-Pilot — Main orchestrator."""

import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Trading Co-Pilot")
    parser.add_argument("--forecast",          action="store_true", help="Run forecasts (includes macro + polymarket + curve)")
    parser.add_argument("--update",            action="store_true", help="Grade forecasts against real prices")
    parser.add_argument("--auto",              action="store_true", help="Full run: macro + poly + forecast + curve + grade + explain + aggregator")
    parser.add_argument("--curve",             action="store_true", help="Fetch futures curves only")
    parser.add_argument("--polymarket",        action="store_true", help="Fetch Polymarket odds only (old-style direct match)")
    parser.add_argument("--poly-signals",      action="store_true", help="Run Polymarket spike detection + signal generation")
    parser.add_argument("--macro",             action="store_true", help="Fetch macro context only")
    parser.add_argument("--explain",           action="store_true", help="Generate LLM signal explanations")
    parser.add_argument("--aggregator",        action="store_true", help="Train and run the learning aggregator model")
    args = parser.parse_args()

    if args.auto:
        print("=== Trading Co-Pilot: Full Auto Run ===")
        from macro_fetcher       import fetch_macro_context
        from polymarket_fetcher  import fetch_all_polymarket
        from run_forecast        import run_all_forecasts
        from futures_curve       import fetch_all_curves
        from update_actuals      import fill_actuals_and_grade
        from explain_forecast    import generate_all_explanations
        from aggregator          import run_aggregator

        fetch_macro_context()
        fetch_all_polymarket()
        run_all_forecasts()
        fetch_all_curves()
        fill_actuals_and_grade()

        # Polymarket spike signals (runs after snapshots accumulate; no-op on first day)
        try:
            from polymarket_predictions.poller import fetch_all_snapshots
            from polymarket_predictions.spike_detector import detect_spikes
            from polymarket_predictions.signal_generator import generate_all_signals
            from polymarket_predictions.correlation_engine import build_all_correlations
            from polymarket_predictions.integrator import apply_to_forecast_csv
            fetch_all_snapshots()
            detect_spikes()
            generate_all_signals()
            build_all_correlations()
            apply_to_forecast_csv()
        except Exception as e:
            print(f"  [poly-signals] {e}")

        generate_all_explanations()
        run_aggregator()
        print("\n=== Auto run complete ===")

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
        from polymarket_predictions.poller import fetch_all_snapshots
        from polymarket_predictions.spike_detector import detect_spikes
        from polymarket_predictions.signal_generator import generate_all_signals
        from polymarket_predictions.correlation_engine import build_all_correlations
        from polymarket_predictions.integrator import apply_to_forecast_csv
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
python main.py --update        Grade forecasts against real prices, regenerate scorecard
python main.py --curve         Fetch futures curves only
python main.py --polymarket    Fetch Polymarket direct-match odds only
python main.py --poly-signals  Polymarket spike detection + signal generation
python main.py --macro         Fetch macro context only
python main.py --explain       Generate LLM signal explanations (requires OPENAI_API_KEY)
python main.py --aggregator    Train and run the learning aggregator model
        """)


if __name__ == "__main__":
    main()
