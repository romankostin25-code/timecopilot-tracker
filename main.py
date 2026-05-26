"""Trading Co-Pilot v3.0 — Main orchestrator."""

import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Trading Co-Pilot v3.0")
    parser.add_argument("--auto",       action="store_true", help="Full run: macro+news+poly+forecast+grade+explain+alerts")
    parser.add_argument("--forecast",   action="store_true", help="Run multi-horizon forecasts (5d/30d/90d)")
    parser.add_argument("--update",     action="store_true", help="Grade actuals, regenerate scorecard")
    parser.add_argument("--macro",      action="store_true", help="Fetch macro context (VIX, DXY, yields)")
    parser.add_argument("--news",       action="store_true", help="Poll RSS feeds + run Claude NLP pipeline")
    parser.add_argument("--polymarket", action="store_true", help="All three Polymarket arms (regime+narrative+bands)")
    parser.add_argument("--curve",      action="store_true", help="Fetch futures term structures")
    parser.add_argument("--explain",    action="store_true", help="Generate Claude signal explanations")
    parser.add_argument("--calendar",   action="store_true", help="Save upcoming economic events")
    parser.add_argument("--alerts",     action="store_true", help="Check and dispatch signal alerts")
    parser.add_argument("--history",    action="store_true", help="Fetch 90-day historical closing prices for all tickers")
    parser.add_argument("--backfill",   action="store_true", help="Walk-forward backfill: re-run model on past 14 trading days + grade")
    parser.add_argument("--poll",       action="store_true", help="15-min price + news poll (for local testing)")
    parser.add_argument("--regrade",    action="store_true", help="Re-grade direction_correct for all historical rows with corrected baseline")
    parser.add_argument("--train",      action="store_true", help="Train direction meta-learner (logistic regression) on graded history")
    parser.add_argument("--pipeline",   action="store_true", help="Run full ML data pipeline: pull news, compute features, assemble dataset")
    parser.add_argument("--news-update",action="store_true", help="Incremental news + features update (last 3 days, for daily inference)")
    parser.add_argument("--cot",        action="store_true", help="Fetch CFTC COT positioning data (weekly)")
    parser.add_argument("--pcr",        action="store_true", help="Fetch CBOE equity put/call ratio (daily contrarian signal)")
    parser.add_argument("--ng-storage", action="store_true", help="Fetch EIA weekly natural gas storage signal for NG=F")
    parser.add_argument("--tft-scores", action="store_true", help="Run TFT inference and save scores cache (requires ML deps)")
    # Legacy v2 flags — still supported
    parser.add_argument("--retrain",    action="store_true", help="Retrain directional classifiers (alias for --train)")
    parser.add_argument("--backtest",   action="store_true", help="[v2] Walk-forward backtest")
    parser.add_argument("--aggregator", action="store_true", help="[v2] Run learning aggregator")
    args = parser.parse_args()

    if args.auto or args.macro:
        from engine.macro_fetcher import fetch_macro_context
        fetch_macro_context()

    if args.auto or args.cot:
        from scripts.fetch_cot import fetch_cot
        fetch_cot()

    if args.auto or args.pcr:
        from scripts.fetch_pcr import fetch_pcr
        fetch_pcr()

    if args.auto or getattr(args, "ng_storage", False):
        from scripts.fetch_ng_storage import fetch_ng_storage
        fetch_ng_storage()

    if args.auto or args.news:
        from intelligence.news_poller import fetch_all_feeds, update_feed
        from intelligence.nlp_pipeline import process_feed
        update_feed(fetch_all_feeds())
        process_feed(max_batch=30)

    if args.auto or args.polymarket:
        from polymarket.poller import fetch_all_snapshots
        from polymarket.regime_engine import run_regime_engine
        from polymarket.narrative_scorer import score_narratives
        from polymarket.signal_combiner import combine_signals
        fetch_all_snapshots()
        run_regime_engine()
        score_narratives()
        combine_signals()

    if args.auto or args.history:
        from engine.fetch_price_history import fetch_price_history
        fetch_price_history()

    if args.auto or args.forecast:
        from engine.run_forecast import run_all_forecasts
        from engine.futures_curve import fetch_all_curves
        run_all_forecasts()
        fetch_all_curves()

    if args.auto or args.update:
        from engine.update_actuals import fill_actuals_and_grade
        fill_actuals_and_grade()

    if args.auto or args.explain:
        from engine.explain_forecast import generate_all_explanations
        generate_all_explanations()

    if args.auto or args.calendar:
        from intelligence.calendar_engine import save_calendar
        save_calendar()

    if args.auto or args.alerts:
        from api.alerts import check_and_dispatch_alerts
        check_and_dispatch_alerts()

    if args.backfill:
        from engine.backfill_forecasts import backfill_forecasts
        backfill_forecasts()

    if args.regrade:
        from engine.update_actuals import regrade_direction_correct
        regrade_direction_correct()

    if args.train or args.retrain:
        from engine.train_direction_model import train_direction_model
        train_direction_model()

    if getattr(args, "tft_scores", False):
        from engine.run_forecast import run_tft_precompute
        run_tft_precompute()

    if args.pipeline:
        import subprocess, sys
        for script in ["scripts/pull_historical_news.py",
                       "scripts/compute_features.py",
                       "scripts/assemble_training_data.py"]:
            print(f"\n=== {script} ===")
            subprocess.run([sys.executable, script], check=True)

    if getattr(args, "news_update", False):
        import subprocess, sys
        print("\n=== Incremental news + features update (3d) ===")
        subprocess.run([sys.executable, "scripts/pull_historical_news.py", "--days", "3"], check=False)
        subprocess.run([sys.executable, "scripts/compute_features.py", "--days", "3"], check=False)

    if args.poll:
        from api.poll import handler
        handler(None)

    if args.curve and not args.forecast and not args.auto:
        from engine.futures_curve import fetch_all_curves
        fetch_all_curves()

    if args.backtest:
        from backtester import run_all_backtests
        tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]
        run_all_backtests(tickers)

    if args.aggregator:
        from aggregator import run_aggregator
        run_aggregator()

    if not any(vars(args).values()):
        print("""
Trading Co-Pilot v3.0 — Command Reference
──────────────────────────────────────────
python main.py --auto        Full run (recommended for CI)
python main.py --forecast    Run multi-horizon forecasts (5d/30d/90d)
python main.py --update      Grade actuals, regenerate scorecard
python main.py --macro       Fetch VIX, DXY, yields, FRED
python main.py --news        Poll RSS feeds + run Claude NLP
python main.py --polymarket  All three Polymarket arms (regime+narrative+bands)
python main.py --curve       Fetch futures term structures
python main.py --explain     Generate Claude signal explanations
python main.py --calendar    Save upcoming economic events
python main.py --alerts      Check and dispatch signal alerts
python main.py --poll        15-min price + news poll
python main.py --cot         Fetch CFTC COT positioning data (weekly)
python main.py --pcr         Fetch CBOE equity put/call ratio (daily)

[v2 legacy]
python main.py --retrain     Retrain directional classifiers
python main.py --backtest    Walk-forward backtest
python main.py --aggregator  Run learning aggregator
        """)


if __name__ == "__main__":
    main()
