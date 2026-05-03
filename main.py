"""Main entry point with interactive menu and CLI flags."""

import argparse
import os
import sys

import pandas as pd

FORECASTS_CSV = "forecasts.csv"


def _stats_summary():
    if not os.path.exists(FORECASTS_CSV):
        print("No forecasts.csv found yet. Run a forecast first.")
        return

    df = pd.read_csv(FORECASTS_CSV)
    total = len(df)
    with_actual = df["actual"].notna() & (df["actual"].astype(str).str.strip() != "")
    n_actual = with_actual.sum()

    print(f"\n{'═'*50}")
    print(f"  TIMECOPILOT ACCURACY TRACKER — SUMMARY")
    print(f"{'═'*50}")
    print(f"  Total forecasts:          {total}")
    print(f"  With actual results:      {n_actual}")

    if n_actual == 0:
        print("  (No actuals yet — run option 2 to update)\n")
        return

    scored = df[with_actual].copy()
    scored["actual"] = pd.to_numeric(scored["actual"], errors="coerce")
    scored = scored.dropna(subset=["actual", "p10", "p90", "p50"])

    in_range = ((scored["actual"] >= scored["p10"]) &
                (scored["actual"] <= scored["p90"]))
    calibration = in_range.mean() * 100

    mae = (scored["actual"] - scored["p50"]).abs().mean()

    color_cal = "\033[92m" if calibration >= 70 else "\033[93m" if calibration >= 50 else "\033[91m"
    reset = "\033[0m"

    print(f"  Calibration (p10–p90 hit): {color_cal}{calibration:.1f}%{reset}")
    print(f"  Median forecast MAE:       {mae:.4f}")

    print(f"\n  By model:")
    for model, grp in scored.groupby("model_used"):
        m_in = ((grp["actual"] >= grp["p10"]) & (grp["actual"] <= grp["p90"])).mean() * 100
        m_mae = (grp["actual"] - grp["p50"]).abs().mean()
        print(f"    {model:<22} hit={m_in:.1f}%  MAE={m_mae:.4f}  (n={len(grp)})")

    print(f"{'═'*50}\n")


def _menu():
    while True:
        print("\n╔══════════════════════════════════╗")
        print("║     TIMECOPILOT TRACKER MENU     ║")
        print("╠══════════════════════════════════╣")
        print("║  1. Make a new forecast           ║")
        print("║  2. Update actual results         ║")
        print("║  3. Show accuracy stats           ║")
        print("║  4. Exit                          ║")
        print("╚══════════════════════════════════╝")

        choice = input("Choose [1-4]: ").strip()
        if choice == "1":
            from run_forecast import run_forecast
            run_forecast()
        elif choice == "2":
            from update_actuals import update_actuals
            update_actuals()
        elif choice == "3":
            _stats_summary()
        elif choice == "4":
            print("Goodbye.")
            sys.exit(0)
        else:
            print("Invalid choice — enter 1, 2, 3, or 4.")


def main():
    parser = argparse.ArgumentParser(description="TimeCopilot Tracker")
    parser.add_argument("--forecast", action="store_true", help="Run forecast")
    parser.add_argument("--update", action="store_true", help="Update actuals")
    parser.add_argument("--auto", action="store_true", help="Forecast + update actuals")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    args = parser.parse_args()

    # Change working directory to script location so CSV paths resolve correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.forecast:
        from run_forecast import run_forecast
        run_forecast()
    elif args.update:
        from update_actuals import update_actuals
        update_actuals()
    elif args.auto:
        from run_forecast import run_forecast
        from update_actuals import update_actuals
        run_forecast()
        update_actuals()
    elif args.stats:
        _stats_summary()
    else:
        _menu()


if __name__ == "__main__":
    main()
