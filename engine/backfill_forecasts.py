"""Walk-forward backfill: re-run current model on past trading days.

For each historical forecast_date, truncates price history to what was
available on that day, runs StatsForecast ensemble + EMA trend signal,
and saves 5d/30d/90d rows. Existing rows are skipped unless force=True.
After adding rows, triggers grading so any matured forecasts get scored.
"""

import os
import pandas as pd
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "data/forecasts.csv"
HORIZONS = [5, 30, 90]


def backfill_forecasts(days_back=14, force=False, replace_ungraded=False):
    """Re-run current signal model on past trading days.

    replace_ungraded=True: drops rows where direction_correct is empty
    (predictions not yet evaluated) and regenerates them with the current
    code. Preserves all graded rows. Use this after signal rule changes to
    update forward-looking predictions without losing historical accuracy data.

    force=True: clears ALL non-today rows and regenerates (destructive).
    """
    from engine.universe import ALL_TICKERS
    from engine.run_forecast import (
        fetch_price_data, _sf_forecast, _extract_quantiles,
        compute_signals, _freq_for, _bday_h_idx,
        _compute_trend_signal, _compute_tech_score,
        _load_macro_signals, _macro_score,
        _load_claude_signals, _claude_score,
        _load_cot, _load_pcr, _load_ng_storage,
    )
    macro          = _load_macro_signals()
    claude_signals = _load_claude_signals()
    cot            = _load_cot()
    pcr            = _load_pcr()
    ng_storage     = _load_ng_storage()

    tickers = [
        t.strip()
        for t in os.getenv("ASSET_TICKERS", ",".join(ALL_TICKERS)).split(",")
        if t.strip()
    ]

    today = date.today()
    forecast_dates = []
    d = today - timedelta(days=1)
    while len(forecast_dates) < days_back:
        if d.weekday() < 5:
            forecast_dates.append(d)
        d -= timedelta(days=1)
    forecast_dates.reverse()

    existing = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

    if force and not existing.empty:
        today_str = str(today)
        keep = existing[existing["forecast_date"].astype(str) == today_str]
        keep.to_csv(CSV_PATH, index=False)
        print(f"[force] Cleared {len(existing) - len(keep)} historical rows, kept {len(keep)} today's forecasts.")
        existing = keep

    elif replace_ungraded and not existing.empty:
        # Drop rows with empty direction_correct whose forecast_date is within our backfill window
        fd_strs = {str(fd) for fd in forecast_dates}
        dc = existing["direction_correct"] if "direction_correct" in existing.columns else pd.Series("", index=existing.index)
        is_ungraded = dc.astype(str).isin(["", "nan", "None"])
        in_window   = existing["forecast_date"].astype(str).isin(fd_strs)
        drop_mask   = is_ungraded & in_window
        n_drop = drop_mask.sum()
        existing = existing[~drop_mask].reset_index(drop=True)
        existing.to_csv(CSV_PATH, index=False)
        print(f"[replace_ungraded] Dropped {n_drop} ungraded rows in window; {len(existing)} rows kept.")

    if not existing.empty:
        existing_keys = set(
            zip(
                existing["ticker"].astype(str),
                existing["forecast_date"].astype(str),
                existing["horizon"].astype(str),
            )
        )
    else:
        existing_keys = set()

    new_rows = []

    for fd in forecast_dates:
        fd_str = str(fd)
        print(f"\n=== Backfilling {fd_str} ===")

        for ticker in tickers:
            if all((ticker, fd_str, str(h)) in existing_keys for h in HORIZONS):
                continue
            try:
                df_full = fetch_price_data(ticker, years=3)
                df_trunc = df_full[df_full["ds"].dt.date <= fd].copy().reset_index(drop=True)
                if len(df_trunc) < 60:
                    print(f"  [{ticker}] skip — only {len(df_trunc)} rows up to {fd_str}")
                    continue

                last_price = float(df_trunc["y"].iloc[-1])
                freq = _freq_for(ticker)
                fcst = _sf_forecast(df_trunc, freq, max(HORIZONS))
                p50_vals, p10_vals, p90_vals = _extract_quantiles(fcst.reset_index(drop=True))
                trend_signal = _compute_trend_signal(df_trunc["y"].values)
                macro_sc     = _macro_score(ticker, macro, cot, pcr, ng_storage)
                claude_sc    = _claude_score(ticker, claude_signals)

                for horizon in HORIZONS:
                    if (ticker, fd_str, str(horizon)) in existing_keys:
                        continue
                    target_date = pd.bdate_range(
                        start=fd + timedelta(days=1), periods=horizon
                    )[-1].date() if freq == "B" else fd + timedelta(days=horizon)
                    if freq == "B":
                        h_idx = _bday_h_idx(fd, target_date, len(p50_vals) - 1)
                    else:
                        h_idx = min(horizon - 1, len(p50_vals) - 1)
                    p50_h = round(float(p50_vals[h_idx]), 6)
                    p10_h = round(float(p10_vals[h_idx]), 6)
                    p90_h = round(float(p90_vals[h_idx]), 6)
                    p50_d1 = float(p50_vals[0])
                    p10_d1 = float(p10_vals[0])
                    p90_d1 = float(p90_vals[0])
                    direction, strength, conviction = compute_signals(
                        p10_d1, p50_d1, p50_h, p90_d1, last_price, horizon,
                        trend_signal=trend_signal,
                        tech_score=_compute_tech_score(df_trunc["y"].values, horizon),
                        macro_score=macro_sc,
                        claude_score=claude_sc,
                        ticker=ticker,
                        macro=macro,
                    )
                    new_rows.append({
                        "forecast_date": fd_str,
                        "target_date":   str(target_date),
                        "ticker":        ticker,
                        "horizon":       horizon,
                        "last_price":    round(last_price, 6),
                        "p10": p10_h, "p50": p50_h, "p90": p90_h,
                        "actual": "",
                        "model_used":       f"StatsForecast_{freq}_backfill",
                        "direction":        direction,
                        "signal_strength":  strength,
                        "conviction_score": conviction,
                        "macro_sc":         round(macro_sc, 4),
                        "cot_signal":       round(float(cot.get(ticker, 0.0)), 4),
                        "pcr_signal":       round(pcr, 4),
                        "poly_signal": "", "poly_regime": "", "poly_confidence": "",
                        "poly_alignment": "", "poly_band_adj_pct": "",
                        "news_signal": "", "news_confidence": "", "news_top_headline": "",
                        "tft_score_raw": "", "ng_storage_signal": "", "news_sc": "",
                        "error_abs": "", "error_pct": "", "hit": "",
                        "direction_correct": "", "graded_at": "", "notes": "",
                    })

                print(f"  [{ticker}] ✓ {fd_str} last={last_price:.4f} freq={freq} trend={trend_signal}")

            except Exception as e:
                print(f"  [{ticker}] ✗ {e}")

    if not new_rows:
        print("\nNo new backfill rows.")
    else:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined.sort_values(["forecast_date", "ticker", "horizon"]).to_csv(CSV_PATH, index=False)
        print(f"\n✓ Added {len(new_rows)} backfill rows.")

    # Grade any matured rows immediately
    print("\n--- Grading matured forecasts ---")
    from engine.update_actuals import fill_actuals_and_grade
    fill_actuals_and_grade()

    # Retrain direction model on updated history
    print("\n--- Retraining direction model ---")
    from engine.train_direction_model import train_direction_model
    train_direction_model()


if __name__ == "__main__":
    backfill_forecasts()
