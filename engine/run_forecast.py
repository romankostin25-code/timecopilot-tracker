"""Multi-horizon forecasting engine (5d / 30d / 90d)."""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

HORIZONS = [int(h) for h in os.getenv("FORECAST_HORIZONS", "5,30,90").split(",")]
FREQ = os.getenv("FORECAST_FREQUENCY", "B")
MAX_HORIZON = max(HORIZONS)
CSV_PATH = "data/forecasts.csv"

_DIR_MODEL_CACHE: dict | None = None
_DIR_MODEL_LOADED = False


def _load_direction_model() -> dict:
    global _DIR_MODEL_CACHE, _DIR_MODEL_LOADED
    if _DIR_MODEL_LOADED:
        return _DIR_MODEL_CACHE or {}
    _DIR_MODEL_LOADED = True
    try:
        import joblib
        payload = joblib.load("data/direction_model.pkl")
        _DIR_MODEL_CACHE = payload.get("models", {})
        trained = [h for h, m in _DIR_MODEL_CACHE.items() if m is not None]
        print(f"[forecaster] direction_model.pkl loaded — trained horizons: {trained}")
    except Exception as e:
        print(f"[forecaster] direction_model.pkl not available ({e}) — using fallback weights")
        _DIR_MODEL_CACHE = {}
    return _DIR_MODEL_CACHE


def fetch_price_data(ticker, years=3):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    period_map = {"DX-Y.NYB": "3y", "^TNX": "3y", "^IRX": "3y", "^VIX": "3y"}
    if ticker in period_map:
        raw = yf.download(ticker, period=period_map[ticker], auto_adjust=True, progress=False)
    else:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    close = raw["Close"].squeeze()
    df = close.reset_index()
    df.columns = ["ds", "y"]
    df["unique_id"] = ticker
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows")
    return df


CRYPTO_TICKERS  = {"BTC-USD", "ETH-USD", "SOL-USD"}
EQUITY_ETFS     = {"SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "EEM", "EFA", "FXI"}
BOND_ETFS       = {"TLT", "IEF", "HYG", "LQD", "BIL"}
COMMODITY_TICKS = {"GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F", "GLD", "SLV"}

def _freq_for(ticker):
    return "D" if ticker in CRYPTO_TICKERS else FREQ


def _ensure_regular_freq(df, freq):
    """Reindex to a gapless frequency so TimeCopilot can infer the interval.

    yfinance data has holiday gaps on business-day series which cause
    TimeCopilot's frequency detector to fail. Forward-filling to a complete
    date range removes those gaps without distorting the series.
    """
    ts = df[["ds", "y", "unique_id"]].copy()
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.set_index("ds").sort_index()
    if freq == "B":
        idx = pd.bdate_range(ts.index.min(), ts.index.max())
    elif freq == "D":
        idx = pd.date_range(ts.index.min(), ts.index.max(), freq="D")
    else:
        return df
    ts = ts.reindex(idx).ffill().dropna()
    ts.index.name = "ds"
    out = ts.reset_index()[["ds", "y", "unique_id"]]
    out["ds"] = pd.to_datetime(out["ds"]).dt.tz_localize(None).dt.normalize()
    return out


def _build_tc_forecaster():
    """Try TimeCopilot. Returns (forecaster, name) or (None, None)."""
    try:
        from timecopilot import TimeCopilotForecaster
        from timecopilot.models.stats import AutoARIMA, AutoETS

        # AutoLGBM excluded: known LightGBM "feature index -1" bug in this env
        models = [AutoARIMA(), AutoETS()]

        for cls_path, label in [
            ("timecopilot.models.foundation.chronos.Chronos", "Chronos"),
            ("timecopilot.models.foundation.toto.Toto",       "Toto"),
        ]:
            try:
                mod_name, cls_name = cls_path.rsplit(".", 1)
                import importlib
                models.insert(0, getattr(importlib.import_module(mod_name), cls_name)())
                print(f"[forecaster] {label} loaded")
            except Exception as e:
                print(f"[forecaster] {label} unavailable: {e}")

        loaded = [type(m).__name__ for m in models]
        print(f"[forecaster] TimeCopilot ready with: {loaded}")
        return TimeCopilotForecaster(models=models), "TimeCopilot_Ensemble"
    except Exception as e:
        print(f"[forecaster] TimeCopilot unavailable: {e}")
        return None, None


def _sf_forecast(df, freq, h):
    """StatsForecast ensemble: median of AutoARIMA + AutoETS + AutoTheta.
    Uses 68% CI (±1σ) for tighter, more actionable forecast bands.
    Returns DataFrame with columns: unique_id, ds, p10, p50, p90."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    sf = StatsForecast(
        models=[AutoARIMA(), AutoETS(), AutoTheta()],
        freq=freq,
        n_jobs=1,
    )
    raw = sf.forecast(df=df, h=h, level=[68]).reset_index(drop=True)
    lo_cols = [c for c in raw.columns if "-lo-68" in c]
    hi_cols = [c for c in raw.columns if "-hi-68" in c]
    pt_cols = [c for c in raw.columns if c not in ("unique_id", "ds")
               and "-lo-" not in c and "-hi-" not in c]
    base = raw[["unique_id", "ds"]] if "unique_id" in raw.columns else raw[["ds"]]
    out = base.copy()
    out["p50"] = raw[pt_cols].median(axis=1) if pt_cols else raw.iloc[:, 2]
    out["p10"] = raw[lo_cols].median(axis=1) if lo_cols else out["p50"] * 0.99
    out["p90"] = raw[hi_cols].median(axis=1) if hi_cols else out["p50"] * 1.01
    return out


def _compute_trend_signal(price_arr):
    """EMA10/EMA30 crossover — 'BULLISH' if short EMA above long, else 'BEARISH'."""
    if len(price_arr) < 30:
        return None
    s = pd.Series(price_arr, dtype=float)
    ema10 = s.ewm(span=10, adjust=False).mean().iloc[-1]
    ema30 = s.ewm(span=30, adjust=False).mean().iloc[-1]
    return "BULLISH" if ema10 > ema30 else "BEARISH"


def _compute_tech_score(price_arr, horizon):
    """RSI + momentum score in [-1, 1]. Horizon-weighted."""
    s = pd.Series(price_arr, dtype=float)
    n = len(s)
    signals = {}

    # RSI-14: overbought/oversold
    if n >= 15:
        delta = s.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).iloc[-1]
        if rsi > 70:
            signals["rsi"] = -0.8
        elif rsi < 30:
            signals["rsi"] = 0.8
        else:
            signals["rsi"] = (50 - rsi) / 50 * 0.3

    # 5-day momentum
    if n >= 5:
        mom5 = (s.iloc[-1] / s.iloc[-5] - 1)
        signals["mom5"] = float(np.clip(mom5 / 0.03, -1.0, 1.0))

    # 20-day momentum
    if n >= 20:
        mom20 = (s.iloc[-1] / s.iloc[-20] - 1)
        signals["mom20"] = float(np.clip(mom20 / 0.10, -1.0, 1.0))

    if not signals:
        return 0.0

    if horizon == 5:
        w = {"rsi": 0.30, "mom5": 0.50, "mom20": 0.20}
    elif horizon == 30:
        w = {"rsi": 0.20, "mom5": 0.20, "mom20": 0.60}
    else:
        w = {"rsi": 0.10, "mom5": 0.10, "mom20": 0.80}

    return float(np.clip(sum(w.get(k, 0) * v for k, v in signals.items()), -1.0, 1.0))


def _load_macro_signals():
    """Load latest macro context row. Returns dict or {}."""
    try:
        df = pd.read_csv("data/macro_context.csv")
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            "vix":          float(row.get("vix", 20)),
            "risk_regime":  str(row.get("risk_regime",  "NEUTRAL")),
            "dollar_regime":str(row.get("dollar_regime","NEUTRAL")),
            "rate_regime":  str(row.get("rate_regime",  "NEUTRAL")),
        }
    except Exception:
        return {}


def _macro_score(ticker, macro):
    """Asset-class-aware macro signal in [-1, 1]."""
    if not macro:
        return 0.0
    vix    = macro.get("vix", 20)
    risk   = macro.get("risk_regime",   "NEUTRAL")
    dollar = macro.get("dollar_regime", "NEUTRAL")
    rate   = macro.get("rate_regime",   "NEUTRAL")
    score  = 0.0

    if ticker in EQUITY_ETFS:
        score += 0.25 if vix < 15 else (0.10 if vix < 20 else (-0.15 if vix > 25 else (-0.30 if vix > 30 else 0.0)))
        score += 0.20 if risk == "RISK_ON" else (-0.20 if risk == "RISK_OFF" else 0.0)
        score -= 0.10 if rate == "HAWKISH" else (-0.10 if rate in ("DOVISH", "EASING") else 0.0)
    elif ticker in BOND_ETFS:
        score -= 0.30 if rate == "HAWKISH" else (-0.30 if rate in ("DOVISH", "EASING") else 0.0)
    elif ticker in COMMODITY_TICKS:
        score -= 0.25 if dollar == "DOLLAR_STRENGTH" else (-0.25 if dollar == "DOLLAR_WEAKNESS" else 0.0)
    elif ticker in CRYPTO_TICKERS:
        score += 0.20 if risk == "RISK_ON" else (-0.25 if risk == "RISK_OFF" else 0.0)
        score -= 0.15 if vix > 25 else 0.0
    elif ticker in ("^VIX", "UVXY"):
        # VIX/UVXY are regime-driven, not mean-reverting — weight regime signal heavily
        score += 0.60 if risk == "RISK_OFF" else (-0.60 if risk == "RISK_ON" else 0.0)
        score += 0.20 if vix > 30 else (-0.20 if vix < 15 else 0.0)
    elif ticker == "DX-Y.NYB":
        score += 0.20 if rate == "HAWKISH" else (-0.20 if rate in ("DOVISH", "EASING") else 0.0)
        score += 0.10 if risk == "RISK_OFF" else 0.0

    return float(np.clip(score, -1.0, 1.0))


def _load_claude_signals():
    """Load signals.json (Claude per-ticker analysis). Returns dict or {}."""
    try:
        import json
        return json.load(open("data/signals.json"))
    except Exception:
        return {}


def _claude_score(ticker, claude_signals):
    """Score from Claude key_drivers: HIGH=1.0, MEDIUM=0.5, LOW=0.25. Returns [-1, 1]."""
    if not claude_signals or ticker not in claude_signals:
        return 0.0
    drivers = claude_signals[ticker].get("analysis", {}).get("key_drivers", [])
    if not drivers:
        return 0.0
    weights = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}
    bull = sum(weights.get(d.get("weight", ""), 0) for d in drivers if d.get("impact") == "BULLISH")
    bear = sum(weights.get(d.get("weight", ""), 0) for d in drivers if d.get("impact") == "BEARISH")
    total = bull + bear
    return float((bull - bear) / total) if total else 0.0


def _extract_quantiles(fcst_rows):
    """Extract (p50_vals, p10_vals, p90_vals) arrays from any forecaster output DataFrame."""
    cols = [c for c in fcst_rows.columns if c not in ("unique_id", "ds")]
    # Clean p10/p50/p90 columns already present (from _sf_forecast)
    if "p50" in cols:
        p50 = fcst_rows["p50"].values
        p10 = fcst_rows["p10"].values if "p10" in cols else p50 * 0.99
        p90 = fcst_rows["p90"].values if "p90" in cols else p50 * 1.01
        return p50, p10, p90
    # Explicit quantile column detection (lo/hi bands from individual models)
    p10_col = next((c for c in cols if any(k in c.lower() for k in ["lo", "q10", "p10", "0.1"])), None)
    p50_col = next((c for c in cols if any(k in c.lower() for k in ["median", "q50", "p50", "mean", "0.5"])), None)
    p90_col = next((c for c in cols if any(k in c.lower() for k in ["hi", "q90", "p90", "0.9"])), None)
    if p50_col:
        p50 = fcst_rows[p50_col].values
        p10 = fcst_rows[p10_col].values if p10_col else p50 * 0.99
        p90 = fcst_rows[p90_col].values if p90_col else p50 * 1.01
        print(f"[forecaster] quantile cols — p10:{p10_col}  p50:{p50_col}  p90:{p90_col}")
        return p50, p10, p90
    # TimeCopilot returns individual model point forecasts as separate columns.
    # Build ensemble: P50=median, P10/P90 from model spread (mean ± 1.28σ).
    arr = fcst_rows[cols].apply(pd.to_numeric, errors="coerce").values
    p50 = np.median(arr, axis=1)
    spread = np.std(arr, axis=1)
    p10 = p50 - 1.28 * spread
    p90 = p50 + 1.28 * spread
    print(f"[forecaster] TC ensemble spread across: {cols}")
    return p50, p10, p90


def _bday_h_idx(forecast_date, target_date, max_idx):
    """0-based step index for a business-day StatsForecast output.

    Counts actual business days from forecast_date (exclusive) to the first
    trading day >= target_date (inclusive), so the correct model step is used
    regardless of whether target_date lands on a weekend or holiday.
    """
    t = pd.Timestamp(target_date)
    if t.dayofweek >= 5:  # weekend → advance to Monday
        t = t + pd.tseries.offsets.BDay(1)
    bdays = pd.bdate_range(
        start=str((pd.Timestamp(forecast_date) + timedelta(days=1)).date()),
        end=str(t.date()),
    )
    return min(max(len(bdays) - 1, 0), max_idx)


_VOL_TICKERS    = {"^VIX", "UVXY"}
_TRENDING_COMMS = {"NG=F", "ZW=F", "SI=F", "SLV", "XLE", "ZC=F"}


def compute_signals(p10_d1, p50_d1, p50_target, p90_d1, last_price, horizon,
                    trend_signal=None, tech_score=0.0, macro_score=0.0, claude_score=0.0,
                    ticker=None):
    """Direction via trained meta-learner or asset-class-specific weighted ensemble.

    Asset-class weights:
      Equities/bonds/default  — model(60%) macro(25%) claude(15%)
      Crypto                  — trend(35%) model(20%) macro(30%) claude(15%)
      Trending commodities    — trend(30%) model(30%) macro(25%) claude(15%)
      VIX/UVXY               — macro(80%) model(20%)  [ARIMA mean-reversion wrong for vol]

    If data/direction_model.pkl has a trained model for this horizon, uses it for
    the model component and applies macro/claude as additive adjustments.
    """
    forecast_return = (p50_target - last_price) / last_price
    band_width      = (p90_d1 - p10_d1) / last_price
    model_sc        = float(np.clip(forecast_return / 0.02, -1.0, 1.0))

    # Use trained meta-learner model_sc if available
    pipe = _load_direction_model().get(str(horizon))
    if pipe is not None:
        model_sc = float(pipe.predict_proba([[forecast_return, band_width]])[0][1]) * 2 - 1  # [0,1] → [-1,1]

    trend_sc = 1.0 if trend_signal == "BULLISH" else (-1.0 if trend_signal == "BEARISH" else model_sc)

    if ticker in _VOL_TICKERS:
        # VIX/UVXY: ARIMA mean-reversion is systematically wrong in trending regimes
        combined = 0.20 * model_sc + 0.80 * macro_score
    elif ticker in CRYPTO_TICKERS:
        # Crypto trends persistently — weight momentum + macro over mean-reversion
        combined = 0.20 * model_sc + 0.35 * trend_sc + 0.30 * macro_score + 0.15 * claude_score
    elif ticker in _TRENDING_COMMS:
        # Energy/ag/silver: respect the trend but still give model some weight
        combined = 0.30 * model_sc + 0.30 * trend_sc + 0.25 * macro_score + 0.15 * claude_score
    else:
        # Equities, bonds, default: mean-reversion model is most reliable
        combined = 0.60 * model_sc + 0.25 * macro_score + 0.15 * claude_score

    direction        = "BULLISH" if combined >= 0 else "BEARISH"
    signal_strength  = round(abs(forecast_return) * 100, 4)
    conviction_score = round(max(0, 1 - (band_width / 0.05)), 4)
    return direction, signal_strength, conviction_score


def run_all_forecasts():
    from engine.universe import ALL_TICKERS
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", ",".join(ALL_TICKERS)).split(",") if t.strip()]

    tc_forecaster, tc_name = _build_tc_forecaster()
    forecast_date  = datetime.today().date()
    macro          = _load_macro_signals()
    claude_signals = _load_claude_signals()
    new_rows, skipped = [], []

    for ticker in tickers:
        print(f"\n[{ticker}] Fetching data...")
        try:
            df = fetch_price_data(ticker)
            last_price = float(df["y"].iloc[-1])
            freq = _freq_for(ticker)
            print(f"[{ticker}] Forecasting {MAX_HORIZON}d ({len(df)} pts, freq={freq})...")

            if tc_forecaster is not None:
                try:
                    df_tc = _ensure_regular_freq(df, freq)
                    fcst = tc_forecaster.forecast(df=df_tc, h=MAX_HORIZON)
                    model_name = tc_name
                    print(f"[{ticker}] TimeCopilot OK — cols: {list(fcst.columns)}")
                except Exception as e:
                    print(f"[{ticker}] TimeCopilot failed: {e} — falling back to StatsForecast")
                    fcst = _sf_forecast(df, freq, MAX_HORIZON)
                    model_name = f"StatsForecast_{freq}"
            else:
                fcst = _sf_forecast(df, freq, MAX_HORIZON)
                model_name = f"StatsForecast_{freq}"

            p50_vals, p10_vals, p90_vals = _extract_quantiles(fcst.reset_index(drop=True))
            trend_signal  = _compute_trend_signal(df["y"].values)
            macro_sc      = _macro_score(ticker, macro)
            claude_sc     = _claude_score(ticker, claude_signals)
            print(f"[{ticker}] signals — ema:{trend_signal} macro:{macro_sc:+.2f} claude:{claude_sc:+.2f}")

            for horizon in HORIZONS:
                target_date = (datetime.today() + timedelta(days=horizon)).date()
                # Use actual business-day step matching the calendar target_date
                if freq == "B":
                    h_idx = _bday_h_idx(forecast_date, target_date, len(p50_vals) - 1)
                else:
                    h_idx = min(horizon - 1, len(p50_vals) - 1)
                p50_h = round(float(p50_vals[h_idx]), 6)
                p10_h = round(float(p10_vals[h_idx]), 6)
                p90_h = round(float(p90_vals[h_idx]), 6)
                p50_d1 = float(p50_vals[0])
                p10_d1 = float(p10_vals[0])
                p90_d1 = float(p90_vals[0])

                direction, signal_strength, conviction = compute_signals(
                    p10_d1, p50_d1, p50_h, p90_d1, last_price, horizon,
                    trend_signal=trend_signal,
                    tech_score=_compute_tech_score(df["y"].values, horizon),
                    macro_score=macro_sc,
                    claude_score=claude_sc,
                    ticker=ticker,
                )
                new_rows.append({
                    "forecast_date": str(forecast_date),
                    "target_date":   str(target_date),
                    "ticker":        ticker,
                    "horizon":       horizon,
                    "last_price":    round(last_price, 6),
                    "p10": p10_h, "p50": p50_h, "p90": p90_h,
                    "actual": "",
                    "model_used":       model_name,
                    "direction":        direction,
                    "signal_strength":  signal_strength,
                    "conviction_score": conviction,
                    "poly_signal": "", "poly_regime": "", "poly_confidence": "",
                    "poly_alignment": "", "poly_band_adj_pct": "",
                    "news_signal": "", "news_confidence": "", "news_top_headline": "",
                    "error_abs": "", "error_pct": "", "hit": "",
                    "direction_correct": "", "graded_at": "", "notes": "",
                })

            print(f"[{ticker}] ✓ {len(HORIZONS)} horizons | last={last_price:.4f} | {model_name}")

        except Exception as e:
            print(f"[{ticker}] ✗ {e}")
            skipped.append(ticker)

    os.makedirs("data", exist_ok=True)
    existing = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    new_df = pd.DataFrame(new_rows)

    if not existing.empty and not new_df.empty:
        for col in ["forecast_date", "target_date"]:
            existing[col] = existing[col].astype(str)
            new_df[col] = new_df[col].astype(str)
        key_cols = ["ticker", "forecast_date", "target_date", "horizon"]
        if all(c in existing.columns for c in key_cols) and all(c in new_df.columns for c in key_cols):
            existing_keys = set(
                zip(existing["ticker"], existing["forecast_date"].astype(str),
                    existing["target_date"].astype(str), existing["horizon"].astype(str))
            )
            new_df = new_df[new_df.apply(
                lambda r: (r["ticker"], str(r["forecast_date"]), str(r["target_date"]), str(r["horizon"]))
                not in existing_keys, axis=1
            )]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"\n✓ {len(new_df)} new rows. {len(skipped)} skipped: {skipped}")


if __name__ == "__main__":
    run_all_forecasts()
