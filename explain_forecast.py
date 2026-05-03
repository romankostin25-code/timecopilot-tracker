"""Trading Co-Pilot — LLM signal explanations via OpenAI."""

import os
import json
import warnings
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

ASSET_NAMES = {
    "GC=F":      "Gold Futures",
    "SI=F":      "Silver Futures",
    "CL=F":      "WTI Crude Oil",
    "NG=F":      "Natural Gas",
    "HG=F":      "Copper Futures",
    "EURUSD=X":  "Euro / US Dollar",
    "GBPUSD=X":  "British Pound / US Dollar",
    "USDJPY=X":  "US Dollar / Japanese Yen",
    "AUDUSD=X":  "Australian Dollar / US Dollar",
    "USDCHF=X":  "US Dollar / Swiss Franc",
    "DX-Y.NYB":  "US Dollar Index (DXY)",
    "^TNX":      "US 10-Year Treasury Yield",
    "^GSPC":     "S&P 500",
}


def _load_context() -> dict:
    ctx = {}
    try:
        macro = pd.read_csv("macro_context.csv").sort_values("date").iloc[-1]
        ctx["vix"]          = macro.get("vix", "N/A")
        ctx["risk_regime"]  = macro.get("risk_regime", "N/A")
        ctx["dollar_regime"]= macro.get("dollar_regime", "N/A")
        ctx["rate_regime"]  = macro.get("rate_regime", "N/A")
        ctx["us10y"]        = macro.get("us10y", "N/A")
        ctx["dxy"]          = macro.get("dxy", "N/A")
    except Exception:
        pass

    try:
        poly = pd.read_csv("polymarket_data.csv")
        ctx["polymarket"] = {
            row["ticker"]: {
                "signal":      row.get("poly_signal", "N/A"),
                "prob_bullish": row.get("poly_prob_bullish", "N/A"),
            }
            for _, row in poly.iterrows()
        }
    except Exception:
        ctx["polymarket"] = {}

    try:
        curve = pd.read_csv("futures_curve.csv")
        latest = curve.sort_values("date").groupby("ticker").last().reset_index()
        ctx["curve"] = {
            row["ticker"]: row.get("regime", "N/A")
            for _, row in latest.iterrows()
        }
    except Exception:
        ctx["curve"] = {}

    return ctx


def _build_prompt(ticker: str, name: str, forecasts: pd.DataFrame, ctx: dict) -> str:
    rows = forecasts[forecasts["ticker"] == ticker].sort_values("target_date").head(5)
    if rows.empty:
        return ""

    today_row = rows.iloc[0]
    direction   = today_row.get("direction", "N/A")
    conviction  = today_row.get("conviction_score", "N/A")
    p10         = today_row.get("p10", "N/A")
    p50         = today_row.get("p50", "N/A")
    p90         = today_row.get("p90", "N/A")
    p50_d5      = rows.iloc[min(4, len(rows)-1)].get("p50", "N/A")

    poly_info   = ctx.get("polymarket", {}).get(ticker, {})
    curve_regime= ctx.get("curve", {}).get(ticker, "N/A")

    prompt = f"""You are a professional macro trading analyst. Provide a concise signal summary for {name} ({ticker}).

FORECAST DATA (as of today):
- Direction: {direction}
- Conviction score: {conviction} (0=low, 1=high)
- Day 1 range: P10={p10}, P50={p50}, P90={p90}
- Day 5 P50 target: {p50_d5}

MACRO CONTEXT:
- VIX: {ctx.get('vix', 'N/A')} | Risk regime: {ctx.get('risk_regime', 'N/A')}
- Dollar regime: {ctx.get('dollar_regime', 'N/A')} | Rate regime: {ctx.get('rate_regime', 'N/A')}
- US 10Y: {ctx.get('us10y', 'N/A')} | DXY: {ctx.get('dxy', 'N/A')}

POLYMARKET SIGNAL: {poly_info.get('signal', 'N/A')} (crowd bullish prob: {poly_info.get('prob_bullish', 'N/A')})
FUTURES CURVE REGIME: {curve_regime}

Write exactly 3 sentences:
1. State the directional outlook and whether the model + Polymarket agree or diverge.
2. Explain the key macro driver(s) supporting or opposing this view right now.
3. Name the single biggest risk that could invalidate this forecast.

No preamble. No headers. Start directly with the outlook."""
    return prompt


def generate_all_explanations():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_key_here":
        print("No OpenAI API key set — skipping agent explanations.")
        _write_placeholder_signals()
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("openai package not installed.")
        _write_placeholder_signals()
        return

    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]

    try:
        forecasts = pd.read_csv("forecasts.csv")
        today_str = str(forecasts["forecast_date"].max())
        forecasts = forecasts[forecasts["forecast_date"].astype(str) == today_str]
    except Exception:
        forecasts = pd.DataFrame()

    ctx = _load_context()
    signals = {}

    for ticker in tickers:
        name = ASSET_NAMES.get(ticker, ticker)
        print(f"[{ticker}] Generating explanation…")
        try:
            prompt = _build_prompt(ticker, name, forecasts, ctx)
            if not prompt:
                raise ValueError("No forecast data available")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4,
            )
            explanation = response.choices[0].message.content.strip()
            print(f"[{ticker}] ✓")
        except Exception as e:
            print(f"[{ticker}] ✗ {e}")
            explanation = "Explanation unavailable."

        signals[ticker] = {
            "ticker":       ticker,
            "name":         name,
            "explanation":  explanation,
            "generated_at": datetime.now().isoformat(),
        }

    with open("signals.json", "w") as f:
        json.dump(signals, f, indent=2)
    print("✓ signals.json saved.")


def _write_placeholder_signals():
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]
    signals = {
        t: {
            "ticker":       t,
            "name":         ASSET_NAMES.get(t, t),
            "explanation":  "Add OPENAI_API_KEY to enable AI explanations.",
            "generated_at": datetime.now().isoformat(),
        }
        for t in tickers
    }
    with open("signals.json", "w") as f:
        json.dump(signals, f, indent=2)


if __name__ == "__main__":
    generate_all_explanations()
