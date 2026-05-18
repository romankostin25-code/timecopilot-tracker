"""Claude-powered agentic signal analysis and forecast explanation."""

import os
import json
import anthropic
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SIGNALS_PATH = "data/signals.json"

SYSTEM_PROMPT = """You are a senior macro trading analyst at a quant hedge fund.
Your role is to produce deep, structured analysis of trading signals — not surface-level summaries.
You think through cross-asset dynamics, regime interactions, and tail risks before writing.
Return ONLY valid JSON matching the exact schema requested. No preamble, no markdown fences."""


def _load_context() -> dict:
    ctx = {}
    try:
        macro = pd.read_csv("data/macro_context.csv").sort_values("date").iloc[-1]
        for k in ["vix", "us10y", "dxy", "risk_regime", "dollar_regime", "rate_regime",
                  "fed_funds_rate", "cpi_yoy_pct", "unemployment_rate"]:
            ctx[k] = macro.get(k, "N/A")
    except Exception:
        pass

    try:
        curve = pd.read_csv("data/futures_curve.csv")
        latest = curve.sort_values("date").groupby("ticker").last().reset_index()
        ctx["curve"] = {r["ticker"]: r.get("regime", "N/A") for _, r in latest.iterrows()}
    except Exception:
        ctx["curve"] = {}

    try:
        with open("data/poly_regimes.json") as f:
            pr = json.load(f)
        ctx["active_regimes"] = list(pr.get("active_regimes", {}).keys())[:3]
        ctx["poly_asset_priors"] = pr.get("asset_priors", {})
    except Exception:
        ctx["active_regimes"] = []
        ctx["poly_asset_priors"] = {}

    try:
        with open("data/intelligence_feed.json") as f:
            feed = json.load(f)
        ctx["recent_headlines"] = [
            {"headline": a["headline"], "sentiment": a.get("sentiment"), "assets": a.get("assets_affected", [])}
            for a in feed[:10] if a.get("nlp_processed")
        ]
    except Exception:
        ctx["recent_headlines"] = []

    return ctx


def _build_prompt(ticker: str, name: str, forecasts: pd.DataFrame, ctx: dict) -> str:
    rows = forecasts[forecasts["ticker"] == ticker].sort_values("horizon")
    if rows.empty:
        return ""

    horizons_data = {}
    for _, row in rows.iterrows():
        h = int(row.get("horizon", 5))
        horizons_data[h] = {
            "direction":       row.get("direction", "N/A"),
            "conviction":      row.get("conviction_score", "N/A"),
            "p10":             row.get("p10", "N/A"),
            "p50":             row.get("p50", "N/A"),
            "p90":             row.get("p90", "N/A"),
            "signal_strength": row.get("signal_strength", "N/A"),
        }

    poly_prior = ctx.get("poly_asset_priors", {}).get(ticker, {})
    relevant_news = [
        h for h in ctx.get("recent_headlines", [])
        if ticker in h.get("assets", [])
    ][:3]

    return f"""Analyze {name} ({ticker}) given the following data.

MULTI-HORIZON FORECASTS:
{json.dumps(horizons_data, indent=2)}

MACRO CONTEXT:
- VIX: {ctx.get('vix', 'N/A')} | Risk regime: {ctx.get('risk_regime', 'N/A')}
- Dollar regime: {ctx.get('dollar_regime', 'N/A')} | Rate regime: {ctx.get('rate_regime', 'N/A')}
- US 10Y: {ctx.get('us10y', 'N/A')} | DXY: {ctx.get('dxy', 'N/A')}
- Fed funds rate: {ctx.get('fed_funds_rate', 'N/A')} | CPI YoY: {ctx.get('cpi_yoy_pct', 'N/A')}%
- Active Polymarket regimes: {ctx.get('active_regimes', [])}

POLYMARKET REGIME SIGNAL FOR {ticker}:
{json.dumps(poly_prior, indent=2)}

FUTURES CURVE (if applicable): {ctx.get('curve', {}).get(ticker, 'N/A')}

RECENT RELEVANT NEWS:
{json.dumps(relevant_news, indent=2)}

Return this exact JSON schema (no extra keys):
{{
  "summary": "2-sentence directional outlook covering near-term and medium-term view",
  "key_drivers": [
    {{"driver": "string", "impact": "BULLISH|BEARISH|NEUTRAL", "weight": "HIGH|MEDIUM|LOW"}}
  ],
  "scenarios": {{
    "base": {{"narrative": "string", "probability": 0.0}},
    "bull": {{"narrative": "string", "probability": 0.0, "catalyst": "string"}},
    "bear": {{"narrative": "string", "probability": 0.0, "catalyst": "string"}}
  }},
  "key_risks": ["string", "string"],
  "conviction_narrative": "1 sentence explaining why conviction is high/medium/low",
  "horizon_divergence": "null or 1 sentence if 5d and 90d outlooks conflict",
  "poly_alignment": "ALIGNED|DIVERGENT|NEUTRAL",
  "poly_alignment_note": "1 sentence on what Polymarket crowd implies vs model"
}}"""


def generate_all_explanations():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_key_here":
        print("No ANTHROPIC_API_KEY set — writing placeholders.")
        _write_placeholders()
        return

    client = anthropic.Anthropic(api_key=api_key)

    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]
    if not tickers:
        from engine.universe import ALL_TICKERS
        tickers = ALL_TICKERS

    try:
        forecasts = pd.read_csv("data/forecasts.csv")
        today_str = str(forecasts["forecast_date"].max())
        forecasts = forecasts[forecasts["forecast_date"].astype(str) == today_str]
    except Exception:
        forecasts = pd.DataFrame()

    from engine.universe import UNIVERSE
    ctx = _load_context()
    signals = {}

    for ticker in tickers:
        name = UNIVERSE.get(ticker, {}).get("name", ticker)
        print(f"[{ticker}] Generating analysis…")
        try:
            prompt = _build_prompt(ticker, name, forecasts, ctx)
            if not prompt:
                raise ValueError("No forecast data")

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            analysis = json.loads(raw)
            print(f"[{ticker}] ✓")
        except Exception as e:
            print(f"[{ticker}] ✗ {e}")
            analysis = {"summary": "Analysis unavailable.", "key_risks": [], "scenarios": {}}

        signals[ticker] = {
            "ticker":       ticker,
            "name":         name,
            "analysis":     analysis,
            # Keep flat `explanation` for backward compat with dashboard
            "explanation":  analysis.get("summary", ""),
            "generated_at": datetime.now().isoformat(),
        }

    os.makedirs("data", exist_ok=True)
    with open(SIGNALS_PATH, "w") as f:
        json.dump(signals, f, indent=2)
    print(f"✓ signals.json saved ({len(signals)} assets).")


def _write_placeholders():
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]
    from engine.universe import UNIVERSE
    signals = {
        t: {
            "ticker": t, "name": UNIVERSE.get(t, {}).get("name", t),
            "analysis": {}, "explanation": "Add ANTHROPIC_API_KEY to enable AI analysis.",
            "generated_at": datetime.now().isoformat(),
        }
        for t in tickers
    }
    os.makedirs("data", exist_ok=True)
    with open(SIGNALS_PATH, "w") as f:
        json.dump(signals, f, indent=2)


if __name__ == "__main__":
    generate_all_explanations()
