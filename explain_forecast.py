"""Trading Co-Pilot — LLM signal explanations via TimeCopilot agent."""

import os
import json
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
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


def fetch_price_data(ticker: str, years: int = 2) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=365 * years)
    raw   = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if raw.empty:
        raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df["unique_id"] = ticker
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None).dt.normalize()
    return df.dropna().sort_values("ds").reset_index(drop=True)


def generate_all_explanations():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_key_here":
        print("No OpenAI API key set — skipping agent explanations.")
        _write_empty_signals()
        return

    try:
        from timecopilot import TimeCopilot
    except ImportError as e:
        print(f"TimeCopilot agent import failed: {e}")
        _write_empty_signals()
        return

    tc      = TimeCopilot(llm="openai:gpt-4o")
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]
    signals = {}

    for ticker in tickers:
        name = ASSET_NAMES.get(ticker, ticker)
        print(f"[{ticker}] Generating explanation…")
        try:
            df     = fetch_price_data(ticker)
            result = tc.forecast(
                df=df,
                query=(
                    f"This is {name} ({ticker}). "
                    f"What is the directional outlook over the next 10 trading days? "
                    f"Is the signal bullish, bearish, or neutral and why? "
                    f"What are the key risks to this forecast? "
                    f"Answer in 3 concise sentences maximum. "
                    f"Start directly with the outlook — no preamble."
                ),
            )
            explanation = result.output.user_query_response
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


def _write_empty_signals():
    tickers = [t.strip() for t in os.getenv("ASSET_TICKERS", "").split(",") if t.strip()]
    signals = {
        t: {
            "ticker":       t,
            "name":         ASSET_NAMES.get(t, t),
            "explanation":  "Add OPENAI_API_KEY to .env to enable AI explanations.",
            "generated_at": datetime.now().isoformat(),
        }
        for t in tickers
    }
    with open("signals.json", "w") as f:
        json.dump(signals, f, indent=2)


if __name__ == "__main__":
    generate_all_explanations()
