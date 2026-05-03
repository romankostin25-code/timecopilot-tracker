# Trading Co-Pilot — Trader's Guide

## What This System Does

Trading Co-Pilot generates 10-day probabilistic forecasts for 13 instruments across commodities, currencies, and rates. Every evening it runs a full pipeline: macro snapshot → Polymarket odds → statistical models → futures term structure → grading → LLM explanations. Every morning it grades overnight outcomes and updates the scorecard.

The output is a live dashboard you can read in under 60 seconds.

---

## Reading the Signal Panel

Each asset card in the left rail shows three things:

- **Direction badge** — `BULLISH`, `BEARISH`, or `NEUTRAL`. This is the model's directional call for the next 10 trading days based on where P50 sits relative to today's price.
- **Polymarket alignment** — `✓ ALIGNED` (model and prediction market agree) or `⚠ DIVERGENT` (they disagree). Divergence doesn't mean the model is wrong; it means there's genuine uncertainty and you should read both signals carefully.
- **Yesterday hit/miss** — whether yesterday's actual close fell inside the P10–P90 forecast band.

**Signal strength** (`LOW / MEDIUM / HIGH`) is derived from the width of the forecast band relative to recent volatility. A narrow band with a clear directional lean = HIGH conviction. A wide band = LOW conviction regardless of direction.

---

## The Accuracy Numbers

### Calibration Rate (target: 80%)
The fraction of graded forecasts where the actual price fell inside the P10–P90 interval. A well-calibrated model should hit 80% — the intervals are designed to be 80% prediction intervals. If calibration is consistently above 90%, the model is too wide (conservative). Below 60% is a warning — the model's uncertainty estimates are too narrow.

### Directional Accuracy (target: 55%)
The fraction of forecasts where the predicted direction (BULLISH/BEARISH/NEUTRAL) matched what actually happened. 50% is a coin flip. 55% over a meaningful sample is useful. Don't over-weight any single day's number — look at the 30-day rolling rate.

**Honest assessment**: these are short-horizon statistical models. They have no alpha on macro surprises, earnings, or geopolitical events. They're useful for mean-reversion signals and regime confirmation — not for predicting black swans.

---

## Futures Curves (Curve Tab)

The curve tab shows the term structure for the five futures markets (CL, GC, NG, SI, HG).

- **BACKWARDATION** (green) — front month trades above back months. Historically associated with physical tightness and bullish spot conditions. For oil and gas, backwardation often signals supply stress.
- **CONTANGO** (red) — front month below back months. Normal carry structure when storage is available. Negative roll yield for long holders.
- **FLAT** — less than 0.5% spread between front and second contract.

**Roll yield** shown is annualized. A deeply negative roll yield in contango is a drag on any long position held through the roll date.

---

## Polymarket Data (Polymarket Tab)

Polymarket is a prediction market where traders bet real money on outcomes. The probabilities here are crowd-sourced price signals — not model outputs.

- **Poly Bullish Prob** — the aggregated probability from Polymarket contracts related to this asset being higher over the relevant horizon.
- **Volume** — only markets above $50,000 volume are included. Low volume markets are noisier.
- **Alignment** — when Polymarket and the model agree directionally, confidence is higher. When they diverge, dig into why.

Note: Polymarket coverage is uneven. Some assets (US equities, oil, major FX) have liquid markets. Others (silver, copper) may have thin or no relevant markets — those will show "No data."

---

## AI Signal Explanations

The FORECAST tab includes an AI-generated plain-language summary for each asset. This is produced by GPT-4o reading the model's P10/P50/P90 path, the current macro regime, futures curve structure, and Polymarket signal together.

These explanations are contextual synthesis, not trading advice. Read them as "here is what the data says" — apply your own judgment about whether the macro regime call matches your view.

---

## Macro Regime Indicators (Header Row)

| Badge | Meaning |
|-------|---------|
| `RISK_OFF` | VIX > 25. Elevated fear. Historically negative for risk assets, positive for gold/JPY/treasuries. |
| `RISK_ON` | VIX < 15. Low fear. Favorable for equities, commodities, high-beta FX. |
| `NEUTRAL` | VIX 15–25. No strong regime signal. |
| `DOLLAR_STRENGTH` | DXY 5-day change > +0.5%. Watch commodity headwinds. |
| `DOLLAR_WEAKNESS` | DXY 5-day change < -0.5%. Typically commodity-supportive. |
| `HAWKISH` | US 10Y > 4.5%. Rate pressure on risk assets. |
| `DOVISH` | US 10Y < 3.5%. Rate support for growth and duration. |

---

## Honest Limitations

1. **Models don't know about scheduled events.** FOMC announcements, NFP, major options expiries — the models have no special handling for these. Check your economic calendar.

2. **10-day horizon is where statistical models start degrading.** Days 1–3 are more reliable than days 8–10. The P10–P90 bands widen appropriately, but the point forecast (P50) should be treated with more skepticism as horizon extends.

3. **Polymarket coverage is patchy.** Don't treat missing Polymarket data as a neutral signal — it may just mean no relevant contract exists.

4. **Calibration over small samples is noisy.** The 7-day calibration rate can swing wildly. The 30-day rate is more meaningful. All-time is most meaningful.

5. **Futures curve front-month contracts are updated monthly.** If you see obviously wrong curve data near a contract roll date, the hardcoded contract codes may need updating in `futures_curve.py`.

6. **This is not financial advice.** The system is a signal aggregator and probabilistic scenario tool. Use it to structure your thinking, not to replace it.

---

## Pipeline Schedule

| Time (ET) | Action |
|-----------|--------|
| 10:00 AM Mon–Fri | Grade yesterday's forecasts, refresh macro + Polymarket |
| 6:00 PM Mon–Fri | Full run: new macro snapshot → Polymarket → 10-day forecasts → curve → grade → AI explanations |

Dashboard auto-refreshes every 5 minutes. All data is served from static CSV/JSON files committed to the repo.

---

## Data Files

| File | Contents |
|------|----------|
| `forecasts.csv` | All forecasts with actuals, hit/miss, direction grades |
| `scorecard.json` | Rolling accuracy rates (7d/30d/all-time) per asset |
| `signals.json` | LLM-generated plain-language signal per asset |
| `macro_context.csv` | Daily macro snapshot (VIX, DXY, 10Y, SP500, Gold, Oil + FRED) |
| `polymarket_data.csv` | Polymarket signals per asset with volume and top question |
| `futures_curve.csv` | Term structure data with contango/backwardation regime |
