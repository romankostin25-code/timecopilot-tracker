# TimeCopilot Accuracy Tracker

Track how well TimeCopilot's probabilistic forecasts hold up against real market data.

---

## Quick Start

### 1. Set up the environment

```bash
cd timecopilot-tracker
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install timecopilot yfinance pandas numpy python-dotenv schedule statsforecast lightgbm
```

> **Note:** The first run of Chronos-style models downloads weights (~1 GB). This is normal and only happens once.

### 2. Configure `.env`

Open `.env` and fill in your values:

| Variable | Description | Example |
|---|---|---|
| `LLM_API_KEY` | OpenAI / Anthropic key (optional — only needed for LLM agent mode) | `sk-...` |
| `LLM_PROVIDER` | Provider for LLM agent mode | `openai` |
| `LLM_MODEL` | Model for LLM agent mode | `gpt-4o` |
| `ASSET_TICKER` | Yahoo Finance ticker symbol | `SPY`, `AAPL`, `BTC-USD` |
| `FORECAST_HORIZON` | How many steps ahead to forecast | `10` |
| `FORECAST_FREQUENCY` | Frequency: `D` = daily, `B` = business days, `W` = weekly | `D` |

If you leave `LLM_API_KEY` as `your_api_key_here`, the system will run statistical/ML models only (no LLM agent) — **this works fine for getting started**.

### 3. Run

```bash
python main.py              # interactive menu
python main.py --forecast   # run forecast immediately
python main.py --update     # fill in past actuals from Yahoo Finance
python main.py --auto       # forecast + update actuals (good for cron)
python main.py --stats      # print accuracy summary
```

---

## Dashboard

Open `dashboard.html` in any browser (no server needed).

1. Drop your `forecasts.csv` onto the page (or click to select it).
2. The chart and table populate instantly — all processing is local.
3. Use **Manual Actual Entry** if auto-update misses a date.
4. **Export CSV** saves the enriched data with status labels.

---

## Automated Daily Runs

```bash
# Mac / Linux — activate venv and start scheduler
bash run_scheduler.sh

# Windows
run_scheduler.bat
```

The scheduler runs `--auto` every weekday at 18:00 local time (after US market close) and logs to `logs/scheduler.log`.

To run at system startup on Mac, create a launchd plist or add to crontab:
```
0 18 * * 1-5 /path/to/timecopilot-tracker/run_scheduler.sh
```

---

## File Overview

| File | Purpose |
|---|---|
| `.env` | Configuration (API keys, ticker, horizon) |
| `main.py` | Entry point — menu + CLI flags |
| `data_fetcher.py` | Downloads historical closes from Yahoo Finance |
| `run_forecast.py` | Runs ensemble models, appends to `forecasts.csv` |
| `update_actuals.py` | Back-fills actual prices for past target dates |
| `scheduler.py` | Weekday 18:00 cron using the `schedule` library |
| `dashboard.html` | Standalone browser dashboard |
| `forecasts.csv` | Auto-created; the source of truth for all forecasts |
| `logs/scheduler.log` | Scheduler run log |

---

## Calibration Score Explained

A well-calibrated probabilistic forecast should have its P10–P90 interval contain the actual value **~80% of the time**.

- **≥ 70%** — good calibration (green)
- **50–70%** — moderate (amber)
- **< 50%** — overconfident or biased (red)

The dashboard tracks this automatically as actuals come in.
