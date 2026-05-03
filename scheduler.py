"""Runs --auto mode every weekday at 18:00 US Eastern."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import schedule
import time

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Change cwd so CSV paths resolve correctly
os.chdir(Path(__file__).parent)


def run_auto():
    if datetime.today().weekday() >= 5:  # Saturday=5, Sunday=6
        log.info("Weekend — skipping run.")
        return
    log.info("Starting scheduled --auto run...")
    try:
        from run_forecast import run_forecast
        from update_actuals import update_actuals
        added = run_forecast()
        updated = update_actuals()
        log.info(f"Done — {added} new forecast rows, {updated} actuals updated.")
    except Exception as exc:
        log.error(f"Scheduled run failed: {exc}", exc_info=True)


# 18:00 local time — adjust to your timezone as needed
schedule.every().day.at("18:00").do(run_auto)

log.info("Scheduler started — will run every weekday at 18:00. Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(30)
