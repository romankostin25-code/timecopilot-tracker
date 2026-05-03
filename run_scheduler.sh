#!/usr/bin/env bash
# Activate virtual environment and run the scheduler (Mac/Linux)
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/.venv/bin/activate"
python "$DIR/scheduler.py"
