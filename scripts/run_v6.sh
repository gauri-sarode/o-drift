#!/bin/zsh
# v6 one-shot runs (protocol v6, frozen): real rewards first (~5 min), then the test6 replication (~5-6 h).
cd "$(dirname "$0")/.."
export PYTHONPATH=src
.venv/bin/python scripts/03_evaluate.py --split real6 --basis select:32 > results/logs/03_real6.log 2>&1
.venv/bin/python scripts/03_evaluate.py --split test6 --basis select:32 > results/logs/03_test6.log 2>&1
touch results/logs/v6.done
