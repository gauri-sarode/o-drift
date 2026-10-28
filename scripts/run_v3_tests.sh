#!/bin/zsh
# v3 one-shot test runs (protocol v3 FROZEN 2026-10-28), sequential: one pool job at a time.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
$PY -u scripts/03_evaluate.py --split test3 --basis select:32 > results/logs/03_test3.log 2>&1
$PY -u scripts/03_evaluate.py --split test3 --basis select:32 --n-anchor 0 > results/logs/03_test3_anchor0.log 2>&1
$PY -u scripts/03_evaluate.py --split obd3 --basis select:32 > results/logs/03_obd3.log 2>&1
$PY scripts/04_analyze.py --split obd3 > results/logs/04_obd3.log 2>&1
$PY scripts/04_analyze.py --split test3 > results/logs/04_test3.log 2>&1
echo done > results/logs/v3_tests.done
