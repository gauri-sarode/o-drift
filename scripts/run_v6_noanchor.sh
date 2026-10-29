#!/bin/zsh
# protocol v6, criterion C5 (abstain without an anchor): test6 with n_anchor 0, after run_v6.sh finishes
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until [ -f results/logs/v6.done ]; do sleep 60; done
.venv/bin/python scripts/03_evaluate.py --split test6 --basis select:32 --n-anchor 0 > results/logs/03_test6_anchor0.log 2>&1
touch results/logs/v6_noanchor.done
