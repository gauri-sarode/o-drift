#!/bin/zsh
# After the v6 runs: test6 criteria, then the post-hoc runs (decision log 39), one at a time.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until [ -f results/logs/v6_noanchor.done ]; do sleep 60; done
.venv/bin/python scripts/04_analyze.py --split test6 > results/logs/04_test6.log 2>&1
.venv/bin/python scripts/18_real_floor.py > results/logs/18_real_floor.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/17_runtime.py > results/logs/17_runtime.log 2>&1   # machine otherwise idle
touch results/logs/v6_posthoc.done
