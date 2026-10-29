#!/bin/zsh
# Post-hoc v3 anchor sweep (protocol v3.anchor_sweep_v3): frozen v3 config, test3 seeds, sequential.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
CELLS=(S0@0.0 S1@2.0 S2@0.0 S3a@1.0 S3b@1.0 S4@1.0 S5@1.0 S8@1.0)
for n in 100 10000; do
  .venv/bin/python -u scripts/03_evaluate.py --split test3 --basis select:32 --n-anchor $n --cells $CELLS \
    > results/logs/03_test3_anchor$n.log 2>&1
done
echo done > results/logs/v3_anchor_sweep.done
