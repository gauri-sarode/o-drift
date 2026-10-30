# O-Drift: observability-aware off-policy evaluation under drift

Code, pre-registered protocol and results for the paper
*O-Drift: Observability-Aware Off-Policy Evaluation in Evolving Recommender Systems*.

O-Drift estimates the value of a target policy for the current period from logs collected
over many earlier periods plus a small sample from the current period (the *anchor*). It
segments the history with a per-action residual test for drift in the conditional reward
(invariant to context and logger drift), fits joint segment reward models, reuses older
segments with a drift correction, and pools the segment estimates at minimum variance.

## Setup

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q          # 46 tests
```

## Data

The simulator needs no data. The Open Bandit experiments use the Open Bandit Dataset
(Saito et al., 2021; CC BY 4.0), which is not redistributed here. Download it into
`data/open_bandit_dataset/` (random and bts logs of the men, women and all campaigns), then run

```bash
.venv/bin/python scripts/07_prepare_obd.py              # semi-synthetic -> data/obd_men.npz
.venv/bin/python scripts/15_prepare_obd_real.py         # real rewards  -> data/obd_real_{men,women,all}.npz
```

## Protocol

`configs/protocol.yaml` is the pre-registration: splits and seeds, tuning grids, the tuning
rule, and the go/no-go criteria of every version (`v2`-`v6` blocks, each with its
status and date). Tuning uses development splits only; each test split was run once per
frozen configuration. `docs/PLAN.md` is the dated decision log (items 1-37), including every
post-hoc finding and why it was made (items 1-40).

| Version | What changed | Outcome |
|---|---|---|
| v1 | first frozen method | simulator test passed; Open Bandit failed (PLAN 21, 25) |
| v2 | reference-validated reuse | tuned to the v1 setting on dev; not tested (PLAN 26-28) |
| v3 | per-episode basis, offset shrinkage, COPE baseline, grouping ablation | **the tested version in the paper** (PLAN 31-33) |
| v4 | reuse gate at z = 1.96 (post-hoc) | failed dev, never tested (PLAN 34) |
| v5 | rolling-origin back-test (post-hoc) | failed dev, never tested (PLAN 35-36) |
| v6 | v3 frozen; OPFV baseline, replication on fresh seeds, real-reward Open Bandit | all criteria pass (PLAN 38-40) |

## Reproducing the paper

All commands run from the repository root with `PYTHONPATH=src`. Raw outputs are in
`results/` (JSON files are stored with Git LFS). Each full run uses all CPU cores; run them
one at a time.

| Paper item | Produced by |
|---|---|
| Dev tuning, v3 (`results/tuned_v3.json`) | `scripts/03_evaluate.py --split dev3 --basis select:32`, then `scripts/04_analyze.py --split dev3` |
| Test runs, v3 (Tables 1-2, criteria) | `scripts/run_v3_tests.sh` (test3, test3 without anchor, Open Bandit, then the analyses) |
| Tables 1-2 | `scripts/12_paper_tables.py` -> `paper/tables/` (needs the v3 and test6 analyses) |
| v4/v5 dev results (post-hoc, Section 6 text) | `scripts/03_evaluate.py --split dev4 --basis select:32`; `scripts/03_evaluate.py --split dev5 --basis select:32 --reps 13 --out results/03_dev5_r13.json`; `scripts/13_revisions_table.py` |
| Anchor-size sweep (post-hoc, Section 6) | `scripts/run_v3_anchor_sweep.sh`, then `scripts/14_anchor_sweep.py` |
| v6 runs (replication, no-anchor run, real rewards) | `scripts/run_v6.sh`, `scripts/run_v6_noanchor.sh` |
| Table 1 replication columns, v6 criteria | `scripts/04_analyze.py --split test6`, then `scripts/12_paper_tables.py` |
| Table 3 (real rewards) | `scripts/16_real_analysis.py` |
| Zero-click anchor floor, runtime (post-hoc) | `scripts/18_real_floor.py`, `scripts/17_runtime.py` (chained by `scripts/run_v6_posthoc.sh`) |
| Paper PDF | `cd paper && latexmk -pdf main.tex` |

Approximate run times on a 10-core laptop: dev3 3-4 h, test3 5-6 h, test3 without anchor
1.5 h, Open Bandit 2-2.5 h, anchor sweep about 1.5 h at 100 records and 5 h at 10,000
(of which S3b alone takes about 3.5 h); real6 about 1 h, test6 about 6.5 h, test6 without anchor
2.5 h.

## Layout

```
src/odrift/        simulator.py, regimes.py (S0-S8b), obd.py (Open Bandit environment),
                   nuisance.py (cross-fitted reward models, bases), monitor.py (residual test),
                   method.py (O-Drift), baselines.py, backtest.py (v5), obd_real.py (real rewards)
scripts/           01-04 main pipeline; 07, 15 Open Bandit preparation; 12-14, 16-18 paper tables,
                   figures and post-hoc analyses;
                   02, 05, 06, 08-11 dev-only diagnostics and prototypes cited in docs/PLAN.md
configs/           protocol.yaml (pre-registration)
docs/              PLAN.md (decision log), notes_monitor_invariance.md (Proposition 1 notes)
results/           raw outputs (03_*.json), analyses (04_*), tuned settings, logs/
paper/             LaTeX source (Springer LNCS)
tests/             unit tests
```
