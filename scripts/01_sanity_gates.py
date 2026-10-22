"""Week-1 gate: standard estimators are unbiased with ~nominal coverage when
their assumptions hold (S0 stationary, S2 logger switch with correct
propensities), and fail in the expected direction when they don't (S3a
reward drift, S6 wrong propensities). Dev seeds only.

    .venv/bin/python scripts/01_sanity_gates.py [--reps 50]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from odrift import estimators as est
from odrift.nuisance import crossfit_rewards
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, LoggedData, Policy, sample_episode, true_value

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
SIM = PROTOCOL["simulator"]
CELLS = [("S0", 0.0), ("S2", 0.0), ("S3a", 0.5), ("S6", 0.0)]


def run_one(args):
    regime_name, intensity, env_seed, data_seed = args
    env = EnvParams.sample(env_seed, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    pi = Policy.near_optimal(env)
    ep = sample_episode(env, regime, SIM["n_per_window"], SIM["n_anchor_default"],
                        SIM["n_target_contexts"], seed=data_seed)
    pooled = LoggedData.concat(ep.history)
    out = {}
    for label, data in (("pooled", pooled), ("anchor", ep.anchor)):
        probs = pi.probs(data.X)
        q_hat, _ = crossfit_rewards(data, env.n_actions, seed=data_seed)
        out[f"{label}_ips"] = est.ips(data, probs)
        out[f"{label}_snips"] = est.snips(data, probs)
        out[f"{label}_dr"] = est.dr(data, probs, q_hat)
    return {k: (e.value, e.se) for k, e in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=PROTOCOL["splits"]["dev"]["data_seeds"]["count"])
    args = ap.parse_args()
    env_seeds = PROTOCOL["splits"]["dev"]["env_seeds"]
    start = PROTOCOL["splits"]["dev"]["data_seeds"]["start"]

    summary = {}
    with ProcessPoolExecutor() as pool:
        for regime_name, intensity in CELLS:
            jobs = [(regime_name, intensity, e, start + s) for e in env_seeds for s in range(args.reps)]
            results = list(pool.map(run_one, jobs))
            truths = {}
            for e in env_seeds:
                env = EnvParams.sample(e, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])
                regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
                truths[e] = true_value(env, Policy.near_optimal(env), regime.target, regime.target,
                                       n_mc=SIM["n_mc_truth"])
            truth = np.array([truths[j[2]] for j in jobs])
            cell = {}
            for key in results[0]:
                v = np.array([r[key][0] for r in results])
                se = np.array([r[key][1] for r in results])
                err = v - truth
                cell[key] = dict(
                    bias=float(err.mean()), bias_se=float(err.std(ddof=1) / np.sqrt(len(err))),
                    rmse=float(np.sqrt((err ** 2).mean())),
                    coverage_95=float((np.abs(err) <= 1.959964 * se).mean()),
                    mean_se=float(se.mean()), emp_sd=float(err.std(ddof=1)),
                )
            summary[f"{regime_name}@{intensity}"] = cell

    print(f"{'cell':10s} {'estimator':14s} {'bias':>9s} {'(se)':>8s} {'rmse':>8s} {'cov95':>6s} {'meanSE/sd':>9s}")
    for cell_name, cell in summary.items():
        for key, m in cell.items():
            print(f"{cell_name:10s} {key:14s} {m['bias']:+9.4f} {m['bias_se']:8.4f} {m['rmse']:8.4f} "
                  f"{m['coverage_95']:6.3f} {m['mean_se'] / m['emp_sd']:9.2f}")
    out = ROOT / "results/01_sanity_gates.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(reps_per_env=args.reps, env_seeds=env_seeds, cells=summary), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
