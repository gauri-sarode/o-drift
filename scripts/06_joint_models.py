"""Dev-only comparison: separately fitted vs jointly fitted segment reward models
(decision log item 16). Same seeds/policies as 03_evaluate; O-DRIFT tuned setting.

    PYTHONPATH=src .venv/bin/python scripts/06_joint_models.py [--reps 20] [--cells ...]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from odrift.method import ODriftConfig, fit_all_nuisances, odrift_balanced
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, Policy, mu, sample_episode, true_value

ROOT = Path(__file__).resolve().parents[1]
P = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
SIM, TUNE, DEV = P["simulator"], P["tuning"], P["splits"]["dev"]
TUNED = json.loads((ROOT / "results/tuned.json").read_text())["odrift"]
CFG = ODriftConfig(delta_variance=TUNE["delta_split"]["variance"], delta_bias=TUNE["delta_split"]["bias"],
                   **TUNE["odrift_fixed"], z_anchor=TUNED[1])


VARIANTS = (("joint", True, "", False), ("joint_anchor", True, "", True),
            ("joint_anchor_shared_trend", True, "shared", True), ("joint_anchor_trend", True, "segment", True))


def truths(regime_name, intensity, e):
    env = EnvParams.sample(e, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])
    reg = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    return [true_value(env, Policy.near_optimal(env, noise=s, seed=e), reg.target, reg.target, n_mc=SIM["n_mc_truth"])
            for s in TUNE["candidate_policies"]]


def run_one(args):
    regime_name, intensity, e, seed, vt = args
    env = EnvParams.sample(e, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])
    reg = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    ep = sample_episode(env, reg, SIM["n_per_window"], SIM["n_anchor_default"], SIM["n_target_contexts"], seed=seed)
    segs = reward_changepoints(ResidualStats(ep.history, env.n_actions), TUNED[0])
    fns = [lambda X, a=(sp if sp.recorded_logger_id is None else replace(sp, logger_id=sp.recorded_logger_id)):
           mu(env, a, X) for sp in reg.windows]
    out = {}
    for name, joint, trend, anc in VARIANTS:
        cfg = replace(CFG, joint_segments=joint, segment_trend=trend, model_anchor=anc)
        nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, env.n_actions, cfg, seed, segments=segs)
        out[name] = [odrift_balanced(ep.history, ep.anchor, ep.target_contexts, Policy.near_optimal(env, noise=s, seed=e),
                                     env.n_actions, cfg, nuis, segs, fns).value - v for s, v in zip(TUNE["candidate_policies"], vt)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cells", nargs="*", default=["S0@0.0", "S1@1.0", "S2@0.0", "S3a@0.5", "S3b@0.5", "S4@0.5", "S8@0.5"])
    ap.add_argument("--out", default="results/06_joint_models.json")
    args = ap.parse_args()
    res = {}
    with ProcessPoolExecutor() as pool:
        for cell in args.cells:
            r, i = cell.split("@")[0], float(cell.split("@")[1])
            vt = dict(zip(DEV["env_seeds"], pool.map(truths, [r] * 4, [i] * 4, DEV["env_seeds"])))
            jobs = [(r, i, e, DEV["data_seeds"]["start"] + s, vt[e]) for e in DEV["env_seeds"] for s in range(args.reps)]
            out = list(pool.map(run_one, jobs))
            env = np.array([j[2] for j in jobs])
            res[cell] = {}
            for v in [n for n, *_ in VARIANTS]:
                err = np.array([o[v] for o in out])
                res[cell][v] = dict(rmse=float(np.sqrt((err ** 2).mean())), bias=float(err.mean()),
                                    errs=err.tolist(), rmse_env=[float(np.sqrt((err[env == e] ** 2).mean())) for e in DEV["env_seeds"]])
            print(cell, {v: (round(d["rmse"] * 1e3, 2), round(d["bias"] * 1e3, 2), [round(x * 1e3, 1) for x in d["rmse_env"]])
                         for v, d in res[cell].items()}, flush=True)
    (ROOT / args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
