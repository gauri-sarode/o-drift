"""Week-2 dev experiment: O-DRIFT vs pooled/anchor DR and an oracle-bias
selector, on the DEV split only (configs/protocol.yaml). Used to pick lambda
and check the bound/coverage machinery; never touches test seeds.

    .venv/bin/python scripts/02_dev_odrift.py [--reps 50] [--cells S3a@0.5 S4@0.5]
"""

import argparse
from dataclasses import replace
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from odrift import estimators as est
from odrift.method import ODriftConfig, build_components, combine, covariance, odrift, select_weights
from odrift.nuisance import crossfit_rewards
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, LoggedData, Policy, sample_episode, true_value

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
SIM, DEV = PROTOCOL["simulator"], PROTOCOL["splits"]["dev"]
LAMS = PROTOCOL["tuning"]["lambda_grid"]
Z_GROUP = PROTOCOL["tuning"]["z_group_grid"]
Z_ANCHOR = PROTOCOL["tuning"]["z_anchor_grid"]
CFG = ODriftConfig(delta_variance=PROTOCOL["tuning"]["delta_split"]["variance"],
                   delta_bias=PROTOCOL["tuning"]["delta_split"]["bias"])


def make_env(seed):
    return EnvParams.sample(seed, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])


def truths_for(regime_name, intensity, env_seed):
    env = make_env(env_seed)
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    pi = Policy.near_optimal(env)
    v_t = true_value(env, pi, regime.target, regime.target, n_mc=SIM["n_mc_truth"])
    v_k = [true_value(env, pi, regime.target, w, n_mc=SIM["n_mc_truth"]) for w in regime.windows]
    return v_t, v_k


def run_one(args):
    regime_name, intensity, env_seed, data_seed, v_t, v_k = args
    env = make_env(env_seed)
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    pi = Policy.near_optimal(env)
    ep = sample_episode(env, regime, SIM["n_per_window"], SIM["n_anchor_default"],
                        SIM["n_target_contexts"], seed=data_seed)
    rec = {}

    def put(name, value, half_width, extra=None):
        rec[name] = dict(err=value - v_t, hw=half_width, **(extra or {}))

    # baselines
    pooled = LoggedData.concat(ep.history)
    for label, data in (("pooled_dr", pooled), ("anchor_dr", ep.anchor)):
        q_hat, _ = crossfit_rewards(data, env.n_actions, seed=data_seed)
        e = est.dr(data, pi.probs(data.X), q_hat)
        put(label, e.value, 1.959964 * e.se)

    # O-DRIFT plug-in, all lambdas from one set of components
    comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, CFG, data_seed)
    b_true = np.array([v - v_t for v in v_k])
    for lam in LAMS:
        r = combine(comps, anc, replace(CFG, selection="mse"), lam)
        bounds = np.array([r.bounds[k] for k in range(len(v_k)) if k in r.bounds])
        held = np.array([k for k in range(len(v_k)) if k in r.bounds])
        put(f"mse_l{lam}", r.value, r.half_width, dict(
            weights={str(k): a for k, a in r.weights.items()},
            bounds_hold=bool(np.all(bounds >= np.abs(b_true[held]))) if len(held) else True,
            n_ineligible=len(r.reasons), reasons=sorted({x for v in r.reasons.values() for x in v})))

    # selector variants (segment is the default; mse/bound are ablations)
    for zg in Z_GROUP:
        for za in Z_ANCHOR:
            r = combine(comps, anc, replace(CFG, z_group=zg, z_anchor=za))
            put(f"seg_g{zg}_a{za}_narrow", r.value, r.z_v * r.se)
            put(f"seg_g{zg}_a{za}", r.value, r.half_width,
                dict(weights={str(k): a for k, a in r.weights.items()}, bounds_hold=True,
                     n_ineligible=len(r.reasons)))
    r = combine(comps, anc, replace(CFG, selection="bound"), 1.0)
    put("ablate_bound_select", r.value, r.half_width)

    # oracle: true transfer bias, same components and covariance, eligible windows only
    keep = [c for c in comps if c.diagnostics.eligible] + [anc]
    b = np.array([b_true[c.index] for c in keep[:-1]] + [0.0])
    S = covariance(keep)
    alpha = select_weights(S, b, 1.0)
    put("oracle", float(sum(a * c.value for a, c in zip(alpha, keep))), np.nan)

    # sample-split variant (Proposition 2 conditions)
    for zg in Z_GROUP:
        r = odrift(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions,
                   replace(CFG, z_group=zg), split=True, seed=data_seed)[CFG.lam]
        put(f"seg_split_g{zg}", r.value, r.half_width)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=DEV["data_seeds"]["count"])
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()
    cells = [(r, i) for r, ints in DEV["regimes"].items() for i in ints]
    if args.cells:
        cells = [(c.split("@")[0], float(c.split("@")[1])) for c in args.cells]
    start = DEV["data_seeds"]["start"]

    summary = {}
    with ProcessPoolExecutor() as pool:
        for regime_name, intensity in cells:
            truth = dict(zip(DEV["env_seeds"], pool.map(truths_for, [regime_name] * 4,
                                                          [intensity] * 4, DEV["env_seeds"])))
            jobs = [(regime_name, intensity, e, start + s, *truth[e])
                    for e in DEV["env_seeds"] for s in range(args.reps)]
            results = list(pool.map(run_one, jobs))
            cell = {}
            for name in results[0]:
                err = np.array([r[name]["err"] for r in results])
                hw = np.array([r[name]["hw"] for r in results])
                m = dict(rmse=float(np.sqrt((err ** 2).mean())), bias=float(err.mean()),
                         coverage=float((np.abs(err) <= hw).mean()) if np.isfinite(hw).all() else None,
                         mean_half_width=float(hw.mean()) if np.isfinite(hw).all() else None)
                if "bounds_hold" in results[0][name]:
                    m["bounds_hold_rate"] = float(np.mean([r[name]["bounds_hold"] for r in results]))
                    m["mean_ineligible"] = float(np.mean([r[name]["n_ineligible"] for r in results]))
                    w = np.zeros(SIM["n_windows"] + 1)
                    for r in results:
                        for k, a in r[name]["weights"].items():
                            w[int(k)] += a / len(results)
                    m["mean_weights"] = w.round(3).tolist()
                cell[name] = m
            key = f"{regime_name}@{intensity}"
            summary[key] = cell
            print(f"\n== {key}  (n={len(results)})")
            print(f"{'method':22s} {'rmse':>8s} {'bias':>9s} {'cov95':>6s} {'halfw':>7s} {'Bhold':>6s} {'inelig':>6s}")
            for name, m in cell.items():
                cov = "" if m["coverage"] is None else f"{m['coverage']:.3f}"
                hw = "" if m["mean_half_width"] is None else f"{m['mean_half_width']:.4f}"
                print(f"{name:22s} {m['rmse']:8.4f} {m['bias']:+9.4f} {cov:>6s} {hw:>7s} "
                      f"{m.get('bounds_hold_rate', ''):>6} {m.get('mean_ineligible', ''):>6}")

    out = ROOT / "results/02_dev_odrift.json"
    out.write_text(json.dumps(dict(reps_per_env=args.reps, split="dev", cells=summary), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
