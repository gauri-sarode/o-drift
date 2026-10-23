"""Main evaluation: all baselines + O-DRIFT on a protocol split, with
policy-selection regret over candidate policies.

    dev  : every hyperparameter in the protocol grids (for tuning)
    test : only the settings in results/tuned.json (run once, after freezing)

    .venv/bin/python scripts/03_evaluate.py --split dev [--reps N] [--cells S3a@0.5 ...] [--n-anchor 1000]
    .venv/bin/python scripts/03_evaluate.py --split test

Writes results/03_{split}[_anchor{n}].json with per-replication errors,
half-widths and choices, so every metric (RMSE, coverage, regret, paired
bootstrap) is computed downstream from the same raw records.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from odrift import baselines as bl
from odrift.method import (ODriftConfig, build_components, combine, covariance, fit_all_nuisances,
                           make_corrected_split, odrift_corrected, odrift_corrected_split, select_weights)
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.nuisance import RewardModel, crossfit_rewards
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, LoggedData, Policy, sample_episode, true_value

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
SIM, TUNE = PROTOCOL["simulator"], PROTOCOL["tuning"]
CFG = ODriftConfig(delta_variance=TUNE["delta_split"]["variance"], delta_bias=TUNE["delta_split"]["bias"],
                   **TUNE["odrift_fixed"])
Z95 = 1.959964


def make_env(seed):
    return EnvParams.sample(seed, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])


def policies(env, env_seed):
    return [Policy.near_optimal(env, noise=s, seed=env_seed) for s in TUNE["candidate_policies"]]


def truths_for(regime_name, intensity, env_seed):
    env = make_env(env_seed)
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    out = []
    for pi in policies(env, env_seed):
        v_t = true_value(env, pi, regime.target, regime.target, n_mc=SIM["n_mc_truth"])
        v_k = [true_value(env, pi, regime.target, w, n_mc=SIM["n_mc_truth"]) for w in regime.windows]
        out.append((v_t, v_k))
    return out


def method_grid(split, tuned):
    """{family: [setting, ...]} -- full grids on dev, tuned singletons on test."""
    grid = {
        "recent": TUNE["baseline_grids"]["recent_windows"],
        "decay": TUNE["baseline_grids"]["decay_gamma"],
        "liu": TUNE["baseline_grids"]["liu_recent_windows"],
        "odrift": [(m, a) for m in TUNE["odrift_grid"]["monitor_alpha"] for a in TUNE["odrift_grid"]["z_anchor"]],
    }
    if split == "test":
        grid = {fam: [tuple(tuned[fam]) if isinstance(tuned[fam], list) else tuned[fam]] for fam in grid}
    return grid


def run_one(args):
    regime_name, intensity, env_seed, data_seed, n_anchor, truths, grid = args
    env = make_env(env_seed)
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    pis = policies(env, env_seed)
    ep = sample_episode(env, regime, SIM["n_per_window"], n_anchor, SIM["n_target_contexts"], seed=data_seed)
    K, A = regime.n_windows, env.n_actions
    has_anchor = n_anchor >= CFG.min_anchor

    # ---- policy-independent fits, shared by all candidate policies
    # per-window, always-transport nuisances for baselines/oracle/selector ablations
    nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, A, replace(CFG, context_alpha=0.0), data_seed)
    stats = ResidualStats(ep.history, A)
    monitor_alphas = sorted({m for m, _ in grid["odrift"]})
    segments = {m: reward_changepoints(stats, m) for m in monitor_alphas}
    seg_nuis = {m: fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, A, CFG, data_seed,
                                     segments=segments[m]) for m in monitor_alphas}
    datasets = {"pooled": LoggedData.concat(ep.history + ([ep.anchor] if n_anchor else []))}
    if has_anchor:
        datasets["anchor"] = ep.anchor
        for B in grid["recent"]:
            datasets[f"recent{B}"] = LoggedData.concat(ep.history[K - B:] + [ep.anchor])
    q_cache = {k: crossfit_rewards(d, A, seed=data_seed)[0] for k, d in datasets.items()}
    liu_models = {B: RewardModel(A).fit(LoggedData.concat(ep.history[K - B:])) for B in grid["liu"]} if has_anchor else {}

    rec = {}  # method -> list over policies of (estimate, half_width, extra)
    for p_idx, pi in enumerate(pis):
        v_t, v_k = truths[p_idx]

        def put(name, value, hw, **extra):
            rec.setdefault(name, []).append(dict(est=float(value), err=float(value - v_t), hw=float(hw), **extra))

        def dr_on(key, h=None):
            data = datasets[key]
            scores = bl.dr_scores(data, pi, A, q_hat=q_cache[key])
            return bl._weighted_estimate(scores, np.ones(len(data)) if h is None else h)

        e = dr_on("pooled")
        put("pooled_dr", e.value, Z95 * e.se)
        if not has_anchor:
            m, za = grid["odrift"][0]
            r = odrift_corrected(ep.history, ep.anchor, ep.target_contexts, pi, A, replace(CFG, z_anchor=za),
                                 seg_nuis[m], segments[m])
            put("odrift", r.value, r.half_width, status=r.status)
            continue
        e = dr_on("anchor")
        put("anchor_dr", e.value, Z95 * e.se)
        for B in grid["recent"]:
            e = dr_on(f"recent{B}")
            put(f"recent_dr[{B}]", e.value, Z95 * e.se)
        pooled = datasets["pooled"]
        for g in grid["decay"]:
            scores = bl.dr_scores(pooled, pi, A, q_hat=q_cache["pooled"])
            e = bl._weighted_estimate(scores, g ** (K - pooled.window))
            put(f"decay_dr[{g}]", e.value, Z95 * e.se)
        scores = bl.dr_scores(pooled, pi, A, q_hat=q_cache["pooled"])
        rho2 = (pi.probs(pooled.X)[np.arange(len(pooled)), pooled.A] / pooled.pscore) ** 2
        h = np.empty(len(pooled))
        for k in np.unique(pooled.window):
            h[pooled.window == k] = 1 / rho2[pooled.window == k].mean()
        e = bl._weighted_estimate(scores, h)
        put("minvar_dr", e.value, Z95 * e.se)
        for B in grid["liu"]:
            e = liu_with_model(liu_models[B], ep.anchor, ep.target_contexts, pi, A)
            put(f"liu_reg[{B}]", e.value, Z95 * e.se)

        comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, pi, A, CFG, nuisances=nuis)
        r = bl.transport_pool(comps, anc, CFG)
        put("transport_dr", r.value, r.z_v * r.se)
        for m, za in grid["odrift"]:
            r = odrift_corrected(ep.history, ep.anchor, ep.target_contexts, pi, A, replace(CFG, z_anchor=za),
                                 seg_nuis[m], segments[m])
            deficient = [k for k, reasons in r.reasons.items() if "target_support" in reasons]
            hw_re = Z95 * (r.se_hetero if r.se_hetero is not None else r.se)
            put(f"odrift[{m},{za}]", r.value, hw_re, hw_narrow=float(Z95 * r.se), hw_certified_bound=float(r.half_width),
                status=r.status, weights=[r.weights.get(k, 0.0) for k in range(K + 1)],
                deficient=deficient, ineligible=sorted(r.reasons), n_segments=len(segments[m]))
        # oracle selector: true transfer biases, same components
        keep = [c for c in comps if c.diagnostics.eligible] + [anc]
        b = np.array([v_k[c.index] - v_t for c in keep[:-1]] + [0.0])
        alpha = select_weights(covariance(keep), b, 1.0)
        put("oracle", sum(a * c.value for a, c in zip(alpha, keep)), np.nan)
        put("ablate_mse", *_val_hw(combine(comps, anc, replace(CFG, selection="mse"), 1.0)))
        put("ablate_bound", *_val_hw(combine(comps, anc, replace(CFG, selection="bound"), 1.0)))

    # sample-split O-DRIFT (Proposition 2 conditions), interval only, per policy
    if has_anchor:
        splits = {m: make_corrected_split(ep.history, ep.anchor, ep.target_contexts, A, CFG, m, data_seed)
                  for m in monitor_alphas}
        for p_idx, pi in enumerate(pis):
            v_t = truths[p_idx][0]
            for m, za in grid["odrift"]:
                r = odrift_corrected_split(splits[m], pi, A, replace(CFG, z_anchor=za))
                rec.setdefault(f"odrift_split[{m},{za}]", []).append(
                    dict(est=r.value, err=r.value - v_t, hw=r.half_width))
    return rec


def _val_hw(r):
    return r.value, r.half_width


def liu_with_model(r_hat, anchor, X_target, pi, A):
    """bl.regression_assisted_dr with a pre-fit, policy-independent reward model."""
    probs = pi.probs(anchor.X)
    rho = probs[np.arange(len(anchor)), anchor.A] / anchor.pscore
    r_a = r_hat.predict_all(anchor.X)[np.arange(len(anchor)), anchor.A]
    Phi = np.c_[np.ones(len(anchor)), r_a]
    beta = np.linalg.solve((Phi * rho[:, None]).T @ Phi + 1e-9 * np.eye(2), (Phi * rho[:, None]).T @ anchor.R)
    dm = (pi.probs(X_target) * (beta[0] + beta[1] * r_hat.predict_all(X_target))).sum(axis=1)
    corr = rho * (anchor.R - Phi @ beta)
    from odrift.estimators import Estimate
    return Estimate(float(dm.mean() + corr.mean()),
                    float(np.sqrt(dm.var(ddof=1) / len(dm) + corr.var(ddof=1) / len(corr))), len(anchor))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--reps", type=int, default=None, help="data seeds per env seed")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--n-anchor", type=int, default=SIM["n_anchor_default"])
    ap.add_argument("--out", default=None, help="output path (default results/03_{split}[_anchorN].json)")
    args = ap.parse_args()
    split = PROTOCOL["splits"][args.split]
    tuned = None
    if args.split == "test":
        assert PROTOCOL["status"] == "FROZEN", "protocol must be frozen before touching test"
        tuned = json.loads((ROOT / "results/tuned.json").read_text())
    grid = method_grid(args.split, tuned)
    cells = [(r, i) for r, ints in split["regimes"].items() for i in ints]
    if args.cells:
        cells = [(c.split("@")[0], float(c.split("@")[1])) for c in args.cells]
    reps = args.reps or split["data_seeds"]["count"]
    start = split["data_seeds"]["start"]

    out = dict(split=args.split, n_anchor=args.n_anchor, reps_per_env=reps, env_seeds=split["env_seeds"],
               grid={k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in grid.items()}, cells={})
    with ProcessPoolExecutor() as pool:
        for regime_name, intensity in cells:
            n_env = len(split["env_seeds"])
            truth = dict(zip(split["env_seeds"], pool.map(truths_for, [regime_name] * n_env,
                                                           [intensity] * n_env, split["env_seeds"])))
            jobs = [(regime_name, intensity, e, start + s, args.n_anchor, truth[e], grid)
                    for e in split["env_seeds"] for s in range(reps)]
            results = list(pool.map(run_one, jobs))
            key = f"{regime_name}@{intensity}"
            out["cells"][key] = dict(
                truths={str(e): [t[0] for t in truth[e]] for e in split["env_seeds"]},
                env_of_rep=[j[2] for j in jobs], reps=results)
            rmse = {m: np.sqrt(np.mean([[p["err"] ** 2 for p in r[m]] for r in results]))
                    for m in results[0]}
            best = sorted(rmse.items(), key=lambda kv: kv[1])[:6]
            print(f"{key:10s} n={len(results)}  best: " + "  ".join(f"{m}={v:.4f}" for m, v in best), flush=True)

    suffix = "" if args.n_anchor == SIM["n_anchor_default"] else f"_anchor{args.n_anchor}"
    path = ROOT / (args.out or f"results/03_{args.split}{suffix}.json")
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
