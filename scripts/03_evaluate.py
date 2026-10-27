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
from odrift import obd
from odrift.method import (ODriftConfig, build_components, combine, covariance, fit_all_nuisances,
                           make_corrected_split, odrift_balanced, ratio_function, odrift_corrected, odrift_corrected_split,
                           select_weights)
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.nuisance import RewardModel, crossfit_rewards, set_basis
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, LoggedData, Policy, mu, sample_episode, true_value

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
SIM, TUNE, V2 = PROTOCOL["simulator"], PROTOCOL["tuning"], PROTOCOL["v2"]
SPLIT_KEY = {"dev": "dev", "test": "test", "obd": "real", "dev2": "dev2", "test2": "test2", "obd2": "real2"}
CFG = ODriftConfig(delta_variance=TUNE["delta_split"]["variance"], delta_bias=TUNE["delta_split"]["bias"],
                   **TUNE["odrift_fixed"])
Z95 = 1.959964


def make_env(seed, dataset="sim", misspec=None):
    if dataset == "obd":
        return obd.OBDEnv.sample(seed)
    return EnvParams.sample(seed, d=SIM["d"], n_actions=SIM["n_actions"],
                            misspec=SIM["misspec"] if misspec is None else misspec)


def cell_key(regime_name, intensity, misspec):
    """'S3a@0.5' at the default misspec; 'S3a@0.5#m3.0' for v2's misspec cells."""
    return f"{regime_name}@{intensity}" + ("" if misspec is None else f"#m{misspec}")


def make_episode(env, regime, n_anchor, seed, dataset="sim"):
    sampler = obd.sample_episode if dataset == "obd" else sample_episode
    return sampler(env, regime, SIM["n_per_window"], n_anchor, SIM["n_target_contexts"], seed=seed)


def policies(env, env_seed):
    noises = PROTOCOL["splits"]["real"]["candidate_policies"] if isinstance(env, obd.OBDEnv) else TUNE["candidate_policies"]
    return [Policy.near_optimal(env, noise=s, seed=env_seed) for s in noises]


def truths_for(regime_name, intensity, env_seed, dataset="sim", misspec=None):
    env = make_env(env_seed, dataset, misspec)
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    out = []
    for pi in policies(env, env_seed):
        if dataset == "obd":  # exact over all real target-block contexts
            out.append((obd.true_value(env, pi, regime.target), [obd.true_value(env, pi, w) for w in regime.windows]))
            continue
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
        "odrift": [(m, a, int(j)) for m in TUNE["odrift_grid"]["monitor_alpha"] for a in TUNE["odrift_grid"]["z_anchor"]
                   for j in TUNE["odrift_grid"]["joint_segments"]],
        "slope": TUNE["baseline_grids"]["slope_kappa"],
        "magic": TUNE["baseline_grids"]["magic_z_ref"],
        "trecent": TUNE["baseline_grids"]["transport_recent"],
        "tdecay": TUNE["baseline_grids"]["transport_decay"],
        "forecast": TUNE["baseline_grids"]["forecast_degree"],
        "timedr": TUNE["baseline_grids"]["time_dr_degree"],
        "tbdecay": TUNE["baseline_grids"]["transport_balanced_decay"],
    }
    if split in ("dev2", "test2"):  # v2: v1's tuned O-DRIFT kept for comparison + the reuse_lam family
        fx = V2["fixed_from_v1"]
        grid["odrift"] = [(fx["monitor_alpha"], fx["z_anchor"], int(fx["joint_segments"]))]
        grid["odrift2"] = V2["reuse_lam_grid"]
    if split in ("test", "test2"):
        full = {k: grid[k] for k in ("recent", "decay")}  # for oracle-tuned upper bounds only
        grid = {fam: [tuple(tuned[fam]) if isinstance(tuned[fam], list) else tuned[fam]] for fam in grid}
        grid.update(full)
    return grid


def run_one(args):
    regime_name, intensity, env_seed, data_seed, n_anchor, truths, grid, dataset, misspec = args
    set_basis(*grid.get("basis", ("linear", 0)))  # context basis of every reward model (not the monitor)
    env = make_env(env_seed, dataset, misspec)
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    pis = policies(env, env_seed)
    ep = make_episode(env, regime, n_anchor, data_seed, dataset)
    K, A = regime.n_windows, env.n_actions
    has_anchor = n_anchor >= CFG.min_anchor

    # ---- policy-independent fits, shared by all candidate policies
    stats = ResidualStats(ep.history, A)
    monitor_alphas = sorted({m for m, _, _ in grid["odrift"]})
    joints = sorted({j for _, _, j in grid["odrift"]})
    segments = {m: reward_changepoints(stats, m) for m in monitor_alphas}
    # per-window, always-transport nuisances for baselines/oracle/selector ablations
    nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, A, replace(CFG, context_alpha=0.0), data_seed)
    seg_nuis = {(m, j): fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, A,
                                          replace(CFG, joint_segments=bool(j)), data_seed, segments=segments[m])
                for m in monitor_alphas for j in joints}
    datasets = {"pooled": LoggedData.concat(ep.history + ([ep.anchor] if n_anchor else []))}
    if has_anchor:
        datasets["anchor"] = ep.anchor
        for B in grid["recent"]:
            datasets[f"recent{B}"] = LoggedData.concat(ep.history[K - B:] + [ep.anchor])
    # monitor-informed selection (selector parity): newest reward segment + anchor, no correction
    m0 = grid["odrift"][0][0]
    if has_anchor:
        datasets["monitor_seg"] = LoggedData.concat([ep.history[k] for k in sorted(segments[m0][0])] + [ep.anchor])
    q_cache = {k: crossfit_rewards(d, A, seed=data_seed)[0] for k, d in datasets.items()}
    artifacts = [sp if sp.recorded_logger_id is None else replace(sp, logger_id=sp.recorded_logger_id)
                 for sp in regime.windows]
    logger_fns = [lambda X, a=a: mu(env, a, X) for a in artifacts]
    # multi-logger balanced propensities from event-time logger artifacts (A4); in S6 the
    # artifact for the faulty windows is the wrongly recorded logger, as in the logs
    if has_anchor:
        pooled_d = datasets["pooled"]
        specs = list(regime.windows) + [regime.target]
        counts = np.bincount(pooled_d.window, minlength=K + 1)
        mixture = np.zeros((len(pooled_d), A))
        for k, spec in enumerate(specs):
            if counts[k]:
                art = spec if spec.recorded_logger_id is None else replace(spec, logger_id=spec.recorded_logger_id)
                mixture += counts[k] / counts.sum() * mu(env, art, pooled_d.X)
    if has_anchor:
        # drift-specific baselines: time-aware reward models; composite transport+balanced+decay
        time_q = {deg: bl.fit_time_reward(pooled_d, K, ep.target_contexts, A, deg, data_seed) for deg in grid["timedr"]}
        q_pool_own, q_pool_tgt = crossfit_rewards(pooled_d, A, seed=data_seed, extra_X=ep.target_contexts)
        ratio_fns = [ratio_function(d.X, ep.target_contexts) for d in ep.history] + [None]
        tgt_art = regime.target if regime.target.recorded_logger_id is None else \
            replace(regime.target, logger_id=regime.target.recorded_logger_id)
        logger_fns_all = logger_fns + [lambda X, a=tgt_art: mu(env, a, X)]
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
            m, za, j = grid["odrift"][0]
            r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, pi, A, replace(CFG, z_anchor=za),
                                seg_nuis[(m, j)], segments[m], logger_fns)
            put("odrift", r.value, r.half_width, status=r.status)
            for lam in grid.get("odrift2", []):
                r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, pi, A,
                                    replace(CFG, z_anchor=za, reuse_lam=lam), seg_nuis[(m, j)], segments[m], logger_fns)
                put(f"odrift2[{lam}]", r.value, r.half_width, status=r.status)
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

        # ---- strong-baseline suite
        probs_pooled = pi.probs(pooled.X)
        e = bl.balanced_mixture_dr(pooled, mixture, probs_pooled, q_cache["pooled"])
        put("balanced_dr", e.value, Z95 * e.se)
        e = dr_on("monitor_seg")
        put("monitor_select_dr", e.value, Z95 * e.se)
        fam_v, fam_var = bl.nested_family(scores, pooled.window, K)
        for kappa in grid["slope"]:
            j = bl.slope_select(fam_v, fam_var, kappa)
            put(f"slope[{kappa}]", fam_v[j], Z95 * np.sqrt(fam_var[j]), chosen=int(j))
        for zr in grid["magic"]:
            w = bl.magic_weights(fam_v, fam_var, zr)
            put(f"magic[{zr}]", float(w @ fam_v), Z95 * float(np.sqrt(w @ np.minimum.outer(fam_var, fam_var) @ w)))
        j = bl.anchor_validated_select(fam_v, fam_var)
        put("anchor_validated", fam_v[j], Z95 * np.sqrt(fam_var[j]), chosen=int(j))

        comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, pi, A, CFG, nuisances=nuis)
        r = bl.transport_pool(comps, anc, CFG)
        put("transport_dr", r.value, r.z_v * r.se)
        elig = [c for c in comps if c.diagnostics.eligible]
        for B in grid["trecent"]:
            sub = [c for c in elig if c.index >= K - B] + [anc]
            Ssub = covariance(sub)
            a_ = select_weights(Ssub, np.zeros(len(sub)), 0.0)
            put(f"trecent[{B}]", float(a_ @ [c.value for c in sub]), Z95 * float(np.sqrt(a_ @ Ssub @ a_)))
        for g in grid["tdecay"]:
            sub = elig + [anc]
            Ssub = covariance(sub)
            a_ = np.array([g ** (K - c.index) for c in sub]) / np.diag(Ssub)
            a_ /= a_.sum()
            put(f"tdecay[{g}]", float(a_ @ [c.value for c in sub]), Z95 * float(np.sqrt(a_ @ Ssub @ a_)))
        for deg in grid["forecast"]:
            sub = elig + [anc]
            v_, se_ = bl.forecast_value(sub, covariance(sub), K, deg)
            put(f"forecast[{deg}]", v_, Z95 * se_)
        for deg in grid["timedr"]:
            e = bl.time_aware_dr(pooled, *time_q[deg], ep.target_contexts, pi)
            put(f"timedr[{deg}]", e.value, Z95 * e.se)
        for g in grid["tbdecay"]:
            e = bl.transport_balanced_decay_dr(pooled, q_pool_own, q_pool_tgt, ep.target_contexts, pi, K, g,
                                               logger_fns_all, ratio_fns)
            put(f"tbdecay[{g}]", e.value, Z95 * e.se)
        for m, za, j in grid["odrift"]:
            r = odrift_corrected(ep.history, ep.anchor, ep.target_contexts, pi, A, replace(CFG, z_anchor=za),
                                 seg_nuis[(m, j)], segments[m])
            put(f"ablate_nobalance[{m},{za},{j}]", r.value, Z95 * (r.se_hetero or r.se))
            r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, pi, A, replace(CFG, z_anchor=za),
                                seg_nuis[(m, j)], segments[m], logger_fns)
            deficient = [k for k, reasons in r.reasons.items() if "target_support" in reasons]
            hw_re = Z95 * (r.se_hetero if r.se_hetero is not None else r.se)
            put(f"odrift[{m},{za},{j}]", r.value, hw_re, hw_narrow=float(Z95 * r.se), hw_certified_bound=float(r.half_width),
                status=r.status, weights=[r.weights.get(k, 0.0) for k in range(K + 1)],
                deficient=deficient, ineligible=sorted(r.reasons), n_segments=len(segments[m]))
            if (m, za, j) == tuple(grid["odrift"][0]):
                for lam in grid.get("odrift2", []):  # v2: reference-validated reuse on the same fits
                    r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, pi, A,
                                        replace(CFG, z_anchor=za, reuse_lam=lam), seg_nuis[(m, j)], segments[m],
                                        logger_fns)
                    deficient = [k for k, reasons in r.reasons.items() if "target_support" in reasons]
                    put(f"odrift2[{lam}]", r.value, Z95 * (r.se_hetero if r.se_hetero is not None else r.se),
                        hw_narrow=float(Z95 * r.se), status=r.status, weights=[r.weights.get(k, 0.0) for k in range(K + 1)],
                        deficient=deficient, n_segments=len(segments[m]))
        # oracle selector: true transfer biases, same components
        keep = [c for c in comps if c.diagnostics.eligible] + [anc]
        b = np.array([v_k[c.index] - v_t for c in keep[:-1]] + [0.0])
        alpha = select_weights(covariance(keep), b, 1.0)
        put("oracle", sum(a * c.value for a, c in zip(alpha, keep)), np.nan)
        put("ablate_mse", *_val_hw(combine(comps, anc, replace(CFG, selection="mse"), 1.0)))
        put("ablate_bound", *_val_hw(combine(comps, anc, replace(CFG, selection="bound"), 1.0)))

    # sample-split O-DRIFT (Proposition 2 conditions), interval only, per policy
    if has_anchor:
        splits = {(m, j): make_corrected_split(ep.history, ep.anchor, ep.target_contexts, A,
                                               replace(CFG, joint_segments=bool(j)), m, data_seed)
                  for m in monitor_alphas for j in joints}
        for p_idx, pi in enumerate(pis):
            v_t = truths[p_idx][0]
            for m, za, j in grid["odrift"]:
                r = odrift_corrected_split(splits[(m, j)], pi, A, replace(CFG, z_anchor=za))
                rec.setdefault(f"odrift_split[{m},{za},{j}]", []).append(
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
    ap.add_argument("--split", choices=list(SPLIT_KEY), required=True)
    ap.add_argument("--reps", type=int, default=None, help="data seeds per env seed")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--n-anchor", type=int, default=SIM["n_anchor_default"])
    ap.add_argument("--basis", default="linear:0", help="reward-model context basis, e.g. rff:32")
    ap.add_argument("--out", default=None, help="output path (default results/03_{split}[_anchorN].json)")
    args = ap.parse_args()
    dataset = "obd" if args.split.startswith("obd") else "sim"
    split = PROTOCOL["splits"][SPLIT_KEY[args.split]]
    v2 = args.split.endswith("2")
    tuned = None
    if args.split in ("test", "obd", "test2", "obd2"):  # frozen config only; OBD is never tuned on
        assert PROTOCOL["status"] == "FROZEN", "protocol must be frozen before touching test"
        tuned = json.loads((ROOT / ("results/tuned_v2.json" if v2 else "results/tuned.json")).read_text())
    grid = method_grid(("test2" if v2 else "test") if dataset == "obd" else args.split, tuned)
    kind, D = args.basis.split(":")
    grid["basis"] = (kind, int(D))
    cells = [(r, i, None) for r, ints in split["regimes"].items() for i in ints]
    cells += [(r, i, V2["misspec_high"]) for r, ints in split.get("misspec_regimes", {}).items() for i in ints]
    if args.cells:
        def parse(c):
            c, _, m = c.partition("#m")
            return c.split("@")[0], float(c.split("@")[1]), float(m) if m else None
        cells = [parse(c) for c in args.cells]
    reps = args.reps or split["data_seeds"]["count"]
    start = split["data_seeds"]["start"]

    suffix = "" if args.n_anchor == SIM["n_anchor_default"] else f"_anchor{args.n_anchor}"
    path = ROOT / (args.out or f"results/03_{args.split}{suffix}.json")
    out = dict(split=args.split, n_anchor=args.n_anchor, reps_per_env=reps, env_seeds=split["env_seeds"],
               grid={k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in grid.items()}, cells={})
    with ProcessPoolExecutor() as pool:
        for regime_name, intensity, misspec in cells:
            n_env = len(split["env_seeds"])
            truth = dict(zip(split["env_seeds"], pool.map(truths_for, [regime_name] * n_env, [intensity] * n_env,
                                                           split["env_seeds"], [dataset] * n_env, [misspec] * n_env)))
            jobs = [(regime_name, intensity, e, start + s, args.n_anchor, truth[e], grid, dataset, misspec)
                    for e in split["env_seeds"] for s in range(reps)]
            results = list(pool.map(run_one, jobs))
            key = cell_key(regime_name, intensity, misspec)
            out["cells"][key] = dict(
                truths={str(e): [t[0] for t in truth[e]] for e in split["env_seeds"]},
                env_of_rep=[j[2] for j in jobs], reps=results)
            rmse = {m: np.sqrt(np.mean([[p["err"] ** 2 for p in r[m]] for r in results]))
                    for m in results[0]}
            best = sorted(rmse.items(), key=lambda kv: kv[1])[:6]
            print(f"{key:10s} n={len(results)}  best: " + "  ".join(f"{m}={v:.4f}" for m, v in best), flush=True)
            path.with_suffix(".partial.json").write_text(json.dumps(out))  # per-cell checkpoint

    path.write_text(json.dumps(out))
    path.with_suffix(".partial.json").unlink(missing_ok=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
