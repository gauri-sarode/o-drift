"""Dev-only comparison of anchor-veto fixes (decision log item 5).

Variants: z_anchor x fallback {anchor, least} x anchor estimator {DR, SN-DR},
z_group fixed to the dev-tuned value. Same dev seeds and candidate policies as
03_evaluate, so RMSEs are directly comparable to results/03_dev.json.

    .venv/bin/python scripts/05_compare_fixes.py [--reps 50]
"""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from odrift.method import ODriftConfig, build_components, combine, fit_all_nuisances
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, Policy, sample_episode

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
SIM, DEV, TUNE = PROTOCOL["simulator"], PROTOCOL["splits"]["dev"], PROTOCOL["tuning"]
CFG = ODriftConfig(delta_variance=TUNE["delta_split"]["variance"], delta_bias=TUNE["delta_split"]["bias"])
Z_ANCHOR = [2.0, 2.5, 3.0]
FALLBACKS = ["anchor", "least"]
SN = (False, True)
Z_GROUP = None  # None -> dev-tuned value from results/tuned.json


def name(zg, za, fb, sn):
    return f"zg={zg},za={za},fb={fb},sn={int(sn)}"


def run_one(args):
    regime_name, intensity, env_seed, data_seed, truths, z_groups, z_anchors, fallbacks, sns, groupings, shrink_refs, shrink_ess, MONITOR_ALPHAS = args
    env = EnvParams.sample(env_seed, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    pis = [Policy.near_optimal(env, noise=s, seed=env_seed) for s in TUNE["candidate_policies"]]
    ep = sample_episode(env, regime, SIM["n_per_window"], SIM["n_anchor_default"], SIM["n_target_contexts"],
                        seed=data_seed)
    nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, env.n_actions, CFG, data_seed)
    stats = ResidualStats(ep.history, env.n_actions) if MONITOR_ALPHAS else None
    monitor_groups = {a: reward_changepoints(stats, a) for a in MONITOR_ALPHAS}
    out = {}
    for p_idx, pi in enumerate(pis):
        for sn in sns:
            cfg = replace(CFG, anchor_sn=sn)
            comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, cfg,
                                          nuisances=nuis)
            for gr, zg, za, fb in itertools.product(groupings, z_groups, z_anchors, fallbacks):
                r = combine(comps, anc, replace(cfg, grouping=gr, z_group=zg, z_anchor=za, fallback=fb))
                out.setdefault(f"{gr[:2]}:" + name(zg, za, fb, sn), []).append(dict(err=r.value - truths[p_idx], hw=r.half_width,
                                                                  anchor_only=r.weights.get(len(comps), 0) > 0.999))
                for a, wg in monitor_groups.items():
                    if gr != groupings[0] or zg != z_groups[0]:
                        continue  # monitor groups do not depend on grouping/z_group
                    r = combine(comps, anc, replace(cfg, z_anchor=za, fallback=fb), window_groups=wg)
                    out.setdefault(f"mon{a}:seg:za={za},fb={fb},sn={int(sn)}", []).append(
                        dict(err=r.value - truths[p_idx], hw=r.half_width, anchor_only=False))
                    r = combine(comps, anc, replace(cfg, z_anchor=za, fallback=fb, selection="shrink"), 1.0,
                                window_groups=wg)
                    out.setdefault(f"mon{a}:shrink:za={za},fb={fb},sn={int(sn)}", []).append(
                        dict(err=r.value - truths[p_idx], hw=r.half_width, anchor_only=False))
                for m in shrink_refs:
                    r = combine(comps, anc, replace(cfg, grouping=gr, z_group=zg, z_anchor=za, fallback=fb,
                                                    selection="shrink", shrink_ref=m), 1.0)
                    out.setdefault(f"shrink{m}:{gr[:2]}:" + name(zg, za, fb, sn), []).append(
                        dict(err=r.value - truths[p_idx], hw=r.half_width, anchor_only=False))
                for t in shrink_ess:
                    r = combine(comps, anc, replace(cfg, grouping=gr, z_group=zg, z_anchor=za, fallback=fb,
                                                    selection="shrink", shrink_ess=t), 1.0)
                    out.setdefault(f"shrinkESS{int(t)}:{gr[:2]}:" + name(zg, za, fb, sn), []).append(
                        dict(err=r.value - truths[p_idx], hw=r.half_width, anchor_only=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=DEV["data_seeds"]["count"])
    ap.add_argument("--z-group", type=float, nargs="*", default=None)
    ap.add_argument("--z-anchor", type=float, nargs="*", default=None)
    ap.add_argument("--fallback", nargs="*", default=None)
    ap.add_argument("--sn", type=int, nargs="*", default=None)
    ap.add_argument("--grouping", nargs="*", default=["greedy"])
    ap.add_argument("--shrink-ref", type=int, nargs="*", default=[])
    ap.add_argument("--shrink-ess", type=float, nargs="*", default=[])
    ap.add_argument("--monitor-alpha", type=float, nargs="*", default=[])
    ap.add_argument("--out", default="results/05_compare_fixes.json")
    args = ap.parse_args()
    global Z_ANCHOR, FALLBACKS, SN
    Z_ANCHOR = args.z_anchor or Z_ANCHOR
    FALLBACKS = args.fallback or FALLBACKS
    SN = tuple(bool(x) for x in args.sn) if args.sn is not None else SN
    dev = json.loads((ROOT / "results/03_dev.json").read_text())
    tuned = json.loads((ROOT / "results/tuned.json").read_text())
    z_groups = args.z_group or [tuned["odrift"][0]]
    start = DEV["data_seeds"]["start"]
    summary = {}
    with ProcessPoolExecutor() as pool:
        for key, cell in dev["cells"].items():
            regime_name, intensity = key.split("@")[0], float(key.split("@")[1])
            jobs = [(regime_name, intensity, e, start + s, cell["truths"][str(e)], z_groups, Z_ANCHOR, FALLBACKS, SN, args.grouping, args.shrink_ref, args.shrink_ess, args.monitor_alpha)
                    for e in DEV["env_seeds"] for s in range(args.reps)]
            res = list(pool.map(run_one, jobs))
            summary[key] = {}
            for v in res[0]:
                err = np.array([[p["err"] for p in r[v]] for r in res])
                summary[key][v] = dict(rmse=float(np.sqrt((err ** 2).mean())),
                                       anchor_only=float(np.mean([[p["anchor_only"] for p in r[v]] for r in res])))
            print(key, flush=True)
    # (grids are passed in the job tuple: worker processes are spawned and would not see globals)
    # mean log RMSE across dev cells -- the protocol's selection criterion
    score = {v: float(np.mean([np.log(summary[k][v]["rmse"]) for k in summary])) for v in summary[next(iter(summary))]}
    print(f"\n{'variant':44s}" + "".join(f"{k:>11s}" for k in summary) + "   meanlog")
    for v in sorted(score, key=score.get):
        print(f"{v:44s}" + "".join(f"{summary[k][v]['rmse'] * 1e3:11.2f}" for k in summary) + f"   {score[v]:.3f}")
    print("(RMSE x1e3; fraction of anchor-only outputs stored in results/05_compare_fixes.json)")
    (ROOT / args.out).write_text(json.dumps(dict(summary=summary, score=score), indent=2))


if __name__ == "__main__":
    main()
