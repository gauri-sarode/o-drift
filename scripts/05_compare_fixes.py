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
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, Policy, sample_episode

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
SIM, DEV, TUNE = PROTOCOL["simulator"], PROTOCOL["splits"]["dev"], PROTOCOL["tuning"]
CFG = ODriftConfig(delta_variance=TUNE["delta_split"]["variance"], delta_bias=TUNE["delta_split"]["bias"])
Z_ANCHOR = [2.0, 2.5, 3.0]
VARIANTS = list(itertools.product(Z_ANCHOR, ["anchor", "least"], [False, True]))


def name(za, fb, sn):
    return f"za={za},fb={fb},sn={int(sn)}"


def run_one(args):
    regime_name, intensity, env_seed, data_seed, truths, z_group = args
    env = EnvParams.sample(env_seed, d=SIM["d"], n_actions=SIM["n_actions"], misspec=SIM["misspec"])
    regime = REGIMES[regime_name](intensity, K=SIM["n_windows"])
    pis = [Policy.near_optimal(env, noise=s, seed=env_seed) for s in TUNE["candidate_policies"]]
    ep = sample_episode(env, regime, SIM["n_per_window"], SIM["n_anchor_default"], SIM["n_target_contexts"],
                        seed=data_seed)
    nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, env.n_actions, CFG, data_seed)
    out = {}
    for p_idx, pi in enumerate(pis):
        for sn in (False, True):
            cfg = replace(CFG, z_group=z_group, anchor_sn=sn)
            comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, cfg,
                                          nuisances=nuis)
            for za, fb in itertools.product(Z_ANCHOR, ["anchor", "least"]):
                r = combine(comps, anc, replace(cfg, z_anchor=za, fallback=fb))
                out.setdefault(name(za, fb, sn), []).append(dict(err=r.value - truths[p_idx], hw=r.half_width,
                                                                  anchor_only=r.weights.get(len(comps), 0) > 0.999))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=DEV["data_seeds"]["count"])
    args = ap.parse_args()
    dev = json.loads((ROOT / "results/03_dev.json").read_text())
    tuned = json.loads((ROOT / "results/tuned.json").read_text())
    z_group = tuned["odrift"][0]
    start = DEV["data_seeds"]["start"]
    summary = {}
    with ProcessPoolExecutor() as pool:
        for key, cell in dev["cells"].items():
            regime_name, intensity = key.split("@")[0], float(key.split("@")[1])
            jobs = [(regime_name, intensity, e, start + s, cell["truths"][str(e)], z_group)
                    for e in DEV["env_seeds"] for s in range(args.reps)]
            res = list(pool.map(run_one, jobs))
            summary[key] = {}
            for v in res[0]:
                err = np.array([[p["err"] for p in r[v]] for r in res])
                summary[key][v] = dict(rmse=float(np.sqrt((err ** 2).mean())),
                                       anchor_only=float(np.mean([[p["anchor_only"] for p in r[v]] for r in res])))
            print(key, flush=True)
    # mean log RMSE across dev cells -- the protocol's selection criterion
    score = {v: float(np.mean([np.log(summary[k][v]["rmse"]) for k in summary])) for v in summary[next(iter(summary))]}
    print(f"\n{'variant':28s}" + "".join(f"{k:>11s}" for k in summary) + "   meanlog")
    for v in sorted(score, key=score.get):
        print(f"{v:28s}" + "".join(f"{summary[k][v]['rmse'] * 1e3:11.2f}" for k in summary) + f"   {score[v]:.3f}")
    print("(RMSE x1e3; fraction of anchor-only outputs stored in results/05_compare_fixes.json)")
    (ROOT / "results/05_compare_fixes.json").write_text(json.dumps(dict(summary=summary, score=score), indent=2))


if __name__ == "__main__":
    main()
