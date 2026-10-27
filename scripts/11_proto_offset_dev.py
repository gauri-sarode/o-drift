"""Dev-only prototype (PLAN 31): segment-offset shrinkage (offset_scale) for O-DRIFT v1 with the
per-episode basis (linear at misspec 0.5, rff32 at misspec 3 -- what log-loss selection picks, PLAN 30);
reuse variants on dev2 cells, same episodes/seeds as 03_evaluate,
compared with the dev2 baselines on matching replications."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("ev", "scripts/03_evaluate.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from odrift.method import fit_all_nuisances, odrift_balanced
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.regimes import REGIMES
from odrift.simulator import mu

VARIANTS = {f"off{v}": dict(offset_scale=v) for v in (1.0, 0.5, 0.25, 0.1)}
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 12
LIN, RFF = (json.load(open(f"results/03_dev2_{b}.json")) for b in ("linear", "rff32"))
DEV = LIN

def one(args):
    key, rg, inten, ms, e, s, truths = args
    from odrift.nuisance import set_basis
    set_basis(*(("rff", 32) if ms else ("linear", 0)))
    env = ev.make_env(e, "sim", ms); regime = REGIMES[rg](inten, K=12)
    pis = ev.policies(env, e)
    ep = ev.make_episode(env, regime, 1000, s)
    segs = reward_changepoints(ResidualStats(ep.history, env.n_actions), 1e-4)
    cfg = replace(ev.CFG, z_anchor=5.0, joint_segments=True)
    fns = [lambda X, a=a: mu(env, a, X) for a in regime.windows]
    out = {}
    for name, kw in VARIANTS.items():
        nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, env.n_actions, replace(cfg, **kw), s, segments=segs)
        out[name] = [odrift_balanced(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, replace(cfg, **kw),
                                     nuis, segs, fns).value - truths[i][0] for i, pi in enumerate(pis)]
    return key, e, s, out

if __name__ == "__main__":
    jobs = []
    for key, c in DEV["cells"].items():
        base, _, m = key.partition("#m"); rg, inten = base.split("@")
        for e in DEV["env_seeds"]:
            tr = [(t, None) for t in c["truths"][str(e)]]
            for s in range(NSEED):
                jobs.append((key, rg, float(inten), float(m) if m else None, e, s, tr))
    with ProcessPoolExecutor() as p:
        res = list(p.map(one, jobs))
    ref = ["pooled_dr", "recent_dr[12]", "decay_dr[0.8]", "minvar_dr", "transport_dr", "slope[3.0]", "magic[1.96]",
           "trecent[8]", "tdecay[0.8]", "timedr[1]", "tbdecay[0.8]", "balanced_dr", "monitor_select_dr"]
    names = list(VARIANTS) + ["best_base"]
    print(f"{'cell':14s}" + "".join(f"{n:>11s}" for n in names))
    logs = {n: [] for n in names}
    for key, c in DEV["cells"].items():
        c = (RFF if "#m" in key else LIN)["cells"][key]
        errs = {n: [] for n in VARIANTS}; base_errs = {b: [] for b in ref}
        for k2, e, s, out in res:
            if k2 != key: continue
            i = next(i for i, ee in enumerate(c["env_of_rep"]) if ee == e) + s  # rep index: env-major, seeds 0..
            for n in VARIANTS: errs[n].append(out[n])
            for b in [b for b in ref if b in c["reps"][i]]: base_errs[b].append([p["err"] for p in c["reps"][i][b]])
        r = {n: np.sqrt(np.mean(np.square(v))) * 1e3 for n, v in errs.items()}
        rb = min(np.sqrt(np.mean(np.square(v))) * 1e3 for v in base_errs.values() if v)
        r["best_base"] = rb
        for n in names: logs[n].append(np.log(r[n]))
        print(f"{key:14s}" + "".join(f"{r[n]:11.1f}" for n in names))
    print(f"{'mean log':14s}" + "".join(f"{np.mean(logs[n]):11.3f}" for n in names))
