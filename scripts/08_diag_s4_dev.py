"""Dev-only diagnosis of O-DRIFT on S4@0.5 at misspec 3: per-segment component bias."""
import sys, numpy as np
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from odrift.method import ODriftConfig, fit_all_nuisances, odrift_balanced, build_components, choose_reference, balanced_segment_components
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, Policy, mu, sample_episode, true_value

MS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
RG = sys.argv[2] if len(sys.argv) > 2 else "S4"
BASIS = (sys.argv[3].split(":")[0], int(sys.argv[3].split(":")[1])) if len(sys.argv) > 3 else ("linear", 0)
CFG = ODriftConfig(z_anchor=5.0, fallback="least", context_alpha=0.01, joint_segments=True, z_group=3.5)

def one(args):
    e, s = args
    from odrift.nuisance import set_basis; set_basis(*BASIS)
    env = EnvParams.sample(seed=e, misspec=MS); regime = REGIMES[RG](0.5)
    pis = [Policy.near_optimal(env, noise=n, seed=e) for n in (0.0, 0.5, 1.0, 2.0)]
    fns = [lambda X, sp=sp: mu(env, sp, X) for sp in regime.windows]
    ep = sample_episode(env, regime, 5000, 1000, 10000, seed=s)
    segs = reward_changepoints(ResidualStats(ep.history, env.n_actions), 1e-4)
    nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, env.n_actions, CFG, s, segments=segs)
    out = []
    for pi in pis:
        vt = true_value(env, pi, regime.target, regime.target, n_mc=200_000)
        comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, CFG, nuisances=nuis)
        ref = choose_reference(comps, anc, CFG, segs)
        sc, anc_c = balanced_segment_components(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, CFG, nuis, segs, fns, ref, comps)
        r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, CFG, nuis, segs, fns)
        # uncorrected own-segment components (each segment's own DM): bias of the reference-type estimate per segment
        sc_own, _ = balanced_segment_components(ep.history, ep.anchor, ep.target_contexts, pi, env.n_actions, CFG, nuis, segs, fns, None, comps)
        out.append(dict(segs=[sorted(g) for g in segs], ref=sorted(ref) if ref else None,
                        comp={c.index - 1000: (c.value - vt, np.sqrt(c.var), c.diagnostics.ess) for c in sc},
                        own={c.index - 1000: c.value - vt for c in sc_own},
                        err=r.value - vt, w={c.index - 1000: r.weights for c in sc}, anc=anc.value - vt))
    return out

if __name__ == "__main__":
    jobs = [(e, s) for e in range(4) for s in range(8)]
    with ProcessPoolExecutor() as p:
        res = list(p.map(one, jobs))
    from collections import Counter
    print("segmentations:", Counter(str(r[0]["segs"]) for r in res).most_common(4))
    print("refs:", Counter(str(r[0]["ref"]) for r in res).most_common(3))
    errs = np.array([[q["err"] for q in r] for r in res]); print(f"O-DRIFT rmse {1e3*np.sqrt((errs**2).mean()):.1f} bias {1e3*errs.mean():+.1f}")
    # per segment-position (0 = newest) bias of corrected component and of own-DM component
    for j in range(3):
        cb = [q["comp"][j][0] for r in res for q in r if j in q["comp"]]
        ob = [q["own"][j] for r in res for q in r if j in q["own"]]
        sd = [q["comp"][j][1] for r in res for q in r if j in q["comp"]]
        ess = [q["comp"][j][2] for r in res for q in r if j in q["comp"]]
        wt = [np.mean(q["w"][j]) if np.ndim(q["w"][j]) else q["w"][j] for r in res for q in r if j in q["w"]]
        if cb: print(f"seg {j}: n={len(cb)} corrected bias {1e3*np.mean(cb):+.1f} rmse {1e3*np.sqrt(np.mean(np.square(cb))):.1f} (model sd {1e3*np.mean(sd):.1f}, ess {np.mean(ess):.0f}) | own-DM bias {1e3*np.mean(ob):+.1f}")
    a = [q["anc"] for r in res for q in r]; print(f"anchor bias {1e3*np.mean(a):+.1f} rmse {1e3*np.sqrt(np.mean(np.square(a))):.1f}")
    for e in range(4):
        sub = errs[[i for i, (ee, s) in enumerate(jobs) if ee == e]]
        print(f"env {e}: rmse {1e3*np.sqrt((sub**2).mean()):.1f} bias by policy", np.round(1e3*sub.mean(0), 1))
