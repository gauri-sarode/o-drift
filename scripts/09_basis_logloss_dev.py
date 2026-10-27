"""Dev-only: held-out log-loss of the pooled per-action reward model for each context basis
(nuisance quality; method-agnostic), on dev2 episodes (one per env seed and data seed < N)."""
import sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("ev", "scripts/03_evaluate.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from odrift.nuisance import RewardModel, set_basis
from odrift.regimes import REGIMES
from odrift.simulator import LoggedData

BASES = [("linear", 0), ("rff", 16), ("rff", 32), ("rff", 64)]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

def one(args):
    rg, inten, ms, e, s = args
    env = ev.make_env(e, "sim", ms); regime = REGIMES[rg](inten, K=12)
    ep = ev.make_episode(env, regime, 1000, s)
    data = LoggedData.concat(ep.history)
    fold = np.random.default_rng(s).permutation(len(data)) % 2 == 0
    out = {}
    for kind, D in BASES:
        set_basis(kind, D, 0)
        t = time.time()
        m = RewardModel(env.n_actions).fit(data.subset(fold))
        q = m.predict_all(data.X[~fold])[np.arange((~fold).sum()), data.A[~fold]]
        r = data.R[~fold]
        out[f"{kind}{D}"] = (float(-np.mean(r * np.log(q) + (1 - r) * np.log(1 - q))), time.time() - t)
    return f"{rg}@{inten}" + ("" if ms is None else f"#m{ms}"), out

if __name__ == "__main__":
    cells = [("S0", 0.0, None), ("S3a", 0.5, None), ("S4", 0.5, None), ("S0", 0.0, 3.0), ("S3a", 0.5, 3.0), ("S4", 0.5, 3.0)]
    jobs = [(rg, i, m, e, s) for rg, i, m in cells for e in range(4) for s in range(N)]
    with ProcessPoolExecutor() as p:
        res = list(p.map(one, jobs))
    names = [f"{k}{D}" for k, D in BASES]
    print(f"{'cell':14s}" + "".join(f"{n:>16s}" for n in names))
    allv = {n: [] for n in names}
    for c in dict.fromkeys(r[0] for r in res):
        rows = [r[1] for r in res if r[0] == c]
        line = f"{c:14s}"
        for n in names:
            ll = np.mean([x[n][0] for x in rows]); tt = np.mean([x[n][1] for x in rows]); allv[n].append(ll)
            line += f"{ll:10.5f} {tt:4.1f}s"
        print(line)
    print(f"{'mean':14s}" + "".join(f"{np.mean(allv[n]):10.5f}      " for n in names))
