"""Dev-only, post-hoc (PLAN 30): per-episode reward-model basis selection by held-out log-loss.

For every dev2 episode, fit the pooled per-action reward model on half of the history with the
linear and the rff32 basis, score log-loss on the other half, and pick the lower. Then every
method's estimate is taken from the run with the chosen basis (results/03_dev2_{linear,rff32}.json,
same episodes), and the RMSE ratio of O-DRIFT to the best tuned baseline is recomputed.
"""
import json
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("ev", "scripts/03_evaluate.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from odrift.nuisance import RewardModel, set_basis
from odrift.regimes import REGIMES
from odrift.simulator import LoggedData

BASES = [("linear", 0), ("rff", 32)]
BASELINES = ["pooled_dr", "anchor_dr", "recent_dr[12]", "decay_dr[0.8]", "minvar_dr", "liu_reg[6]", "transport_dr",
             "slope[3.0]", "magic[1.96]", "trecent[8]", "tdecay[0.8]", "forecast[1]", "timedr[1]", "tbdecay[0.8]",
             "anchor_validated", "balanced_dr", "monitor_select_dr"]
OD = ["odrift[0.0001,5.0,1]", "odrift2[0.5]"]


def parse(key):
    rg, rest = key.split("@")
    inten, _, ms = rest.partition("#m")
    return rg, float(inten), (float(ms) if ms else None)


def logloss(args):
    key, e, s, n_anchor = args
    rg, inten, ms = parse(key)
    env = ev.make_env(e, "sim", ms)
    ep = ev.make_episode(env, REGIMES[rg](inten, K=12), n_anchor, s)
    data = LoggedData.concat(ep.history)
    fold = np.random.default_rng(s).permutation(len(data)) % 2 == 0
    out = []
    for kind, D in BASES:
        set_basis(kind, D, 0)
        q = RewardModel(env.n_actions).fit(data.subset(fold)).predict_all(data.X[~fold])
        q = q[np.arange((~fold).sum()), data.A[~fold]]
        r = data.R[~fold]
        out.append(float(-np.mean(r * np.log(q) + (1 - r) * np.log(1 - q))))
    return out


def errs(rep, m):
    return np.array([p["err"] for p in rep[m]])


def main():
    lin, rff = (json.load(open(f"results/03_dev2_{b}.json")) for b in ("linear", "rff32"))
    reps, start = lin["reps_per_env"], 0
    jobs = [(k, e, start + i % reps, lin["n_anchor"]) for k, c in lin["cells"].items()
            for i, e in enumerate(c["env_of_rep"])]
    with ProcessPoolExecutor() as p:
        ll = list(p.map(logloss, jobs, chunksize=4))
    pos = 0
    print(f"{'cell':14s} {'rff%':>5s} " + " ".join(f"{b:>8s}" for b in ["lin", "rff", "select"]) + "   (O-DRIFT v1 / v2[0.5] ratio to best tuned baseline)")
    for k in lin["cells"]:
        cl, cr = lin["cells"][k]["reps"], rff["cells"][k]["reps"]
        pick = [ll[pos + i][1] < ll[pos + i][0] for i in range(len(cl))]
        pos += len(cl)
        sel = [r if p else l for l, r, p in zip(cl, cr, pick)]
        line = f"{k:14s} {100 * np.mean(pick):5.0f} "
        for reps_ in (cl, cr, sel):
            rm = {m: np.sqrt(np.mean([errs(r, m) ** 2 for r in reps_])) for m in BASELINES + OD if m in reps_[0]}
            b = min(rm[m] for m in BASELINES if m in rm)
            line += f"  {rm[OD[0]] / b:.2f}/{rm[OD[1]] / b:.2f}"
        print(line)


if __name__ == "__main__":
    main()
