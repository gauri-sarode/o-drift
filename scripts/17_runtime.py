"""Wall-clock cost per episode of O-Drift and of the main baselines (post-hoc; decision log 39).

Each method runs end to end on its own (its own reward models, ratio models and monitor; nothing
shared), single process, on the same episodes: S4@1.0 on dev2 seeds (simulator) and the Men
campaign of the real-reward Open Bandit task (timing only; no estimate is reported). The reward
model basis is chosen once per episode by the v3 rule and timed separately ("basis choice"),
because every reward-model method needs it. Method order is shuffled per episode and each episode
is timed REPEATS times; the table reports the median over episodes of the per-episode median.

    PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python scripts/17_runtime.py   # machine otherwise idle
"""
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from odrift import baselines as bl
from odrift.method import (build_components, fit_all_nuisances, odrift_balanced, ratio_function)
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.nuisance import crossfit_rewards, set_basis
from odrift.simulator import LoggedData, mu

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/03_evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
TUNED = json.loads((ROOT / "results/tuned_v6.json").read_text())
N_EP, REPEATS = 5, 3
TASKS = {"Simulator (S4@1.0)": ("S4", 1.0, [0, 1, 2, 3, 0], "sim"),
         "Open Bandit, real rewards (Men)": ("REAL", 0.0, [12, 13, 14, 15, 16], "real")}


def setup(regime_name, intensity, env_seed, data_seed, dataset):
    env = ev.make_env(env_seed, dataset)
    regime = ev.get_regime(regime_name, intensity)
    pi = ev.policies(env, env_seed)[0]
    ep = ev.make_episode(env, regime, ev.SIM["n_anchor_default"], data_seed, dataset)
    arts = [sp if sp.recorded_logger_id is None else replace(sp, logger_id=sp.recorded_logger_id)
            for sp in list(regime.windows) + [regime.target]]
    logger_fns_all = [lambda X, a=a: mu(env, a, X) for a in arts]
    return env, regime, pi, ep, logger_fns_all


def methods(env, regime, pi, ep, logger_fns_all, seed):
    K, A, CFG = regime.n_windows, env.n_actions, ev.CFG
    hist, anc, Xt = ep.history, ep.anchor, ep.target_contexts
    pooled = lambda: LoggedData.concat(hist + [anc])

    def pooled_dr():
        d = pooled()
        q = crossfit_rewards(d, A, seed=seed)[0]
        return bl._weighted_estimate(bl.dr_scores(d, pi, A, q_hat=q), np.ones(len(d))).value

    def slope():
        d = pooled()
        q = crossfit_rewards(d, A, seed=seed)[0]
        v, var = bl.nested_family(bl.dr_scores(d, pi, A, q_hat=q), d.window, K)
        return v[bl.slope_select(v, var, TUNED["slope"])]

    def transported(h_fn):
        d = pooled()
        q_own, q_tgt = crossfit_rewards(d, A, seed=seed, extra_X=Xt)
        ratio_fns = [ratio_function(w.X, Xt) for w in hist] + [None]
        return bl.window_weighted_balanced_dr(d, q_own, q_tgt, Xt, pi, K, h_fn(d), logger_fns_all, ratio_fns).value

    def opfv():
        d = pooled()
        q_own, q_tgt = bl.fit_time_reward(d, K, Xt, A, TUNED["opfv"], seed)
        return bl.opfv(d, q_own, q_tgt, Xt, pi, K, bl.opfv_features(K))[0].value

    def transport_dr():
        nu = fit_all_nuisances(hist, anc, Xt, A, replace(CFG, context_alpha=0.0), seed)
        comps, a = build_components(hist, anc, Xt, pi, A, CFG, nuisances=nu)
        return bl.transport_pool(comps, a, CFG).value

    def odrift():
        m, za, _ = TUNED["odrift"]
        segs = reward_changepoints(ResidualStats(hist, A), m)
        cfg = replace(CFG, joint_segments=True, offset_scale=TUNED["odrift3"], z_anchor=za)
        nu = fit_all_nuisances(hist, anc, Xt, A, cfg, seed, segments=segs)
        return odrift_balanced(hist, anc, Xt, pi, A, cfg, nu, segs, logger_fns_all[:-1]).value

    return {"Pooled DR": pooled_dr, "SLOPE": slope,
            "COPE": lambda: transported(lambda d: bl.cope_cluster(d, K, TUNED["cope"])),
            "Transport bal. decay": lambda: transported(lambda d: TUNED["tbdecay"] ** (K - np.arange(K + 1))),
            "OPFV": opfv, "Transport DR": transport_dr, "O-Drift": odrift}


def main():
    rng = np.random.default_rng(0)
    out = {}
    for task, (rname, inten, envs, dataset) in TASKS.items():
        times = {}
        for i, e in enumerate(envs):
            seed = 90_000 + i  # outside every protocol split's data seeds
            env, regime, pi, ep, lf = setup(rname, inten, e, seed, dataset)
            t0 = time.perf_counter()
            basis = ev.select_basis(ep.history, env.n_actions, seed, 32)
            times.setdefault("Basis choice (all methods)", []).append(time.perf_counter() - t0)
            set_basis(*basis)
            fns = methods(env, regime, pi, ep, lf, seed)
            per = {n: [] for n in fns}
            for _ in range(REPEATS):
                for n in rng.permutation(list(fns)):
                    t0 = time.perf_counter()
                    fns[n]()
                    per[n].append(time.perf_counter() - t0)
            for n, ts in per.items():
                times.setdefault(n, []).append(float(np.median(ts)))
            print(f"{task} episode {i}: " + "  ".join(f"{n} {np.median(ts):.2f}s" for n, ts in per.items()), flush=True)
        out[task] = {n: float(np.median(ts)) for n, ts in times.items()}
    (ROOT / "results/17_runtime.json").write_text(json.dumps(out, indent=2))

    names = list(next(iter(out.values())))
    tex = "\n".join([
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Wall-clock seconds per episode and policy (median of " + str(N_EP) + r" episodes, single core), "
        r"each method end to end with its own models. The reward-model basis choice is common to all methods "
        r"and listed separately.}",
        r"\label{tab:runtime}",
        r"\begin{tabular}{@{}l" + "r" * len(out) + r"@{}}\toprule",
        "Method & " + " & ".join(t.split(" (")[0] if "Simulator" in t else "Open Bandit" for t in out) + r" \\\midrule",
        *[(r"\odrift{}" if n == "O-Drift" else n) + " & " + " & ".join(f"{out[t][n]:.2f}" for t in out) + r" \\"
          for n in names],
        r"\bottomrule\end{tabular}", r"\end{table}"])
    (ROOT / "paper/tables/tab_runtime.tex").write_text(tex + "\n")
    print("wrote results/17_runtime.json, paper/tables/tab_runtime.tex")


if __name__ == "__main__":
    main()
