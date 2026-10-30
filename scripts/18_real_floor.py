"""Post-hoc (decision log 39): the real-reward Open Bandit test (real6) with the anchor variance floor.

In 4 of the 420 real6 episodes the 1,000-record anchor has no click, its component variance is
estimated as 0, and O-Drift returns no estimate. This reruns the frozen O-Drift (odrift3[0.25]) on
the same episodes with and without ODriftConfig.anchor_var_floor. The run without the floor must
reproduce results/03_real6.json exactly (checked); every other method is taken from that file.

    PYTHONPATH=src .venv/bin/python scripts/18_real_floor.py
"""
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from odrift.method import fit_all_nuisances, odrift_balanced
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.nuisance import set_basis
from odrift.simulator import mu

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/03_evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
spec16 = importlib.util.spec_from_file_location("real", ROOT / "scripts/16_real_analysis.py")
real = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(real)
TUNED = json.loads((ROOT / "results/tuned_v6.json").read_text())
OD = f"odrift3[{TUNED['odrift3']}]"


def run(args):
    env_seed, data_seed = args
    env = ev.make_env(env_seed, "real")
    regime = ev.get_regime("REAL", 0.0)
    pi = ev.policies(env, env_seed)[0]
    ep = ev.make_episode(env, regime, ev.SIM["n_anchor_default"], data_seed, "real")
    set_basis(*ev.select_basis(ep.history, env.n_actions, data_seed, 32))
    A = env.n_actions
    m, za, _ = TUNED["odrift"]
    segs = reward_changepoints(ResidualStats(ep.history, A), m)
    nu = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, A,
                           replace(ev.CFG, joint_segments=True, offset_scale=TUNED["odrift3"]), data_seed, segments=segs)
    logger_fns = [lambda X, a=a: mu(env, a, X) for a in regime.windows]
    out = {}
    for floor in (False, True):
        r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, pi, A,
                            replace(ev.CFG, z_anchor=za, offset_scale=TUNED["odrift3"], anchor_var_floor=floor),
                            nu, segs, logger_fns)
        se = r.se_hetero if r.se_hetero is not None else r.se
        out["floor" if floor else "frozen"] = dict(est=float(r.value), hw=float(ev.Z95 * se))
    out["anchor_clicks"] = int(ep.anchor.R.sum())
    return out


def main():
    cell = json.loads((ROOT / "results/03_real6.json").read_text())["cells"]["REAL@0.0"]
    split = ev.PROTOCOL["splits"]["real6"]
    reps, start = split["data_seeds"]["count"], split["data_seeds"]["start"]
    jobs = [(e, start + s) for e in split["env_seeds"] for s in range(reps)]
    assert [j[0] for j in jobs] == cell["env_of_rep"]
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(run, jobs))
    stored = np.array([r[OD][0]["est"] for r in cell["reps"]])
    frozen = np.array([r["frozen"]["est"] for r in res])
    same = np.isfinite(stored) == np.isfinite(frozen)
    assert same.all() and np.allclose(stored[np.isfinite(stored)], frozen[np.isfinite(frozen)], rtol=0, atol=1e-12), \
        "rerun without the floor does not reproduce results/03_real6.json"
    floor = np.array([r["floor"]["est"] for r in res])
    changed = ~np.isclose(floor, frozen, rtol=0, atol=1e-12) | ~np.isfinite(frozen)
    print(f"reproduced frozen O-Drift on all {len(res)} episodes; floor changes {changed.sum()} episodes "
          f"(anchor clicks there: {[res[i]['anchor_clicks'] for i in np.flatnonzero(changed)]}); "
          f"non-finite with floor: {(~np.isfinite(floor)).sum()}")
    # all 420 episodes: O-Drift with the floor vs every baseline that returned an estimate in all of them
    envs = np.array(cell["env_of_rep"])
    truth = np.array([cell["truths"][str(e)][0] for e in envs])
    # the tuned baselines of scripts/16_real_analysis.py (not the oracle-tuned recent/decay grids)
    bases = [m for m in real.BASELINES if all(np.isfinite(r[m][0]["est"]) for r in cell["reps"])]
    sq = {m: np.array([(r[m][0]["est"] - t) ** 2 for r, t in zip(cell["reps"], truth)]) for m in bases}
    sq["odrift_floor"] = (floor - truth) ** 2
    rm = {m: float(np.sqrt(v.mean())) for m, v in sq.items()}
    best = min(bases, key=rm.get)
    tasks = np.unique(envs)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10_000):
        ts = rng.choice(tasks, len(tasks))
        sel = np.concatenate([np.flatnonzero(envs == t) for t in ts])
        boots.append(np.sqrt(sq["odrift_floor"][sel].mean() / sq[best][sel].mean()))
    ci = np.percentile(boots, [2.5, 97.5]).tolist()
    hw = np.array([r["floor"]["hw"] for r in res])
    summary = dict(changed=int(changed.sum()), rmse=rm, best_baseline=best, ratio=rm["odrift_floor"] / rm[best],
                   ratio_ci=ci, coverage=float(np.mean(np.abs(floor - truth) <= hw)))
    print(f"all {len(res)} episodes: O-Drift+floor RMSE {rm['odrift_floor'] * 1e3:.3f}e-3, best baseline {best} "
          f"{rm[best] * 1e3:.3f}e-3, ratio {summary['ratio']:.3f} CI {np.round(ci, 3).tolist()}, "
          f"coverage {summary['coverage']:.3f}")
    (ROOT / "results/18_real6_floor.json").write_text(json.dumps(dict(summary=summary, env_of_rep=cell["env_of_rep"],
                                                                      reps=res), indent=1))
    print("wrote results/18_real6_floor.json")


if __name__ == "__main__":
    main()
