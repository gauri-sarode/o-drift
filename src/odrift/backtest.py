"""v5: rolling-origin back-test to choose how much history to reuse (per episode).

The anchor is too small to validate drift-corrected reuse (v4, PLAN 34). A history window
is not: pretend window o (o = K-1, K-2) is the target, split it into a pseudo-anchor of the
anchor's size and a validation part, run every candidate estimator on windows < o, and
score it against the DR value of the validation part. The candidate with the smallest mean
squared back-test error over origins and candidate policies is used for the real target.
Candidates are fixed a priori (protocol v5); nothing here is tuned.
"""
from dataclasses import replace

import numpy as np

from . import baselines as bl
from .method import ODriftConfig, fit_all_nuisances, odrift_balanced
from .monitor import ResidualStats, reward_changepoints
from .nuisance import crossfit_rewards
from .simulator import LoggedData

CANDIDATES = ("reuse", "shrink", "reference")  # v1 corrected reuse | offset_scale 0.25 | newest segment only


def candidate_estimates(history: list, anchor: LoggedData, X_target: np.ndarray, pis: list, n_actions: int,
                        cfg: ODriftConfig, logger_fns: list, seed: int, monitor_alpha: float,
                        shrink: float) -> dict:
    """Each candidate's estimate for each policy (policy-independent fits shared)."""
    segs = reward_changepoints(ResidualStats(history, n_actions), monitor_alpha)
    nu1 = fit_all_nuisances(history, anchor, X_target, n_actions, replace(cfg, joint_segments=True), seed, segments=segs)
    nus = fit_all_nuisances(history, anchor, X_target, n_actions,
                            replace(cfg, joint_segments=True, offset_scale=shrink), seed, segments=segs)
    ref = LoggedData.concat([history[k] for k in sorted(segs[0])] + [anchor])
    q_ref = crossfit_rewards(ref, n_actions, seed=seed)[0]
    out = {c: [] for c in CANDIDATES}
    for pi in pis:
        out["reuse"].append(odrift_balanced(history, anchor, X_target, pi, n_actions, replace(cfg, joint_segments=True),
                                            nu1, segs, logger_fns).value)
        out["shrink"].append(odrift_balanced(history, anchor, X_target, pi, n_actions,
                                             replace(cfg, joint_segments=True, offset_scale=shrink),
                                             nus, segs, logger_fns).value)
        out["reference"].append(bl._weighted_estimate(bl.dr_scores(ref, pi, n_actions, q_hat=q_ref),
                                                      np.ones(len(ref))).value)
    return out


def backtest_choice(history: list, n_anchor: int, pis: list, n_actions: int, cfg: ODriftConfig,
                    logger_fns: list, seed: int, monitor_alpha: float = 1e-4, shrink: float = 0.25,
                    origins: int = 2) -> tuple:
    """Return (chosen candidate, {candidate: mean squared back-test error})."""
    K = len(history)
    sq = {c: [] for c in CANDIDATES}
    for o in range(K - 1, K - 1 - origins, -1):
        win = history[o]
        perm = np.random.default_rng(seed + 7919 * o).permutation(len(win))
        n_a = min(n_anchor, len(win) // 3)
        p_anchor, val = win.subset(perm[:n_a]), win.subset(perm[n_a:])
        q_val = crossfit_rewards(val, n_actions, seed=seed + o)[0]
        truth = [bl.dr_scores(val, pi, n_actions, q_hat=q_val).mean() for pi in pis]
        est = candidate_estimates(history[:o], p_anchor, val.X, pis, n_actions, cfg, logger_fns[:o],
                                  seed + o, monitor_alpha, shrink)
        for c in CANDIDATES:
            sq[c] += [(e - t) ** 2 for e, t in zip(est[c], truth)]
    mse = {c: float(np.mean(v)) for c, v in sq.items()}
    return min(CANDIDATES, key=lambda c: mse[c]), mse
