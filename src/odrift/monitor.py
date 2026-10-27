"""Reward-drift monitor: detect changes in the conditional reward q_k(x, a)
from per-action residuals, rather than from the scalar policy value.

A single pooled reward model (per-action quadratic logistic) is fit on all
historical windows. Under no reward drift, residuals R - q_hat(X, A) have
the same conditional mean in every window; reward drift shows up as
window-dependent per-action residual shifts. Testing all actions jointly
(chi-square) accumulates evidence that the scalar value test averages away.
"""

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from .simulator import LoggedData


def _quad_features(X: np.ndarray, A: np.ndarray, n_actions: int) -> np.ndarray:
    onehot = np.eye(n_actions)[A]
    feats = [onehot, (X[:, :, None] * onehot[:, None, :]).reshape(len(A), -1),
             ((X ** 2)[:, :, None] * onehot[:, None, :]).reshape(len(A), -1)]
    return np.hstack(feats)


class ResidualStats:
    """Per-(window, action) residual sums and variance sums from a pooled model.

    kind (grouping ablation, PLAN 31): "residual" is the monitor (per-action residuals of the
    pooled quadratic model: invariant to context and logger drift under correct specification).
    The mechanism-blind statistics keep the same test and segmentation: "action" compares
    per-action raw mean rewards (accounts for logger drift, not context drift); "mean" compares
    the raw mean reward over all actions (1 df; COPE's similarity statistic).
    """

    def __init__(self, history: list, n_actions: int, C: float = 1.0, kind: str = "residual"):
        pooled = LoggedData.concat(history)
        if kind == "residual":
            model = LogisticRegression(C=C, max_iter=2000, fit_intercept=False)
            F = _quad_features(pooled.X, pooled.A, n_actions)
            model.fit(F, pooled.R)
            p = model.predict_proba(F)[:, 1]
        elif kind == "action":
            p = (np.bincount(pooled.A, pooled.R, n_actions) / np.maximum(np.bincount(pooled.A, minlength=n_actions), 1))[pooled.A]
        elif kind == "mean":
            p = np.full(len(pooled), pooled.R.mean())
        else:
            raise ValueError(kind)
        self.pool_actions = kind == "mean"
        resid, var = pooled.R - p, p * (1 - p)
        K = len(history)
        self.sum = np.zeros((K, n_actions))
        self.var = np.zeros((K, n_actions))
        self.n = np.zeros((K, n_actions))
        np.add.at(self.sum, (pooled.window, pooled.A), resid)
        np.add.at(self.var, (pooled.window, pooled.A), var)
        np.add.at(self.n, (pooled.window, pooled.A), 1)

    def block_stat(self, left: list, right: list) -> tuple:
        """Chi-square statistic for equal per-action mean residuals in two blocks."""
        S1, S2 = self.sum[left].sum(0), self.sum[right].sum(0)
        V1, V2 = self.var[left].sum(0), self.var[right].sum(0)
        n1, n2 = self.n[left].sum(0), self.n[right].sum(0)
        if self.pool_actions:
            S1, S2, V1, V2, n1, n2 = (np.atleast_1d(a.sum()) for a in (S1, S2, V1, V2, n1, n2))
        ok = (n1 > 10) & (n2 > 10)
        m1 = S1[ok] / n1[ok]
        m2 = S2[ok] / n2[ok]
        v = V1[ok] / n1[ok] ** 2 + V2[ok] / n2[ok] ** 2
        return float(((m1 - m2) ** 2 / v).sum()), int(ok.sum())


def reward_changepoints(stats: ResidualStats, alpha: float) -> list:
    """Binary segmentation of windows using the per-action residual test.

    A segment is split at its best point if the chi-square p-value, Bonferroni-
    corrected over candidate split points, is below alpha. Returns segments
    newest-first (same contract as method.changepoint_groups).
    """
    K = stats.sum.shape[0]
    segments, stack = [], [list(range(K))]
    while stack:
        seg = stack.pop()
        best_p, best_s = 1.0, None
        for s in range(1, len(seg)):
            t, df = stats.block_stat(seg[:s], seg[s:])
            p = chi2.sf(t, df) if df else 1.0
            if p < best_p:
                best_p, best_s = p, s
        if best_s is not None and best_p * (len(seg) - 1) < alpha:
            stack += [seg[:best_s], seg[best_s:]]
        else:
            segments.append(seg)
    return sorted(segments, key=lambda g: -max(g))
