"""Baselines with the same access to the anchor as O-DRIFT (protocol §7).

All pooled baselines include the anchor as the most recent data. Window and
decay hyperparameters are tuned on the dev split only.
"""

import numpy as np

from .estimators import Estimate, importance_weights
from .method import ODriftConfig, combine
from .nuisance import RewardModel, crossfit_rewards
from .simulator import LoggedData


def _weighted_estimate(scores: np.ndarray, h: np.ndarray) -> Estimate:
    """Weighted mean sum h_i G_i / sum h_i with weights treated as fixed."""
    w = h / h.sum()
    v = float(w @ scores)
    return Estimate(v, float(np.sqrt(((w * (scores - v)) ** 2).sum())), len(scores))


def dr_scores(data: LoggedData, pi, n_actions: int, seed: int = 0, q_hat: np.ndarray | None = None) -> np.ndarray:
    probs = pi.probs(data.X)
    if q_hat is None:
        q_hat, _ = crossfit_rewards(data, n_actions, seed=seed)
    rho = importance_weights(data, probs)
    return (probs * q_hat).sum(axis=1) + rho * (data.R - q_hat[np.arange(len(data)), data.A])


def recent_dr(history: list, anchor: LoggedData, pi, n_actions: int, n_recent: int, seed: int = 0) -> Estimate:
    """Sliding-window DR on the last `n_recent` windows plus the anchor (0 = anchor only)."""
    data = LoggedData.concat(history[len(history) - n_recent:] + [anchor]) if n_recent else anchor
    return _weighted_estimate(dr_scores(data, pi, n_actions, seed), np.ones(len(data)))


def decay_dr(history: list, anchor: LoggedData, pi, n_actions: int, gamma: float, seed: int = 0) -> Estimate:
    """Exponential-decay DR: records from window k weighted gamma^(K - k); anchor weight 1."""
    data = LoggedData.concat(history + [anchor])
    K = len(history)
    return _weighted_estimate(dr_scores(data, pi, n_actions, seed), gamma ** (K - data.window))


def minvar_dr(history: list, anchor: LoggedData, pi, n_actions: int, seed: int = 0) -> Estimate:
    """Zhan et al. (KDD 2021) adaptive weighting, non-contextual MinVar variant.

    Each window's DR scores get weight h_k = 1 / E[sum_a pi^2/mu_k], estimated
    by the window mean of rho^2 (since E_{A~mu}[(pi/mu)^2] = sum_a pi^2/mu).
    Assumes a stationary reward; adapted here with windows as the time unit.
    """
    data = LoggedData.concat(history + [anchor])
    scores = dr_scores(data, pi, n_actions, seed)
    rho2 = importance_weights(data, pi.probs(data.X)) ** 2
    h = np.empty(len(data))
    for k in np.unique(data.window):
        m = data.window == k
        h[m] = 1.0 / rho2[m].mean()
    return _weighted_estimate(scores, h)


def regression_assisted_dr(history: list, anchor: LoggedData, X_target: np.ndarray, pi, n_actions: int,
                           n_recent: int | None = None) -> Estimate:
    """Liu et al. (AISTATS 2023) regression-assisted DR for the current period.

    A reward model r_hat is fit on past windows (the last `n_recent`, or all).
    With phi(x, a) = (1, r_hat(x, a)), coefficients beta are fit on the
    anchor by importance-weighted least squares (their Eq. 3); the population
    total of phi^T beta is taken over current target contexts and the
    correction uses only anchor rewards (their Eq. 4), so old rewards never
    enter the correction term.
    """
    past = history if n_recent is None else history[len(history) - n_recent:]
    r_hat = RewardModel(n_actions).fit(LoggedData.concat(past))
    probs = pi.probs(anchor.X)
    rho = importance_weights(anchor, probs)
    idx = np.arange(len(anchor))
    r_a = r_hat.predict_all(anchor.X)[idx, anchor.A]
    Phi = np.c_[np.ones(len(anchor)), r_a]
    beta = np.linalg.solve((Phi * rho[:, None]).T @ Phi + 1e-9 * np.eye(2), (Phi * rho[:, None]).T @ anchor.R)
    r_t = r_hat.predict_all(X_target)
    dm = (pi.probs(X_target) * (beta[0] + beta[1] * r_t)).sum(axis=1)
    corr = rho * (anchor.R - Phi @ beta)
    v = float(dm.mean() + corr.mean())
    se = float(np.sqrt(dm.var(ddof=1) / len(dm) + corr.var(ddof=1) / len(corr)))
    return Estimate(v, se, len(anchor))


def transport_pool(comps: list, anchor_comp, cfg: ODriftConfig = ODriftConfig()):
    """Context-transport DR: every transported window + anchor at minimum
    variance, ignoring reward drift (O-DRIFT with the homogeneity test off).
    Diagnostics-based eligibility is kept."""
    from dataclasses import replace
    return combine(comps, anchor_comp, replace(cfg, selection="segment", z_group=np.inf, z_anchor=np.inf))
