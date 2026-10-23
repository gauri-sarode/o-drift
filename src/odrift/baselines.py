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


# ------------------------------------------------- data-driven selectors over a nested family
# Family j = 0..J: estimator on the anchor plus the newest j windows (more data, more
# potential drift bias). Values/SEs come from the same per-record DR scores, so for nested
# record sets Cov(V_i, V_j) = Var(V_max(i,j)) (the larger set's variance).

def nested_family(scores: np.ndarray, window: np.ndarray, K: int) -> tuple:
    """(values, variances) for j = 0..K: anchor (window K) + windows K-j..K-1."""
    vals, vars_ = [], []
    for j in range(K + 1):
        s = scores[window >= K - j]
        vals.append(s.mean())
        vars_.append(s.var(ddof=1) / len(s))
    return np.array(vals), np.array(vars_)


def _nested_cov(var: np.ndarray) -> np.ndarray:
    return np.minimum.outer(var, var)


def slope_select(vals: np.ndarray, var: np.ndarray, kappa: float) -> int:
    """SLOPE (Su, Srinath & Krishnamurthy, ICML 2020): Lepski-style rule over a family
    ordered by decreasing variance -- the largest j whose interval intersects all
    intervals of the less-biased members j' <= j."""
    lo, hi, best = -np.inf, np.inf, 0
    for j in range(len(vals)):
        lo = max(lo, vals[j] - kappa * np.sqrt(var[j]))
        hi = min(hi, vals[j] + kappa * np.sqrt(var[j]))
        if lo > hi:
            break
        best = j
    return best


def magic_weights(vals: np.ndarray, var: np.ndarray, z_ref: float) -> np.ndarray:
    """MAGIC (Thomas & Brunskill, ICML 2016): blend weights minimizing x'Omega x + (x'b)^2,
    with b_j the distance from V_j to the reference (anchor-only, j=0) confidence interval."""
    from .method import select_weights
    lo, hi = vals[0] - z_ref * np.sqrt(var[0]), vals[0] + z_ref * np.sqrt(var[0])
    b = np.maximum(0.0, np.maximum(lo - vals, vals - hi))
    Q = _nested_cov(var) + np.outer(b, b)
    return select_weights(Q, b, 0.0, Q=Q)


def anchor_validated_select(vals: np.ndarray, var: np.ndarray) -> int:
    """Pick the member minimizing estimated MSE: debiased squared distance to the anchor
    estimate (j=0) plus own variance. For nested sets Var(V_j - V_0) = var_0 - var_j."""
    bias2 = np.maximum(0.0, (vals - vals[0]) ** 2 - (var[0] - var))
    bias2[0] = 0.0
    return int(np.argmin(bias2 + var))


def balanced_mixture_dr(data: LoggedData, mixture_probs: np.ndarray, pi_probs: np.ndarray,
                        q_hat: np.ndarray) -> Estimate:
    """Multi-logger DR with the balanced (mixture) propensity mu_bar(a|x) = sum_k (n_k/N) mu_k(a|x)
    (Agarwal et al. 2017; Kallus, Saito & Uehara, ICML 2021). Optimal under a stationary reward."""
    idx = np.arange(len(data))
    rho = pi_probs[idx, data.A] / mixture_probs[idx, data.A]
    scores = (pi_probs * q_hat).sum(axis=1) + rho * (data.R - q_hat[idx, data.A])
    return _weighted_estimate(scores, np.ones(len(data)))


# ------------------------------------------------- drift-specific baselines (added before freeze)

def forecast_value(comps: list, S: np.ndarray, K: int, degree: int) -> tuple:
    """Time-series forecast of per-window value estimates (Chandak et al., NeurIPS 2020,
    "Optimizing for the future in non-stationary MDPs"; here with windows as time steps).

    Per-window estimates V_k (index k = 0..K-1; the anchor is index K, the target period)
    are regressed on a polynomial in t = k/K by generalized least squares with their joint
    covariance S, and the fit is evaluated at t = 1. Returns (value, se).
    """
    t = np.array([c.index for c in comps], dtype=float) / K
    Phi = np.vander(t, degree + 1, increasing=True)
    L = np.linalg.cholesky(S + 1e-12 * np.trace(S) / len(S) * np.eye(len(S)))
    Pw, yw = np.linalg.solve(L, Phi), np.linalg.solve(L, np.array([c.value for c in comps]))
    cov = np.linalg.pinv(Pw.T @ Pw)
    beta = cov @ Pw.T @ yw
    phi_t = np.ones(degree + 1)
    return float(phi_t @ beta), float(np.sqrt(max(phi_t @ cov @ phi_t, 0.0)))


def _time_features(X: np.ndarray, A: np.ndarray, t: np.ndarray, n_actions: int, degree: int) -> np.ndarray:
    """[onehot(a), x*onehot(a)] interacted with (1, t, ..., t^degree)."""
    onehot = np.eye(n_actions)[A]
    base = np.hstack([onehot, (X[:, :, None] * onehot[:, None, :]).reshape(len(A), -1)])
    return np.hstack([base * (t ** p)[:, None] for p in range(degree + 1)])


def fit_time_reward(data: LoggedData, K: int, X_target: np.ndarray, n_actions: int, degree: int,
                    seed: int = 0) -> tuple:
    """Cross-fitted time-aware reward model q(x, a, t), t = window/K (policy-independent).

    Returns (out-of-fold q at each record's own (x, a, t), fold-averaged q(x*, ., 1) on
    the target contexts, shape (M, A))."""
    from sklearn.linear_model import LogisticRegression
    t = data.window / K
    folds = np.random.default_rng(seed).permutation(len(data)) % 2
    M = len(X_target)
    q_own, q_tgt = np.empty(len(data)), np.zeros((M, n_actions))
    F_tgt = _time_features(np.tile(X_target, (n_actions, 1)), np.repeat(np.arange(n_actions), M),
                           np.ones(n_actions * M), n_actions, degree)
    for f in range(2):
        tr = folds != f
        model = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False)
        model.fit(_time_features(data.X[tr], data.A[tr], t[tr], n_actions, degree), data.R[tr])
        q_own[~tr] = model.predict_proba(_time_features(data.X[~tr], data.A[~tr], t[~tr], n_actions, degree))[:, 1]
        q_tgt += model.predict_proba(F_tgt)[:, 1].reshape(n_actions, M).T / 2
    return q_own, q_tgt


def time_aware_dr(data: LoggedData, q_own: np.ndarray, q_tgt: np.ndarray, X_target: np.ndarray, pi) -> Estimate:
    """Pooled DR with a time-aware reward model: DM term at the target time (t = 1) on
    target contexts + every record's residual against the model at its own time."""
    rho = importance_weights(data, pi.probs(data.X))
    dm = (pi.probs(X_target) * q_tgt).sum(axis=1)
    corr = rho * (data.R - q_own)
    v = float(dm.mean() + corr.mean())
    return Estimate(v, float(np.sqrt(dm.var(ddof=1) / len(dm) + corr.var(ddof=1) / len(corr))), len(data))


def transport_balanced_decay_dr(data: LoggedData, q_own: np.ndarray, q_tgt: np.ndarray, X_target: np.ndarray,
                                pi, K: int, gamma: float, logger_fns: list, ratio_fns: list) -> Estimate:
    """Composite baseline: context transport + multi-logger balancing + exponential decay,
    with one pooled reward model and no drift detection or correction.

    Record weights h_i = gamma^(K - k_i); importance weight from the target to the
    h-weighted sampling mixture sum_k c_k p_k(x) mu_k(a|x), c_k ∝ h_k n_k, with
    p_k = p_T / w_k:  W_i = pi(a_i|x_i) / sum_k c_k mu_k(a_i|x_i) / w_k(x_i).
    logger_fns / ratio_fns have K + 1 entries (index K = anchor; ratio_fns[k] None = 1).
    gamma = 1 is transport + balanced pooling of everything.
    """
    N, idx = len(data), np.arange(len(data))
    counts = np.bincount(data.window, minlength=K + 1).astype(float)
    c = counts * gamma ** (K - np.arange(K + 1))
    c /= c.sum()
    denom = np.zeros(N)
    for k in range(K + 1):
        if counts[k]:
            w_k = ratio_fns[k](data.X) if ratio_fns[k] is not None else 1.0
            denom += c[k] * logger_fns[k](data.X)[idx, data.A] / w_k
    W = pi.probs(data.X)[idx, data.A] / denom
    h = gamma ** (K - data.window)
    corr = W * (data.R - q_own[idx, data.A])
    dm = (pi.probs(X_target) * q_tgt).sum(axis=1)
    hn = h / h.sum()
    v = float(dm.mean() + hn @ corr)
    se = float(np.sqrt(dm.var(ddof=1) / len(dm) + ((hn * (corr - hn @ corr)) ** 2).sum()))
    return Estimate(v, se, N)
