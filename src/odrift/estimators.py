"""Standard single-source OPE estimators with per-record scores and Wald intervals.

Every estimator returns an `Estimate` built from per-record influence scores,
so standard errors, paired comparisons and cross-window covariances all come
from the same object.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .simulator import LoggedData


@dataclass
class Estimate:
    value: float
    se: float
    n: int
    scores: np.ndarray | None = None  # centered influence scores (for covariance)

    def interval(self, level: float = 0.95) -> tuple:
        z = norm.ppf(0.5 + level / 2)
        return self.value - z * self.se, self.value + z * self.se

    def covers(self, truth: float, level: float = 0.95) -> bool:
        lo, hi = self.interval(level)
        return lo <= truth <= hi


def _from_scores(scores: np.ndarray) -> Estimate:
    n = len(scores)
    v = scores.mean()
    return Estimate(float(v), float(scores.std(ddof=1) / np.sqrt(n)), n, scores - v)


def importance_weights(data: LoggedData, pi_probs: np.ndarray) -> np.ndarray:
    """rho_i = pi(A_i|X_i) / mu(A_i|X_i), using the propensity as logged."""
    return pi_probs[np.arange(len(data)), data.A] / data.pscore


def ips(data: LoggedData, pi_probs: np.ndarray) -> Estimate:
    return _from_scores(importance_weights(data, pi_probs) * data.R)


def snips(data: LoggedData, pi_probs: np.ndarray) -> Estimate:
    rho = importance_weights(data, pi_probs)
    v = float((rho * data.R).sum() / rho.sum())
    infl = rho * (data.R - v) / rho.mean()  # delta-method influence function
    return Estimate(v, float(infl.std(ddof=1) / np.sqrt(len(rho))), len(rho), infl)


def dm(pi_probs_eval: np.ndarray, q_hat_eval: np.ndarray) -> Estimate:
    """Direct method over an evaluation context sample (no reward correction)."""
    return _from_scores((pi_probs_eval * q_hat_eval).sum(axis=1))


def dr(data: LoggedData, pi_probs: np.ndarray, q_hat: np.ndarray) -> Estimate:
    """Doubly robust: sum_a pi q_hat + rho (R - q_hat(A)) on logged contexts."""
    idx = np.arange(len(data))
    rho = importance_weights(data, pi_probs)
    return _from_scores((pi_probs * q_hat).sum(axis=1) + rho * (data.R - q_hat[idx, data.A]))


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / (weights ** 2).sum())
