"""Open Bandit semi-synthetic environment (protocol splits.real; decision log 23).

Real: contexts and their temporal order (data/obd_men.npz from scripts/07_prepare_obd.py) and
the shape of the base reward f (fit on real clicks). Synthetic: actions, propensities, rewards
and the injected logger/reward drift, through the same WindowSpec / q / mu machinery as the
simulator. A regime's context_shift is ignored: context drift is whatever the real time order has.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .simulator import Episode, Regime, log_contexts

DATA = Path(__file__).resolve().parents[2] / "data/obd_men.npz"
TARGET_BLOCK = 12


@lru_cache(maxsize=1)
def load() -> dict:
    d = np.load(DATA)
    return {k: d[k] for k in ("X", "block", "W", "b", "Q")}


@dataclass(frozen=True)
class OBDEnv:
    """EnvParams-compatible: f = real-fit base logits; g, h, loggers drawn per env seed.
    W, b (the linear part of f) define the near-optimal target policies."""

    d: int
    n_actions: int
    W: np.ndarray
    b: np.ndarray
    Q: np.ndarray
    G: np.ndarray
    gb: np.ndarray
    H: np.ndarray
    loggers: np.ndarray

    @classmethod
    def sample(cls, seed: int) -> "OBDEnv":
        data = load()
        d, A = data["W"].shape
        rng = np.random.default_rng(seed)
        return cls(d=d, n_actions=A, W=data["W"], b=data["b"], Q=data["Q"],
                   G=rng.normal(scale=0.5, size=(d, A)), gb=rng.normal(scale=1.0, size=A),
                   H=rng.normal(scale=0.5, size=(d, A)), loggers=rng.normal(scale=0.5, size=(3, d, A)))

    def f(self, X):
        return X @ self.W + self.b + (X ** 2) @ self.Q

    def g(self, X):
        return X @ self.G + self.gb

    def h(self, X, intercept: float = 1.0):
        return intercept * self.gb + 2.0 * np.tanh(X @ self.G) * np.tanh(X @ self.H)


def block_contexts(k: int) -> np.ndarray:
    data = load()
    return data["X"][data["block"] == k]


def sample_episode(env: OBDEnv, regime: Regime, n_per_window: int, n_anchor: int,
                   n_target_contexts: int, seed: int) -> Episode:
    """History window k = a random subsample of real block k; anchor and target contexts are
    disjoint subsamples of the real target block."""
    rng = np.random.default_rng(seed)
    assert regime.n_windows == TARGET_BLOCK
    history = []
    for k, spec in enumerate(regime.windows):
        Xk = block_contexts(k)
        history.append(log_contexts(env, spec, Xk[rng.choice(len(Xk), n_per_window, replace=False)], k, rng))
    Xt = block_contexts(TARGET_BLOCK)
    perm = rng.permutation(len(Xt))
    anchor = log_contexts(env, regime.target, Xt[perm[:n_anchor]], TARGET_BLOCK, rng)
    return Episode(history, anchor, Xt[perm[n_anchor:n_anchor + n_target_contexts]])


def true_value(env: OBDEnv, policy, reward_spec) -> float:
    """Exact V over ALL real target-block contexts: mean_x sum_a pi(a|x) q_spec(x, a)."""
    from .simulator import q
    Xt = block_contexts(TARGET_BLOCK)
    return float((policy.probs(Xt) * q(env, reward_spec, Xt)).sum(axis=1).mean())
