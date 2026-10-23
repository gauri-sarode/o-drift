"""Nonstationary contextual-bandit simulator with tractable ground truth.

Windows k = 0..K-1 are historical; window K is the current target period T.
Each window has its own context distribution P_k(X), logging policy mu_k(A|X)
and conditional reward q_k(X, A), and the three can be shifted independently:

    X ~ N(c_k * u, I_d)                                  context shift along u
    mu_k(a|x) = (1 - eps_k) softmax(beta_k x^T M_{l(k)})_a + eps_k / A
    q_k(x, a) = sigmoid(f(x, a) + r_k * g(x, a))         reward shift
    (S8: g replaced by an out-of-class interaction h -- misspecified drift)

f and g are fixed random functions that are linear-per-action plus a
nonlinear term (weight `misspec`), so a per-action logistic reward model is
misspecified when misspec > 0. The target is (c_K, r_K) = (0, 0); historical
drift is expressed as deviations from it.

Ground truth V_T(pi) = E_{P_T}[sum_a pi(a|X) q_T(X, a)] is exact over actions
and Monte Carlo over contexts. The transported historical value
V_{T<-k}(pi) = E_{P_T}[sum_a pi(a|X) q_k(X, a)] is available the same way,
so the transfer bias b_k = V_{T<-k} - V_T can be checked directly.
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import expit, softmax


@dataclass(frozen=True)
class EnvParams:
    """Fixed (seeded) structural parameters shared by every window."""

    d: int
    n_actions: int
    W: np.ndarray        # (d, A) base reward weights
    b: np.ndarray        # (A,) base reward intercepts
    V: np.ndarray        # (d, A) nonlinear-term weights
    G: np.ndarray        # (d, A) reward-drift weights
    gb: np.ndarray       # (A,) reward-drift intercepts
    u: np.ndarray        # (d,) unit context-shift direction
    loggers: np.ndarray  # (n_loggers, d, A) logging-policy score matrices
    misspec: float
    H: np.ndarray | None = None  # (d, A) second direction of the out-of-class drift term (S8)

    @classmethod
    def sample(cls, seed: int, d: int = 5, n_actions: int = 10,
               n_loggers: int = 3, misspec: float = 0.5) -> "EnvParams":
        rng = np.random.default_rng(seed)
        u = rng.normal(size=d)
        return cls(
            d=d,
            n_actions=n_actions,
            W=rng.normal(scale=0.5, size=(d, n_actions)),
            b=rng.normal(loc=-1.0, scale=0.5, size=n_actions),
            V=rng.normal(scale=0.5, size=(d, n_actions)),
            G=rng.normal(scale=0.5, size=(d, n_actions)),
            gb=rng.normal(scale=1.0, size=n_actions),
            u=u / np.linalg.norm(u),
            loggers=rng.normal(scale=0.5, size=(n_loggers, d, n_actions)),
            misspec=misspec,
            H=rng.normal(scale=0.5, size=(d, n_actions)),  # drawn last: earlier draws unchanged
        )

    def f(self, X: np.ndarray) -> np.ndarray:
        """Base reward logits for all actions, shape (n, A)."""
        return X @ self.W + self.b + self.misspec * np.tanh(X @ self.V) ** 2

    def g(self, X: np.ndarray) -> np.ndarray:
        """Reward-drift logits for all actions, shape (n, A)."""
        return X @ self.G + self.gb

    def h(self, X: np.ndarray, intercept: float = 1.0) -> np.ndarray:
        """Out-of-class reward-drift logits (interaction of two directions), shape (n, A).
        Not representable by the per-action linear reward model; intercept = 0 drops the
        in-class per-action intercept, leaving only the out-of-class part (S8b)."""
        return intercept * self.gb + 2.0 * np.tanh(X @ self.G) * np.tanh(X @ self.H)


@dataclass(frozen=True)
class WindowSpec:
    """Drift state of one window."""

    context_shift: float = 0.0   # c_k
    reward_shift: float = 0.0    # r_k
    drift_nonlinear: float = 0.0  # nu_k: drift direction (1 - nu) g + nu h (S8)
    drift_intercept: float = 1.0  # weight of the per-action intercept inside h (S8b: 0)
    logger_id: int = 0           # l(k)
    logger_beta: float = 1.0     # beta_k (greediness)
    logger_eps: float = 0.2      # eps_k (uniform exploration)
    unsupported_actions: tuple = ()  # actions with mu_k(a|x) = 0
    # S6 metadata fault: propensities written to the log are recomputed from
    # this logger instead of the one that actually acted (None = correct).
    recorded_logger_id: int | None = None


@dataclass(frozen=True)
class Regime:
    name: str
    windows: tuple  # K historical WindowSpecs, oldest first
    target: WindowSpec = field(default_factory=WindowSpec)

    @property
    def n_windows(self) -> int:
        return len(self.windows)


def q(env: EnvParams, spec: WindowSpec, X: np.ndarray) -> np.ndarray:
    drift = env.g(X) if not spec.drift_nonlinear else \
        (1 - spec.drift_nonlinear) * env.g(X) + spec.drift_nonlinear * env.h(X, spec.drift_intercept)
    return expit(env.f(X) + spec.reward_shift * drift)


def mu(env: EnvParams, spec: WindowSpec, X: np.ndarray) -> np.ndarray:
    scores = spec.logger_beta * (X @ env.loggers[spec.logger_id])
    probs = (1 - spec.logger_eps) * softmax(scores, axis=1) + spec.logger_eps / env.n_actions
    if spec.unsupported_actions:
        probs[:, list(spec.unsupported_actions)] = 0.0
        probs /= probs.sum(axis=1, keepdims=True)
    return probs


def sample_contexts(env: EnvParams, spec: WindowSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n, env.d)) + spec.context_shift * env.u


@dataclass
class LoggedData:
    X: np.ndarray        # (n, d)
    A: np.ndarray        # (n,) int
    R: np.ndarray        # (n,) {0, 1}
    pscore: np.ndarray   # (n,) mu(A|X) as logged at event time
    window: np.ndarray   # (n,) window index (K = target/anchor)
    logger_id: np.ndarray  # (n,)
    probs_all: np.ndarray | None = None  # (n, A) logged mu(.|X) at event time (A4 artifacts)

    def __len__(self) -> int:
        return len(self.A)

    def subset(self, mask: np.ndarray) -> "LoggedData":
        return LoggedData(**{k: (None if v is None else v[mask]) for k, v in vars(self).items()})

    @staticmethod
    def concat(parts: list) -> "LoggedData":
        out = {}
        for k in vars(parts[0]):
            vals = [getattr(p, k) for p in parts]
            out[k] = None if any(v is None for v in vals) else np.concatenate(vals)
        return LoggedData(**out)


def log_window(env: EnvParams, spec: WindowSpec, n: int, window: int,
               rng: np.random.Generator) -> LoggedData:
    X = sample_contexts(env, spec, n, rng)
    probs = mu(env, spec, X)
    # inverse-CDF sampling of one action per row
    A = (probs.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1)
    A = np.minimum(A, env.n_actions - 1)
    R = (rng.random(n) < q(env, spec, X)[np.arange(n), A]).astype(np.int8)
    if spec.recorded_logger_id is not None:
        probs = mu(env, replace(spec, logger_id=spec.recorded_logger_id), X)
    return LoggedData(
        X=X, A=A, R=R, pscore=probs[np.arange(n), A], probs_all=probs,
        window=np.full(n, window), logger_id=np.full(n, spec.logger_id),
    )


@dataclass
class Episode:
    """One sampled evaluation problem."""

    history: list            # K LoggedData, oldest first
    anchor: LoggedData       # small recent logged sample from the target period
    target_contexts: np.ndarray  # unlabeled X* ~ P_T


def sample_episode(env: EnvParams, regime: Regime, n_per_window: int, n_anchor: int,
                   n_target_contexts: int, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    K = regime.n_windows
    history = [log_window(env, spec, n_per_window, k, rng) for k, spec in enumerate(regime.windows)]
    anchor = log_window(env, regime.target, n_anchor, K, rng)
    return Episode(history, anchor, sample_contexts(env, regime.target, n_target_contexts, rng))


class Policy:
    """Stochastic target policy pi(a|x) = (1-eps) softmax(beta * score(x)) + eps/A."""

    def __init__(self, env: EnvParams, weights: np.ndarray, intercept: np.ndarray,
                 beta: float = 2.0, eps: float = 0.05):
        self.env, self.weights, self.intercept, self.beta, self.eps = env, weights, intercept, beta, eps

    def probs(self, X: np.ndarray) -> np.ndarray:
        p = softmax(self.beta * (X @ self.weights + self.intercept), axis=1)
        return (1 - self.eps) * p + self.eps / self.env.n_actions

    @classmethod
    def near_optimal(cls, env: EnvParams, noise: float = 0.0, seed: int = 0, **kw) -> "Policy":
        """Policy greedy-ish w.r.t. the base linear reward, optionally perturbed.

        Varying `noise` gives a family of candidate policies of different quality
        for policy-selection regret.
        """
        rng = np.random.default_rng(seed)
        return cls(env, env.W + noise * rng.normal(size=env.W.shape),
                   env.b + noise * rng.normal(size=env.b.shape), **kw)


def true_value(env: EnvParams, policy: Policy, context_spec: WindowSpec,
               reward_spec: WindowSpec, n_mc: int = 1_000_000, seed: int = 12345,
               chunk: int = 200_000) -> float:
    """E_{X~P(context_spec)}[sum_a pi(a|X) q_{reward_spec}(X, a)] by Monte Carlo.

    V_T(pi): context_spec = reward_spec = regime.target.
    V_{T<-k}(pi): context_spec = regime.target, reward_spec = regime.windows[k].
    The fixed seed gives common random numbers across calls, so differences
    such as b_k = V_{T<-k} - V_T have far less MC noise than either term.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for start in range(0, n_mc, chunk):
        X = sample_contexts(env, context_spec, min(chunk, n_mc - start), rng)
        total += (policy.probs(X) * q(env, reward_spec, X)).sum()
    return total / n_mc
