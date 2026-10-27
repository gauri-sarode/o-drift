"""Cross-fitted reward models q_hat(x, a)."""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .simulator import LoggedData

# Context basis shared by EVERY reward model (all estimators, baselines included; not the monitor):
# "linear" phi(x) = x, or "rff" phi(x) = [x, sqrt(2) cos(x Omega + b)] with D random Fourier features
# (Gaussian kernel, bandwidth sqrt(d)). Omega, b are fixed by (d, D, seed): one feature map for all data.
_BASIS = {"kind": "linear", "D": 0, "seed": 0, "cache": {}}


def set_basis(kind: str = "linear", D: int = 0, seed: int = 0) -> None:
    if kind not in ("linear", "rff"):
        raise ValueError(kind)
    _BASIS.update(kind=kind, D=int(D), seed=int(seed), cache={})


def basis_dim(d: int) -> int:
    return d + (_BASIS["D"] if _BASIS["kind"] == "rff" else 0)


def expand(X: np.ndarray) -> np.ndarray:
    """phi(X), shape (n, basis_dim(d))."""
    if _BASIS["kind"] == "linear":
        return X
    d = X.shape[1]
    if d not in _BASIS["cache"]:
        rng = np.random.default_rng(_BASIS["seed"])
        _BASIS["cache"][d] = (rng.normal(scale=1 / np.sqrt(d), size=(d, _BASIS["D"])),
                              rng.uniform(0, 2 * np.pi, _BASIS["D"]))
    omega, b = _BASIS["cache"][d]
    return np.hstack([X, np.sqrt(2) * np.cos(X @ omega + b)])


def _features(X: np.ndarray, A: np.ndarray, n_actions: int) -> np.ndarray:
    """Per-action features in the context basis: [onehot(a), phi(x) * onehot(a)]."""
    P = expand(X)
    onehot = np.eye(n_actions)[A]
    return np.hstack([onehot, (P[:, :, None] * onehot[:, None, :]).reshape(len(A), -1)])


def block_logits(X: np.ndarray, coef: np.ndarray, n_actions: int) -> np.ndarray:
    """Logits for every action from coefficients laid out as _features: [b (A), W (p*A, phi-major)].
    Equal to scoring _features(x, a) for each a, without materializing the (n*A, A(1+p)) matrix."""
    P = expand(X)
    return P @ coef[n_actions:].reshape(P.shape[1], n_actions) + coef[:n_actions]


def fit_per_action(X: np.ndarray, A: np.ndarray, R: np.ndarray, n_actions: int,
                   mult: np.ndarray | None = None, C: float = 1.0, max_iter: int = 1000,
                   tol: float = 1e-8) -> np.ndarray:
    """L2 logistic regression on per-action features, fit action by action.

    The model has B feature blocks, block k = mult[:, k] * [onehot(a), phi(x) * onehot(a)]
    (mult = None: one block of ones). Every column belongs to exactly one action
    (column index % n_actions) and the penalty is separable, so the joint objective
    C * sum(log-loss) + ||w||^2 / 2 splits into one independent fit per action on its own
    rows -- the same optimum as one big fit, without the (n, B*A*(1+p)) matrix. The small
    per-action problems are solved to a tight tolerance (the old single big fit stopped at
    tol 1e-4 with coefficient errors up to ~0.2 for time-aware features).
    Returns coefficients in the joint layout (B blocks, each laid out as _features).
    """
    from sklearn.linear_model import LogisticRegression
    P = expand(X)
    mult = np.ones((len(X), 1)) if mult is None else mult
    B, p1 = mult.shape[1], 1 + P.shape[1]
    coef = np.zeros(B * n_actions * p1)
    # column of (block k, feature f in [1, phi], action a) in the joint layout
    col = lambda k, f, a: k * n_actions * p1 + (a if f == 0 else n_actions + (f - 1) * n_actions + a)
    for a in range(n_actions):
        rows = A == a
        if not rows.any():
            continue
        Z = np.hstack([mult[rows, k:k + 1] * np.c_[np.ones(rows.sum()), P[rows]] for k in range(B)])
        y = R[rows]
        idx = [col(k, f, a) for k in range(B) for f in range(p1)]
        if y.min() == y.max():  # one class: intercept-only fit of a smoothed constant
            coef[col(0, 0, a)] = np.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5))
            continue
        coef[idx] = LogisticRegression(C=C, max_iter=max_iter, fit_intercept=False, tol=tol).fit(Z, y).coef_.ravel()
    return coef


class RewardModel:
    """Per-action logistic regression. Misspecified when the simulator's misspec > 0."""

    def __init__(self, n_actions: int, C: float = 1.0):
        self.n_actions = n_actions
        self.C = C
        self.coef = None
        self._constant = None

    def fit(self, data: LoggedData) -> "RewardModel":
        if data.R.min() == data.R.max():
            self._constant = float(data.R[0])
        else:
            self.coef = fit_per_action(data.X, data.A, data.R, self.n_actions, C=self.C)
        return self

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """q_hat(x, a) for every action, shape (n, A)."""
        n = len(X)
        if self._constant is not None:
            return np.full((n, self.n_actions), self._constant)
        return expit(block_logits(X, self.coef, self.n_actions))


def crossfit_rewards(data: LoggedData, n_actions: int, n_folds: int = 2, seed: int = 0,
                     extra_X: np.ndarray | None = None) -> tuple:
    """Out-of-fold q_hat for `data`, plus fold-averaged q_hat on `extra_X`.

    Returns (q_hat_data (n, A), q_hat_extra (m, A) or None). Each logged record
    is scored only by a model that did not see it, so DR residuals are not
    shrunk by overfitting.
    """
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(data)) % n_folds
    q_data = np.empty((len(data), n_actions))
    q_extra = None if extra_X is None else np.zeros((len(extra_X), n_actions))
    for f in range(n_folds):
        model = RewardModel(n_actions).fit(data.subset(folds != f))
        q_data[folds == f] = model.predict_all(data.X[folds == f])
        if extra_X is not None:
            q_extra += model.predict_all(extra_X) / n_folds
    return q_data, q_extra
