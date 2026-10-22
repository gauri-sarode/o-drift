"""Cross-fitted reward models q_hat(x, a)."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .simulator import LoggedData


def _features(X: np.ndarray, A: np.ndarray, n_actions: int) -> np.ndarray:
    """Per-action linear features: [onehot(a), x * onehot(a)]."""
    onehot = np.eye(n_actions)[A]
    return np.hstack([onehot, (X[:, :, None] * onehot[:, None, :]).reshape(len(A), -1)])


class RewardModel:
    """Per-action logistic regression. Misspecified when the simulator's misspec > 0."""

    def __init__(self, n_actions: int, C: float = 1.0):
        self.n_actions = n_actions
        self.model = LogisticRegression(C=C, max_iter=1000, fit_intercept=False)
        self._constant = None

    def fit(self, data: LoggedData) -> "RewardModel":
        if data.R.min() == data.R.max():
            self._constant = float(data.R[0])
        else:
            self.model.fit(_features(data.X, data.A, self.n_actions), data.R)
        return self

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """q_hat(x, a) for every action, shape (n, A)."""
        n = len(X)
        if self._constant is not None:
            return np.full((n, self.n_actions), self._constant)
        A = np.repeat(np.arange(self.n_actions), n)
        Xr = np.tile(X, (self.n_actions, 1))
        return self.model.predict_proba(_features(Xr, A, self.n_actions))[:, 1].reshape(self.n_actions, n).T


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
