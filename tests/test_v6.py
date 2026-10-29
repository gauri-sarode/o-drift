import numpy as np

from odrift import baselines as bl
from odrift.simulator import LoggedData


class Uniform:
    def __init__(self, A):
        self.A = A

    def probs(self, X):
        return np.full((len(X), self.A), 1 / self.A)


def _data(K=12, n=400, A=4, shift=0.3, seed=0):
    rng = np.random.default_rng(seed)
    w = np.repeat(np.arange(K + 1), n)
    p = np.where(w < K // 2, 0.2 + shift, 0.2)  # older half reward-drifted
    N = len(w)
    return LoggedData(X=rng.normal(size=(N, 2)), A=rng.integers(A, size=N), R=(rng.random(N) < p).astype(int),
                      pscore=np.full(N, 1 / A), window=w, logger_id=np.zeros(N, int))


def test_opfv_feature_masks():
    f = bl.opfv_features(12)
    assert f["recent0"].tolist() == [False] * 12 + [True]
    assert f["recent12"].all()
    assert np.flatnonzero(f["period6"]).tolist() == [0, 6, 12]


def test_opfv_reduces_to_mean_reward_and_avoids_biased_windows():
    d = _data()
    A, K = 4, 12
    q0, qt = np.zeros(len(d)), np.zeros((50, A))
    feats = bl.opfv_features(K)
    e, _ = bl.opfv(d, q0, qt, np.zeros((50, 2)), Uniform(A), K, {"recent0": feats["recent0"], "recent12": feats["recent12"]})
    assert e.value in (d.R[d.window == K].mean(), d.R.mean())
    # all candidates: the chosen feature must not pool the drifted older half
    e, chosen = bl.opfv(d, q0, qt, np.zeros((50, 2)), Uniform(A), K, feats)
    assert chosen in ("recent0", "recent1", "recent2", "recent3", "recent6", "period4", "period6", "period3", "period2") \
        and not feats[chosen][:K // 2].any() or abs(e.value - 0.2) < 0.03


def test_laplace_var_floor_only_for_constant_rewards():
    from odrift.method import laplace_var_floor
    from odrift.simulator import LoggedData
    n, A = 200, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    base = dict(X=X, A=rng.integers(A, size=n), pscore=np.full(n, 1 / A), probs_all=np.full((n, A), 1 / A),
                window=np.zeros(n, dtype=int), logger_id=np.zeros(n, dtype=int))

    class Uniform:
        def probs(self, X):
            return np.full((len(X), A), 1 / A)

    zero = LoggedData(R=np.zeros(n), **base)
    p = 1 / (n + 2)
    assert np.isclose(laplace_var_floor(zero, Uniform()), p * (1 - p) / n)  # rho = 1
    varied = LoggedData(R=(rng.random(n) < 0.1).astype(float), **base)
    assert laplace_var_floor(varied, Uniform()) == 0.0
