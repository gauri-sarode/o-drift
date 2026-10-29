"""Real-reward Open Bandit evaluation (protocol v6.real; decision log 38).

Everything is real: contexts, items, clicks and the time order of the uniform-random log, and the
evaluation policy and its ground truth from the Bernoulli Thompson Sampling (BTS) log of the same
A/B test (data/obd_real_{campaign}.npz from scripts/15_prepare_obd_real.py).

A task is (campaign, target block b): history windows are blocks b-12..b-1, the anchor and the
target contexts are disjoint random subsamples of block b, and the estimand is the value of
pi_b = BTS's item distribution per position in block b, whose ground truth is BTS's on-policy click
rate in block b. The env is EnvParams-compatible for mu(): all-zero logger weights with
logger_eps = 1 give exactly the uniform random logger.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .simulator import Episode, LoggedData, Regime, WindowSpec

DATA = Path(__file__).resolve().parents[2] / "data"
CAMPAIGNS = ("men", "women", "all")
UNIFORM = WindowSpec(logger_eps=1.0)


@lru_cache(maxsize=3)
def load(campaign: str) -> dict:
    d = np.load(DATA / f"obd_real_{campaign}.npz")
    return {k: d[k] for k in d.files}


def task_of(env_seed: int) -> tuple:
    """env seed = 100 * campaign index + target block."""
    return CAMPAIGNS[env_seed // 100], env_seed % 100


@dataclass(frozen=True)
class RealEnv:
    campaign: str
    target_block: int
    d: int
    n_actions: int
    loggers: np.ndarray  # (1, d, A) zeros: mu() with logger_eps = 1 is uniform

    @classmethod
    def sample(cls, env_seed: int) -> "RealEnv":
        camp, b = task_of(env_seed)
        data = load(camp)
        d, A = data["X"].shape[1], data["pi"].shape[2]
        return cls(camp, b, d, A, np.zeros((1, d, A)))


def regime(K: int) -> Regime:
    return Regime("REAL", (UNIFORM,) * K, UNIFORM)


class PositionPolicy:
    """Context-free (per-position) policy: pi(a | x) = table[position(x), a]; the position is the
    one-hot in the last three context columns."""

    def __init__(self, table: np.ndarray):
        self.table = table

    def probs(self, X: np.ndarray) -> np.ndarray:
        return X[:, -3:] @ self.table


def policy(env: RealEnv) -> PositionPolicy:
    return PositionPolicy(load(env.campaign)["pi"][env.target_block])


def truths(env: RealEnv, K: int) -> list:
    """[(V_T, [V_k for the history windows])]: BTS on-policy click rates of the target block and of
    each history block (the latter only feed the oracle, as the transfer values V_{T<-k})."""
    ctr = load(env.campaign)["ctr"]
    b = env.target_block
    return [(float(ctr[b]), [float(ctr[k]) for k in range(b - K, b)])]


def _logged(data: dict, idx: np.ndarray, window: int, n_actions: int) -> LoggedData:
    n = len(idx)
    return LoggedData(X=data["X"][idx], A=data["A"][idx].astype(int), R=data["R"][idx], pscore=np.full(n, 1 / n_actions),
                      probs_all=np.full((n, n_actions), 1 / n_actions), window=np.full(n, window),
                      logger_id=np.zeros(n, dtype=int))


def sample_episode(env: RealEnv, K: int, n_per_window: int, n_anchor: int, n_target_contexts: int,
                   seed: int) -> Episode:
    data = load(env.campaign)
    rng = np.random.default_rng(seed)
    b = env.target_block
    assert b >= K, "target block needs K history blocks before it"
    history = []
    for k, blk in enumerate(range(b - K, b)):
        rows = np.flatnonzero(data["block"] == blk)
        history.append(_logged(data, rng.choice(rows, n_per_window, replace=False), k, env.n_actions))
    rows = rng.permutation(np.flatnonzero(data["block"] == b))
    anchor = _logged(data, rows[:n_anchor], K, env.n_actions)
    return Episode(history, anchor, data["X"][rows[n_anchor:n_anchor + n_target_contexts]])
