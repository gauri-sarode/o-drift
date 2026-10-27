import numpy as np

from odrift import baselines as bl
from odrift.monitor import ResidualStats, reward_changepoints
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, LoggedData, sample_episode


def _episode(regime, intensity, seed=0):
    env = EnvParams.sample(seed=seed)
    return env, sample_episode(env, REGIMES[regime](intensity, K=12), 3000, 1000, 2000, seed=seed)


def test_cope_cluster_keeps_anchor_and_closest_mean_reward_windows():
    env, ep = _episode("S0", 0.0)
    data = LoggedData.concat(ep.history + [ep.anchor])
    h = bl.cope_cluster(data, 12, 4)
    assert h[12] == 1 and h.sum() == 4
    rbar = np.array([data.R[data.window == k].mean() for k in range(13)])
    chosen = np.where(h[:12] > 0)[0]
    others = np.where(h[:12] == 0)[0]
    assert np.abs(rbar[chosen] - rbar[12]).max() <= np.abs(rbar[others] - rbar[12]).min()
    assert bl.cope_cluster(data, 12, 13).sum() == 13


def test_mean_reward_statistic_is_one_df_and_residual_is_per_action():
    env, ep = _episode("S0", 0.0)
    left, right = list(range(6)), list(range(6, 12))
    assert ResidualStats(ep.history, env.n_actions, kind="mean").block_stat(left, right)[1] == 1
    assert ResidualStats(ep.history, env.n_actions, kind="residual").block_stat(left, right)[1] == env.n_actions


def test_blind_statistics_split_under_logger_switch_monitor_does_not():
    """S2 (logger switches, q stable): the overall raw-reward test sees the switch as drift;
    the per-action residual monitor does not -- the mechanism-awareness the paper claims."""
    env, ep = _episode("S2", 0.0, seed=1)
    assert len(reward_changepoints(ResidualStats(ep.history, env.n_actions, kind="residual"), 1e-4)) == 1
    assert len(reward_changepoints(ResidualStats(ep.history, env.n_actions, kind="mean"), 1e-4)) > 1
