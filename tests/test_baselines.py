import numpy as np
import pytest

from odrift import baselines as bl
from odrift.method import ODriftConfig, build_components
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, Policy, sample_episode, true_value

ENV = EnvParams.sample(seed=0)
PI = Policy.near_optimal(ENV)
REGIME = REGIMES["S2"](0.0)
EP = sample_episode(ENV, REGIME, 4000, 3000, 20_000, seed=0)
TRUTH = true_value(ENV, PI, REGIME.target, REGIME.target)


@pytest.mark.parametrize("name,fn", [
    ("recent3", lambda: bl.recent_dr(EP.history, EP.anchor, PI, ENV.n_actions, 3)),
    ("anchor", lambda: bl.recent_dr(EP.history, EP.anchor, PI, ENV.n_actions, 0)),
    ("decay", lambda: bl.decay_dr(EP.history, EP.anchor, PI, ENV.n_actions, 0.8)),
    ("minvar", lambda: bl.minvar_dr(EP.history, EP.anchor, PI, ENV.n_actions)),
    ("liu", lambda: bl.regression_assisted_dr(EP.history, EP.anchor, EP.target_contexts, PI, ENV.n_actions)),
])
def test_baselines_unbiased_without_reward_drift(name, fn):
    e = fn()
    assert abs(e.value - TRUTH) < 4 * e.se, (name, e.value, TRUTH, e.se)


def test_transport_pool_unbiased_without_reward_drift():
    comps, anc = build_components(EP.history, EP.anchor, EP.target_contexts, PI, ENV.n_actions, ODriftConfig())
    r = bl.transport_pool(comps, anc)
    assert abs(r.value - TRUTH) < 4 * r.se
    assert sum(a > 0 for a in r.weights.values()) > 6  # actually pools


def test_selectors_on_synthetic_family():
    vals = np.array([0.50, 0.505, 0.502, 0.60, 0.61])   # members 3,4 clearly biased
    var = np.array([0.02, 0.01, 0.007, 0.004, 0.003]) ** 2
    assert bl.slope_select(vals, var, 2.0) == 2
    assert bl.anchor_validated_select(vals, var) == 2
    w = bl.magic_weights(vals, var, 1.0)
    assert abs(w.sum() - 1) < 1e-8 and w[3:].sum() < 0.05


def test_forecast_recovers_linear_trend():
    from types import SimpleNamespace
    from odrift.baselines import forecast_value
    K = 12
    comps = [SimpleNamespace(index=k, value=0.5 + 0.01 * k) for k in range(K + 1)]
    v, se = forecast_value(comps, 1e-6 * np.eye(K + 1), K, 1)
    assert abs(v - (0.5 + 0.01 * K)) < 1e-6 and se < 1e-2


def test_time_aware_dr_and_composite_unbiased_when_stationary():
    from odrift.baselines import fit_time_reward, time_aware_dr, transport_balanced_decay_dr
    from odrift.method import ratio_function
    from odrift.nuisance import crossfit_rewards
    from odrift.regimes import REGIMES
    from odrift.simulator import EnvParams, LoggedData, Policy, mu, sample_episode, true_value
    env = EnvParams.sample(0)
    pi = Policy.near_optimal(env)
    regime = REGIMES["S2"](0.0)
    truth = true_value(env, pi, regime.target, regime.target)
    K = regime.n_windows
    fns = [lambda X, sp=sp: mu(env, sp, X) for sp in list(regime.windows) + [regime.target]]
    e_t, e_c = [], []
    for seed in range(3):
        ep = sample_episode(env, regime, 2000, 1000, 5000, seed=seed)
        data = LoggedData.concat(ep.history + [ep.anchor])
        e_t.append(time_aware_dr(data, *fit_time_reward(data, K, ep.target_contexts, env.n_actions, 1), ep.target_contexts, pi))
        qo, qt = crossfit_rewards(data, env.n_actions, extra_X=ep.target_contexts)
        rfs = [ratio_function(d.X, ep.target_contexts) for d in ep.history] + [None]
        e_c.append(transport_balanced_decay_dr(data, qo, qt, ep.target_contexts, pi, K, 0.9, fns, rfs))
    for es in (e_t, e_c):
        assert abs(np.mean([e.value for e in es]) - truth) < 3 * np.mean([e.se for e in es]), ([e.value for e in es], truth)
