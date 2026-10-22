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
