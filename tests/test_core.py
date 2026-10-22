import numpy as np
import pytest

from odrift import estimators as est
from odrift.nuisance import crossfit_rewards
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, LoggedData, Policy, log_window, mu, q, sample_episode, true_value

ENV = EnvParams.sample(seed=0)


def test_logger_probs_are_distributions():
    for name, make in REGIMES.items():
        regime = make(1.0)
        X = np.random.default_rng(0).normal(size=(50, ENV.d))
        for spec in (*regime.windows, regime.target):
            p = mu(ENV, spec, X)
            assert np.allclose(p.sum(axis=1), 1), name
            assert (p >= 0).all(), name


def test_logged_actions_follow_logger():
    spec = REGIMES["S2"](0.0).windows[0]
    data = log_window(ENV, spec, 200_000, 0, np.random.default_rng(1))
    # empirical action frequencies match the mean logging probability
    X = np.random.default_rng(2).normal(size=(200_000, ENV.d))
    expected = mu(ENV, spec, X).mean(axis=0)
    observed = np.bincount(data.A, minlength=ENV.n_actions) / len(data)
    assert np.abs(observed - expected).max() < 0.01


def test_recorded_propensity_fault_only_changes_pscore():
    regime = REGIMES["S6"](0.0)
    faulty = next(s for s in regime.windows if s.recorded_logger_id is not None)
    a = log_window(ENV, faulty, 1000, 0, np.random.default_rng(3))
    from dataclasses import replace
    b = log_window(ENV, replace(faulty, recorded_logger_id=None), 1000, 0, np.random.default_rng(3))
    assert (a.A == b.A).all() and (a.R == b.R).all()
    assert not np.allclose(a.pscore, b.pscore)


def test_true_value_matches_on_policy_simulation():
    """V_T(pi) via MC over contexts equals the mean reward of acting with pi."""
    pi = Policy.near_optimal(ENV)
    target = REGIMES["S0"](0.0).target
    v = true_value(ENV, pi, target, target, n_mc=400_000)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400_000, ENV.d))
    p = pi.probs(X)
    A = (p.cumsum(axis=1) < rng.random((len(X), 1))).sum(axis=1).clip(max=ENV.n_actions - 1)
    R = rng.random(len(X)) < q(ENV, target, X)[np.arange(len(X)), A]
    assert abs(R.mean() - v) < 4 * R.std() / np.sqrt(len(X))


def test_transfer_bias_zero_without_reward_shift_and_nonzero_with_it():
    pi = Policy.near_optimal(ENV)
    s1, s3 = REGIMES["S1"](2.0), REGIMES["S3a"](1.0)
    v_t = true_value(ENV, pi, s1.target, s1.target, n_mc=200_000)
    assert true_value(ENV, pi, s1.target, s1.windows[0], n_mc=200_000) == pytest.approx(v_t)
    assert abs(true_value(ENV, pi, s3.target, s3.windows[0], n_mc=200_000) - v_t) > 0.01


@pytest.mark.parametrize("estimator", ["ips", "snips", "dr"])
def test_estimators_recover_truth_on_large_stationary_sample(estimator):
    pi = Policy.near_optimal(ENV)
    regime = REGIMES["S0"](0.0)
    data = log_window(ENV, regime.target, 200_000, 0, np.random.default_rng(5))
    probs = pi.probs(data.X)
    if estimator == "dr":
        q_hat, _ = crossfit_rewards(data, ENV.n_actions)
        e = est.dr(data, probs, q_hat)
    else:
        e = getattr(est, estimator)(data, probs)
    truth = true_value(ENV, pi, regime.target, regime.target)
    assert abs(e.value - truth) < 4 * e.se


def test_episode_shapes():
    regime = REGIMES["S4"](1.0)
    ep = sample_episode(ENV, regime, n_per_window=500, n_anchor=100, n_target_contexts=300, seed=0)
    assert len(ep.history) == regime.n_windows
    assert len(LoggedData.concat(ep.history)) == 500 * regime.n_windows
    assert ep.anchor.window[0] == regime.n_windows
    assert ep.target_contexts.shape == (300, ENV.d)
