import numpy as np

from odrift.method import ODriftConfig, context_ratio, odrift, transported_dr
from odrift.regimes import REGIMES
from odrift.simulator import EnvParams, LoggedData, Policy, log_window, sample_contexts, sample_episode, true_value

ENV = EnvParams.sample(seed=0)
PI = Policy.near_optimal(ENV)
Z = 3.0


def _window(regime, k, n=20_000, seed=0):
    return log_window(ENV, regime.windows[k], n, k, np.random.default_rng(seed))


def test_context_ratio_matches_gaussian_mean_shift():
    regime = REGIMES["S1"](2.0)
    rng = np.random.default_rng(0)
    Xs = sample_contexts(ENV, regime.windows[0], 20_000, rng)
    Xt = sample_contexts(ENV, regime.target, 20_000, rng)
    w = context_ratio(Xs, Xt)
    shift = regime.windows[0].context_shift * ENV.u
    true_w = np.exp(-(Xs @ shift) + shift @ shift / 2)
    true_w /= true_w.mean()
    assert np.corrcoef(np.log(w), np.log(true_w))[0, 1] > 0.99


def test_transported_dr_recovers_transported_value_under_context_shift():
    regime = REGIMES["S1"](1.0)
    data = _window(regime, 0)
    Xt = sample_contexts(ENV, regime.target, 20_000, np.random.default_rng(1))
    c = transported_dr(data, Xt, PI, ENV.n_actions, 0, Z, 50)
    truth = true_value(ENV, PI, regime.target, regime.windows[0], n_mc=400_000)
    assert abs(c.value - truth) < 4 * np.sqrt(c.var)
    assert c.diagnostics.eligible


def test_diagnostics_flag_wrong_propensities():
    regime = REGIMES["S6"](0.0)
    k = next(i for i, s in enumerate(regime.windows) if s.recorded_logger_id is not None)
    Xt = sample_contexts(ENV, regime.target, 5000, np.random.default_rng(1))
    c = transported_dr(_window(regime, k, n=5000), Xt, PI, ENV.n_actions, k, Z, 50)
    assert "propensity_inconsistent" in c.diagnostics.reasons


def test_diagnostics_flag_missing_support():
    regime = REGIMES["S7"](0.0)
    Xt = sample_contexts(ENV, regime.target, 5000, np.random.default_rng(1))
    c = transported_dr(_window(regime, regime.n_windows - 1, n=5000), Xt, PI, ENV.n_actions, 0, Z, 50)
    assert "target_support" in c.diagnostics.reasons


def test_clean_windows_pass_diagnostics():
    regime = REGIMES["S2"](0.0)
    Xt = sample_contexts(ENV, regime.target, 5000, np.random.default_rng(1))
    for k in (0, 5, 11):
        c = transported_dr(_window(regime, k, n=5000, seed=k), Xt, PI, ENV.n_actions, k, Z, 50)
        assert c.diagnostics.eligible, (k, c.diagnostics)


def test_no_anchor_is_not_certified():
    ep = sample_episode(ENV, REGIMES["S0"](0.0), 2000, 0, 2000, seed=0)
    res = odrift(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions)[1.0]
    assert res.status == "not_certified" and np.isinf(res.half_width)


def test_reward_shifted_windows_get_less_weight():
    regime = REGIMES["S3a"](2.0)
    ep = sample_episode(ENV, regime, 3000, 2000, 5000, seed=0)
    res = odrift(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions)[1.0]
    old = sum(res.weights.get(k, 0) for k in range(regime.n_windows // 2))
    assert res.status == "certified" and old < 0.1, res.weights


def test_split_variant_runs_and_weights_sum_to_one():
    ep = sample_episode(ENV, REGIMES["S4"](1.0), 2000, 1000, 4000, seed=0)
    out = odrift(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, split=True, lams=[0.5, 2.0])
    for res in out.values():
        assert abs(sum(res.weights.values()) - 1) < 1e-6


def test_changepoint_finds_abrupt_reward_change():
    from odrift.method import ODriftConfig, build_components, changepoint_groups, covariance
    regime = REGIMES["S3a"](2.0)
    ep = sample_episode(ENV, regime, 5000, 1000, 5000, seed=1)
    comps, _ = build_components(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, ODriftConfig())
    vals = np.array([c.value for c in comps])
    groups = changepoint_groups(vals, covariance(comps), 3.0)
    assert sorted(groups[0]) == list(range(6, 12)), groups


def test_changepoint_keeps_stationary_history_whole():
    from odrift.method import ODriftConfig, build_components, changepoint_groups, covariance
    ep = sample_episode(ENV, REGIMES["S1"](1.0), 5000, 1000, 5000, seed=2)
    comps, _ = build_components(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, ODriftConfig())
    groups = changepoint_groups(np.array([c.value for c in comps]), covariance(comps), 3.0)
    assert len(groups) == 1, groups


def test_reward_monitor_no_false_split_under_context_shift_and_finds_reward_split():
    from odrift.monitor import ResidualStats, reward_changepoints
    ep = sample_episode(ENV, REGIMES["S1"](1.0), 5000, 100, 100, seed=0)
    assert len(reward_changepoints(ResidualStats(ep.history, ENV.n_actions), 0.001)) == 1
    ep = sample_episode(ENV, REGIMES["S3a"](0.5), 5000, 100, 100, seed=0)
    segs = reward_changepoints(ResidualStats(ep.history, ENV.n_actions), 0.001)
    assert sorted(segs[0]) == list(range(6, 12)), segs


def test_combine_with_monitor_groups_drops_reward_shifted_windows():
    from odrift.method import ODriftConfig, build_components, combine
    ep = sample_episode(ENV, REGIMES["S3a"](1.0), 4000, 1000, 4000, seed=0)
    comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, ODriftConfig())
    r = combine(comps, anc, ODriftConfig(fallback="least"), window_groups=[list(range(6, 12)), list(range(6))])
    assert sum(r.weights.get(k, 0) for k in range(6)) < 1e-6


def test_segment_reward_models_unbiased_in_stationary_history():
    from odrift.method import ODriftConfig, build_components, combine, fit_all_nuisances
    regime = REGIMES["S0"](0.0)
    ep = sample_episode(ENV, regime, 4000, 1000, 10000, seed=3)
    cfg = ODriftConfig(fallback="least")
    nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, ENV.n_actions, cfg,
                             segments=[list(range(regime.n_windows))])
    comps, anc = build_components(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, cfg, nuisances=nuis)
    r = combine(comps, anc, cfg)
    truth = true_value(ENV, PI, regime.target, regime.target)
    assert abs(r.value - truth) < 4 * r.se


def test_drift_corrected_reuse_removes_reward_bias_of_old_windows():
    from odrift.method import ODriftConfig, fit_all_nuisances, odrift_corrected
    from odrift.monitor import ResidualStats, reward_changepoints
    regime = REGIMES["S3a"](1.0)
    truth = true_value(ENV, PI, regime.target, regime.target)
    errs = []
    for seed in range(4):
        ep = sample_episode(ENV, regime, 4000, 1000, 8000, seed=seed)
        segs = reward_changepoints(ResidualStats(ep.history, ENV.n_actions), 0.001)
        cfg = ODriftConfig(z_anchor=3.0, fallback="least")
        nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, ENV.n_actions, cfg, segments=segs)
        r = odrift_corrected(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, cfg, nuis, segs)
        assert sum(r.weights.get(k, 0) for k in range(6)) > 0.2  # old windows are reused
        errs.append(r.value - truth)
    assert abs(np.mean(errs)) < 0.01


def test_corrected_split_variant_runs():
    from odrift.method import ODriftConfig, make_corrected_split, odrift_corrected_split
    ep = sample_episode(ENV, REGIMES["S3a"](1.0), 4000, 1000, 6000, seed=0)
    cfg = ODriftConfig(z_anchor=3.0, fallback="least")
    sp = make_corrected_split(ep.history, ep.anchor, ep.target_contexts, ENV.n_actions, cfg, 0.001)
    r = odrift_corrected_split(sp, PI, ENV.n_actions, cfg)
    assert r.status == "certified" and np.isfinite(r.half_width)
    assert abs(sum(r.weights.values()) - 1) < 1e-6


def test_balanced_segment_pooling_unbiased_under_logger_switch():
    from dataclasses import replace as dc_replace
    from odrift.method import ODriftConfig, fit_all_nuisances, odrift_balanced
    from odrift.simulator import mu
    regime = REGIMES["S2"](0.0)
    truth = true_value(ENV, PI, regime.target, regime.target)
    fns = [lambda X, sp=sp: mu(ENV, sp, X) for sp in regime.windows]
    cfg = ODriftConfig(z_anchor=5.0, fallback="least", context_alpha=0.01)
    errs, ses = [], []
    for seed in range(4):
        ep = sample_episode(ENV, regime, 3000, 1000, 6000, seed=seed)
        segs = [list(range(regime.n_windows))]
        nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, ENV.n_actions, cfg, segments=segs)
        r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, cfg, nuis, segs, fns)
        errs.append(r.value - truth); ses.append(r.se)
    assert abs(np.mean(errs)) < 2 * np.mean(ses)


def test_balanced_weights_reported_per_window_and_exclude_deficient():
    from odrift.method import ODriftConfig, fit_all_nuisances, odrift_balanced
    from odrift.simulator import mu
    regime = REGIMES["S7"](0.0)
    fns = [lambda X, sp=sp: mu(ENV, sp, X) for sp in regime.windows]
    cfg = ODriftConfig(z_anchor=5.0, fallback="least", context_alpha=0.01)
    ep = sample_episode(ENV, regime, 3000, 1000, 4000, seed=0)
    segs = [list(range(regime.n_windows))]
    nuis = fit_all_nuisances(ep.history, ep.anchor, ep.target_contexts, ENV.n_actions, cfg, segments=segs)
    r = odrift_balanced(ep.history, ep.anchor, ep.target_contexts, PI, ENV.n_actions, cfg, nuis, segs, fns)
    assert abs(sum(r.weights.values()) - 1) < 1e-6 and set(r.weights) == set(range(regime.n_windows + 1))
    deficient = [k for k, why in r.reasons.items() if "target_support" in why]
    assert deficient and sum(r.weights[k] for k in deficient) == 0
    assert sum(r.weights[k] for k in range(regime.n_windows) if k not in deficient) > 0.5
