"""Drift regimes S0-S7 (proposal §6.1), each parameterized by one intensity.

All regimes use K historical windows, oldest first, and a target period whose
logger is the most recent one. `intensity` is in logit units for reward shift
and in context-standard-deviation units for context shift.
"""

import math

from .simulator import Regime, WindowSpec

K_DEFAULT = 12

# Three deployments, as in the proposal's May-Oct example: A (exploratory),
# B (greedier, less exploration), C (current).
LOGGER_A = dict(logger_id=0, logger_beta=1.0, logger_eps=0.3)
LOGGER_B = dict(logger_id=1, logger_beta=2.0, logger_eps=0.1)
LOGGER_C = dict(logger_id=2, logger_beta=1.0, logger_eps=0.2)


def _logger_for(k: int, K: int) -> dict:
    return LOGGER_A if k < K // 3 else LOGGER_B if k < 2 * K // 3 else LOGGER_C


def s0_stationary(intensity: float = 0.0, K: int = K_DEFAULT) -> Regime:
    spec = WindowSpec(**LOGGER_C)
    return Regime("S0", tuple(spec for _ in range(K)), spec)


def s1_covariate(intensity: float, K: int = K_DEFAULT) -> Regime:
    """Older windows have increasingly shifted contexts; q and mu are stable."""
    return Regime("S1", tuple(WindowSpec(context_shift=intensity * (K - k) / K, **LOGGER_C)
                              for k in range(K)), WindowSpec(**LOGGER_C))


def s2_logger(intensity: float = 0.0, K: int = K_DEFAULT) -> Regime:
    """Logging policy switches A -> B -> C; P(X) and q are stable."""
    return Regime("S2", tuple(WindowSpec(**_logger_for(k, K)) for k in range(K)), WindowSpec(**LOGGER_C))


def s3_reward_abrupt(intensity: float, K: int = K_DEFAULT) -> Regime:
    """The older half of the history has a different conditional reward."""
    return Regime("S3a", tuple(WindowSpec(reward_shift=intensity if k < K // 2 else 0.0, **LOGGER_C)
                               for k in range(K)), WindowSpec(**LOGGER_C))


def s3_reward_gradual(intensity: float, K: int = K_DEFAULT) -> Regime:
    """Conditional reward drifts linearly toward the target over time."""
    return Regime("S3b", tuple(WindowSpec(reward_shift=intensity * (K - k) / K, **LOGGER_C)
                               for k in range(K)), WindowSpec(**LOGGER_C))


def s4_mixed(intensity: float, K: int = K_DEFAULT) -> Regime:
    """The proposal's worked example (§5), the title's central test case.

    Oldest third (logger A): small context shift, reward shift -> should be rejected.
    Middle third (logger B): large context shift, stable reward -> should be transported.
    Recent third (logger C): no shift but the target's favoured actions are
    poorly explored (low eps) -> recency alone is not enough.
    """
    windows = []
    for k in range(K):
        if k < K // 3:
            windows.append(WindowSpec(context_shift=0.25 * intensity, reward_shift=intensity, **LOGGER_A))
        elif k < 2 * K // 3:
            windows.append(WindowSpec(context_shift=1.5 * intensity, **LOGGER_B))
        else:
            windows.append(WindowSpec(**{**LOGGER_C, "logger_beta": 3.0, "logger_eps": 0.02}))
    return Regime("S4", tuple(windows), WindowSpec(**LOGGER_C))


def s5_seasonal(intensity: float, K: int = K_DEFAULT, period: int = 6) -> Regime:
    """Context and reward vary periodically; windows in phase with the target match it."""
    windows = []
    for k in range(K):
        phase = 2 * math.pi * (K - k) / period
        windows.append(WindowSpec(context_shift=intensity * math.sin(phase),
                                  reward_shift=intensity * (1 - math.cos(phase)) / 2,
                                  **_logger_for(k, K)))
    return Regime("S5", tuple(windows), WindowSpec(**LOGGER_C))


def s6_missing_metadata(intensity: float = 0.0, K: int = K_DEFAULT) -> Regime:
    """S2 loggers, but logger-B windows record propensities recomputed from
    the newest deployed logger C (assumption A4 violated)."""
    windows = []
    for k in range(K):
        spec = WindowSpec(**_logger_for(k, K))
        if spec.logger_id == LOGGER_B["logger_id"]:
            spec = WindowSpec(**LOGGER_B, recorded_logger_id=LOGGER_C["logger_id"])
        windows.append(spec)
    return Regime("S6", tuple(windows), WindowSpec(**LOGGER_C))


def s7_support_loss(intensity: float = 0.0, K: int = K_DEFAULT, actions: tuple = (0, 1, 2)) -> Regime:
    """Recent windows never play `actions`; the target logger does."""
    windows = tuple(WindowSpec(**LOGGER_C, unsupported_actions=actions if k >= K // 2 else ())
                    for k in range(K))
    return Regime("S7", windows, WindowSpec(**LOGGER_C))


def s8_misspecified_drift(intensity: float, K: int = K_DEFAULT) -> Regime:
    """S3a with the reward drift along an out-of-class direction h (two-direction
    interaction): the per-action linear reward model cannot represent the drift, so
    drift-corrected reuse is no longer exact. Robustness test of assumption A-corr."""
    return Regime("S8", tuple(WindowSpec(reward_shift=intensity if k < K // 2 else 0.0, drift_nonlinear=1.0,
                                         **LOGGER_C) for k in range(K)), WindowSpec(**LOGGER_C))


REGIMES = {
    "S0": s0_stationary, "S1": s1_covariate, "S2": s2_logger,
    "S8": s8_misspecified_drift,
    "S3a": s3_reward_abrupt, "S3b": s3_reward_gradual, "S4": s4_mixed,
    "S5": s5_seasonal, "S6": s6_missing_metadata, "S7": s7_support_loss,
}
