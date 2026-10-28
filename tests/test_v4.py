import numpy as np

from odrift.method import gate_components, covariance


class _C:
    def __init__(self, value, rng, n=4000, sd=0.5):
        self.value = value
        self.dm_scores = np.full(200, value) + rng.normal(scale=1e-3, size=200)
        self.corr_scores = rng.normal(scale=sd, size=n)


def _comps(biases, seed=0):
    rng = np.random.default_rng(seed)
    return [_C(0.5 + b, rng) for b in biases]  # positions: old..., reference, anchor (last)


def test_gate_drops_a_detectably_biased_segment_and_keeps_an_unbiased_one():
    comps = _comps([0.10, 0.0, 0.0, 0.0])  # biased old, clean old, reference, anchor
    kept = gate_components(comps, covariance(comps), ref_pos=[2], z=1.96)
    assert 0 not in kept and {1, 2, 3} <= set(kept)


def test_infinite_gate_keeps_everything():
    comps = _comps([0.10, 0.0, 0.0, 0.0])
    assert gate_components(comps, covariance(comps), ref_pos=[2], z=np.inf) == [0, 1, 2, 3]
