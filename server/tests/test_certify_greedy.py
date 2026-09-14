"""The greedy correction loop (USER's mechanic, 2026-09-14).

One duplicate at a time, and the measurement — points of an instance landing
inside its mask — decides; a failed candidate returns to the pool; a full pass
with no improvement is convergence.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.certify.iterate import (Agreement, Candidate, GreedyLoop,   # noqa: E402
                                            _offset_m, _rate_order)


# ── the rule itself ──────────────────────────────────────────────────────

def test_more_points_in_mask_is_better_and_fewer_is_worse():
    base = Agreement(1000, 2000, {}, {})
    assert Agreement(1400, 2000, {}, {}).better_than(base)
    assert not Agreement(600, 2000, {}, {}).better_than(base)


def test_a_tie_is_not_an_improvement():
    """The chain only grows on evidence. A trial that changes nothing must not
    consume a candidate, or the loop would 'converge' by exhausting the pool."""
    base = Agreement(1000, 2000, {}, {})
    assert not Agreement(1000, 2000, {}, {}).better_than(base)


def test_the_measure_is_the_count_not_the_fraction():
    """Seeing more of the object must not be punished: a state where 1400 of
    2500 points land in their masks is better than 1000 of 2000 even though the
    FRACTION fell, because more of the cloud is where the images say it is."""
    assert Agreement(1400, 2500, {}, {}).better_than(Agreement(1000, 2000, {}, {}))


# ── the sample is fixed, which is what removes the threshold ─────────────

def _cand(key=0, n_a=10000, n_b=8000, samples=1000):
    edge = {"i": 60, "j": 5, "instance_id": 7, "label": "desk",
            "X": np.eye(4), "later_kfs": [58, 62]}
    return Candidate(key, edge, np.arange(n_a), np.arange(n_b),
                     [1, 2], [3, 4], oid=6, samples=samples)


def test_the_sample_is_deterministic_across_trials():
    a, b = _cand(), _cand()
    assert np.array_equal(a.sample_a, b.sample_a)
    assert np.array_equal(a.sample_b, b.sample_b)
    assert len(a.sample_a) == 1000 and len(a.sample_b) == 1000


def test_different_candidates_sample_differently():
    """Same seed for the same candidate, a different draw for another — one
    candidate's sample must not stand in for the whole scene."""
    assert not np.array_equal(_cand(key=0).sample_a, _cand(key=1).sample_a)


def test_a_small_copy_keeps_every_point():
    c = _cand(n_a=300, n_b=200, samples=1000)
    assert np.array_equal(c.sample_a, np.arange(300))
    assert np.array_equal(c.sample_b, np.arange(200))


# ── the trial order is a heuristic, never a filter ───────────────────────

def test_rate_order_puts_the_consensus_first_and_keeps_everyone():
    """Under E(d)=eps*d a closure is only comparable once divided by the walk
    between its ends. The outlier is tried LAST, never dropped."""
    d_kf = np.linspace(0.0, 20.0, 80)

    def c(key, i, j, t):
        e = {"i": i, "j": j, "instance_id": key, "label": f"o{key}",
             "X": np.array([[1, 0, 0, t], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float),
             "later_kfs": [i, i]}
        return Candidate(key, e, np.arange(50), np.arange(50), [], [], oid=key)

    # three agree on ~2 cm/m over their own stretch, one demands 20x that
    pool = [c(0, 70, 10, 3.0), c(1, 60, 20, 2.0), c(2, 75, 5, 3.5), c(3, 40, 30, 5.0)]
    ordered = _rate_order(pool, d_kf)
    assert len(ordered) == len(pool), "the order must never drop a candidate"
    assert ordered[-1].key == 3, "the outlier is tried last, not discarded"


def test_a_failed_candidate_is_ordered_last_but_stays():
    d_kf = np.linspace(0.0, 20.0, 80)
    pool = [_cand(0), _cand(1)]
    pool[0].failures = 3
    ordered = _rate_order(pool, d_kf)
    assert len(ordered) == 2
    assert ordered[-1].key == 0


# ── the deviation reported alongside the count ───────────────────────────

def test_offset_measures_the_separation_between_copies():
    """A tight cluster displaced by 1.20 m reads 1.20 m. The cluster has to be
    tight for the assertion to be exact: nearest-neighbour distance between two
    clouds of real extent is shorter than the displacement of their centres —
    two blobs of sigma 5 cm at 1.20 m read 1.02 — which is why this is the
    number REPORTED alongside the curve and never the one that decides."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.002, (2000, 3))
    b = a + np.array([1.20, 0.0, 0.0])
    assert _offset_m(a, b) == pytest.approx(1.20, abs=0.01)
    assert _offset_m(a, a) == pytest.approx(0.0, abs=1e-6)


def test_offset_is_nan_on_an_empty_copy():
    assert np.isnan(_offset_m(np.zeros((0, 3)), np.ones((5, 3))))


# ── composition of the chain ─────────────────────────────────────────────

class _FakeLoop:
    """GreedyLoop.composed() without a session: the composition rule is pure."""

    def __init__(self, chain, n_kf):
        self.chain, self.n_kf = chain, n_kf

    composed = GreedyLoop.composed


def _step(n, R, t, k=1.0):
    return {"R_kf": np.repeat(np.asarray(R, float)[None], n, 0),
            "t_kf": np.repeat(np.asarray(t, float)[None], n, 0),
            "k_kf": np.full(n, k)}


def test_an_empty_chain_composes_to_nothing():
    assert _FakeLoop([], 10).composed() is None


def test_the_chain_composes_left_to_right_like_every_other_stage():
    """R = Rn...R1 and t = Rn*t_prev + tn — the convention the transaction and
    the pose transform already use. Get it backwards and the epoch replays a
    different geometry than the loop measured."""
    n = 4
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    Rz = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    chain = [_step(n, np.eye(3), [1.0, 0.0, 0.0]), _step(n, Rz, [0.0, 0.0, 0.0])]
    R, t, k = _FakeLoop(chain, n).composed()
    # first translate by +x, then rotate 90 deg about z: the translation follows
    assert np.allclose(R[0], Rz, atol=1e-9)
    assert np.allclose(t[0], [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(k, 1.0)


def test_depth_factors_multiply_along_the_chain():
    n = 3
    chain = [_step(n, np.eye(3), [0, 0, 0], k=1.02),
             _step(n, np.eye(3), [0, 0, 0], k=0.98)]
    _R, _t, k = _FakeLoop(chain, n).composed()
    assert np.allclose(k, 1.02 * 0.98)


def test_a_single_step_composes_to_itself():
    n = 5
    chain = [_step(n, np.eye(3), [0.3, -0.2, 0.1], k=1.01)]
    R, t, k = _FakeLoop(chain, n).composed()
    assert np.allclose(R, np.repeat(np.eye(3)[None], n, 0))
    assert np.allclose(t[0], [0.3, -0.2, 0.1])
    assert np.allclose(k, 1.01)


# ── the config carries every decision ────────────────────────────────────

def test_no_decision_literal_lives_in_the_package():
    """Same rule the correction package is held to: the loop's knobs come from
    config.yaml. The one number that must NOT exist is a minimum gain — USER
    2026-09-14 rejected it as arbitrary, and the fixed sample makes it
    unnecessary."""
    from tests.synth_metric import raw_server_cfg
    from reconstruction.loops.config import load_loops_config
    g = load_loops_config(raw_server_cfg()).certify.greedy
    assert {"enabled", "max_epochs", "window_kf", "offset_samples"} <= set(vars(g))
    assert not hasattr(g, "min_gain")
    assert not hasattr(g, "max_reprojection_drop")
    src = (Path(__file__).resolve().parents[1]
           / "reconstruction" / "certify" / "iterate.py").read_text()
    assert "min_gain" not in src


def test_a_missing_greedy_key_fails_at_load_naming_it():
    from tests.synth_metric import raw_server_cfg, certify_cfg
    from reconstruction.loops.config import load_loops_config, LoopsConfigError
    ce = certify_cfg()
    ce["greedy"].pop("max_epochs")
    with pytest.raises(LoopsConfigError, match="certify.greedy.max_epochs"):
        load_loops_config(raw_server_cfg(certify=ce))
