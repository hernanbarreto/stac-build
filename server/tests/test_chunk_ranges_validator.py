"""The fork's explicit chunk layout (vendor/VGGT-Long/loop_utils/metric_lock.py):
validate_chunk_ranges — what Model.chunk_ranges must satisfy, each violation failing
loudly by name — uniform_chunk_ranges (the vendor's layout, unchanged) and
seam_overlap (each seam's own shared-frame count). Also: every layout the
co-visibility planner (reconstruction/chunk_covis.py) produces is one the fork runs.
CPU only, no model."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "vendor", "VGGT-Long")))

from loop_utils.metric_lock import (MIN_SEAM_FRAMES, ChunkRangesError,  # noqa: E402
                                    frame_owner, seam_overlap, uniform_chunk_ranges,
                                    validate_chunk_ranges)

# The pccr plan of 2026-10-06 (289 keyframes, cap 883): five chunks, seams 33/106/20/22.
PCCR = [(0, 63), (30, 169), (63, 189), (169, 211), (189, 289)]


def _vendor_layout(n, chunk_size, overlap):
    """VGGT-Long's process_long_sequence layout, verbatim from before the explicit ranges."""
    if overlap >= chunk_size:
        raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size "
                         f"({chunk_size})")
    if n <= chunk_size:
        return [(0, n)]
    step = chunk_size - overlap
    num_chunks = (n - overlap + step - 1) // step
    out = []
    for i in range(num_chunks):
        start_idx = i * step
        end_idx = min(start_idx + chunk_size, n)
        out.append((start_idx, end_idx))
    return out


_UNIFORM_CASES = [(100, 60, 30), (90, 60, 30), (95, 60, 30), (66, 40, 20), (227, 80, 40),
                  (57, 60, 30), (61, 60, 30), (289, 296, 148), (2000, 296, 148),
                  (1998, 120, 60), (500, 75, 30), (300, 60, 20), (121, 60, 59), (10, 3, 1)]


# ── the uniform layout: unchanged, and its seams ARE `overlap` ───────────────

@pytest.mark.parametrize("n,size,ov", _UNIFORM_CASES)
def test_uniform_ranges_are_the_vendor_layout(n, size, ov):
    from reconstruction.chunk_plan import chunk_ranges
    assert uniform_chunk_ranges(n, size, ov) == _vendor_layout(n, size, ov)
    assert uniform_chunk_ranges(n, size, ov) == chunk_ranges(n, size, ov)


def test_uniform_overlap_not_below_size_is_the_vendor_setting_error():
    with pytest.raises(ValueError, match=r"\[SETTING ERROR\] Overlap \(60\)"):
        uniform_chunk_ranges(100, 60, 60)


@pytest.mark.parametrize("n,size,ov", _UNIFORM_CASES)
def test_uniform_seams_share_exactly_the_configured_overlap(n, size, ov):
    """seam_overlap == overlap on every seam of a uniform layout: the seam fits that now
    slice [-ov:] / [:ov] cut exactly the vendor's [-overlap:] / [:overlap]."""
    ci = uniform_chunk_ranges(n, size, ov)
    assert [seam_overlap(ci, k) for k in range(len(ci) - 1)] == [ov] * (len(ci) - 1)


@pytest.mark.parametrize("n,size", [(100, 60), (227, 80), (2000, 296), (289, 296), (61, 60)])
def test_a_uniform_half_overlap_layout_is_a_valid_explicit_one(n, size):
    ci = uniform_chunk_ranges(n, size, size // 2)
    assert validate_chunk_ranges([list(r) for r in ci], n) == ci


# ── a valid explicit layout ─────────────────────────────────────────────────

def test_the_pccr_plan_passes_with_its_own_seams():
    ci = validate_chunk_ranges([list(r) for r in PCCR], 289)
    assert ci == PCCR
    assert [seam_overlap(ci, k) for k in range(len(ci) - 1)] == [33, 106, 20, 22]
    owner = frame_owner(ci, 289)
    assert owner[0] == 0 and owner[-1] == len(ci) - 1
    assert set(np.diff(owner).tolist()) <= {0, 1}


def test_numpy_and_tuple_inputs_come_back_as_python_int_tuples():
    for ranges in (np.asarray(PCCR, dtype=np.int64),
                   [(np.int32(a), np.int64(b)) for a, b in PCCR],
                   tuple(tuple(r) for r in PCCR)):
        ci = validate_chunk_ranges(ranges, np.int64(289))
        assert ci == PCCR
        assert all(type(x) is int for r in ci for x in r)
        assert all(type(r) is tuple for r in ci)


def test_one_pass_is_one_range():
    assert validate_chunk_ranges([[0, 183]], 183) == [(0, 183)]
    assert validate_chunk_ranges([[0, 3]], 3) == [(0, 3)]


def test_the_structural_check_runs_before_the_frame_list_exists():
    """n_frames None (the early check in __init__): the end is not compared, the rest is."""
    assert validate_chunk_ranges([list(r) for r in PCCR]) == PCCR
    with pytest.raises(ChunkRangesError, match="three chunks"):
        validate_chunk_ranges([[0, 60], [20, 80], [50, 120]])


def test_the_seam_floor_is_eight_frames_and_inclusive():
    assert MIN_SEAM_FRAMES == 8
    assert validate_chunk_ranges([[0, 60], [52, 120]], 120) == [(0, 60), (52, 120)]
    with pytest.raises(ChunkRangesError, match="seam 0->1 has 7 shared frame"):
        validate_chunk_ranges([[0, 60], [53, 120]], 120)


# ── every violation fails loudly, naming the problem ────────────────────────

@pytest.mark.parametrize("ranges,n,msg", [
    (None, 10, "non-empty list"),
    ([], 10, "non-empty list"),
    ("0,10", 10, "non-empty list"),
    ({"a": 1}, 10, "non-empty list"),
    ([[0, 10, 20]], 20, r"chunk 0 is \[0, 10, 20\], not a \[start, end\] pair"),
    ([5], 5, "chunk 0 is 5, not a"),
    ([[0, 60.0]], 60, "integer keyframe indices"),
    ([[0, "60"]], 60, "integer keyframe indices"),
    ([[False, 60]], 60, "integer keyframe indices"),
    ([[0, 60], [30, 30]], 60, r"chunk 1 \[30, 30\) is empty"),
    ([[0, 60], [40, 20]], 60, r"chunk 1 \[40, 20\) is empty"),
    ([[-5, 60]], 60, "negative"),
    ([[1, 60]], 60, "first chunk starts at 1, it must start at 0"),
    ([[0, 60], [30, 100]], 99, "last chunk ends at 100 but this run has 99 frames"),
    ([[0, 60], [30, 100]], 120, "planned for another frame list"),
    ([[0, 60], [0, 100]], 100, "not sorted — chunk 1"),
    ([[0, 60], [30, 60], [40, 100]], 100, "not sorted — chunk 1"),
    ([[0, 60], [62, 120]], 120, r"seam 0->1 has a gap of 2 frame\(s\) \[60, 62\)"),
    ([[0, 60], [60, 120]], 120, r"seam 0->1 has 0 shared frame\(s\)"),
    ([[0, 60], [55, 120]], 120, "at least 8 shared frames"),
    ([[0, 60], [20, 80], [50, 120]], 120,
     r"frames \[50, 60\) are in three chunks \(0, 1, 2\)"),
    ([[0, 63], [30, 169], [63, 189], [160, 211], [189, 289]], 289,
     r"frames \[160, 169\) are in three chunks \(1, 2, 3\)"),
])
def test_every_malformed_layout_fails_naming_the_problem(ranges, n, msg):
    with pytest.raises(ChunkRangesError, match=msg):
        validate_chunk_ranges(ranges, n)


def test_the_error_is_a_value_error_and_names_the_config_key():
    with pytest.raises(ValueError) as ei:
        validate_chunk_ranges([[0, 60], [62, 120]], 120)
    assert str(ei.value).startswith("Model.chunk_ranges")


# ── the producer and the consumer agree ─────────────────────────────────────

def test_every_planner_layout_is_a_valid_fork_layout():
    """chunk_covis.plan over budgets cheap, expensive, mixed and noisy, at capacities that
    force chunking: each plan passes the fork's validator unchanged, and every seam
    is a whole block (>= MIN_CHUNK_FRAMES // 2 > MIN_SEAM_FRAMES)."""
    from reconstruction.chunk_covis import MIN_CHUNK_FRAMES, PlanError, plan
    rng = np.random.default_rng(20261006)
    checked = 0
    for trial in range(40):
        n = int(rng.integers(30, 400))
        kind = trial % 4
        if kind == 0:
            d = np.full(n, float(rng.uniform(0.01, 0.6)))
        elif kind == 1:
            d = rng.uniform(0.0, 0.5, n)
        elif kind == 2:
            cut = int(rng.integers(1, n))
            d = np.concatenate([np.full(cut, 0.02), np.full(n - cut, 0.4)])
        else:
            d = np.abs(rng.normal(0.1, 0.2, n))
        z = rng.uniform(0.8, 12.0, n)
        t = rng.uniform(0.0, 8.0, n - 1)
        try:
            # no card capacity in the plan (USER 2026-10-06): Omega's resolution adapts
            ranges = plan(d, z, t)
        except PlanError:
            continue                       # a walk the planner refuses: nothing to run
        ci = validate_chunk_ranges([list(r) for r in ranges], n)
        assert ci == [tuple(r) for r in ranges]
        for k in range(len(ci) - 1):
            assert seam_overlap(ci, k) >= MIN_CHUNK_FRAMES // 2 > MIN_SEAM_FRAMES - 1
        checked += 1
    assert checked >= 30
