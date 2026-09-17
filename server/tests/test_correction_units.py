"""Units: keyframe is atomic; chunks come ONLY from the persisted plan; no
fixed divisor exists anywhere in the package."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.units import (chunks_of_keyframe,            # noqa: E402
                              load_chunk_plan,
                              visits_from_keyframes)

PKG = Path(__file__).resolve().parents[1] / "correction"


def test_visits_grouping():
    kfs = np.array([0, 1, 2, 5, 6, 20, 21, 23])
    v = visits_from_keyframes(kfs, gap_kf=2)
    assert v == [[0, 1, 2, 5, 6], [20, 21, 23]]
    v0 = visits_from_keyframes(kfs, gap_kf=0)
    assert v0 == [[0, 1, 2], [5, 6], [20, 21], [23]]


def test_no_plan_means_no_chunks(tmp_path):
    assert load_chunk_plan(tmp_path) is None
    assert chunks_of_keyframe(None, 7) == []


def test_plan_respects_real_ranges_and_overlap(tmp_path):
    # the exact shape map_worker persists: chunk_plan.chunk_ranges with
    # 50% overlap (reconstruction/chunk_plan.py:chunk_ranges)
    from reconstruction.chunk_plan import chunk_ranges
    n_kf, size, ov = 100, 40, 20
    plan = {"version": 1, "phase": "chunked-metric", "n_keyframes": n_kf,
            "chunk_size": size, "overlap": ov,
            "chunk_ranges": [[a, b] for a, b in
                             chunk_ranges(n_kf, size, ov)], "walk_m": 50.0}
    (tmp_path / "chunk_plan.json").write_text(json.dumps(plan))
    loaded = load_chunk_plan(tmp_path)
    assert loaded["chunk_ranges"] == plan["chunk_ranges"]
    # a keyframe inside an overlap belongs to BOTH real chunks
    assert chunks_of_keyframe(loaded, 25) == [0, 1]
    assert chunks_of_keyframe(loaded, 5) == [0]
    assert chunks_of_keyframe(loaded, 95) == [len(plan["chunk_ranges"]) - 1]


def test_corrupt_plan_is_an_error(tmp_path):
    (tmp_path / "chunk_plan.json").write_text('{"chunk_ranges": "nope"}')
    with pytest.raises(RuntimeError, match="chunk_plan.json"):
        load_chunk_plan(tmp_path)


def test_no_fixed_divisor_in_the_package():
    """H1: `// 30` (or any fixed keyframe-bucket divisor) must not exist."""
    pattern = re.compile(r"//\s*\d|%\s*30|\* 30 \+")
    offenders = []
    for py in sorted(PKG.glob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                offenders.append(f"{py.name}:{i}: {line.strip()}")
    assert not offenders, offenders


# ── a block seen continuously still has two visits (USER 2026-09-17) ─────

def _visits_from_pairs():
    from correction.revisit import _visits_from_pairs as f
    return f


def test_the_pairs_split_a_block_that_has_no_temporal_gap():
    """pccr 2026-09-17, the regression this pins: ``covisibility`` keeps only
    keyframe pairs at least min_gap_kf apart — every one of them asserts a
    revisit — and then the block was split into visits by looking for a gap in
    the flat list of its observers. A surface in view continuously along the
    walk has no such gap, so the block collapsed to ONE visit and was thrown
    away: both blocks died there, with 2,424 and 565 hits against a 300
    minimum, the session reported 0 revisit regions, and the pose graph was
    left with no clean edge to close (the 09-14 run closed on exactly these).
    """
    import numpy as np
    f = _visits_from_pairs()
    # observers 0..60 with no 30-keyframe gap anywhere, and pairs that DO span
    i = np.array([0, 5, 10, 15, 20, 25, 30])
    j = np.array([40, 45, 50, 55, 58, 59, 60])
    visits = f(i, j)
    assert len(visits) == 2, visits
    early, late = visits
    assert max(early) < min(late)
    assert set(early) | set(late) == set(i) | set(j)
    # every pair must cross the split — that is what the pairs asserted
    assert all((a in early and b in late) or (a in late and b in early)
               for a, b in zip(i, j))


def test_the_cut_is_where_most_pairs_cross_not_an_arbitrary_place():
    import numpy as np
    f = _visits_from_pairs()
    # six pairs cross between 12 and 40; one outlier pair crosses between 1 and 2
    i = np.array([10, 11, 12, 10, 11, 12, 1])
    j = np.array([40, 41, 42, 43, 44, 45, 2])
    early, late = f(i, j)
    assert max(early) <= 12 and min(late) >= 40, (early, late)


def test_a_block_with_one_observer_or_no_pair_yields_nothing():
    import numpy as np
    f = _visits_from_pairs()
    assert f(np.array([7]), np.array([7])) == []
    assert f(np.array([], dtype=int), np.array([], dtype=int)) == []


def test_the_order_of_a_pair_does_not_matter():
    import numpy as np
    f = _visits_from_pairs()
    a = f(np.array([5, 6]), np.array([50, 51]))
    b = f(np.array([50, 51]), np.array([5, 6]))
    assert a == b == [[5, 6], [50, 51]]


# ── the projection removes DOF from the MOTION, not from a raw vector ────

def test_projecting_a_rotating_closure_needs_the_point_it_turns_about():
    """pccr 2026-09-17, desk#201: a closure of 165.3° that lands its copy on
    its twin 60 cm away has |t| = 10.1 m, because t is the translation part of
    a rotation about a DISTANT origin, not a displacement. Projecting that raw
    t perpendicular to the desk's axis asked for a 14 m translation — the
    greedy loop measured "its own copies 60.7 → 823.5 cm", and with 19 edges
    like it the pose graph closed 0% and the session stayed at epoch 0."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from correction.solve import project_solution
    ax = np.array([0.2, 0.95, 0.24]); ax /= np.linalg.norm(ax)
    R = Rotation.from_rotvec(np.deg2rad(165.3) * ax).as_matrix()
    c = np.array([4.0, 1.2, 7.8])
    t = np.array([0.55, 0.05, 0.25]) + c - R @ c        # moves the object ~0.99 m
    pts = c + np.random.default_rng(0).normal(0, 0.35, (2000, 3))
    real = np.median(np.linalg.norm(pts @ R.T + t - pts, axis=1))
    assert 0.9 < real < 1.1 and np.linalg.norm(t) > 10.0   # the trap, in one line
    spec = {"mode": "perp_axis", "axis": np.array([1.0, 0.0, 0.15])}
    _R1, t1 = project_solution(R, t, spec)                     # no reference point
    _R2, t2 = project_solution(R, t, spec, about=c)            # about the object
    assert np.linalg.norm(t1) > 10.0, "the raw t is not a displacement"
    assert np.linalg.norm(t2) < real, f"{np.linalg.norm(t2):.2f} m"


def test_about_changes_nothing_when_the_solution_carries_no_rotation():
    """Every caller that runs its ICP with rotation=full only ever projects a
    pure translation; those must be untouched."""
    import numpy as np
    from correction.solve import project_solution
    I = np.eye(3); t = np.array([0.3, -0.1, 0.2]); c = np.array([4.0, 1.2, 7.8])
    ax = np.array([1.0, 0.0, 0.15])
    for spec in ({"mode": "perp_axis", "axis": ax}, {"mode": "normal", "normal": ax},
                 {"mode": "translation"}):
        r1, t1 = project_solution(I, t, spec)
        r2, t2 = project_solution(I, t, spec, about=c)
        assert np.allclose(r1, r2) and np.allclose(t1, t2), spec["mode"]


def test_full_observability_is_never_projected_with_or_without_about():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from correction.solve import project_solution
    R = Rotation.from_rotvec([0.3, 0.2, 0.1]).as_matrix(); t = np.array([1.0, 2.0, 3.0])
    for about in (None, np.array([4.0, 1.2, 7.8])):
        R2, t2 = project_solution(R, t, {"mode": "full"}, about=about)
        assert np.allclose(R2, R) and np.allclose(t2, t)
