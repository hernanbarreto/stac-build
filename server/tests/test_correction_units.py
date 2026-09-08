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
