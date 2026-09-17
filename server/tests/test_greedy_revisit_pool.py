"""USER 2026-09-17: the greedy loop must try the GEOMETRIC revisit regions too.

pccr spent three days unable to fix a floor duplicate that the measurement had
already found. The loop tried 19 instance candidates and rejected every one,
correctly: `white_tiled_floor#44` is ONE instance of 5,699,353 points — 29.5%
of the segmented cloud — spanning the whole building, so its two "copies" are
two stretches of the same floor 7.4 m apart and closing them slides the floor
along itself (766 k observations worse, its own copies "closing" 736.6 -> 20.2
cm only by moving the floor 7 m).

The regions the revisit detector produces are 3 m blocks — local, and they
measured the real thing: region 1 reads 22.0 cm, which is exactly the vertical
duplication the user sees under the two desks (the point histogram of the zone
is bimodal at 0.00 m and +0.22 m), and closes it to 4.2 cm.

They never reached the loop because `_build_pool` skipped every edge without an
`instance_id`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.certify.iterate import Candidate, GreedyLoop   # noqa: E402


class _Session:
    """Points in and out of a 3 m block, owned by an early and a late visit."""
    xyz = np.array([[0.0, 0.0, 0.0],      # in block, early visit
                    [1.0, 1.0, 1.0],      # in block, late visit
                    [0.5, 0.5, 0.5],      # in block, late visit
                    [5.0, 5.0, 5.0],      # OUTSIDE the block, late visit
                    [0.2, 0.2, 0.2]],     # in block, early visit
                   dtype=float)
    ks = np.array([5, 208, 210, 207, 3])


def _loop():
    g = GreedyLoop.__new__(GreedyLoop)
    g.base = _Session()
    return g


def _edge(**over):
    e = {"volume_m": {"lo": [-1.0, -1.0, -1.0], "hi": [2.0, 2.0, 2.0]},
         "earlier_kfs": [0, 9], "later_kfs": [206, 215],
         "source": "revisit", "label": "revisit_region_1", "trusted": True,
         "i": 210, "j": 5, "X": np.eye(4)}
    e.update(over)
    return e


def test_a_region_yields_the_points_of_each_visit():
    a, b = GreedyLoop._region_indices(_loop(), _edge())
    assert a.tolist() == [1, 2], "copy A is the LATER visit, inside the block"
    assert b.tolist() == [0, 4], "copy B is the EARLIER visit, inside the block"


def test_a_point_outside_the_block_is_not_a_copy():
    """Index 3 is owned by the late visit but sits 5 m away: the region is a
    place, not a keyframe range."""
    a, _b = GreedyLoop._region_indices(_loop(), _edge())
    assert 3 not in a.tolist()


def test_a_region_without_a_volume_is_skipped_not_fatal():
    """Old reports have no volume_m; the pool must pass over them."""
    for missing in ({"volume_m": None}, {"earlier_kfs": None}, {"later_kfs": []}):
        assert GreedyLoop._region_indices(_loop(), _edge(**missing)) == (None, None)


def test_the_visits_never_share_a_point():
    a, b = GreedyLoop._region_indices(_loop(), _edge())
    assert not (set(a.tolist()) & set(b.tolist())), "a point belongs to ONE visit"


def test_a_revisit_candidate_names_itself_without_an_instance():
    """`<revisit_region_1#-1 ...>` would read as an instance with a broken id."""
    c = Candidate(0, _edge(), np.array([1, 2]), np.array([0, 4]), [1], [2], None)
    assert c.instance_id == -1
    assert repr(c) == "<revisit_region_1 kf 210<->5>"


def test_an_instance_candidate_still_shows_its_id():
    e = _edge(label="desk", instance_id=201)
    c = Candidate(0, e, np.array([1]), np.array([0]), [1], [2], 7)
    assert repr(c) == "<desk#201 kf 210<->5>"


def test_the_pool_builder_reaches_the_region_branch():
    """Pinned on the source: an edge with no instance must NOT be dropped."""
    src = (Path(__file__).resolve().parents[1] / "reconstruction" / "certify"
           / "iterate.py").read_text()
    body = src[src.index("def _build_pool"):src.index("def _region_indices")]
    assert "ia, ib = self._region_indices(e)" in body
    assert "if inst is None:\n                continue" not in body, \
        "an edge without an instance must no longer be skipped outright"


def test_the_volume_travels_from_the_region_to_the_edge():
    """visit_edges has to carry volume_m or the pool can never select points."""
    src = (Path(__file__).resolve().parents[1] / "reconstruction" / "certify"
           / "loops_posthoc.py").read_text()
    ve = src[src.index("def visit_edges"):]
    assert '"volume_m": rg.get("volume_m")' in ve
    assert '"label": f"revisit_region_{rg.get(\'region\')}"' in ve
