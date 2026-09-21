"""One surface against itself is not a duplicate.

Measured on the pccr relaunch of 2026-09-21, epoch 1 — the certification
wrote these as pose-graph edges:

    white_tiled_floor#45  kf 119<->25  copies 1036.0 -> 47.8 cm,  |t| 1036 cm / 0.0 deg, full
    white_tiled_floor#45  kf 119<->20  copies 1057.5 -> 31.4 cm,  |t| 1057 cm / 0.0 deg, full
    white_wall#98         kf 115<->26  copies 1299.5 ->  8.5 cm,  |t| 1299 cm / 0.0 deg, perp_axis
    exposed_ceiling_ducts#116          EIGHT edges, 3.8 to 10.7 m each

A floor does not move ten metres between two visits. Those are the two ENDS
of one floor, one wall, one duct: the spatial gate accepted them on the
separation it can OBSERVE (the offset ACROSS the shared plane, ~0 by
construction) and the measurement then took the copies by keyframe window,
got the two ends, and "closed" them by sliding one onto the other.

They are not harmless. In that same run 11 of 22 acta edges were this, and
they carried the median loop closure from 1.06 m to 2.35 m while the REAL
closures improved from a median of 80.5 cm to 47.7 cm — the acta reported a
71.7 % regression over a cloud that got better.

The fix measures the one thing that separates the two cases: do the copies
OVERLAP on the surface they share? Two copies of one desk top do, and they
observe their in-plane displacement through their own edges. Two ends of a
ceiling do not, and along a plane a plane determines nothing.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.loops import spatial_gate as sg  # noqa: E402


def _cfg():
    return {"dims_pct_lo": 5.0, "dims_pct_hi": 95.0,
            "same_surface_angle_deg": 10.0, "same_surface_axis_ratio": 0.15,
            "same_surface_planar_ratio": 0.05}


def _plane_patch(rng, n=4000, w=3.0, thickness=0.004):
    return np.c_[rng.uniform(0, w, n), rng.uniform(0, w, n),
                 rng.normal(0, thickness, n)]


def _bar(rng, n=3000, length=4.0):
    """An elongated cluster along x — a wall strip seen edge-on."""
    return np.c_[rng.uniform(0, length, n), rng.normal(0, 0.03, n),
                 rng.normal(0, 0.03, n)]


# ── the two cases the rule exists to tell apart ────────────────────────────

def test_two_ends_of_one_floor_are_two_parts():
    rng = np.random.default_rng(0)
    a = _plane_patch(rng)
    b = a + np.array([10.0, 0.0, 0.0])
    geom = sg.same_surface_rule(a, b, _cfg())
    assert geom["kind"] == "plane" and geom["same_geometry"]
    # the gate's own number: across the plane they are flush
    assert geom["distance_m"] < 0.01
    assert geom["centroid_distance_m"] > 9.0
    ov = sg.surface_overlap(a, b, geom, _cfg())
    assert ov["applies"] and ov["disjoint"]
    assert ov["overlap_frac"] == 0.0
    assert ov["gap_m"] > 1.0


def test_two_copies_of_one_desk_top_are_one_piece():
    """The case that must SURVIVE: a real duplicate, displaced by drift."""
    rng = np.random.default_rng(1)
    a = _plane_patch(rng, w=1.4)
    b = a + np.array([0.40, 0.05, 0.01])
    geom = sg.same_surface_rule(a, b, _cfg())
    ov = sg.surface_overlap(a, b, geom, _cfg())
    assert ov["applies"] and not ov["disjoint"]
    assert ov["overlap_frac"] > 0.5


def test_two_stretches_of_one_wall_are_two_parts():
    rng = np.random.default_rng(2)
    a = _bar(rng)
    b = a + np.array([13.0, 0.0, 0.0])
    geom = sg.same_surface_rule(a, b, _cfg())
    assert geom["kind"] == "axis"
    ov = sg.surface_overlap(a, b, geom, _cfg())
    assert ov["applies"] and ov["disjoint"]


def test_a_compact_pair_is_not_this_question():
    """Two blobs share no surface: the centroid distance IS the separation
    and the rule must stand down instead of answering."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.2, (800, 3))
    b = a + np.array([9.0, 0, 0])
    geom = sg.same_surface_rule(a, b, _cfg())
    assert geom["kind"] == "centroid"
    assert sg.surface_overlap(a, b, geom, _cfg())["applies"] is False


def test_a_starved_side_never_decides():
    rng = np.random.default_rng(4)
    a = _plane_patch(rng)
    geom = sg.same_surface_rule(a, a[:2], _cfg())
    assert sg.surface_overlap(a, a[:2], geom, _cfg())["applies"] is False


def test_the_numbers_are_reported_so_a_refusal_can_say_why():
    rng = np.random.default_rng(5)
    a = _plane_patch(rng)
    ov = sg.surface_overlap(a, a + np.array([10.0, 0, 0]),
                            sg.same_surface_rule(a, a + np.array([10.0, 0, 0]), _cfg()),
                            _cfg())
    assert len(ov["directions"]) == 2          # a plane is judged on both axes
    for d in ov["directions"]:
        assert {"support_a", "support_b", "overlap_m", "overlap_frac",
                "gap_m"} <= set(d)


# ── the callers ────────────────────────────────────────────────────────────

class _Session:
    """One PLANE whose two clusters sit ten metres apart — pccr's floor."""

    def __init__(self, tmp_path, n=2000, gap=10.0):
        import json
        self.output_dir = tmp_path
        rng = np.random.default_rng(7)
        a = _plane_patch(rng, n=n)
        b = a + np.array([gap, 0.0, 0.0])
        self.xyz = np.vstack([a, b])
        self.ks = np.concatenate([np.full(n, 10), np.full(n, 200)])
        self.n_points = len(self.xyz)
        self.n_kf = 216
        self.poses = np.tile(np.eye(4), (216, 1, 1))
        (tmp_path / "segmentation_result.json").write_text(json.dumps(
            {"instances": [{"instance_id": 45, "id": 45, "label": "white_tiled_floor",
                            "globalIndices": list(range(2 * n))}]}))


def _scfg():
    return types.SimpleNamespace(min_copy_points=10, icp_iters=3, icp_trim=0.8,
                                 max_copy_residual_m=0.05)


def _patch_indices(monkeypatch, lp):
    def _idx(sess, inst, i, j, window_kf):
        gi = np.arange(len(sess.xyz))
        ks = sess.ks
        return (gi[np.abs(ks - i) <= int(window_kf)],
                gi[np.abs(ks - j) <= int(window_kf)], ks)
    monkeypatch.setattr(lp, "_copy_indices", _idx)


def test_the_scale_stage_refuses_the_pair_before_paying_for_its_icp(tmp_path, monkeypatch):
    from reconstruction.certify import loops_posthoc as lp
    sess = _Session(tmp_path)
    _patch_indices(monkeypatch, lp)
    called = []
    monkeypatch.setattr(lp, "measure_copy",
                        lambda *a, **k: called.append(1) or None)
    cand = [{"instance_id": 45, "label": "white_tiled_floor", "i": 200, "j": 10,
             "verdict": "ambiguous", "kind": "instance"}]
    rows = lp.copy_scale_rows(sess, cand, _scfg(), 30, 3, log=lambda m: None,
                              spatial_cfg=_cfg())
    assert not called, "the ICP ran on two parts of one floor"
    assert len(rows) == 1 and "two parts of one plane" in rows[0]["reason"]
    assert rows[0]["surface_overlap"]["disjoint"] is True


def test_without_the_spatial_config_the_stage_behaves_as_before(tmp_path, monkeypatch):
    """The refusal is an optimisation there, never a change of verdict."""
    from reconstruction.certify import loops_posthoc as lp
    sess = _Session(tmp_path)
    _patch_indices(monkeypatch, lp)
    called = []
    monkeypatch.setattr(lp, "measure_copy",
                        lambda *a, **k: called.append(1) or None)
    cand = [{"instance_id": 45, "label": "white_tiled_floor", "i": 200, "j": 10,
             "verdict": "ambiguous", "kind": "instance"}]
    lp.copy_scale_rows(sess, cand, _scfg(), 30, 3, log=lambda m: None)
    assert called, "without spatial_cfg the old path must still measure"


def test_the_graph_writes_no_edge_for_one_surface_against_itself(tmp_path, monkeypatch):
    from reconstruction.certify import loops_posthoc as lp
    sess = _Session(tmp_path)
    _patch_indices(monkeypatch, lp)
    called = []
    monkeypatch.setattr(lp, "measure_copy",
                        lambda *a, **k: called.append(1) or None)
    cfg = types.SimpleNamespace(
        loops=types.SimpleNamespace(
            spatial=_cfg(), duplicate_min_sep_m=0.20,
            semantic=types.SimpleNamespace(default_class="structural",
                                           nonstructural_sigma_factor=2.0)),
        certify=types.SimpleNamespace(
            scale=_scfg(),
            visit_loops=types.SimpleNamespace(window_kf=30, sigma_floor_m=0.01,
                                              unobserved_sigma_deg=30.0,
                                              unobserved_sigma_m=5.0)),
        graph=types.SimpleNamespace(loop_sigma_rot_deg=5.0),
        loop=types.SimpleNamespace(ambiguous_sigma_factor=3.0))
    cand = [{"instance_id": 45, "label": "white_tiled_floor", "i": 200, "j": 10,
             "verdict": "ambiguous", "kind": "instance", "class": "structural"}]
    out = lp.instance_edges(sess, cand, ccfg=None, cfg=cfg, log=lambda m: None)
    assert not called, "the rigid fit ran on two ends of one floor"
    assert len(out) == 1 and out[0]["accepted"] is False
    assert "two parts of one plane" in out[0]["reason"]
    # nothing is dropped silently: the numbers travel with the refusal
    assert out[0]["surface_overlap"]["gap_m"] > 1.0


# ── the shape belongs to the OBJECT, not to the window it was sampled through ─

def test_a_stub_of_one_duct_has_no_direction_of_its_own():
    """Why `surface_of_instance` exists. pccr 2026-09-21, epoch 1: asked of the
    two COPIES, `exposed_ceiling_ducts#116` answered "axis" on kf 169<->115 and
    was refused, and answered neither plane nor axis on kf 203<->92, where the
    window held a stub with no direction — that pair went on to close 10.30 m
    at 0.0 deg. Seven of the acta's thirteen voting pairs were one duct and one
    floor against themselves."""
    rng = np.random.default_rng(11)
    # a duct with a real cross-section: 12 m long, ~25 cm across
    n = 12000
    bar = np.c_[rng.uniform(0, 12.0, n), rng.normal(0, 0.12, n),
                rng.normal(0, 0.12, n)]
    # what a 15-keyframe window sees of it: a stub no longer than it is thick,
    # which has no direction of its own
    stub_a = bar[bar[:, 0] < 0.30]
    stub_b = bar[bar[:, 0] > 11.70]
    pair = sg.same_surface_rule(stub_a, stub_b, _cfg())
    assert not pair["same_geometry"], "the stubs already agreed — pick a harder pair"
    whole = sg.surface_of_instance(bar, _cfg())
    assert whole["kind"] == "axis" and whole["same_geometry"]
    ov = sg.surface_overlap(stub_a, stub_b, whole, _cfg())
    assert ov["applies"] and ov["disjoint"]
    assert ov["gap_m"] > 5.0
    # and the pair-level rule, asked alone, would have let it through
    assert sg.surface_overlap(stub_a, stub_b, pair, _cfg())["applies"] is False


def test_the_object_rule_measures_along_the_objects_own_axes():
    rng = np.random.default_rng(12)
    bar = _bar(rng, length=6.0)
    whole = sg.surface_of_instance(bar, _cfg())
    assert whole["source"] == "instance" and whole.get("axes") is not None
    # the basis travels: surface_overlap must use it instead of re-deriving one
    ov = sg.surface_overlap(bar[bar[:, 0] < 1.0], bar[bar[:, 0] > 5.0], whole, _cfg())
    assert len(ov["directions"]) == 1          # an axis is judged on one


def test_a_compact_object_is_never_refused_by_the_object_rule():
    """The guard that keeps the REAL duplicates. `computer_monitor#178` closed
    a genuine 59 cm duplicate whose copies sit FARTHER apart than the monitor
    is wide — a blanket disjointness test would have thrown it away."""
    rng = np.random.default_rng(13)
    blob = rng.normal(0, 0.15, (2000, 3))
    whole = sg.surface_of_instance(blob, _cfg())
    assert whole["kind"] == "centroid" and not whole["same_geometry"]
    a = blob[:1000]
    b = blob[1000:] + np.array([0.59, 0.0, 0.0])
    assert sg.surface_overlap(a, b, whole, _cfg())["applies"] is False


def test_the_instance_rule_needs_points_like_every_other_one():
    assert sg.surface_of_instance(np.zeros((2, 3)), _cfg())["same_geometry"] is False


def test_the_graph_falls_back_to_the_object_when_the_copies_cannot_say():
    """`instance_edges` asks the copies first — when they AGREE on a surface
    that is the strongest statement available — and the object only when they
    do not."""
    src = (Path(__file__).resolve().parents[1]
           / "reconstruction" / "certify" / "loops_posthoc.py").read_text()
    body = src[src.index("def instance_edges("):src.index("def _info_from_projection(")]
    assert "same_surface_rule(" in body and "surface_of_instance(" in body
    assert body.index("same_surface_rule(") < body.index("surface_of_instance("), \
        "the pair must speak before the object"
