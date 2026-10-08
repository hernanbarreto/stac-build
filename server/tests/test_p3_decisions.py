"""The projection's knife-edge decisions judged by THE USER'S RULE (docs/plan_determinismo.md
2026-10-08): 93 (mask identity), 104 (space dedupe / fragment contiguity), 105 (co-visible
split), 109 / 143 (OBB yaw), 145 (OBB extent), 116 (the 1000-point bar's margin). Every
test drives the function the pipeline calls."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import pipeline as P                                # noqa: E402


def _params():
    from config import cfg
    return P._decision_params(cfg)


def test_the_rule_reads_the_users_factor_and_five_judges():
    fac, conf, min_j = _params()
    assert fac == 1.1 and conf == 0.95 and min_j == 5


# ── 93: two SAM3 ids are one observation only by the majority of their shared frames ──

def _m(px):
    a = np.zeros((6, 6), np.uint8)
    a.flat[px] = 1
    return a


def test_one_frame_of_perfect_overlap_no_longer_collapses_two_ids():
    """The vendor's rule unioned two ids for the whole batch on ONE frame at IoU 1.0."""
    masks = {0: {1: _m([0, 1, 2, 3]), 2: _m([0, 1, 2, 3])}}          # identical in one frame
    for f in range(1, 8):
        masks[f] = {1: _m([0, 1, 2, 3]), 2: _m([20, 21, 22, 23])}    # disjoint in seven
    rec = []
    out, n = P._dedupe_masks_per_frame(masks, 0.9, record=rec)
    assert n == 0 and rec and rec[0]["collapsed"] is False
    assert rec[0]["n_judges"] == 8 and rec[0]["n_frames_above_bar"] == 1


def test_ids_that_coincide_in_the_majority_of_their_frames_collapse():
    masks = {f: {1: _m([0, 1, 2, 3]), 2: _m([0, 1, 2, 3])} for f in range(8)}
    rec = []
    out, n = P._dedupe_masks_per_frame(masks, 0.9, record=rec)
    assert n == 1 and rec[0]["collapsed"] is True and rec[0]["n_judges"] == 8
    assert all(set(fm) == {1} for fm in out.values()), "the lowest id survives"


def test_fewer_than_five_judge_frames_never_collapse():
    masks = {f: {1: _m([0, 1, 2, 3]), 2: _m([0, 1, 2, 3])} for f in range(4)}
    rec = []
    out, n = P._dedupe_masks_per_frame(masks, 0.9, record=rec)
    assert n == 0 and rec[0]["judges_margin"] == -1 and "judges" in rec[0]["reason"]


def test_a_batch_link_needs_the_same_majority():
    prev = {f: {5: _m([0, 1, 2, 3])} for f in range(10)}
    curr = {f: {1: _m([0, 1, 2, 3]), 2: _m([30, 31])} for f in range(10)}
    rec = []
    remap, nxt = P._match_ids_iou(prev, curr, 0, 9, 0.3, next_global_id=6, record=rec)
    assert remap[1] == 5 and remap[2] == 6 and nxt == 7
    assert [r["linked"] for r in rec] == [True]
    # the same link over THREE overlap frames: not enough judges → a fresh id
    rec = []
    remap, _ = P._match_ids_iou({f: prev[f] for f in range(3)}, {f: curr[f] for f in range(3)},
                                0, 2, 0.3, next_global_id=6, record=rec)
    assert remap[1] == 6 and rec[0]["linked"] is False


# ── 104: fragment contiguity needs five adjacent voxel pairs ──────────────────────────

def _frag_instances(touching_pairs):
    """Two same-label slabs; `touching_pairs` distinct 10 cm voxel adjacencies between them."""
    rng = np.random.default_rng(0)
    a = np.c_[rng.uniform(0.0, 0.95, 4000), rng.uniform(0.0, 0.05, 4000), rng.uniform(0.0, 0.95, 4000)]
    b = np.c_[rng.uniform(2.0, 2.95, 4000), rng.uniform(0.0, 0.05, 4000), rng.uniform(0.0, 0.95, 4000)]
    bridge = np.array([[1.05, 0.02, 0.05 + 0.1 * k] for k in range(touching_pairs)])   # cells x=10
    a_cells = np.array([[0.95, 0.02, 0.05 + 0.1 * k] for k in range(touching_pairs)])  # cells x=9
    xyz = np.vstack([a, a_cells, b, bridge])
    na = len(a) + len(a_cells)
    inst = [{"id": 0, "instance_id": 1, "label": "floor", "globalIndices": list(range(na))},
            {"id": 1, "instance_id": 2, "label": "floor", "globalIndices": list(range(na, len(xyz)))}]
    return inst, xyz


def test_one_adjacent_voxel_pair_does_not_weld_two_fragments():
    inst, xyz = _frag_instances(1)
    dec = []
    n = P._merge_label_fragments(inst, xyz, gap_m=0.10, record={}, decisions=dec)
    assert n == 0 and len(inst) == 2
    assert dec[0]["adjacent_voxel_pairs"] >= 1 and dec[0]["merged"] is False and dec[0]["margin"] < 0


def test_five_adjacent_voxel_pairs_do():
    inst, xyz = _frag_instances(6)
    dec = []
    n = P._merge_label_fragments(inst, xyz, gap_m=0.10, record={}, decisions=dec)
    assert n == 1 and len(inst) == 1 and dec[0]["merged"] is True


# ── 105: the split's judges ────────────────────────────────────────────────────────────

def test_covisibility_needs_five_judge_keyframes_above_the_bar():
    fac, conf, mj = _params()
    d = P._covisible({0, 1, 2}, {0, 1, 2, 7}, 0.5, factor=fac, confidence=conf, min_judges=mj)
    assert d["improves"] is False and "judges" in d["failed"]
    d = P._covisible(set(range(8)), set(range(8)) | {20}, 0.5, factor=fac, confidence=conf, min_judges=mj)
    assert d["improves"] is True and d["n_judges"] == 8
    d = P._covisible(set(range(8)), set(range(4)) | {20, 21, 22, 23}, 0.5, factor=fac,
                     confidence=conf, min_judges=mj)
    assert d["improves"] is False, "half the keyframes: not above the bar significantly"


def test_split_children_are_numbered_from_the_raw_store_and_in_spatial_order():
    rng = np.random.default_rng(0)
    desks = [rng.uniform(-0.5, 0.5, (6000, 3)) * [1.3, 0.05, 0.7] + [x, 0.75, 0.0] for x in (4.0, 0.0, 2.0)]
    floor = rng.uniform(-0.5, 0.5, (60000, 3)) * [8.0, 0.02, 6.0] + [2.0, 0.0, -1.0]
    xyz = np.vstack(desks + [floor])
    fr = np.concatenate([rng.choice(10, len(d)) for d in desks] + [np.full(len(floor), 999)]).astype(np.int32)
    n_inst = sum(len(d) for d in desks)
    cams = {f: np.asarray((x, 1.5, 3.0), float) for f, x in zip(range(10), np.linspace(-1, 5, 10))}
    rec = []
    inst = [{"id": 9, "instance_id": 10, "label": "desk", "globalIndices": list(range(n_inst))}]
    added = P._split_covisible_components(inst, xyz, fr, gap_m=0.10, min_points=1000, covis_share=0.5,
                                          cam_centre=cams, min_walk_m=1.0, id_base=40, record=rec)
    assert added == 2 and len(inst) == 3
    assert [i["id"] for i in inst] == [9, 41, 42], "children from id_base + 1, parent keeps its id"
    centres = [np.mean(xyz[i["globalIndices"]], 0)[0] for i in inst[1:]]
    assert centres == sorted(centres), "children ordered by their stable spatial key"
    assert rec and all(r["kind"] == "covisible_split" for r in rec)
    assert all(r["free_space"]["n_rays"] >= 5 for r in rec if r["distinct"])


# ── 109 / 143 / 145: the OBB ──────────────────────────────────────────────────────────

def _table(yaw_deg, rng, n_front=2500, n_side=1200):
    top = np.stack([rng.uniform(-0.8, 0.8, 6000), np.full(6000, 0.75), rng.uniform(-0.4, 0.4, 6000)], 1)
    front = np.stack([rng.uniform(-0.8, 0.8, n_front), rng.uniform(0, 0.75, n_front), np.full(n_front, 0.4)], 1)
    left = np.stack([np.full(n_side, -0.8), rng.uniform(0, 0.75, n_side), rng.uniform(-0.4, 0.4, n_side)], 1)
    right = left * [-1, 1, 1]
    Pz = np.concatenate([top, front, left, right]) + rng.normal(0, 0.004, (6000 + n_front + 2 * n_side, 3))
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    return Pz @ R.T + [3.0, 0.0, -2.0]


def _yaw(obb):
    R = np.asarray(obb["rotation"])
    return float(np.degrees(np.arctan2(R[2, 0], R[0, 0])))


def test_the_yaw_is_canonical_and_perpendicular_faces_give_one_box():
    rng = np.random.default_rng(3)
    P1 = _table(30.0, rng)
    a = P._compute_obb(P1)
    b = P._compute_obb(P1 @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T)   # the same table turned 90°
    assert 0.0 <= _yaw(a) < 90.0 and 0.0 <= _yaw(b) < 90.0
    assert abs(_yaw(a) - _yaw(b)) < 1.0, "one representative of the 90°-periodic yaw"
    assert a["decisions"]["yaw"]["chosen"]["n_tied"] >= 1
    assert a["decisions"]["yaw"]["candidates"][0]["tied_with_minimum"] is True


def test_among_tied_footprints_the_dominant_plane_wins_and_the_margin_is_recorded():
    rng = np.random.default_rng(4)
    obb = P._compute_obb(_table(17.0, rng, n_front=2500, n_side=1200))
    ch = obb["decisions"]["yaw"]["chosen"]
    tied = [c for c in obb["decisions"]["yaw"]["candidates"] if c["tied_with_minimum"]]
    assert ch["support"] == max(c["support"] for c in tied)
    assert all("margin_m2" in c and "footprint_error_m2" in c for c in obb["decisions"]["yaw"]["candidates"])


def test_the_sample_is_stable_under_one_extra_point():
    rng = np.random.default_rng(5)
    Pz = _table(30.0, rng)
    s1 = P._stable_sample(Pz, 2000)
    s2 = P._stable_sample(np.vstack([Pz, [[9.0, 9.0, 9.0]]]), 2000)
    assert np.array_equal(Pz[s1], np.vstack([Pz, [[9.0, 9.0, 9.0]]])[s2]), \
        "one point more never redraws the sample (point 109)"


def test_the_extent_cut_is_judged_with_its_sampling_error():
    fac = _params()[0]
    assert P._share_verdict(1, 1000, 0.02, fac)[0] == "below"          # a lone flyer leaves
    assert P._share_verdict(19, 1000, 0.02, fac)[0] == "tie"            # 1.9 %: within its error, stays
    assert P._share_verdict(21, 1000, 0.02, fac)[0] == "tie"
    assert P._share_verdict(60, 1000, 0.02, fac)[0] == "above"
    rng = np.random.default_rng(6)
    body = rng.uniform(0, 0.3, (3000, 3))                           # dense: one 5 cm-voxel component
    flyer = np.array([[5.0, 5.0, 5.0]])
    core, dropped, rec = P._obb_core_points(np.vstack([body, flyer]),
                                            {"enabled": True, "voxel_m": 0.05, "min_component_frac": 0.02,
                                             "min_points": 200, "min_keep_frac": 0.5}, fac)
    assert dropped == 1 and len(core) == 3000 and rec["used"] == "core"
    assert [c["verdict"] for c in rec["components"]] == ["above", "below"]


def test_a_leftover_min_plane_frac_fails_the_load():
    with pytest.raises(KeyError, match="min_plane_frac was removed"):
        P._vertical_plane_yaws(np.zeros((20, 3)), {"ransac_iters": 1, "dist_m": 0.02, "vertical_tol_deg": 10,
                                                   "max_planes": 1, "sample": 10, "seed": 0,
                                                   "min_plane_frac": 0.1})
