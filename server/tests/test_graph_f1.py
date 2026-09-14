"""claude_stac.txt F1 on the server side: §12.5 (instance detector: two
clusters → candidate; after the graph, zero duplicates), §12.5b (spatial
gate: fused identity → split, drift within budget → loop, out of frustum →
rejected, re-evaluation with a measured budget), §12.10 (config + zero
decision literals)."""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.loops import spatial_gate as sg                    # noqa: E402
from reconstruction.loops.config import (LoopsConfigError, load_loops_config,  # noqa: E402
                                         fork_model_loops, fork_model_scale)
from tests.synth_metric import (make_session, write_session_dir, raw_server_cfg,  # noqa: E402
                                yaw_R)

PKG = Path(__file__).resolve().parents[1] / "reconstruction" / "loops"


@pytest.fixture(scope="module")
def sess():
    return make_session(n_kf=150, H=40, W=56)


class _View:
    """Minimal TrajectoryView over a synthetic session (GT poses)."""

    def __init__(self, s, drift=None):
        self.s = s
        self.n_frames = s.n_kf
        self.hw = (s.H, s.W)
        self._poses = s.poses if drift is None else np.stack([drift[g] @ s.poses[g]
                                                              for g in range(s.n_kf)])

    def pose(self, g):
        return self._poses[int(g)]

    def K(self, g):
        return self.s.K

    def centres(self):
        return self._poses[:, :3, 3]

    def depth(self, g):
        return self.s.depth[int(g)]

    def points(self, g, n, seed=0):
        d = self.s.depth[int(g)]
        P = self.s.points[int(g)][d > 0]
        if len(P) > n:
            P = P[np.random.default_rng(seed).choice(len(P), int(n), replace=False)]
        return P


def _cfg():
    return load_loops_config(raw_server_cfg())


# ── §12.5b spatial gate ─────────────────────────────────────────────────────

def test_drift_budget_grows_with_walk():
    c = _cfg().loops.spatial
    b0 = sg.drift_budget(0.0, c)
    b1 = sg.drift_budget(100.0, c)
    assert b0["delta_m"] == c.drift_floor_m and b1["delta_m"] == pytest.approx(1.3)
    ov = sg.drift_budget(100.0, c, {"delta_m": 0.05, "theta_deg": 0.2, "source": "chain"})
    assert ov["delta_m"] == 0.05 and ov["source"] == "chain"


def test_salad_pair_revisit_accepted_far_pair_rejected(sess):
    c = _cfg().loops.spatial
    view = _View(sess)
    ok = sg.gate_frame_pair(140, 8, view, c)          # the walk returns past the start
    assert ok["verdict"] in ("accept", "ambiguous"), ok
    bad = sg.gate_frame_pair(75, 8, view, c)          # opposite sides of the block
    assert bad["verdict"] == "reject"
    assert bad["rules"]["frustum"]["passed"] is False


def test_fused_identity_splits_drifted_copy_loops(sess):
    c = _cfg().loops.spatial
    view = _View(sess)
    # column near the start (oid 11 at x=-8, z=7.2 side? pick by scene labels)
    cols = [p for p in sess.scene.prims if getattr(p, "label", "") == "column"]
    col_a, col_b = cols[5], cols[4]        # (0,-7.2): seen at the start and at the lap's end
    pts_a = np.array([col_a.c + [col_a.r, h, 0.0] for h in np.linspace(0.1, 2.9, 200)])
    pts_far = np.array([col_b.c + [col_b.r, h, 0.0] for h in np.linspace(0.1, 2.9, 200)])
    # find keyframes that see each column
    def kf_seeing(oid):
        return [g for g in range(sess.n_kf) if (sess.oid[g] == oid).sum() > 30]
    ka, kb = kf_seeing(col_a.oid), kf_seeing(col_b.oid)
    assert ka and kb
    # (a) fused: two different columns 16 m apart under one id → split
    g = sg.gate_instance_pair(kb[0], ka[0], view, c, pts_far, pts_a, "column")
    assert g["verdict"] == "split" and g["rules"]["separation"]["verdict"] == "split"
    # (b) the same column seen again at the end of the lap, displaced by drift
    # within budget → loop
    later = [g_ for g_ in ka if g_ > 120]
    assert later, "the lap must revisit the column"
    pts_drift = pts_a + np.array([0.25, 0.0, 0.10])
    g2 = sg.gate_instance_pair(later[0], ka[0], view, c, pts_drift, pts_a, "column")
    assert g2["verdict"] in ("loop", "ambiguous"), g2
    assert g2["rules"]["separation"]["verdict"] == "loop"
    # (c) an object the other visit could never see (behind the camera, out of
    # every frustum) → rejected
    pts_hidden = pts_a + np.array([0.0, 0.0, -30.0])
    g3 = sg.gate_instance_pair(later[0], ka[0], view, c, pts_hidden, pts_a, "column")
    assert g3["verdict"] in ("reject", "split")
    # (d) size rule (compact objects): a box-like object and a copy twice its
    # size 0.4 m away are two objects — partial views of a plane/axis are
    # exempt (their extents are not observable), recorded as skipped
    rng = np.random.default_rng(1)
    blob = col_a.c + np.array([1.0, 0.5, 0.0]) + rng.uniform(-0.5, 0.5, (400, 3)) * [1.0, 1.0, 1.0]
    blob_big = col_a.c + np.array([1.4, 1.0, 0.0]) + rng.uniform(-1.0, 1.0, (400, 3)) * [1.0, 1.0, 1.0]
    g4 = sg.gate_instance_pair(later[0], ka[0], view, c, blob_big, blob, "cabinet")
    assert g4["rules"]["geometry"]["kind"] == "centroid"
    assert g4["verdict"] == "split" and g4["rules"]["size"]["passed"] is False
    g5 = sg.gate_instance_pair(later[0], ka[0], view, c, pts_drift, pts_a, "column")
    assert g5["rules"]["size"].get("skipped")


def test_reevaluation_with_measured_budget_flips_intermediate(sess):
    c = _cfg().loops.spatial
    view = _View(sess)
    col = [p for p in sess.scene.prims if getattr(p, "label", "") == "column"][5]
    pts = np.array([col.c + [col.r, h, 0.0] for h in np.linspace(0.1, 2.9, 200)])
    ks = [g for g in range(sess.n_kf) if (sess.oid[g] == col.oid).sum() > 30]
    later = [g_ for g_ in ks if g_ > 120][0]
    # separation between δ and 3δ → ambiguous under the default budget
    L = sg.walked_length_m(view.centres(), later, ks[0])
    delta = sg.drift_budget(L, c)["delta_m"]
    pts_mid = pts + np.array([1.8 * delta, 0.0, 0.0])
    amb = sg.gate_instance_pair(later, ks[0], view, c, pts_mid, pts, "column")
    assert amb["verdict"] == "ambiguous"
    # after the pose graph: the measured chain uncertainty is LARGER here → loop
    loop = sg.gate_instance_pair(later, ks[0], view, c, pts_mid, pts, "column",
                                 budget_override={"delta_m": 2.0 * delta, "theta_deg": 3.0})
    assert loop["verdict"] == "loop"
    # ...or SMALLER → the fusion is confirmed (split)
    split = sg.gate_instance_pair(later, ks[0], view, c, pts_mid, pts, "column",
                                  budget_override={"delta_m": 0.3 * delta, "theta_deg": 1.0})
    assert split["verdict"] == "split"


def test_corridor_detection(sess):
    c = _cfg().loops.spatial
    view = _View(sess)
    cor = sg.corridor_between(2, 20, view, c)         # a straight leg of the walk
    assert cor["corridor"] is True
    cor2 = sg.corridor_between(2, 100, view, c)       # around the block
    assert cor2["corridor"] is False


# ── §12.5 instance detector on a session on disk ────────────────────────────

def _drift_field(n, total_m=0.35):
    """Accumulated rigid drift along the walk (translation growing linearly)."""
    D = np.tile(np.eye(4), (n, 1, 1))
    for g in range(n):
        D[g, :3, 3] = np.array([total_m, 0.0, 0.4 * total_m]) * g / max(n - 1, 1)
    return D


def test_instance_detector_two_clusters_then_zero_after_correction(tmp_path, sess):
    cols = [p for p in sess.scene.prims if getattr(p, "label", "") == "column"]
    walls = [p for p in sess.scene.prims if getattr(p, "label", "") == "wall"]
    # instance 1: the first column (seen at the start AND at the end of the lap)
    # instance 2: a wall; instance 3: FUSED identity = two distant columns
    instances = {1: {"label": "column", "oids": [cols[5].oid]},
                 2: {"label": "wall", "oids": [walls[0].oid]},
                 3: {"label": "column", "oids": [cols[2].oid, cols[7].oid]}}
    # 0.9 m of accumulated drift: larger than the column's own diameter, so
    # the two copies are DISJOINT clusters (a smaller drift makes overlapping
    # copies — the onion — that only the temporal detector sees)
    root = write_session_dir(tmp_path / "s1", sess, instances, point_stride=2,
                             drift_by_kf=_drift_field(sess.n_kf, total_m=0.9))
    from reconstruction.loops.instance_loops import detect_instance_loops
    cfg = load_loops_config(raw_server_cfg(**{"loops.cluster_min_points": 150,
                                              "loops.dbscan_min_samples": 8}))
    rep = detect_instance_loops(root / "output", root, cfg=cfg, log=lambda m: None)
    by_iid = {}
    for c in rep["candidates"]:
        by_iid.setdefault(c["instance_id"], []).append(c)
    # instance 1: revisit with drift within budget → loop candidate written
    assert 1 in by_iid, rep["candidates"]
    assert any(c["verdict"] in ("loop", "ambiguous") for c in by_iid[1])
    assert rep["n_written"] >= 1
    # instance 3: two columns 8+ m apart → split, no loop
    assert 3 in by_iid and any(c["verdict"] == "split" for c in by_iid[3])
    assert rep["splits"] and {s["source_instance"] for s in rep["splits"]} == {3}
    # the wall (one plane, sampled with gaps, drifted) is never split — its
    # observable separation is the plane offset, not the centroid distance
    assert all(c["verdict"] != "split" for c in by_iid.get(2, []))
    res = json.loads((root / "output" / "segmentation_result.json").read_text())
    ids = sorted(int(i["instance_id"]) for i in res["instances"])
    assert ids == [1, 2, 3, 4]
    led = (root / "output" / "segmentation_ledger.jsonl").read_text().strip().splitlines()
    assert json.loads(led[-1])["type"] == "instance_split"
    # loop_closures.txt carries the instance candidate with its source
    txt = (root / "output" / "maplong_run" / "loop_closures.txt").read_text()
    assert "instance" in txt
    # a non-structural class never drops the candidate (USER 2026-09-13): the
    # same session with the default class 'movable' writes the same pairs,
    # tagged with the class so the verifier inflates σ instead
    root_m = write_session_dir(tmp_path / "s1m", sess, instances, point_stride=2,
                               drift_by_kf=_drift_field(sess.n_kf, total_m=0.9))
    cfg_m = load_loops_config(raw_server_cfg(**{"loops.cluster_min_points": 150,
                                                "loops.dbscan_min_samples": 8,
                                                "loops.semantic.default_class": "movable"}))
    rep_m = detect_instance_loops(root_m / "output", root_m, cfg=cfg_m, log=lambda m: None)
    assert rep_m["n_written"] == rep["n_written"] >= 1
    txt_m = (root_m / "output" / "maplong_run" / "loop_closures.txt").read_text()
    assert "instance:movable" in txt_m
    # duplicates metric present
    dup = json.loads((root / "output" / "duplicates.json").read_text())
    assert dup["n_duplicates"] >= 1
    # after the (simulated) graph — the drift removed — the same detector
    # finds NO duplicate of instance 1
    root2 = write_session_dir(tmp_path / "s2", sess, {1: instances[1], 2: instances[2]},
                              point_stride=2, drift_by_kf=None)
    rep2 = detect_instance_loops(root2 / "output", root2, cfg=cfg, log=lambda m: None)
    dups1 = [d for d in rep2["duplicates"] if d["instance_id"] == 1]
    assert dups1 == []


# ── §12.10 config ───────────────────────────────────────────────────────────

def test_missing_key_names_the_key():
    with pytest.raises(LoopsConfigError, match="loops.spatial.drift_floor_m"):
        load_loops_config(raw_server_cfg(**{"loops.spatial.drift_floor_m": None}))
    with pytest.raises(LoopsConfigError, match="scale.sigma_loop"):
        load_loops_config(raw_server_cfg(**{"scale.sigma_loop": None}))
    with pytest.raises(LoopsConfigError, match="correction_graph.loop.max_edge_sigma_m"):
        load_loops_config(raw_server_cfg(**{"correction_graph.loop.max_edge_sigma_m": -1}))
    with pytest.raises(LoopsConfigError, match="starved_sigma_m"):
        load_loops_config(raw_server_cfg(**{"correction_graph.loop.starved_sigma_m": 0.01}))


def test_production_yaml_loads_and_flattens():
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    cfg = load_loops_config(raw)
    d = fork_model_loops(cfg, "/srv")
    for k in ("anchors_per_bridge", "max_edge_sigma_m", "scale_tol_log", "spatial",
              "bridge_extra_frames", "stac_server_dir", "intra_chunk_loops"):
        assert k in d
    assert fork_model_scale(cfg)["verify_max_dev"] == cfg.scale.verify_max_dev


# Mathematics and machine limits, not decisions: 1.4826 = 1/Φ⁻¹(0.75), the
# factor that makes the MAD a consistent estimator of σ for a normal
# distribution (kf_graph's robust drift consensus), and 9e18 ≈ 2⁶³, the int64
# overflow guard of the voxel packing in instance_loops.
_FLOAT_WHITELIST = {0.0, 1.0, -1.0, 2.0, 0.5, 1e-9, 1e-6, 1e-12, 1e-18, 100.0, 1000.0, 255.0,
                    1.4826, 9e18}


def test_no_decision_literals_outside_config():
    offenders = []
    for py in sorted(PKG.glob("*.py")):
        if py.name == "config.py":
            continue
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if node.value not in _FLOAT_WHITELIST:
                    offenders.append(f"{py.name}:{node.lineno} = {node.value}")
    assert not offenders, offenders
