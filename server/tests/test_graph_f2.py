"""claude_stac.txt F2 on the server: §12.7b (structural constraints on the
out-and-back corridor: one end loop alone leaves a lateral bend; floor datum
+ wall planarity bring it under tolerance; a real step is not flattened),
loop coverage (§4.6a), regulated dimensions → absolute rows (§4.6/§5.2),
config validation of the F2 sections, zero decision literals."""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.loops import structural as st                     # noqa: E402
from reconstruction.loops.config import LoopsConfigError, load_loops_config  # noqa: E402
from tests.synth_metric import (make_session, write_session_dir, raw_server_cfg,  # noqa: E402
                                corridor_scene, out_and_back_trajectory, yaw_R,
                                chain_drift)

PKG = Path(__file__).resolve().parents[1] / "reconstruction" / "loops"
N_KF = 120


def _bend_field(poses, yaw_deg_per_kf=0.04, yaw_kfs=(30, 60), lateral_per_kf=0.002,
                vertical_per_kf=0.0008):
    """The out-and-back failure mode as ODOMETRIC drift: a yaw bias confined to
    the second half of the outbound leg (kf 30–60) plus a small lateral +
    vertical bias every step. A single closure at the end sees the total but
    not WHERE the bend happened and spreads it along the whole walk; the
    walls and the floor datum localise it."""
    n = len(poses)
    steps = np.zeros((n, 6))
    for g in range(n):
        steps[g, 3:] = [0.0, vertical_per_kf, lateral_per_kf]
        if yaw_kfs[0] <= g < yaw_kfs[1]:
            steps[g, 1] = np.radians(yaw_deg_per_kf)
    return chain_drift(poses, steps)


def _instances(sess):
    prims = sess.scene.prims
    walls = [p for p in prims if getattr(p, "label", "") == "wall"]
    cols = [p for p in prims if getattr(p, "label", "") == "column"]
    beams = [p for p in prims if getattr(p, "label", "") == "beam"]
    inst = {}
    iid = 1
    for wl in walls[:2]:                       # the two LONG walls
        inst[iid] = {"label": "wall", "oids": [wl.oid]}; iid += 1
    for c in cols:
        inst[iid] = {"label": "column", "oids": [c.oid]}; iid += 1
    for b in beams:
        inst[iid] = {"label": "beam", "oids": [b.oid]}; iid += 1
    return inst


def _end_loop_edge(sess):
    """What an exact bridge measures between the first and last keyframes."""
    i, j = sess.n_kf - 1, 0
    Z = np.linalg.inv(sess.poses[i]) @ sess.poses[j]
    return {"i": i, "j": j, "Z": Z, "sigma_m": 0.02, "sigma_deg": 0.5, "bridge": 0}


def _run(root, cfg, sess, structural, extra):
    from reconstruction.loops.kf_graph import run_keyframe_graph
    return run_keyframe_graph(root / "output", root, cfg=cfg, apply=False,
                              extra_loop_edges=extra, use_structural=structural,
                              log=lambda m: None)


def _pose_errors(sess, rep, D):
    """Per-keyframe position error vs ground truth after the graph, in the
    truth frame (node 0 is the gauge)."""
    from reconstruction.loops.kf_graph import _vendor_on_path
    _vendor_on_path()
    from loop_utils.lie import se3_exp
    xi = rep["xi"]
    errs = []
    for g in range(sess.n_kf):
        X = se3_exp(np.asarray(xi[g]))
        T = X @ (D[g] @ sess.poses[g])
        errs.append(T[:3, 3] - sess.poses[g][:3, 3])
    return np.asarray(errs)


def _bend_errors(sess, rep, D):
    """The BEND: per-keyframe position error after the best rigid alignment
    of the solved trajectory onto the truth. A constant lateral/vertical
    odometric bias is, to first order, a rotation of the whole walk about
    its start — straight walls and a flat floor are rotation-invariant, so no
    structural constraint can (or should) see it; what they remove is the
    internal deformation (the kink), which no rigid motion absorbs."""
    e = _pose_errors(sess, rep, D)
    P_true = sess.poses[:, :3, 3]
    P_sol = P_true + e
    c_t, c_s = P_true.mean(0), P_sol.mean(0)
    H = (P_sol - c_s).T @ (P_true - c_t)
    U, _, Vt = np.linalg.svd(H)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return (P_sol - c_s) @ R.T + c_t - P_true


@pytest.fixture(scope="module")
def corridor():
    sess = make_session(H=40, W=56, scene=corridor_scene(),
                        poses=out_and_back_trajectory(N_KF))
    return sess


def _cfg(**over):
    base = {"loops.cluster_min_points": 150,
            "structural.wall_planarity.min_points_per_kf": 60,   # synthetic 40x56 px keyframes
            "structural.column_vertical.min_points_per_kf": 30,
            "structural.repeated_parallel.min_points_per_kf": 30}
    base.update(over)
    return load_loops_config(raw_server_cfg(**base))


def test_end_loop_alone_leaves_a_lateral_bend_structural_removes_it(tmp_path, corridor):
    sess = corridor
    D = _bend_field(sess.poses)
    root = write_session_dir(tmp_path / "s", sess, _instances(sess), point_stride=2, drift_by_kf=D)
    cfg = _cfg()
    edge = _end_loop_edge(sess)
    rep_loop = _run(root, cfg, sess, structural=False, extra=[edge])
    assert rep_loop["verdict"] == "APPLY", rep_loop["gates"]
    b_loop = _bend_errors(sess, rep_loop, D)
    lateral_loop = float(np.abs(b_loop[:, 2]).max())
    rep_st = _run(root, cfg, sess, structural=True, extra=[edge])
    assert rep_st["verdict"] == "APPLY", rep_st["gates"]
    assert any(w["used"] for w in rep_st["structural"]["wall_planarity"])
    b_st = _bend_errors(sess, rep_st, D)
    lateral_st = float(np.abs(b_st[:, 2]).max())
    wall_tol = cfg.structural.wall_planarity.wall_tol_m
    # the end closure alone leaves the middle of the walk bent by more than the
    # wall tolerance; the walls bring the lateral bend under it
    assert lateral_loop > wall_tol, lateral_loop
    assert lateral_st < 0.5 * lateral_loop, (lateral_loop, lateral_st)
    assert lateral_st < wall_tol, lateral_st
    # the solved walls are one plane each, parallel to the corridor within the
    # angle the patches observe, at the true half-width (the corridor's width
    # is what the walls measure absolutely)
    planes = rep_st["planes_solved"]
    walls = [v for k, v in planes.items() if k.startswith("wall")]
    assert len(walls) == 2
    width = abs(abs(walls[0]["offset_m"]) + abs(walls[1]["offset_m"]))
    assert abs(width - 4.0) < 2 * wall_tol, width
    assert rep_st["loop_coverage"]["n_anchors"] == 2


def test_loop_coverage_metric():
    C = np.stack([np.linspace(0, 100, 101), np.zeros(101), np.zeros(101)], 1)
    cov = st.loop_coverage(C, [(0, 100)], radius_m=10.0)
    assert 0.20 <= cov["coverage"] <= 0.24 and cov["uncovered"]
    cov2 = st.loop_coverage(C, [(0, 100), (25, 75), (50, 50)], radius_m=10.0)
    assert cov2["coverage"] > 0.5


def test_regulated_dimension_rows(tmp_path, corridor):
    sess = corridor
    root = write_session_dir(tmp_path / "s", sess, _instances(sess), point_stride=2)
    from correction.session import load_session
    session = load_session(root / "output")
    instances = json.loads((root / "output" / "segmentation_result.json").read_text())["instances"]
    classes = {int(i["instance_id"]): "structural" for i in instances}
    cfg = _cfg(**{"structural.regulated_dims": [
        {"label": "column", "dimension": "diameter", "value_m": 0.50, "tol_m": 0.005}]})
    rows = st.regulated_rows(session, instances, classes, cfg.structural.regulated_dims,
                             cfg.scale.sigma_regulated, np.array([0.0, 1.0, 0.0]),
                             chunk_ranges=[[0, 60], [30, 90], [60, 120]],
                             dims_pct=(cfg.loops.spatial.dims_pct_lo, cfg.loops.spatial.dims_pct_hi))
    assert rows and all(abs(r["log_s"]) < 0.10 for r in rows)      # 0.5 m columns, measured ≈ 0.5
    assert all(r["sigma"] >= cfg.scale.sigma_regulated for r in rows)
    p = st.write_absolute_rows(root / "output", rows, [[0, 60], [30, 90], [60, 120]])
    assert json.loads(p.read_text())["rows"]


def test_f2_config_validation():
    with pytest.raises(LoopsConfigError, match="correction_graph.graph.min_loop_gain"):
        load_loops_config(raw_server_cfg(**{"correction_graph.graph.min_loop_gain": 2.0}))
    with pytest.raises(LoopsConfigError, match="authority.saturation_warn"):
        load_loops_config(raw_server_cfg(**{"authority.saturation_warn": None}))
    with pytest.raises(LoopsConfigError, match="structural.wall_planarity.wall_tol_m"):
        load_loops_config(raw_server_cfg(**{"structural.wall_planarity.wall_tol_m": -1}))
    with pytest.raises(LoopsConfigError, match="regulated_dims"):
        load_loops_config(raw_server_cfg(**{"structural.regulated_dims": [{"label": "x"}]}))
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    cfg = load_loops_config(raw)
    assert cfg.graph.dense_max_unknowns == 12000 and cfg.certify.ensemble_offset_frames == 0
    assert cfg.structural.regulated_dims == ()


_FLOAT_WHITELIST = {0.0, 1.0, -1.0, 2.0, 0.5, 1e-9, 1e-6, 1e-12, 1e-18, 100.0, 1000.0, 255.0}


def test_no_decision_literals_in_f2_modules():
    offenders = []
    for name in ("structural.py", "kf_graph.py"):
        tree = ast.parse((PKG / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if node.value not in _FLOAT_WHITELIST:
                    offenders.append(f"{name}:{node.lineno} = {node.value}")
    assert not offenders, offenders
