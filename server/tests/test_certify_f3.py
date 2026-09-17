"""claude_stac.txt F3 — §12.8 the certification loop converges and stops,
a rejected iteration leaves the previous epoch bit for bit and every epoch of
the chain stays selectable bit for bit; §12.7d known answer within tolerance
and an envelope monotone with the loop density; §12.7e determinism;
§12.9 provenance survives every epoch and the epoch replay is exact;
§10.12 the adversarial suite declares every failure with its cause."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.loops.config import load_loops_config                        # noqa: E402
from reconstruction.witness.frames import frames_from_arrays                     # noqa: E402
from reconstruction.witness.depth_tracks import load_tracks                      # noqa: E402
from tests.synth_metric import (make_session, write_session_dir, write_aligned_chunks,   # noqa: E402
                                write_images, synthetic_tracks, raw_server_cfg, drift_field,
                                corridor_loop_scene, loop_trajectory)
from tests.synth_correction import make_correction_cfg, session_files_snapshot   # noqa: E402

N_KF = 72
GEOMETRY_FILES = ("cleaned_cloud.ply", "camera_poses.txt", "segmentation_result.json",
                  "depth_correction.json", "geometry_epoch.json")


def _instances(sess):
    inst, iid = {}, 1
    for p in sess.scene.prims:
        if getattr(p, "label", "") in ("wall", "column", "box"):
            inst[iid] = {"label": p.label, "oids": [p.oid]}
            iid += 1
    return inst


def _cfg(**over):
    base = {"loops.cluster_min_points": 150,
            # the synthetic planes are exact: their patch normals carry the
            # precision of a plane fitted on hundreds of noise-free points
            "structural.wall_planarity.sigma_angle_deg": 0.2,
            "structural.wall_planarity.min_points_per_kf": 60,
            "structural.column_vertical.min_points_per_kf": 30,
            "structural.repeated_parallel.min_points_per_kf": 30,
            "witness.mask_erosion_px": 1, "witness.tracks.min_obs_per_frame": 10,
            "witness.depth.pair_samples": 3000, "witness.contours.enabled": False}
    base.update(over)
    return load_loops_config(raw_server_cfg(**base))


@pytest.fixture(scope="module")
def truth():
    return make_session(H=40, W=56, scene=corridor_loop_scene(), poses=loop_trajectory(N_KF, extra_laps=0.15))


def _drift(sess):
    return drift_field(sess.n_kf, yaw_deg_per_kf=0.01, t_per_kf=(0.004, 0.0, 0.002))


def _write(root, sess, D):
    write_session_dir(root, sess, _instances(sess), point_stride=2, drift_by_kf=D)
    write_aligned_chunks(root, sess, chunk_size=24, overlap=12, drift_by_kf=D)
    write_images(root, sess)
    synthetic_tracks(root, sess, win=12, stride=6, per_frame=30)
    return root


def _run(root, sess, cfg, **kw):
    from reconstruction.certify.run import certify_session
    base = frames_from_arrays(sess.depth, sess.K, sess.poses, sess.frame_numbers)
    return certify_session(root, cfg, operator="test", log=lambda m: None, correction_cfg=make_correction_cfg(),
                           device="cpu", base_frames=base, tracks=load_tracks(root / "output"),
                           use_fork_edges=False, **kw)


def _pose_errors(root, sess):
    from correction.session import read_poses
    P = read_poses(root / "output" / "camera_poses.txt")
    return np.array([np.linalg.norm(P[g][:3, 3] - sess.poses[g][:3, 3]) for g in range(sess.n_kf)])


def test_loop_converges_stops_and_every_epoch_is_selectable(tmp_path, truth):
    sess = truth
    D = _drift(sess)
    root = _write(tmp_path / "s", sess, D)
    out = root / "output"
    snap0 = session_files_snapshot(out)
    err0 = np.array([np.linalg.norm((D[g] @ sess.poses[g])[:3, 3] - sess.poses[g][:3, 3]) for g in range(N_KF)])
    cfg = _cfg()
    acta = _run(root, sess, cfg, max_iters=3)
    assert acta["stop_reason"] and acta["stopped_at"] is not None
    its = acta["iterations"]
    assert its and its[0]["verdict"] == "applied", its[0]
    assert all(it["verdict"] in ("applied", "identity") for it in its), [it["verdict"] for it in its]
    assert acta["epoch_final"] >= 1
    # the geometry moved toward the truth: the revisited stretch is closed
    err1 = _pose_errors(root, sess)
    revisited = np.arange(int(N_KF * 0.85), N_KF)
    assert err1[revisited].max() < 0.5 * err0[revisited].max(), (err0[revisited].max(), err1[revisited].max())
    assert np.median(err1) < np.median(err0)
    # the objective went down and the loop stopped for a declared reason
    assert acta["metrics_final"]["objective"] < acta["metrics_initial"]["objective"]
    assert ("converged" in acta["stop_reason"]) or ("max_iters" in acta["stop_reason"]) \
        or ("nothing to close" in acta["stop_reason"])
    # one epoch per applied iteration, every one of them on disk and selectable
    n_applied = sum(1 for it in its if it["verdict"] == "applied")
    assert acta["epoch_final"] == n_applied
    from correction.apply import available_epochs
    from correction import ledger
    assert [e["epoch"] for e in available_epochs(out)] == list(range(n_applied + 1))
    assert len(ledger.applied_runs(out)) == n_applied
    snap_top = session_files_snapshot(out)
    for q in (out / "quality").glob("report_epoch_*.json"):
        rep = json.loads(q.read_text())
        assert rep["metrics"]["witnesses"]["status_counts"] and "seam_residual" in rep["metrics"]
    # selecting the original gives it back bit for bit, and the whole chain of
    # certified epochs is still there to go back up to (USER 2026-09-16)
    from correction.run import run_select
    from correction.epoch import current_epoch
    run_select(out, 0, "test")
    assert current_epoch(out) == 0
    snap_back = session_files_snapshot(out)
    for rel in GEOMETRY_FILES:
        assert snap_back.get(rel) == snap0.get(rel), f"{rel} not restored bit for bit"
    assert [e["epoch"] for e in available_epochs(out)] == list(range(n_applied + 1))
    run_select(out, n_applied, "test")
    assert current_epoch(out) == n_applied
    for rel in GEOMETRY_FILES:
        assert session_files_snapshot(out).get(rel) == snap_top.get(rel), \
            f"{rel} not restored bit for bit going back up"


def _liar_depth_stage(monkeypatch):
    """A depth stage that lies: a 10 % compression of every frame 'applied'."""
    import reconstruction.witness.depth_tracks as dt
    real = dt.depth_stage

    def liar(frames, wcfg, tracks=None, contour_obs=None, log=print, seed=0):
        rep = real(frames, wcfg, tracks, contour_obs, log, seed)
        rep["applied"] = True
        rep["a"] = {f: 0.9 for f in rep["a"]}
        rep["b"] = {f: 0.0 for f in rep["b"]}
        rep["reason"] = "liar"
        return rep

    monkeypatch.setattr(dt, "depth_stage", liar)


def test_veto_mode_rejected_iteration_keeps_the_previous_epoch_bit_for_bit(tmp_path, truth, monkeypatch):
    """certify.gates.mode = veto (evaluation): a failed gate rejects the
    iteration and the previous epoch stays bit for bit."""
    sess = truth
    root = _write(tmp_path / "s", sess, _drift(sess))
    out = root / "output"
    cfg = _cfg(**{"certify.gates.mode": "veto"})
    acta1 = _run(root, sess, cfg, max_iters=1)
    assert acta1["iterations"][0]["verdict"] == "applied"
    snap1 = session_files_snapshot(out)
    _liar_depth_stage(monkeypatch)
    acta2 = _run(root, sess, cfg, max_iters=1)
    it = acta2["iterations"][0]
    assert it["verdict"] == "rejected", it
    assert it["gate_mode"] == "veto"
    assert "vs" in it["reason"] and any(not g["passed"] for g in it["gates"])
    assert "rejected" in acta2["stop_reason"]
    snap2 = session_files_snapshot(out)
    for rel in GEOMETRY_FILES:
        assert snap2.get(rel) == snap1.get(rel), f"{rel} changed by a rejected iteration"
    from correction import ledger
    rows = ledger.ledger_view(out)
    assert rows[-1]["kind"] == "certify" and rows[-1]["verdict"] == "rejected"


def test_advisory_mode_failed_gate_is_applied_and_declared(tmp_path, truth, monkeypatch):
    """Production (certify.gates.mode = advisory, USER 2026-09-13: the cloud
    the UI receives is the corrected one): the same lying depth stage fails
    a gate, the gate is DECLARED (acta warning, kit ⚠, attention list) and the
    iteration is still APPLIED as a new epoch — the user judges it by
    selecting between the epochs, which all stay on disk (USER 2026-09-16)."""
    sess = truth
    root = _write(tmp_path / "s", sess, _drift(sess))
    out = root / "output"
    cfg = _cfg()
    assert cfg.certify.gates.mode == "advisory"
    acta1 = _run(root, sess, cfg, max_iters=1)
    assert acta1["iterations"][0]["verdict"] == "applied"
    snap1 = session_files_snapshot(out)
    _liar_depth_stage(monkeypatch)
    acta2 = _run(root, sess, cfg, max_iters=1)
    it = acta2["iterations"][0]
    assert it["verdict"] == "applied", it
    assert it["gate_mode"] == "advisory" and it["gate_warnings"], it["gates"]
    failed = [g for g in it["gates"] if not g["passed"]]
    assert failed and all(g.get("advisory") for g in failed)
    assert acta2["epoch_final"] == acta1["epoch_final"] + 1
    snap2 = session_files_snapshot(out)
    assert snap2.get("cleaned_cloud.ply") != snap1.get("cleaned_cloud.ply")
    from correction import ledger
    rows = ledger.ledger_view(out)
    assert rows[-1]["kind"] == "certify" and rows[-1]["verdict"] == "applied"
    from reconstruction.certify.attention import attention_list
    att = attention_list(out)
    warn = [a for a in att["items"] if a["kind"] == "gate_warning"]
    assert warn and all("advisory" in a["text"] for a in warn)
    # the kit can still SHOW the previous epoch, bit for bit — and the declared
    # one stays on disk instead of being destroyed by the comparison
    from correction.run import run_select
    from correction.apply import available_epochs
    from correction.epoch import current_epoch
    run_select(out, acta1["epoch_final"], "test")
    assert current_epoch(out) == acta1["epoch_final"]
    for rel in GEOMETRY_FILES:
        assert session_files_snapshot(out).get(rel) == snap1.get(rel), rel
    assert acta2["epoch_final"] in [e["epoch"] for e in available_epochs(out)]
    run_select(out, acta2["epoch_final"], "test")
    for rel in GEOMETRY_FILES:
        assert session_files_snapshot(out).get(rel) == snap2.get(rel), rel


def test_instance_copies_become_pose_edges_whatever_the_class(tmp_path, truth):
    """§4.4 (USER 2026-09-13: "nunca debe descartarse un duplicado detectado
    por SAM3"): the copies of a revisited SAM3 instance are measured on the
    cloud and become pose-graph edges; a non-structural class (here the
    default when the VLM is unavailable) only inflates σ, never drops the
    candidate; the certification closes the duplicates."""
    from reconstruction.loops.instance_loops import detect_instance_loops
    from reconstruction.certify.loops_posthoc import instance_edges
    from correction.session import load_session
    sess = truth
    root = _write(tmp_path / "s", sess, _drift(sess))
    out = root / "output"
    cfg = _cfg(**{"loops.semantic.default_class": "movable"})
    det = detect_instance_loops(out, root, cfg, log=lambda m: None, apply_splits=True)
    cands = [c for c in det["candidates"] if c["verdict"] in ("loop", "ambiguous")]
    assert cands, det["candidates"]
    assert all(c["class"] == "movable" for c in cands)
    assert det["n_written"] == len(cands)
    txt = (out / "maplong_run" / "loop_closures.txt").read_text()
    assert "instance:movable" in txt
    session = load_session(out)
    edges = instance_edges(session, cands, make_correction_cfg(), cfg, log=lambda m: None)
    ok = [e for e in edges if e.get("accepted")]
    assert ok, edges
    for e in ok:
        assert e["source"] == "instance" and "Z" in e and e["info_t"].shape == (3, 3)
        assert e["offset_after_m"] < e["offset_before_m"]
        assert e["sigma_factors"].get("class:movable") == cfg.loops.semantic.nonstructural_sigma_factor
        assert e["sigma_m"] >= cfg.certify.visit_loops.sigma_floor_m * cfg.loops.semantic.nonstructural_sigma_factor
    # the loop applies them: the acta records instance edges and the
    # duplicates go down
    acta = _run(root, sess, cfg, max_iters=2)
    it0 = acta["iterations"][0]
    assert it0["verdict"] == "applied", it0
    assert any(l.get("source") == "instance" and l.get("accepted") for l in it0["loops"])
    assert acta["metrics_final"]["duplicates"]["n"] <= acta["metrics_initial"]["duplicates"]["n"]
    assert acta["metrics_final"]["closure"]["median_m"] < acta["metrics_initial"]["closure"]["median_m"]


def test_known_answer_and_envelope_monotone_with_loop_density(tmp_path, truth):
    from reconstruction.quality.known_answer import known_answer, envelope
    sess = truth
    root = _write(tmp_path / "s", sess, None)
    cfg = _cfg(**{"certify.known_answer.yaw_deg": 0.5, "certify.known_answer.t_m": 0.15,
                  "certify.known_answer.scale": 1.02,
                  "certify.envelope.levels_t_m": [0.1, 0.3], "certify.envelope.levels_scale_pct": [2, 6],
                  "certify.envelope.loop_densities": [0.5, 1.0]})
    base = frames_from_arrays(sess.depth, sess.K, sess.poses, sess.frame_numbers)
    kw = dict(device="cpu", base_frames=base, tracks=load_tracks(root / "output"), use_fork_edges=False)
    rep = known_answer(root, cfg, make_correction_cfg(), work=tmp_path / "ka", log=lambda m: None, **kw)
    # §10.10 asks to MEASURE and DECLARE the recovery, not to pass or fail it
    # against a tolerance nothing derived (USER 2026-09-14: "que mida y
    # declare, no que falle ... quién dijo que 5 cm es lógico"). So what is
    # asserted is that the injection landed, that the instrument recovers, and
    # that the numbers reach the report — all comparisons, no thresholds.
    assert rep["error_before"]["t_m_max"] > 0.0, rep["error_before"]
    assert rep["improved"], rep["error_after"]
    assert rep["error_after"]["t_m_max"] < rep["error_before"]["t_m_max"]
    assert 0.0 < rep["recovered_fraction"]["t"] <= 1.0, rep["recovered_fraction"]
    assert rep["provenance"] == "tool_measured"
    env = envelope(root, cfg, make_correction_cfg(), work=tmp_path / "env", log=lambda m: None, **kw)
    assert env["monotone_with_loop_density"], env["per_density"]
    assert env["declared"]["max_correctable_t_m"] is not None
    assert env["declared"]["min_loop_density_recovering"] is not None


def test_determinism_two_runs_same_acta(tmp_path, truth):
    from reconstruction.quality.determinism import determinism
    sess = truth
    root = _write(tmp_path / "s", sess, _drift(sess))
    cfg = _cfg()
    base = frames_from_arrays(sess.depth, sess.K, sess.poses, sess.frame_numbers)
    rep = determinism(root, cfg, make_correction_cfg(), work=tmp_path / "det", log=lambda m: None,
                      device="cpu", base_frames=base, tracks=load_tracks(root / "output"), use_fork_edges=False)
    assert rep["identical_within_tolerance"], rep["differences"]


def test_provenance_survives_the_loop_and_the_epoch_replays_exactly(tmp_path, truth):
    sess = truth
    root = _write(tmp_path / "s", sess, _drift(sess))
    out = root / "output"
    from correction.session import read_ply, read_poses
    _, before = read_ply(out / "cleaned_cloud.ply")
    poses0 = read_poses(out / "camera_poses.txt")
    cfg = _cfg()
    acta = _run(root, sess, cfg, max_iters=1)
    assert acta["iterations"][0]["verdict"] == "applied"
    _, after = read_ply(out / "cleaned_cloud.ply")
    assert len(after) == len(before)
    for fld in ("frame_global", "pixel_row", "pixel_col", "confidence"):
        assert np.array_equal(after[fld], before[fld]), fld
    for fld in ("mv_votes", "mask_votes", "mask_conflicts", "status"):
        assert fld in after.dtype.names
    res = json.loads((out / "segmentation_result.json").read_text())["instances"]
    for r in res:
        gi = np.asarray(r["globalIndices"], np.int64)
        if len(gi):
            assert gi.min() >= 0 and gi.max() < len(after)
    # the epoch npz replays the exact transform (bit-faithful, keyed by frame_global)
    from correction.replay import apply_epoch_to_arrays
    from correction import ledger
    npz = ledger.load_epoch_npz(out, 1)
    xyz = np.stack([before["x"], before["y"], before["z"]], 1).astype(np.float64)
    frames = [int(x) for x in (out / "camera_frames.txt").read_text().split()]
    poses = poses0.copy()
    apply_epoch_to_arrays(xyz, before["frame_global"].astype(np.int64), poses, frames, npz)
    cur = np.stack([after["x"], after["y"], after["z"]], 1).astype(np.float64)
    assert np.abs(xyz - cur).max() < 1e-4
    assert np.allclose(poses, read_poses(out / "camera_poses.txt"), atol=1e-6)


def test_adversarial_suite_declares_every_failure(tmp_path):
    from reconstruction.quality.adversarial import run_suite, SCENARIOS
    cfg = _cfg()
    rep = run_suite(cfg, make_correction_cfg(), scenarios=SCENARIOS, work=tmp_path / "adv",
                    out_json=tmp_path / "adv" / "report.json", log=lambda m: None, device="cpu", max_iters=1)
    assert len(rep["scenarios"]) == len(SCENARIOS)
    assert rep["silent_failures"] == [], rep["silent_failures"]
    for r in rep["scenarios"]:
        assert r["status"] in ("passed", "failed", "not_runnable")
        if r["status"] != "passed":
            assert r["cause"], r
    assert rep["n_not_runnable"] == 2
    assert (tmp_path / "adv" / "report.json").exists()
