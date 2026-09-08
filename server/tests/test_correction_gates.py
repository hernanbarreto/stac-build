"""Gates veto the application: scene exam, plausibility caps, continuity.
A rejected run modifies NO geometry file."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_objects                       # noqa: E402
from correction.session import read_ply, write_ply           # noqa: E402
from tests.synth_correction import (build_scene, make_correction_cfg,  # noqa: E402
                                    session_files_snapshot)

GEOMETRY_FILES = ("cleaned_cloud.ply", "cleaned_cloud_raw.ply",
                  "camera_poses.txt", "camera_frames.txt",
                  "segmentation_result.json", "scale_diagnostics.json")


def _assert_geometry_untouched(output_dir, snap_before):
    snap_after = session_files_snapshot(output_dir)
    for rel in GEOMETRY_FILES:
        assert snap_after.get(rel) == snap_before.get(rel), \
            f"{rel} was modified by a REJECTED run"
    assert not (output_dir / "geometry_epoch.json").exists()
    assert not (output_dir / "depth_correction.json").exists()
    assert not any(output_dir.glob("_tx_epoch_*"))
    assert not any(output_dir.glob("_epoch_*"))


def _displace_marked_objects_only(scene, iids, offset):
    """Corrupt ONLY the marked objects' revisit copies (the floor and the
    witnesses stay correct) — collapsing the copies then breaks the rest of
    the scene, which the exam must catch."""
    header, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    rev = set(scene.gt["revisit_kfs"])
    import json
    res = json.loads(
        (scene.output_dir / "segmentation_result.json").read_text())
    for inst in res["instances"]:
        if int(inst["instance_id"]) in iids:
            g = np.asarray(inst["globalIndices"], dtype=np.int64)
            m = g[np.isin(scene.ks[g], list(rev))]
            xyz[m] += np.asarray(offset)
    data["x"] = xyz[:, 0].astype(np.float32)
    data["y"] = xyz[:, 1].astype(np.float32)
    data["z"] = xyz[:, 2].astype(np.float32)
    write_ply(scene.output_dir / "cleaned_cloud.ply", header, data)
    write_ply(scene.output_dir / "cleaned_cloud_raw.ply", header, data)


def test_scene_exam_vetoes_and_touches_nothing(tmp_path):
    """The marked copies collapse, but the (correct) floor and witnesses of
    the same keyframes would be dragged along → rejection, zero changes."""
    scene = build_scene(tmp_path)          # NO real drift
    _displace_marked_objects_only(scene, [1, 2], (0.45, 0.0, 0.20))
    snap = session_files_snapshot(scene.output_dir)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "rejected", "the exam must veto"
    failed = [g["name"] for g in rep["gates"] if not g["passed"]]
    assert "scene_exam" in failed, failed
    _assert_geometry_untouched(scene.output_dir, snap)


def test_plausibility_caps(tmp_path):
    """MEJORAS §3 (box1): a transform beyond max_rot/max_t means broken
    anchors — rejected even when the copies collapse."""
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(3.6, 0.0, 0.0))
    snap = session_files_snapshot(scene.output_dir)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "rejected"
    failed = [g["name"] for g in rep["gates"] if not g["passed"]]
    assert "plausibility" in failed, failed
    _assert_geometry_untouched(scene.output_dir, snap)


def test_continuity_gate(tmp_path):
    """A per-keyframe step beyond max_step_mm vetoes."""
    scene = build_scene(tmp_path, drift_yaw_deg=1.0,
                        drift_t=(0.30, 0.0, 0.10))
    snap = session_files_snapshot(scene.output_dir)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg(**{"gates.max_step_mm": 5.0}))
    assert rep["status"] == "rejected"
    failed = [g["name"] for g in rep["gates"] if not g["passed"]]
    assert "continuity" in failed, failed
    _assert_geometry_untouched(scene.output_dir, snap)


def test_rejected_run_lands_in_the_ledger(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=0.5, drift_t=(3.6, 0, 0))
    run_objects(scene.output_dir, [1, 2], "test",
                cfg=make_correction_cfg())
    from correction.ledger import ledger_view
    rows = ledger_view(scene.output_dir)
    assert len(rows) == 1 and rows[0]["verdict"] == "rejected"
    # the per-run report file persists (H5: never overwritten away)
    rp = scene.output_dir / rows[0]["report"]
    assert rp.exists()
