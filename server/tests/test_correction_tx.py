"""Transactionality: a failing Potree build changes NOTHING; undo/approve
leave exactly the expected states; a concurrent operation gets 409."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_objects, run_verdict          # noqa: E402
from tests.synth_correction import (build_scene, make_correction_cfg,  # noqa: E402
                                    session_files_snapshot)

GEOMETRY_FILES = ("cleaned_cloud.ply", "cleaned_cloud_raw.ply",
                  "camera_poses.txt", "segmentation_result.json",
                  "scale_diagnostics.json")


def test_potree_failure_discards_the_transaction(tmp_path, monkeypatch):
    scene = build_scene(tmp_path, drift_yaw_deg=1.5,
                        drift_t=(0.20, 0.0, 0.08))
    snap = session_files_snapshot(scene.output_dir)
    import potree_converter
    monkeypatch.setattr(potree_converter, "convert_ply_to_potree",
                        lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="Potree build FAILED"):
        run_objects(scene.output_dir, [1, 2], "test",
                    cfg=make_correction_cfg(**{"apply.potree_rebuild": True}))
    after = session_files_snapshot(scene.output_dir)
    for rel in GEOMETRY_FILES:
        assert after.get(rel) == snap.get(rel), f"{rel} changed"
    assert not (scene.output_dir / "geometry_epoch.json").exists()
    assert not any(scene.output_dir.glob("_tx_epoch_*"))
    assert not any(scene.output_dir.glob("_epoch_*"))


def test_undo_restores_exactly_then_reapply_and_approve(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=1.5,
                        drift_t=(0.20, 0.0, 0.08))
    snap0 = session_files_snapshot(scene.output_dir)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending"
    assert (scene.output_dir / "_epoch_0").exists()
    res = run_verdict(scene.output_dir, "undone", "test")
    assert res["verdict"] == "undone" and res["epoch"] == 0
    after = session_files_snapshot(scene.output_dir)
    for rel in GEOMETRY_FILES:
        assert after.get(rel) == snap0.get(rel), f"{rel} not restored"
    assert not (scene.output_dir / "geometry_epoch.json").exists()
    assert not (scene.output_dir / "depth_correction.json").exists()
    assert not any(scene.output_dir.glob("_epoch_*"))
    # run again on the RESTORED session, then approve
    rep2 = run_objects(scene.output_dir, [1, 2], "test",
                       cfg=make_correction_cfg())
    assert rep2["status"] == "pending"
    res2 = run_verdict(scene.output_dir, "approved", "test")
    assert res2["verdict"] == "approved" and res2["epoch"] == 1
    assert not any(scene.output_dir.glob("_epoch_*")), \
        "approve leaves no remains"
    from correction.ledger import ledger_view
    rows = ledger_view(scene.output_dir)
    verdicts = [r["verdict"] for r in rows]
    assert verdicts == ["undone", "approved"], verdicts


def test_pending_blocks_a_new_run(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=1.5,
                        drift_t=(0.20, 0.0, 0.08))
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending"
    with pytest.raises(RuntimeError, match="pending"):
        run_objects(scene.output_dir, [1, 2], "test",
                    cfg=make_correction_cfg())


def test_session_lock_409():
    """The API-level per-session lock: the second caller gets 409 with the
    blocking task id."""
    from fastapi import HTTPException
    from correction.api import _acquire, _release
    _acquire("sess-a", "task-1")
    try:
        with pytest.raises(HTTPException) as ei:
            _acquire("sess-a", "task-2")
        assert ei.value.status_code == 409
        assert ei.value.detail["blocking_task_id"] == "task-1"
        _acquire("sess-b", "task-3")     # other sessions unaffected
        _release("sess-b")
    finally:
        _release("sess-a")
    _acquire("sess-a", "task-4")         # released → free again
    _release("sess-a")
