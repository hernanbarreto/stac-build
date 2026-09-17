"""Transactionality: a failing Potree build changes NOTHING; selecting an
epoch leaves exactly the expected states and destroys none of the others; a
concurrent operation gets 409."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_objects, run_select           # noqa: E402
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


def test_selecting_epoch_0_restores_it_exactly_and_keeps_the_other(tmp_path):
    """USER 2026-09-16: "todas viven, solo se seleccionan y la que se selecciona
    se muestra". Selecting epoch 0 must restore it byte for byte AND leave
    epoch 1 on disk — Undo used to delete it."""
    scene = build_scene(tmp_path, drift_yaw_deg=1.5,
                        drift_t=(0.20, 0.0, 0.08))
    snap0 = session_files_snapshot(scene.output_dir)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "applied"
    assert (scene.output_dir / "_epoch_0").exists()
    res = run_select(scene.output_dir, 0)
    assert res["epoch"] == 0 and res["changed"]
    after = session_files_snapshot(scene.output_dir)
    for rel in GEOMETRY_FILES:
        assert after.get(rel) == snap0.get(rel), f"{rel} not restored"
    assert not (scene.output_dir / "geometry_epoch.json").exists()
    assert not (scene.output_dir / "depth_correction.json").exists()
    # epoch 1 is STILL THERE — this is the whole point
    assert (scene.output_dir / "_epoch_1").is_dir()
    assert res["available"] == [0, 1]

    # and we can go back up to it, which Undo made impossible
    back = run_select(scene.output_dir, 1)
    assert back["epoch"] == 1
    assert (scene.output_dir / "geometry_epoch.json").exists()
    assert sorted(back["available"]) == [0, 1]


def test_a_new_run_stacks_on_the_epoch_being_shown(tmp_path):
    """It used to refuse while an epoch was "pending approval". There is no
    approval any more (USER 2026-09-16), so a second correction simply runs on
    top of whichever epoch is on screen and both stay selectable."""
    scene = build_scene(tmp_path, drift_yaw_deg=1.5,
                        drift_t=(0.20, 0.0, 0.08))
    r1 = run_objects(scene.output_dir, [1, 2], "test", cfg=make_correction_cfg())
    assert r1["status"] == "applied"
    r2 = run_objects(scene.output_dir, [1, 2], "test", cfg=make_correction_cfg())
    assert r2["status"] in ("applied", "rejected")
    from correction.apply import available_epochs
    got = [e["epoch"] for e in available_epochs(scene.output_dir)]
    assert got == sorted(got) and 0 in got and len(got) >= 2

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
