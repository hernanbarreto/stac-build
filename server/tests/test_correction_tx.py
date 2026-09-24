"""Transactionality: a failing Potree build changes NOTHING; selecting an
epoch leaves exactly the expected states and destroys none of the others.

Driven through ``run_floor`` since 2026-09-24: the manual object correction
(``run_objects``) went with the UI "Corrections" button, and the floor
alignment is the run that still goes through the same transactional apply.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_floor, run_select             # noqa: E402
from tests.synth_correction import (build_scene, make_correction_cfg,  # noqa: E402
                                    session_files_snapshot)

GEOMETRY_FILES = ("cleaned_cloud.ply", "cleaned_cloud_raw.ply",
                  "camera_poses.txt", "segmentation_result.json",
                  "scale_diagnostics.json")


def _drifted_ramp(tmp_path):
    """A ramp floor with a y-drift along the walk: the plane model has a real
    correction to apply (same scene as test_correction_floor)."""
    return build_scene(tmp_path, floor="ramp", floor_slope=0.04,
                       drift_yaw_deg=0.0, drift_t=(0.0, 0.10, 0.0))


def test_potree_failure_discards_the_transaction(tmp_path, monkeypatch):
    scene = _drifted_ramp(tmp_path)
    snap = session_files_snapshot(scene.output_dir)
    import potree_converter
    monkeypatch.setattr(potree_converter, "convert_ply_to_potree",
                        lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="Potree build FAILED"):
        run_floor(scene.output_dir, "plane", None, "test",
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
    scene = _drifted_ramp(tmp_path)
    snap0 = session_files_snapshot(scene.output_dir)
    rep = run_floor(scene.output_dir, "plane", None, "test",
                    cfg=make_correction_cfg())
    assert rep["status"] == "applied", rep.get("rejection_reason")
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
    scene = _drifted_ramp(tmp_path)
    r1 = run_floor(scene.output_dir, "plane", None, "test",
                   cfg=make_correction_cfg())
    assert r1["status"] == "applied", r1.get("rejection_reason")
    r2 = run_floor(scene.output_dir, "plane", None, "test",
                   cfg=make_correction_cfg())
    assert r2["status"] in ("applied", "rejected")
    from correction.apply import available_epochs
    got = [e["epoch"] for e in available_epochs(scene.output_dir)]
    assert got == sorted(got) and 0 in got and len(got) >= 2
