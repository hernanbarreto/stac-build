"""Downstream consistency after an applied correction: raw==cleaned coherence,
globalIndices stability, epoch stamping.

Driven through the floor alignment since 2026-09-24 (the manual object
correction went with the UI "Corrections" button). The depth sidecar written
by the depth stage is covered by test_depth_f3."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.epoch import check, current_epoch, stamp     # noqa: E402
from correction.run import run_floor                         # noqa: E402
from correction.session import read_ply                      # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def _run_floor_scene(tmp_path):
    scene = build_scene(tmp_path, floor="ramp", floor_slope=0.04,
                        drift_yaw_deg=0.0, drift_t=(0.0, 0.10, 0.0))
    rep = run_floor(scene.output_dir, "plane", None, "test",
                    cfg=make_correction_cfg())
    assert rep["status"] == "applied", rep.get("rejection_reason")
    return scene, rep


def test_raw_and_cleaned_share_the_transform(tmp_path):
    scene, rep = _run_floor_scene(tmp_path)
    _, a = read_ply(scene.output_dir / "cleaned_cloud.ply")
    _, b = read_ply(scene.output_dir / "cleaned_cloud_raw.ply")
    for f in ("x", "y", "z", "frame_global"):
        assert np.array_equal(a[f], b[f]), f


def test_global_indices_stay_valid(tmp_path):
    scene, rep = _run_floor_scene(tmp_path)
    _, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    res = json.loads(
        (scene.output_dir / "segmentation_result.json").read_text())
    # order preserved → each instance's indices still carry the SAME
    # provenance rows as at build time
    for inst in res["instances"]:
        g = np.asarray(inst["globalIndices"], dtype=np.int64)
        assert np.array_equal(data["frame_global"][g], scene.fg[g])


def test_epoch_stamping(tmp_path):
    scene, rep = _run_floor_scene(tmp_path)
    assert current_epoch(scene.output_dir) == 1
    meta = stamp({"method": "x"}, scene.output_dir)
    assert meta["geometry_epoch"] == 1
    # every applied correction counts — there is no approving any more
    # (USER 2026-09-16: the epochs are selected, not approved)
    assert meta["human_directed_corrections"] == 1
    meta2 = stamp({"method": "x"}, scene.output_dir)
    assert meta2["human_directed_corrections"] == 1
    # an artifact stamped before the correction reads as stale
    assert check({"geometry_epoch": 0}, scene.output_dir)["stale"]
    assert not check(meta2, scene.output_dir)["stale"]
