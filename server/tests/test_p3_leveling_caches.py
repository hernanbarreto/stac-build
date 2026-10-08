"""docs/plan_determinismo.md points 110 (the floor leveling: seeded, cached by the cloud),
113 (the mask audit's camera is the cloud's own), 115 (the evidence cache is keyed by its
inputs), 118 (no clock in scene_r.db), 119 (the audit's per-point view counts) — 2026-10-08."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _floor_cloud(seed, n=20000):
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(-5, 5, n), rng.normal(0.3, 0.01, n), rng.uniform(-5, 5, n)]
    a = np.radians(4.0)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return pts @ R.T


def test_the_leveling_is_seeded_and_keyed_by_the_cloud():
    from alignment_manager import AlignmentManager
    am = AlignmentManager()
    cloud = _floor_cloud(0)
    s1, R1, t1 = am.compute_leveling_from_points(cloud)
    s2, R2, t2 = AlignmentManager().compute_leveling_from_points(cloud.copy())
    assert np.array_equal(R1, R2) and np.array_equal(t1, t2), "one cloud, one transform (point 110)"
    assert abs(np.degrees(np.arccos(np.clip(R1[1, 1], -1, 1))) - 4.0) < 0.5
    # the cache serves THIS cloud only: another cloud through the same manager gets its own fit
    other = _floor_cloud(1) + [0.0, 1.0, 0.0]
    s3, R3, t3 = am.compute_leveling_from_points(other)
    assert not (np.array_equal(R3, R1) and np.array_equal(t3, t1))
    s4, R4, t4 = am.compute_leveling_from_points(cloud)
    assert np.array_equal(R4, R1) and np.array_equal(t4, t1)


def test_the_instance_store_carries_no_clock(tmp_path):
    from phase_r.instance_store import InstanceStore
    paths = []
    for k in range(2):
        p = tmp_path / f"scene_{k}.db"
        st = InstanceStore(p)
        st.upsert_instance(1, "desk")
        st.set_points(1, np.zeros((3, 3), np.float32))
        st.add_user_volume("box", [0, 0, 0], [1, 1, 1])
        st.close()
        paths.append(p)
    con = sqlite3.connect(paths[0])
    assert con.execute("SELECT created_ns FROM instances").fetchone()[0] == 0
    assert con.execute("SELECT created_ns FROM user_volumes").fetchone()[0] == 0
    con.close()
    assert paths[0].read_bytes() == paths[1].read_bytes(), "two builds, one byte sequence (point 118)"


def test_the_evidence_is_keyed_by_its_inputs(tmp_path, monkeypatch):
    from reconstruction.surface_fit import hole_audit as HA
    out = tmp_path / "output"
    out.mkdir()
    (out / "seg_masks.npz").write_bytes(b"x")
    (out / "camera_poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
    k1 = HA.evidence_identity(out)
    (out / "camera_poses.txt").write_text("1 0 0 0.5 0 1 0 0 0 0 1 0 0 0 0 1\n")   # another epoch's poses
    k2 = HA.evidence_identity(out)
    assert k1 != k2
    built = []

    class _Ev:
        ok = False
        reason = "synthetic"
        masks = None

        def __init__(self, o, s):
            built.append(str(o))

    monkeypatch.setattr(HA, "_Evidence", _Ev)
    HA._EVIDENCE_CACHE.clear()
    HA._evidence(out, tmp_path)
    HA._evidence(out, tmp_path)
    assert len(built) == 1, "the same inputs are built once"
    (out / "seg_masks.npz").write_bytes(b"y")
    HA._evidence(out, tmp_path)
    assert len(built) == 2, "another mask store is another evidence (point 115)"
    assert [k for k in HA._EVIDENCE_CACHE if k[0] == str(out)] and \
        len([k for k in HA._EVIDENCE_CACHE if k[0] == str(out)]) == 1, "the previous version is dropped"


def test_the_cloud_camera_is_camera_json_on_the_record_grid_never_stray(tmp_path):
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    from segmentation.session_io import CloudCameraError, load_cloud_camera_source
    out = tmp_path / "output"
    out.mkdir()
    with pytest.raises(CloudCameraError, match="camera.json does not exist"):
        load_cloud_camera_source(out)
    g = grid_full_frame_resize(400, 300, 200, 150, "omega")
    cam = CameraModel(width=400, height=300, params=(380.0, 378.0, 199.5, 149.5, 0, 0, 0, 0),
                      source="refine", camera_epoch=1, omega_grid=g, mask_grid=None)
    save_camera_json(out / "camera.json", cam, geometry_epoch=0)
    (out / "camera_frames.txt").write_text("10\n20\n")
    (out / "camera_poses.txt").write_text(" ".join(str(v) for v in np.eye(4).ravel()) + "\n"
                                         + " ".join(str(v) for v in np.eye(4).ravel()) + "\n")
    # a Stray capture beside the scan must NOT be the camera of the cloud
    stray = tmp_path / "inputs" / "stray"
    stray.mkdir(parents=True)
    (stray / "odometry.csv").write_text("x")
    (stray / "camera_matrix.csv").write_text("x")
    src = load_cloud_camera_source(out)
    assert sorted(src.pose_map) == [10, 20] and src.backend.startswith("camera.json")
    K = src.K_for(10)
    assert abs(K[0, 0] - 380.0 / 2.0) < 1e-9 and src.source_resolution == (150, 200), \
        "K on the record grid (half the native frame here)"
    (out / "camera_frames.txt").write_text("10\n")
    with pytest.raises(CloudCameraError, match="not one list"):
        load_cloud_camera_source(out)


def test_the_audit_tie_breaks_equal_areas_by_keyframe_and_counts_views():
    from segmentation.mask_filter import MaskAudit
    src = (Path(__file__).resolve().parents[1] / "segmentation" / "mask_filter.py").read_text()
    assert "ranked.sort(key=lambda r: (-r[0], r[1]))" in src
    assert "judged_views" in MaskAudit._PER_POINT and "off_views" in MaskAudit._PER_POINT
    assert 'raise RuntimeError(f"mask audit: no mask/camera evidence' in src, \
        "an audit that cannot run fails the projection (point 122)"
