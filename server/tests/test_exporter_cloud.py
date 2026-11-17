"""The exporter's cloud (USER 2026-11-17: "podriamos exportar la nube de puntos compatible con
varios programas"): the published cloud written as PLY, LAS and E57 holds the SAME points — in
the display frame turned Z-up, filtered by the viewer's confidence — and PLY / LAS keep each
point's origin."""
from __future__ import annotations

import json

import numpy as np
import pytest

from exporter import cloud as C

DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                  ("confidence", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")])


def _scan(tmp_path, n=5000, floor=None, octree_range=None):
    rng = np.random.default_rng(1)
    rec = np.zeros(n, DTYPE)
    rec["x"], rec["y"], rec["z"] = rng.uniform(-8, 8, n), rng.uniform(0, 3, n), rng.uniform(-5, 5, n)
    for c in ("red", "green", "blue"):
        rec[c] = rng.integers(0, 256, n)
    rec["confidence"] = rng.uniform(1.0, 9.0, n)
    rec["frame_global"], rec["pixel_row"], rec["pixel_col"] = rng.integers(0, 9000, n), rng.integers(0, 720, n), rng.integers(0, 1280, n)
    out = tmp_path / "output"
    out.mkdir()
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    head += [f"property {C._PLY_NAMES[DTYPE[k].str.replace('|', '')]} {k}" for k in DTYPE.names] + ["end_header"]
    (out / "cleaned_cloud.ply").write_bytes(("\n".join(head) + "\n").encode() + rec.tobytes())
    if floor is not None:
        np.savez(out / "floor_transform.npz", s=floor[0], R=floor[1], t=floor[2])
    if octree_range is not None:
        (out / "potree").mkdir()
        (out / "potree" / "metadata.json").write_text(json.dumps(
            {"attributes": [{"name": "confidence", "min": [octree_range[0]], "max": [octree_range[1]]}]}))
    return out, rec


def test_the_three_formats_hold_the_same_points(tmp_path):
    laspy = pytest.importorskip("laspy")
    pye57 = pytest.importorskip("pye57")
    out, rec = _scan(tmp_path)
    cloud = C.Cloud(out)
    assert cloud.identity and cloud.stats()["points"] == len(rec) == cloud.total
    want = np.column_stack([rec["x"], -rec["z"], rec["y"]])                 # Z up: (x, −z, y), exact
    C.write_ply(cloud, tmp_path / "c.ply")
    back = C.open_ply(tmp_path / "c.ply")
    assert back.dtype == DTYPE and np.array_equal(np.column_stack([back["x"], back["y"], back["z"]]), want)
    for k in DTYPE.names[3:]:
        assert np.array_equal(back[k], rec[k]), k                           # colour, confidence, the origin
    C.write_las(cloud, tmp_path / "c.las")
    las = laspy.read(str(tmp_path / "c.las"))
    assert las.header.version.minor == 4 and las.header.point_format.id == 7 and len(las.points) == len(rec)
    assert np.abs(np.column_stack([las.x, las.y, las.z]) - want.astype(np.float64)).max() <= C.LAS_SCALE / 2 + 1e-12
    assert np.array_equal(np.asarray(las.red), rec["red"].astype(np.uint16) * 257)
    for k in ("frame_global", "pixel_row", "pixel_col"):
        assert np.array_equal(np.asarray(las[k]), rec[k]), k
    assert np.array_equal(np.asarray(las["confidence"]), rec["confidence"])
    C.write_e57(cloud, tmp_path / "c.e57", name="test")
    e57 = pye57.E57(str(tmp_path / "c.e57"))
    raw = e57.read_scan_raw(0)
    assert np.array_equal(np.column_stack([raw["cartesianX"], raw["cartesianY"], raw["cartesianZ"]]), want.astype(np.float64))
    assert np.array_equal(raw["colorGreen"], rec["green"])


def test_the_confidence_filter_is_the_viewers(tmp_path):
    # normalised by the octree metadata's range, as the viewer and GenRecon do
    out, rec = _scan(tmp_path, octree_range=(0.0, 10.0))
    cloud = C.Cloud(out, confidence=0.5)
    keep = (rec["confidence"].astype(np.float64) - 0.0) / 10.0 >= 0.5
    assert cloud.stats()["points"] == int(keep.sum()) < len(rec)
    C.write_ply(cloud, tmp_path / "c.ply")
    assert np.array_equal(C.open_ply(tmp_path / "c.ply")["frame_global"], rec["frame_global"][keep])
    with pytest.raises(C.CloudError):
        C.Cloud(out, confidence=1.0)
    with pytest.raises(C.CloudError, match="keeps no point"):
        C.Cloud(out, confidence=0.99).stats()


def test_a_floor_transform_is_applied_before_the_turn(tmp_path):
    yaw = np.deg2rad(20.0)
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    out, rec = _scan(tmp_path, floor=(1.5, R, np.array([0.5, -0.2, 3.0])))
    cloud = C.Cloud(out)
    assert not cloud.identity
    display = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64) @ (1.5 * R).T + [0.5, -0.2, 3.0]
    want = np.column_stack([display[:, 0], -display[:, 2], display[:, 1]])
    got = np.vstack([xyz for _rec, xyz in cloud.blocks()])
    assert np.abs(got - want).max() < 1e-9
