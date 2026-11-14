"""THE REAL RESCALE OF A SCAN (USER 2026-11-13: "el reescalado debe ser real, porque hay humano de
referencia, comparacion contra otras nubes, debe generarse la nube real escalada"): a new geometry
epoch in which every point, pose and box is the old one × s, exactly, and everything the epoch does
not carry (volumes, placed objects, meshes, the calibration, the project's composition) follows.
Synthetic (tests/synth_correction.build_scene), no GPU."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import calibration3d as C                                              # noqa: E402
from correction import rescale as RS                                    # noqa: E402
from correction.epoch import current_epoch                              # noqa: E402
from correction.session import load_session, read_poses                 # noqa: E402
from phase_r.instance_store import InstanceStore                        # noqa: E402
from segmentation.pipeline import rebuild_instance_store, write_npz_canonical   # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg     # noqa: E402

S = 1.0375                      # 3.75 % larger


def _glb(path: Path) -> None:
    """A minimal binary glTF: one node, one buffer."""
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"name": "mesh", "translation": [1.0, 2.0, 3.0]}], "buffers": [{"byteLength": 8}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    binary = bytes(range(8))
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def _read_glb(path: Path):
    data = path.read_bytes()
    magic, version, total = struct.unpack("<III", data[:12])
    clen, ctype = struct.unpack("<II", data[12:20])
    doc = json.loads(data[20:20 + clen])
    blen, btype = struct.unpack("<II", data[20 + clen:28 + clen])
    return doc, data[28 + clen:28 + clen + blen], total == len(data), clen % 4 == 0, btype


def _scene(tmp_path):
    scene = build_scene(tmp_path)
    out = scene.output_dir
    # a floor transform with a translation: the display frame is NOT the scan's own
    floor = {"s": np.float64(1.0), "R": np.eye(3), "t": np.array([0.2, -0.35, 0.1])}
    write_npz_canonical(out / "floor_transform.npz", floor)
    # the objects' boxes, as a projection leaves them in the result (display frame)
    from segmentation.pipeline import _compute_obb
    sess = load_session(out)
    res = json.loads((out / "segmentation_result.json").read_text())
    for inst in res["instances"]:
        g = np.asarray(inst["globalIndices"], dtype=np.int64)
        inst["obb"] = _compute_obb(sess.xyz[g] @ floor["R"].T + floor["t"])
    (out / "segmentation_result.json").write_text(json.dumps(res))
    assert rebuild_instance_store(out)                                   # scene_r.db, display frame
    store = InstanceStore(out / "scene_r.db")
    vid = store.add_user_volume("bay", [1.0, 0.5, -2.0], [2.0, 1.0, 0.5], yaw_deg=30.0)
    store.close()
    (out / "scene_objects.json").write_text(json.dumps({"objects": [
        {"id": 1, "name": "human", "url": "/x.glb", "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4.0, 0.0, -1.5, 1]}]}))
    mesh = out / "tsdf" / "genrecon_photo"
    (mesh / "tiles").mkdir(parents=True)
    _glb(mesh / "genrecon_photo.glb")
    (mesh / "genrecon_photo.meta.json").write_text(json.dumps({"bounds_published": [[-1.0, 0.0, 2.0], [3.0, 2.5, 4.0]], "faces": 12}))
    (mesh / "tiles" / "tileset.json").write_text(json.dumps({"asset": {"version": "1.1"}, "geometricError": 4.0, "root": {
        "boundingVolume": {"box": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]}, "geometricError": 2.0,
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5.0, 6.0, 7.0, 1], "children": []}}))
    epoch = int(current_epoch(out))
    doc = C.empty_document()
    C.add(doc, {"a": [0.0, 0.0, 0.0], "b": [2.0, 0.0, 0.0], "value_m": 2.0 * S, "sigma_m": 0.002, "role": "gauge"}, epoch)
    C.add(doc, {"a": [0.0, 0.0, 1.0], "b": [0.0, 1.5, 1.0], "value_m": 1.5 * S, "role": "test"}, epoch)
    C.add(doc, {"a": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0], "value_m": 1.0}, epoch - 1 if epoch else 99)   # another epoch: stale
    C.add_plane(doc, {"points": [[0, 0, 0], [2, 0, 0], [2, 0, 3], [0, 0.01, 3]], "label": "floor"}, epoch)
    C.save(out, doc)
    return scene, out, vid, epoch


def test_a_rescale_is_a_new_epoch_with_everything_at_the_new_scale(tmp_path):
    scene, out, vid, epoch = _scene(tmp_path)
    before = load_session(out)
    xyz0, poses0 = before.xyz.copy(), before.poses.copy()
    raw0 = np.stack([before.raw_data["x"], before.raw_data["y"], before.raw_data["z"]], 1).astype(np.float64)
    res0 = json.loads((out / "segmentation_result.json").read_text())
    store = InstanceStore(out / "scene_r.db")
    iid = int(store.list_instances()[0]["instance_id"])
    pts0, obb0 = store.get_points(iid).copy(), store.get_obb(iid)
    store.close()
    assert obb0 is not None and len(pts0) > 100
    assert C.summary(C.load(out), epoch)["scale_factor"] == pytest.approx(S, rel=1e-9)

    logs = []
    rep = RS.apply_scale(out, S, source={"mode": "global", "factor": S}, cfg=make_correction_cfg(),
                         rebuild_octree=False, log=logs.append)

    # ── the epoch ──
    assert rep["epoch_from"] == epoch and rep["epoch_to"] == epoch + 1 == int(current_epoch(out))
    rec = json.loads((out / "geometry_epoch.json").read_text())
    assert rec["kind"] == "transform" and rec["stage"] == "rescale" and rec["scale"] == S
    after = load_session(out)
    assert after.n_points == before.n_points == rep["points"]            # every point, none deleted
    np.testing.assert_allclose(after.xyz, xyz0 * S, rtol=0, atol=2e-6)   # the cloud's float32
    raw1 = np.stack([after.raw_data["x"], after.raw_data["y"], after.raw_data["z"]], 1).astype(np.float64)
    np.testing.assert_allclose(raw1, raw0 * S, rtol=0, atol=2e-6)
    np.testing.assert_allclose(after.poses[:, :3, 3], poses0[:, :3, 3] * S, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(after.poses[:, :3, :3], poses0[:, :3, :3])        # the cameras do not turn
    # the display frame scales about its own origin: the floor transform's translation × s, re-sealed
    ft = np.load(out / "floor_transform.npz")
    np.testing.assert_allclose(ft["t"], np.array([0.2, -0.35, 0.1]) * S, atol=1e-12)
    np.testing.assert_array_equal(ft["R"], np.eye(3))
    seal = json.loads((out / "floor_level.json").read_text())
    assert "rescale" in seal["leveled_by"] and seal["stamp"]["cloud_sha256"]
    # the boxes: scaled exactly, never refit
    res1 = json.loads((out / "segmentation_result.json").read_text())
    for a, b in zip(res0["instances"], res1["instances"]):
        if "obb" not in a:
            continue
        np.testing.assert_allclose(b["obb"]["center"], np.array(a["obb"]["center"]) * S, rtol=1e-12)
        np.testing.assert_allclose(b["obb"]["half_extents"], np.array(a["obb"]["half_extents"]) * S, rtol=1e-12)
        assert b["obb"]["rotation"] == a["obb"]["rotation"] and b["globalIndices"] == a["globalIndices"]
    # the instance store follows: points and boxes × s in the display frame
    store = InstanceStore(out / "scene_r.db")
    try:
        np.testing.assert_allclose(store.get_points(iid), pts0 * S, rtol=0, atol=3e-6)
        T1, aabb1, pos1 = store.get_obb(iid)[:3]
        np.testing.assert_allclose(np.asarray(T1)[:3, 3], np.asarray(obb0[0])[:3, 3] * S, atol=1e-5)
        np.testing.assert_allclose(aabb1, np.asarray(obb0[1]) * S, atol=1e-5)
        np.testing.assert_allclose(np.asarray(T1)[:3, :3], np.asarray(obb0[0])[:3, :3], atol=1e-7)
        # ── what the epoch does not carry ──
        v = store.get_user_volume(vid)
        np.testing.assert_allclose(v["center"], np.array([1.0, 0.5, -2.0]) * S, atol=1e-6)
        assert v["size"] == [2.0, 1.0, 0.5] and v["yaw_deg"] == 30.0     # the user's own metres
    finally:
        store.close()
    obj = json.loads((out / "scene_objects.json").read_text())["objects"][0]["matrix"]
    assert obj[:12] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] and obj[12:15] == pytest.approx([4.0 * S, 0.0, -1.5 * S])
    gltf, binary, whole, padded, btype = _read_glb(out / "tsdf/genrecon_photo/genrecon_photo.glb")
    assert whole and padded and binary == bytes(range(8)) and btype == 0x004E4942
    assert gltf["scenes"][0]["nodes"] == [1] and gltf["nodes"][1] == {"name": "stac_scale", "scale": [S, S, S], "children": [0]}
    assert gltf["nodes"][0] == {"name": "mesh", "translation": [1.0, 2.0, 3.0]}
    tiles = json.loads((out / "tsdf/genrecon_photo/tiles/tileset.json").read_text())["root"]
    assert tiles["transform"] == pytest.approx([S, 0, 0, 0, 0, S, 0, 0, 0, 0, S, 0, 5.0 * S, 6.0 * S, 7.0 * S, 1])
    assert tiles["geometricError"] == 2.0 and tiles["boundingVolume"]["box"][3] == 1       # local: the transform scales them
    meta = json.loads((out / "tsdf/genrecon_photo/genrecon_photo.meta.json").read_text())
    assert meta["bounds_published"] == [pytest.approx([-S, 0.0, 2 * S]), pytest.approx([3 * S, 2.5 * S, 4 * S])]
    assert meta["geometry_epoch"] == epoch + 1 and meta["faces"] == 12 and meta["rescaled"] == [{"scale": S, "epoch_to": epoch + 1}]
    assert rep["volumes_moved"] == 1 and rep["placed_objects_moved"] == 1 and len(rep["meshes_scaled"]) == 2

    # ── the calibration that was applied still stands, and reads a factor of 1 ──
    doc = C.load(out)
    assert C.validate_document(doc) == []
    live = [m for m in doc["measurements"] if m["geometry_epoch"] == epoch + 1]
    assert len(live) == 2 and live[0]["b"] == pytest.approx([2.0 * S, 0.0, 0.0]) and live[0]["measured_m"] == pytest.approx(2.0 * S)
    assert [m["geometry_epoch"] for m in doc["measurements"]][-1] != epoch + 1          # the stale one stays stale
    assert doc["planes"][0]["geometry_epoch"] == epoch + 1 and doc["planes"][0]["label"] == "floor"
    assert doc["applied"] == [{"scale": S, "mode": "global", "epoch_from": epoch, "epoch_to": epoch + 1}]
    summ = C.summary(doc, epoch + 1)
    assert summ["scale_factor"] == pytest.approx(1.0, abs=1e-9)
    assert rep["calibration"] == {"measurements": 2, "planes": 1}
    # the report and the ledger
    report = json.loads((out / "corrections" / f"report_{rep['correction_id']}.json").read_text())
    assert report["scale"] == S and report["undo"].startswith("apply the scale 0.96385")
    assert any(json.loads(line)["kind"] == "scale" for line in (out / "corrections.jsonl").read_text().splitlines())

    # ── UNDO: the inverse scale, another epoch — everything comes back ──
    back = RS.apply_scale(out, 1.0 / S, cfg=make_correction_cfg(), rebuild_octree=False, log=logs.append)
    assert back["epoch_to"] == epoch + 2
    again = load_session(out)
    np.testing.assert_allclose(again.xyz, xyz0, rtol=0, atol=4e-6)
    np.testing.assert_allclose(again.poses[:, :3, 3], poses0[:, :3, 3], atol=1e-9)
    np.testing.assert_allclose(np.load(out / "floor_transform.npz")["t"], [0.2, -0.35, 0.1], atol=1e-12)
    gltf, *_ = _read_glb(out / "tsdf/genrecon_photo/genrecon_photo.glb")
    assert gltf["nodes"][1]["scale"] == pytest.approx([1.0, 1.0, 1.0]) and len(gltf["nodes"]) == 2    # one wrapper, multiplied
    assert json.loads((out / "scene_objects.json").read_text())["objects"][0]["matrix"][12] == pytest.approx(4.0)


def test_the_rescale_refuses_what_is_no_scale(tmp_path):
    scene = build_scene(tmp_path)
    for bad in (0.0, -1.0, float("nan"), float("inf"), 1.0):
        with pytest.raises(RS.RescaleError):
            RS.apply_scale(scene.output_dir, bad, cfg=make_correction_cfg(), log=lambda *_: None)
    assert int(current_epoch(scene.output_dir)) == 0                     # nothing was staged


def test_the_per_keyframe_form_of_a_uniform_scale_is_the_scale(tmp_path):
    """The arrays handed to the epoch transaction (they move the poses and the findings) are
    x → s·x exactly: the transaction's own warp on them gives the same cloud."""
    from correction.apply import transform_poses, warp_full_cloud_numpy
    scene = build_scene(tmp_path)
    sess = load_session(scene.output_dir)
    R, t, k = RS.uniform_scale_arrays(sess, S)
    xyz, _n = warp_full_cloud_numpy(sess, R, t, k)
    has_kf = sess.ks >= 0
    np.testing.assert_allclose(xyz[has_kf], sess.xyz[has_kf] * S, rtol=0, atol=1e-12)
    poses = transform_poses(sess.poses, R, t)
    np.testing.assert_allclose(poses[:, :3, 3], sess.poses[:, :3, 3] * S, atol=1e-12)
    np.testing.assert_array_equal(poses[:, :3, :3], sess.poses[:, :3, :3])


def test_the_projects_composition_follows_a_rescaled_scan(tmp_path):
    class _Paths:
        def __init__(self):
            self.meta = {"scans": [{"key": "a/default"}, {"key": "b/default"}, {"key": "c/default"}], "composition": {
                "reference": "a/default", "reference_scale": 1.02, "transforms": {
                    "b/default": {"matrix": (np.diag([1.04, 1.04, 1.04, 1.0]) + np.array([[0, 0, 0, 2.0], [0] * 4, [0] * 4, [0] * 4])).reshape(-1).tolist(),
                                  "method": "objects", "fused": "fused/x"},
                    "c/default": {"matrix": np.eye(4).reshape(-1).tolist(), "method": "manual"}}}}

        def save_project_meta(self, meta):
            self.meta = json.loads(json.dumps(meta))
    import project_scans
    paths = _Paths()
    orig = project_scans.load_meta
    project_scans.load_meta = lambda p: p.meta
    try:
        # a non-reference scan grows by 4 %: its transform to the reference loses that 4 %
        out = RS.rescale_composition(paths, "b/default", 1.04)
        M = np.array(paths.meta["composition"]["transforms"]["b/default"]["matrix"]).reshape(4, 4)
        np.testing.assert_allclose(M[:3, :3], np.eye(3), atol=1e-12)
        assert M[0, 3] == 2.0 and paths.meta["composition"]["transforms"]["b/default"]["scale_scan_to_ref"] == pytest.approx(1.0)
        assert out == {"transforms": ["b/default"], "fused_to_redo": ["fused/x"]}
        assert paths.meta["composition"]["transforms"]["c/default"]["matrix"] == np.eye(4).reshape(-1).tolist()
        # the reference shrinks by 2 %: every other scan's transform lands in the smaller frame
        RS.rescale_composition(paths, "a/default", 0.98)
        for key in ("b/default", "c/default"):
            M = np.array(paths.meta["composition"]["transforms"][key]["matrix"]).reshape(4, 4)
            np.testing.assert_allclose(M[:3, :3], 0.98 * np.eye(3), atol=1e-12)
        assert np.array(paths.meta["composition"]["transforms"]["b/default"]["matrix"]).reshape(4, 4)[0, 3] == pytest.approx(1.96)
        assert RS.rescale_composition(paths, "zzz/default", 2.0) is None        # a scan with no transform: nothing
    finally:
        project_scans.load_meta = orig


def test_which_calibration_may_be_applied_to_the_geometry():
    """Only a GLOBAL factor (USER 2026-11-14, on applying the per-axis factors to the geometry:
    "si, me parece" — to: only the global factor is applied, per axis stays a diagnosis)."""
    import calibration_api as A
    ok = {"state": "verified", "scale_factor": 1.0375, "mode_used": "global"}
    assert A.apply_refusal(ok) is None
    assert A.apply_refusal({**ok, "state": "calibrated"}) is None
    assert "no calibration measurement" in A.apply_refusal({"state": "uncalibrated", "scale_factor": None, "mode_used": "global"})
    assert "per axis" in A.apply_refusal({**ok, "mode_used": "axes"}) and "GLOBAL" in A.apply_refusal({**ok, "mode_used": "axes"})
    assert "exactly 1" in A.apply_refusal({**ok, "scale_factor": 1.0})
    assert "not a scale" in A.apply_refusal({**ok, "scale_factor": 0.0})
    assert "not a scale" in A.apply_refusal({**ok, "scale_factor": float("nan")})
