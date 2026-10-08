"""The mask→cloud projection is a PURE function of its inputs (docs/plan_determinismo.md
points 99, 100, 107, 108, 120, 121, 122, 123 — 2026-10-08), on a rendered scene: a plate
10 cm in front of a wall, seen by seven keyframes of the SESSION camera (camera.json),
the cloud carrying every point's birth keyframe and pixel on the declared record grid.

* two projections give the same bytes — the result, the class byte, the 16-bit ids, the
  audit, its marks, the store, the fusion map — and the raw mask store is untouched;
* the result carries its stamp; a result whose stamp matches is reused and the octree is
  not rebuilt; a changed input makes it stale;
* a cloud frame that is not a keyframe, an undeclared grid, an empty projection FAIL and
  delete the previous result — nothing of another state survives.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import mask_space                                   # noqa: E402
from segmentation import pipeline as P                                # noqa: E402

W, H = 120, 90
K_SESSION = (100.0, 99.0, 60.2, 44.8)
CAMS_X = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)
FRAMES = [10 * (k + 1) for k in range(len(CAMS_X))]
Z_PLATE, Z_WALL = 1.4, 1.5
HALF_PLATE, WALL_X, WALL_Y = 0.2, 1.5, 1.0

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                      ("blue", "u1"), ("frame_global", "<i4"), ("pixel_row", "<u2"),
                      ("pixel_col", "<u2"), ("confidence", "<f4")])


def _poses():
    Pz = np.tile(np.eye(4), (len(CAMS_X), 1, 1))
    Pz[:, 0, 3] = CAMS_X
    return Pz


def _render(k):
    fx, fy, cx, cy = K_SESSION
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = (u - cx) / fx, (v - cy) / fy
    c = CAMS_X[k]
    plate = (np.abs(c + dx * Z_PLATE) <= HALF_PLATE) & (np.abs(dy * Z_PLATE) <= HALF_PLATE)
    wall = ~plate & (np.abs(c + dx * Z_WALL) <= WALL_X) & (np.abs(dy * Z_WALL) <= WALL_Y)
    depth = np.where(plate, Z_PLATE, np.where(wall, Z_WALL, 0.0))
    label = np.where(plate, 0, np.where(wall, 1, -1))
    return depth, label


def _write_ply(path, data):
    names = {"red": "red", "green": "green", "blue": "blue"}
    types = {"<f4": "float", "|u1": "uchar", "<i4": "int", "<u2": "ushort"}
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    head += [f"property {types[data.dtype[n].str]} {names.get(n, n)}" for n in data.dtype.names]
    head += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode())
        f.write(data.tobytes())


def _session(tmp_path, with_camera=True):
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    out = tmp_path / "output"
    out.mkdir(parents=True)
    poses = _poses()
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in FRAMES) + "\n")
    (out / "camera_poses.txt").write_text("\n".join(
        " ".join(repr(float(x)) for x in p.ravel()) for p in poses) + "\n")
    (out / ".orientation_applied").write_text("")
    if with_camera:
        g = grid_full_frame_resize(W, H, W, H, "omega")
        cam = CameraModel(width=W, height=H, params=(*K_SESSION, 0.0, 0.0, 0.0, 0.0),
                          source="refine", camera_epoch=1, omega_grid=g,
                          mask_grid=grid_full_frame_resize(W, H, W, H, "mask"))
        save_camera_json(out / "camera.json", cam, geometry_epoch=1)
    masks = {mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME),
             mask_space.KEYFRAMES_KEY: np.asarray(FRAMES, np.int32),
             "obj_ids": np.array([0, 1], np.int32),
             "frames": np.arange(len(FRAMES), dtype=np.int32),
             "scaled_res": np.array([H, W], np.int32)}
    rows = []
    fx, fy, cx, cy = K_SESSION
    n_plate = 0
    for k in range(len(CAMS_X)):
        depth, label = _render(k)
        masks[f"f{k}_o0"] = (label == 0).astype(np.uint8)
        masks[f"f{k}_o1"] = (label == 1).astype(np.uint8)
        v, u = np.nonzero(depth > 0)
        z = depth[v, u]
        Pw = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1) + poses[k][:3, 3]
        d = np.zeros(len(u), PLY_DTYPE)
        d["x"], d["y"], d["z"] = Pw[:, 0], Pw[:, 1], Pw[:, 2]
        d["red"] = 100; d["green"] = 120; d["blue"] = 140
        d["frame_global"] = FRAMES[k]
        d["pixel_row"], d["pixel_col"] = v, u
        d["confidence"] = 0.9
        rows.append(d)
        n_plate += int((label[v, u] == 0).sum())
    data = np.concatenate(rows)
    _write_ply(out / "cleaned_cloud.ply", data)
    P.write_canonical_store(out / "seg_masks.npz", masks)
    (out / "segmentation.json").write_text(json.dumps({
        "version": "3.0", "prompt": "monitor;wall", "prompts": ["monitor", "wall"],
        "resolution": {"scaled": [H, W], "original": [H, W]}, "mask_file": "seg_masks.npz",
        "instances": [{"id": 0, "instance_id": 1, "label": "monitor", "color": [1, 2, 3]},
                      {"id": 1, "instance_id": 2, "label": "wall", "color": [4, 5, 6]}]}, indent=1))
    return out, data, n_plate


PRODUCTS = ("segmentation_result.json", "classification.npy", "instance_ids.npy", "class_map.json",
            "mask_audit.json", "out_of_place.npy", P.OUT_OF_PLACE_VIEWS_NAME, "scene_r.db",
            P.POTREE_STAMP_NAME)


@pytest.fixture
def fake_potree(monkeypatch):
    import potree_converter as PC
    calls = []

    def _convert(session_dir, force=False, ply_override=None, potree_dir_override=None):
        calls.append(str(ply_override))
        d = Path(session_dir) / "output" / "potree"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text("{}")
        return True

    monkeypatch.setattr(PC, "convert_ply_to_potree", _convert)
    return calls


def _bytes(out):
    return {n: (out / n).read_bytes() for n in PRODUCTS}


def test_two_projections_write_the_same_bytes_and_leave_the_raw_store_alone(tmp_path, fake_potree):
    out, data, n_plate = _session(tmp_path)
    raw = [(out / n).read_bytes() for n in ("seg_masks.npz", "segmentation.json")]
    res1 = P._match_and_save_result(out)
    first = _bytes(out)
    res2 = P._match_and_save_result(out)
    assert _bytes(out) == first
    assert [(out / n).read_bytes() for n in ("seg_masks.npz", "segmentation.json")] == raw   # point 100
    assert not (out / "_sam3_raw").exists() and not (out / "corrected_cloud.ply").exists()
    by = {i["instance_id"]: i for i in res1["instances"]}
    assert sorted(by) == [1, 2]
    assert by[1]["total_points"] == n_plate, "every plate point on its mask through the exact grid maps"
    assert by[1]["min_points_margin"] == n_plate - 1000                                        # point 116
    ids = np.load(out / "instance_ids.npy")
    assert ids.dtype == np.uint16 and set(ids.tolist()) == {1, 2}                               # point 107
    assert np.array_equal(np.load(out / "classification.npy"), ids.astype(np.uint8))
    assert res1["decisions"]["error_factor"] == 1.1 and res1["decisions"]["min_judges"] == 5
    assert res1["stamp"]["inputs"]["cloud"] and "server/segmentation/pipeline.py" in res1["stamp"]["code"]
    assert fake_potree == [str(out / "cleaned_cloud.ply")], "the octree once, from the cloud projected on"
    audit = json.loads((out / "mask_audit.json").read_text())
    assert "geometry_epoch" not in audit and "reconstruction_id" in audit                      # point 118
    assert res2["stamp"] == res1["stamp"]


def test_a_result_with_an_identical_stamp_is_reused_and_a_changed_input_is_stale(tmp_path, fake_potree,
                                                                                 monkeypatch):
    out, data, _ = _session(tmp_path)
    P.map_segmentation_to_cloud(out)
    assert P.segmentation_result_is_stale(out) == (False, "identical stamp (inputs, code and configuration)")
    monkeypatch.setattr(P, "_match_and_save_result",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-projected")))
    res = P.map_segmentation_to_cloud(out)
    assert len(res["instances"]) == 2 and len(fake_potree) == 1
    # the cloud changes (one more point): stale, naming the input
    _write_ply(out / "cleaned_cloud.ply", np.concatenate([data, data[:1]]))
    stale, why = P.segmentation_result_is_stale(out)
    assert stale and "input 'cloud' changed" in why
    # a result with no stamp (written before 2026-10-08) is stale
    doc = json.loads((out / "segmentation_result.json").read_text())
    doc.pop("stamp")
    (out / "segmentation_result.json").write_text(json.dumps(doc))
    assert P.segmentation_result_is_stale(out)[0]


def test_a_cloud_frame_that_is_not_a_keyframe_fails_and_the_previous_result_goes(tmp_path, fake_potree):
    out, data, _ = _session(tmp_path)
    P._match_and_save_result(out)
    assert (out / "segmentation_result.json").exists()
    bad = data.copy()
    bad["frame_global"][:5] = 3                     # a video number that is also a valid position
    _write_ply(out / "cleaned_cloud.ply", bad)
    with pytest.raises(RuntimeError, match="cloud frame 3 is not in camera_frames.txt"):
        P._match_and_save_result(out)
    assert not (out / "segmentation_result.json").exists(), "no result of another state survives (99)"


def test_an_undeclared_grid_fails_never_a_guess(tmp_path, fake_potree):
    from precision.camera import CameraError
    out, _, _ = _session(tmp_path, with_camera=False)
    with pytest.raises((CameraError, RuntimeError)):
        P._match_and_save_result(out)
    assert not (out / "segmentation_result.json").exists()


def test_a_store_of_another_keyframe_list_is_refused_by_the_projection(tmp_path, fake_potree):
    out, _, _ = _session(tmp_path)
    (out / "camera_frames.txt").write_text("\n".join(str(f + 1) for f in FRAMES) + "\n")
    with pytest.raises(mask_space.KeyframeListMismatch):
        P._match_and_save_result(out)


def test_a_configured_audit_that_cannot_run_fails_the_projection(tmp_path, fake_potree, monkeypatch):
    out, _, _ = _session(tmp_path)
    from reconstruction.surface_fit import hole_audit as HA

    class _NoEv:
        ok = False
        reason = "synthetic: no evidence"
        masks = None

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(HA, "_Evidence", _NoEv)
    HA._EVIDENCE_CACHE.clear()
    with pytest.raises(RuntimeError, match="mask audit: no mask/camera evidence"):
        P._match_and_save_result(out)
