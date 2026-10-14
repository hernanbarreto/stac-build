"""The closing of wave 2 of docs/plan_determinismo.md for the cloud + segmentation package
(2026-10-08): 96 (every product of a previous projection goes on EVERY path of the cloud
stage, one list with the pipeline manager), 103 / 57 (one converter thread, pinned in code and
recorded in the octree's stamp), 108 (the mask audit addresses the masks through the EXACT
grid maps, never a scale), 109 (the footprint error of an OBB candidate is MEASURED by a
seeded Poisson bootstrap), 112 (one configuration for the whole SAM3 stage), 122 (a projection
that fails leaves no product), 123 (the octree is reused only on an identical stamp — no file
date), 158 (floor_transform.npz applies only with the seal of this geometry), and the store
seal (a store caught half way through the SAM3 stage's swap is refused). Every test drives the
function the pipeline calls."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import mask_space                                   # noqa: E402
from segmentation import pipeline as P                                # noqa: E402
from segmentation import republish as RP                              # noqa: E402
import potree_converter as PC                                         # noqa: E402

from test_p3_projection import FRAMES, PLY_DTYPE, _session, _write_ply, fake_potree  # noqa: E402,F401


# ── 96: one list of projection products, removed on every path of the cloud stage ───────

def test_the_projection_products_are_one_list_with_the_pipeline_manager():
    from pipeline_manager import PipelineManager, StageId
    assert set(RP.PROJECTION_PRODUCTS) == set(PipelineManager.STAGE_OUTPUT_FILES[StageId.PROJECTION])
    for name in ("instance_ids.npy", "fusion_map.json", "potree_stamp.json", "mask_audit.json",
                 "out_of_place.npy", "out_of_place_views.npz", "scene_r.db"):
        assert name in RP.PROJECTION_PRODUCTS


class _Pipe:
    def __init__(self):
        self.lines = []

    def send_log(self, msg, level=None):
        self.lines.append(str(msg))

    def send_progress(self, pct, msg, stage=None):
        pass

    def check_cancel(self):
        return False


def test_the_cloud_stage_removes_the_previous_projection_products_on_the_light_path(tmp_path):
    """Production always takes the light path (the precision core published the cloud): the
    cleanup used to sit in the merge branch and the previous run's result, class arrays, store
    and fusion map stayed beside the new cloud."""
    from workers.cloudcompy_worker import _cloudcompy_work
    session = tmp_path / "scan"
    out = session / "output"
    out.mkdir(parents=True)
    (out / "cleaned_cloud.ply").write_bytes(b"ply\n")
    (out / ".orientation_applied").write_text("")
    for name in RP.PROJECTION_PRODUCTS + ("segmentation.json", "seg_masks.npz"):
        (out / name).write_text("x")
    pipe = _Pipe()
    _cloudcompy_work(pipe, str(session), {"postprocessing": {"build_potree": False}})
    assert not any((out / n).exists() for n in RP.PROJECTION_PRODUCTS)
    assert (out / "segmentation.json").exists() and (out / "seg_masks.npz").exists(), \
        "the raw mask store is the SAM3 stage's, not a projection product"
    assert any("previous mask→cloud projection removed" in m for m in pipe.lines)
    assert any("not projected here" in m for m in pipe.lines)


# ── 123 / 103 / 57: the octree is reused only on an identical stamp; one thread, recorded ──

def _fake_converter(calls):
    def fake_converter(las_path, potree_dir):
        calls.append(str(las_path))
        Path(potree_dir).mkdir(parents=True)
        (Path(potree_dir) / "metadata.json").write_text("{}")
        return True
    return fake_converter


def test_the_octree_is_reused_only_on_an_identical_stamp_never_by_date(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    ply = out / "cleaned_cloud.ply"
    ply.write_bytes(b"cloud A")
    calls = []
    monkeypatch.setattr(PC, "_ply_to_las", lambda p, l: 10)
    monkeypatch.setattr(PC, "_run_potree_converter", _fake_converter(calls))
    assert PC.convert_ply_to_potree(tmp_path, force=False) and len(calls) == 1
    # the stamp lives INSIDE the octree it describes, written before the swap
    stamp_path = out / "potree" / PC.OCTREE_STAMP_NAME
    assert PC.potree_stamp_path(out / "potree") == stamp_path and stamp_path.is_file()
    assert not (out / PC.POTREE_STAMP_NAME).exists(), "a build writes no record beside the octree"
    stamp = json.loads(stamp_path.read_text())
    assert stamp["threads"] == PC.POTREE_THREADS == "1" and stamp["encoding"] == "UNCOMPRESSED"
    assert stamp["classification_sha256"] is None and stamp["instance_ids_sha256"] is None
    assert stamp["cloud_sha256"] == PC.potree_stamp_for(ply)["cloud_sha256"]
    # the same inputs: reused, whatever the file dates say
    assert PC.convert_ply_to_potree(tmp_path, force=False) and len(calls) == 1
    later = os.stat(out / "potree" / "metadata.json").st_mtime + 3600
    os.utime(ply, (later, later))                          # the PLY "newer" than the octree
    assert PC.convert_ply_to_potree(tmp_path, force=False) and len(calls) == 1
    assert PC.octree_matches(out / "potree", ply)[0]
    # the identity travels with the directory (a transaction swap, select_epoch, an epoch store)
    (out / "potree").rename(out / "_tx_epoch_1")
    assert PC.octree_matches(out / "_tx_epoch_1", ply)[0]
    (out / "_tx_epoch_1").rename(out / "potree")
    # other cloud bytes: rebuilt
    ply.write_bytes(b"cloud B")
    ok, why = PC.octree_matches(out / "potree", ply)
    assert not ok and "cloud_sha256" in why
    assert PC.convert_ply_to_potree(tmp_path, force=False) and len(calls) == 2
    # class arrays beside the cloud: another octree
    np.save(out / "classification.npy", np.zeros(3, np.uint8))
    assert PC.convert_ply_to_potree(tmp_path, force=False) and len(calls) == 3
    assert json.loads(stamp_path.read_text())["classification_sha256"]
    # a chunk preview, or no stamp at all, is never reused
    (out / "potree" / PC.PREVIEW_MARKER).write_text("preview")
    assert PC.octree_matches(out / "potree", ply) == (False, "the octree is a preview of the raw chunks")
    (out / "potree" / PC.PREVIEW_MARKER).unlink()
    stamp_path.unlink()
    assert not PC.octree_matches(out / "potree", ply)[0]


def test_the_cloud_stage_reuses_the_published_octree_on_the_light_path(tmp_path, monkeypatch):
    """Production: f6_bend builds the octree of the cloud it publishes INSIDE its transaction
    and the swap brings it to output/potree with its stamp inside; the cloud stage then
    removes the previous projection's products (its record among them) and must still
    recognise that octree — the stamp beside the directory was discarded by the swap and the
    stage rebuilt every octree (zaragoza 114 M points ≈ 4 min)."""
    from workers.cloudcompy_worker import _cloudcompy_work
    session = tmp_path / "scan"
    out = session / "output"
    tx = out / "_tx_epoch_0"
    tx.mkdir(parents=True)
    (tx / "cleaned_cloud.ply").write_bytes(b"the published cloud")
    calls = []
    monkeypatch.setattr(PC, "_ply_to_las", lambda p, l, *a: 10)
    monkeypatch.setattr(PC, "_run_potree_converter", _fake_converter(calls))
    assert PC.convert_ply_to_potree(session, force=True, ply_override=tx / "cleaned_cloud.ply",
                                    potree_dir_override=tx / "potree")
    assert len(calls) == 1 and (tx / "potree" / PC.OCTREE_STAMP_NAME).is_file()
    (tx / "cleaned_cloud.ply").rename(out / "cleaned_cloud.ply")      # the transaction swap
    (tx / "potree").rename(out / "potree")
    (out / ".orientation_applied").write_text("")
    for name in RP.PROJECTION_PRODUCTS:                                # a previous projection
        (out / name).write_text("x")
    pipe = _Pipe()
    _cloudcompy_work(pipe, str(session), {"postprocessing": {"build_potree": True}})
    assert len(calls) == 1, "the published octree is reused, not rebuilt"
    assert any("Potree octree reused" in m for m in pipe.lines)
    assert not (out / PC.POTREE_STAMP_NAME).exists() and (out / "potree" / PC.OCTREE_STAMP_NAME).is_file()
    # the next projection bakes class arrays in: a different octree, rebuilt once
    np.save(out / "classification.npy", np.zeros(3, np.uint8))
    assert not PC.octree_matches(out / "potree", out / "cleaned_cloud.ply")[0]


# ── 122: a projection that fails leaves NO product; the products are staged ─────────────

def test_a_failed_projection_leaves_no_product_and_a_finished_one_no_staging(tmp_path, fake_potree,
                                                                                monkeypatch):
    out, _data, _ = _session(tmp_path)
    res = P._match_and_save_result(out)
    assert not (out / P.PROJECTION_STAGING_DIRNAME).exists()
    # the synthetic store is written by hand, unsealed — the interactive manager's state:
    # projected, and the edit declared in the result
    seal = res["decisions"]["store_seal"]
    assert seal["intact"] is False and "carries no seal" in seal["declared"][0]
    assert res["decisions"]["display_frame"] == "identity (.orientation_applied)"
    on_disk = [n for n in RP.PROJECTION_PRODUCTS if (out / n).exists()]
    assert {"segmentation_result.json", "classification.npy", "instance_ids.npy", "class_map.json",
            "scene_r.db", "mask_audit.json", "out_of_place.npy", RP.OUT_OF_PLACE_VIEWS,
            RP.POTREE_STAMP} <= set(on_disk)
    (out / "seg_broadcast.json").write_text("{}")           # the manager's snapshot of the result

    def boom(*a, **k):
        raise OSError(28, "No space left on device")        # the publish, the last step

    monkeypatch.setattr(P, "_publish_staged_products", boom)
    with pytest.raises(OSError):
        P._match_and_save_result(out)
    assert not any((out / n).exists() for n in RP.PROJECTION_PRODUCTS), \
        [n for n in RP.PROJECTION_PRODUCTS if (out / n).exists()]
    assert not (out / P.PROJECTION_STAGING_DIRNAME).exists()
    assert (out / "seg_masks.npz").exists() and (out / "segmentation.json").exists()


# ── the store seal: a swap caught half way is refused; an edit after the seal is declared ──

def test_a_half_swapped_store_is_refused_by_the_projection(tmp_path, fake_potree):
    out, _, _ = _session(tmp_path)
    (out / P.SAM3_SWAP_MARKER).write_text("replacing\n")
    with pytest.raises(RuntimeError, match="swap did not complete"):
        P._match_and_save_result(out)
    assert not (out / "segmentation_result.json").exists()
    (out / P.SAM3_SWAP_MARKER).unlink()
    (out / "segmentation.json").unlink()
    with pytest.raises(RuntimeError, match="no segmentation.json"):
        P._match_and_save_result(out)


def test_a_seal_naming_other_bytes_is_refused_and_no_seal_is_declared(tmp_path, fake_potree):
    """A seal that lies — segmentation.json names mask bytes that are not on disk (erase /
    delete rewrote the npz and kept the stamp) — is REFUSED on every path, as the
    certification refuses it; a store with NO seal (the interactive manager's upsert, a store
    written before the seal existed) is projected as it is, declared."""
    out, _, _ = _session(tmp_path)
    doc = json.loads((out / "segmentation.json").read_text())
    doc["stamp"] = {"sha256": "x", "inputs": {"seg_masks.npz": "0" * 64}}
    (out / "segmentation.json").write_text(json.dumps(doc))
    fatal, declared = P.store_seal_verdict(out)
    assert declared == [] and len(fatal) == 1 and "changed since it was sealed" in fatal[0]
    with pytest.raises(RuntimeError, match="changed since it was sealed"):
        P._match_and_save_result(out)
    assert not (out / "segmentation_result.json").exists()
    del doc["stamp"]
    (out / "segmentation.json").write_text(json.dumps(doc))                     # the manager's upsert
    fatal, declared = P.store_seal_verdict(out)
    assert fatal == [] and "carries no seal" in declared[0]
    res = P._match_and_save_result(out)
    assert res["decisions"]["store_seal"] == {"intact": False, "declared": declared}
    (out / "seg_masks.npz").unlink()
    assert P.store_seal_verdict(out)[0] == ["seg_masks.npz is missing"]


def test_a_marker_left_by_a_completed_swap_is_removed_and_the_store_accepted(tmp_path, monkeypatch):
    """A kill between the last rename and the marker's removal leaves a COMPLETE, sealed
    store under the marker; refusing it would re-run SAM3's hours for nothing. The old seal
    goes first in the swap, so a sealed segmentation.json naming the npz on disk can only
    be the new store's."""
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: object())
    out = tmp_path / "output"
    out.mkdir()
    cfg = {"models": {"segmentation": {"batch_size": 1}}, "segmentation": {}}
    staging = out / P.SAM3_STAGING_DIRNAME
    _staged_store(staging)
    P.finalize_sam3_store(staging, out, [0], ["a"], cfg, log=lambda m: None)
    (out / P.SAM3_SWAP_MARKER).write_text("replacing\n")                        # the kill
    logs = []
    assert P.check_store_seal(out, log=logs.append) == []
    assert not (out / P.SAM3_SWAP_MARKER).exists() and any("had completed" in m for m in logs)
    # the same marker beside a store whose seal names OTHER bytes: halves of two runs, refused
    (out / P.SAM3_SWAP_MARKER).write_text("replacing\n")
    (out / "seg_masks.npz").write_bytes(b"another run's masks")
    fatal, _ = P.store_seal_verdict(out)
    assert fatal and "swap did not complete" in fatal[0] and (out / P.SAM3_SWAP_MARKER).exists()


def _staged_store(staging: Path):
    staging.mkdir(parents=True)
    members = {"f0_o0": np.ones((4, 4), np.uint8), "obj_ids": np.array([0], np.int32),
               "frames": np.array([0], np.int32), "scaled_res": np.array([4, 4], np.int32),
               mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME)}
    P.write_canonical_store(staging / "seg_masks.npz", members)
    (staging / "segmentation.json").write_text(json.dumps(
        {"version": "3.0", "prompts": ["a"], "mask_file": "seg_masks.npz",
         "instances": [{"id": 0, "instance_id": 1, "label": "a", "color": [1, 1, 1]}]}))


def test_the_swap_deletes_the_old_seal_first_and_a_kill_leaves_a_refused_store(tmp_path, monkeypatch):
    """A kill between the two renames used to leave the NEW npz beside the OLD
    segmentation.json — a pair nothing refused."""
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: object())
    out = tmp_path / "output"
    out.mkdir()
    (out / "seg_masks.npz").write_bytes(b"old masks")
    (out / "segmentation.json").write_text(json.dumps({"instances": [], "stamp": {"sha256": "old"}}))
    (out / "segmentation_result.json").write_text("{}")
    cfg = {"models": {"segmentation": {"batch_size": 1}}, "segmentation": {}}
    staging = out / P.SAM3_STAGING_DIRNAME
    _staged_store(staging)
    real = os.replace

    def killed(src, dst, *a, **k):
        if Path(dst) == out / "seg_masks.npz":              # the move INTO the session
            raise OSError("killed between the renames")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(P.os, "replace", killed)
    with pytest.raises(OSError, match="killed"):
        P.finalize_sam3_store(staging, out, [0], ["a"], cfg, log=lambda m: None)
    assert not (out / "segmentation.json").exists(), "the old seal goes BEFORE the masks move"
    assert (out / P.SAM3_SWAP_MARKER).exists()
    assert not (out / "segmentation_result.json").exists(), "the previous store's products go with it"
    fatal, _ = P.store_seal_verdict(out)
    assert fatal and "swap did not complete" in fatal[0]
    # the swap that completes: sealed, marker gone, the seal intact
    monkeypatch.setattr(P.os, "replace", real)
    _staged_store(staging) if not staging.exists() else None
    P.finalize_sam3_store(staging, out, [0], ["a"], cfg, log=lambda m: None)
    assert not (out / P.SAM3_SWAP_MARKER).exists()
    assert P.check_store_seal(out) == [] and P.store_seal_verdict(out) == ([], [])


# ── 158: floor_transform.npz only with the seal of THIS reconstruction and THIS cloud ────

def _rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _name_reconstruction(out):
    """A reconstruction to name — the seal needs its identity."""
    from correction.epoch import reconstruction_id_or_none
    rec = out / "omega_run" / "results_output"
    rec.mkdir(parents=True, exist_ok=True)
    np.savez(rec / "frame_000000.npz", depth=np.zeros((2, 2), np.float32))
    rid = reconstruction_id_or_none(out)
    assert rid
    return rid


def test_floor_transform_applies_only_with_the_seal_of_this_geometry(tmp_path, fake_potree):
    """An npz on disk is the frame the VIEWER shows the cloud in; a stage applies it only
    with the seal of this reconstruction and this cloud, and an npz it cannot verify FAILS
    the projection naming the file — projecting in another frame would put every box off by
    the leveling delta (2026-08-28), and an unverifiable input never becomes a default."""
    from repro import sha256_file
    out, _, _ = _session(tmp_path)
    plain = P._match_and_save_result(out)
    assert plain["decisions"]["display_frame"] == "identity (.orientation_applied)"
    c0 = np.asarray(next(i for i in plain["instances"] if i["instance_id"] == 1)["obb"]["center"])
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(30.0), t=np.array([0.0, 0.5, 0.0]))
    # unsealed (no floor_level.json): the projection FAILS naming the file, nothing published
    with pytest.raises(RuntimeError, match=r"floor_transform\.npz cannot be applied .* no floor_level\.json"):
        P._match_and_save_result(out)
    assert not (out / "segmentation_result.json").exists()
    rid = _name_reconstruction(out)
    seal = {"reconstruction_id": rid, "cloud_sha256": sha256_file(out / "cleaned_cloud.ply")}
    (out / "floor_level.json").write_text(json.dumps({"selected_instance_id": 2, "stamp": seal}))
    res = P._match_and_save_result(out)
    assert res["decisions"]["display_frame"].startswith("floor_transform.npz (sealed for this")
    c1 = np.asarray(next(i for i in res["instances"] if i["instance_id"] == 1)["obb"]["center"])
    assert np.allclose(c1, _rot_y(30.0) @ c0 + [0.0, 0.5, 0.0], atol=1e-6), "the OBBs sit in the levelled frame"
    # the seal of ANOTHER cloud, or another reconstruction: refused again, naming which
    (out / "floor_level.json").write_text(json.dumps({"stamp": {**seal, "cloud_sha256": "0" * 64}}))
    with pytest.raises(RuntimeError, match="the seal names cloud 000000000000"):
        P._match_and_save_result(out)
    (out / "floor_level.json").write_text(json.dumps({"stamp": {**seal, "reconstruction_id": "f" * 64}}))
    assert P._floor_seal_status(out, out / "cleaned_cloud.ply")[0] is False
    # a seal that names the npz's own bytes (the cloud stage's) is held to them
    full = {**seal, "floor_transform_sha256": sha256_file(out / "floor_transform.npz")}
    (out / "floor_level.json").write_text(json.dumps({"stamp": full}))
    assert P._floor_seal_status(out, out / "cleaned_cloud.ply")[0] is True
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(31.0), t=np.array([0.0, 0.5, 0.0]))
    ok, why = P._floor_seal_status(out, out / "cleaned_cloud.ply")
    assert ok is False and "rewritten after it was sealed" in why
    assert "floor_level.json" in P.projection_stamp(out)["inputs"], "the seal is an input of the stamp"


def test_the_cloud_stage_seals_the_leveling_it_writes_and_keeps_only_a_sealed_one(tmp_path, monkeypatch):
    """The stage that makes the transform is the one that seals it: floor_level.json beside
    the npz with the reconstruction id, the cloud's sha256 and the npz's own; on a light
    resume a transform sealed for this cloud is kept, an unsealed one (the gizmo, a session
    levelled before the seal) is replaced by the seeded leveling and the replacement declared
    — the next projection would otherwise FAIL on it."""
    import alignment_manager as AM
    from workers.cloudcompy_worker import _cloudcompy_work
    session = tmp_path / "scan"
    out = session / "output"
    out.mkdir(parents=True)
    d = np.zeros(3, PLY_DTYPE)
    d["x"], d["y"], d["z"] = [0.0, 1.0, 2.0], [0.0, 0.1, 0.2], [1.0, 1.0, 1.0]
    _write_ply(out / "cleaned_cloud.ply", d)
    (out / "camera_frames.txt").write_text("10\n")
    rid = _name_reconstruction(out)

    class _Leveling:
        def compute_leveling_from_points(self, xyz):
            return 1.0, _rot_y(10.0), np.array([0.0, 0.3, 0.0])

    monkeypatch.setattr(AM, "get_alignment_manager", lambda: _Leveling())
    pipe = _Pipe()
    _cloudcompy_work(pipe, str(session), {"postprocessing": {"build_potree": False}})
    fl = json.loads((out / "floor_level.json").read_text())
    from repro import sha256_file
    assert fl["stamp"] == {"reconstruction_id": rid,
                           "cloud_sha256": sha256_file(out / "cleaned_cloud.ply"),
                           "floor_transform_sha256": sha256_file(out / "floor_transform.npz")}
    assert P._floor_seal_status(out, out / "cleaned_cloud.ply")[0] is True
    assert any("sealed for this cloud" in m for m in pipe.lines)
    npz_bytes = (out / "floor_transform.npz").read_bytes()
    # light resume: the sealed transform is kept
    pipe = _Pipe()
    _cloudcompy_work(pipe, str(session), {"postprocessing": {"build_potree": False}})
    assert any("kept (light resume" in m for m in pipe.lines)
    assert (out / "floor_transform.npz").read_bytes() == npz_bytes
    # an unsealed transform on disk (the gizmo wrote it, no seal): replaced and declared
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(45.0), t=np.zeros(3))
    (out / "floor_level.json").unlink()
    pipe = _Pipe()
    _cloudcompy_work(pipe, str(session), {"postprocessing": {"build_potree": False}})
    assert any("REPLACED" in m and "no floor_level.json" in m for m in pipe.lines)
    assert (out / "floor_transform.npz").read_bytes() == npz_bytes
    assert P._floor_seal_status(out, out / "cleaned_cloud.ply")[0] is True
    # a baked orientation: no transform, no seal
    (out / ".orientation_applied").write_text("")
    _cloudcompy_work(_Pipe(), str(session), {"postprocessing": {"build_potree": False}})
    assert not (out / "floor_transform.npz").exists() and not (out / "floor_level.json").exists()


def test_the_cleanup_refits_boxes_in_the_stages_display_frame_or_fails(tmp_path):
    """`_refit_obbs` used to read floor_transform.npz unconditionally and turn any error
    into 'boxes not refitted' (stale boxes shipped): one rule for every stage, a failure
    names the step."""
    from segmentation.geometric_cleanup import _refit_obbs
    out, _, _ = _session(tmp_path)
    xyz = _bytes_to_xyz(out / "cleaned_cloud.ply")
    inst = {"instance_id": 1, "globalIndices": list(range(0, 200))}
    logs = []
    assert _refit_obbs(out, [inst], len(xyz), log=logs.append) == 1
    assert inst["obb"]["n_points"] == 200 and inst["obb"]["decisions"]["yaw"]["chosen"]
    assert any("display frame: identity (.orientation_applied)" in m for m in logs)
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(30.0), t=np.zeros(3))
    with pytest.raises(RuntimeError, match="floor_transform.npz cannot be applied"):
        _refit_obbs(out, [inst], len(xyz), log=lambda m: None)


def _bytes_to_xyz(ply):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(ply)).points)


# ── 109: the footprint error is a seeded bootstrap over the points ──────────────────────

def test_the_footprint_error_is_measured_by_a_seeded_bootstrap():
    import time
    from correction.floor import N_BOOT_FLOOR
    rng = np.random.default_rng(7)
    pts = np.c_[rng.uniform(-0.8, 0.8, 4000), rng.uniform(-0.4, 0.4, 4000)] + rng.normal(0, 0.004, (4000, 2))
    P._footprint_error(pts, 0.0)                                 # warm (imports), then measure
    t0 = time.perf_counter()
    sig, n_boot, seed = P._footprint_error(pts, 0.0)
    cost = time.perf_counter() - t0
    print(f"[footprint] one candidate, 4000 points, {n_boot} resamples: {cost * 1000:.1f} ms")
    assert np.isfinite(sig) and sig > 0 and n_boot == N_BOOT_FLOOR
    assert P._footprint_error(pts, 0.0) == (sig, n_boot, seed), "seeded: one input, one error"
    perm = pts[rng.permutation(len(pts))]
    assert P._footprint_error(perm, 0.0) == (sig, n_boot, seed), \
        "a function of the point SET: the seed AND the error, in any order of the points"
    assert P._footprint_error(pts, 0.3)[2] != seed, "and from the yaw"
    # the identity computed once per box is the one every candidate uses
    ident = P._footprint_identity(pts)
    assert P._footprint_error(pts, 0.0, identity=ident) == (sig, n_boot, seed)
    assert P._footprint_error(perm, 0.0, identity=P._footprint_identity(perm))[0] == sig
    assert cost < 0.25, f"the vectorised bootstrap: {cost:.3f} s (the resample loop cost 0.65 s)"
    # the error reaches the OBB record, with the bootstrap's size and seed
    from test_p3_decisions import _table
    obb = P._compute_obb(_table(17.0, np.random.default_rng(4)))
    cand = obb["decisions"]["yaw"]["candidates"]
    assert cand and all(c["footprint_error_m2"] > 0 and c["error_n_boot"] == N_BOOT_FLOOR
                        and isinstance(c["error_seed"], int) for c in cand)
    assert obb["decisions"]["yaw"]["chosen"]["error_factor"] == P._decision_params(__import__("config").cfg)[0]
    assert obb["decisions"]["yaw"]["chosen"]["footprint_error_m2"] == \
        min(cand, key=lambda c: c["footprint_m2"])["footprint_error_m2"]


# ── 108: the mask audit addresses the masks through the exact grid maps ─────────────────

def test_mask_pixels_follow_the_exact_grid_maps_not_a_scale(tmp_path):
    """A record grid that is Omega's CROP of the native frame (x 40..160 of 200) against a
    full-frame mask grid: a scale alone lands 20 mask pixels off."""
    from correction.visit_drift import SessionProjection
    from precision.camera import CameraModel, GridMap, mask_grid_for, save_camera_json
    from reconstruction.surface_fit import hole_audit as HA
    out = tmp_path / "output"
    out.mkdir()
    rec = GridMap(name="omega", w=120, h=90, content_w=120, content_h=90, pad_left=0, pad_top=0,
                  crop_x=40, crop_y=0, crop_w=120, crop_h=90, native_w=200, native_h=90)
    cam = CameraModel(width=200, height=90, params=(100.0, 100.0, 100.0, 45.0, 0.0, 0.0, 0.0, 0.0),
                      source="refine", camera_epoch=1, omega_grid=rec, mask_grid=None)
    save_camera_json(out / "camera.json", cam, geometry_epoch=1)
    (out / "camera_frames.txt").write_text("10\n")
    (out / "camera_poses.txt").write_text(" ".join(repr(float(x)) for x in np.eye(4).ravel()) + "\n")
    mh, mw = 45, 100
    P.write_canonical_store(out / "seg_masks.npz", {
        mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME),
        mask_space.KEYFRAMES_KEY: np.array([10], np.int32), "obj_ids": np.array([0], np.int32),
        "frames": np.array([0], np.int32), "scaled_res": np.array([mh, mw], np.int32),
        "f0_o0": np.ones((mh, mw), np.uint8)})
    d = np.zeros(3, PLY_DTYPE)
    d["x"], d["y"], d["z"] = [0.0, 0.1, -0.1], [0.0, 0.05, 0.0], [1.0, 1.2, 1.5]
    d["frame_global"] = 10
    d["pixel_row"], d["pixel_col"] = [45, 50, 40], [60, 70, 50]
    _write_ply(out / "cleaned_cloud.ply", d)
    HA._EVIDENCE_CACHE.clear()
    ev = HA._Evidence(out, tmp_path)
    assert ev.ok, ev.reason
    assert ev.mask_hw == (mh, mw) and (ev.kw, ev.kh) == (120, 90)
    u = np.array([0.0, 60.0, 119.0])
    v = np.array([0.0, 45.0, 89.0])
    mu, mv = ev.mask_pixels(u, v, mh, mw)
    rows, cols = SessionProjection(cam, rec, mask_grid_for(200, 90, (mh, mw)), "t").record_to_mask(v, u)
    assert np.array_equal(mu, cols) and np.array_equal(mv, rows), "the certification's maps"
    assert not np.array_equal(mu, (u * mw / ev.kw).astype(np.int64)), "a scale ignores the crop"
    assert mu[0] == 20, "record x = 0 is native x = 40, mask x = 20"
    with pytest.raises(RuntimeError, match="one store, one grid"):
        ev.mask_pixels(u, v, 90, 120)
    HA._EVIDENCE_CACHE.clear()


# ── 112: one configuration for the whole SAM3 stage ─────────────────────────────────────

def test_run_segmentation_runs_on_the_configuration_it_is_handed(tmp_path, monkeypatch):
    import copy
    import torch
    import segmentation.sam3_wrapper as w
    from config import cfg as server_cfg
    from test_p3_sam3_store import SCRIPT, _FakeSAM3, _session as _sam3_session
    seen = []

    def wrapper(config=None):
        seen.append(config)
        return _FakeSAM3(SCRIPT)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(w, "get_sam3_wrapper", wrapper)
    handed = copy.deepcopy(server_cfg)
    handed["models"]["segmentation"]["multiplex_count"] = 12345
    frames, out = _sam3_session(tmp_path)
    P.run_segmentation(str(frames), str(out), "chair;box", prompt_status={}, config=handed)
    assert seen and all(c is handed for c in seen), "the wrapper is built on the handed configuration"
    doc = json.loads((out / "segmentation.json").read_text())
    kf = mask_space.declared_keyframes(np.load(out / "seg_masks.npz"))
    sealed = doc["stamp"]["config"]["sam3"]
    of = lambda c: P.sam3_store_seal(c, ["chair", "box"], kf, [4, 4], None,   # noqa: E731
                                     out / "seg_masks.npz")["config"]["sam3"]
    assert sealed == of(handed) != of(server_cfg), "the seal records the handed SAM3 block, not config.yaml's"
    assert P._cfg_get(handed, "models.segmentation.multiplex_count", 0) == 12345
    assert P._cfg_get({"a": {}}, "a.b.c", "dflt") == "dflt"


def test_the_geometric_cleanup_reads_the_sessions_frozen_configuration(tmp_path):
    """ONE dict for the whole cleanup: the audit's `segmentation` section and the OBB refit's
    error factor come from the same configuration (the frozen file, or the one handed)."""
    from intake.run_config import freeze_run_config
    from segmentation.geometric_cleanup import _cfg, _full_cfg
    session = tmp_path / "scan"
    frozen = {"segmentation": {"mask_filter": {"visit_gap_kf": 1}, "frozen_marker": 1},
              "correction_graph": {"graph": {"improvement_error_factor": 1.1, "heldout_confidence": 0.95}}}
    freeze_run_config(session, frozen, log=lambda m: None)
    assert _cfg(None, session / "output")["frozen_marker"] == 1
    assert _full_cfg(None, session / "output") == frozen
    handed = {"segmentation": {"given": True}, "correction_graph": frozen["correction_graph"]}
    assert _cfg(handed, session / "output") == {"given": True}
    assert _full_cfg(handed, session / "output") is handed


def test_the_result_stamp_names_the_configuration_by_content_never_by_caller(tmp_path, fake_potree):
    """`config_source` named the CALLER (the frozen file / 'handed by the caller'): a result
    projected in-run with the handed dict was judged stale by the next check (cfg=None →
    the frozen file) and re-projected; two projections of identical inputs through two
    callers wrote different bytes. The identity is the content of the sections read."""
    import copy
    from intake.run_config import freeze_run_config, load_run_config
    from config import cfg as server_cfg
    out, _, _ = _session(tmp_path)
    freeze_run_config(tmp_path, server_cfg, log=lambda m: None)
    frozen, _sha = load_run_config(tmp_path)
    handed = copy.deepcopy(frozen)
    a = P.projection_stamp(out, out / "cleaned_cloud.ply")                            # the frozen file
    b = P.projection_stamp(out, out / "cleaned_cloud.ply", handed, "the configuration handed by the caller")
    assert a == b and "config_source" not in a["config"]
    res = P._match_and_save_result(out, config=handed, config_source="handed by a worker")
    assert "config_source" not in res["decisions"]
    stale, why = P.segmentation_result_is_stale(out)
    assert not stale, why
    first = (out / "segmentation_result.json").read_bytes()
    assert P.map_segmentation_to_cloud(out)["instances"] and len(fake_potree) == 1, "reused, not re-projected"
    assert (out / "segmentation_result.json").read_bytes() == first
    # another configuration of a section the projection reads: another identity
    other = copy.deepcopy(handed)
    other["segmentation"]["__other__"] = 1
    assert P.projection_stamp(out, out / "cleaned_cloud.ply", other)["config"]["segmentation"] != a["config"]["segmentation"]


def test_the_mask_audit_uses_the_declared_grid_of_the_evidence(tmp_path, fake_potree):
    """`MaskAudit._mask_res` probed the first mask's shape when `scaled_res` was unreadable —
    a grid deduced from the data (point 108 forbids it): the audit's grid is the evidence's
    declared one, validated there."""
    from reconstruction.surface_fit import hole_audit as HA
    from segmentation.mask_filter import MaskAudit
    from test_p3_projection import H, W
    out, _, _ = _session(tmp_path)
    HA._EVIDENCE_CACHE.clear()
    audit = MaskAudit(out, tmp_path, {"visit_gap_kf": 1}, log=lambda m: None)
    assert audit.res == audit.ev.mask_hw == (H, W)
    assert not hasattr(MaskAudit, "_mask_res")
    HA._EVIDENCE_CACHE.clear()
