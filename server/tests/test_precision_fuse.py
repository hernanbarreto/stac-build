"""F7 — witness fusion, provenance v2 and the rejected set, on the synthetic F6
session (textured scene with ground truth). CPU only."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from precision import depth_sweep as DS                       # noqa: E402
from precision import fuse as FU                               # noqa: E402
from precision import provenance as PV                         # noqa: E402

import test_precision_depth as T                               # noqa: E402


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    from precision.config import load_precision_config
    p = load_precision_config()
    pc = replace(p, depth=replace(p.depth, n_views=3, best_k=2, n_hyp=16, null_frames=2,
                                  min_scale_samples=50, calib_min_bin_samples=50,
                                  calib_samples_per_frame=2000, view_samples=128,
                                  propagation_iters=1, min_consistent_views=2),
                 # the synthetic cloud is sparser than the cleaner's reach (15 mm): the
                 # fusion is tested without the recipe, the recipe's wiring separately
                 fuse=replace(p.fuse, cleaning=False))
    s, _ = T._session(tmp_path_factory.mktemp("fuse"), pc)
    DS.run_sweep(s, pc, log=lambda *a: None, device="cpu")
    return s, pc


def test_every_point_has_its_witnesses_and_rebuilds_its_frame_and_pixel(swept):
    s, pc = swept
    res = FU.fuse(s, pc, log=lambda *a: None)
    o, data = res["origins"], res["data"]
    t0 = o["source"] == DS.SOURCE_SWEEP
    assert np.all(o["n_consistent"][t0] >= pc.fuse.min_witness_views)
    assert np.all(o["n_consistent"][~t0] >= pc.depth.prior_fill_min_views)
    # exact frame + pixel: the depth at that undistorted pixel, back-projected with
    # the session camera and that frame's pose, IS the point
    out = s / "output"
    from precision.camera import load_camera_json, undistort_maps
    _m1, _m2, K = undistort_maps(load_camera_json(out / "camera.json"))
    kf, c2w = DS._read_poses(out / "camera_poses.txt", out / "camera_frames.txt")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1)
    for f in np.unique(o["frame_global"]):
        m = o["frame_global"] == f
        with np.load(out / DS.DEPTH_DIRNAME / f"frame_{int(f)}.npz") as z:
            X = FU.backproject(z["depth"], o["pixel_v_und"][m], o["pixel_u_und"][m], K,
                               c2w[kf.index(int(f))])
        assert np.array_equal(X, xyz[m])
    # one point per voxel
    keys = FU.pack_keys(np.floor(xyz / pc.fuse.voxel_m).astype(np.int64))
    assert len(np.unique(keys)) == len(keys)
    # the accounting closes: cloud + rejected = candidates
    rep = res["report"]
    assert rep["n_points"] + rep["n_rejected"] == rep["n_candidates"]
    # the copies of a surface seen by several keyframes are CONSUMED by the winner
    # within τ_rel × depth (the 4 mm voxel dedup is only the safety net behind it)
    assert rep["rejected_by_reason"]["fused"] > 0
    assert rep["fusion"]["n_fused"] == rep["rejected_by_reason"]["fused"]
    assert len(res["rejected"]["reason"]) == rep["n_rejected"]
    # the viewer channels
    assert np.all(data["confidence"][~t0] == 0) and np.all(data["mv_votes"] == o["n_consistent"])


def test_fusion_is_deterministic(swept):
    s, pc = swept
    a = FU.fuse(s, pc, log=lambda *a: None)["data"]
    b = FU.fuse(s, pc, log=lambda *a: None)["data"]
    assert a.tobytes() == b.tobytes()


def test_cleaning_recipe_drops_with_reasons_and_the_accounting_closes(swept, monkeypatch):
    """The cloud stage's recipe (voxel pick + SOR, gpu_cloud_clean's own functions and the
    postprocessing parameters) runs inside F7; every dropped point carries its reason."""
    from dataclasses import replace
    import reconstruction.gpu_cloud_clean as GC
    s, pc = swept
    calls = {}

    def fake_sor(xyz, knn, n_sigma, cell_h, *, device, query_block, candidate_budget):
        calls["sor"] = (knn, n_sigma)
        keep = np.ones(len(xyz), bool)
        keep[:3] = False                                  # three outliers, by fiat
        return keep, 0.005, 0.001

    monkeypatch.setattr(GC, "_sor_keep", fake_sor)
    res = FU.fuse(s, replace(pc, fuse=replace(pc.fuse, cleaning=True)), log=lambda *a: None)
    rep = res["report"]
    from config import cfg as raw
    assert calls["sor"] == (int(raw["postprocessing"]["sor_knn"]), float(raw["postprocessing"]["sor_sigma"]))
    assert rep["params"]["cleaning_recipe"]["voxel_size"] == raw["postprocessing"]["voxel_size"]
    assert rep["rejected_by_reason"]["sor"] == 3 and rep["rejected_by_reason"]["voxel"] >= 0
    assert rep["n_points"] + rep["n_rejected"] == rep["n_candidates"]
    reasons = res["rejected"]["reason"]
    assert (reasons == PV.REJECT_REASONS["sor"]).sum() == 3
    off = FU.fuse(s, pc, log=lambda *a: None)
    assert off["report"]["params"]["cleaning_recipe"] is None
    assert off["report"]["n_points"] == rep["n_points"] + 3 + rep["rejected_by_reason"]["voxel"]


def test_publish_is_a_selectable_epoch_readable_by_v1_consumers(swept, monkeypatch, tmp_path):
    import shutil
    s0, pc = swept
    s = tmp_path / "pub"
    shutil.copytree(s0, s)
    out = s / "output"
    # an epoch-1 cloud with v1 provenance, as the pipeline leaves it
    from correction.session import write_ply
    d1 = np.zeros(5, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("frame_global", "<i4"),
                            ("pixel_row", "<i4"), ("pixel_col", "<i4")])
    hdr = [b"ply\n", b"format binary_little_endian 1.0\n", b"element vertex 5\n",
           b"property float x\n", b"property float y\n", b"property float z\n",
           b"property int frame_global\n", b"property int pixel_row\n", b"property int pixel_col\n",
           b"end_header\n"]
    write_ply(out / "cleaned_cloud.ply", hdr, d1)
    (out / "segmentation_result.json").write_text(json.dumps({"instances": [{"id": 1}]}))
    assert PV.load_origins(out)["version"] == 1

    def fake_potree(session_dir, force=False, ply_override=None, potree_dir_override=None):
        Path(potree_dir_override).mkdir(parents=True)
        (Path(potree_dir_override) / "metadata.json").write_text("{}")
        return True
    import potree_converter
    monkeypatch.setattr(potree_converter, "convert_ply_to_potree", fake_potree)
    rep = FU.run_fuse(s, pc, log=lambda *a: None)
    assert rep["epoch_from"] == 1 and rep["epoch_to"] == 2
    org = PV.load_origins(out)
    assert org["version"] == 2 and int(org["geometry_epoch"][0]) == 2
    # the previous epoch is kept and selectable
    assert (out / "_epoch_1" / "cleaned_cloud.ply").exists()
    # v1 consumers read the new cloud (correction session: provenance + poses)
    from correction.session import load_session
    sess = load_session(out)
    assert sess.n_points == rep["n_points"] and np.all(sess.ks >= 0)
    with np.load(out / PV.REJECTED_NAME) as z:
        assert len(z["reason"]) == rep["n_rejected"]
    seg = json.loads((out / "segmentation_result.json").read_text())
    assert seg["instances"] == [] and "F8" in seg["pending"]
    assert json.loads((out / "_epoch_1" / "segmentation_result.json").read_text())["instances"]
