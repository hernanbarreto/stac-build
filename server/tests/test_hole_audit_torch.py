"""The hole-audit evidence on torch (USER 2026-10-15: "toda esa etapa debe correr en gpu"): the
z-buffers, mask votes and see-through tallies of ``hole_audit._Evidence`` equal the numpy recipe
they replaced (kept here as the reference), on the CPU and — when a card is visible — on CUDA
under torch's STRICT deterministic mode, bit for bit across two runs; the device is declared
(`surface_fit.evidence_device`) or set explicitly, never assumed."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch touches CUDA

import numpy as np                                               # noqa: E402
import pytest                                                    # noqa: E402
import torch                                                     # noqa: E402

from reconstruction.surface_fit import hole_audit as HA          # noqa: E402
from segmentation import mask_space                              # noqa: E402
from segmentation import pipeline as P                           # noqa: E402
from test_p3_projection import PLY_DTYPE, _write_ply             # noqa: E402

NW, NH = 320, 180            # the native frame
RW, RH = 120, 90             # the record grid: Omega's crop x 40..280 of the frame, resized 2:1
MH, MW = 45, 80              # the mask grid: a full-frame resize (4:1)
FRAMES = [10, 20, 30]
CAM_X = (-0.3, 0.0, 0.3)
LENS = (0.05, -0.01, 0.001, -0.0005)      # k1 k2 p1 p2 — the lens is part of the projection


def _camera():
    from precision.camera import CameraModel, GridMap
    rec = GridMap(name="omega", w=RW, h=RH, content_w=RW, content_h=RH, pad_left=0, pad_top=0,
                  crop_x=40, crop_y=0, crop_w=240, crop_h=180, native_w=NW, native_h=NH)
    return CameraModel(width=NW, height=NH, params=(150.0, 150.0, 160.0, 90.0, *LENS),
                       source="refine", camera_epoch=1, omega_grid=rec, mask_grid=None)


def _session(tmp_path):
    """A wall at z = 3 m, a plate at z = 1.5 m in front of it (oid 0), three cameras along x."""
    from precision.camera import save_camera_json
    out = tmp_path / "output"
    out.mkdir()
    save_camera_json(out / "camera.json", _camera(), geometry_epoch=1)
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in FRAMES) + "\n")
    poses = []
    for x in CAM_X:
        T = np.eye(4)
        T[0, 3] = x
        poses.append(T)
    (out / "camera_poses.txt").write_text("\n".join(
        " ".join(repr(float(v)) for v in T.ravel()) for T in poses) + "\n")
    rng = np.random.default_rng(3)
    wall = np.column_stack([rng.uniform(-1.6, 1.6, 5000), rng.uniform(-0.9, 0.9, 5000),
                            np.full(5000, 3.0) + rng.normal(0, 0.005, 5000)])
    plate = np.column_stack([rng.uniform(-0.3, 0.3, 1200), rng.uniform(-0.2, 0.2, 1200),
                             np.full(1200, 1.5) + rng.normal(0, 0.003, 1200)])
    xyz = np.vstack([wall, plate])
    d = np.zeros(len(xyz), PLY_DTYPE)
    d["x"], d["y"], d["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    d["red"] = 90; d["green"] = 100; d["blue"] = 110
    d["frame_global"] = np.asarray(FRAMES)[np.arange(len(xyz)) % 3]
    d["pixel_row"] = rng.integers(0, RH, len(xyz))
    d["pixel_col"] = rng.integers(0, RW, len(xyz))
    d["confidence"] = 0.9
    _write_ply(out / "cleaned_cloud.ply", d)
    masks = {mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME),
             mask_space.KEYFRAMES_KEY: np.asarray(FRAMES, np.int32),
             "obj_ids": np.array([0, 1], np.int32),
             "frames": np.arange(len(FRAMES), dtype=np.int32),
             "scaled_res": np.array([MH, MW], np.int32)}
    for k in range(len(FRAMES)):
        m0 = np.zeros((MH, MW), np.uint8)
        m0[13:33, 22:59] = 1                  # the plate, generously, in every frame
        masks[f"f{k}_o0"] = m0
        masks[f"f{k}_o1"] = (1 - m0).astype(np.uint8)
    P.write_canonical_store(out / "seg_masks.npz", masks)
    return out, xyz, plate


# ── the numpy recipe this port replaced (reconstruction/surface_fit/hole_audit.py before
# 2026-10-15), kept as the reference the tensors are held to ────────────────────────────

def _ref_zbuf(ev, cloud_fidx, mh, mw, cache):
    from scipy import ndimage as _ndi
    fidx = int(cloud_fidx)
    if fidx in cache:
        return cache[fidx]
    c2w = ev.cam.pose_map.get(fidx)
    K = ev.cam.K_for(fidx)
    if c2w is None or K is None:
        cache[fidx] = None
        return None
    c2w4 = np.eye(4)
    c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
    M = np.linalg.inv(c2w4)
    p = (M[:3, :3] @ ev._cloud.T).T + M[:3, 3]
    z = p[:, 2]
    ok = z > 0.05
    u, v = ev.cam.pixels(fidx, p[ok])
    mu, mv = ev.mask_pixels(u, v, mh, mw)
    inb = (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
    zb = np.full((mh, mw), np.inf, dtype=np.float64)
    np.minimum.at(zb, (mv[inb], mu[inb]), z[ok][inb])
    zb = _ndi.minimum_filter(zb, size=5, mode="nearest")
    cache[fidx] = zb
    return zb


def _ref_vote(ev, world_pts, oid, cache, depth_tol=0.15):
    n = len(world_pts)
    covered = np.zeros(n, dtype=np.int32)
    occluded = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=np.int32)
    for fidx, key in ev.frames_for(oid):
        c2w = ev.pose_at(fidx)
        K = ev.K_at(fidx)
        if c2w is None or K is None:
            continue
        m = ev.masks[key]
        mh, mw = m.shape
        zb = _ref_zbuf(ev, ev.cloud_frame(fidx), mh, mw, cache)
        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        pcam = (M[:3, :3] @ world_pts.T).T + M[:3, 3]
        z = pcam[:, 2]
        front = z > 0.05
        u = np.full(n, -1.0)
        v = np.full(n, -1.0)
        u[front], v[front] = ev.cam.pixels(ev.cloud_frame(fidx), pcam[front])
        mu, mv = ev.mask_pixels(u, v, mh, mw)
        inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
        valid[inb] += 1
        occ = np.zeros(n, dtype=bool)
        behind = np.zeros(n, dtype=bool)
        if zb is not None:
            zpix = np.full(n, np.inf)
            zpix[inb] = zb[mv[inb], mu[inb]]
            occ = zpix < (z - depth_tol)
            behind = np.isfinite(zpix) & (zpix > (z + depth_tol))
        occluded[occ] += 1
        hit = np.zeros(n, dtype=bool)
        hit[inb] = m[mv[inb], mu[inb]] > 0
        covered[hit & ~occ & ~behind] += 1
    return covered, occluded, valid


def _ref_see_through(ev, world_pts, cache, depth_tol=0.15):
    mh, mw = ev.mask_hw
    n = len(world_pts)
    valid = np.zeros(n, dtype=np.int32)
    at_depth = np.zeros(n, dtype=np.int32)
    behind = np.zeros(n, dtype=np.int32)
    for fidx in sorted(ev.cam.pose_map.keys()):
        c2w = ev.cam.pose_map.get(fidx)
        K = ev.cam.K_for(fidx)
        if c2w is None or K is None:
            continue
        zb = _ref_zbuf(ev, fidx, mh, mw, cache)
        if zb is None:
            continue
        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        pcam = (M[:3, :3] @ world_pts.T).T + M[:3, 3]
        z = pcam[:, 2]
        front = z > 0.05
        u = np.full(n, -1.0)
        v = np.full(n, -1.0)
        u[front], v[front] = ev.cam.pixels(fidx, pcam[front])
        mu, mv = ev.mask_pixels(u, v, mh, mw)
        inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
        zpix = np.full(n, np.inf)
        zpix[inb] = zb[mv[inb], mu[inb]]
        measured = np.isfinite(zpix)
        valid[measured] += 1
        at_depth[measured & (np.abs(zpix - z) <= depth_tol)] += 1
        behind[measured & (zpix > z + depth_tol)] += 1
    return valid, at_depth, behind


def _probe_points(plate):
    """Plate points, wall points behind the plate (occluded), points past the wall (seen
    through nothing: no measured geometry behind them) and points behind the cameras."""
    rng = np.random.default_rng(5)
    behind_plate = np.column_stack([rng.uniform(-0.25, 0.25, 300), rng.uniform(-0.15, 0.15, 300),
                                    np.full(300, 3.0)])
    beyond = np.column_stack([rng.uniform(-1.0, 1.0, 300), rng.uniform(-0.5, 0.5, 300),
                              np.full(300, 4.5)])
    back = np.column_stack([rng.uniform(-1.0, 1.0, 50), rng.uniform(-0.5, 0.5, 50),
                            np.full(50, -1.0)])
    edge = np.column_stack([rng.uniform(-3.0, 3.0, 300), rng.uniform(-2.0, 2.0, 300),
                            np.full(300, 2.0)])
    return np.vstack([plate[::4], behind_plate, beyond, back, edge]).astype(np.float64)


def _build(out, tmp_path):
    HA._EVIDENCE_CACHE.clear()
    ev = HA._Evidence(out, tmp_path)
    assert ev.ok, ev.reason
    assert ev.cam.lens is not None, "the synthetic camera carries a lens"
    return ev


def _assert_same_zbuf(a, b):
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    f = np.isfinite(a)
    assert np.allclose(a[f], b[f], rtol=0, atol=1e-9)


def test_the_tensor_evidence_equals_the_numpy_recipe_on_the_cpu(tmp_path):
    out, xyz, plate = _session(tmp_path)
    ev = _build(out, tmp_path)
    assert ev.device == "cpu" and ev.torch_record["device"] == "cpu"
    cache = {}
    pts = _probe_points(plate)
    for f in FRAMES:
        zb = ev._zbuf(f, MH, MW)
        ref = _ref_zbuf(ev, f, MH, MW, cache)
        assert zb is not None and ref is not None
        assert np.isfinite(zb).sum() > 0.2 * MH * MW, "the cloud covers the frames"
        _assert_same_zbuf(zb, ref)
    got = ev.vote(pts, 0)
    want = _ref_vote(ev, pts, 0, cache)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)
    assert got[2].max() == len(FRAMES) and got[1].sum() > 0 and got[0].sum() > 0, \
        "every frame sees the probes; some are occluded, some covered"
    got = ev.see_through(pts)
    want = _ref_see_through(ev, pts, cache)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)
    n_plate = len(plate[::4])
    valid, at_depth, behind = got
    assert (at_depth[:n_plate] > 0).mean() > 0.9, "the plate is measured at its own depth"
    beyond = slice(n_plate + 300, n_plate + 600)
    assert behind[beyond].sum() == 0 and at_depth[beyond].sum() == 0, \
        "past the wall the nearest geometry is IN FRONT — occluded, never 'behind'"
    edge = slice(n_plate + 650, n_plate + 950)
    assert behind[edge].sum() > 0, "a point floating before the wall has measured geometry behind it"
    assert ev.calibrate_oid(plate, 1) == 0, "the plate's own mask covers its points"
    assert ev.frames_for(0) == [(0, "f0_o0"), (1, "f1_o0"), (2, "f2_o0")]
    assert ev.frames_for(7) == []


def test_a_missing_frame_pose_yields_no_zbuffer_and_counts_nothing(tmp_path):
    out, xyz, plate = _session(tmp_path)
    ev = _build(out, tmp_path)
    assert ev._zbuf(99, MH, MW) is None
    with pytest.raises(RuntimeError, match="one store, one grid"):
        ev._zbuf(FRAMES[0], MH + 1, MW)
    cov, occ, val = ev.vote(np.zeros((0, 3)), 0)
    assert len(cov) == len(occ) == len(val) == 0


def test_the_device_is_declared_never_assumed(tmp_path, monkeypatch):
    import config as C
    out, xyz, plate = _session(tmp_path)
    try:
        HA.use_device(None)
        monkeypatch.setattr(C, "cfg", {"surface_fit": {}})
        with pytest.raises(RuntimeError, match="surface_fit.evidence_device"):
            HA.evidence_device()
        monkeypatch.setattr(C, "cfg", {"surface_fit": {"evidence_device": "tpu"}})
        with pytest.raises(RuntimeError, match="evidence_device"):
            HA.evidence_device()
        monkeypatch.setattr(C, "cfg", {"surface_fit": {"evidence_device": "cpu"}})
        assert HA.evidence_device() == "cpu"
        HA._EVIDENCE_CACHE.clear()
        ev = HA._Evidence(out, tmp_path)
        assert ev.ok and ev.device == "cpu"
        if not torch.cuda.is_available():
            monkeypatch.setattr(C, "cfg", {"surface_fit": {"evidence_device": "cuda"}})
            with pytest.raises(RuntimeError, match="no CUDA device"):
                HA.evidence_device()
    finally:
        HA.use_device("cpu")
        HA._EVIDENCE_CACHE.clear()


def test_the_production_config_declares_the_device():
    from config import cfg
    assert (cfg.get("surface_fit") or {}).get("evidence_device") in ("cuda", "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_on_cuda_the_evidence_matches_the_cpu_and_repeats_bit_for_bit(tmp_path):
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        pytest.skip("CUDA initialised without the pinned cuBLAS workspace")
    import repro
    out, xyz, plate = _session(tmp_path)
    ev_cpu = _build(out, tmp_path)
    pts = _probe_points(plate)
    cpu_z = [ev_cpu._zbuf(f, MH, MW) for f in FRAMES]
    cpu_vote = ev_cpu.vote(pts, 0)
    cpu_st = ev_cpu.see_through(pts)
    runs = []
    try:
        HA.use_device("cuda")
        with repro.deterministic_torch(0):
            for _ in range(2):
                ev = _build(out, tmp_path)
                assert ev.device == "cuda" and ev.torch_record["deterministic_strict"]
                z = [ev._zbuf(f, MH, MW) for f in FRAMES]
                assert all(zz is not None for zz in z)
                runs.append((z, ev.vote(pts, 0), ev.see_through(pts), ev.calibrate_oid(plate, 1)))
    finally:
        HA.use_device("cpu")
        HA._EVIDENCE_CACHE.clear()
    for z, vote, st, oid in runs:
        for a, b in zip(z, cpu_z):
            _assert_same_zbuf(a, b)
        for g, w in zip(vote, cpu_vote):
            assert np.array_equal(g, w)
        for g, w in zip(st, cpu_st):
            assert np.array_equal(g, w)
        assert oid == 0
    (z1, v1, s1, _), (z2, v2, s2, _) = runs
    assert all(a.tobytes() == b.tobytes() for a, b in zip(z1, z2)), "the same bytes twice"
    assert all(a.tobytes() == b.tobytes() for a, b in zip(v1, v2))
    assert all(a.tobytes() == b.tobytes() for a, b in zip(s1, s2))
