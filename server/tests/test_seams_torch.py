"""One seam of the cube graph measured on a device (USER 2026-11-19: "Todo debe hacerse en placa") against
cube_run.CubeRun.seams, the path of record: the same masks and sample element by element, the fits to 1e-9."""
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch

FORK = Path(os.environ.get("STAC_FORK_DIR") or Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
sys.path.insert(0, str(FORK))
from loop_utils import cube_run as CR  # noqa: E402
from loop_utils import seams_torch as ST  # noqa: E402
from loop_utils import stable_sample as SS  # noqa: E402
from loop_utils.loop_bridges import robust_sim3  # noqa: E402
from loop_utils.metric_lock import robust_rigid  # noqa: E402

DEV = "cpu"


def test_the_hash_and_the_stable_pick_are_the_records_bit_for_bit():
    rng = np.random.default_rng(2)
    keys = rng.integers(0, 2 ** 63, size=50000, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
    keys[100:140] = keys[60]                                              # equal keys: ties fall to the earlier
    t = torch.as_tensor(keys.astype(np.int64))
    assert np.array_equal(ST.mix64(t).numpy().astype(np.uint64), SS.mix64(keys))
    for k in (0, 1, 37, 4999, 49999, 50000, 70000):
        assert np.array_equal(ST.stable_pick(t, k).numpy(), SS.stable_pick(keys, k)), k
    pix = SS.pixel_keys(np.repeat(np.arange(2700, 2706), 5000), np.tile(np.arange(5000), 6))
    assert np.array_equal(ST.stable_pick(torch.as_tensor(pix.astype(np.int64)), 12345).numpy(), SS.stable_pick(pix, 12345))


def test_the_medians_are_numpys():
    rng = np.random.default_rng(4)
    for n in (1, 2, 7, 8, 1001, 2000):
        x = rng.normal(size=n)
        assert float(ST.median(torch.as_tensor(x))) == float(np.median(x))
        y = rng.normal(size=(n, 3))
        assert np.array_equal(ST.median0(torch.as_tensor(y)).numpy(), np.median(y, axis=0))


def test_the_range_is_decided_on_the_squared_distance_exactly_as_the_records_norm():
    for r in (15.0, 10.0, 0.1, 7.3):
        b = ST.range_sq_bound(r)
        assert np.sqrt(b) <= r < np.sqrt(np.nextafter(b, np.inf))


def _pairs(rng, n=60000, s=1.07, outliers=0.08):
    src = rng.normal(size=(n, 3)) * np.array([3.0, 1.5, 4.0])
    w = rng.normal(size=3) * 0.3
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    dst = s * (src @ R.T) + np.array([0.4, -1.1, 2.0]) + rng.normal(size=(n, 3)) * 0.01
    bad = rng.random(n) < outliers
    dst[bad] += rng.normal(size=(int(bad.sum()), 3)) * 0.8
    return src, dst


def test_the_two_fits_are_the_records_to_1e9():
    rng = np.random.default_rng(6)
    src, dst = _pairs(rng)
    a = robust_sim3(src, dst)
    b = ST.robust_sim3(torch.as_tensor(src), torch.as_tensor(dst))
    assert abs(a[0] - b[0]) < 1e-9 and np.abs(a[1] - b[1]).max() < 1e-9 and np.abs(a[2] - b[2]).max() < 1e-9
    assert abs(a[3] - b[3]) < 1e-9 and a[4] == b[4]
    c = robust_rigid(src, dst)
    d = ST.robust_rigid(torch.as_tensor(src), torch.as_tensor(dst))
    assert np.abs(c[0] - d[0]).max() < 1e-9 and np.abs(c[1] - d[1]).max() < 1e-9 and abs(c[2] - d[2]) < 1e-9 and c[3] == d[3]


def _cubes(tmp_path, rng, H=420, W=440, per=4, K=3):
    """K cubes of ``per`` frames along a walk, neighbours sharing 3 of their 4 frames: one true surface, each cube
    in its own frame and unit, with noise, low-confidence pixels, far pixels and a few non-finite ones."""
    n_frames = per + (K - 1)
    world = rng.normal(size=(n_frames, H, W, 3)) * np.array([2.0, 1.0, 3.0]) + np.array([0.0, 0.0, 6.0])
    world[:, : H // 6] += np.array([0.0, 0.0, 30.0])                    # a band beyond the range
    cams = np.cumsum(np.c_[rng.normal(size=(n_frames, 1)) * 0.02 + 0.1, np.zeros((n_frames, 2))], 0)
    sets = []
    for k in range(K):
        F = np.arange(k, k + per)
        sets.append(F)
        s = float(np.exp(rng.normal() * 0.1))
        w = rng.normal(size=3) * 0.2
        th = np.linalg.norm(w)
        Kx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
        R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
        t = rng.normal(size=3)
        wp = (s * (world[F].reshape(-1, 3) @ R.T) + t).reshape(per, H, W, 3) + rng.normal(size=(per, H, W, 3)) * 0.01
        wp[rng.random((per, H, W)) < 0.002] = np.nan
        conf = rng.random((per, H, W)).astype(np.float32) + 0.2
        conf[rng.random((per, H, W)) < 0.15] = 0.0
        ext = np.tile(np.eye(4), (per, 1, 1))
        ext[:, :3, 3] = s * (cams[F] @ R.T) + t
        np.save(tmp_path / f"chunk_{k}.npy", {"world_points": wp.astype(np.float32), "world_points_conf": conf,
                                               "extrinsic": ext.astype(np.float32), "_stac_frames": F.tolist()})
    return sets


def _seams(tmp_path, sets, device):
    r = CR.CubeRun.__new__(CR.CubeRun)
    r.sets, r.K = sets, len(sets)
    r.local = [{int(g): i for i, g in enumerate(F)} for F in sets]
    r.pairs = CR.overlap_pairs(sets)
    r.un = str(tmp_path)
    r.range_m = 15.0
    r.cfg = {"cube_seams_device": device}
    out = io.StringIO()
    with redirect_stdout(out):
        edges = r.seams()
    return edges, out.getvalue(), r.scale_bound


def test_the_seams_on_a_device_are_the_records(tmp_path):
    sets = _cubes(tmp_path, np.random.default_rng(9))
    rec, said_rec, sb_rec = _seams(tmp_path, sets, "numpy")
    got, said_got, sb_got = _seams(tmp_path, sets, DEV)
    assert len(rec) == len(got) == 3 and "3 cube read(s)" in said_got and "3 cube read(s)" in said_rec
    left = lambda t: sorted(ln for ln in t.splitlines() if "left out" in ln or "not an edge" in ln)  # noqa: E731
    assert left(said_rec) == left(said_got) and left(said_rec), "the pixels left out by the range are counted alike"
    for a, b in zip(rec, got):
        same = ("a", "b", "fit", "exact", "shared_frames", "correspondences")
        assert [a[k] for k in same] == [b[k] for k in same]
        assert a["correspondences"] == 200000, "the stable sample was drawn"
        for k in ("s", "sigma_m", "lever_m"):
            assert abs(float(a[k]) - float(b[k])) < 1e-9, k
        for k in ("R", "t", "centroid"):
            assert np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)).max() < 1e-9, k
    assert abs(sb_rec["median"] - sb_got["median"]) < 1e-9 and abs(sb_rec["bound"] - sb_got["bound"]) < 1e-9


def test_the_mask_and_the_sample_are_the_records_element_by_element(tmp_path):
    sets = _cubes(tmp_path, np.random.default_rng(13), K=2)
    da = np.load(tmp_path / "chunk_0.npy", allow_pickle=True).item()
    db = np.load(tmp_path / "chunk_1.npy", allow_pickle=True).item()
    sh = np.asarray(sorted(set(sets[0]) & set(sets[1])))
    la = np.asarray([list(sets[0]).index(g) for g in sh])
    lb = np.asarray([list(sets[1]).index(g) for g in sh])
    pa, pb = da["world_points"][la], db["world_points"][lb]
    ca, cb = da["world_points_conf"][la], db["world_points_conf"][lb]
    cma = np.asarray(da["extrinsic"], np.float64)[la][:, :3, 3]
    cmb = np.asarray(db["extrinsic"], np.float64)[lb][:, :3, 3]
    t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), dtype=dt)  # noqa: E731
    px = pa.shape[1] * pa.shape[2]
    seen = {"far": 0, "not_finite": 0}
    for rng_m in (15.0, 0.0):                  # with the range, and without it (the non-finite points reach the fits' own mask)
        # the record's lines (cube_run.seams, robust_sim3)
        ok = (ca > 1e-5) & (cb > 1e-5)
        if rng_m > 0:
            near = ((np.linalg.norm(pa - cma[:, None, None, :], axis=-1) <= rng_m)
                    & (np.linalg.norm(pb - cmb[:, None, None, :], axis=-1) <= rng_m))
            seen["far"] += int((ok & ~near).sum())
            ok &= near
        flat = np.flatnonzero(ok.reshape(-1))
        keys = SS.pixel_keys(sh[flat // px], flat % px)
        P_a, P_b = pa.reshape(-1, 3)[flat].astype(np.float64), pb.reshape(-1, 3)[flat].astype(np.float64)
        m = np.isfinite(P_b).all(1) & np.isfinite(P_a).all(1)
        seen["not_finite"] += int((~m).sum())
        pick = SS.stable_pick(keys[m], 200000)
        keep = {}
        ST.measure(t(pa), t(ca), t(cma, torch.float64), t(pb), t(cb), t(cmb, torch.float64), sh, rng_m, keep=keep)
        assert np.array_equal(keep["ok"], ok) and np.array_equal(keep["finite"], m), rng_m
        assert int(m.sum()) > 200000 and np.array_equal(keep["pick"], pick), rng_m
    assert seen["far"] > 0 and seen["not_finite"] > 0, "both masks were exercised"


def test_an_unknown_device_is_refused_naming_itself():
    with pytest.raises(ValueError, match="unknown device"):
        ST.device_of("numpy")
