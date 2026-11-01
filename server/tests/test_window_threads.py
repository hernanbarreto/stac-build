"""The I3 window passes on threads (intake.walk._window_records / write_anchors /
window_content_sha256, reconstruction.chunk_covis.keyframe_views / measured_tol) against the loops
they replaced, kept here as the references: the same records, the same anchors' arrays, the same
digests, the same views and the same tolerance — exactly."""
import hashlib
import json

import numpy as np
import pytest

from intake import walk as W
from reconstruction import chunk_covis as CC


def _windows(root, n=30, wf=8, h=12, w=16, seed=0):
    rng = np.random.default_rng(seed)
    wdir = root / "output" / W.WINDOWS_DIRNAME
    wdir.mkdir(parents=True)
    spec = []
    for k, (a, b) in enumerate(W.plan_windows(n, wf, 0.5)):
        m = b - a
        idx = np.arange(a, b)
        ext = np.zeros((m, 3, 4))
        ext[:, :3, :3] = np.eye(3)
        ext[:, 0, 3] = -0.1 * idx
        depth = rng.uniform(1.0, 4.0, (m, h, w)).astype(np.float32)
        depth[:, :2, :3] = 0.0                                  # invalid pixels
        conf = rng.uniform(0.0, 2.0, (m, h, w)).astype(np.float32)
        K = np.repeat(np.array([[10.0 + k, 0, w / 2], [0, 10.0 + k, h / 2], [0, 0, 1]])[None], m, 0)
        np.savez(wdir / f"window_{k:04d}.npz", frames=(10 * idx).astype(np.int64), depth=depth,
                 depth_mono=depth * 1.01, conf=conf, extrinsics=ext, intrinsics=K,
                 scale_factor=np.float64(1.0 + 0.01 * k), is_metric=np.int64(1),
                 ref_views=np.array([m // 2], np.int64), stamp=np.array(f"stamp-{k:03d}"))
        spec.append([f"frames/{10 * i:06d}.jpg" for i in idx])
    (wdir / "windows.json").write_text(json.dumps({"windows": spec, "process_res": w, "model_id": "synthetic"}))
    return wdir, [W.load_window(wdir / f"window_{k:04d}.npz") for k in range(len(spec))]


def _sha_reference(path):
    h = hashlib.sha256()
    with np.load(path) as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            h.update(f"{k}\0{a.dtype.str}\0{list(a.shape)}\0".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def _records_reference(windows, anchors_dir):
    best = {}
    for k, w in enumerate(windows):
        n = len(w["frames"])
        for i, f in enumerate(w["frames"]):
            d = abs(i - (n - 1) / 2.0)
            if f not in best or d < best[f][0]:
                best[f] = (d, k, i)
    anchors_dir.mkdir(parents=True, exist_ok=True)
    by_window = {}
    for f, (_d, k, i) in best.items():
        by_window.setdefault(k, []).append((f, i))
    for k, items in by_window.items():
        with np.load(windows[k]["path"]) as z:
            depth, conf, K = z["depth"], z["conf"], z["intrinsics"]
            for f, i in items:
                np.savez_compressed(anchors_dir / f"frame_{f}.npz", depth=depth[i].astype(np.float32),
                                    conf=conf[i].astype(np.float32), intrinsics=K[i].astype(np.float64))
    rec = []
    for i, w in enumerate(windows):
        with np.load(w["path"]) as z:
            refs = [int(x) for x in z["ref_views"]] if "ref_views" in z.files else None
            stamp_i = str(z["stamp"]) if "stamp" in z.files else None
            d, c, K = z["depth"], z["conf"], z["intrinsics"]
            valid = (c > 0) & np.isfinite(d) & (d > 0)
            med = float(np.median(d[valid])) if valid.any() else None
            hfov = float(np.median(2.0 * np.arctan(d.shape[-1] / (2.0 * K[:, 0, 0]))))
        rec.append({"index": i, "frames": [w["frames"][0], w["frames"][-1]], "n": len(w["frames"]),
                    "scale_factor": w["scale_factor"], "is_metric": w["is_metric"],
                    "reference_views": refs, "stamp": stamp_i, "median_depth_m": med, "hfov_rad": hfov,
                    "content_sha256": _sha_reference(w["path"])})
    return len(best), rec


@pytest.mark.parametrize("threads", [1, 3, 8])
def test_window_records_and_anchors_equal_the_loop(tmp_path, monkeypatch, threads):
    monkeypatch.setattr(W, "WINDOW_THREADS", threads)
    wdir, windows = _windows(tmp_path)
    n_ref, rec_ref = _records_reference(windows, tmp_path / "ref")
    n, rec = W._window_records(windows, tmp_path / "new")
    assert n == n_ref and rec == rec_ref
    names = sorted(p.name for p in (tmp_path / "ref").glob("frame_*.npz"))
    assert names == sorted(p.name for p in (tmp_path / "new").glob("frame_*.npz")) and len(names) == n
    for nm in names:
        with np.load(tmp_path / "ref" / nm) as a, np.load(tmp_path / "new" / nm) as b:
            assert sorted(a.files) == sorted(b.files)
            for k in a.files:
                assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])
    # write_anchors on its own: the same files and arrays
    assert W.write_anchors(wdir, windows, tmp_path / "wa") == n
    for nm in names:
        with np.load(tmp_path / "ref" / nm) as a, np.load(tmp_path / "wa" / nm) as b:
            for k in a.files:
                assert np.array_equal(a[k], b[k])


def test_window_content_sha256_is_the_tobytes_digest(tmp_path):
    wdir, windows = _windows(tmp_path, h=600, w=800, n=10)        # arrays over 1 MB: the buffer path
    for w in windows:
        assert W.window_content_sha256(w["path"]) == _sha_reference(w["path"])


def _views_reference(wdir):
    windows = CC._load_windows(wdir)
    G = CC._placements(windows)
    best = {}
    for k, w in enumerate(windows):
        n = len(w["frames"])
        for i, f in enumerate(w["frames"]):
            d = abs(i - (n - 1) / 2.0)
            if f not in best or d < best[f][0]:
                best[f] = (d, k, i)
    views, cache = {}, {}
    for f, (_d, k, i) in sorted(best.items()):
        if k not in cache:
            with np.load(windows[k]["path"]) as z:
                cache = {k: {"depth": z["depth"].astype(np.float32), "conf": z["conf"].astype(np.float32),
                             "K": z["intrinsics"].astype(np.float64)}}
        z = cache[k]
        views[f] = {"c2w": G[k] @ windows[k]["c2w"][i], "depth": np.array(z["depth"][i]),
                    "conf": np.array(z["conf"][i]), "K": np.array(z["K"][i]), "window": k}
    return sorted(views), views, windows


def _tol_reference(windows):
    pool = []
    for k in range(1, len(windows)):
        a, b = windows[k - 1], windows[k]
        shared = [f for f in b["frames"] if f in a["frames"]]
        if not shared:
            continue
        with np.load(a["path"]) as za, np.load(b["path"]) as zb:
            for f in shared[:: max(1, len(shared) // 4)]:
                da = za["depth"][a["frames"].index(f)]; db = zb["depth"][b["frames"].index(f)]
                ok = (da > 0) & (db > 0)
                r = np.abs(da[ok] - db[ok]) / (0.5 * (da[ok] + db[ok]))
                pool.append(r[:: max(1, r.size // 4000)])
    return float(np.quantile(np.concatenate(pool), CC.TOL_QUANTILE)) if pool else 0.05


@pytest.mark.parametrize("threads", [1, 4])
def test_keyframe_views_and_tolerance_equal_the_loops(tmp_path, monkeypatch, threads):
    monkeypatch.setattr(W, "WINDOW_THREADS", threads)
    wdir, _ = _windows(tmp_path, n=40)
    f_ref, v_ref, _w = _views_reference(wdir)
    f_new, v_new, windows = CC.keyframe_views(wdir)
    assert f_new == f_ref
    for f in f_ref:
        assert v_new[f]["window"] == v_ref[f]["window"]
        for key in ("c2w", "depth", "conf", "K"):
            assert v_new[f][key].dtype == v_ref[f][key].dtype and np.array_equal(v_new[f][key], v_ref[f][key])
    assert CC.measured_tol(windows) == _tol_reference(windows)
