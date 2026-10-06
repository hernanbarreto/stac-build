"""Fuse: ONE global metric scale (DA3 / LoGeR), a per-frame affine fit of PointDiT's depth onto
LoGeR's (scaled) depth, unprojection with LoGeR's poses (scaled), voxel cleaning → cloud.ply.

Every choice is a declared default (DEFAULTS below), reported in fuse_report.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULTS = {
    "pixel_stride": 2,          # unproject every 2nd native pixel (density; the voxel decides the rest)
    "voxel_m": 0.01,            # 1 cm voxel average
    "conf_min_norm": 0.10,      # LoGeR confidence gate: a min-max fraction per frame (as the viewer slider)
    "huber_k": 1.345,           # Huber constant of the affine fit (the standard 95 % efficiency value)
    "irls_iterations": 10,      # as the F6 bend
    "fit_samples": 200000,      # pixels per frame used by the fit (deterministic stride)
    "reject_mad_k": 3.0,        # a frame whose fit residual is beyond median + k·MAD of the frames is dropped
}


def _resize(a: np.ndarray, W: int, H: int, nearest: bool = False) -> np.ndarray:
    import cv2
    return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def global_scale(Lz, Lc, Dz, Dc) -> float:
    """median(DA3 / LoGeR) over the pixels both trust (each above its own frame median confidence)."""
    ok = (Lz > 1e-6) & (Dz > 1e-6) & (Lc >= np.median(Lc)) & (Dc >= np.median(Dc))
    return float(np.median(Dz[ok] / Lz[ok])) if ok.sum() > 100 else float("nan")


def fit_frame(Pz, Pv, ref, w, cfg) -> tuple:
    """(a, b, residual): ref ≈ a·Pz + b on the supported pixels (Huber IRLS)."""
    from precision.mono_detail import irls_affine
    sup = Pv & (ref > 1e-6) & (w > 0)
    idx = np.flatnonzero(sup.ravel())
    if idx.size < 1000:
        return float("nan"), float("nan"), float("inf")
    step = max(1, idx.size // int(cfg["fit_samples"]))
    idx = idx[::step]
    x = Pz.ravel()[idx].astype(np.float64); y = ref.ravel()[idx].astype(np.float64)
    a, b = irls_affine(x, y, w.ravel()[idx].astype(np.float64), cfg["huber_k"], cfg["irls_iterations"])
    r = (y - (a * x + b)) / y
    return a, b, _mad(r)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement
    v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def voxel_mean(xyz: np.ndarray, rgb: np.ndarray, v: float):
    q = np.floor(xyz / v).astype(np.int64)
    q -= q.min(axis=0)
    key = (q[:, 0] * (q[:, 1].max() + 1) + q[:, 1]) * (q[:, 2].max() + 1) + q[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out = np.stack([np.bincount(inv, xyz[:, k]) / cnt for k in range(3)], 1)
    col = np.stack([np.bincount(inv, rgb[:, k].astype(np.float64)) / cnt for k in range(3)], 1)
    return out.astype(np.float32), np.clip(np.rint(col), 0, 255).astype(np.uint8)


def fuse(out_dir: Path, frames_dir: Path, cfg: dict = None, log=print) -> dict:
    import cv2
    cfg = dict(DEFAULTS, **(cfg or {}))
    L = np.load(out_dir / "loger.npz")
    names = [str(n) for n in L["names"]]
    H, W = (int(x) for x in L["grid_hw"]); H0, W0 = (int(x) for x in L["native_hw"])
    da3 = out_dir / "da3"; pdit = out_dir / "pointdit"
    # 1) ONE global metric scale
    ratios = []
    for i, n in enumerate(names):
        st = Path(n).stem
        dp, cp = da3 / f"{st}_depth.npy", da3 / f"{st}_conf.npy"
        if not dp.exists():
            ratios.append(np.nan); continue
        Dz = np.load(dp).astype(np.float32); Dc = np.load(cp).astype(np.float32)
        Lz = _resize(L["depth"][i], Dz.shape[1], Dz.shape[0]); Lc = _resize(L["conf"][i], Dz.shape[1], Dz.shape[0])
        ratios.append(global_scale(Lz, Lc, Dz, Dc))
    r = np.asarray(ratios, np.float64)
    s = float(np.nanmedian(r))
    rel_mad = _mad(np.log(r[np.isfinite(r)]))
    trend = float(np.corrcoef(np.arange(len(r))[np.isfinite(r)], np.log(r[np.isfinite(r)]))[0, 1]) if np.isfinite(r).sum() > 3 else float("nan")
    log(f"[fuse] metric scale s = {s:.4f} (DA3/LoGeR, {np.isfinite(r).sum()} frames, spread ±{rel_mad * 100:.1f} %, "
        f"trend along the walk r = {trend:+.2f})")
    # 2) per-frame affine fit of PointDiT onto s·LoGeR depth
    fits = []
    for i, n in enumerate(names):
        st = Path(n).stem
        P = np.load(pdit / f"{st}.npz")
        Pz = _resize(P["z"], W0, H0); Pv = _resize(P["valid"].astype(np.float32), W0, H0, nearest=True) > 0.5
        ref = s * _resize(L["depth"][i], W0, H0); w = _resize(L["conf"][i], W0, H0)
        a, b, res = fit_frame(Pz, Pv, ref, w, cfg)
        fits.append((a, b, res))
    res = np.array([f[2] for f in fits])
    ok_res = np.isfinite(res)
    bar = float(np.median(res[ok_res]) + cfg["reject_mad_k"] * _mad(res[ok_res]))
    keep = [i for i, (a, b, rr) in enumerate(fits) if np.isfinite(a) and a > 0 and rr <= bar]
    log(f"[fuse] PointDiT fits: median residual {np.median(res[ok_res]) * 100:.1f} %, bar {bar * 100:.1f} %; "
        f"{len(keep)}/{len(names)} frames kept")
    # 3) unproject with the scaled poses and K on the native grid
    K = L["K"].copy(); K[0] *= W0 / W; K[1] *= H0 / H
    c2w = L["c2w"].copy(); c2w[:, :3, 3] *= s
    st_px = int(cfg["pixel_stride"])
    vv, uu = np.mgrid[0:H0:st_px, 0:W0:st_px]
    rays = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    pts, cols = [], []
    for i in keep:
        stem = Path(names[i]).stem
        a, b, _ = fits[i]
        P = np.load(pdit / f"{stem}.npz")
        Z = a * _resize(P["z"], W0, H0) + b
        valid = (_resize(P["valid"].astype(np.float32), W0, H0, nearest=True) > 0.5) & (Z > 0)
        c = _resize(L["conf"][i], W0, H0)
        lo, hi = float(c.min()), float(c.max())
        valid &= c >= lo + cfg["conf_min_norm"] * (hi - lo)
        Zs = Z[::st_px, ::st_px]; m = valid[::st_px, ::st_px]
        Xc = rays[m] * Zs[m][:, None]
        Xw = Xc @ c2w[i][:3, :3].T + c2w[i][:3, 3]
        img = cv2.cvtColor(cv2.imread(str(frames_dir / names[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        pts.append(Xw.astype(np.float32)); cols.append(img[::st_px, ::st_px][m])
    xyz = np.concatenate(pts); rgb = np.concatenate(cols)
    n_raw = len(xyz)
    xyz, rgb = voxel_mean(xyz, rgb, float(cfg["voxel_m"]))
    write_ply(out_dir / "cloud.ply", xyz, rgb)
    with open(out_dir / "camera_poses_c2w.txt", "w") as f:
        for i, n in enumerate(names):
            f.write(n + " " + " ".join(f"{x:.9g}" for x in c2w[i].ravel()) + "\n")
    walk = float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum())
    rep = {"defaults": cfg, "metric_scale": s, "scale_spread_rel": rel_mad, "scale_trend_r": trend,
           "frames": len(names), "frames_kept": len(keep), "fit_residual_bar": bar,
           "K_native": K.tolist(), "walk_m": walk, "points_raw": int(n_raw), "points": int(len(xyz)),
           "per_frame": [{"name": names[i], "ratio": None if not np.isfinite(r[i]) else float(r[i]),
                          "a": fits[i][0], "b": fits[i][1], "residual": fits[i][2], "kept": i in keep}
                         for i in range(len(names))]}
    json.dump(rep, open(out_dir / "fuse_report.json", "w"), indent=1)
    log(f"[fuse] camera walk {walk:.2f} m (metric); cloud {n_raw:,} → {len(xyz):,} points at "
        f"{cfg['voxel_m'] * 100:.0f} cm → {out_dir / 'cloud.ply'}")
    return rep


if __name__ == "__main__":
    fuse(Path(sys.argv[1]), Path(sys.argv[2]))


def depth_edges(z: np.ndarray, rtol: float) -> np.ndarray:
    """True where the depth jumps by more than rtol (relative) inside a 3x3 neighbourhood — LoGeR's
    own demo zeroes the confidence there (loger.utils.geometry.depth_edge, rtol 0.03)."""
    import cv2
    k = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(z.astype(np.float32), k); zmin = cv2.erode(z.astype(np.float32), k)
    return (zmax - zmin) > rtol * np.maximum(z, 1e-6)


def fuse_loger_only(out_dir: Path, frames_dir: Path, cfg: dict = None, log=print, name: str = "cloud_loger.ply") -> dict:
    """The control cloud: LoGeR's OWN depth on its grid (no PointDiT), the same global metric scale,
    LoGeR's depth-edge mask, the same confidence gate and voxel."""
    import cv2
    cfg = dict(DEFAULTS, edge_rtol=0.03, **(cfg or {}))
    L = np.load(out_dir / "loger.npz"); rep = json.load(open(out_dir / "fuse_report.json"))
    names = [str(n) for n in L["names"]]
    H, W = (int(x) for x in L["grid_hw"])
    s = float(rep["metric_scale"]); K = L["K"]; c2w = L["c2w"].copy(); c2w[:, :3, 3] *= s
    vv, uu = np.mgrid[0:H, 0:W] + 0.5
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    pts, cols = [], []
    for i, n in enumerate(names):
        Z = s * L["depth"][i]; c = L["conf"][i]
        lo, hi = float(c.min()), float(c.max())
        m = (Z > 0) & (c >= lo + cfg["conf_min_norm"] * (hi - lo)) & ~depth_edges(Z, cfg["edge_rtol"])
        Xw = (rays[m] * Z[m][:, None]) @ c2w[i][:3, :3].T + c2w[i][:3, 3]
        img = cv2.cvtColor(cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        pts.append(Xw.astype(np.float32)); cols.append(img[m])
    xyz = np.concatenate(pts); rgb = np.concatenate(cols); n_raw = len(xyz)
    xyz, rgb = voxel_mean(xyz, rgb, float(cfg["voxel_m"]))
    write_ply(out_dir / name, xyz, rgb)
    log(f"[fuse] LoGeR-only cloud: {n_raw:,} → {len(xyz):,} points at {cfg['voxel_m'] * 100:.0f} cm → {out_dir / name}")
    return {"points_raw": int(n_raw), "points": int(len(xyz))}


def fuse_da3_cond(out_dir: Path, frames_dir: Path, cfg: dict = None, log=print, name: str = "cloud_da3cond.ply") -> dict:
    """Cloud from DA3 conditioned on LoGeR's metric poses (da3_cond/), DA3's own grid and K, the
    same confidence gate (DA3's conf), the depth-edge mask and the voxel."""
    import cv2
    cfg = dict(DEFAULTS, edge_rtol=0.03, **(cfg or {}))
    names = [Path(l.strip()).name for l in open(out_dir / "images.txt") if l.strip()]
    pts, cols = [], []
    for n in names:
        d = np.load(out_dir / "da3_cond" / (Path(n).stem + ".npz"))
        Z, c, K, T = d["depth"], d["conf"], d["K"], d["c2w"]
        h, w = Z.shape
        vv, uu = np.mgrid[0:h, 0:w] + 0.5
        lo, hi = float(c.min()), float(c.max())
        m = (Z > 0) & (c >= lo + cfg["conf_min_norm"] * (hi - lo)) & ~depth_edges(Z, cfg["edge_rtol"])
        X = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * Z[m], (vv[m] - K[1, 2]) / K[1, 1] * Z[m], Z[m]], 1)
        Xw = X @ T[:3, :3].T + T[:3, 3]
        img = cv2.cvtColor(cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        pts.append(Xw.astype(np.float32)); cols.append(img[m])
    xyz = np.concatenate(pts); rgb = np.concatenate(cols); n_raw = len(xyz)
    xyz, rgb = voxel_mean(xyz, rgb, float(cfg["voxel_m"]))
    write_ply(out_dir / name, xyz, rgb)
    log(f"[fuse] DA3-conditioned cloud: {n_raw:,} → {len(xyz):,} points at {cfg['voxel_m'] * 100:.0f} cm → {out_dir / name}")
    return {"points_raw": int(n_raw), "points": int(len(xyz))}
