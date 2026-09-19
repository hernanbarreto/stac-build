"""§6.4 depth by correspondences: tracks (vendored VGGSfM tracker over
windows of nearby keyframes AND verified loop pairs) and SAM3 mask
contours, triangulated with the FINAL poses, feed the SAME per-frame affine
model of the depth graph (``z' = a·z + b``, ``metric_lock.solve_depth_graph``)
together with the current pairwise projections. A triangulated depth is
pose-consistent: it fixes the gauge the pairwise graph cannot see (every
frame compressed alike). Applied along the rays (``apply_depth_correction``
semantics: poses and provenance untouched). Gate: the multi-view depth
disagreement (``depth_graph_verdict``) must fall on HELD-OUT pairs — else
identity.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.witness.frames import unproject

TRACKS_REL = Path("ba_run") / "tracks.npz"


def _vendor_on_path():
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


@dataclass
class Tracks:
    obs_track: np.ndarray      # (P,) int64 track id
    obs_frame: np.ndarray      # (P,) int64 REAL frame number
    obs_uv: np.ndarray         # (P,2) float32 pixel (x=col, y=row) at (res_h, res_w)
    obs_weight: np.ndarray     # (P,) float32 visibility × score
    res_h: int
    res_w: int

    @property
    def n_obs(self) -> int:
        return int(len(self.obs_track))


def load_tracks(output_dir) -> Tracks:
    p = Path(output_dir) / TRACKS_REL
    if not p.exists():
        raise RuntimeError(f"{p} is missing — run reconstruction.vggt_tracks (or hand the tracks in)")
    z = np.load(p)
    meta = z["meta"] if "meta" in z.files else None
    if meta is None or len(meta) < 2:
        raise RuntimeError(f"{p}: meta (res_h, res_w) missing")
    return Tracks(obs_track=np.asarray(z["obs_track"], np.int64),
                  obs_frame=np.asarray(z["obs_frame"], np.int64),
                  obs_uv=np.asarray(z["obs_uv"], np.float32).reshape(-1, 2),
                  obs_weight=(np.asarray(z["obs_vis"], np.float32) * np.asarray(z["obs_score"], np.float32)),
                  res_h=int(meta[0]), res_w=int(meta[1]))


def ensure_tracks(output_dir, frames_dir, tcfg, log: Callable[[str], None] = print) -> Tracks:
    """The session's tracks; extracted with the vendored tracker (its own
    env, ``witness.tracks.python``) when absent. Failure is loud."""
    output_dir = Path(output_dir)
    if not (output_dir / TRACKS_REL).exists():
        server_dir = Path(__file__).resolve().parents[2]
        cmd = [tcfg.python, "-m", "reconstruction.vggt_tracks", "--output-dir", str(output_dir),
               "--frames-dir", str(frames_dir), "--win", str(int(tcfg.win)), "--stride", str(int(tcfg.stride)),
               "--loop-window", str(int(tcfg.loop_window))]
        log(f"[depth-tracks] extracting tracks: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(server_dir), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"track extraction failed (exit {proc.returncode}): "
                               f"{proc.stderr[-2000:]}")
    return load_tracks(output_dir)


# ── triangulation ────────────────────────────────────────────────────────────

def _proj_matrices(frames: Dict[int, dict], res_h: int, res_w: int) -> Dict[int, dict]:
    out = {}
    for n, f in frames.items():
        d = np.asarray(f["depth"])
        Hd, Wd = d.shape
        sx, sy = res_w / float(Wd), res_h / float(Hd)
        K = np.asarray(f["K"], np.float64).copy()
        K[0, :] *= sx
        K[1, :] *= sy
        w2c = np.linalg.inv(np.asarray(f["T"], np.float64))
        out[int(n)] = {"P": K @ w2c[:3], "w2c": w2c, "K": K, "sx": sx, "sy": sy}
    return out


def _dlt(Ps: np.ndarray, uv: np.ndarray) -> np.ndarray:
    A = []
    for P, (u, v) in zip(Ps, uv):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def _reproj(Ps: np.ndarray, X: np.ndarray):
    Xh = np.append(X, 1.0)
    p = Ps @ Xh                       # (n,3)
    z = p[:, 2]
    uv = p[:, :2] / np.where(np.abs(z) > 1e-12, z, 1e-12)[:, None]
    return uv, z


def _refine(X: np.ndarray, Ps: np.ndarray, uv: np.ndarray, iters: int = 6) -> np.ndarray:
    """Gauss-Newton on the reprojection error with Cauchy weights (robust to
    one wrong observation)."""
    for _ in range(iters):
        pu, z = _reproj(Ps, X)
        r = (pu - uv).reshape(-1)
        c = max(1.0, float(np.median(np.abs(r))) * 1.4826 * 2.0)
        w = 1.0 / (1.0 + (r / c) ** 2)
        J = []
        for P, zz in zip(Ps, z):
            zz = zz if abs(zz) > 1e-12 else 1e-12
            Xh = np.append(X, 1.0)
            p = P @ Xh
            Ju = (P[0, :3] * zz - p[0] * P[2, :3]) / zz ** 2
            Jv = (P[1, :3] * zz - p[1] * P[2, :3]) / zz ** 2
            J.append(Ju); J.append(Jv)
        J = np.asarray(J)
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r
        try:
            d = np.linalg.solve(H + 1e-9 * np.eye(3), -g)
        except np.linalg.LinAlgError:
            break
        X = X + d
        if np.linalg.norm(d) < 1e-7:
            break
    return X


def triangulate(tracks: Tracks, frames: Dict[int, dict], min_views: int,
                reproj_max_px: float) -> dict:
    """Per observation of every well-triangulated track: REAL frame, (u,v) at
    the tracks' resolution, z_tri (depth of the triangulated point in that
    camera), reprojection error, weight. DLT + robust refinement; a track
    needs ≥ min_views inlier views."""
    pm = _proj_matrices(frames, tracks.res_h, tracks.res_w)
    order = np.argsort(tracks.obs_track, kind="stable")
    tr = tracks.obs_track[order]
    bounds = np.flatnonzero(np.diff(tr)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(tr)]])
    o_frame, o_u, o_v, o_z, o_err, o_w, o_track = [], [], [], [], [], [], []
    n_tracks = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        fr = tracks.obs_frame[idx]
        keep = np.array([int(f) in pm for f in fr])
        if keep.sum() < min_views:
            continue
        idx = idx[keep]; fr = fr[keep]
        Ps = np.stack([pm[int(f)]["P"] for f in fr])
        uv = tracks.obs_uv[idx].astype(np.float64)
        X = _dlt(Ps, uv)
        if not np.all(np.isfinite(X)):
            continue
        X = _refine(X, Ps, uv)
        pu, z = _reproj(Ps, X)
        err = np.linalg.norm(pu - uv, axis=1)
        inl = (err <= float(reproj_max_px)) & (z > 1e-3)
        if inl.sum() < min_views:
            continue
        n_tracks += 1
        for k in np.flatnonzero(inl):
            o_frame.append(int(fr[k])); o_u.append(float(uv[k, 0])); o_v.append(float(uv[k, 1]))
            o_z.append(float(z[k])); o_err.append(float(err[k]))
            o_w.append(float(tracks.obs_weight[idx[k]])); o_track.append(int(tr[s]))
    return {"frame": np.asarray(o_frame, np.int64), "u": np.asarray(o_u), "v": np.asarray(o_v),
            "z_tri": np.asarray(o_z), "err_px": np.asarray(o_err), "weight": np.asarray(o_w),
            "track": np.asarray(o_track, np.int64), "n_tracks": n_tracks}


def _bilinear(depth: np.ndarray, u: np.ndarray, v: np.ndarray, edge_tol_rel: Optional[float] = None
              ) -> np.ndarray:
    """Depth at sub-pixel (u, v); NaN where a neighbour is invalid or — with
    ``edge_tol_rel`` — where the 2×2 neighbourhood spreads more than that
    fraction (a depth edge: an interpolated value there is a blend of two
    surfaces, not a measurement of either)."""
    H, W = depth.shape
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    u1, v1 = u0 + 1, v0 + 1
    ok = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)
    out = np.full(len(u), np.nan)
    if not ok.any():
        return out
    fu = (u - u0)[ok]; fv = (v - v0)[ok]
    d00 = depth[v0[ok], u0[ok]]; d01 = depth[v0[ok], u1[ok]]
    d10 = depth[v1[ok], u0[ok]]; d11 = depth[v1[ok], u1[ok]]
    valid = (d00 > 0) & (d01 > 0) & (d10 > 0) & (d11 > 0)
    if edge_tol_rel is not None:
        lo = np.minimum(np.minimum(d00, d01), np.minimum(d10, d11))
        hi = np.maximum(np.maximum(d00, d01), np.maximum(d10, d11))
        valid &= (hi - lo) <= float(edge_tol_rel) * np.where(lo > 0, lo, 1.0)
    val = (d00 * (1 - fu) * (1 - fv) + d01 * fu * (1 - fv) + d10 * (1 - fu) * fv + d11 * fu * fv)
    res = np.where(valid, val, np.nan)
    out[ok] = res
    return out


def track_observations(tri: dict, tracks: Tracks, frames: Dict[int, dict], sigma_rel: float,
                       edge_tol_rel: Optional[float] = None) -> List[Tuple[int, float, float, float]]:
    """(real_frame, z_obs, z_tri, weight): the frame's OWN depth at the
    track pixel vs the triangulated depth (no observation on depth edges)."""
    out = []
    for f in np.unique(tri["frame"]):
        sel = tri["frame"] == f
        fr = frames[int(f)]
        d = np.asarray(fr["depth"], np.float32)
        Hd, Wd = d.shape
        u = tri["u"][sel] * (Wd / float(tracks.res_w))
        v = tri["v"][sel] * (Hd / float(tracks.res_h))
        z_obs = _bilinear(d, u, v, edge_tol_rel)
        z_tri = tri["z_tri"][sel]
        w = tri["weight"][sel]
        ok = np.isfinite(z_obs) & (z_obs > 0)
        for zo, zt, ww in zip(z_obs[ok], z_tri[ok], w[ok]):
            out.append((int(f), float(zo), float(zt), float(ww) / (float(sigma_rel) * float(zo)) ** 2))
    return out


# ── mask contours where tracks are sparse ───────────────────────────────────

def _contour(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    m = mask.astype(bool)
    return m & ~binary_erosion(m, iterations=1)


def contour_observations(frames: Dict[int, dict], instances: List[dict], store,
                         images: Dict[int, np.ndarray], ccfg, pairs: Sequence[Tuple[int, int]],
                         seed: int = 0) -> List[Tuple[int, float, float, float]]:
    """(real_frame, z_obs, z_tri, weight) from mask-contour correspondences:
    contour samples of an instance in frame i are pushed along their ray
    over ±search_rel of the current depth; the depth whose projection lands
    on the SAME instance's contour in frame j (2-D Chamfer via a distance
    transform) is the observation. Only where the image gradient at the
    contour is at least min_gradient (a sharp edge), and only with a clear
    unique minimum."""
    import cv2
    rng = np.random.default_rng(seed)
    Hm, Wm = store.res
    out: List[Tuple[int, float, float, float]] = []
    ids = [int(i.get("instance_id", i.get("id"))) for i in instances]
    grads: Dict[int, np.ndarray] = {}

    def _grad(f):
        if f not in grads:
            img = images.get(f)
            if img is None:
                grads[f] = None
            else:
                g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if g.shape != (Hm, Wm):
                    g = cv2.resize(g, (Wm, Hm), interpolation=cv2.INTER_AREA)
                gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                grads[f] = np.hypot(gx, gy) / 4.0        # Sobel kernel sum → 0–255 units
        return grads[f]

    steps = int(ccfg.search_steps)
    rel = float(ccfg.search_rel)
    for iid in ids:
        obj = store.obj_of.get(iid)
        if obj is None:
            continue
        for (i, j) in pairs:
            # i, j are CLOUD/pose frames (that is what `frames` and `images`
            # are keyed by); the mask store is keyed by keyframe POSITION
            ki, kj = store.mask_key(i, obj), store.mask_key(j, obj)
            if ki is None or kj is None:
                continue
            if ki not in store.masks or kj not in store.masks or i not in frames or j not in frames:
                continue
            gi = _grad(i)
            if gi is None:
                continue
            ci = _contour(store.masks[ki]); cj = _contour(store.masks[kj])
            if not cj.any() or not ci.any():
                continue
            dt = cv2.distanceTransform((~cj).astype(np.uint8), cv2.DIST_L2, 3)
            pts = np.argwhere(ci & (gi >= float(ccfg.min_gradient)))
            if len(pts) == 0:
                continue
            pick = pts[rng.choice(len(pts), min(int(ccfg.samples_per_instance), len(pts)), replace=False)]
            fi, fj = frames[i], frames[j]
            di = np.asarray(fi["depth"], np.float32)
            Hd, Wd = di.shape
            Ki = np.asarray(fi["K"], np.float64); Ti = np.asarray(fi["T"], np.float64)
            Kj = np.asarray(fj["K"], np.float64); w2cj = np.linalg.inv(np.asarray(fj["T"], np.float64))
            sx, sy = Wm / float(Wd), Hm / float(Hd)
            for r, c in pick:
                ud, vd = c / sx, r / sy
                z0 = _bilinear(di, np.array([ud]), np.array([vd]), float(ccfg.sigma_rel))[0]
                if not np.isfinite(z0) or z0 <= 0:
                    continue
                zs = z0 * np.linspace(1.0 - rel, 1.0 + rel, steps)
                ray = np.array([(ud - Ki[0, 2]) / Ki[0, 0], (vd - Ki[1, 2]) / Ki[1, 1], 1.0])
                Pw = Ti[:3, 3][None] + (Ti[:3, :3] @ ray)[None] * zs[:, None]
                X = Pw @ w2cj[:3, :3].T + w2cj[:3, 3]
                zj = X[:, 2]
                ok = zj > 1e-3
                if ok.sum() < 3:
                    continue
                u = (Kj[0, 0] * X[:, 0] / np.where(ok, zj, 1.0) + Kj[0, 2]) * sx
                v = (Kj[1, 1] * X[:, 1] / np.where(ok, zj, 1.0) + Kj[1, 2]) * sy
                ui = np.clip(np.round(u).astype(int), 0, Wm - 1)
                vi = np.clip(np.round(v).astype(int), 0, Hm - 1)
                inb = ok & (u >= 0) & (u < Wm) & (v >= 0) & (v < Hm)
                dist = np.where(inb, dt[vi, ui], np.inf)
                k = int(np.argmin(dist))
                if not np.isfinite(dist[k]) or dist[k] > 1.5:
                    continue
                # unique minimum: the distance must rise clearly on both sides
                lo = dist[max(0, k - steps // 4)]; hi = dist[min(steps - 1, k + steps // 4)]
                if not (np.isfinite(lo) and np.isfinite(hi)) or min(lo, hi) < dist[k] + 1.0:
                    continue
                z_star = float(zs[k])
                if 0 < k < steps - 1 and np.isfinite(dist[k - 1]) and np.isfinite(dist[k + 1]):
                    a_, b_, c_ = dist[k - 1], dist[k], dist[k + 1]
                    den = a_ - 2 * b_ + c_
                    if den > 1e-9:
                        z_star = float(zs[k] + 0.5 * (a_ - c_) / den * (zs[1] - zs[0]))
                out.append((int(i), float(z0), z_star, 1.0 / (float(ccfg.sigma_rel) * float(z0)) ** 2))
    return out


# ── the pairwise sensor (current projections) ───────────────────────────────

def _pair_samples(wp_src: np.ndarray, valid_src: np.ndarray, depth_dst: np.ndarray, w2c_dst: np.ndarray,
                  K_dst: np.ndarray, max_samples: int, seed: int, edge_tol_rel: float, min_samples: int = 300):
    """Exact-surface depth samples between two frames (the association of
    metric_lock.depth_pair_samples) with the depth-edge rejection of the
    track sampler: src's points projected into dst, dst's OWN depth read at
    the hit pixel — unless the 2×2 neighbourhood there spreads more than
    ``edge_tol_rel`` (an association across a depth edge — the corner of two
    walls — is a blend of surfaces, the systematic bias of the pair sensor
    measured on clean frames). Returns (z_src_in_dst, z_dst) or None."""
    H, W = depth_dst.shape
    p = np.asarray(wp_src, np.float64).reshape(-1, 3)
    idx = np.flatnonzero(np.asarray(valid_src).reshape(-1))
    if len(idx) < min_samples:
        return None
    if len(idx) > int(max_samples):
        idx = np.random.default_rng(seed).choice(idx, int(max_samples), replace=False)
    p = p[idx]
    w2c = np.asarray(w2c_dst, np.float64)
    X = p @ w2c[:3, :3].T + w2c[:3, 3]
    z = X[:, 2]
    m = z > 1e-3
    if m.sum() < min_samples:
        return None
    fx, fy, cx, cy = float(K_dst[0, 0]), float(K_dst[1, 1]), float(K_dst[0, 2]), float(K_dst[1, 2])
    u = X[m, 0] / z[m] * fx + cx
    v = X[m, 1] / z[m] * fy + cy
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    inb = (u0 >= 0) & (u0 + 1 < W) & (v0 >= 0) & (v0 + 1 < H)
    if inb.sum() < min_samples:
        return None
    u0, v0, zs = u0[inb], v0[inb], z[m][inb]
    d = np.asarray(depth_dst, np.float64)
    q = np.stack([d[v0, u0], d[v0, u0 + 1], d[v0 + 1, u0], d[v0 + 1, u0 + 1]], 1)
    lo, hi = q.min(1), q.max(1)
    good = (lo > 1e-3) & ((hi - lo) <= float(edge_tol_rel) * lo)
    if good.sum() < min_samples:
        return None
    ui = np.clip(np.round(u[inb][good]).astype(int), 0, W - 1)
    vi = np.clip(np.round(v[inb][good]).astype(int), 0, H - 1)
    return zs[good], d[vi, ui]


def pairwise_rows(frames: Dict[int, dict], nums: List[int], offsets: Sequence[int], samples: int,
                  seed: int = 0, clip_sigma: float = 5.0, edge_tol_rel: float = 0.05
                  ) -> List[Tuple[int, int, float, float, float, int]]:
    """(i, j, alpha, beta, median_rel_before, n, scatter, z_med) for frame
    index pairs at the given offsets — the exact-surface pair sensor on the
    frames' own depth (unprojected with the final poses), depth edges
    rejected, metric_lock's robust affine relation per pair."""
    _vendor_on_path()
    from loop_utils.metric_lock import pair_depth_relation
    cache = {}

    def _wp(i):
        if i not in cache:
            f = frames[nums[i]]
            P = unproject(f["depth"], f["K"])
            T = np.asarray(f["T"], np.float64)
            wp = P @ T[:3, :3].T + T[:3, 3]
            valid = np.asarray(f["depth"]) > 0
            wp = np.where(valid[..., None], wp, 0.0)
            cache[i] = (wp, valid, np.linalg.inv(T), np.asarray(f["K"], np.float64),
                        np.asarray(f["depth"], np.float64))
        return cache[i]

    rows = []
    n = len(nums)
    for i in range(n):
        for d in offsets:
            j = i + int(d)
            if j >= n:
                continue
            wi, vi_, _, _, _ = _wp(i)
            _, _, w2cj, Kj, dj = _wp(j)
            zs = _pair_samples(wi, vi_, dj, w2cj, Kj, int(samples), seed, float(edge_tol_rel))
            if zs is None:
                continue
            rel = pair_depth_relation(zs[0], zs[1])
            if rel is None:
                continue
            al, be, before, cnt = rel
            # the row's own precision: the residual scatter AFTER the affine fit
            # (relative). A compressed frame fits its neighbour exactly (small
            # scatter, high weight); a corner frame whose two walls associate
            # across the depth edge keeps a structured residual (low weight).
            # Weighting by the mismatch BEFORE the fit did the opposite — the
            # rows that carried the biggest correction weighed least (measured)
            r = np.asarray(zs[1], np.float64) - (al * np.asarray(zs[0], np.float64) + be)
            mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            clip = max(float(clip_sigma) * mad, 1e-6)
            rms = float(np.sqrt(np.mean(np.minimum(r ** 2, clip ** 2))))
            scatter = rms / max(float(np.median(zs[1])), 1e-6)
            rows.append((i, j, float(al), float(be), float(before), int(cnt), float(scatter),
                         float(np.median(zs[1]))))
    return rows


def _robust_affine(z_obs: np.ndarray, z_tri: np.ndarray, w: np.ndarray, iters: int = 8):
    """z_tri ≈ alpha·z_obs + beta, weighted IRLS with Cauchy weights."""
    x = np.asarray(z_obs, np.float64); y = np.asarray(z_tri, np.float64); w0 = np.asarray(w, np.float64)
    w = w0.copy()
    a, b = 1.0, 0.0
    for _ in range(iters):
        sw = w.sum()
        if sw <= 0:
            return None
        mx = (x * w).sum() / sw; my = (y * w).sum() / sw
        vx = (w * (x - mx) ** 2).sum()
        if vx < 1e-12:
            return None
        a = (w * (x - mx) * (y - my)).sum() / vx
        b = my - a * mx
        r = y - (a * x + b)
        c = max(3.0 * 1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-4)
        w = w0 / (1.0 + (r / c) ** 2)
    return float(a), float(b)


def solve_depth(pair_rows, obs_by_index: Dict[int, list], n: int, scale_only: bool,
                min_obs: int, sigma_rel: float, pair_sigma_floor: float, prior_sigma: float = 0.0,
                zref: float = 1.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """The depth graph over pairwise rows PLUS the correspondence
    observations as rows against a VIRTUAL frame (the pose-consistent
    triangulated field); the solution is re-gauged so that virtual frame is
    identity. Rows are weighted by their precision: a pair row by its
    post-fit residual scatter, a track row by σ_rel/√n of its observations,
    the identity prior by prior_sigma. Returns (a, b, n_absolute_rows)."""
    _vendor_on_path()
    from loop_utils.metric_lock import solve_depth_graph
    V = n
    rows = [(int(i), int(j), al, be) for i, j, al, be, *_ in pair_rows]
    # scale-system weights (relative) and offset-system weights (metres: the
    # same relative precision at the row's depth)
    wts = [1.0 / max(float(r[6]), float(pair_sigma_floor)) for r in pair_rows]
    wts_b = [1.0 / (max(float(r[6]), float(pair_sigma_floor)) * max(float(r[7]), 1e-6)) for r in pair_rows]
    n_abs = 0
    for f, obs in obs_by_index.items():
        if len(obs) < int(min_obs):
            continue
        arr = np.asarray(obs, np.float64)
        fit = _robust_affine(arr[:, 0], arr[:, 1], arr[:, 2])
        if fit is None:
            continue
        rows.append((int(f), V, fit[0], fit[1]))
        wts.append(np.sqrt(len(obs)) / float(sigma_rel))
        wts_b.append(np.sqrt(len(obs)) / (float(sigma_rel) * max(float(np.median(arr[:, 1])), 1e-6)))
        n_abs += 1
    if prior_sigma > 0:
        # identity prior per frame against the virtual (pose-consistent)
        # frame: what the evidence cannot resolve stays identity
        for f in range(n):
            rows.append((int(f), V, 1.0, 0.0))
            wts.append(1.0 / float(prior_sigma))
            wts_b.append(1.0 / (float(prior_sigma) * float(zref)))
    a, b = solve_depth_graph(rows, n + 1, scale_only=scale_only, weights=wts, weights_b=wts_b)
    if n_abs or prior_sigma > 0:
        aV, bV = float(a[V]), float(b[V])
        a = a / aV
        b = (b - bV) / aV
    return np.asarray(a[:n]), np.asarray(b[:n]), n_abs


def _verdict_with_correspondences(a, b, held4, obs_by_index: Dict[int, list], dcfg, sigma_rel: float,
                                  min_obs: int) -> dict:
    """The held-out judgement when the defect can be LOCAL (a few frames
    compressed): the multi-view disagreement must go down where it exists
    (p90 over the held-out pairs, by the configured factor) and may not rise
    beyond the sensor's noise floor in the median; a frame with its own
    correspondence rows must agree with them (|log a − log α| within
    bound·σ), frames without keep the chain-integration guard of
    depth_graph_verdict (``bounded``)."""
    zref = float(dcfg.zref_m)
    floor = float(dcfg.pair_sigma_floor_rel)
    rb = np.array([abs(al * zref + be - zref) / zref for _f, _g, al, be in held4])
    ra = np.array([abs((a[f] * zref + b[f]) - (a[g] * (al * zref + be) + b[g])) / zref
                   for f, g, al, be in held4])
    p90_b, p90_a = float(np.percentile(rb, 90)), float(np.percentile(ra, 90))
    med_b, med_a = float(np.median(rb)), float(np.median(ra))
    improves = (p90_a <= float(dcfg.improve) * p90_b or p90_b <= floor) and (med_a <= med_b + floor)
    per_frame_ok = True
    worst = 0.0
    for f, obs in obs_by_index.items():
        if len(obs) < int(min_obs):
            continue                  # never a row of the solve — not a judge either
        arr = np.asarray(obs, np.float64)
        fit = _robust_affine(arr[:, 0], arr[:, 1], arr[:, 2])
        if fit is None:
            continue
        sig = float(sigma_rel) / np.sqrt(len(obs))
        dev = abs(float(np.log(a[f])) - float(np.log(max(fit[0], 1e-9))))
        worst = max(worst, dev / max(float(dcfg.bound) * sig + floor, 1e-12))
        if dev > float(dcfg.bound) * sig + floor:
            per_frame_ok = False
    return {"improves": bool(improves), "p90_before": p90_b, "p90_after": p90_a,
            "med_before": med_b, "med_after": med_a,
            "bounded": bool(per_frame_ok), "correspondence_bound_worst": float(worst)}


def depth_stage(frames: Dict[int, dict], wcfg, tracks: Optional[Tracks] = None,
                contour_obs: Optional[list] = None, log: Callable[[str], None] = print,
                seed: int = 0) -> dict:
    """Solve the per-frame affine depth correction of a session's keyframes.
    Returns {"a": {real: a}, "b": {real: b}, "applied": bool, "verdict": {...},
    "reason": str, counts}. Identity when the held-out verdict says no."""
    _vendor_on_path()
    from loop_utils.metric_lock import depth_graph_verdict
    dcfg = wcfg.depth
    t0 = time.time()
    nums = sorted(int(k) for k in frames)
    index = {f: i for i, f in enumerate(nums)}
    n = len(nums)
    rows = pairwise_rows(frames, nums, dcfg.pair_offsets, dcfg.pair_samples, seed=seed,
                         clip_sigma=dcfg.pair_scatter_clip_sigma, edge_tol_rel=wcfg.tracks.depth_edge_tol_rel)
    k = max(2, int(round(1.0 / float(dcfg.holdout_fraction))))
    held = [r for q, r in enumerate(rows) if q % k == 0]
    fit = [r for q, r in enumerate(rows) if q % k != 0]
    obs_by_index: Dict[int, list] = {}
    n_track = n_contour = 0
    if tracks is not None:
        tri = triangulate(tracks, frames, wcfg.tracks.min_views, wcfg.tracks.reproj_max_px)
        for f, zo, zt, w in track_observations(tri, tracks, frames, wcfg.tracks.sigma_rel,
                                               wcfg.tracks.depth_edge_tol_rel):
            obs_by_index.setdefault(index[f], []).append((zo, zt, w))
            n_track += 1
    for f, zo, zt, w in (contour_obs or []):
        if f in index:
            obs_by_index.setdefault(index[f], []).append((zo, zt, w))
            n_contour += 1
    identity = {"a": {f: 1.0 for f in nums}, "b": {f: 0.0 for f in nums}, "applied": False,
                "n_pair_rows": len(rows), "n_pair_rows_fit": len(fit), "n_pair_rows_held": len(held),
                "n_track_obs": n_track, "n_contour_obs": n_contour, "verdict": None,
                "pair_rel_median": (float(np.median([r[4] for r in rows])) if rows else None)}
    if len(fit) < int(dcfg.min_pairs) or len(held) < 1:
        identity["reason"] = (f"pairwise sensor starved: {len(fit)} fitted / {len(held)} held-out "
                              f"pair(s), min {dcfg.min_pairs}")
        log(f"[depth-tracks] IDENTITY — {identity['reason']}")
        return identity
    a, b, n_abs = solve_depth(fit, obs_by_index, n, dcfg.scale_only, wcfg.tracks.min_obs_per_frame,
                              wcfg.tracks.sigma_rel, dcfg.pair_sigma_floor_rel, dcfg.prior_sigma_rel,
                              float(dcfg.zref_m))
    # refinement: the pair sensor's association was made on the WRONG depth;
    # correct the depth, re-measure, solve the residual, compose
    for _ in range(int(dcfg.refine_iters) - 1):
        corrected = {}
        for f in nums:
            fr = frames[f]; d = np.asarray(fr["depth"], np.float32)
            valid = np.isfinite(d) & (d > 0)
            i = index[f]
            corrected[f] = {"depth": np.where(valid, d * a[i] + b[i], d).astype(np.float32),
                            "K": fr["K"], "T": fr["T"]}
        rows2 = pairwise_rows(corrected, nums, dcfg.pair_offsets, dcfg.pair_samples, seed=seed,
                              clip_sigma=dcfg.pair_scatter_clip_sigma,
                              edge_tol_rel=wcfg.tracks.depth_edge_tol_rel)
        fit2 = [r for q, r in enumerate(rows2) if q % k != 0]
        obs2 = {i: [(a[i] * zo + b[i], zt, w) for zo, zt, w in obs] for i, obs in obs_by_index.items()}
        if len(fit2) < int(dcfg.min_pairs):
            break
        a2, b2, _ = solve_depth(fit2, obs2, n, dcfg.scale_only, wcfg.tracks.min_obs_per_frame,
                                wcfg.tracks.sigma_rel, dcfg.pair_sigma_floor_rel, dcfg.prior_sigma_rel,
                              float(dcfg.zref_m))
        b = a2 * b + b2
        a = a2 * a
    meas4 = [(i, j, al, be) for i, j, al, be, *_ in fit]
    held4 = [(i, j, al, be) for i, j, al, be, *_ in held]
    verdict = depth_graph_verdict(a, b, meas4, held4, zref=float(dcfg.zref_m),
                                  improve=float(dcfg.improve), bound=float(dcfg.bound))
    verdict = {k_: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k_, v in verdict.items()}
    verdict.update(_verdict_with_correspondences(a, b, held4, obs_by_index, dcfg, wcfg.tracks.sigma_rel,
                                                 wcfg.tracks.min_obs_per_frame))
    # the correspondence observations judge too: their residual must go down
    # where it exists (p90) and may not rise beyond the noise floor (median)
    abs_before = abs_after = None
    corr_ok = True
    if obs_by_index:
        rb, ra = [], []
        for i, obs in obs_by_index.items():
            arr = np.asarray(obs, np.float64)
            rb.extend(np.abs(arr[:, 1] - arr[:, 0]) / arr[:, 1])
            ra.extend(np.abs(arr[:, 1] - (a[i] * arr[:, 0] + b[i])) / arr[:, 1])
        abs_before, abs_after = float(np.median(rb)), float(np.median(ra))
        p90b, p90a = float(np.percentile(rb, 90)), float(np.percentile(ra, 90))
        corr_ok = bool(p90a <= p90b and abs_after <= abs_before + float(dcfg.pair_sigma_floor_rel))
        verdict["correspondence_p90_before"], verdict["correspondence_p90_after"] = p90b, p90a
    significant = bool(np.max(np.abs(np.log(a))) > float(dcfg.pair_sigma_floor_rel)
                       or np.max(np.abs(b)) > float(dcfg.pair_sigma_floor_rel) * float(dcfg.zref_m))
    applied = bool(verdict["bounded"] and verdict["improves"] and significant and corr_ok)
    rep = dict(identity)
    rep.update({"verdict": verdict, "n_absolute_rows": n_abs,
                "correspondence_rel_before": abs_before, "correspondence_rel_after": abs_after,
                "elapsed_s": round(time.time() - t0, 1)})
    if applied:
        rep["a"] = {f: float(a[i]) for f, i in index.items()}
        rep["b"] = {f: float(b[i]) for f, i in index.items()}
        rep["applied"] = True
        rep["reason"] = "held-out disagreement improves and the corrections are bounded"
    elif not significant:
        rep["reason"] = (f"correction within the sensor noise (max |log a| {np.max(np.abs(np.log(a))):.5f}, "
                         f"max |b| {np.max(np.abs(b)):.4f} m) — identity")
    else:
        rep["reason"] = (f"held-out verdict: improves={verdict['improves']} bounded={verdict['bounded']} "
                         f"(median rel {verdict['med_before']:.4f} → {verdict['med_after']:.4f}, p90 "
                         f"{verdict['p90_before']:.4f} → {verdict['p90_after']:.4f})"
                         + ("" if abs_after is None else
                            f"; correspondence rel {abs_before:.4f} → {abs_after:.4f}"))
    log(f"[depth-tracks] {n} frames, {len(fit)} pair rows (+{len(held)} held-out), {n_track} track obs, "
        f"{n_contour} contour obs, {n_abs} absolute rows → {'APPLY' if applied else 'IDENTITY'}: "
        f"{rep['reason']}")
    return rep


def load_images(session_dir, frame_numbers: Sequence[int]) -> Dict[int, np.ndarray]:
    """frames/<name>.jpg per real frame (frame_list.json names); missing
    images are absent from the dict (the contour witness skips them)."""
    import cv2
    session_dir = Path(session_dir)
    names = {}
    flp = session_dir / "output" / "frame_list.json"
    if flp.exists():
        for nm in json.loads(flp.read_text()):
            digits = "".join(ch for ch in Path(str(nm)).stem if ch.isdigit())
            if digits:
                names[int(digits)] = str(nm)
    out = {}
    for f in frame_numbers:
        cand = [session_dir / "frames" / names[f]] if f in names else []
        cand += [session_dir / "frames" / f"{int(f):06d}.jpg", session_dir / "frames" / f"{int(f):06d}.png"]
        for p in cand:
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    out[int(f)] = img
                    break
    return out


def run_depth_stage(output_dir, session_dir, cfg=None, frames: Optional[Dict[int, dict]] = None,
                    tracks: Optional[Tracks] = None, log: Callable[[str], None] = print,
                    epoch: bool = True, operator: str = "auto", correction_cfg=None,
                    loop_pairs: Sequence[Tuple[int, int]] = ()) -> dict:
    """The §6.4 stage on a session: tracks (extracted if absent and enabled),
    contours (if enabled and images exist), solve, gate, apply as an epoch
    (per-keyframe affine along the rays through the correction package)."""
    from correction.session import load_session
    from reconstruction.loops.config import load_loops_config
    from reconstruction.witness.frames import load_session_frames
    from reconstruction.witness.mask_votes import load_mask_store
    cfg = cfg or load_loops_config()
    wcfg = cfg.witness
    output_dir, session_dir = Path(output_dir), Path(session_dir)
    frames = frames or load_session_frames(output_dir, log)
    if tracks is None and wcfg.tracks.enabled:
        tracks = ensure_tracks(output_dir, session_dir / "frames", wcfg.tracks, log)
    contour_obs = None
    res_path = output_dir / "segmentation_result.json"
    if wcfg.contours.enabled and res_path.exists():
        instances = json.loads(res_path.read_text()).get("instances") or []
        store = load_mask_store(output_dir)
        if instances and store is not None:
            nums = sorted(frames)
            pairs = [(nums[i], nums[i + d]) for d in wcfg.depth.pair_offsets for i in range(len(nums) - d)]
            pairs += [(int(i), int(j)) for i, j in loop_pairs]
            images = load_images(session_dir, nums)
            contour_obs = contour_observations(frames, instances, store, images, wcfg.contours, pairs)
    rep = depth_stage(frames, wcfg, tracks, contour_obs, log)
    if rep["applied"] and epoch:
        session = load_session(output_dir)
        _depth_epoch(output_dir, session, rep, operator, log, correction_cfg)
    (output_dir / "depth_tracks.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def affine_per_keyframe(rep: dict, frames_list: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """(k_kf, b_kf) in keyframe order from a depth_stage report."""
    a = rep["a"]; b = rep["b"]
    k_kf = np.array([float(a.get(int(f), a.get(str(int(f)), 1.0))) for f in frames_list])
    b_kf = np.array([float(b.get(int(f), b.get(str(int(f)), 0.0))) for f in frames_list])
    return k_kf, b_kf


def _depth_epoch(output_dir: Path, session, rep: dict, operator: str, log, ccfg=None):
    from correction.apply import stage_transaction, swap_transaction, assert_no_interrupted_swap
    from correction.config import load_correction_config
    from correction import ledger
    assert_no_interrupted_swap(output_dir)
    ccfg = ccfg or load_correction_config()
    N = session.n_kf
    k_kf, b_kf = affine_per_keyframe(rep, session.frames)
    R_kf = np.tile(np.eye(3), (N, 1, 1)); t_kf = np.zeros((N, 3))
    cid = ledger.new_correction_id()
    tx = stage_transaction(session, ccfg, R_kf, t_kf, k_kf, correction_id=cid, scale_diag_new=None,
                           floor_npz=None, log=log, progress=None, b_kf=b_kf)
    swap_transaction(output_dir, tx, log=log)
    rep_path = output_dir / "corrections" / f"report_{cid}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(rep, indent=1, default=float))
    v = rep["verdict"] or {}
    ledger.record_run(output_dir, correction_id=cid, epoch_from=tx["epoch_from"], epoch_to=tx["epoch_to"],
                      kind="depth_tracks", operator=operator, instance_ids=[], visits=[],
                      observability=[], diagnosis=[], anchors=[],
                      gates=[{"name": "holdout_depth_disagreement", "passed": bool(v.get("improves")),
                              "value": v.get("med_after"), "threshold": v.get("med_before")},
                             {"name": "bounded_corrections", "passed": bool(v.get("bounded")),
                              "value": None, "threshold": None}],
                      overrides={}, report_path=str(rep_path.relative_to(output_dir)))
    rep["correction_id"] = cid
    rep["epoch"] = tx["epoch_to"]
