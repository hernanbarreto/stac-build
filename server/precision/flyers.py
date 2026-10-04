"""Flyer diagnosis of a published cloud — Phase 0 of the mono-detail work (claude_stac.txt, 2026-10-04).

A FLYER is a point of the final cloud that is either
  * ISOLATED — its mean distance to its ``knn_k`` nearest neighbours, in units of its own pixel
    footprint (z / f: the spacing a surface has at that range), lies above the
    ``isolated_quantile`` percentile of the session's own k-NN distances — or
  * BETWEEN TWO CONFIRMED SURFACES — in the depth map of its OWN keyframe (the cloud's points
    re-projected to the pixels they came from) it sits inside a discontinuity band and at a depth
    strictly between the near and the far surface of that step.
The definition does not involve PointDiT; Phase 8's A/B measures the same thing with the flag on.

Every flyer is classified by its provenance, in this order of precedence:
  a  ``mixed_edge``          mixed pixel of its source frame: in the discontinuity band of the
                             calibrated depth with a depth between front and back
  b  ``view_inconsistent``   supported in its own frame, contradicted by the neighbouring keyframes
                             (``bend.neighbors``): more of them see a surface BEHIND the point than
                             agree with it, at the session's own agreement tolerance
  c  ``low_texture``         photometric score of its source pixel (gradient energy over
                             ``texture_window_px``) under the ``texture_quantile`` percentile of the
                             session's pixels, or a saturated highlight (mean gray >= ``specular_gray``)
  d  ``other``

Every bar is the session's own percentile (declared in ``precision.flyers``), never a fixed
number. Outputs, all under ``output/precision/``: ``flyers.json`` (counts, shares, the measured
bars), ``flyers.csv`` (one row per class), ``flyers_class.npy`` (uint8 per cloud point, 0 = not a
flyer, 1..4 = a..d, same row order as the cloud) and ``flyers.glb`` — the flyers as a coloured point
cloud the viewer shows as a layer (View → Flyers: a red, b orange, c blue, d grey). Provenance
``tool_measured``; nothing is deleted or moved.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_TAG = "[flyers]"
REPORT = "flyers.json"
CSV_NAME = "flyers.csv"
CLASS_NPY = "flyers_class.npy"
LAYER_GLB = "flyers.glb"
PROVENANCE = "tool_measured"
CLASSES = ("mixed_edge", "view_inconsistent", "low_texture", "other")
CLASS_LETTER = ("a", "b", "c", "d")
CLASS_RGB = np.array([[230, 30, 30], [245, 150, 20], [40, 90, 230], [140, 140, 140]], np.uint8)


class FlyersError(RuntimeError):
    pass


# ── pure helpers (tested on synthetic data) ───────────────────────────────

def depth_maps_from_cloud(xyz: np.ndarray, frame: np.ndarray, row: np.ndarray, col: np.ndarray,
                          frames: Sequence[int], w2c: Dict[int, np.ndarray], H: int, W: int
                          ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Per keyframe the depth map of the cloud's OWN points at the pixels they were born at
    (nearest point wins a shared pixel), the index map (row of the cloud, -1 = empty) and, per
    point, its depth in its own camera (0 for a point whose keyframe has no pose)."""
    zmap: Dict[int, np.ndarray] = {}
    imap: Dict[int, np.ndarray] = {}
    z_own = np.zeros(len(xyz), np.float64)
    order = np.argsort(frame, kind="stable")
    fs = frame[order]
    bounds = np.searchsorted(fs, np.asarray(frames), side="left"), np.searchsorted(fs, np.asarray(frames), side="right")
    for i, f in enumerate(frames):
        idx = order[bounds[0][i]:bounds[1][i]]
        z = np.full((H, W), 0.0, np.float32)
        im = np.full((H, W), -1, np.int64)
        if len(idx):
            M = w2c[f]
            zc = xyz[idx] @ M[2, :3] + M[2, 3]
            z_own[idx] = zc
            ok = (zc > 0) & (row[idx] >= 0) & (row[idx] < H) & (col[idx] >= 0) & (col[idx] < W)
            idx, zc = idx[ok], zc[ok]
            o = np.argsort(-zc, kind="stable")             # nearest written last → wins
            z[row[idx[o]], col[idx[o]]] = zc[o].astype(np.float32)
            im[row[idx[o]], col[idx[o]]] = idx[o]
        zmap[f] = z
        imap[f] = im
    return zmap, imap, z_own


def window_spread(z: np.ndarray, band_px: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per valid pixel the nearest and the farthest VALID depth of its (2·band_px+1)² window and
    their relative spread (z_back − z_front) / z_front. The cloud's depth maps are SPARSE (a voxel-
    downsampled cloud fills a fraction of the pixels), so a step is read over a window, never
    between two adjacent pixels."""
    from scipy.ndimage import maximum_filter, minimum_filter
    size = 2 * int(band_px) + 1
    valid = z > 0
    big = np.where(valid, z, np.inf)
    small = np.where(valid, z, -np.inf)
    z_front = minimum_filter(big, size=size, mode="nearest")
    z_back = maximum_filter(small, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.where(valid & np.isfinite(z_front) & np.isfinite(z_back),
                          (z_back - z_front) / np.maximum(z_front, 1e-9), 0.0)
    return z_front, z_back, spread


def step_threshold(zmaps: Dict[int, np.ndarray], band_px: int, quantile: float,
                   rng: np.random.Generator, sample: int) -> float:
    """The session's own discontinuity bar: the ``quantile`` percentile of the window spread over
    every valid pixel (a sample of ``sample`` of them)."""
    pool: List[np.ndarray] = []
    n = 0
    for z in zmaps.values():
        _, _, sp = window_spread(z, band_px)
        v = sp[z > 0]
        if len(v):
            pool.append(v); n += len(v)
    if n == 0:
        raise FlyersError("no valid pixel in any depth map — no step can be measured")
    allv = np.concatenate(pool)
    if len(allv) > sample:
        allv = allv[rng.choice(len(allv), sample, replace=False)]
    return float(np.percentile(allv, quantile))


def discontinuity_band(z: np.ndarray, tau_step: float, band_px: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(band, z_front, z_back): the valid pixels whose window spread exceeds ``tau_step`` (a step
    within ``band_px`` of them), with the window's nearest and farthest depth."""
    z_front, z_back, spread = window_spread(z, band_px)
    band = (z > 0) & (spread > tau_step)
    return band, z_front, z_back


def between_surfaces(z: np.ndarray, band: np.ndarray, z_front: np.ndarray, z_back: np.ndarray,
                     tau_step: float) -> np.ndarray:
    """Band pixels whose own depth lies strictly between the window's front and back surfaces
    (clear of both by the step bar)."""
    return band & (z > 0) & (z > z_front * (1.0 + tau_step)) & (z < z_back * (1.0 - tau_step))


def isolation(xyz: np.ndarray, k: int, quantile: float, rng: np.random.Generator, sample: int,
              footprint: Optional[np.ndarray] = None, workers: int = 1) -> Tuple[np.ndarray, float]:
    """(isolated mask, bar): mean distance to the ``k`` nearest OTHER points, in units of the
    point's own pixel FOOTPRINT (``footprint`` = z / f of its source camera: the spacing a surface
    has at that range by construction — a far wall is sparser than a near desk, not more isolated),
    isolated above the ``quantile`` percentile of that ratio over a sample of the cloud. Without a
    footprint the raw distance is used."""
    from scipy.spatial import cKDTree
    n = len(xyz)
    if n <= k:
        return np.zeros(n, bool), float("nan")
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1, workers=workers)
    dk = d[:, 1:].mean(1)
    if footprint is not None:
        dk = dk / np.maximum(np.asarray(footprint, np.float64), 1e-9)
    smp = dk if n <= sample else dk[rng.choice(n, sample, replace=False)]
    bar = float(np.percentile(smp, quantile))
    return dk > bar, bar


def project(X: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, z) of world points in a camera (z may be ≤ 0: behind it)."""
    Pc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    zz = np.where(np.abs(z) > 1e-12, z, 1e-12)
    u = K[0, 0] * Pc[:, 0] / zz + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / zz + K[1, 2]
    return u, v, z


def view_votes(X: np.ndarray, src: np.ndarray, frames: Sequence[int], zmaps: Dict[int, np.ndarray],
               w2c: Dict[int, np.ndarray], K: np.ndarray, neighbors: Sequence[int], tau_view: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Per point (agree, contra) over the neighbouring keyframes of its source: a neighbour whose
    own depth at the projected pixel is within ``tau_view`` (relative) AGREES; one whose depth is
    farther than the point by more than that sees free space through it and CONTRADICTS. A
    neighbour with no depth there, or seeing a nearer surface (the point is occluded), is silent."""
    pos = {f: i for i, f in enumerate(frames)}
    H, W = next(iter(zmaps.values())).shape
    agree = np.zeros(len(X), np.int32)
    contra = np.zeros(len(X), np.int32)
    for f in np.unique(src):
        sel = np.nonzero(src == f)[0]
        i = pos.get(int(f))
        if i is None:
            continue
        for off in neighbors:
            j = i + int(off)
            if j < 0 or j >= len(frames):
                continue
            g = frames[j]
            u, v, z = project(X[sel], w2c[g], K)
            ui = np.rint(u).astype(np.int64); vi = np.rint(v).astype(np.int64)
            ok = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not ok.any():
                continue
            zn = zmaps[g][vi[ok], ui[ok]]
            seen = zn > 0
            rel = (zn - z[ok]) / np.maximum(z[ok], 1e-9)
            a = seen & (np.abs(rel) <= tau_view)
            c = seen & (rel > tau_view)
            idx = sel[ok]
            agree[idx[a]] += 1
            contra[idx[c]] += 1
    return agree, contra


def view_tolerance(X: np.ndarray, src: np.ndarray, frames: Sequence[int], zmaps: Dict[int, np.ndarray],
                   w2c: Dict[int, np.ndarray], K: np.ndarray, neighbors: Sequence[int],
                   quantile: float) -> float:
    """The session's own view-agreement bar: the ``quantile`` percentile of |z_neighbour − z| / z
    over the sampled (interior) points their neighbours see — the construction of epoch 8's τ."""
    pos = {f: i for i, f in enumerate(frames)}
    H, W = next(iter(zmaps.values())).shape
    pool: List[np.ndarray] = []
    for f in np.unique(src):
        sel = np.nonzero(src == f)[0]
        i = pos.get(int(f))
        if i is None:
            continue
        for off in neighbors:
            j = i + int(off)
            if j < 0 or j >= len(frames):
                continue
            g = frames[j]
            u, v, z = project(X[sel], w2c[g], K)
            ui = np.rint(u).astype(np.int64); vi = np.rint(v).astype(np.int64)
            ok = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not ok.any():
                continue
            zn = zmaps[g][vi[ok], ui[ok]]
            seen = zn > 0
            if seen.any():
                pool.append(np.abs(zn[seen] - z[ok][seen]) / np.maximum(z[ok][seen], 1e-9))
    if not pool:
        raise FlyersError("no sampled point is seen by a neighbouring keyframe — the view tolerance "
                          "cannot be measured")
    return float(np.percentile(np.concatenate(pool), quantile))


def texture_maps(gray: np.ndarray, window_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """(gradient energy, mean gray) over a ``window_px`` box of a gray image (float 0..255)."""
    from scipy.ndimage import sobel, uniform_filter
    g = gray.astype(np.float64)
    gx, gy = sobel(g, axis=1), sobel(g, axis=0)
    size = max(1, int(window_px))
    return uniform_filter(gx * gx + gy * gy, size=size, mode="nearest"), uniform_filter(g, size=size, mode="nearest")


def classify(is_flyer: np.ndarray, mixed: np.ndarray, agree: np.ndarray, contra: np.ndarray,
             low_texture: np.ndarray) -> np.ndarray:
    """uint8 per point: 0 not a flyer, 1 a (mixed edge), 2 b (contra > agree), 3 c (low texture or
    highlight), 4 d (other) — a before b before c."""
    cls = np.zeros(len(is_flyer), np.uint8)
    f = is_flyer
    cls[f & mixed] = 1
    b = f & (cls == 0) & (contra > agree)
    cls[b] = 2
    c = f & (cls == 0) & low_texture
    cls[c] = 3
    cls[f & (cls == 0)] = 4
    return cls


def counts(cls: np.ndarray) -> Dict[str, dict]:
    n = int(len(cls)); nf = int((cls > 0).sum())
    out = {"points": n, "flyers": nf, "flyer_share": (nf / n if n else 0.0), "by_class": {}}
    for k, name in enumerate(CLASSES, start=1):
        c = int((cls == k).sum())
        out["by_class"][name] = {"letter": CLASS_LETTER[k - 1], "count": c,
                                 "share_of_flyers": (c / nf if nf else 0.0), "share_of_points": (c / n if n else 0.0)}
    return out


def write_layer_glb(path: Path, xyz: np.ndarray, cls: np.ndarray) -> Path:
    """The flyers as a coloured point cloud (GLB, POINTS primitive) for the viewer's layer."""
    import trimesh
    m = cls > 0
    pts = xyz[m].astype(np.float32)
    rgb = np.concatenate([CLASS_RGB[cls[m] - 1], np.full((int(m.sum()), 1), 255, np.uint8)], 1)
    if len(pts) == 0:
        pts = np.zeros((1, 3), np.float32); rgb = np.array([[0, 0, 0, 0]], np.uint8)
    pc = trimesh.PointCloud(pts, colors=rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    pc.export(str(path), file_type="glb")
    return path


def write_csv(path: Path, session: str, rep: dict) -> Path:
    rows = [{"session": session, "class": name, "letter": d["letter"], "count": d["count"],
             "share_of_flyers_pct": round(100 * d["share_of_flyers"], 3),
             "share_of_points_pct": round(100 * d["share_of_points"], 4)}
            for name, d in rep["by_class"].items()]
    rows.append({"session": session, "class": "all_flyers", "letter": "", "count": rep["flyers"],
                 "share_of_flyers_pct": 100.0, "share_of_points_pct": round(100 * rep["flyer_share"], 4)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path


# ── the session ────────────────────────────────────────────────────────────

def load_session(session_dir: Path, log: Callable = print):
    """The live cloud with its provenance, the camera that built it and its poses."""
    from correction.session import read_ply
    from precision.camera import load_camera_json
    from precision.depth_sweep import _read_poses
    out = Path(session_dir) / "output"
    ply = out / "cleaned_cloud.ply"
    if not ply.exists():
        raise FlyersError(f"{ply} is missing — no published cloud to diagnose")
    _, data = read_ply(ply)
    names = data.dtype.names or ()
    for k in ("frame_global", "pixel_row", "pixel_col"):
        if k not in names:
            raise FlyersError(f"{ply.name} carries no per-point provenance ('{k}') — the flyer "
                              f"classes are defined by the source pixel")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    frame = np.asarray(data["frame_global"]).astype(np.int64)
    row = np.asarray(data["pixel_row"]).astype(np.int64)
    col = np.asarray(data["pixel_col"]).astype(np.int64)
    cam = load_camera_json(out / "camera.json")
    frames, c2w = _read_poses(out / "camera_poses.txt", out / "camera_frames.txt")
    w2c = {f: np.linalg.inv(c2w[i]) for i, f in enumerate(frames)}
    log(f"{LOG_TAG} {len(xyz):,} points from {len(np.unique(frame)):,} keyframes; camera "
        f"{cam.width}x{cam.height} fx {cam.params[0]:.1f}; {len(frames)} poses")
    return xyz, frame, row, col, cam, [int(f) for f in frames], w2c


def texture_scores(session_dir: Path, cam, frames: Sequence[int], frame: np.ndarray, row: np.ndarray,
                   col: np.ndarray, window_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per cloud point the gradient energy and the mean gray at its source pixel (frames resized to the
    camera grid when they are not on it, as the cloud's pixels are)."""
    from PIL import Image
    from intake.content import frame_file
    frames_dir = Path(session_dir) / "frames"
    H, W = int(cam.height), int(cam.width)
    energy = np.full(len(frame), np.nan); gray_mean = np.full(len(frame), np.nan)
    order = np.argsort(frame, kind="stable")
    fs = frame[order]
    for f in np.unique(frame):
        a, b = np.searchsorted(fs, f, "left"), np.searchsorted(fs, f, "right")
        idx = order[a:b]
        try:
            img = Image.open(frame_file(frames_dir, int(f))).convert("L")
        except (FileNotFoundError, OSError):
            continue
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        g = np.asarray(img, np.float64)
        e, m = texture_maps(g, window_px)
        ok = (row[idx] >= 0) & (row[idx] < H) & (col[idx] >= 0) & (col[idx] < W)
        energy[idx[ok]] = e[row[idx[ok]], col[idx[ok]]]
        gray_mean[idx[ok]] = m[row[idx[ok]], col[idx[ok]]]
    return energy, gray_mean


def diagnose(xyz, frame, row, col, K, H, W, frames, w2c, fcfg, neighbors, energy, gray_mean,
             log: Callable = print, workers: int = 1) -> Tuple[np.ndarray, dict]:
    """The whole measurement on arrays (no I/O): (class per point, the measured bars)."""
    rng = np.random.default_rng(int(fcfg.seed))
    t0 = time.time()
    zmaps, _, z_own = depth_maps_from_cloud(xyz, frame, row, col, frames, w2c, H, W)
    tau_step = step_threshold(zmaps, int(fcfg.band_px), float(fcfg.step_quantile), rng, int(fcfg.sample_points))
    mixed = np.zeros(len(xyz), bool)
    in_band = np.zeros(len(xyz), bool)
    order = np.argsort(frame, kind="stable"); fs = frame[order]
    for f in frames:
        a, b = np.searchsorted(fs, f, "left"), np.searchsorted(fs, f, "right")
        idx = order[a:b]
        if not len(idx):
            continue
        band, zf, zb = discontinuity_band(zmaps[f], tau_step, int(fcfg.band_px))
        bs = between_surfaces(zmaps[f], band, zf, zb, tau_step)
        ok = (row[idx] >= 0) & (row[idx] < H) & (col[idx] >= 0) & (col[idx] < W)
        in_band[idx[ok]] = band[row[idx[ok]], col[idx[ok]]]
        mixed[idx[ok]] = bs[row[idx[ok]], col[idx[ok]]]
    log(f"{LOG_TAG} step bar {tau_step * 100:.2f} % (p{fcfg.step_quantile:g} of the window spread); "
        f"{in_band.mean() * 100:.2f} % of the points in a discontinuity band, {mixed.mean() * 100:.3f} % between "
        f"two surfaces ({time.time() - t0:.0f} s)")
    t1 = time.time()
    footprint = np.abs(z_own) / float(K[0, 0])              # one pixel at the point's own range
    isolated, iso_bar = isolation(xyz, int(fcfg.knn_k), float(fcfg.isolated_quantile), rng,
                                  int(fcfg.sample_points), footprint=footprint, workers=workers)
    log(f"{LOG_TAG} isolation bar {iso_bar:.2f} pixel footprints (p{fcfg.isolated_quantile:g} of the {fcfg.knn_k}-NN "
        f"mean distance over z/f); {isolated.mean() * 100:.2f} % isolated ({time.time() - t1:.0f} s)")
    is_flyer = isolated | mixed
    # the view tolerance on interior points (not in a band, not isolated), as epoch 8 measured τ
    interior = np.nonzero(~in_band & ~isolated)[0]
    if len(interior) > int(fcfg.sample_points):
        interior = rng.choice(interior, int(fcfg.sample_points), replace=False)
    tau_view = view_tolerance(xyz[interior], frame[interior], frames, zmaps, w2c, K, neighbors,
                              float(fcfg.view_quantile))
    fl = np.nonzero(is_flyer)[0]
    agree_f, contra_f = view_votes(xyz[fl], frame[fl], frames, zmaps, w2c, K, neighbors, tau_view)
    agree = np.zeros(len(xyz), np.int32); contra = np.zeros(len(xyz), np.int32)
    agree[fl] = agree_f; contra[fl] = contra_f
    fin = np.isfinite(energy)
    if not fin.any():
        raise FlyersError("no frame image could be read — the photometric class cannot be measured")
    smp = energy[fin]
    if len(smp) > int(fcfg.sample_points):
        smp = smp[rng.choice(len(smp), int(fcfg.sample_points), replace=False)]
    tex_bar = float(np.percentile(smp, float(fcfg.texture_quantile)))
    low_tex = fin & ((energy <= tex_bar) | (np.nan_to_num(gray_mean, nan=0.0) >= float(fcfg.specular_gray)))
    cls = classify(is_flyer, mixed, agree, contra, low_tex)
    bars = {"tau_step_rel": tau_step, "isolation_footprints": iso_bar, "tau_view_rel": tau_view,
            "texture_energy": tex_bar, "n_interior_sampled": int(len(interior)),
            "flyers_isolated": int(isolated.sum()), "flyers_between_surfaces": int(mixed.sum()),
            "points_in_band": int(in_band.sum()), "seconds": round(time.time() - t0, 1)}
    log(f"{LOG_TAG} view bar {tau_view * 100:.2f} % (p{fcfg.view_quantile:g} of the neighbour disagreement, "
        f"{len(interior):,} interior points); texture bar {tex_bar:.1f} (p{fcfg.texture_quantile:g})")
    return cls, bars


def run(session_dir: Path, pcfg, log: Callable = print, write: bool = True, workers: int = 1) -> dict:
    """Diagnose the live cloud of ``session_dir``; write the report, the CSV, the class array and the
    viewer layer under output/precision/ (``write``)."""
    session_dir = Path(session_dir)
    fcfg = pcfg.flyers
    xyz, frame, row, col, cam, frames, w2c = load_session(session_dir, log)
    H, W = int(cam.height), int(cam.width)
    K = np.asarray(cam.K(), np.float64)
    energy, gray_mean = texture_scores(session_dir, cam, frames, frame, row, col, int(fcfg.texture_window_px))
    cls, bars = diagnose(xyz, frame, row, col, K, H, W, frames, w2c, fcfg, pcfg.bend.neighbors,
                         energy, gray_mean, log=log, workers=workers)
    rep = {"version": 1, "stage": "flyers", "provenance": PROVENANCE, "session": session_dir.name,
           "params": {k: getattr(fcfg, k) for k in ("knn_k", "isolated_quantile", "step_quantile", "band_px",
                                                     "view_quantile", "texture_quantile", "texture_window_px",
                                                     "specular_gray", "sample_points", "seed")},
           "neighbors": list(int(x) for x in pcfg.bend.neighbors), "measured": bars, **counts(cls)}
    a = rep["by_class"]["mixed_edge"]
    log(f"{LOG_TAG} {rep['flyers']:,} flyers of {rep['points']:,} points ({rep['flyer_share'] * 100:.2f} %): "
        + ", ".join(f"{d['letter']} {name} {d['count']:,} ({d['share_of_flyers'] * 100:.1f} %)"
                    for name, d in rep["by_class"].items()))
    print(f"{LOG_TAG} class (a) mixed edge: {a['share_of_flyers'] * 100:.1f} % of the flyers "
          f"({a['share_of_points'] * 100:.3f} % of the points)", flush=True)
    if write:
        pdir = session_dir / "output" / "precision"
        pdir.mkdir(parents=True, exist_ok=True)
        np.save(pdir / CLASS_NPY, cls)
        write_csv(pdir / CSV_NAME, session_dir.name, rep)
        write_layer_glb(pdir / LAYER_GLB, xyz, cls)
        rep["files"] = {"report": str(pdir / REPORT), "csv": str(pdir / CSV_NAME), "classes": str(pdir / CLASS_NPY),
                        "layer": str(pdir / LAYER_GLB)}
        (pdir / REPORT).write_text(json.dumps(rep, indent=1, default=float))
        log(f"{LOG_TAG} written {pdir / REPORT}, {CSV_NAME}, {CLASS_NPY}, {LAYER_GLB}")
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", nargs="+", required=True, help="session directories (each holds output/)")
    ap.add_argument("--csv", default=None, help="combined CSV over every session given")
    ap.add_argument("--workers", type=int, default=1, help="BOUND (cost): threads of the k-NN query")
    ap.add_argument("--no-write", action="store_true", help="measure and print only")
    args = ap.parse_args(argv)
    from precision.config import load_precision_config
    pcfg = load_precision_config()
    rows = []
    for s in args.session:
        try:
            rep = run(Path(s), pcfg, write=not args.no_write, workers=args.workers)
        except FlyersError as e:
            print(f"{LOG_TAG} {s}: NOT measured — {e}", flush=True)
            continue
        for name, d in rep["by_class"].items():
            rows.append({"session": rep["session"], "class": name, "letter": d["letter"], "count": d["count"],
                         "share_of_flyers_pct": round(100 * d["share_of_flyers"], 3),
                         "share_of_points_pct": round(100 * d["share_of_points"], 4)})
        rows.append({"session": rep["session"], "class": "all_flyers", "letter": "", "count": rep["flyers"],
                     "share_of_flyers_pct": 100.0, "share_of_points_pct": round(100 * rep["flyer_share"], 4)})
    if args.csv and rows:
        p = Path(args.csv); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"{LOG_TAG} combined CSV → {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
