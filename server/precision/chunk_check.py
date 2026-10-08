"""Per-keyframe and per-chunk verification of the corrected geometry against the FLOOR
and the CEILING, with DA3 as the independent instrument — what says whether something
has to be corrected, and WHAT (USER 2026-09-29: *"el sistema en el pipeline no puede
verificar esas diferencias para ver si tiene que corregir algo? … por chunk … verificación
interna e intrachunk, ambos, construila"*).

Measured per keyframe from the DEPTH MAPS and the refined poses (no cloud needed, so it
runs before anything is published):

    floor_h   height of the keyframe's floor band over the session's dominant floor plane
              (Omega depth × s_k, the session camera, the refined pose)
    ceil_h    the same for its ceiling band
    cam_h     the camera centre over that plane
    cf_omega  camera-to-floor distance by Omega          (cam_h − floor_h)
    cf_da3    camera-to-floor distance by DA3's own metric depth, same pose

Per chunk (the unit of Omega's gauge) and per keyframe (intra-chunk, against the chunk's
own trend along the walk). The bar is the sample's own noise — a bootstrap interval at the
declared confidence, never an invented threshold:

    depth   cf_omega / cf_da3 departs from the session's ratio → Omega's depth in that
            chunk is long/short by r  → route: depth scale about the camera (2026-09-19)
    pose    at a seam the floor jumps AND the ceiling jumps with it → the cameras of one
            side are off vertically   → route: vertical pose alignment per keyframe
    level   at a seam the floor jumps, the ceiling does not → a real level change,
            nothing to correct
    ok      nothing departs
    undecided  the floor departs but neither instrument can say why — said, not assumed

pccr 2026-09-29, the case that asked for this: chunk 0 (kf 0–62) had its floor 12–15 cm
under the rest while Omega and DA3 agreed on camera-to-floor (150 cm vs 130 elsewhere):
not depth; the per-keyframe floor solver had kept the 13.6 cm as a "level change" on an
invented repeatability bar, and only the user's testimony said it was one floor.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

CHECK_NAME = "chunk_check.json"
LOG_TAG = "[chunk-check]"
PROVENANCE = "tool_measured"
VERDICTS = ("ok", "depth", "pose", "level", "undecided")
UP = np.array([0.0, 1.0, 0.0])
_MAD_TO_SIGMA = 1.0 / 0.6744897501960817      # Φ⁻¹(3/4): MAD → σ of a normal sample


class ChunkCheckError(RuntimeError):
    """A structural impossibility of the check — always with the exact reason."""


# ── geometry helpers ─────────────────────────────────────────────────────

def unproject(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, stride: int) -> np.ndarray:
    """(N,3) world points of the valid pixels of a depth map, every ``stride``-th pixel."""
    d = np.asarray(depth, np.float64)[::stride, ::stride]
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64) * stride
    ok = np.isfinite(d) & (d > 0)
    X = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)[ok]
    T = np.asarray(c2w, np.float64)
    return X @ T[:3, :3].T + T[:3, 3]


def band_height(h: np.ndarray, pct: float, band_m: float, min_points: int) -> Tuple[float, int, float]:
    """(median height, n, spread) of the band around the ``pct`` percentile of ``h`` (the
    floor for a low percentile, the ceiling for a high one); the spread is the band's
    robust σ (1.4826·MAD) — the resolution with which this keyframe measures that surface.
    NaN when the band holds fewer than ``min_points`` samples — the keyframe does not see it."""
    h = np.asarray(h, np.float64)
    h = h[np.isfinite(h)]
    if h.size < min_points:
        return float("nan"), int(h.size), float("nan")
    q = np.percentile(h, pct)
    band = h[np.abs(h - q) <= band_m]
    if band.size < min_points:
        return float("nan"), int(band.size), float("nan")
    med = float(np.median(band))
    return med, int(band.size), float(_MAD_TO_SIGMA * np.median(np.abs(band - med)))


class NoFloorPlane(ChunkCheckError):
    """The pooled low bands hold too few points for ANY plane (fewer than three). Since
    2026-10-07 (docs/plan_determinismo.md point 140) there is no acceptance bar any more: the
    best plane is always fitted and reported with its inlier share and interval (zaragoza
    2026-10-05 measured 19.0 % against a 20 % bar and the whole session was UNDECIDED for one
    point). The check REPORTS this case; it is not a failure of the reconstruction and must not
    stop the pipeline."""

    def __init__(self, msg: str, best_frac: float, n_points: int):
        super().__init__(msg)
        self.best_frac = float(best_frac)
        self.n_points = int(n_points)


def dominant_plane(points: np.ndarray, band_m: float, seed: int, confidence: float,
                   n_boot: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """The session's floor plane (unit normal towards +Y, a point on it, its report) — the SEEDED
    numpy RANSAC (point 50) over the pooled floor bands of every keyframe, WITHOUT an acceptance
    bar (point 140): the best plane is taken whatever share of the band it holds, and that share
    is REPORTED with its bootstrap interval at ``confidence`` (resampling the pooled points, the
    plane fixed) so the reader sees how dominant the floor is. Only fewer than three points leave
    no plane to fit (NoFloorPlane, declared)."""
    from reconstruction.geometry.primitives import fit_plane_ransac
    P = np.asarray(points, np.float64)
    if len(P) > 400_000:
        P = P[np.random.default_rng(seed).choice(len(P), 400_000, replace=False)]
    thr = band_m / 4
    pf = fit_plane_ransac(P, dist_thresh=thr, iters=400, min_inlier_frac=0.0,
                          measure_curvature=False, seed=int(seed))
    if pf is None:
        raise NoFloorPlane(f"no plane can be fitted to {len(P):,} low-band point(s) (three are needed)",
                           0.0, int(len(P)))
    inl = P[pf.inliers]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][2]      # least-squares normal of the inliers
    n = n / np.linalg.norm(n)
    if n[1] < 0:
        n = -n
    within = np.abs((P - c) @ n) <= thr
    frac = float(within.mean())
    rng = np.random.default_rng(int(seed))
    a = (1.0 - float(confidence)) / 2.0
    fr = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        fr[b] = float(within[rng.integers(0, len(P), len(P))].mean())
    info = {"inlier_frac": frac, "inlier_frac_ci": [float(np.quantile(fr, a)), float(np.quantile(fr, 1 - a))],
            "confidence": float(confidence), "n_points": int(len(P)), "dist_thresh_m": float(thr),
            "acceptance_bar": "none (point 140): the plane is always fitted and reported with its share; "
                              "the per-chunk verdicts carry their own judges"}
    return n, c, info


# ── the bar: the sample's own noise ──────────────────────────────────────

def ci_median(x: np.ndarray, confidence: float, seed: int, n_boot: int) -> Tuple[float, float]:
    """Bootstrap interval of the median of ``x`` (fixed seed → reproducible)."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, x.size, (n_boot, x.size))], axis=1)
    a = (1.0 - confidence) / 2.0
    return float(np.quantile(meds, a)), float(np.quantile(meds, 1.0 - a))


# ── the measurement ──────────────────────────────────────────────────────

@dataclass
class Row:
    i: int
    frame: int
    chunk: int
    chainage: float
    floor_h: float
    n_floor: int
    ceil_h: float
    n_ceil: int
    cam_h: float
    cf_omega: float
    cf_da3: float
    s_k: float
    floor_res: float = float("nan")     # the keyframe's floor-band spread (its resolution)
    ceil_res: float = float("nan")


def _epochs(out: Path) -> Dict[str, Optional[int]]:
    ge = out / "geometry_epoch.json"
    cam = out / "camera.json"
    g = json.loads(ge.read_text()).get("epoch") if ge.exists() else None
    c = json.loads(cam.read_text()).get("camera_epoch") if cam.exists() else None
    return {"geometry_epoch": g, "camera_epoch": c}


def load_inputs(session_dir: Path, log: Callable = print):
    """Frames, poses, camera, s_k, chunk per keyframe, chainage — from the session."""
    out = Path(session_dir) / "output"
    fp, pp = out / "camera_frames.txt", out / "camera_poses.txt"
    if not fp.exists() or not pp.exists():
        raise ChunkCheckError(f"{fp.name} / {pp.name} missing — no keyframe poses to check")
    frames = [int(float(x)) for x in fp.read_text().split()]
    c2w = np.loadtxt(pp).reshape(-1, 4, 4)
    if len(c2w) != len(frames):
        raise ChunkCheckError(f"{len(frames)} keyframes but {len(c2w)} poses")
    cam_json = out / "camera.json"
    if not cam_json.exists():
        raise ChunkCheckError("camera.json missing — F0 did not run")
    cam = json.loads(cam_json.read_text())
    fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    # Omega's record grid and the camera's lens are handled where the record is read (measure_rows
    # carries it onto the undistorted native grid when they differ — zaragoza 2026-10-05: 1920x1088
    # records, k1 -0.0035); this check used to refuse a grid whose x differed and miss one whose y did
    s_k = {f: 1.0 for f in frames}
    bend = {f: (0.0, 0.0) for f in frames}
    offset = {f: 0.0 for f in frames}
    cloud_epoch = None
    s_k_source = "absent (1.0 — neither the corrected cloud nor F6 measured the per-keyframe scale)"
    for rel in ("corrected_cloud.json", "depth_native/report.json"):
        rep = out / rel
        if not rep.exists():
            continue
        doc = json.loads(rep.read_text())
        pf = doc.get("per_frame") or {}
        if all(str(f) in pf and "s_k" in pf[str(f)] for f in frames):
            s_k = {f: float(pf[str(f)]["s_k"]) for f in frames}
            # the depth on F5 (f6_bend) publishes k(u, v) = s_k + c1·u' + c2·v' per keyframe:
            # the check measures the cloud that was PUBLISHED, bend included
            bend = {f: tuple(float(x) for x in (pf[str(f)].get("bend") or (0.0, 0.0))) for f in frames}
            s_k_source = f"{rel} per_frame.s_k" + (" + bend" if any(b != (0.0, 0.0) for b in bend.values()) else "")
            cloud_epoch = doc.get("epoch_to")
            break
    # the certification warps that cloud per keyframe (depth × k + b about the camera, then
    # rigid): the depth that is LIVE is the cloud epoch's composed with every transform epoch
    # above it — the rigid part is already in camera_poses.txt, the depth part is composed here
    epochs_composed: List[int] = []
    if cloud_epoch is not None:
        s_k, bend, offset, epochs_composed = compose_transform_epochs(out, frames, int(cloud_epoch), s_k, bend)
        if epochs_composed:
            s_k_source += (f" × depth factor of transform epoch(s) {epochs_composed} "
                           f"(corrections/epoch_<N>.npz k_kf" + (", b_kf" if any(offset.values()) else "")
                           + f"; live epoch {epochs_composed[-1]})")
        else:
            s_k_source += f" (live epoch {cloud_epoch} is the cloud epoch)"
    rec = out / "omega_run" / "results_output"
    chunk = {}
    for f in frames:
        p = rec / f"frame_{f}.npz"
        if not p.exists():
            raise ChunkCheckError(f"{p} missing — Omega's per-keyframe record is the input")
        with np.load(p) as z:
            chunk[f] = int(z["chunk"]) if "chunk" in z.files else 0
    from precision.depth_sweep import keyframe_chainage
    chain = keyframe_chainage(Path(session_dir), frames)
    log(f"{LOG_TAG} {len(frames)} keyframes, {len(set(chunk.values()))} Omega chunk(s), "
        f"s_k from {s_k_source}, chainage {'measured' if chain is not None else 'NOT measured (no walk)'}")
    return (frames, c2w, K, s_k, s_k_source, chunk, chain, rec, out / "da3_run" / "results_output", bend,
            {"offset": offset, "epochs_composed": epochs_composed})


def compose_transform_epochs(out: Path, frames: Sequence[int], cloud_epoch: int,
                             s_k: Dict[int, float], bend: Dict[int, tuple]
                             ) -> Tuple[Dict[int, float], Dict[int, tuple], Dict[int, float], List[int]]:
    """``(s_k, bend, offset, epochs)``: the published cloud's per-keyframe depth model
    composed with every TRANSFORM epoch between the cloud epoch and the live one.

    A transform epoch moves keyframe f's points along their rays, z' = k·z + b about
    the keyframe's own camera (correction.apply.warp_full_cloud; k_kf, b_kf of
    corrections/epoch_<N>.npz), then rigidly — and the rigid part moves the camera with
    them, so camera_poses.txt already carries it. Epochs compose in lineage order:
    K ← k·K, B ← k·B + b, and the cloud's k(u, v) = s_k + c1·u' + c2·v' becomes
    K·k(u, v) + B. A new-cloud epoch on the way, or a keyframe an epoch does not
    name, is a structural impossibility of the check and is refused by name."""
    from correction.epoch import EPOCH_KIND_TRANSFORM, current_epoch, epoch_kind, epoch_lineage
    from correction.ledger import load_epoch_npz
    live = int(current_epoch(out))
    lineage = epoch_lineage(out, live)
    if int(cloud_epoch) not in lineage:
        raise ChunkCheckError(f"the live epoch {live} does not descend from the cloud epoch {cloud_epoch} "
                              f"(ancestry {lineage}) — the check cannot say what depth is live")
    after = lineage[lineage.index(int(cloud_epoch)) + 1:]
    K = {f: 1.0 for f in frames}
    B = {f: 0.0 for f in frames}
    for e in after:
        if epoch_kind(out, e) != EPOCH_KIND_TRANSFORM:
            raise ChunkCheckError(f"epoch {e} is a new cloud, not a transform of the cloud epoch "
                                  f"{cloud_epoch} — its depth is not Omega's × a factor")
        npz = load_epoch_npz(out, e)
        by_frame = {int(f): j for j, f in enumerate(npz["frames"])}
        missing = [f for f in frames if f not in by_frame]
        if missing:
            raise ChunkCheckError(f"corrections/epoch_{e}.npz names no transform for keyframe(s) "
                                  f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
        for f in frames:
            j = by_frame[f]
            k, b = float(npz["k_kf"][j]), float(npz["b_kf"][j])
            K[f] = k * K[f]
            B[f] = k * B[f] + b
    s_k2 = {f: float(s_k[f]) * K[f] for f in frames}
    bend2 = {f: (float(bend[f][0]) * K[f], float(bend[f][1]) * K[f]) for f in frames}
    return s_k2, bend2, B, [int(e) for e in after]


def scale_map(s_k: float, bend, H: int, W: int):
    """The per-pixel factor the published cloud applied to Omega's depth: s_k alone, or the
    f6_bend ratio model k(u, v) = s_k + c1·(u − W/2)/W + c2·(v − H/2)/H (precision.depth_on_f5.design)."""
    c1, c2 = (float(bend[0]), float(bend[1])) if bend is not None else (0.0, 0.0)
    if c1 == 0.0 and c2 == 0.0:
        return float(s_k)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return float(s_k) + c1 * (uu - W / 2) / W + c2 * (vv - H / 2) / H


def measure_rows(frames, c2w, K, s_k, chunk, chain, omega_dir: Path, da3_dir: Path, cfg,
                 log: Callable = print, bend: Optional[Dict[int, tuple]] = None,
                 offset: Optional[Dict[int, float]] = None) -> Tuple[List[Row], dict]:
    """Two passes over the depth maps: the dominant floor plane, then every height.
    ``offset`` (metres along the ray, per keyframe) is the affine part a transform epoch
    composed on the published depth — added only where Omega measured a depth."""
    t0 = time.time()
    floor_pool = []
    world_pts: Dict[int, np.ndarray] = {}
    # K is the camera's (undistorted native frame); Omega's record is the ORIGINAL frame on Omega's own
    # grid — the same frame only without a lens and on the camera's grid (pccr). Otherwise the record
    # is carried onto the undistorted native grid first (the bend's own grid, corrected_cloud.record_on_native)
    from precision.camera import load_camera_json, undistort_maps
    from precision import corrected_cloud as CC
    cm = load_camera_json(Path(omega_dir).parents[1] / "camera.json")
    lens = bool(np.any(cm.dist()))
    carry, maps = None, None
    for i, f in enumerate(frames):
        with np.load(omega_dir / f"frame_{f}.npz") as z:
            d0 = np.asarray(z["depth"], np.float64)
            if carry is None:
                carry = lens or tuple(d0.shape) != (cm.height, cm.width)
                if carry:
                    maps = undistort_maps(cm)[:2]
                    log(f"{LOG_TAG} Omega's record {d0.shape[1]}x{d0.shape[0]}" + (" + lens" if lens else "")
                        + f" carried onto the undistorted native grid {cm.width}x{cm.height} before unprojecting with K")
            if carry:
                d0 = CC.record_on_native(d0.astype(np.float32), cm, maps).astype(np.float64)
            d = d0 * scale_map(s_k[f], bend.get(f) if bend else None, d0.shape[0], d0.shape[1])
            off = float(offset.get(f, 0.0)) if offset else 0.0
            if off != 0.0:
                d = np.where(np.isfinite(d0) & (d0 > 0), d + off, d)
        X = unproject(d, K, c2w[i], cfg.pixel_stride)
        world_pts[f] = X
        if len(X):
            y = X[:, 1]
            q = np.percentile(y, cfg.low_pct)
            floor_pool.append(X[np.abs(y - q) <= cfg.band_m])
    if not floor_pool:
        raise ChunkCheckError("no Omega depth to measure")
    n, c, pinfo = dominant_plane(np.concatenate(floor_pool), cfg.band_m, cfg.seed, cfg.confidence,
                                 cfg.bootstrap)
    tilt = float(np.degrees(np.arccos(np.clip(n @ UP, -1, 1))))
    rows: List[Row] = []
    n_da3 = 0
    for i, f in enumerate(frames):
        X = world_pts[f]
        h = (X - c) @ n
        floor_h, n_floor, floor_res = band_height(h, cfg.low_pct, cfg.band_m, cfg.min_points)
        ceil_h, n_ceil, ceil_res = band_height(h, cfg.high_pct, cfg.band_m, cfg.min_points)
        cam_h = float((c2w[i][:3, 3] - c) @ n)
        cf_da3 = float("nan")
        p = da3_dir / f"frame_{f}.npz"
        if p.exists():
            with np.load(p) as z:
                dd = np.asarray(z["depth"], np.float64)
                Kd = np.asarray(z["intrinsics"], np.float64) if "intrinsics" in z.files else K
            hd = (unproject(dd, Kd, c2w[i], cfg.pixel_stride) - c) @ n
            fd, _, _ = band_height(hd, cfg.low_pct, cfg.band_m, cfg.min_points)
            if np.isfinite(fd):
                cf_da3 = cam_h - fd
                n_da3 += 1
        rows.append(Row(i, int(f), int(chunk[f]), float(chain[i]) if chain is not None else float("nan"),
                        floor_h, n_floor, ceil_h, n_ceil, cam_h,
                        cam_h - floor_h if np.isfinite(floor_h) else float("nan"), cf_da3, float(s_k[f]),
                        floor_res, ceil_res))
    plane = {"normal": [float(v) for v in n], "point": [float(v) for v in c], "tilt_deg": tilt,
             "n_floor_band_points": int(sum(len(p) for p in floor_pool)), **pinfo}
    log(f"{LOG_TAG} floor plane tilt {tilt:.2f}° from {plane['n_floor_band_points']:,} band points, holding "
        f"{pinfo['inlier_frac'] * 100:.1f} % of them (CI {pinfo['inlier_frac_ci'][0] * 100:.1f}–"
        f"{pinfo['inlier_frac_ci'][1] * 100:.1f} %, no acceptance bar); "
        f"{sum(np.isfinite(r.floor_h) for r in rows)} keyframes see the floor, "
        f"{sum(np.isfinite(r.ceil_h) for r in rows)} the ceiling, DA3 on {n_da3} ({time.time() - t0:.0f} s)")
    return rows, plane


# ── the verdicts ─────────────────────────────────────────────────────────

def _arr(rows: Sequence[Row], attr: str) -> np.ndarray:
    return np.array([getattr(r, attr) for r in rows], np.float64)


def _pool(rows: Sequence[Row], at: float, pool_m: float) -> List[Row]:
    return [r for r in rows if np.isfinite(r.chainage) and abs(r.chainage - at) <= pool_m]


def rule_departure(values: np.ndarray, ref, error: float, *, error_factor: float, confidence: float,
                   n_boot: int, seed: int) -> dict:
    """THE USER'S RULE for "does this band depart from the reference by more than it resolves"
    (docs/plan_determinismo.md point 140): the JUDGES are the finite ``values`` (one per keyframe);
    the departure must be (a) SIGNIFICANT at ``confidence`` — the whole bootstrap interval of the
    median departure on one side of zero, (b) testified by at least min_judge_closures(confidence)
    keyframes (5 at 0.95) and (c) at least ``error_factor`` × ``error``, the band's MEASURED
    resolution (its own spread). Returns ``departs`` (True / False / None = nothing to judge)
    with every margin for the report. A non-finite resolution enters as 0 and is declared.

    ``ref`` a scalar (a fixed level: the chunk's own median for the intra test, 0 for the
    'same amount' test): metric_lock.decide_change, paired against it, in both directions.
    ``ref`` an ARRAY (the other chunks' keyframes, the pool across a seam): the same three
    conditions on median(values) − median(ref) with BOTH samples resampled (seeded) — the
    reference's own uncertainty is part of the interval. A fixed pooled median is wrong here:
    when another chunk is off, the pool is bimodal and its median sits between the modes,
    so every sound chunk 'departed' from it (the synthetic depth case called chunks 0 and 2
    'depth' too); decide_change itself is paired and cannot resample a second sample."""
    from precision.refine import _vendor_path
    _vendor_path()
    from loop_utils.loop_judge import min_judge_closures
    from loop_utils.metric_lock import decide_change
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    two_sample = isinstance(ref, np.ndarray) or (not np.isscalar(ref) and ref is not None)
    out = {"n_judges": int(v.size), "error_m": float(error) if np.isfinite(error) else None,
           "error_factor": float(error_factor), "median_delta_m": None, "ci_m": [None, None],
           "required_m": None, "margin_m": None, "ci_margin_m": None, "judges_margin": None,
           "min_judges": int(min_judge_closures(float(confidence))), "failed": None, "departs": None,
           "resolution_unmeasured": not np.isfinite(error)}
    err = float(error) if np.isfinite(error) else 0.0
    fac = float(error_factor)
    if two_sample:
        r = np.asarray(ref, np.float64).ravel()
        r = r[np.isfinite(r)]
        out.update({"form": "two-sample (both resampled)", "n_reference": int(r.size),
                    "reference_m": float(np.median(r)) if r.size else None})
        if v.size == 0 or r.size == 0:
            return out
        rng = np.random.default_rng(int(seed))
        meds = np.empty(int(n_boot))
        for b in range(int(n_boot)):
            meds[b] = np.median(v[rng.integers(0, v.size, v.size)]) - np.median(r[rng.integers(0, r.size, r.size)])
        a = (1.0 - float(confidence)) / 2.0
        lo, hi = float(np.percentile(meds, 100 * a)), float(np.percentile(meds, 100 * (1 - a)))
        d = float(np.median(v) - np.median(r))
        need = out["min_judges"]
        enough = v.size >= need
        significant = lo > 0.0 or hi < 0.0
        beyond = abs(d) >= fac * err
        failed = [k for k, ok in (("judges", enough), ("significance", significant), ("error", beyond)) if not ok]
        out.update({"median_delta_m": d, "ci_m": [lo, hi], "required_m": fac * err, "margin_m": abs(d) - fac * err,
                    "ci_margin_m": (lo if d >= 0 else -hi), "judges_margin": int(v.size - need),
                    "failed": failed, "departs": bool(enough and significant and beyond)})
        return out
    out.update({"form": "paired against a fixed level (decide_change)",
                "reference_m": float(ref) if ref is not None and np.isfinite(ref) else None})
    if v.size == 0 or ref is None or not np.isfinite(ref):
        return out
    ref_arr = np.full(v.size, float(ref))
    up = decide_change(v, ref_arr, error=err, error_factor=fac, confidence=float(confidence),
                       n_boot=int(n_boot), seed=int(seed))           # d = value − ref: departs UPWARD
    down = decide_change(ref_arr, v, error=err, error_factor=fac, confidence=float(confidence),
                         n_boot=int(n_boot), seed=int(seed))         # d = ref − value: departs DOWNWARD
    side = up if up["median_delta"] >= 0.0 else down
    out.update({"median_delta_m": float(up["median_delta"]), "ci_m": [float(up["ci_low"]), float(up["ci_high"])],
                "required_m": float(side["required_delta"]), "margin_m": float(side["error_margin"]),
                "ci_margin_m": float(side["ci_margin"]), "judges_margin": int(side["judges_margin"]),
                "min_judges": int(side["min_judges"]), "failed": list(side["failed"]),
                "departs": bool(up["improves"] or down["improves"])})
    return out


def judge(rows: List[Row], cfg, pool_m: float, log: Callable = print, *, error_factor: float) -> dict:
    """Per-chunk verdicts, the seams and the per-keyframe intra-chunk residuals.

    The FLOOR and the CEILING decide, together: a depth error moves them in OPPOSITE
    directions (along the rays, away from the camera), a vertical pose error moves them
    in the SAME direction by the same amount, a real level change moves the floor alone.
    DA3 is a CONFIRMATION of a depth verdict, never evidence on its own: its per-chunk
    scale wanders ±12 % on pccr (chunks 4/5 with a perfect floor read 0.89× / 1.12×),
    which is larger than what is being hunted.

    Every "departs" is THE USER'S RULE (point 140, ``rule_departure``): the chunk's (or the
    pool's) keyframes are the judges — at least 5 at 0.95 — the departure is significant at
    ``confidence`` and at least ``error_factor`` × the band's measured resolution; the margins
    are in the report. A chunk with fewer judges than required is UNDECIDED, said so.
    """
    conf, seed, B = cfg.confidence, cfg.seed, cfg.bootstrap
    fac = float(error_factor)
    a_q = (1.0 - conf) / 2.0
    chunks = sorted({r.chunk for r in rows})
    by = {k: [r for r in rows if r.chunk == k] for k in chunks}
    logq = np.log(_arr(rows, "cf_omega") / _arr(rows, "cf_da3"))
    logq[~np.isfinite(logq)] = np.nan
    session = {"cf_omega_m": float(np.nanmedian(_arr(rows, "cf_omega"))),
               "cf_da3_m": float(np.nanmedian(_arr(rows, "cf_da3"))),
               "omega_over_da3": float(np.exp(np.nanmedian(logq))) if np.isfinite(np.nanmedian(logq)) else None,
               "floor_h_m": float(np.nanmedian(_arr(rows, "floor_h"))),
               "ceil_h_m": float(np.nanmedian(_arr(rows, "ceil_h")))}

    res_f = float(np.nanmedian(_arr(rows, "floor_res")))          # the instruments' resolution:
    res_c = float(np.nanmedian(_arr(rows, "ceil_res")))           # the bands' own spread

    def rule(values, ref, error, sd):
        return rule_departure(values, ref, error, error_factor=fac, confidence=conf, n_boot=B, seed=sd)

    def compare(A: Sequence[Row], Br: Sequence[Row], sd: int) -> dict:
        """B against A: the floor's departure (judged now, the floor's resolution is known), the
        ceiling's and the 'same amount' values (judged in pattern(), once the ceiling's resolution
        is known from the quiet seams)."""
        fA, fB = _arr(A, "floor_h"), _arr(Br, "floor_h")
        cA, cB = _arr(A, "ceil_h"), _arr(Br, "ceil_h")
        ref_f, ref_c = float(np.nanmedian(fA)), float(np.nanmedian(cA))
        floor = rule(fB, fA, res_f, sd)                              # two-sample: the reference resampled too
        dc = float(np.nanmedian(cB) - ref_c) if np.isfinite(ref_c) and np.isfinite(cB).any() else float("nan")
        same = (cB - ref_c) - (fB - ref_f)                           # per keyframe: ceiling jump − floor jump
        return {"floor_jump_m": floor["median_delta_m"], "floor_ci": floor["ci_m"], "floor_rule": floor,
                "ceiling_jump_m": dc, "ceiling_ci": [None, None], "ceiling_rule": None, "same_rule": None,
                "_ceiling_values": cB, "_ceiling_ref": cA, "_same_values": same, "_seed": sd}

    # seams: adjacent chunks in keyframe order, the keyframes within pool_m of the boundary
    order = []
    for r in rows:
        if not order or order[-1] != r.chunk:
            order.append(r.chunk)
    seams = []
    for a, b in zip(order[:-1], order[1:]):
        bnd = max(by[a], key=lambda r: r.i)
        if np.isfinite(bnd.chainage) and pool_m > 0:
            L, Rr = _pool(by[a], bnd.chainage, pool_m), _pool(by[b], bnd.chainage, pool_m)
        else:
            L, Rr = by[a], by[b]
        seams.append({"left_chunk": a, "right_chunk": b, "n_left": len(L), "n_right": len(Rr),
                      **compare(L, Rr, seed + 100 * (a + 1))})
    # THE CEILING'S OWN RESOLUTION: at the seams where the floor does not move, whatever the
    # ceiling band jumps is what it jumps for nothing (pccr: ±45–57 cm — ducts, beams and
    # fixtures enter and leave the top band as the camera turns). It can only testify to
    # jumps larger than that.
    # a QUIET floor is one MEASURED not to move — judged by enough keyframes and found within its
    # resolution / noise; a seam the rule could not judge (too few keyframes) says nothing about
    # the ceiling's noise and must not widen its resolution
    quiet = [abs(sm["ceiling_jump_m"]) for sm in seams
             if sm["floor_rule"]["departs"] is False and "judges" not in (sm["floor_rule"]["failed"] or [])
             and np.isfinite(sm["ceiling_jump_m"])]
    res_c_eff = max([res_c] + quiet) if np.isfinite(res_c) else (max(quiet) if quiet else float("nan"))
    session.update({"floor_resolution_m": res_f, "ceiling_resolution_m": res_c_eff,
                    "ceiling_resolution_source": ("the ceiling band's jumps at seams with a quiet floor"
                                                  if quiet and res_c_eff > res_c else "the ceiling band's spread"),
                    "rule": "metric_lock.decide_change per band: >= min_judge_closures(confidence) keyframe judges, "
                            "significant at confidence, |median| >= improvement_error_factor x the band's resolution",
                    "improvement_error_factor": fac})
    log(f"{LOG_TAG} resolution: floor {res_f * 100:.1f} cm (band spread), ceiling {res_c_eff * 100:.1f} cm "
        f"({session['ceiling_resolution_source']}); a departure must clear {fac:g} x that with >= 5 keyframes")

    def pattern(cmp: dict) -> Tuple[str, str]:
        """(verdict, why): the floor and the ceiling, together — every departure by the rule."""
        fr = cmp["floor_rule"]
        df = cmp["floor_jump_m"]
        if fr["departs"] is None:
            return "unmeasured", "no floor on one side"
        need = fr["min_judges"]
        if fr["n_judges"] < need:
            return "undecided", f"{fr['n_judges']} keyframe(s) judge the floor, {need} required"
        if not fr["departs"]:
            return "ok", (f"the floor does not depart ({df * 100:+.1f} cm; CI [{fr['ci_m'][0] * 100:+.1f}, "
                          f"{fr['ci_m'][1] * 100:+.1f}] cm, required {fr['required_m'] * 100:.1f} cm)")
        cr = rule(cmp["_ceiling_values"], cmp["_ceiling_ref"], res_c_eff, cmp["_seed"] + 1)
        cmp["ceiling_rule"], cmp["ceiling_ci"] = cr, cr["ci_m"]
        dc = cmp["ceiling_jump_m"]
        if cr["departs"] is None:
            return "undecided", f"floor {df * 100:+.1f} cm, no ceiling to ask"
        if cr["n_judges"] < need:
            return "undecided", f"floor {df * 100:+.1f} cm; {cr['n_judges']} keyframe(s) see the ceiling, {need} required"
        if not np.isfinite(res_c_eff) or res_c_eff >= abs(df):
            return "undecided", (f"floor {df * 100:+.1f} cm; the ceiling cannot resolve a jump of this size "
                                 f"(its own resolution here is {res_c_eff * 100:.1f} cm)")
        if not cr["departs"]:
            return "level", (f"floor {df * 100:+.1f} cm, ceiling {dc * 100:+.1f} cm (within {fac:g} x its "
                             f"resolution / noise): a real level change")
        if np.sign(dc) != np.sign(df):
            return "depth", f"floor {df * 100:+.1f} cm and ceiling {dc * 100:+.1f} cm move APART: depth along the rays"
        sr = rule(cmp["_same_values"], 0.0, max(res_f, res_c_eff), cmp["_seed"] + 2)
        cmp["same_rule"] = sr
        if sr["departs"] is False:
            return "pose", f"floor {df * 100:+.1f} cm and ceiling {dc * 100:+.1f} cm move TOGETHER: the cameras are off"
        return "undecided", f"floor {df * 100:+.1f} cm, ceiling {dc * 100:+.1f} cm, same direction but not the same amount"

    for sm in seams:
        sm["verdict"], sm["why"] = pattern(sm)
        for k_ in ("_ceiling_values", "_ceiling_ref", "_same_values", "_seed"):
            sm.pop(k_, None)
    # the session's own band of intra-chunk residuals (each keyframe against ITS chunk's median)
    resid = {k: _arr(by[k], "floor_h") - np.nanmedian(_arr(by[k], "floor_h")) for k in chunks}
    out_chunks, to_correct = [], []
    for k in chunks:
        rs, others = by[k], [r for r in rows if r.chunk != k]
        if others:
            cmp = compare(others, rs, seed + 3 + k)
            cmp["verdict"], cmp["why"] = pattern(cmp)
            for k_ in ("_ceiling_values", "_ceiling_ref", "_same_values", "_seed"):
                cmp.pop(k_, None)
        else:
            cmp = {"verdict": "unmeasured", "why": "a single chunk", "floor_jump_m": float("nan"),
                   "ceiling_jump_m": float("nan"), "floor_ci": [None, None], "ceiling_ci": [None, None],
                   "floor_rule": None, "ceiling_rule": None, "same_rule": None}
        fo = cmp["floor_jump_m"]
        # DA3, as confirmation only — the same rule, the error = the scatter of the others' own ratios
        lq_k = np.log(_arr(rs, "cf_omega") / _arr(rs, "cf_da3"))
        lq_o = np.log(_arr(others, "cf_omega") / _arr(others, "cf_da3")) if others else np.array([])
        lq_o = lq_o[np.isfinite(lq_o)]
        da3_rule = rule(lq_k, lq_o,
                        float(_MAD_TO_SIGMA * np.median(np.abs(lq_o - np.median(lq_o)))) if lq_o.size else float("nan"),
                        seed + 7 + k)
        da3_ratio = float(np.exp(da3_rule["median_delta_m"])) if da3_rule["median_delta_m"] is not None else None
        da3_dep = da3_rule["departs"]
        # intra-chunk: the chunk's floor trend along its own walk; the keyframes around the trend's
        # peak are the judges of a departure from the chunk's own level (the rule, point 140)
        fh, ch = _arr(rs, "floor_h"), _arr(rs, "chainage")
        trend = np.full(len(rs), np.nan)
        for j, r in enumerate(rs):
            w = (np.abs(ch - r.chainage) <= pool_m) if (np.isfinite(r.chainage) and pool_m > 0) else np.ones(len(rs), bool)
            v = fh[w]; v = v[np.isfinite(v)]
            trend[j] = np.median(v) if v.size else np.nan
        exc = trend - np.nanmedian(fh)
        intra, exc_kf, intra_rule = False, None, None
        if np.isfinite(exc).any():
            jj = int(np.nanargmax(np.abs(exc)))
            exc_kf = {"keyframe": rs[jj].i, "excursion_m": float(exc[jj])}
            wj = (np.abs(ch - rs[jj].chainage) <= pool_m) if (np.isfinite(rs[jj].chainage) and pool_m > 0) \
                else np.ones(len(rs), bool)
            intra_rule = rule(fh[wj], float(np.nanmedian(fh)), res_f, seed + 11 + k)
            intra = bool(intra_rule["departs"])
        verdict, why = cmp["verdict"], cmp["why"]
        if verdict in ("ok", "undecided") and intra:
            why = (f"the floor drifts INSIDE the chunk: trend excursion {np.nanmin(exc) * 100:+.1f} … {np.nanmax(exc) * 100:+.1f} cm "
                   f"(peak at kf {exc_kf['keyframe']}, {intra_rule['n_judges']} keyframes around it depart from the chunk's "
                   f"level by {intra_rule['median_delta_m'] * 100:+.1f} cm, required {intra_rule['required_m'] * 100:.1f})"
                   + ("; its seams are consistent, so the error builds up and returns within the chunk" if verdict == "ok"
                      else f"; as a whole: {why}"))
            verdict = "intra"
        if verdict == "depth":
            r_floor = float(np.nanmedian(_arr(rs, "cf_omega")) / np.nanmedian(_arr(others, "cf_omega")))
            why += f"; camera-to-floor {r_floor:.3f}× the others" + (
                f", DA3 confirms ({da3_ratio:.3f}× the session ratio)" if da3_dep and da3_ratio and (da3_ratio > 1) == (r_floor > 1)
                else ", DA3 does not confirm")
            to_correct.append({"chunk": k, "kind": "scale_about_camera", "factor": 1.0 / r_floor, "keyframes": [r.i for r in rs]})
        elif verdict == "pose":
            to_correct.append({"chunk": k, "kind": "vertical_pose", "delta_m": -fo, "keyframes": [r.i for r in rs]})
        elif verdict == "intra":
            to_correct.append({"chunk": k, "kind": "vertical_alignment_per_keyframe", "keyframes": [r.i for r in rs],
                               "excursion_m": [float(np.nanmin(exc)), float(np.nanmax(exc))]})
        out_chunks.append({"chunk": k, "keyframes": [rs[0].i, rs[-1].i], "n": len(rs),
                           "floor_h_m": float(np.nanmedian(fh)), "floor_vs_others_m": fo, "floor_ci": cmp["floor_ci"],
                           "ceil_vs_others_m": cmp["ceiling_jump_m"], "ceiling_ci": cmp["ceiling_ci"],
                           "cam_h_m": float(np.nanmedian(_arr(rs, "cam_h"))),
                           "cf_omega_m": float(np.nanmedian(_arr(rs, "cf_omega"))),
                           "cf_da3_m": float(np.nanmedian(_arr(rs, "cf_da3"))),
                           "da3_ratio_vs_session": da3_ratio, "da3_departs": da3_dep, "da3_rule": da3_rule,
                           "floor_rule": cmp["floor_rule"], "ceiling_rule": cmp["ceiling_rule"],
                           "same_amount_rule": cmp["same_rule"],
                           "trend_excursion_m": [float(np.nanmin(exc)) if np.isfinite(exc).any() else None,
                                                 float(np.nanmax(exc)) if np.isfinite(exc).any() else None],
                           "intra": intra, "intra_peak": exc_kf, "intra_rule": intra_rule,
                           "seams": [s["verdict"] for s in seams if k in (s["left_chunk"], s["right_chunk"])],
                           "verdict": verdict, "why": why})
        log(f"{LOG_TAG} chunk {k} (kf {rs[0].i}-{rs[-1].i}): floor {np.nanmedian(fh) * 100:+.1f} cm, ceiling "
            f"{np.nanmedian(_arr(rs, 'ceil_h')) * 100:+.1f} cm, cam−floor {np.nanmedian(_arr(rs, 'cf_omega')) * 100:.0f} cm "
            f"(DA3 {np.nanmedian(_arr(rs, 'cf_da3')) * 100:.0f}), intra excursion "
            f"{np.nanmin(exc) * 100:+.1f}…{np.nanmax(exc) * 100:+.1f} cm → {verdict.upper()}: {why}")
    # per keyframe
    finq = logq[np.isfinite(logq)]
    q_band = (np.quantile(finq, a_q), np.quantile(finq, 1 - a_q)) if finq.size else (np.nan, np.nan)
    keyframes = []
    for k in chunks:
        allres = np.concatenate([resid[o] for o in chunks if o != k]) if len(chunks) > 1 else resid[k]
        allres = allres[np.isfinite(allres)]
        rb = (np.quantile(allres, a_q), np.quantile(allres, 1 - a_q)) if allres.size else (np.nan, np.nan)
        for r, res in zip(by[k], resid[k]):
            lq = logq[r.i]
            keyframes.append({"i": r.i, "frame": r.frame, "chunk": r.chunk, "chainage_m": r.chainage,
                              "floor_h_m": r.floor_h, "ceil_h_m": r.ceil_h, "cam_h_m": r.cam_h,
                              "cf_omega_m": r.cf_omega, "cf_da3_m": r.cf_da3, "s_k": r.s_k,
                              "resid_vs_chunk_m": float(res) if np.isfinite(res) else None,
                              "intra_outlier": bool(np.isfinite(res) and (res < rb[0] or res > rb[1])),
                              "da3_outlier": bool(np.isfinite(lq) and (lq < q_band[0] or lq > q_band[1]))})
    keyframes.sort(key=lambda d: d["i"])
    return {"session": session, "chunks": out_chunks, "seams": seams, "keyframes": keyframes,
            "to_correct": to_correct}


# ── the run ──────────────────────────────────────────────────────────────

def run_check(session_dir: Path, pcfg, log: Callable = print, chainage: Optional[np.ndarray] = None) -> dict:
    """Measure, judge, write ``output/precision/chunk_check.json``; return the report."""
    t0 = time.time()
    session_dir = Path(session_dir)
    cfg = pcfg.chunk_check
    from config import cfg as raw_cfg
    from reconstruction.loops.config import improvement_error_factor
    fac = float(improvement_error_factor(raw_cfg))                  # the user's 2 (point 1), ONE place
    (frames, c2w, K, s_k, s_k_source, chunk, chain, omega_dir, da3_dir, bend,
     composed) = load_inputs(session_dir, log)
    if chainage is not None:
        chain = np.asarray(chainage, np.float64)
    pool_m = float(pcfg.gauge.knot_walk_m) / 2.0
    try:
        rows, plane = measure_rows(frames, c2w, K, s_k, chunk, chain, omega_dir, da3_dir, cfg, log,
                                   bend=bend, offset=composed["offset"])
        verdicts = judge(rows, cfg, pool_m, log, error_factor=fac)
    except NoFloorPlane as e:
        # DECLARED, not fatal (USER 2026-10-05): a report step that cannot measure says so and the
        # chain goes on — the cloud is published, the certification has its own floor machinery
        log(f"{LOG_TAG} ⚠ {e} — the per-chunk floor/ceiling check is UNDECIDED for this session; "
            f"the cloud stands, the certification measures its own floor")
        plane = None
        verdicts = {"session": {"verdict": "undecided", "reason": str(e),
                                "floor_plane": {"found": False, "low_band_points": e.n_points}},
                    "chunks": [], "seams": [], "keyframes": [], "to_correct": []}
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    from precision.corrected_cloud import write_timing
    out = session_dir / "output"
    # the epoch numbers say where in the session's history this was measured; the
    # reconstruction id says WHICH reconstruction (point 63: the former depends on the history)
    rep = {"version": 1, "provenance": PROVENANCE, **_epochs(out),
           RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(out),
           "params": {"low_pct": cfg.low_pct, "high_pct": cfg.high_pct, "band_m": cfg.band_m,
                      "min_points": cfg.min_points, "pixel_stride": cfg.pixel_stride,
                      "confidence": cfg.confidence, "bootstrap": cfg.bootstrap, "pool_walk_m": pool_m,
                      "seed": int(cfg.seed), "improvement_error_factor": fac,
                      "floor_plane": "seeded numpy RANSAC (point 50), no acceptance bar (point 140)",
                      "s_k_source": s_k_source, "depth_epochs_composed": composed["epochs_composed"]},
           "plane": plane, **verdicts}
    seconds_check = round(time.time() - t0, 1)
    pdir = out / "precision"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / CHECK_NAME).write_text(json.dumps(rep, indent=1, default=float))
    summary = (", ".join(f"chunk {c['chunk']} {c['verdict']}" for c in rep["chunks"])
               or f"session {rep['session'].get('verdict', '?')}")
    log(f"{LOG_TAG} {summary}; {len(rep['to_correct'])} correction(s) indicated → {pdir / CHECK_NAME} "
        f"({seconds_check} s)")
    # the floor metric of the published cloud (precision/cloud_metrics.py; edges need the projection)
    from precision.cloud_metrics import run_cloud_metrics
    try:
        rep["cloud_metrics"] = run_cloud_metrics(session_dir, pcfg, stage="f6_check", log=log, edges=False)
    except Exception as e:  # noqa: BLE001 — a metric that cannot be measured is declared, never fatal
        log(f"{LOG_TAG} ⚠ cloud metrics not measured: {e}")
        rep["cloud_metrics"] = {"measured": False, "reason": str(e)}
    (pdir / CHECK_NAME).write_text(json.dumps(rep, indent=1, default=float))
    # the wall clock lives next to the report, never in it (point 36 / 56)
    write_timing(pdir / CHECK_NAME, {"seconds_check": seconds_check,
                                     "seconds_total": round(time.time() - t0, 1)})
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    from precision.config import load_precision_config
    run_check(Path(args.session), load_precision_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
