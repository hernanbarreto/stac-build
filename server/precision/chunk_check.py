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


def dominant_plane(points: np.ndarray, band_m: float, seed: int,
                   min_inlier_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """The session's floor plane (unit normal towards +Y, a point on it) — RANSAC over the
    pooled floor bands of every keyframe (reconstruction.geometry.primitives)."""
    from reconstruction.geometry.primitives import fit_plane_ransac
    P = np.asarray(points, np.float64)
    if len(P) > 400_000:
        P = P[np.random.default_rng(seed).choice(len(P), 400_000, replace=False)]
    pf = fit_plane_ransac(P, dist_thresh=band_m / 4, iters=400, min_inlier_frac=min_inlier_frac,
                          measure_curvature=False)
    if pf is None:
        raise ChunkCheckError("no dominant floor plane in the pooled floor bands")
    inl = P[pf.inliers]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][2]      # least-squares normal of the inliers
    n = n / np.linalg.norm(n)
    if n[1] < 0:
        n = -n
    return n, c


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


def ci_median_diff(a: np.ndarray, b: np.ndarray, confidence: float, seed: int,
                   n_boot: int) -> Tuple[float, float, float]:
    """(median(a) − median(b), lo, hi): the difference of two independent samples with
    its bootstrap interval."""
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    b = np.asarray(b, np.float64); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, a.size, (n_boot, a.size))], axis=1)
    mb = np.median(b[rng.integers(0, b.size, (n_boot, b.size))], axis=1)
    d = ma - mb
    al = (1.0 - confidence) / 2.0
    return float(np.median(a) - np.median(b)), float(np.quantile(d, al)), float(np.quantile(d, 1.0 - al))


def departs(lo: float, hi: float) -> Optional[bool]:
    """True when the interval lies entirely on one side of zero, False when it holds zero,
    None when there was nothing to measure."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return bool(lo > 0.0 or hi < 0.0)


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
    g = cam.get("omega_grid") or {}
    if g and not (float(g.get("scale_x", 1.0)) == 1.0 and int(g.get("crop_x", 0)) == 0
                  and int(g.get("pad_left", 0)) == 0):
        raise ChunkCheckError("Omega's grid is not the native grid — the check needs the "
                              "F0 grid mapping before unprojecting Omega's depth")
    s_k = {f: 1.0 for f in frames}
    s_k_source = "absent (1.0 — neither the corrected cloud nor F6 measured the per-keyframe scale)"
    for rel in ("corrected_cloud.json", "depth_native/report.json"):
        rep = out / rel
        if not rep.exists():
            continue
        pf = json.loads(rep.read_text()).get("per_frame") or {}
        if all(str(f) in pf and "s_k" in pf[str(f)] for f in frames):
            s_k = {f: float(pf[str(f)]["s_k"]) for f in frames}
            s_k_source = f"{rel} per_frame.s_k"
            break
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
    return frames, c2w, K, s_k, s_k_source, chunk, chain, rec, out / "da3_run" / "results_output"


def measure_rows(frames, c2w, K, s_k, chunk, chain, omega_dir: Path, da3_dir: Path, cfg,
                 log: Callable = print) -> Tuple[List[Row], dict]:
    """Two passes over the depth maps: the dominant floor plane, then every height."""
    t0 = time.time()
    floor_pool = []
    world_pts: Dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        with np.load(omega_dir / f"frame_{f}.npz") as z:
            d = np.asarray(z["depth"], np.float64) * s_k[f]
        X = unproject(d, K, c2w[i], cfg.pixel_stride)
        world_pts[f] = X
        if len(X):
            y = X[:, 1]
            q = np.percentile(y, cfg.low_pct)
            floor_pool.append(X[np.abs(y - q) <= cfg.band_m])
    if not floor_pool:
        raise ChunkCheckError("no Omega depth to measure")
    n, c = dominant_plane(np.concatenate(floor_pool), cfg.band_m, cfg.seed, cfg.plane_min_inlier_frac)
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
             "n_floor_band_points": int(sum(len(p) for p in floor_pool))}
    log(f"{LOG_TAG} dominant floor plane tilt {tilt:.2f}° from {plane['n_floor_band_points']:,} band points; "
        f"{sum(np.isfinite(r.floor_h) for r in rows)} keyframes see the floor, "
        f"{sum(np.isfinite(r.ceil_h) for r in rows)} the ceiling, DA3 on {n_da3} ({time.time() - t0:.0f} s)")
    return rows, plane


# ── the verdicts ─────────────────────────────────────────────────────────

def _arr(rows: Sequence[Row], attr: str) -> np.ndarray:
    return np.array([getattr(r, attr) for r in rows], np.float64)


def _pool(rows: Sequence[Row], at: float, pool_m: float) -> List[Row]:
    return [r for r in rows if np.isfinite(r.chainage) and abs(r.chainage - at) <= pool_m]


def judge(rows: List[Row], cfg, pool_m: float, log: Callable = print) -> dict:
    """Per-chunk verdicts, the seams and the per-keyframe intra-chunk residuals.

    The FLOOR and the CEILING decide, together: a depth error moves them in OPPOSITE
    directions (along the rays, away from the camera), a vertical pose error moves them
    in the SAME direction by the same amount, a real level change moves the floor alone.
    DA3 is a CONFIRMATION of a depth verdict, never evidence on its own: its per-chunk
    scale wanders ±12 % on pccr (chunks 4/5 with a perfect floor read 0.89× / 1.12×),
    which is larger than what is being hunted.
    """
    conf, seed, B = cfg.confidence, cfg.seed, cfg.bootstrap
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

    def beyond(lo, hi, val, res):
        """Departs by the sample's own noise AND by more than the instrument resolves."""
        d = departs(lo, hi)
        if d is None:
            return None
        return bool(d and np.isfinite(val) and (not np.isfinite(res) or abs(val) > res))

    def compare(A: Sequence[Row], Br: Sequence[Row], sd: int) -> dict:
        """B against A: floor jump, ceiling jump, whether they are the same amount."""
        df, flo, fhi = ci_median_diff(_arr(Br, "floor_h"), _arr(A, "floor_h"), conf, sd, B)
        dc, clo, chi = ci_median_diff(_arr(Br, "ceil_h"), _arr(A, "ceil_h"), conf, sd + 1, B)
        same_lo = same_hi = float("nan")
        if np.isfinite(df) and np.isfinite(dc):
            _, same_lo, same_hi = ci_median_diff(_arr(Br, "ceil_h") - np.nanmedian(_arr(A, "ceil_h")),
                                                 _arr(Br, "floor_h") - np.nanmedian(_arr(A, "floor_h")), conf, sd + 2, B)
        return {"floor_jump_m": df, "floor_ci": [flo, fhi], "ceiling_jump_m": dc, "ceiling_ci": [clo, chi],
                "same_ci": [same_lo, same_hi]}

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
    quiet = [abs(sm["ceiling_jump_m"]) for sm in seams
             if beyond(sm["floor_ci"][0], sm["floor_ci"][1], sm["floor_jump_m"], res_f) is False
             and np.isfinite(sm["ceiling_jump_m"])]
    res_c_eff = max([res_c] + quiet) if np.isfinite(res_c) else (max(quiet) if quiet else float("nan"))
    session.update({"floor_resolution_m": res_f, "ceiling_resolution_m": res_c_eff,
                    "ceiling_resolution_source": ("the ceiling band's jumps at seams with a quiet floor"
                                                  if quiet and res_c_eff > res_c else "the ceiling band's spread")})
    log(f"{LOG_TAG} resolution: floor {res_f * 100:.1f} cm (band spread), ceiling {res_c_eff * 100:.1f} cm "
        f"({session['ceiling_resolution_source']})")

    def pattern(cmp: dict) -> Tuple[str, str]:
        """(verdict, why): the floor and the ceiling, together."""
        df, (flo, fhi) = cmp["floor_jump_m"], cmp["floor_ci"]
        dc, (clo, chi) = cmp["ceiling_jump_m"], cmp["ceiling_ci"]
        f_dep = beyond(flo, fhi, df, res_f)
        if f_dep is None:
            return "unmeasured", "no floor on one side"
        if not f_dep:
            return "ok", "the floor does not move"
        if not np.isfinite(dc):
            return "undecided", f"floor {df * 100:+.1f} cm, no ceiling to ask"
        if not np.isfinite(res_c_eff) or res_c_eff >= abs(df):
            return "undecided", (f"floor {df * 100:+.1f} cm; the ceiling cannot resolve a jump of this size "
                                 f"(its own resolution here is {res_c_eff * 100:.1f} cm)")
        c_dep = beyond(clo, chi, dc, res_c_eff)
        if not c_dep:
            return "level", f"floor {df * 100:+.1f} cm, ceiling {dc * 100:+.1f} cm (within noise): a real level change"
        if np.sign(dc) != np.sign(df):
            return "depth", f"floor {df * 100:+.1f} cm and ceiling {dc * 100:+.1f} cm move APART: depth along the rays"
        if departs(cmp["same_ci"][0], cmp["same_ci"][1]) is False:
            return "pose", f"floor {df * 100:+.1f} cm and ceiling {dc * 100:+.1f} cm move TOGETHER: the cameras are off"
        return "undecided", f"floor {df * 100:+.1f} cm, ceiling {dc * 100:+.1f} cm, same direction but not the same amount"

    for sm in seams:
        sm["verdict"], sm["why"] = pattern(sm)
    # the session's own band of intra-chunk residuals (each keyframe against ITS chunk's median)
    resid = {k: _arr(by[k], "floor_h") - np.nanmedian(_arr(by[k], "floor_h")) for k in chunks}
    out_chunks, to_correct = [], []
    for k in chunks:
        rs, others = by[k], [r for r in rows if r.chunk != k]
        if others:
            cmp = compare(others, rs, seed + 3 + k)
            cmp["verdict"], cmp["why"] = pattern(cmp)
        else:
            cmp = {"verdict": "unmeasured", "why": "a single chunk", "floor_jump_m": float("nan"),
                   "ceiling_jump_m": float("nan"), "floor_ci": [None, None], "ceiling_ci": [None, None]}
        fo = cmp["floor_jump_m"]
        # DA3, as confirmation only
        lq_k = np.log(_arr(rs, "cf_omega") / _arr(rs, "cf_da3"))
        lq_o = np.log(_arr(others, "cf_omega") / _arr(others, "cf_da3")) if others else np.array([])
        dq, qlo, qhi = ci_median_diff(lq_k, lq_o, conf, seed + 7 + k, B)
        da3_ratio = float(np.exp(dq)) if np.isfinite(dq) else None
        da3_dep = departs(qlo, qhi)
        # intra-chunk: the chunk's floor trend along its own walk against the session's band
        fh, ch = _arr(rs, "floor_h"), _arr(rs, "chainage")
        trend = np.full(len(rs), np.nan)
        for j, r in enumerate(rs):
            w = (np.abs(ch - r.chainage) <= pool_m) if (np.isfinite(r.chainage) and pool_m > 0) else np.ones(len(rs), bool)
            v = fh[w]; v = v[np.isfinite(v)]
            trend[j] = np.median(v) if v.size else np.nan
        exc = trend - np.nanmedian(fh)
        oth = np.concatenate([resid[o] for o in chunks if o != k]) if others else np.array([])
        oth = oth[np.isfinite(oth)]
        band = (float(np.quantile(oth, a_q)), float(np.quantile(oth, 1 - a_q))) if oth.size else (float("nan"), float("nan"))
        intra = bool(np.isfinite(exc).any() and np.isfinite(band[0]) and
                     (np.nanmin(exc) < min(band[0], -res_f) or np.nanmax(exc) > max(band[1], res_f)))
        exc_kf = None
        if intra:
            jj = int(np.nanargmax(np.abs(exc)))
            exc_kf = {"keyframe": rs[jj].i, "excursion_m": float(exc[jj])}
        verdict, why = cmp["verdict"], cmp["why"]
        if verdict in ("ok", "undecided") and intra:
            why = (f"the floor drifts INSIDE the chunk: trend excursion {np.nanmin(exc) * 100:+.1f} … {np.nanmax(exc) * 100:+.1f} cm "
                   f"(peak at kf {exc_kf['keyframe']}) against the other chunks' band [{band[0] * 100:+.1f}, {band[1] * 100:+.1f}] cm"
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
                           "da3_ratio_vs_session": da3_ratio, "da3_departs": da3_dep,
                           "trend_excursion_m": [float(np.nanmin(exc)) if np.isfinite(exc).any() else None,
                                                 float(np.nanmax(exc)) if np.isfinite(exc).any() else None],
                           "intra_band_m": [band[0] if np.isfinite(band[0]) else None, band[1] if np.isfinite(band[1]) else None],
                           "intra": intra, "intra_peak": exc_kf,
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
    frames, c2w, K, s_k, s_k_source, chunk, chain, omega_dir, da3_dir = load_inputs(session_dir, log)
    if chainage is not None:
        chain = np.asarray(chainage, np.float64)
    rows, plane = measure_rows(frames, c2w, K, s_k, chunk, chain, omega_dir, da3_dir, cfg, log)
    pool_m = float(pcfg.gauge.knot_walk_m) / 2.0
    verdicts = judge(rows, cfg, pool_m, log)
    rep = {"version": 1, "provenance": PROVENANCE, **_epochs(session_dir / "output"),
           "params": {"low_pct": cfg.low_pct, "high_pct": cfg.high_pct, "band_m": cfg.band_m,
                      "min_points": cfg.min_points, "pixel_stride": cfg.pixel_stride,
                      "confidence": cfg.confidence, "bootstrap": cfg.bootstrap, "pool_walk_m": pool_m,
                      "s_k_source": s_k_source},
           "plane": plane, **verdicts, "seconds": round(time.time() - t0, 1)}
    pdir = session_dir / "output" / "precision"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / CHECK_NAME).write_text(json.dumps(rep, indent=1, default=float))
    summary = ", ".join(f"chunk {c['chunk']} {c['verdict']}" for c in rep["chunks"])
    log(f"{LOG_TAG} {summary}; {len(rep['to_correct'])} correction(s) indicated → {pdir / CHECK_NAME} "
        f"({rep['seconds']} s)")
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
