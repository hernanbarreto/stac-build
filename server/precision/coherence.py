"""THE COHERENCE MEASURE — the one ruler every stage of the chain is judged with (USER 2026-11-20: "si empeora ese
paso o esa decisión se vuelve al estado mejor y se sigue con el siguiente paso del proceso. Debemos llegar a la
publicación de la nube habiendo pasado por todas las etapas y habiéndonos quedado con el mejor estado del sistema" /
"hay que tomar decisiones sobre los resultados que se van obteniendo").

A reconstruction is, per keyframe, a camera pose, a depth map with its scale, and one camera model. It is COHERENT
when two keyframes that see the same surface put it in the same place (docs/pipeline_audit_2026-11-20.md, Part I).
This module measures exactly that, and nothing of it depends on the stage that produced the state:

  one keyframe's pixels are CARRIED into another keyframe with the state's geometry, and compared with what that
  keyframe itself has there —
    ALONG THE RAY   the carried depth against the other keyframe's own depth: the median relative difference of the
                    pixels that land on the same surface (the same surface = within ``same_surface_sigmas`` robust
                    sigmas of the pair's own median: no fixed band), in cm at the surface;
    ACROSS THE RAY  the carried IMAGE against the other keyframe's image: the shift that best matches them
                    (normalised cross-correlation over a window of shifts), in cm at the surface. Two photographs:
                    independent of the depth network.

Pairs: NEIGHBOURS (the offsets F6's own neighbour disagreement uses) and DIFFERENT PASSES (for each keyframe, the
keyframes the first pass's map says share the most with it among those the walk does not tie). A pair is measured in
both directions; each direction is a judge, clustered by its pair.

A ZONE is a stretch of the walk (the keyframes a cube owns). ``judge_zones`` compares two states zone by zone with
the USER's rule (``decide_change``: the median of the paired differences, a 95 % bootstrap interval that excludes
zero, at least five judges, at least 1.1 x the measured error) — the error being the measure's own repeatability on
those pairs (each pair measured on two halves of its pixels).

Held by tests on a scene whose answer is known (tests/test_coherence.py). Everything is float64 on ONE declared
device; nothing is drawn at random.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from precision.epoch0_cloud import SKY_CONF   # the fork's sentinel: a pixel the sky mask zeroed (no threshold)

KIND_NEIGHBOUR, KIND_PASS = 0, 1
KIND_NAMES = {KIND_NEIGHBOUR: "neighbours", KIND_PASS: "different passes"}
METRICS = ("along_cm", "across_cm")
ROBUST = 1.0 / 0.6744897501960817   # MAD -> sigma of a normal distribution: 1 / Phi^-1(3/4), a mathematical constant
SECTION = "reconstruction.precision.coherence"
NCC_BLOCK = 16384               # pixels a block of the correlation's sums (memory; the result does not depend on it
#                                 beyond the order of its additions, which is fixed)
BLUR_REACH = 3                  # the smoothing's sums reach this many sigmas (a numerical truncation)


class CoherenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Params:
    """What the measure and the rule work with: the section's own keys (precision.config.Params) and the
    values other sections own. No number is this module's."""
    device: str                 # cpu | cuda — every array of the measure lives there
    pixel_stride: int           # every n-th pixel of a keyframe (rows and columns) is carried
    search_px: int              # half-width of the window of shifts searched across the ray, in pixels
    blur_sigma_px: float        # the grey images are smoothed by this before they are compared
    min_pixels: int             # fewest pixels on a common surface for a pair to be measured at all
    same_surface_sigmas: float  # same surface = depths within this many robust sigmas of the pair's own differences
    neighbour_offsets: Tuple[int, ...]   # the vote's own, as distances (reconstruction.precision.bend.neighbors)
    pass_min_gap: int           # keyframes apart from which the walk does not tie two keyframes (tracks.window_frames)
    range_m: float              # reconstruction.view_range_m — his range: what lies beyond it judges nothing
    conf_floor_norm: float      # reconstruction.simple.conf_min_norm — the pipeline's one confidence floor
    confidence: float           # the rule's confidence (coherence.heldout_confidence)
    error_factor: float         # his factor (correction_graph.graph.improvement_error_factor)
    min_judges: int             # fewest judges for the rule's interval to exclude zero at that confidence (the fork's
    #                             min_judge_closures: 5 at 95 %) — also how many partners of other passes a keyframe gets
    bootstrap: int              # resamples of the rule's interval
    seed: int                   # their fixed seed


def min_judges_of(confidence: float) -> int:
    from precision.judge_torch import _vendor
    _vendor()
    from loop_utils.loop_judge import min_judge_closures
    return int(min_judge_closures(float(confidence)))


def params(pcfg, raw_cfg: dict, device: Optional[str] = None) -> Params:
    """The measure's parameters from the job's configuration (``pcfg``: precision.config.PrecisionConfig).
    ``device`` overrides the section's (the tests run on the CPU); cuda with no card FAILS naming the key."""
    from reconstruction.loops.config import improvement_error_factor
    co = pcfg.coherence
    dev = str(device or co.device)
    if dev == "cuda" and not torch.cuda.is_available():
        raise CoherenceError(f"'{SECTION}.device' is cuda and this process sees no card")
    return Params(
        device=dev, pixel_stride=int(co.pixel_stride), search_px=int(co.search_px),
        blur_sigma_px=float(co.blur_sigma_px), min_pixels=int(co.min_pixels),
        same_surface_sigmas=float(co.same_surface_sigmas),
        neighbour_offsets=tuple(sorted(set(abs(int(o)) for o in pcfg.bend.neighbors))),
        pass_min_gap=int(pcfg.tracks.window_frames), range_m=float(pcfg.view_range_m),
        conf_floor_norm=float(raw_cfg["reconstruction"]["simple"]["conf_min_norm"]),
        confidence=float(co.heldout_confidence), error_factor=float(improvement_error_factor(raw_cfg)),
        min_judges=min_judges_of(co.heldout_confidence), bootstrap=int(co.bootstrap), seed=int(co.seed))


# ── the camera ────────────────────────────────────────────────────────────────────────────────────────────────────────
def project(x: torch.Tensor, y: torch.Tensor, K: Sequence[float], dist: Sequence[float]):
    """Normalised camera coordinates -> pixels, with the lens (OpenCV's k1, k2, p1, p2)."""
    fx, fy, cx, cy = (float(v) for v in K)
    k1, k2, p1, p2 = (float(v) for v in dist)
    r2 = x * x + y * y
    rad = 1.0 + r2 * (k1 + k2 * r2)
    xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return fx * xd + cx, fy * yd + cy


def unproject(u: torch.Tensor, v: torch.Tensor, K: Sequence[float], dist: Sequence[float], *, iterations: int = 100,
              tol_px: float = 1e-9):
    """Pixels -> normalised camera coordinates (the lens removed by the fixed-point iteration every implementation
    of this model uses); the result is VERIFIED: projected back it must give the pixel within ``tol_px``."""
    fx, fy, cx, cy = (float(q) for q in K)
    k1, k2, p1, p2 = (float(q) for q in dist)
    xd, yd = (u - cx) / fx, (v - cy) / fy
    x, y = xd.clone(), yd.clone()
    if any(c != 0.0 for c in (k1, k2, p1, p2)):
        for _ in range(int(iterations)):
            r2 = x * x + y * y
            rad = 1.0 + r2 * (k1 + k2 * r2)
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x, y = (xd - dx) / rad, (yd - dy) / rad
    ub, vb = project(x, y, K, dist)
    err = float(torch.maximum((ub - u).abs().max(), (vb - v).abs().max()))
    if not err <= tol_px:
        raise CoherenceError(f"the lens could not be removed on this grid: {err:.3e} px left after {iterations} "
                             f"iterations (camera {tuple(K)}, lens {tuple(dist)})")
    return x, y


# ── what is measured: the maps (state independent) and a state ────────────────────────────────────────────────────────
@dataclass
class Maps:
    """The keyframes' own data, the same for every state: depth (in the records' units), the pixels that judge, the
    smoothed grey image — all on the device, on ONE grid."""
    depth: torch.Tensor         # (N, H, W) float32
    valid: torch.Tensor         # (N, H, W) bool
    grey: torch.Tensor          # (N, H, W) float32
    frames: Tuple[int, ...]

    @property
    def hw(self) -> Tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[2])


@dataclass
class State:
    """A reconstruction as the measure reads it."""
    label: str
    c2w: torch.Tensor           # (N, 4, 4) float64 — camera to world
    K: Tuple[float, float, float, float]   # fx, fy, cx, cy on the maps' grid
    dist: Tuple[float, float, float, float]  # k1, k2, p1, p2
    factor: torch.Tensor        # (N,) float64 — what multiplies each keyframe's depth map (one number repeated when
    #                             the scale is one for the session)


def blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """A Gaussian smoothing as explicit weighted sums of shifted copies (edges replicated): the same bits on any
    device, no convolution library."""
    if sigma <= 0:
        return img
    r = max(1, int(math.ceil(BLUR_REACH * sigma)))
    w = torch.tensor([math.exp(-0.5 * (k / sigma) ** 2) for k in range(-r, r + 1)], dtype=img.dtype, device=img.device)
    w = w / w.sum()
    out = img
    for axis in (-1, -2):
        n = out.shape[axis]
        idx = torch.arange(n, device=img.device)
        acc = torch.zeros_like(out)
        for k in range(-r, r + 1):
            acc = acc + w[k + r] * out.index_select(axis, (idx + k).clamp(0, n - 1))
        out = acc
    return out


def grey_of(rgb: torch.Tensor, cfg: Params) -> torch.Tensor:
    """(n, H, W, 3) uint8 -> (n, H, W) float32 on the device: the luma (ITU-R BT.601), smoothed."""
    g = rgb.to(cfg.device).to(torch.float64)
    grey = (299 * g[..., 0] + 587 * g[..., 1] + 114 * g[..., 2]) / 1000.0
    return blur(grey, cfg.blur_sigma_px).to(torch.float32)


def make_maps(depth: np.ndarray, conf: np.ndarray, chunk: Sequence[int], grey: torch.Tensor, frames: Sequence[int],
              cfg: Params) -> Maps:
    """Maps from the keyframes' records (depth, confidence, the cube that owns each) and their grey images
    (:func:`grey_of`, on the records' grid). The pixels that judge are the chain's own: not sky, a positive depth,
    and a confidence over the pipeline's ONE floor — per cube, ``min + conf_min_norm x (max - min)``."""
    dev = cfg.device
    d = torch.as_tensor(np.asarray(depth, np.float32), device=dev)
    c = torch.as_tensor(np.asarray(conf, np.float32), device=dev)
    ck = np.asarray(chunk, np.int64)
    ok = torch.isfinite(d) & (d > 0) & torch.isfinite(c) & (c > SKY_CONF)
    valid = torch.zeros_like(ok)
    for k in np.unique(ck):
        rows = torch.as_tensor(np.flatnonzero(ck == k), device=dev)
        vals = c[rows][ok[rows]]
        if vals.numel() == 0:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        valid[rows] = ok[rows] & (c[rows] >= lo + cfg.conf_floor_norm * (hi - lo))
    grey = grey.to(dev)
    if grey.shape != d.shape:
        raise CoherenceError(f"images {tuple(grey.shape)} and depth maps {tuple(d.shape)} are not on one grid")
    return Maps(depth=d, valid=valid, grey=grey, frames=tuple(int(f) for f in frames))


# ── the pairs (state independent: the same judges for every state) ────────────────────────────────────────────────────
def neighbour_pairs(n: int, offsets: Sequence[int]) -> List[Tuple[int, int]]:
    return [(k, k + int(o)) for o in sorted(set(int(o) for o in offsets)) for k in range(n - int(o))]


def pass_pairs(covis: Sequence[Sequence[float]], frames: Sequence[int], min_gap: int, per_keyframe: int
               ) -> List[Tuple[int, int]]:
    """For each keyframe, its ``per_keyframe`` partners of OTHER PASSES: among the keyframes more than ``min_gap``
    keyframes away, those the first pass's map says share the most with it (the smaller of the two directed shares;
    ties to the nearer in the walk, then the lower index). ``covis``: rows [frame_a, frame_b, c_ab, c_ba]."""
    pos = {int(f): i for i, f in enumerate(frames)}
    cand: Dict[int, List[Tuple[float, int, int]]] = {}
    for a, b, cab, cba in covis:
        ia, ib = pos.get(int(a)), pos.get(int(b))
        if ia is None or ib is None or abs(ia - ib) <= int(min_gap):
            continue
        c = float(min(cab, cba))
        if not c > 0.0:
            continue
        cand.setdefault(ia, []).append((-c, abs(ia - ib), ib))
        cand.setdefault(ib, []).append((-c, abs(ia - ib), ia))
    out = set()
    for i, lst in cand.items():
        for _c, _gap, j in sorted(lst)[:int(per_keyframe)]:
            out.add((min(i, j), max(i, j)))
    return sorted(out)


# ── the measure ───────────────────────────────────────────────────────────────────────────────────────────────────────
COLUMNS = ("i", "j", "kind", "measured", "n", "z_m", "bias_rel", "spread_rel", "along_cm", "along_err_cm",
           "shift_x_px", "shift_y_px", "across_cm", "across_err_cm", "ncc")


def _median(x: torch.Tensor) -> torch.Tensor:
    return torch.quantile(x, 0.5)


def _peak(sc: torch.Tensor, R: int) -> Tuple[float, float, float]:
    """The shift (dx, dy) of the highest correlation of the (2R+1, 2R+1) table ``sc`` [dy, dx], refined by the
    parabola through its neighbours along each axis when it is not on the window's edge."""
    flat = int(torch.argmax(sc))
    by, bx = flat // (2 * R + 1), flat % (2 * R + 1)

    def sub(v0, v1, v2):
        den = float(v0 - 2.0 * v1 + v2)
        return 0.0 if abs(den) < 1e-12 else 0.5 * float(v0 - v2) / den
    sy = (by - R) + (sub(sc[by - 1, bx], sc[by, bx], sc[by + 1, bx]) if 0 < by < 2 * R else 0.0)
    sx = (bx - R) + (sub(sc[by, bx - 1], sc[by, bx], sc[by, bx + 1]) if 0 < bx < 2 * R else 0.0)
    return float(sx), float(sy), float(sc[by, bx])


def _ncc_table(sums, n: int, R: int) -> torch.Tensor:
    """Normalised cross-correlation for every shift from the additive sums (Sa, Saa, Sb, Sbb, Sab)."""
    Sa, Saa, Sb, Sbb, Sab = sums
    va = Saa - Sa * Sa / n
    vb = Sbb - Sb * Sb / n
    den = torch.sqrt(torch.clamp(va, min=0.0) * torch.clamp(vb, min=0.0))
    sc = torch.where(den > 0, (Sab - Sa * Sb / n) / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(den))
    return sc.reshape(2 * R + 1, 2 * R + 1)


class _Carrier:
    """A state's geometry prepared once: the rays of the carried pixels (the lens removed) and the window of shifts."""

    def __init__(self, maps: Maps, state: State, cfg: Params):
        self.maps, self.state, self.cfg = maps, state, cfg
        dev = cfg.device
        H, W = maps.hw
        s = int(cfg.pixel_stride)
        vv, uu = torch.meshgrid(torch.arange(0, H, s, device=dev), torch.arange(0, W, s, device=dev), indexing="ij")
        self.rows, self.cols = vv.reshape(-1), uu.reshape(-1)
        xn, yn = unproject(self.cols.to(torch.float64), self.rows.to(torch.float64), state.K, state.dist)
        self.xn, self.yn = xn, yn
        self.ray = torch.sqrt(xn * xn + yn * yn + 1.0)          # distance along the ray per unit of depth
        R = int(cfg.search_px)
        dy, dx = torch.meshgrid(torch.arange(-R, R + 1, device=dev), torch.arange(-R, R + 1, device=dev), indexing="ij")
        self.dy, self.dx, self.R = dy.reshape(-1), dx.reshape(-1), R

    def source(self, i: int):
        """Keyframe i's carried pixels in the world: (points, their grey, which of the sampled pixels they are)."""
        m, st = self.maps, self.state
        z = m.depth[i][self.rows, self.cols].to(torch.float64) * st.factor[i]
        ok = m.valid[i][self.rows, self.cols] & (z * self.ray <= self.cfg.range_m)
        idx = torch.nonzero(ok, as_tuple=False).reshape(-1)
        z = z[idx]
        cam = torch.stack([self.xn[idx] * z, self.yn[idx] * z, z], 1)
        R, C = st.c2w[i, :3, :3], st.c2w[i, :3, 3]
        return cam @ R.T + C, m.grey[i][self.rows[idx], self.cols[idx]].to(torch.float64)

    def into(self, Xw: torch.Tensor, grey_i: torch.Tensor, j: int) -> Optional[dict]:
        m, st, cfg, R = self.maps, self.state, self.cfg, self.R
        H, W = m.hw
        Rj, Cj = st.c2w[j, :3, :3], st.c2w[j, :3, 3]
        Xc = (Xw - Cj) @ Rj
        zz = Xc[:, 2]
        front = zz > 0
        zs = torch.where(front, zz, torch.ones_like(zz))
        pu, pv = project(Xc[:, 0] / zs, Xc[:, 1] / zs, st.K, st.dist)
        iu, iv = torch.round(pu).to(torch.int64), torch.round(pv).to(torch.int64)
        u0, v0 = torch.floor(pu).to(torch.int64), torch.floor(pv).to(torch.int64)
        # every shifted sample is read BETWEEN four pixels: the window must fit around (u0, v0) and (u0 + 1, v0 + 1)
        inb = front & (u0 >= R) & (u0 <= W - R - 2) & (v0 >= R) & (v0 <= H - R - 2)
        k = torch.nonzero(inb, as_tuple=False).reshape(-1)
        if k.numel() < cfg.min_pixels:
            return None
        iu, iv, zz, a = iu[k], iv[k], zz[k], grey_i[k]
        u0, v0, fu, fv = u0[k], v0[k], (pu - torch.floor(pu))[k], (pv - torch.floor(pv))[k]
        dj = m.depth[j][iv, iu].to(torch.float64) * st.factor[j]
        mm = m.valid[j][iv, iu]
        k = torch.nonzero(mm, as_tuple=False).reshape(-1)
        if k.numel() < cfg.min_pixels:
            return None
        iu, iv, zz, a, dj = iu[k], iv[k], zz[k], a[k], dj[k]
        u0, v0, fu, fv = u0[k], v0[k], fu[k], fv[k]
        e = (zz - dj) / dj
        med = _median(e)
        mad = _median((e - med).abs())
        same = (e - med).abs() <= cfg.same_surface_sigmas * ROBUST * mad if float(mad) > 0 else (e == med)
        k = torch.nonzero(same, as_tuple=False).reshape(-1)
        n = int(k.numel())
        if n < cfg.min_pixels:
            return None
        zz, a, e = zz[k], a[k], e[k]
        u0, v0, fu, fv = u0[k], v0[k], fu[k], fv[k]
        zmed = float(_median(zz))
        half = torch.arange(n, device=e.device) % 2 == 0                 # two halves of the pixels, in raster order
        bias, spread = float(_median(e)), float(_median((e - _median(e)).abs()))
        bA, bB = float(_median(e[half])), float(_median(e[~half]))
        # across the ray: the carried grey against the other keyframe's grey around where each pixel lands — read at
        # the EXACT landing position (between four pixels), so that a coherent state reads a shift of zero and not
        # the rounding of a pixel. The sums the correlation is made of are additive: they are accumulated block by
        # block of pixels (a fixed block: the same additions in the same order on any device), for the two halves.
        g = m.grey[j]
        acc = {True: None, False: None}
        for s0 in range(0, n, NCC_BLOCK):
            sl = slice(s0, min(n, s0 + NCC_BLOCK))
            rr, cc = v0[sl, None] + self.dy[None, :], u0[sl, None] + self.dx[None, :]                        # (b, S)
            wu, wv = fu[sl, None], fv[sl, None]
            B = ((1.0 - wv) * ((1.0 - wu) * g[rr, cc].to(torch.float64) + wu * g[rr, cc + 1].to(torch.float64))
                 + wv * ((1.0 - wu) * g[rr + 1, cc].to(torch.float64) + wu * g[rr + 1, cc + 1].to(torch.float64)))
            aa, hh = a[sl], half[sl]
            for side in (True, False):
                sel = hh if side else ~hh
                x, Y = aa[sel], B[sel]
                part = (x.sum(), (x * x).sum(), Y.sum(0), (Y * Y).sum(0), (x[:, None] * Y).sum(0), int(x.numel()))
                acc[side] = part if acc[side] is None else tuple(p + q for p, q in zip(acc[side], part))
        sA, sB = acc[True], acc[False]
        tabA, tabB = _ncc_table(sA[:5], sA[5], R), _ncc_table(sB[:5], sB[5], R)
        tab = _ncc_table(tuple(x + y for x, y in zip(sA[:5], sB[:5])), sA[5] + sB[5], R)
        sx, sy, ncc = _peak(tab, R)
        ax, ay, _ = _peak(tabA, R)
        bx, by, _ = _peak(tabB, R)
        cm_px = 100.0 * zmed / float(st.K[0])                              # one pixel at the surface, in cm
        return {"n": n, "z_m": zmed, "bias_rel": bias, "spread_rel": spread,
                "along_cm": abs(bias) * zmed * 100.0, "along_err_cm": abs(bA - bB) * zmed * 100.0,
                "shift_x_px": sx, "shift_y_px": sy, "across_cm": math.hypot(sx, sy) * cm_px,
                "across_err_cm": math.hypot(ax - bx, ay - by) * cm_px, "ncc": ncc}


def measure(maps: Maps, state: State, pairs: Sequence[Tuple[int, int]], kinds: Sequence[int],
            cfg: Params, progress=None) -> Dict[str, np.ndarray]:
    """Every pair in both directions (2 x len(pairs) rows, pair p at rows 2p and 2p + 1). A direction with fewer
    than ``min_pixels`` pixels on a common surface is not measured (``measured`` 0, its numbers NaN)."""
    if len(pairs) != len(kinds):
        raise CoherenceError("one kind per pair")
    N = len(maps.frames)
    if tuple(state.c2w.shape) != (N, 4, 4) or tuple(state.factor.shape) != (N,):
        raise CoherenceError(f"the state '{state.label}' is not of these {N} keyframes")
    car = _Carrier(maps, state, cfg)
    rows = np.full((2 * len(pairs), len(COLUMNS)), np.nan)
    order = sorted(range(2 * len(pairs)), key=lambda r: (pairs[r // 2][r % 2], pairs[r // 2][1 - r % 2], r))
    cur, src = -1, None
    for q, r in enumerate(order):
        p, d = r // 2, r % 2
        i, j = pairs[p][d], pairs[p][1 - d]
        if i != cur:
            cur, src = i, car.source(i)
        res = car.into(src[0], src[1], j) if src[0].shape[0] >= cfg.min_pixels else None
        rows[r, :4] = (i, j, int(kinds[p]), 0.0 if res is None else 1.0)
        if res is not None:
            rows[r, 4:] = [res[c] for c in COLUMNS[4:]]
        if progress is not None and (q + 1) % 2000 == 0:
            progress(q + 1, len(order))
    return {c: rows[:, k].copy() for k, c in enumerate(COLUMNS)}


def summary(meas: Dict[str, np.ndarray]) -> dict:
    """The session's figures of one state: per kind and metric the median and the 90 % of the measured directions."""
    out = {}
    for kind, name in KIND_NAMES.items():
        sel = (meas["kind"] == kind) & (meas["measured"] == 1)
        rep = {"directions": int((meas["kind"] == kind).sum()), "measured": int(sel.sum())}
        for metric in METRICS:
            v = meas[metric][sel]
            rep[metric] = ({"median": float(np.median(v)), "p90": float(np.percentile(v, 90))} if len(v) else None)
        out[name] = rep
    return out


# ── the decision, zone by zone ────────────────────────────────────────────────────────────────────────────────────────
def judge_zones(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray], zone_of: Sequence[int],
                cfg: Params, decide=None) -> dict:
    """``after`` against ``before`` (the same pairs, row by row), zone by zone, with the USER's rule. A direction
    judges the zones of its two keyframes. Per zone, kind and metric: WORSE when the rule says ``before`` improves on
    ``after``, BETTER the other way round, else THE SAME (or too few judges). A zone is WORSE when any of its series
    is, BETTER when one is better and none worse."""
    if decide is None:
        from precision.judge_torch import decide_change_t

        def decide(b, a, error, clusters):
            return decide_change_t(b, a, error=error, error_factor=cfg.error_factor, confidence=cfg.confidence,
                                   min_judges=cfg.min_judges, clusters=clusters, n_boot=cfg.bootstrap, seed=cfg.seed,
                                   device=cfg.device)
    for c in ("i", "j", "kind"):
        if not np.array_equal(before[c], after[c]):
            raise CoherenceError("the two measurements are not of the same pairs in the same order")
    zone_of = np.asarray(zone_of, np.int64)
    both = (before["measured"] == 1) & (after["measured"] == 1)
    zi, zj = zone_of[before["i"].astype(np.int64)], zone_of[before["j"].astype(np.int64)]
    pair_id = np.arange(len(both)) // 2
    zones = {}
    for z in sorted(set(int(v) for v in np.unique(zone_of))):
        rep = {"verdict": "same", "series": {}}
        for kind, kname in KIND_NAMES.items():
            sel = both & (before["kind"] == kind) & ((zi == z) | (zj == z))
            for metric in METRICS:
                b, a = before[metric][sel], after[metric][sel]
                key = f"{kname} / {metric}"
                n_judges = int(len(np.unique(pair_id[sel])))
                if n_judges < cfg.min_judges:
                    rep["series"][key] = {"verdict": "too few judges", "judges": n_judges}
                    continue
                err = float(max(np.median(before[metric.replace("_cm", "_err_cm")][sel]),
                                np.median(after[metric.replace("_cm", "_err_cm")][sel])))
                better = decide(b, a, err, pair_id[sel])
                worse = decide(a, b, err, pair_id[sel])
                v = "worse" if worse["improves"] else ("better" if better["improves"] else "same")
                rep["series"][key] = {"verdict": v, "judges": n_judges, "before_median_cm": float(np.median(b)),
                                      "after_median_cm": float(np.median(a)), "error_cm": err,
                                      "median_change_cm": float(np.median(a - b)),
                                      "ci_cm": [-float(better["ci_high"]), -float(better["ci_low"])]}
        vs = [s["verdict"] for s in rep["series"].values()]
        rep["verdict"] = "worse" if "worse" in vs else ("better" if "better" in vs else "same")
        zones[z] = rep
    return {"zones": zones, "worse": [z for z, r in zones.items() if r["verdict"] == "worse"],
            "better": [z for z, r in zones.items() if r["verdict"] == "better"],
            "same": [z for z, r in zones.items() if r["verdict"] == "same"]}


# ── a session's own data ──────────────────────────────────────────────────────────────────────────────────────────────
RECORDS_REL = Path("omega_run") / "results_output"
FRAME_BATCH = 256               # frames decoded and smoothed at a time (memory; the result does not depend on it)


def _load_record(path: Path):
    with np.load(path) as z:
        return (np.asarray(z["depth"], np.float32), np.asarray(z["conf"], np.float32), int(z["chunk"]),
                np.asarray(z["pose_c2w"], np.float64))


def session_frames(out: Path) -> List[int]:
    return [int(float(v)) for v in (Path(out) / "camera_frames.txt").read_text().split()]


def grid_camera(cam, hw: Tuple[int, int]):
    """(K, lens, grid map) of the session camera (precision.camera.CameraModel, native pixels) on the records' grid
    of ``hw`` — through the pipeline's own exact relation between the two grids (the crop, the resize and the
    padding Omega's run made)."""
    from precision.camera import K_native_to_grid
    from precision.depth_sweep import record_grid
    g = record_grid(cam, (int(hw[0]), int(hw[1])))
    Kg = K_native_to_grid(np.asarray(cam.K(), np.float64), g)
    K = (float(Kg[0, 0]), float(Kg[1, 1]), float(Kg[0, 2]), float(Kg[1, 2]))
    lens = tuple(float(v) for v in np.asarray(cam.dist(), np.float64).ravel())
    if len(lens) != len(("k1", "k2", "p1", "p2")):
        raise CoherenceError(f"the session camera's lens has {len(lens)} coefficient(s); the measure reads k1, k2, p1, p2")
    return K, lens, g


def session_maps(session_dir: Path, cam, cfg: Params, threads: int = 8, log=print):
    """(maps, zone_of, record poses) of a session: its keyframes' records and frames. ``zone_of``: the cube that owns
    each keyframe; the record poses are the cameras the depth records were made with (their units are the depth's).
    The records are read on threads (inflating a file is the disk's side); the frames are decoded on the device
    (jpeg_gpu) and brought to the records' grid as Omega's run did — the crop, the resize, the padding; what lies in
    the padding judges nothing."""
    import jpeg_gpu
    session_dir = Path(session_dir)
    out = session_dir / "output"
    frames = session_frames(out)
    rdir = out / RECORDS_REL
    missing = [f for f in frames if not (rdir / f"frame_{f}.npz").exists()]
    if missing:
        raise CoherenceError(f"{len(missing)} keyframe(s) have no depth record under {rdir} (first: {missing[:5]}) — "
                             f"the measure needs every keyframe's depth")
    with ThreadPoolExecutor(max_workers=int(threads)) as ex:
        recs = list(ex.map(_load_record, [rdir / f"frame_{f}.npz" for f in frames]))
    depth, conf = np.stack([r[0] for r in recs]), np.stack([r[1] for r in recs])
    zone_of = np.array([r[2] for r in recs], np.int64)
    rec_c2w = np.stack([r[3] for r in recs])
    del recs
    N, H, W = depth.shape
    _K, _lens, g = grid_camera(cam, (H, W))
    dev = cfg.device
    grey = torch.zeros((N, H, W), dtype=torch.float32, device=dev)
    paths = [session_dir / "frames" / f"{f:06d}.jpg" for f in frames]
    for s0 in range(0, N, FRAME_BATCH):
        st = jpeg_gpu.read_rgb_stack(paths[s0:s0 + FRAME_BATCH], dev if dev != "cpu" else None).to(dev)
        if tuple(st.shape[1:3]) != (g.native_h, g.native_w):
            raise CoherenceError(f"frames of {tuple(st.shape[1:3])} for a camera of {(g.native_h, g.native_w)}")
        st = st[:, g.crop_y:g.crop_y + g.crop_h, g.crop_x:g.crop_x + g.crop_w]
        if tuple(st.shape[1:3]) != (g.content_h, g.content_w):
            st = jpeg_gpu.resize_bilinear(st.permute(0, 3, 1, 2), g.content_h, g.content_w).permute(0, 2, 3, 1)
        grey[s0:s0 + st.shape[0], g.pad_top:g.pad_top + g.content_h, g.pad_left:g.pad_left + g.content_w] = \
            grey_of(st, cfg)
        del st
    maps = make_maps(depth, conf, zone_of, grey, frames, cfg)
    inside = torch.zeros((H, W), dtype=torch.bool, device=dev)
    inside[g.pad_top:g.pad_top + g.content_h, g.pad_left:g.pad_left + g.content_w] = True
    maps.valid &= inside
    log(f"[coherence] maps: {N} keyframes of {len(np.unique(zone_of))} cube(s) on a grid of {W}x{H}, on {dev}")
    return maps, zone_of, rec_c2w


def session_pairs(session_dir: Path, frames: Sequence[int], cfg: Params) -> Tuple[list, list]:
    """(pairs, kinds) of a session: its neighbours, and its partners of other passes by the first pass's map
    (intake/cubes.json — the plan's co-visibility, the same list the chain's revisits come from)."""
    nb = neighbour_pairs(len(frames), cfg.neighbour_offsets)
    cv = json.loads((Path(session_dir) / "intake" / "cubes.json").read_text()).get("kf_covisibility") or {}
    ps = pass_pairs(cv.get("pairs") or [], frames, cfg.pass_min_gap, cfg.min_judges)
    return nb + ps, [KIND_NEIGHBOUR] * len(nb) + [KIND_PASS] * len(ps)
