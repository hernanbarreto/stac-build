"""F6 — prior-guided plane-sweep depth at native resolution (claude_stac.txt §4-F6).

Per keyframe i, in the UNDISTORTED native frame (F0's maps, K_new = K):

Prior. z0 = Omega's depth × s_k, taken to native with ``native_depth.
guided_upsample`` (joint bilateral: space + colour + depth) on the Omega grid
(F0's GridMap), then through the undistortion map. s_k is MEASURED per keyframe:
the median ratio of the F4 tracks' depth — triangulated with F5's camera and
poses — to Omega's depth at the same pixel (the nearest keyframes pool their
ratios until ``min_scale_samples``). Omega's record carries the units of the
reconstruction before the metric lock; the ratio carries it to the current epoch
whatever happened in between, and needs no bookkeeping of which scale was applied.

Hypotheses. ``n_hyp`` samples uniform in INVERSE depth over [z0(1−β), z0(1+β)]
plus z0 itself. β per pixel is the calibrated |error| quantile of the prior at
its confidence and distance (``confidence.py``: tier-0 reference when this epoch
has one, else the landmarks), capped by ``beta_max``. Hypothesis t of pixel p is
the prior surface SCALED ABOUT THE CAMERA CENTRE: for a locally planar prior that
is exactly the plane (z, n0) — a scaling about the centre keeps a plane's normal
— so warping every view through the per-pixel hypothesis depth and correlating
P×P windows is the plane-induced homography of the prior's own normal, one
``grid_sample`` per view and hypothesis. (Inside one window β, and so the scale,
may vary pixel to pixel: the window then follows a slightly bent plane. Declared.)

Score. ZNCC over P×P windows (box filters); a window with any sample outside a
view, or whose standard deviation is under the 8-bit quantisation noise
(1/(255·√12)), has no signal (−1). Views aggregate as the mean of the best
``best_k`` (occlusion-robust). Best hypothesis → parabolic sub-sample refinement
in inverse depth (kept only when the re-evaluated score does not fall).
``propagation_iters`` rounds: each pixel's candidate is the colour-weighted
median of its window's depths (weight exp(−|ΔI|/σ_c) × the neighbour's own
score; σ_c = the image's median absolute neighbour difference), re-evaluated,
kept when it scores higher.

Noise floor. MEASURED: the same pipeline on ``null_frames`` keyframes with every
view's image replaced by a keyframe that shares no surface with the reference —
the scores a pixel reaches when nothing corresponds. The floor is their
``null_confidence`` quantile, per texture bin (the reference window's standard
deviation — weak texture scores higher by chance, so it gets its own floor).

Consistency (at native). Pixel p with depth d → view j: consistent when
|z_ij − d_j| / d_j < τ_rel and the round trip through d_j lands within τ_px of
p, in ≥ ``min_consistent_views`` keyframes (neighbours' depths count only where
they have signal themselves). τ_px = ``tau_px_k`` × the RMS of F5's held-out
reprojection; τ_rel = the median relative disagreement between neighbouring
priors on the surfaces they share — the intra-window quantity F3 measures, here
over the whole session at the production resolution.

Tiers (``source``): 0 = ZNCC over the floor AND consistent; 1 = no ZNCC signal
but the PRIOR is consistent prior-against-prior in ≥ ``prior_fill_min_views``
(``prior_fill: keep``); ≥ 2 = discarded, the code names why. Nothing is invented:
a pixel that has neither stays discarded.

Outputs: ``output/depth_native/frame_<n>.npz`` {depth f32, ncc f32,
n_consistent u8, source u8, normal f32, residual_rel f32} +
``output/depth_native/report.json``; ``output/precision/confidence_calibration.json``
re-measured against tier 0.

Determinism: the whole run executes with torch's deterministic algorithms on (an op
without a deterministic kernel raises instead of running), TF32 off and the seed set
(``precision.tracks.deterministic_torch``); ties in the view ranking and in the
propagation's weighted median are broken by a stable sort; the CLI fixes cuBLAS's
workspace (CUBLAS_WORKSPACE_CONFIG) before CUDA initialises. Identical inputs give
bit-identical arrays.

Needs F5 APPLIED (its camera, its poses, its localised witnesses) — the witness
poses live in F5's world. GPU when available.

CLI: ``python -m precision.depth_sweep --session <dir>``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEPTH_DIRNAME = "depth_native"
WORK_DIRNAME = "_work"
REPORT_NAME = "report.json"
REPORT_VERSION = 1
PROVENANCE = "tool_measured"
LOG_TAG = "[depth-sweep]"

SOURCE_SWEEP = 0                 # tier 0
SOURCE_PRIOR_FILL = 1            # tier 1
DISCARD_NO_PRIOR = 2             # Omega left no depth here: no hypothesis to sweep
DISCARD_NO_SIGNAL = 3            # no ZNCC over the floor and the prior is not confirmed
DISCARD_INCONSISTENT = 4         # ZNCC signal, but the other views do not confirm the depth
DISCARD_PRIOR_FILL_DROPPED = 5   # tier 1 under prior_fill: drop
DISCARD_EXCLUDED = 6             # I2 exclusion mask
DISCARD_LOW_CONF = 7             # tier-1 prior under the pipeline's ONE confidence floor
DISCARD_CONTRADICTED = 8         # a consistency view sees FREE SPACE through the point
DISCARD_NOT_INDEPENDENT = 9      # tier-1 prior confirmed only by views within one visit of walk
SOURCE_NAMES = {SOURCE_SWEEP: "tier0", SOURCE_PRIOR_FILL: "tier1_prior_fill",
                DISCARD_NO_PRIOR: "no_prior", DISCARD_NO_SIGNAL: "no_signal",
                DISCARD_INCONSISTENT: "inconsistent",
                DISCARD_PRIOR_FILL_DROPPED: "prior_fill_dropped",
                DISCARD_EXCLUDED: "excluded_mask",
                DISCARD_LOW_CONF: "prior_low_conf",
                DISCARD_CONTRADICTED: "contradicted",
                DISCARD_NOT_INDEPENDENT: "prior_not_independent"}

GRAY_MAX = float(np.iinfo(np.uint8).max)          # 8-bit frames, read on [0, 1]
# the std of 8-bit rounding noise on the [0, 1] scale (a uniform error of one
# level has variance 1/12): a window flatter than this carries no signal
QUANT_STD = 1.0 / (GRAY_MAX * math.sqrt(12))
# a bilinear sample of a validity mask is 1 only when all four neighbours are valid
FULL = 1.0 - 1e-6


class DepthSweepError(RuntimeError):
    """A structural impossibility of the sweep — with the exact reason."""


# ── small geometry ───────────────────────────────────────────────────────

def parabola_vertex(x0, x1, x2, y0, y1, y2):
    """Vertex x of the parabola through three points (any spacing, arrays), and
    whether it is a maximum inside [x0, x2]."""
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    den = np.where(den == 0, np.nan, den) if isinstance(den, np.ndarray) else den
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
    xv = -b / (2 * a)
    lo, hi = np.minimum(x0, x2), np.maximum(x0, x2)
    ok = (a < 0) & (xv >= lo) & (xv <= hi)
    return xv, ok


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame unit normals by central differences of the back-projected
    points, facing the camera; NaN where a neighbour has no depth."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    z = depth.astype(np.float64)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    n = np.full((H, W, 3), np.nan)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    c = np.cross(dx, dy)
    ok = ((z[1:-1, 2:] > 0) & (z[1:-1, :-2] > 0) & (z[2:, 1:-1] > 0) & (z[:-2, 1:-1] > 0)
          & (z[1:-1, 1:-1] > 0))
    nn = np.linalg.norm(c, axis=-1, keepdims=True)
    c = np.where(nn > 0, c / np.where(nn > 0, nn, 1), np.nan)
    c = np.where((np.sum(c * P[1:-1, 1:-1], -1, keepdims=True) > 0), -c, c)
    n[1:-1, 1:-1] = np.where(ok[..., None], c, np.nan)
    return n


# ── the sweep core (torch) ───────────────────────────────────────────────

class _Frame:
    """A reference keyframe and its views on one device."""

    def __init__(self, ref: np.ndarray, K: np.ndarray, w2c_ref: np.ndarray,
                 views: Sequence[np.ndarray], view_valid: Sequence[np.ndarray],
                 view_w2c: Sequence[np.ndarray], patch: int, best_k: int, device):
        import torch
        self.t = torch
        self.dev = device
        self.H, self.W = ref.shape
        self.P = int(patch)
        self.k = int(min(best_k, len(views)))
        f32 = dict(dtype=torch.float32, device=device)
        self.I = torch.as_tensor(np.ascontiguousarray(ref), **f32)[None, None]
        self.J = torch.as_tensor(np.stack(views), **f32)[:, None]            # V,1,H,W
        self.Jv = torch.as_tensor(np.stack(view_valid).astype(np.float32), **f32)[:, None]
        Kt = torch.as_tensor(K, dtype=torch.float64, device=device)
        v, u = torch.meshgrid(torch.arange(self.H, device=device, dtype=torch.float64),
                              torch.arange(self.W, device=device, dtype=torch.float64),
                              indexing="ij")
        rays = torch.stack([(u - Kt[0, 2]) / Kt[0, 0], (v - Kt[1, 2]) / Kt[1, 1],
                            torch.ones_like(u)]).reshape(3, -1)             # 3,HW
        c2w = np.linalg.inv(np.asarray(w2c_ref, np.float64))
        A, b = [], []
        for T in view_w2c:
            M = np.asarray(T, np.float64) @ c2w                               # ref cam → view cam
            A.append(K @ M[:3, :3])
            b.append(K @ M[:3, 3])
        At = torch.as_tensor(np.stack(A), dtype=torch.float64, device=device)
        self.Ar = (At @ rays).float()                                         # V,3,HW
        self.b = torch.as_tensor(np.stack(b), **f32)[:, :, None]              # V,3,1
        self.mI = self._box(self.I)
        self.vI = (self._box(self.I * self.I) - self.mI ** 2).clamp(min=0)

    def _box(self, x):
        return self.t.nn.functional.avg_pool2d(x, self.P, stride=1, padding=self.P // 2,
                                               count_include_pad=False)

    def warp(self, depth):
        """Views sampled at the reference pixels through ``depth`` (H,W): (V,1,H,W)
        images and validity."""
        F = self.t.nn.functional
        x = self.Ar * depth.reshape(1, 1, -1) + self.b                        # V,3,HW
        z = x[:, 2]
        ok = z > 1e-6
        zs = z.clamp(min=1e-6)
        u, v = x[:, 0] / zs, x[:, 1] / zs
        gx = u / max(self.W - 1, 1) * 2 - 1
        gy = v / max(self.H - 1, 1) * 2 - 1
        grid = self.t.stack([gx, gy], -1).reshape(len(x), self.H, self.W, 2)
        Jw = F.grid_sample(self.J, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True)
        Vw = F.grid_sample(self.Jv, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True)
        valid = (Vw > FULL) & ok.reshape(len(x), 1, self.H, self.W) \
            & (depth > 0)[None, None]
        return Jw, valid

    def score(self, depth):
        """Best-k mean ZNCC of the views at ``depth`` (H,W); −1 = no signal."""
        Jw, valid = self.warp(depth)
        vf = valid.float()
        Jw = Jw * vf
        mJ = self._box(Jw)
        vJ = (self._box(Jw * Jw) - mJ ** 2).clamp(min=0)
        cov = self._box(self.I * Jw) - self.mI * mJ
        full = self._box(vf) > FULL
        tex = (self.vI > QUANT_STD ** 2) & (vJ > QUANT_STD ** 2)
        ncc = self.t.where(full & tex, cov / self.t.sqrt((self.vI * vJ).clamp(min=1e-12)),
                           self.t.full_like(cov, -1.0)).clamp(-1, 1)[:, 0]    # V,H,W
        top = self.t.topk(ncc, self.k, dim=0).values
        return top.mean(0)

    def ref_std(self):
        return self.t.sqrt(self.vI)[0, 0]


def sweep_frame(fr: _Frame, z0: np.ndarray, beta: np.ndarray, n_hyp: int,
                propagation_iters: int, contrast_sigma: float) -> Dict[str, np.ndarray]:
    """Depth and score of one keyframe (numpy out). z0 ≤ 0 → no prior, no sweep."""
    t = fr.t
    dev = fr.dev
    z0t = t.as_tensor(z0, dtype=t.float32, device=dev)
    bt = t.as_tensor(beta, dtype=t.float32, device=dev)
    has = z0t > 0
    rho0 = t.where(has, 1.0 / z0t.clamp(min=1e-6), t.zeros_like(z0t))
    lo = rho0 / (1 + bt)
    hi = rho0 / (1 - bt).clamp(min=1e-6)       # β < 1 (config)
    s = t.linspace(0, 1, int(n_hyp), device=dev)[:, None, None]
    rho = t.cat([lo[None] + (hi - lo)[None] * s, rho0[None]], 0)
    rho, _ = t.sort(rho, 0)
    S = t.stack([fr.score(t.where(has, 1.0 / rho[h].clamp(min=1e-9), t.zeros_like(z0t)))
                 for h in range(rho.shape[0])])
    best = t.argmax(S, 0)
    n = rho.shape[0]
    g = lambda a, off: t.gather(a, 0, (best + off).clamp(0, n - 1)[None])[0]   # noqa: E731
    r1, s1 = g(rho, 0), g(S, 0)
    boundary = (best == 0) | (best == n - 1)
    xv, ok = parabola_vertex(g(rho, -1).cpu().numpy(), r1.cpu().numpy(), g(rho, 1).cpu().numpy(),
                             g(S, -1).cpu().numpy(), s1.cpu().numpy(), g(S, 1).cpu().numpy())
    ok = ok & ~boundary.cpu().numpy()
    rref = t.as_tensor(np.where(ok, xv, r1.cpu().numpy()), dtype=t.float32, device=dev)
    depth = t.where(has, 1.0 / rref.clamp(min=1e-9), t.zeros_like(z0t))
    sc = fr.score(depth)
    disc = t.where(has, 1.0 / r1.clamp(min=1e-9), t.zeros_like(z0t))
    keep = sc >= s1
    depth = t.where(keep, depth, disc)
    sc = t.where(keep, sc, s1)
    radius = fr.P // 2
    k = 2 * radius + 1
    F = t.nn.functional
    Iref = fr.I[0, 0]
    Iq = F.unfold(Iref[None, None], k, padding=radius)[0]                      # k²,HW
    wcol = t.exp(-t.abs(Iq - Iref.reshape(1, -1)) / max(contrast_sigma, QUANT_STD))
    for _ in range(int(propagation_iters)):
        wq = F.unfold(sc.clamp(min=0)[None, None], k, padding=radius)[0] * wcol
        # the weighted median of the window, weights = colour × neighbour score
        cand = _weighted_median_w(depth, wq, radius)
        cs = fr.score(cand)
        better = (cs > sc) & has
        depth = t.where(better, cand, depth)
        sc = t.where(better, cs, sc)
    sc = t.where(has, sc, t.full_like(sc, -1.0))
    return {"depth": depth.cpu().numpy().astype(np.float32),
            "score": sc.cpu().numpy().astype(np.float32),
            "ref_std": fr.ref_std().cpu().numpy().astype(np.float32),
            "boundary": (boundary & has).cpu().numpy()}


def _weighted_median_w(depth, w_unfolded, radius: int):
    """Per pixel: the weighted median of its (2r+1)² window's depths, the window
    weights already unfolded (k², HW)."""
    import torch
    F = torch.nn.functional
    k = 2 * radius + 1
    H, W = depth.shape
    d = F.unfold(depth[None, None], k, padding=radius)[0]
    w = torch.where(d > 0, w_unfolded, torch.zeros_like(w_unfolded))
    # STABLE: equal depths keep their window order, so the weights accumulate in one
    # order on every device and run (an unstable sort may permute ties)
    ds, o = torch.sort(d, dim=0, stable=True)
    ws = torch.gather(w, 0, o)
    # prefix sum in a fixed sequential order over the k² rows: torch.cumsum on a CUDA
    # float tensor has no deterministic kernel (use_deterministic_algorithms raises)
    cw = ws.clone()
    for r in range(1, cw.shape[0]):
        cw[r] += cw[r - 1]
    tot = cw[-1:]
    idx = torch.searchsorted(cw.T.contiguous(), (0.5 * tot).T.contiguous()).T
    med = torch.gather(ds, 0, idx.clamp(max=k * k - 1))[0]
    return torch.where(tot[0] > 0, med, depth.reshape(-1)).reshape(H, W)


def contrast_sigma(img: np.ndarray, valid: np.ndarray) -> float:
    """The image's own contrast scale: median |neighbour difference| (valid pairs)."""
    dx = np.abs(np.diff(img, axis=1))[valid[:, 1:] & valid[:, :-1]]
    dy = np.abs(np.diff(img, axis=0))[valid[1:] & valid[:-1]]
    d = np.concatenate([dx, dy])
    return float(np.median(d)) if d.size else QUANT_STD


# ── the noise floor ──────────────────────────────────────────────────────

def floor_table(null_scores: np.ndarray, null_std: np.ndarray, confidence: float,
                n_bins: int) -> Dict[str, Any]:
    """Per texture bin (reference window std): the ``confidence`` quantile of the
    scores reached with non-corresponding views."""
    ok = np.isfinite(null_scores) & (null_scores > -1)
    s, d = null_scores[ok], null_std[ok]
    if s.size == 0:
        raise DepthSweepError("the null run produced no scored pixel — no floor can be measured")
    edges = np.unique(np.quantile(d, np.linspace(0, 1, int(n_bins) + 1)))
    if len(edges) < 2:
        edges = np.array([d.min(), d.max() + 1e-12])
    b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2)
    fl = [float(np.quantile(s[b == i], confidence)) if np.any(b == i) else float(np.quantile(s, confidence))
          for i in range(len(edges) - 1)]
    return {"std_edges": edges.tolist(), "floor": fl, "confidence": float(confidence),
            "n": int(s.size), "global_floor": float(np.quantile(s, confidence))}


def floor_of(table: Dict[str, Any], std: np.ndarray) -> np.ndarray:
    e = np.asarray(table["std_edges"])
    f = np.asarray(table["floor"])
    return f[np.clip(np.searchsorted(e, std, side="right") - 1, 0, len(f) - 1)]


# ── consistency at native ────────────────────────────────────────────────

def _nanmedian0(R):
    """``torch.nanmedian(R, 0).values`` without its indices: the CUDA kernel that returns
    them has no deterministic implementation (torch refuses it under deterministic
    algorithms). The lower median of the non-NaN values along dim 0 — the same element
    nanmedian picks — NaN where there is none (NaN sorts last)."""
    import torch
    n = (~torch.isnan(R)).sum(0)
    s = torch.sort(R, dim=0, stable=True).values
    med = torch.gather(s, 0, ((n - 1).clamp(min=0) // 2)[None])[0]
    return torch.where(n > 0, med, torch.full_like(med, float("nan")))


def consistency(depth_i: np.ndarray, K: np.ndarray, w2c_i: np.ndarray,
                nbr_depths: Sequence[np.ndarray], nbr_w2c: Sequence[np.ndarray],
                tau_rel: float, tau_px: float, device=None, return_good: bool = False,
                nbr_bad_margin: Optional[Sequence] = None):
    """(n_consistent u8, residual_rel f32 = median |z_ij − d_j|/d_j over the
    consistent views, NaN where none, n_contradict u8 = views that measured a surface
    FARTHER along the ray — free space through the point) for one depth map against
    the given views'. With ``return_good`` a fourth item: the per-view agreement
    masks (bool H×W each), for rules about WHICH views agree. ``nbr_bad_margin`` (one
    per view: a scalar or that VIEW's H×W map) is the relative margin beyond which the
    view's farther surface counts as a contradiction — the view's OWN depth error
    (τ_rel for a measured depth, the calibrated prior error for Omega's), default
    τ_rel. A view's depth ≤ 0 is 'no measurement there'."""
    import torch
    dev = device or "cpu"
    H, W = depth_i.shape
    Kt = torch.as_tensor(K, dtype=torch.float64, device=dev)
    di = torch.as_tensor(depth_i, dtype=torch.float64, device=dev).reshape(-1)
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64),
                          torch.arange(W, device=dev, dtype=torch.float64), indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)
    Pc = torch.stack([(u - Kt[0, 2]) / Kt[0, 0] * di, (v - Kt[1, 2]) / Kt[1, 1] * di, di])
    c2w_i = torch.as_tensor(np.linalg.inv(w2c_i), dtype=torch.float64, device=dev)
    w2c_it = torch.as_tensor(w2c_i, dtype=torch.float64, device=dev)
    Pw = c2w_i[:3, :3] @ Pc + c2w_i[:3, 3:4]
    n_cons = torch.zeros(H * W, dtype=torch.int32, device=dev)
    n_bad = torch.zeros(H * W, dtype=torch.int32, device=dev)
    rels = []
    goods = []
    margins = list(nbr_bad_margin) if nbr_bad_margin is not None else [None] * len(nbr_depths)
    for dj, Tj, mj in zip(nbr_depths, nbr_w2c, margins):
        Tj = torch.as_tensor(Tj, dtype=torch.float64, device=dev)
        Pj = Tj[:3, :3] @ Pw + Tj[:3, 3:4]
        z = Pj[2]
        zs = z.clamp(min=1e-9)
        uj, vj = Kt[0, 0] * Pj[0] / zs + Kt[0, 2], Kt[1, 1] * Pj[1] / zs + Kt[1, 2]
        djt = torch.as_tensor(dj, dtype=torch.float64, device=dev)
        gx = uj / max(W - 1, 1) * 2 - 1
        gy = vj / max(H - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
        F = torch.nn.functional
        samp = F.grid_sample(djt[None, None], grid, mode="bilinear", padding_mode="zeros",
                             align_corners=True).reshape(-1)
        vs = F.grid_sample((djt > 0).double()[None, None], grid, mode="bilinear",
                           padding_mode="zeros", align_corners=True).reshape(-1)
        ok = (di > 0) & (z > 1e-6) & (vs > FULL) & (samp > 0)
        rel = torch.abs(z - samp) / samp.clamp(min=1e-9)
        # the round trip: the neighbour's point back into the reference
        Qj = torch.stack([(uj - Kt[0, 2]) / Kt[0, 0] * samp, (vj - Kt[1, 2]) / Kt[1, 1] * samp, samp])
        Tj_inv = torch.linalg.inv(Tj)
        Qw = Tj_inv[:3, :3] @ Qj + Tj_inv[:3, 3:4]
        Qi = w2c_it[:3, :3] @ Qw + w2c_it[:3, 3:4]
        qz = Qi[2].clamp(min=1e-9)
        ui, vi = Kt[0, 0] * Qi[0] / qz + Kt[0, 2], Kt[1, 1] * Qi[1] / qz + Kt[1, 2]
        err = torch.sqrt((ui - u) ** 2 + (vi - v) ** 2)
        good = ok & (rel < tau_rel) & (err < tau_px) & (Qi[2] > 1e-6)
        # CONTRADICTION: the neighbour measured a surface FARTHER along the ray than
        # this point — it sees free space where the point claims to be (pccr
        # 2026-09-29: agreements alone let every frame-coherent Omega blob in)
        if mj is None:
            marg = tau_rel
        elif np.isscalar(mj):
            marg = float(mj)
        else:
            mt = torch.as_tensor(np.asarray(mj, np.float64), device=dev)
            marg = F.grid_sample(mt[None, None], grid, mode="nearest", padding_mode="border",
                                 align_corners=True).reshape(-1)
        bad = ok & (samp > z * (1.0 + marg))
        n_cons += good.int()
        n_bad += bad.int()
        rels.append(torch.where(good, rel, torch.full_like(rel, float("nan"))))
        if return_good:
            goods.append(good.reshape(H, W).cpu().numpy())
    if rels:
        res = _nanmedian0(torch.stack(rels))
    else:
        res = torch.full((H * W,), float("nan"), dtype=torch.float64, device=dev)
    out = (n_cons.clamp(max=255).to(torch.uint8).reshape(H, W).cpu().numpy(),
           res.reshape(H, W).cpu().numpy().astype(np.float32),
           n_bad.clamp(max=255).to(torch.uint8).reshape(H, W).cpu().numpy())
    return out + (goods,) if return_good else out


def pair_mismatch(depth_i: np.ndarray, K: np.ndarray, w2c_i: np.ndarray,
                  depth_j: np.ndarray, w2c_j: np.ndarray, device=None) -> Optional[float]:
    """Median |z_ij − d_j| / d_j over the pixels of i that land on measured pixels of
    j — the relative disagreement of two depth maps about the surfaces both see."""
    import torch
    dev = device or "cpu"
    H, W = depth_i.shape
    Kt = torch.as_tensor(K, dtype=torch.float64, device=dev)
    di = torch.as_tensor(depth_i, dtype=torch.float64, device=dev).reshape(-1)
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64),
                          torch.arange(W, device=dev, dtype=torch.float64), indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)
    M = torch.as_tensor(np.asarray(w2c_j) @ np.linalg.inv(w2c_i), dtype=torch.float64, device=dev)
    Pc = torch.stack([(u - Kt[0, 2]) / Kt[0, 0] * di, (v - Kt[1, 2]) / Kt[1, 1] * di, di])
    Pj = M[:3, :3] @ Pc + M[:3, 3:4]
    z = Pj[2].clamp(min=1e-9)
    uj, vj = Kt[0, 0] * Pj[0] / z + Kt[0, 2], Kt[1, 1] * Pj[1] / z + Kt[1, 2]
    iu, iv = torch.round(uj).long(), torch.round(vj).long()
    ok = (di > 0) & (Pj[2] > 1e-6) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    if not bool(ok.any()):
        return None
    djt = torch.as_tensor(depth_j, dtype=torch.float64, device=dev)
    d = djt[iv[ok], iu[ok]]
    m = d > 0
    if not bool(m.any()):
        return None
    return float(torch.median(torch.abs(Pj[2][ok][m] - d[m]) / d[m]))


def assign_tiers(sig: np.ndarray, n_s: np.ndarray, z0: np.ndarray, n_p: np.ndarray,
                 excl: Optional[np.ndarray], min_consistent_views: int,
                 prior_fill_min_views: int, prior_fill: str,
                 bad_s: Optional[np.ndarray] = None, bad_p: Optional[np.ndarray] = None,
                 low_conf: Optional[np.ndarray] = None,
                 independent: Optional[np.ndarray] = None) -> np.ndarray:
    """``source`` per pixel from the ZNCC signal, the sweep's and the prior's consistent
    view counts, their CONTRADICTION counts (views that see free space through the
    point), the prior, the confidence floor and the exclusion mask.

    A measured (tier 0) depth stands while the views confirming it outnumber the
    views contradicting it; a prior (tier 1) — Omega's own depth with no image
    evidence of its own — stands only when NO view contradicts it and Omega's
    confidence there is above the pipeline's floor (pccr 2026-09-29: 95 % of the
    fused cloud was tier 1 admitted on agreements alone → layers and floaters), and
    — when ``independent`` is given — at least one agreeing view stands a VISIT of
    walk away (pccr's door: 24 cm of layers, all confirmed by adjacent keyframes
    whose Omega depths share the same error)."""
    src = np.full(sig.shape, DISCARD_NO_SIGNAL, np.uint8)
    bs = np.zeros(sig.shape, np.uint8) if bad_s is None else bad_s
    bp = np.zeros(sig.shape, np.uint8) if bad_p is None else bad_p
    enough0 = sig & (n_s >= int(min_consistent_views))
    tier0 = enough0 & (bs.astype(np.int32) < n_s.astype(np.int32))
    enough1 = ~sig & (z0 > 0) & (n_p >= int(prior_fill_min_views))
    tier1 = enough1 & (bp == 0)
    src[sig & ~tier0] = DISCARD_INCONSISTENT
    src[enough0 & ~tier0] = DISCARD_CONTRADICTED
    src[tier1] = SOURCE_PRIOR_FILL if prior_fill == "keep" else DISCARD_PRIOR_FILL_DROPPED
    src[enough1 & ~tier1] = DISCARD_CONTRADICTED
    if low_conf is not None:
        src[tier1 & low_conf] = DISCARD_LOW_CONF
    if independent is not None:
        src[tier1 & ~low_conf_or_false(low_conf, sig.shape) & ~independent] = DISCARD_NOT_INDEPENDENT
    src[tier0] = SOURCE_SWEEP
    src[z0 <= 0] = DISCARD_NO_PRIOR
    if excl is not None:
        src[excl] = DISCARD_EXCLUDED
    return src


def low_conf_or_false(low_conf: Optional[np.ndarray], shape) -> np.ndarray:
    return np.zeros(shape, bool) if low_conf is None else np.asarray(low_conf, bool)


def conf_floor_mask(conf: np.ndarray, valid: np.ndarray, conf_min_norm: float) -> np.ndarray:
    """The pipeline's ONE confidence floor (reconstruction.simple.conf_min_norm, USER
    2026-09-23) on a keyframe's Omega confidence: the min-max fraction over the
    frame's valid pixels — the arithmetic of the viewer slider. True where the
    prior is UNDER the floor."""
    c = np.asarray(conf, np.float64)
    ok = np.asarray(valid, bool) & np.isfinite(c)
    if float(conf_min_norm) <= 0.0 or not ok.any():
        return np.zeros(c.shape, bool)
    lo, hi = float(c[ok].min()), float(c[ok].max())
    if hi <= lo:
        return np.zeros(c.shape, bool)
    return ok & ((c - lo) / (hi - lo) < float(conf_min_norm))


# ── view selection ───────────────────────────────────────────────────────

def select_views(X: np.ndarray, C_ref: np.ndarray, cand_w2c: np.ndarray, K: np.ndarray,
                 wh: Tuple[int, int], n_views: int, min_tri_deg: float,
                 max_tri_deg: float) -> Tuple[List[int], np.ndarray]:
    """Candidate indices ranked by covisibility of the reference's prior points
    ``X`` (world, M×3), among those whose median triangulation angle lies in
    [min_tri_deg, max_tri_deg]. Also returns every candidate's covisible share."""
    W, H = wh
    R = cand_w2c[:, :3, :3].astype(np.float32)
    t = cand_w2c[:, :3, 3].astype(np.float32)
    Xf = X.astype(np.float32)
    P = np.einsum("nij,mj->nmi", R, Xf) + t[:, None, :]                      # N,M,3
    z = P[..., 2]
    zs = np.where(z > 1e-6, z, 1.0)
    u = K[0, 0] * P[..., 0] / zs + K[0, 2]
    v = K[1, 1] * P[..., 1] / zs + K[1, 2]
    vis = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    covis = vis.mean(1)
    C = -np.einsum("nji,nj->ni", R, t)                                          # centres
    a = Xf[None] - C_ref[None, None].astype(np.float32)
    b = Xf[None] - C[:, None]
    cosang = np.sum(a * b, -1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1),
                                            1e-12)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a candidate that sees none
        med = np.nanmedian(np.where(vis, ang, np.nan), axis=1) if vis.any() \
            else np.full(len(P), np.nan)
    ok = (covis > 0) & (med >= min_tri_deg) & (med <= max_tri_deg)
    # STABLE: equal covisibility (common — it is a count over view_samples) keeps the
    # candidate order; numpy's default sort may order ties differently across builds / CPUs
    order = [int(i) for i in np.argsort(-covis, kind="stable") if ok[i]]
    return order[:int(n_views)], covis


# ── session inputs ───────────────────────────────────────────────────────

@dataclass
class SweepInputs:
    cam: Any
    K: np.ndarray
    wh: Tuple[int, int]
    maps: Tuple[np.ndarray, np.ndarray]
    kf: List[int]
    kf_w2c: np.ndarray
    wit: List[int]
    wit_w2c: np.ndarray
    tau_px: float
    heldout_rms_px: float
    epochs: Dict[str, int]
    frames_dir: Path
    records_dir: Path


def _read_poses(poses_txt: Path, frames_txt: Path) -> Tuple[List[int], np.ndarray]:
    poses = np.loadtxt(poses_txt).reshape(-1, 4, 4)
    frames = [int(float(x)) for x in frames_txt.read_text().split()]
    if len(frames) != len(poses):
        raise DepthSweepError(f"{poses_txt.name} ({len(poses)}) and {frames_txt.name} "
                              f"({len(frames)}) disagree")
    return frames, poses


def load_inputs(session_dir: Path, pcfg) -> SweepInputs:
    from intake.quality import read_session_epochs
    from precision.camera import load_camera_json, undistort_maps
    from precision.refine import REFINE_NAME, RESIDUALS_NAME, WITNESS_FRAMES_NAME, WITNESS_POSES_NAME
    session_dir = Path(session_dir)
    out = session_dir / "output"
    pdir = out / "precision"
    rj = pdir / REFINE_NAME
    if not rj.exists():
        raise DepthSweepError(f"{rj} is missing — F6 sweeps with F5's camera and poses "
                              f"(python -m precision.refine --session <dir>)")
    ref = json.loads(rj.read_text())
    epochs = read_session_epochs(session_dir)
    if not ref.get("applied"):
        raise DepthSweepError("F5 ran with --no-apply: its witness poses live in the refined "
                              "world, the keyframes do not — apply F5 first (it publishes a "
                              "selectable epoch; epoch 0 stays intact)")
    if ref.get("epoch_to") != epochs["geometry_epoch"]:
        # a NEW-CLOUD epoch (F7's fused cloud) moves no camera: F5's poses and witness
        # poses still hold across it, so F6/F7 can be re-run after F7 published
        from correction.epoch import epoch_kind
        later = list(range(int(ref.get("epoch_to")) + 1, int(epochs["geometry_epoch"]) + 1))
        kinds = {e: epoch_kind(out, e) for e in later} if later else {}
        if not later or any(k != "new_cloud" for k in kinds.values()):
            raise DepthSweepError(f"F5 published epoch {ref.get('epoch_to')}, the session is at "
                                  f"{epochs['geometry_epoch']} and the epochs in between are "
                                  f"{kinds or 'none'} — the witness poses belong to F5's epoch; "
                                  f"re-run F5 on the current one")
    cam = load_camera_json(out / "camera.json")
    kf, kf_c2w = _read_poses(out / "camera_poses.txt", out / "camera_frames.txt")
    wp, wf = pdir / WITNESS_POSES_NAME, pdir / WITNESS_FRAMES_NAME
    if wp.exists() and wf.exists() and wp.read_text().strip():
        wit, wit_c2w = _read_poses(wp, wf)
    else:
        wit, wit_c2w = [], np.zeros((0, 4, 4))
    with np.load(pdir / RESIDUALS_NAME) as z:
        h = z["heldout_rms_px"].astype(np.float64)
    if h.size == 0:
        raise DepthSweepError(f"{pdir / RESIDUALS_NAME} holds no held-out residual — τ_px "
                              f"cannot be measured")
    # the RMS over the tracks F5 could reproject: a DEGENERATE track (non-finite, or
    # farther than the image diagonal) is a landmark the solve sent to infinity, not a
    # residual — pccr 2026-09-29: 2.3 % of them (up to 90,850 px) made the RMS 259 px
    # and τ_px 518 px on a 464-px-wide image, so the round-trip test never rejected
    diag = float(np.hypot(cam.width, cam.height))
    good = np.isfinite(h) & (h >= 0.0) & (h <= diag)
    if not good.any():
        raise DepthSweepError("every held-out residual of F5 is degenerate — τ_px cannot be "
                              "measured")
    rms = float(np.sqrt(np.mean(h[good] ** 2)))
    m1, m2, K = undistort_maps(cam)
    return SweepInputs(cam=cam, K=K, wh=(cam.width, cam.height), maps=(m1, m2), kf=kf,
                       kf_w2c=np.linalg.inv(kf_c2w), wit=wit,
                       wit_w2c=np.linalg.inv(wit_c2w) if len(wit) else np.zeros((0, 4, 4)),
                       tau_px=float(pcfg.depth.tau_px_k) * rms, heldout_rms_px=rms,
                       epochs=epochs, frames_dir=session_dir / "frames",
                       records_dir=out / "omega_run" / "results_output")


def read_gray_undistorted(frames_dir: Path, frame: int, maps) -> Tuple[np.ndarray, np.ndarray]:
    """(gray in [0, 1], valid) of one frame in the undistorted native frame."""
    import cv2
    from intake.content import frame_file
    p = frame_file(frames_dir, frame)
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise DepthSweepError(f"cannot read {p}")
    g = g.astype(np.float32) / GRAY_MAX
    und = cv2.remap(g, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    val = cv2.remap(np.ones_like(g), maps[0], maps[1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > FULL
    return und, val


def read_exclusion(session_dir: Path, frame: int, maps) -> Optional[np.ndarray]:
    import cv2
    from intake.content import EXCLUSION_MASKS_DIRNAME, read_mask_png
    p = Path(session_dir) / "intake" / EXCLUSION_MASKS_DIRNAME / f"{frame:06d}.png"
    if not p.exists():
        return None
    m = read_mask_png(p).astype(np.float32)
    return cv2.remap(m, maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0) > 0


def record_grid(cam, shape_hw: Tuple[int, int]):
    from precision.camera import grid_like
    h, w = shape_hw
    g = cam.omega_grid
    return g if (g.w, g.h) == (w, h) else grid_like(g, w, h, "omega_npz")


def prior_native(rec_depth: np.ndarray, rec_conf: np.ndarray, s_k: float, cam, maps,
                 guide_gray_distorted: np.ndarray, device=None) -> Tuple[np.ndarray, np.ndarray]:
    """(z0, conf0) of one keyframe in the undistorted native frame: Omega's depth ×
    s_k through ``guided_upsample`` on its grid, then the undistortion map."""
    import cv2
    from reconstruction.native_depth import guided_upsample
    g = record_grid(cam, rec_depth.shape)
    h, w = rec_depth.shape
    f = int(math.ceil(max(g.scale_x, g.scale_y)))
    crop = guide_gray_distorted[g.crop_y:g.crop_y + g.crop_h, g.crop_x:g.crop_x + g.crop_w]
    guide_hi = cv2.resize(crop, (w * f, h * f), interpolation=cv2.INTER_AREA) * GRAY_MAX
    z = np.where(np.isfinite(rec_depth) & (rec_depth > 0), rec_depth * float(s_k), 0.0)
    z_hi, v_hi = guided_upsample(z.astype(np.float32), z > 0, guide_hi, f, device=device)
    gx = ((maps[0] - g.crop_x + 0.5) / g.scale_x - 0.5).astype(np.float32)
    gy = ((maps[1] - g.crop_y + 0.5) / g.scale_y - 0.5).astype(np.float32)
    hx, hy = (gx + 0.5) * f - 0.5, (gy + 0.5) * f - 0.5
    z0 = cv2.remap(np.where(v_hi, z_hi, 0).astype(np.float32), hx, hy, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    c0 = cv2.remap(np.asarray(rec_conf, np.float32), gx, gy, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return z0, c0


# ── per-keyframe scale and the landmark calibration ──────────────────────

def triangulated_landmarks(session_dir: Path, pcfg, inp: SweepInputs):
    """F4's tracks triangulated with F5's camera and poses: ({track: X}, {track: [(kf index,
    native px)]})."""
    from precision.camera import undistort_solver
    from precision.refine import group_tracks, triangulate_tracks
    from precision.tracks import load_tracks_v2
    tr = load_tracks_v2(session_dir)
    groups = group_tracks(tr["obs_track"], tr["obs_frame"], tr["obs_uv_native"], inp.kf)
    X = triangulate_tracks(groups, inp.kf_w2c, inp.cam.params, undistort_solver(pcfg.camera),
                           pcfg.refine.min_tri_deg)
    return X, groups


def landmark_samples(session_dir: Path, pcfg, inp: SweepInputs) -> Dict[int, Dict[str, np.ndarray]]:
    """Per keyframe index: z_tri (depth of F4's tracks triangulated with F5's camera and
    poses), Omega's record depth and confidence at the observing pixel."""
    from precision.camera import native_to_grid
    X, groups = triangulated_landmarks(session_dir, pcfg, inp)
    by_kf: Dict[int, List[Tuple[float, Tuple[float, float]]]] = {}
    for t, P in X.items():
        for i, p in groups[t]:
            z = float((inp.kf_w2c[i, :3, :3] @ P + inp.kf_w2c[i, :3, 3])[2])
            if z > 0:
                by_kf.setdefault(i, []).append((z, (float(p[0]), float(p[1]))))
    out = {}
    for i, rows in by_kf.items():
        npz = inp.records_dir / f"frame_{inp.kf[i]}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as zf:
            d = zf["depth"]
            c = zf["conf"] if "conf" in zf.files else np.full(d.shape, np.nan, np.float32)
        g = record_grid(inp.cam, d.shape)
        uv = native_to_grid(np.array([p for _, p in rows]), g)
        iu, iv = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (iu >= 0) & (iu < d.shape[1]) & (iv >= 0) & (iv < d.shape[0])
        zt = np.array([z for z, _ in rows])[ok]
        zr = d[iv[ok], iu[ok]].astype(np.float64)
        cr = c[iv[ok], iu[ok]].astype(np.float64)
        m = np.isfinite(zr) & (zr > 0)
        out[i] = {"z_tri": zt[m], "z_rec": zr[m], "conf": cr[m]}
    return out


def keyframe_scales(samples: Dict[int, Dict[str, np.ndarray]], n_kf: int,
                    min_samples: int, chainage: Optional[np.ndarray] = None,
                    window_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """(s_k, n used) per keyframe: the median of z_tri / z_rec over the ratios of every
    keyframe within ``window_m / 2`` of WALK (``chainage`` per keyframe, metres) — the
    drift of Omega's depth scale is smooth along the walk, so the estimate is pooled
    along it. pccr 2026-09-29: independent per-keyframe medians jumped up to 27 %
    between consecutive keyframes (38 of 288 pairs > 5 %) while the gauge's model
    moves 0.25 %; each keyframe at its own scale layered the cloud. Without
    chainage (or window 0) the pool is the nearest keyframes by index until
    ``min_samples``. NaN where the session has fewer than ``min_samples``."""
    ratios = {i: s["z_tri"] / s["z_rec"] for i, s in samples.items() if s["z_rec"].size}
    s = np.full(n_kf, np.nan)
    n = np.zeros(n_kf, np.int64)
    total = sum(r.size for r in ratios.values())
    if total < min_samples:
        return s, n
    half = float(window_m) / 2.0
    use_walk = chainage is not None and half > 0

    def in_window(j, k):
        return (use_walk and np.isfinite(chainage[j]) and np.isfinite(chainage[k])
                and abs(chainage[j] - chainage[k]) <= half)

    for k in range(n_kf):
        pool: List[np.ndarray] = [ratios[j] for j in ratios if in_window(j, k)]
        r = 0
        while sum(p.size for p in pool) < min_samples and r <= n_kf:
            for j in ({k - r, k + r} if r else {k}):
                if j in ratios and not in_window(j, k):
                    pool.append(ratios[j])
            r += 1
        v = np.concatenate(pool)
        s[k], n[k] = float(np.median(v)), int(v.size)
    return s, n


def keyframe_chainage(session_dir: Path, kf: Sequence[int]) -> Optional[np.ndarray]:
    """Chainage (metres of walk) per keyframe from intake/walk.json (I3), NaN where a
    keyframe has none; None when the walk was never measured."""
    from intake.walk import load_walk
    walk = load_walk(Path(session_dir))
    if not walk or not walk.get("chainage"):
        return None
    c = {int(r["frame"]): float(r["chainage_m"]) for r in walk["chainage"]}
    return np.array([c.get(int(f), np.nan) for f in kf], np.float64)


# ── the run ──────────────────────────────────────────────────────────────

def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _hb(cfg_runner, t_last: float, msg: str, log: Callable) -> float:
    now = time.time()
    if now - t_last >= float(cfg_runner.heartbeat_s):
        log(msg)
        return now
    return t_last


def run_sweep(session_dir: Path, pcfg, log: Callable = print, device=None) -> Dict[str, Any]:
    """The F6 run under deterministic torch (see the module docstring)."""
    from precision.tracks import deterministic_torch
    with deterministic_torch(pcfg.depth.seed):
        return _run_sweep(session_dir, pcfg, log=log, device=device)


def _run_sweep(session_dir: Path, pcfg, log: Callable = print, device=None) -> Dict[str, Any]:
    from precision import confidence as CAL
    from intake.content import frame_file
    import cv2
    session_dir = Path(session_dir)
    out = session_dir / "output"
    dcfg = pcfg.depth
    dev = device or _device()
    t_start = time.time()
    inp = load_inputs(session_dir, pcfg)
    n_kf = len(inp.kf)
    ddir = out / DEPTH_DIRNAME
    work = ddir / WORK_DIRNAME
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    rng = np.random.default_rng(int(dcfg.seed))

    # s_k and β: measured before any pixel is swept
    lm = landmark_samples(session_dir, pcfg, inp)
    # the scale of Omega's depth per keyframe, pooled along the WALK over the gauge's
    # knot spacing (the drift is smooth; an independent median per keyframe is not)
    chain = keyframe_chainage(session_dir, inp.kf)
    s_k_raw, _ = keyframe_scales(lm, n_kf, int(dcfg.min_scale_samples))
    s_k, s_n = keyframe_scales(lm, n_kf, int(dcfg.min_scale_samples), chainage=chain,
                               window_m=float(pcfg.gauge.knot_walk_m))
    if chain is None:
        log(f"{LOG_TAG} intake/walk.json absent — s_k pooled by keyframe index, not along "
            f"the walk")
    else:
        jumps = np.abs(np.diff(s_k)) / np.maximum(s_k[:-1], 1e-9)
        jraw = np.abs(np.diff(s_k_raw)) / np.maximum(s_k_raw[:-1], 1e-9)
        log(f"{LOG_TAG} s_k pooled over ±{pcfg.gauge.knot_walk_m / 2:g} m of walk: neighbour "
            f"jumps median {np.nanmedian(jumps) * 100:.2f} % max {np.nanmax(jumps) * 100:.1f} % "
            f"(independent medians: {np.nanmedian(jraw) * 100:.2f} % / {np.nanmax(jraw) * 100:.1f} %)")
    if not np.isfinite(s_k).any():
        raise DepthSweepError(f"fewer than {dcfg.min_scale_samples} track depths in the whole "
                              f"session — Omega's prior cannot be carried to this epoch")
    cal_params = {"conf_bins": dcfg.calib_conf_bins, "dist_bins": dcfg.calib_dist_bins,
                  "quantile": dcfg.beta_quantile, "min_bin_samples": dcfg.calib_min_bin_samples}
    prev = CAL.load_calibration(out, inp.epochs, reference="tier0")
    if prev is not None:
        beta_table, beta_source = prev["models"]["omega"], "tier0"
    else:
        e, c, d = [], [], []
        for i, smp in lm.items():
            if np.isfinite(s_k[i]):
                e.append(smp["z_tri"] / (smp["z_rec"] * s_k[i]) - 1.0)
                c.append(smp["conf"])
                d.append(smp["z_tri"])
        beta_table = CAL.calibrate(np.concatenate(e), np.concatenate(c), np.concatenate(d),
                                   **cal_params)
        beta_source = "landmarks"
    log(f"{LOG_TAG} {n_kf} keyframes, {len(inp.wit)} localized witnesses, {dev}; s_k "
        f"{np.nanmin(s_k):.4f}–{np.nanmax(s_k):.4f}; β from {beta_source} "
        f"(global q{dcfg.beta_quantile:g} {beta_table['global_abs_err_quantile'] * 100:.2f} %, "
        f"cap {dcfg.beta_max:g}); τ_px {inp.tau_px:.2f} px")

    # priors at native + view selection
    all_w2c = np.concatenate([inp.kf_w2c, inp.wit_w2c], 0)
    photo_views: Dict[int, List[int]] = {}
    cons_views: Dict[int, List[int]] = {}
    contra_views: Dict[int, List[int]] = {}      # EVERY keyframe whose frustum sees the prior
    no_cover: Dict[int, List[int]] = {}
    has_prior = np.zeros(n_kf, bool)
    t_hb = time.time()
    for i, f in enumerate(inp.kf):
        rec = inp.records_dir / f"frame_{f}.npz"
        if not rec.exists() or not np.isfinite(s_k[i]):
            continue
        with np.load(rec) as zf:
            rd = zf["depth"]
            rc = zf["conf"] if "conf" in zf.files else np.full(rd.shape, np.nan, np.float32)
        gd = cv2.imread(str(frame_file(inp.frames_dir, f)), cv2.IMREAD_GRAYSCALE).astype(np.float32) / GRAY_MAX
        z0, c0 = prior_native(rd, rc, s_k[i], inp.cam, inp.maps, gd, device=dev)
        np.save(work / f"prior_{f}.npy", z0)
        np.save(work / f"pconf_{f}.npy", c0)
        has_prior[i] = bool((z0 > 0).any())
        vv, uu = np.nonzero(z0 > 0)
        if vv.size == 0:
            continue
        pick = rng.choice(vv.size, size=min(int(dcfg.view_samples), vv.size), replace=False)
        z = z0[vv[pick], uu[pick]].astype(np.float64)
        Pc = np.stack([(uu[pick] - inp.K[0, 2]) / inp.K[0, 0] * z,
                       (vv[pick] - inp.K[1, 2]) / inp.K[1, 1] * z, z], 1)
        c2w = np.linalg.inv(inp.kf_w2c[i])
        Xw = Pc @ c2w[:3, :3].T + c2w[:3, 3]
        others = np.array([j for j in range(len(all_w2c)) if j != i])
        order, covis = select_views(Xw, c2w[:3, 3], all_w2c[others], inp.K, inp.wh,
                                    dcfg.n_views, dcfg.min_tri_deg, dcfg.max_tri_deg)
        photo_views[i] = [int(others[o]) for o in order]
        kf_others = others[others < n_kf]
        order_k, covis_k = select_views(Xw, c2w[:3, 3], all_w2c[kf_others], inp.K, inp.wh,
                                        dcfg.n_views, dcfg.min_tri_deg, dcfg.max_tri_deg)
        cons_views[i] = [int(kf_others[o]) for o in order_k]
        # the CONTRADICTION test asks every view that sees the point, whatever its
        # angle: a floater is seen through by the far side of the room, not by the
        # eight nearest keyframes (USER 2026-09-29: "ampliar las vistas que ven a
        # los puntos")
        seen = np.flatnonzero(covis_k > 0)
        contra_views[i] = [int(kf_others[j]) for j in seen[np.argsort(-covis_k[seen], kind="stable")]]
        no_cover[i] = [int(others[j]) for j in np.flatnonzero(covis == 0) if others[j] < n_kf]
        t_hb = _hb(pcfg.runner, t_hb, f"{LOG_TAG} priors + views: {i + 1}/{n_kf}", log)

    frame_of = lambda j: inp.kf[j] if j < n_kf else inp.wit[j - n_kf]        # noqa: E731
    img_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def image(j):
        if j not in img_cache:
            if len(img_cache) > 4 * int(dcfg.n_views):
                img_cache.pop(next(iter(img_cache)))
            g, val = read_gray_undistorted(inp.frames_dir, frame_of(j), inp.maps)
            ex = read_exclusion(session_dir, frame_of(j), inp.maps)
            img_cache[j] = (g, val & ~ex if ex is not None else val)
        return img_cache[j]

    def beta_map(i):
        z0 = np.load(work / f"prior_{inp.kf[i]}.npy")
        c0 = np.load(work / f"pconf_{inp.kf[i]}.npy")
        b = np.minimum(CAL.lookup(beta_table, np.nan_to_num(c0, nan=np.nanmedian(c0)),
                                  np.where(z0 > 0, z0, 1.0)), float(dcfg.beta_max))
        return z0, b

    def run_one(i, view_idx, image_of):
        ref, rval = image(i)
        z0, b = beta_map(i)
        z0 = np.where(rval, z0, 0)
        imgs = [image_of(j) for j in view_idx]
        fr = _Frame(ref, inp.K, inp.kf_w2c[i], [m[0] for m in imgs], [m[1] for m in imgs],
                    [all_w2c[j] for j in view_idx], dcfg.patch_px, dcfg.best_k, dev)
        return sweep_frame(fr, z0, b, dcfg.n_hyp, dcfg.propagation_iters,
                           contrast_sigma(ref, rval))

    # the noise floor: the same pipeline, every view's image from a keyframe that
    # shares no surface with the reference
    swept = [i for i in range(n_kf) if photo_views.get(i)]
    if not swept:
        raise DepthSweepError("no keyframe has a prior and a view within the triangulation "
                              "range — nothing to sweep")
    null_idx = [swept[int(round(x))] for x in np.linspace(0, len(swept) - 1,
                                                           min(int(dcfg.null_frames), len(swept)))]
    ns, nd = [], []
    for i in null_idx:
        pool = no_cover.get(i) or []
        if not pool:
            continue
        sub = rng.choice(pool, size=len(photo_views[i]), replace=len(pool) < len(photo_views[i]))
        swap = dict(zip(photo_views[i], [int(x) for x in sub]))
        r = run_one(i, photo_views[i], lambda j: image(swap[j]))
        m = r["depth"] > 0
        ns.append(r["score"][m])
        nd.append(r["ref_std"][m])
    if not ns:
        raise DepthSweepError("no sampled keyframe has a keyframe that shares no surface with "
                              "it — the ZNCC null distribution cannot be measured")
    floor = floor_table(np.concatenate(ns), np.concatenate(nd), dcfg.null_confidence,
                        dcfg.null_texture_bins)
    log(f"{LOG_TAG} ZNCC noise floor over {len(ns)} keyframe(s): q{dcfg.null_confidence:g} "
        f"{floor['global_floor']:.3f} (per texture bin {min(floor['floor']):.3f}–"
        f"{max(floor['floor']):.3f})")

    # pass 1: sweep every keyframe
    t_hb, t1 = time.time(), time.time()
    boundary_frac = {}
    for n_done, i in enumerate(swept):
        r = run_one(i, photo_views[i], image)
        sig = (r["score"] > floor_of(floor, r["ref_std"])) & (r["depth"] > 0)
        np.save(work / f"sweep_{inp.kf[i]}.npy", r["depth"])
        np.save(work / f"score_{inp.kf[i]}.npy", r["score"])
        np.save(work / f"signal_{inp.kf[i]}.npy", sig)
        boundary_frac[i] = float(r["boundary"][r["depth"] > 0].mean()) if (r["depth"] > 0).any() else 0.0
        rate = (n_done + 1) / max(time.time() - t1, 1e-9)
        t_hb = _hb(pcfg.runner, t_hb, f"{LOG_TAG} sweep {n_done + 1}/{len(swept)} "
                   f"({rate:.2f} kf/s, ETA {(len(swept) - n_done - 1) / rate / 60:.1f} min)", log)

    # τ_rel: the neighbouring priors' disagreement on the surfaces they share
    mm = []
    for i in swept:
        zi = np.load(work / f"prior_{inp.kf[i]}.npy")
        for j in cons_views.get(i, []):
            pj = work / f"prior_{inp.kf[j]}.npy"
            if pj.exists():
                v = pair_mismatch(zi, inp.K, inp.kf_w2c[i], np.load(pj), inp.kf_w2c[j], device=dev)
                if v is not None:
                    mm.append(v)
    if not mm:
        raise DepthSweepError("no pair of neighbouring priors shares a surface — τ_rel cannot "
                              "be measured")
    tau_rel = float(np.median(mm))
    log(f"{LOG_TAG} τ_rel {tau_rel * 100:.2f} % (median prior disagreement over {len(mm)} "
        f"pair(s)), τ_px {inp.tau_px:.2f} px")

    # pass 2: consistency, tiers, outputs
    from config import cfg as _raw_cfg
    try:
        conf_floor = float(_raw_cfg["reconstruction"]["simple"]["conf_min_norm"])
    except (KeyError, TypeError, ValueError) as e:
        raise DepthSweepError("reconstruction.simple.conf_min_norm is missing — the ONE "
                              "confidence floor of the pipeline (USER 2026-09-23) gates the "
                              "tier-1 prior too") from e
    log(f"{LOG_TAG} tier-1 prior gated by the pipeline's confidence floor "
        f"(reconstruction.simple.conf_min_norm {conf_floor:g}, min-max per keyframe)")
    # a tier-1 prior needs a witness from ANOTHER VISIT: the system's one definition of
    # a visit (correction.visit_drift.min_walk_m) — adjacent keyframes share Omega's error
    try:
        min_walk = float(_raw_cfg["correction"]["visit_drift"]["min_walk_m"])
    except (KeyError, TypeError, ValueError) as e:
        raise DepthSweepError("correction.visit_drift.min_walk_m is missing — the tier-1 "
                              "independence rule reads the one definition of a visit") from e
    if min_walk > 0 and chain is None:
        raise DepthSweepError("intake/walk.json is missing — the independence of a tier-1 "
                              "witness is measured in metres of walk")
    log(f"{LOG_TAG} tier-1 prior needs an agreeing view ≥ {min_walk:g} m of walk away "
        f"(correction.visit_drift.min_walk_m)")
    # every keyframe's depth for the contradiction test: the swept depth where it has
    # signal, Omega's prior elsewhere — loaded once (≈ 1.5 MB per keyframe)
    best_depth: Dict[int, np.ndarray] = {}
    best_margin: Dict[int, np.ndarray] = {}
    for j in swept:
        fj = inp.kf[j]
        pj = np.load(work / f"prior_{fj}.npy")
        sj = np.load(work / f"sweep_{fj}.npy")
        gj = np.load(work / f"signal_{fj}.npy")
        meas = gj & (sj > 0)
        best_depth[j] = np.where(meas, sj, pj).astype(np.float32)
        # the VIEW's own depth error decides when its farther surface contradicts:
        # a measured depth → τ_rel; Omega's prior → its calibrated |error| quantile β
        # (pccr 2026-09-29: judged at τ_rel 1.08 % against priors 5-27 % off, ~100
        # views contradicted almost every point — 220 k points left of 9 M)
        _, bj = beta_map(j)
        best_margin[j] = np.where(meas, tau_rel, bj).astype(np.float32)
    n_far_only = 0
    n_contra_views_tot = 0
    counts_tot = {k: 0 for k in SOURCE_NAMES.values()}
    per_frame = {}
    cal_omega, cal_da3 = [], []
    da3_dir = out / "da3_run" / "results_output"
    t_hb = time.time()
    for n_done, i in enumerate(swept):
        f = inp.kf[i]
        d = np.load(work / f"sweep_{f}.npy")
        sc = np.load(work / f"score_{f}.npy")
        sig = np.load(work / f"signal_{f}.npy")
        z0 = np.load(work / f"prior_{f}.npy")
        c0 = np.load(work / f"pconf_{f}.npy")
        nb = [j for j in cons_views.get(i, []) if j in best_depth]
        nd_ = [np.where(np.load(work / f"signal_{inp.kf[j]}.npy"),
                        np.load(work / f"sweep_{inp.kf[j]}.npy"), 0) for j in nb]
        np_ = [np.load(work / f"prior_{inp.kf[j]}.npy") for j in nb]
        nw = [inp.kf_w2c[j] for j in nb]
        # AGREEMENT over the nearest views (the tier thresholds keep their meaning)
        n_s, r_s, b_s_near, good_s = consistency(np.where(sig, d, 0), inp.K, inp.kf_w2c[i], nd_,
                                                 nw, tau_rel, inp.tau_px, device=dev,
                                                 return_good=True)
        n_p, r_p, b_p_near, good_p = consistency(z0, inp.K, inp.kf_w2c[i], np_, nw, tau_rel,
                                                 inp.tau_px, device=dev, return_good=True)
        # CONTRADICTION: the most covisible views that see the point (depth.
        # contradiction_views) VOTE with their best depth and their own error margin;
        # a point is contradicted when more of them see through it than confirm it
        ca = [j for j in contra_views.get(i, []) if j in best_depth][:int(dcfg.contradiction_views)]
        n_contra_views_tot += len(ca)
        if ca:
            cd_ = [best_depth[j] for j in ca]
            cw_ = [inp.kf_w2c[j] for j in ca]
            cm_ = [best_margin[j] for j in ca]
            g_s_all, _, b_s_all = consistency(np.where(sig, d, 0), inp.K, inp.kf_w2c[i], cd_, cw_,
                                              tau_rel, inp.tau_px, device=dev, nbr_bad_margin=cm_)
            g_p_all, _, b_p_all = consistency(z0, inp.K, inp.kf_w2c[i], cd_, cw_, tau_rel,
                                              inp.tau_px, device=dev, nbr_bad_margin=cm_)
            # a contradiction counts only where it outvotes the confirmations of the same views
            b_s = np.where(b_s_all.astype(np.int32) > g_s_all.astype(np.int32), b_s_all, 0).astype(np.uint8)
            b_p = np.where(b_p_all.astype(np.int32) > g_p_all.astype(np.int32), b_p_all, 0).astype(np.uint8)
        else:
            b_s, b_p = b_s_near, b_p_near
        # INDEPENDENCE: an agreeing view a visit of walk away from this keyframe
        indep = None
        if min_walk > 0:
            indep = np.zeros(z0.shape, bool)
            for j, g in zip(nb, good_p):
                if np.isfinite(chain[j]) and np.isfinite(chain[i]) and \
                        abs(chain[j] - chain[i]) >= min_walk:
                    indep |= g
        excl = read_exclusion(session_dir, f, inp.maps)
        low = conf_floor_mask(c0, z0 > 0, conf_floor)
        src = assign_tiers(sig, n_s, z0, n_p, excl, dcfg.min_consistent_views,
                           dcfg.prior_fill_min_views, dcfg.prior_fill,
                           bad_s=b_s, bad_p=b_p, low_conf=low, independent=indep)
        # what the far views alone caught: rejected now, not by the nearest ones
        near_src = assign_tiers(sig, n_s, z0, n_p, excl, dcfg.min_consistent_views,
                                dcfg.prior_fill_min_views, dcfg.prior_fill,
                                bad_s=b_s_near, bad_p=b_p_near, low_conf=low, independent=indep)
        n_far_only += int(((src == DISCARD_CONTRADICTED) & (near_src != DISCARD_CONTRADICTED)).sum())
        depth = np.where(src == SOURCE_SWEEP, d, np.where(src == SOURCE_PRIOR_FILL, z0, 0))
        ncons = np.where(src == SOURCE_PRIOR_FILL, n_p, n_s).astype(np.uint8)
        nbad = np.where(src == SOURCE_PRIOR_FILL, b_p, b_s).astype(np.uint8)
        res = np.where(src == SOURCE_PRIOR_FILL, r_p, r_s)
        nrm = normals_from_depth(depth, inp.K)
        # float32 everywhere: a precision product is not stored at half precision
        # (float16 kept 3 significant digits of a normal and of the residual)
        np.savez_compressed(ddir / f"frame_{f}.npz", depth=depth.astype(np.float32),
                            ncc=np.where(z0 > 0, sc, np.nan).astype(np.float32),
                            n_consistent=ncons, n_contradict=nbad, source=src,
                            normal=nrm.astype(np.float32), residual_rel=res.astype(np.float32))
        cnt = {SOURCE_NAMES[k]: int((src == k).sum()) for k in SOURCE_NAMES}
        for k, v in cnt.items():
            counts_tot[k] += v
        t0m = src == SOURCE_SWEEP
        per_frame[str(f)] = {
            **cnt, "n_photometric_views": len(photo_views[i]),
            "n_witness_views": int(sum(j >= n_kf for j in photo_views[i])),
            "n_consistency_views": len(nb),
            "photometric_views": [frame_of(j) for j in photo_views[i]],
            "consistency_views": [inp.kf[j] for j in nb],
            "zncc_median_tier0": float(np.median(sc[t0m])) if t0m.any() else None,
            "prior_vs_tier0_median_rel": (float(np.median(np.abs(z0[t0m] / d[t0m] - 1)))
                                          if t0m.any() else None),
            "boundary_hypothesis_frac": boundary_frac.get(i),
            "n_contradiction_views": len(ca),
            "s_k": float(s_k[i]), "s_k_samples": int(s_n[i]),
            "s_k_independent": float(s_k_raw[i])}
        cal_omega.append(CAL.sample_pairs(d, z0, c0, t0m, dcfg.calib_samples_per_frame, rng))
        da3 = da3_dir / f"frame_{f}.npz"
        if da3.exists() and t0m.any():
            with np.load(da3) as zf:
                dd, dc = zf["depth"], zf["conf"]
            from precision.camera import grid_full_frame_resize
            g = grid_full_frame_resize(inp.wh[0], inp.wh[1], dd.shape[1], dd.shape[0], "da3")
            gx = ((inp.maps[0] - g.crop_x + 0.5) / g.scale_x - 0.5).astype(np.float32)
            gy = ((inp.maps[1] - g.crop_y + 0.5) / g.scale_y - 0.5).astype(np.float32)
            dn = cv2.remap(dd.astype(np.float32), gx, gy, cv2.INTER_NEAREST, borderValue=0)
            # extract_da3_depth stores DA3's confidence already as expp1 − 1
            cn = cv2.remap(np.asarray(dc, np.float32), gx, gy, cv2.INTER_NEAREST,
                           borderValue=np.nan)
            cal_da3.append(CAL.sample_pairs(d, dn, cn, t0m, dcfg.calib_samples_per_frame, rng))
        t_hb = _hb(pcfg.runner, t_hb, f"{LOG_TAG} consistency {n_done + 1}/{len(swept)}", log)

    tables = {}
    for name, smp in (("omega", CAL.concat(cal_omega)), ("da3", CAL.concat(cal_da3))):
        try:
            tables[name] = CAL.calibrate(smp["err_rel"], smp["conf"], smp["dist"], **cal_params)
        except CAL.CalibrationError as e:
            tables[name] = {"unmeasured": str(e)}
    cal_path = CAL.write_calibration(out, "tier0", tables, inp.epochs, cal_params)

    total = sum(counts_tot.values())
    probe = out / "omega_probe.json"
    doc = {"version": REPORT_VERSION, "provenance": PROVENANCE, **inp.epochs,
           "params": asdict(dcfg),
           "device": str(dev), "seconds": round(time.time() - t_start, 1),
           "n_keyframes": n_kf, "n_swept": len(swept),
           "no_prior_keyframes": [inp.kf[i] for i in range(n_kf) if not has_prior[i]],
           "n_witness_views_available": len(inp.wit),
           "beta_source": beta_source, "zncc_floor": floor,
           "tau_rel": tau_rel, "tau_rel_pairs": len(mm),
           "tau_rel_f3_probe": (json.loads(probe.read_text()).get("results") if probe.exists() else None),
           "tau_px": inp.tau_px, "heldout_rms_px": inp.heldout_rms_px,
           "s_k": {"min": float(np.nanmin(s_k)), "max": float(np.nanmax(s_k)),
                   "median": float(np.nanmedian(s_k))},
           "counts": counts_tot,
           "percent": {k: (100.0 * v / total if total else 0.0) for k, v in counts_tot.items()},
           "source_codes": {str(k): v for k, v in SOURCE_NAMES.items()},
           "calibration": str(cal_path.relative_to(out)),
           "per_frame": per_frame, "colmap_ab": None}
    (ddir / REPORT_NAME).write_text(json.dumps(doc, indent=1, default=float))
    shutil.rmtree(work, ignore_errors=True)
    log(f"{LOG_TAG} tier 0 {doc['percent']['tier0']:.1f} %, tier 1 "
        f"{doc['percent']['tier1_prior_fill']:.1f} %, discarded "
        f"{100 - doc['percent']['tier0'] - doc['percent']['tier1_prior_fill']:.1f} % in "
        f"{doc['seconds'] / 60:.1f} min → {ddir / REPORT_NAME}")
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.depth_sweep",
                                 description="Prior-guided native-resolution plane sweep (F6).")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    from precision.tracks import ensure_cublas_workspace
    ensure_cublas_workspace()                   # before torch initialises CUDA
    run_sweep(Path(args.session), load_precision_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
