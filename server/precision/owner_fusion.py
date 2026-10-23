"""Owner fusion — ONE point per surface element (audit 2026-10-23; USER: "es el más importante y
debe necesariamente corregirse … el costo del tiempo en la reconstrucción debe arrojar como
resultado una nube perfecta").

THE DEFECT. The depth on F5 published every voted pixel of every keyframe. A surface element seen
by K keyframes became K points, each on its own ray with its own keyframe's residual depth error
(after the bend the session's held-out is 1-3 % relative; the vote's tau — p75 of the neighbour
disagreement — ~2 % on pccr). The cleaner's 5 mm voxel pick keeps every layer farther apart than
5 mm, so a wall at 3 m came out as a sheaf of layers up to 6 cm thick: the crumpled surfaces, the
doubled walls, the bent edges — and no evidence rule could act on a point that had K−1 near-copies.

THE ESTIMATOR. An observation is (keyframe f, pixel p): the ray r_f(p) from the camera centre C_f
and the depth z_f(p), with a relative uncertainty of the order of tau. Two observations see the
SAME surface element when one's point, carried into the other's camera, agrees with the other's
depth at the landing point within tau — the vote's own agreement test, in the other camera's own
frame; the depth at the landing point is read at the exact sub-pixel position when the 2x2 patch
around it is one surface (its four depths within tau), at the nearest pixel otherwise (a patch that
spans a step is never interpolated across it). For every observation its MEMBERS are the agreeing
observations among the COVISIBLE keyframes — chosen by frustum overlap measured on the depth maps
themselves (loops and revisits included), never by index. The fused depth along the ray is the
MEDIAN of the own depth and the members' depths carried onto the ray, t = (d_g − a) / b: the point
of the ray whose depth in g equals g's depth at the landing point (the vote's zmed, over every
covisible keyframe instead of ±3/6/12). The median is the maximum-likelihood consensus under
Laplacian errors; it is never a mean an outlier drags; and the result stays ON an observed ray, so
the point keeps its pixel provenance.

OWNERSHIP. Each surface element is published ONCE. Its owner is the observation of best accuracy
among the mutually agreeing ones, by a TOTAL ORDER on observations
    key(f, p) = ( min(members, evidence_cap),  −bucket(|X − C_f|),  −f ):
confirmed by at least `evidence_cap` other views first; then the NEAREST camera (Omega's error is
relative to depth, so the nearest view measures the element best — and its pixel has the smallest
footprint, so the cloud keeps the finest resolution any keyframe reached), the distance quantized
on a logarithmic scale of relative step `nearest_tie_rel` so that two observations from (nearly)
the same distance share a bucket; then the earliest keyframe. The key is computed from the
observation ALONE, so keys compare transitively — a chain of consumptions is strictly increasing
and can never cycle — and nothing depends on the order anything was read.
An observation is an OWNER iff no member outranks it; otherwise it is CONSUMED by its best-ranked
member (rejected_points.npz, reason `fused`, with the winner). This is a local argmax — parallel,
deterministic, no sequential state. Every chain of consumptions is strictly increasing in the
total order and therefore ends at an owner, so every element is represented; the report measures
the one-hop share (consumed pixels whose winner is itself an owner).

FREE-SPACE CARVING AND EVIDENCE (phase 3c, USER 2026-10-23 "no voladores ni puntos donde no van").
Over the same covisible keyframes every observation counts the views that see FREE SPACE through
its point — g measured a surface farther along g's ray than the point, beyond the pair's tolerance
— judged by instruments at least as good as its own (a learned map never vetoes a photometric
point). More such views than agreeing ones: rejected (reason ``contradicted``) — a floater is seen
through by every camera that looks past it, and a surface element seen by many cameras is never
contradicted by a single wrong map. An owner confirmed by fewer than ``min_evidence`` views (itself
included) is rejected as ``unverified`` (``insufficient_witnesses``): what one camera alone saw is
not a measurement two instruments agree on. Both leave with their counts in rejected_points.npz;
the contradiction count of every published point goes to origins ``n_contradict``.

WHAT IT DOES NOT DO. Two observations of one wall whose poses disagree by more than tau·z (drift
between a visit and a revisit) do not agree and BOTH stay: the fusion never hides drift — the
certification's visit_drift measures it. A real step smaller than tau·z (a 4 cm moulding at 3 m)
is below the depth's own resolution and merges, as any instrument does at its resolution.

Every map here is a torch tensor on f6_torch's device (the card in the pipeline, the CPU in the
tests); gathers, sorts and reductions only — no scatter, no atomics (USER 2026-10-08).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from precision import f6_torch as FT

LOG_TAG = "[fusion]"
REJECT_FUSED = 10                      # precision.provenance.REJECT_REASONS["fused"]
REJECT_CONTRADICTED = 9                # ...["contradicted"]: a view sees free space through the point
REJECT_UNVERIFIED = 1                  # ...["insufficient_witnesses"]: fewer than min_evidence views


class OwnerFusionError(RuntimeError):
    pass


@dataclass
class FusionResult:
    owner: Dict[int, np.ndarray]                 # frame → [H, W] bool: the pixel publishes its point
    depth: Dict[int, np.ndarray]                 # frame → [H, W] float32 fused depth along the pixel's ray (0 = none)
    members: Dict[int, np.ndarray]               # frame → [H, W] int16: agreeing covisible observations
    winner: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]   # frame → (winner frame i32, row i16, col i16) of consumed pixels
    candidates: Dict[int, List[int]]             # frame → the covisible keyframes it was fused with
    report: dict = field(default_factory=dict)
    fused_tier: Dict[int, np.ndarray] = field(default_factory=dict)   # frame → [H, W] int8: the instrument tier the fused depth came from
    contra: Dict[int, np.ndarray] = field(default_factory=dict)       # frame → [H, W] int16: covisible observations that see free space through the point
    free_space: Dict[int, np.ndarray] = field(default_factory=dict)   # frame → [H, W] bool: rejected, more views see through it than confirm it
    unverified: Dict[int, np.ndarray] = field(default_factory=dict)   # frame → [H, W] bool: rejected, fewer than min_evidence views


# ── covisibility: who sees what, measured on the depth maps ────────────────

def _project_float(Xc: torch.Tensor, K: np.ndarray, H: int, W: int):
    """Camera-frame points [..., 3] → sub-pixel (u, v), depth, the in-front-and-inside mask (the
    image is [0, W−1] x [0, H−1] of pixel CENTRES)."""
    z = Xc[..., 2]
    ok = z > 0
    zs = torch.where(ok, z, torch.ones_like(z))
    u = float(K[0, 0]) * Xc[..., 0] / zs + float(K[0, 2])
    v = float(K[1, 1]) * Xc[..., 1] / zs + float(K[1, 2])
    ok = ok & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    return u, v, z, ok


def covisibility(frames: Sequence[int], depth: Dict[int, np.ndarray], valid: Dict[int, np.ndarray],
                 K: np.ndarray, c2w: Dict[int, np.ndarray], stride: int) -> np.ndarray:
    """cov[i, j] = the share of keyframe i's sampled valid points (every ``stride``-th row and column)
    that land inside keyframe j's image, in front of it. Measured on the depth maps: a revisit sees
    the same wall whatever its index."""
    n = len(frames)
    K = np.asarray(K, np.float64)
    cov = np.zeros((n, n), np.float64)
    pts = []
    for f in frames:
        H, W = depth[f].shape
        d = FT._t(depth[f]); v = FT._t(valid[f], torch.bool) & (d > 0)
        sub = torch.zeros((H, W), dtype=torch.bool, device=d.device)
        sub[::int(stride), ::int(stride)] = True
        m = v & sub
        rays = FT.camera_rays(H, W, K)
        Xc = rays[m] * d[m][:, None]
        R = FT._t(c2w[f][:3, :3]); t = FT._t(c2w[f][:3, 3])
        pts.append(Xc @ R.T + t)
    for i, f in enumerate(frames):
        if pts[i].shape[0] == 0:
            continue
        for j, g in enumerate(frames):
            if j == i:
                continue
            H, W = depth[g].shape
            T = FT._t(np.linalg.inv(c2w[g]))
            Xg = pts[i] @ T[:3, :3].T + T[:3, 3]
            _, _, _, ok = _project_float(Xg, K, H, W)
            cov[i, j] = float(ok.to(torch.float64).mean())
    return cov


def candidates_of(cov: np.ndarray, min_frac: float, max_candidates: int) -> Dict[int, List[int]]:
    """Per keyframe index the candidate indices: covisible both ways at least ``min_frac`` (the larger
    of the two directions decides), the ``max_candidates`` most covisible first — a total order
    (share, then index) so the list never depends on floating noise alone."""
    n = cov.shape[0]
    out: Dict[int, List[int]] = {}
    for i in range(n):
        share = np.maximum(cov[i], cov[:, i])
        js = [j for j in range(n) if j != i and share[j] >= float(min_frac)]
        js.sort(key=lambda j: (-float(share[j]), j))
        out[i] = js[:int(max_candidates)]
    return out


# ── one neighbour carried onto a keyframe's rays ─────────────────────────────

def carry(z_f: torch.Tensor, valid_f: torch.Tensor, ray_w: torch.Tensor, C_f: torch.Tensor,
          w2c_g: np.ndarray, z_g: torch.Tensor, valid_g: torch.Tensor, K: np.ndarray, tau,
          interpolate: bool, tau_g=None):
    """Keyframe g's surface at the landing points of f's pixels: returns (agree [H, W], t [H, W] the
    agreeing depth carried onto f's ray (NaN elsewhere), q_row, q_col the nearest landing pixel,
    contra [H, W]: g measured a surface FARTHER than f's point by more than the tolerance at the
    landing pixel — g sees free space through the point).
    The landing depth of g is read bilinearly when the 2x2 patch is one surface (four valid depths
    within tau of their mean), at the nearest pixel otherwise. ``tau``: one tolerance, or f's
    per-pixel map (its instrument's); with ``tau_g`` (g's per-pixel map) a pair's tolerance is the
    LARGER of the two observations' — the worse instrument of the pair decides what agreement is."""
    H, W = z_g.shape
    T = FT._t(w2c_g)
    Xw = C_f + z_f[..., None] * ray_w
    Xg = Xw @ T[:3, :3].T + T[:3, 3]
    u, v, zg, ok = _project_float(Xg, K, H, W)
    ok = ok & valid_f & (z_f > 0)
    ur = torch.round(u).long().clamp(0, W - 1); vr = torch.round(v).long().clamp(0, H - 1)
    flat_g = z_g.reshape(-1); flat_v = valid_g.reshape(-1)
    idx_n = vr * W + ur
    d_n = flat_g[idx_n]; v_n = flat_v[idx_n] & (d_n > 0)
    tau_t = tau if isinstance(tau, torch.Tensor) else torch.full_like(zg, float(tau))
    if tau_g is not None:
        tau_t = torch.maximum(tau_t, tau_g.reshape(-1)[idx_n])
    d_at = d_n
    if interpolate:
        u0 = torch.floor(u).long().clamp(0, W - 2); v0 = torch.floor(v).long().clamp(0, H - 2)
        du = (u - u0.to(u.dtype)).clamp(0, 1); dv = (v - v0.to(v.dtype)).clamp(0, 1)
        i00 = v0 * W + u0; i01 = i00 + 1; i10 = i00 + W; i11 = i10 + 1
        d00, d01, d10, d11 = flat_g[i00], flat_g[i01], flat_g[i10], flat_g[i11]
        all_v = flat_v[i00] & flat_v[i01] & flat_v[i10] & flat_v[i11] & (d00 > 0) & (d01 > 0) & (d10 > 0) & (d11 > 0)
        stack = torch.stack([d00, d01, d10, d11])
        mean = stack.mean(0)
        flat = all_v & ((stack.amax(0) - stack.amin(0)) <= tau_t * mean)
        d_b = (d00 * (1 - du) * (1 - dv) + d01 * du * (1 - dv) + d10 * (1 - du) * dv + d11 * du * dv)
        d_at = torch.where(flat, d_b, d_n)
        v_n = v_n | flat
    ok = ok & v_n
    one = torch.ones_like(zg)
    e = (zg - d_at) / torch.where(ok, d_at, one)
    agree = ok & (torch.abs(e) <= tau_t)
    contra = ok & (e < -tau_t)
    # the point of f's ray whose depth in g equals g's depth at the landing point
    a = T[2, :3] @ C_f + T[2, 3]
    b = ray_w @ T[2, :3]
    bb = torch.where(torch.abs(b) > 0, b, one)
    t = torch.where(agree & (torch.abs(b) > 0), (d_at - a) / bb, torch.full_like(zg, float("nan")))
    return agree, t, vr, ur, contra


# ── the two passes ───────────────────────────────────────────────────────────

def _frame_geometry(f: int, H: int, W: int, K: np.ndarray, c2w: Dict[int, np.ndarray]):
    rays = FT.camera_rays(H, W, K)
    R = FT._t(c2w[f][:3, :3]); C = FT._t(c2w[f][:3, 3])
    return rays @ R.T, C


def fuse(frames: Sequence[int], depth: Dict[int, np.ndarray], valid: Dict[int, np.ndarray],
         K: np.ndarray, c2w: Dict[int, np.ndarray], tau: float, *, covis_stride: int, min_covis_frac: float,
         max_candidates: int, evidence_cap: int, interpolate: bool, nearest_tie_rel: float,
         chain_max_hops: int = 8, chain_stride: int = 4,
         tier: Optional[Dict[int, np.ndarray]] = None, tau_by_tier: Optional[Sequence[float]] = None,
         free_space: bool = True, min_evidence: int = 1,
         log: Callable = print, progress: Optional[Callable[[float, str], None]] = None) -> FusionResult:
    """The owner fusion over the voted maps (``depth[f]`` > 0 where ``valid[f]``), in two passes:
    1. members and the fused depth of every observation; 2. ownership by the total order.

    INSTRUMENT TIERS (phase 3): ``tier[f]`` names per pixel the instrument that measured its depth —
    0 the photometric reference (COLMAP PatchMatch, geometrically consistent), 1 the learned depth
    (Omega bent and voted) — and ``tau_by_tier`` each instrument's own agreement tolerance (measured).
    A pair of observations agrees within the LARGER of their two tolerances; the fused depth of an
    observation is the median over the BEST tier present among itself and its members (a tier-0
    member never averages with tier-1 ones: the better instrument decides, the worse one fills);
    and the tier is the first component of the owner's key — a photometric observation outranks
    every learned one of the same element.

    FREE-SPACE CARVING (phase 3c, USER 2026-10-23 "no voladores ni puntos donde no van"): over the
    same covisible keyframes, an observation counts the views that see FREE SPACE through its point
    — g measured a surface farther along g's ray than the point, by more than the pair's tolerance
    — judged only by instruments at least as good as its own (a learned map never vetoes a
    photometric point). An observation with more such views than agreeing ones is rejected
    (``free_space``, provenance reason ``contradicted``): a floater is seen through by every view
    that looks past it. MINIMUM EVIDENCE: an owner confirmed by fewer than ``min_evidence`` views
    (itself included) is rejected as ``unverified`` (reason ``insufficient_witnesses``) — a point one
    camera alone saw is not a measurement two instruments agree on. Both go to rejected_points.npz
    with their counts; nothing is deleted."""
    t0 = time.time()
    K = np.asarray(K, np.float64)
    frames = [int(f) for f in frames]
    w2c = {f: np.linalg.inv(c2w[f]) for f in frames}
    cov = covisibility(frames, depth, valid, K, c2w, covis_stride)
    cands = candidates_of(cov, min_covis_frac, max_candidates)
    n_cand = [len(cands[i]) for i in range(len(frames))]
    tiers = tier is not None
    taus = [float(x) for x in (tau_by_tier if tau_by_tier is not None else (tau,))]
    if tiers and len(taus) < 2:
        raise OwnerFusionError("tier maps need one tau per tier (tau_by_tier)")
    log(f"{LOG_TAG} covisibility over {len(frames)} keyframes (stride {covis_stride}): candidates per keyframe "
        f"min {min(n_cand)}, median {int(np.median(n_cand))}, max {max(n_cand)} (share >= {min_covis_frac:g}, "
        f"cap {max_candidates}); tau " + (", ".join(f"tier {k} {x * 100:.2f} %" for k, x in enumerate(taus))
                                         if tiers else f"{tau * 100:.2f} %"))
    zt = {f: FT._t(depth[f]) for f in frames}
    vt = {f: FT._t(valid[f], torch.bool) & (zt[f] > 0) for f in frames}
    tt: Dict[int, torch.Tensor] = {}
    tau_m: Dict[int, torch.Tensor] = {}
    for f in frames:
        if tiers:
            tt[f] = FT._t(tier[f], torch.int64).clamp(0, len(taus) - 1)
            tau_m[f] = torch.as_tensor(taus, dtype=torch.float64, device=zt[f].device)[tt[f]]
        else:
            tt[f] = torch.zeros(zt[f].shape, dtype=torch.int64, device=zt[f].device)
            tau_m[f] = torch.full(zt[f].shape, float(tau), dtype=torch.float64, device=zt[f].device)
    # pass 1: members and the fused depth
    members: Dict[int, torch.Tensor] = {}
    contra_n: Dict[int, torch.Tensor] = {}
    fused: Dict[int, torch.Tensor] = {}
    best_tier: Dict[int, torch.Tensor] = {}
    for i, f in enumerate(frames):
        H, W = zt[f].shape
        ray_w, C = _frame_geometry(f, H, W, K, c2w)
        cols = [zt[f]]
        col_tier = [tt[f]]
        n_ag = torch.zeros((H, W), dtype=torch.int32, device=zt[f].device)
        n_ct = torch.zeros((H, W), dtype=torch.int32, device=zt[f].device)
        for j in cands[i]:
            g = frames[j]
            agree, t, qr, qc, contra = carry(zt[f], vt[f], ray_w, C, w2c[g], zt[g], vt[g], K, tau_m[f],
                                             interpolate, tau_g=tau_m[g])
            n_ag = n_ag + agree.to(torch.int32)
            cols.append(t)
            tq = tt[g].reshape(-1)[qr * W + qc]
            col_tier.append(torch.where(agree, tq, torch.full_like(tq, len(taus))))   # non-member: no tier
            # a view sees through the point: counted when its instrument is at least as good
            n_ct = n_ct + (contra & (tq <= tt[f])).to(torch.int32)
        C_all = torch.stack(cols)
        if len(cols) > 1:
            T_all = torch.stack(col_tier)
            bt = T_all.amin(0)                                      # the best tier present (own included)
            nan = torch.full_like(C_all, float("nan"))
            med = FT.nanmedian0(torch.where(T_all == bt[None], C_all, nan))
        else:
            bt = tt[f]
            med = zt[f].to(torch.float64)
        fused[f] = torch.where(vt[f], med, torch.zeros_like(med))
        members[f] = torch.where(vt[f], n_ag, torch.zeros_like(n_ag))
        contra_n[f] = torch.where(vt[f], n_ct, torch.zeros_like(n_ct))
        best_tier[f] = torch.where(vt[f], bt, torch.full_like(bt, len(taus)))
        if progress is not None and (i % 10 == 0 or i == len(frames) - 1):
            progress(100.0 * 0.5 * (i + 1) / len(frames), f"fusion pass 1: {i + 1}/{len(frames)} keyframes")
    # pass 2: ownership by a TOTAL ORDER on observations. Every observation carries its own KEY,
    # computed from itself alone: (min(members, cap), −bucket(distance to its own camera), −frame
    # index); the distance is quantized on a logarithmic scale of relative step `nearest_tie_rel`,
    # so two observations of one element seen from (nearly) the same distance share a bucket and
    # the earlier keyframe decides. Keys compare transitively, so a chain of consumptions is
    # strictly increasing and can never cycle; an owner is an observation no agreeing observation
    # outranks. (An ε-tolerant comparison of two distances per pair — "nearer by more than ε, else
    # the index" — is not transitive: it produced cycles p → q → r → p and left 45 % of the
    # consumed observations without an owner in a synthetic plane; measured, tests.)
    cap = int(evidence_cap)
    eps = float(nearest_tie_rel)
    log_step = float(np.log1p(eps)) if eps > 0 else 0.0
    bucket: Dict[int, torch.Tensor] = {}
    for f in frames:
        H, W = zt[f].shape
        rays = FT.camera_rays(H, W, K)
        d_own = zt[f] * torch.linalg.norm(rays, dim=-1)
        if log_step > 0:
            b = torch.floor(torch.log(torch.clamp(d_own, min=1e-9)) / log_step).to(torch.int64)
        else:
            b = (d_own * 1e9).to(torch.int64)                   # no band: the raw distance decides
        bucket[f] = torch.where(vt[f], b, torch.full_like(b, 0))
    owner: Dict[int, np.ndarray] = {}
    winner: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    free_m: Dict[int, np.ndarray] = {}
    unver_m: Dict[int, np.ndarray] = {}
    tot_owner = tot_valid = tot_consumed = tot_free = tot_unver = 0
    min_ev = max(1, int(min_evidence))
    # the observations that stay in the game: not seen through by more views than confirm them
    alive: Dict[int, torch.Tensor] = {}
    for f in frames:
        alive[f] = vt[f] & ~(contra_n[f] > members[f]) if free_space else vt[f]
    def _prim(f: int) -> torch.Tensor:
        # the first component of the key: the instrument first (a tier-0 observation outranks every
        # tier-1 one), then the confirmed tier
        return (len(taus) - 1 - tt[f]) * (cap + 1) + torch.clamp(members[f], max=cap)

    for i, f in enumerate(frames):
        H, W = zt[f].shape
        ray_w, C = _frame_geometry(f, H, W, K, c2w)
        prim_f = _prim(f)
        buck_f = bucket[f]
        z_f = zt[f]
        lost = torch.zeros((H, W), dtype=torch.bool, device=z_f.device)
        best_prim = torch.full((H, W), -1, dtype=torch.int32, device=z_f.device)
        best_buck = torch.full((H, W), 2 ** 62, dtype=torch.int64, device=z_f.device)
        best_j = torch.full((H, W), -1, dtype=torch.int64, device=z_f.device)
        best_r = torch.zeros((H, W), dtype=torch.int64, device=z_f.device)
        best_c = torch.zeros((H, W), dtype=torch.int64, device=z_f.device)
        for j in cands[i]:
            g = frames[j]
            agree, _, qr, qc, _c = carry(zt[f], vt[f], ray_w, C, w2c[g], zt[g], vt[g], K, tau_m[f], interpolate,
                                         tau_g=tau_m[g])
            qi = qr * W + qc
            prim_g = _prim(g).reshape(-1)[qi]
            buck_g = bucket[g].reshape(-1)[qi]
            alive_g = alive[g].reshape(-1)[qi]
            better = agree & alive_g & ((prim_g > prim_f) | ((prim_g == prim_f) & ((buck_g < buck_f) | ((buck_g == buck_f) & (j < i)))))
            lost = lost | better
            # the best-ranked winner so far, for the rejected record (the same total order)
            up = better & ((prim_g > best_prim) | ((prim_g == best_prim) & ((buck_g < best_buck) | ((buck_g == best_buck) & (j < best_j)))))
            best_prim = torch.where(up, prim_g, best_prim); best_buck = torch.where(up, buck_g, best_buck)
            best_j = torch.where(up, torch.full_like(best_j, j), best_j)
            best_r = torch.where(up, qr, best_r); best_c = torch.where(up, qc, best_c)
        free_f = vt[f] & ~alive[f]
        unver_f = alive[f] & ~lost & (members[f] + 1 < min_ev)
        own = alive[f] & ~lost & ~unver_f
        owner[f] = FT._np(own)
        free_m[f] = FT._np(free_f)
        unver_m[f] = FT._np(unver_f)
        tot_free += int(free_f.sum()); tot_unver += int(unver_f.sum())
        consumed = FT._np(alive[f] & lost)
        bj = FT._np(best_j, np.int64)
        frames_arr = np.asarray(frames + [-1], np.int32)
        wf = np.where(consumed & (bj >= 0), frames_arr[np.where(bj >= 0, bj, len(frames))], -1).astype(np.int32)
        winner[f] = (wf, np.where(consumed, FT._np(best_r), 0).astype(np.int16),
                     np.where(consumed, FT._np(best_c), 0).astype(np.int16))
        tot_owner += int(own.sum()); tot_valid += int(vt[f].sum()); tot_consumed += int(consumed.sum())
        del lost, best_prim, best_buck, best_j, best_r, best_c
        if progress is not None and (i % 10 == 0 or i == len(frames) - 1):
            progress(100.0 * 0.5 * (1.0 + (i + 1) / len(frames)), f"fusion pass 2: {i + 1}/{len(frames)} keyframes")
    # representation: every consumed observation follows its winners to an OWNER (the chain is
    # strictly increasing in the total order, so it ends); the owner's fused point must stand where
    # the consumed observation stood — measured in the consumed pixel's own footprint (z / f)
    chain = chain_check(frames, zt, vt, owner, winner, fused, K, c2w, max_hops=chain_max_hops,
                        stride=chain_stride)
    mem_all = np.concatenate([FT._np(members[f])[owner[f]] for f in frames]) if tot_owner else np.zeros(0)
    q = np.percentile(mem_all, [5, 25, 50, 75, 95]).tolist() if mem_all.size else [None] * 5
    rep = {"rule": "owner = the observation no agreeing covisible observation outranks; rank = (min(members, "
                   "evidence_cap), nearest camera, earliest keyframe, pixel); fused depth = median of the own depth "
                   "and the members' depths carried onto the ray",
           "tau": float(tau), "evidence_cap": cap, "interpolate": bool(interpolate), "nearest_tie_rel": eps,
           "covisibility": {"stride": int(covis_stride), "min_frac": float(min_covis_frac), "cap": int(max_candidates),
                            "candidates_per_keyframe": {"min": min(n_cand), "median": float(np.median(n_cand)), "max": max(n_cand)}},
           "observations": tot_valid, "owners": tot_owner, "consumed": tot_consumed,
           "free_space": {"enabled": bool(free_space), "rejected": tot_free,
                          "rule": "more covisible views (instrument >= the point's) see free space through the "
                                  "point than agree with it"},
           "unverified": {"min_evidence": min_ev, "rejected": tot_unver,
                          "rule": "an owner confirmed by fewer than min_evidence views (itself included)"},
           "owners_share": (tot_owner / tot_valid) if tot_valid else None,
           "representation": chain,
           "members_of_owners": {"p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
                                 "share_zero": float((mem_all == 0).mean()) if mem_all.size else None},
           "seconds": round(time.time() - t0, 1)}
    log(f"{LOG_TAG} {tot_owner:,} owners of {tot_valid:,} observations ({rep['owners_share'] * 100 if tot_valid else 0:.1f} %), "
        f"{tot_consumed:,} consumed, {tot_free:,} seen through (free space), {tot_unver:,} unverified "
        f"(< {min_ev} views); represented by an owner within {chain_max_hops} hops: "
        f"{(chain['resolved_share'] * 100 if chain['sampled'] else 100):.1f} % of {chain['sampled']:,} sampled, "
        f"at p50 {chain['distance_footprints']['p50']} / p95 {chain['distance_footprints']['p95']} footprints; "
        f"members of the owners p50 "
        f"{q[2]}, unconfirmed {rep['members_of_owners']['share_zero'] * 100 if mem_all.size else 0:.1f} %; "
        f"{rep['seconds']} s")
    if tiers:
        own_t = np.concatenate([FT._np(best_tier[f])[owner[f]] for f in frames]) if tot_owner else np.zeros(0)
        rep["tiers"] = {"tau_by_tier": taus,
                        "owners_by_fused_tier": {str(k): int((own_t == k).sum()) for k in range(len(taus))},
                        "observations_by_tier": {str(k): int(sum(int(((tt[f] == k) & vt[f]).sum()) for f in frames))
                                                 for k in range(len(taus))}}
        log(f"{LOG_TAG} owners by the instrument of their fused depth: "
            + ", ".join(f"tier {k}: {v:,}" for k, v in rep["tiers"]["owners_by_fused_tier"].items()))
    return FusionResult(owner=owner, depth={f: FT._np(fused[f], np.float32) for f in frames},
                        members={f: FT._np(torch.clamp(members[f], max=32767), np.int16) for f in frames},
                        winner=winner, candidates={f: [frames[j] for j in cands[i]] for i, f in enumerate(frames)},
                        fused_tier={f: FT._np(torch.clamp(best_tier[f], max=127), np.int8) for f in frames},
                        contra={f: FT._np(torch.clamp(contra_n[f], max=32767), np.int16) for f in frames},
                        free_space=free_m, unverified=unver_m, report=rep)


def chain_check(frames: Sequence[int], zt: Dict[int, torch.Tensor], vt: Dict[int, torch.Tensor],
                owner: Dict[int, np.ndarray], winner: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                fused: Dict[int, torch.Tensor], K: np.ndarray, c2w: Dict[int, np.ndarray], max_hops: int,
                stride: int) -> dict:
    """The representation check: a strided sample of the consumed observations follows winner →
    winner … up to ``max_hops`` until an owner; the distance between the consumed observation's
    point and that owner's fused point, in the consumed pixel's footprint (z / f). Returns the
    counts, the resolved share, the hop distribution and the distance quantiles."""
    K = np.asarray(K, np.float64)
    fx = float(K[0, 0])
    owner_m = {f: owner[f] for f in frames}
    wf_m = {f: winner[f][0] for f in frames}; wr_m = {f: winner[f][1] for f in frames}; wc_m = {f: winner[f][2] for f in frames}
    P0, Z0, cur_f, cur_r, cur_c = [], [], [], [], []
    for f in frames:
        H, W = zt[f].shape
        consumed = FT._np(vt[f]) & ~owner[f] & (wf_m[f] >= 0)
        sub = np.zeros((H, W), bool); sub[::int(stride), ::int(stride)] = True
        r, c = np.nonzero(consumed & sub)
        if not len(r):
            continue
        z = FT._np(zt[f])[r, c].astype(np.float64)
        X = np.stack([(c - K[0, 2]) / K[0, 0] * z, (r - K[1, 2]) / K[1, 1] * z, z], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
        P0.append(X); Z0.append(z)
        cur_f.append(np.full(len(r), int(f), np.int64)); cur_r.append(r.astype(np.int64)); cur_c.append(c.astype(np.int64))
    if not P0:
        return {"sampled": 0, "stride": int(stride), "max_hops": int(max_hops), "resolved": 0, "resolved_share": None,
                "hops": None, "distance_footprints": {"p50": None, "p95": None, "max": None}}
    P0 = np.concatenate(P0); Z0 = np.concatenate(Z0)
    cf = np.concatenate(cur_f); cr = np.concatenate(cur_r); cc = np.concatenate(cur_c)
    n = len(cf)
    resolved = np.zeros(n, bool); hops = np.zeros(n, np.int32)
    alive = np.ones(n, bool)
    for h in range(1, int(max_hops) + 1):
        # step every alive chain to its winner
        nf = np.full(n, -1, np.int64); nr = np.zeros(n, np.int64); nc = np.zeros(n, np.int64)
        for f in np.unique(cf[alive]):
            sel = alive & (cf == f)
            nf[sel] = wf_m[int(f)][cr[sel], cc[sel]]; nr[sel] = wr_m[int(f)][cr[sel], cc[sel]]; nc[sel] = wc_m[int(f)][cr[sel], cc[sel]]
        dead = alive & (nf < 0)
        alive = alive & ~dead
        cf = np.where(alive, nf, cf); cr = np.where(alive, nr, cr); cc = np.where(alive, nc, cc)
        done = np.zeros(n, bool)
        for f in np.unique(cf[alive]):
            sel = alive & (cf == f)
            done[sel] = owner_m[int(f)][cr[sel], cc[sel]]
        resolved |= done; hops[done] = h
        alive = alive & ~done
        if not alive.any():
            break
    # the distance to the terminal owner's fused point, in the consumed pixel's footprint
    d = np.full(n, np.nan)
    for f in np.unique(cf[resolved]):
        sel = resolved & (cf == f)
        z = FT._np(fused[int(f)])[cr[sel], cc[sel]].astype(np.float64)
        X = np.stack([(cc[sel] - K[0, 2]) / K[0, 0] * z, (cr[sel] - K[1, 2]) / K[1, 1] * z, z], 1) @ c2w[int(f)][:3, :3].T + c2w[int(f)][:3, 3]
        d[sel] = np.linalg.norm(X - P0[sel], axis=1) / (Z0[sel] / fx)
    dd = d[np.isfinite(d)]
    qs = np.percentile(dd, [50, 95]).tolist() if dd.size else [None, None]
    return {"sampled": int(n), "stride": int(stride), "max_hops": int(max_hops), "resolved": int(resolved.sum()),
            "resolved_share": float(resolved.mean()),
            "hops": {str(h): int((hops == h).sum()) for h in range(1, int(max_hops) + 1) if (hops == h).any()},
            "distance_footprints": {"p50": (round(float(qs[0]), 3) if qs[0] is not None else None),
                                    "p95": (round(float(qs[1]), 3) if qs[1] is not None else None),
                                    "max": (round(float(dd.max()), 3) if dd.size else None)}}


def write_rejected(path, frames: Sequence[int], result: FusionResult, valid: Dict[int, np.ndarray]) -> int:
    """Every observation that left the cloud, as ``rejected_points.npz``: consumed (reason `fused`,
    with the winner that represents it), seen through (`contradicted`) and unverified
    (`insufficient_witnesses`), each with its agreeing and contradicting counts. Returns the count."""
    fg, pr, pc, wf, wr, wc, rs, na, nc = [], [], [], [], [], [], [], [], []
    for f in frames:
        v = np.asarray(valid[f], bool)
        free = result.free_space.get(f)
        unv = result.unverified.get(f)
        consumed = v & ~result.owner[f] & (result.winner[f][0] >= 0)
        if free is not None:
            consumed &= ~free
        if unv is not None:
            consumed &= ~unv
        for m, reason in ((consumed, REJECT_FUSED),
                          (v & free if free is not None else np.zeros_like(v), REJECT_CONTRADICTED),
                          (v & unv if unv is not None else np.zeros_like(v), REJECT_UNVERIFIED)):
            if not m.any():
                continue
            r, c = np.nonzero(m)
            fg.append(np.full(len(r), int(f), np.int32)); pr.append(r.astype(np.int16)); pc.append(c.astype(np.int16))
            if reason == REJECT_FUSED:
                wf.append(result.winner[f][0][m]); wr.append(result.winner[f][1][m]); wc.append(result.winner[f][2][m])
            else:
                wf.append(np.full(len(r), -1, np.int32)); wr.append(np.zeros(len(r), np.int16)); wc.append(np.zeros(len(r), np.int16))
            rs.append(np.full(len(r), reason, np.uint8))
            na.append(result.members[f][m].astype(np.int16))
            nc.append((result.contra[f][m] if f in result.contra else np.zeros(len(r), np.int16)).astype(np.int16))
    n = int(sum(len(x) for x in fg))
    cat = lambda xs, dt: (np.concatenate(xs) if n else np.zeros(0, dt))  # noqa: E731
    np.savez_compressed(path, version=np.int64(4), stage=np.array("owner_fusion"),
                        frame_global=cat(fg, np.int32), pixel_row=cat(pr, np.int16), pixel_col=cat(pc, np.int16),
                        winner_frame=cat(wf, np.int32), winner_row=cat(wr, np.int16), winner_col=cat(wc, np.int16),
                        reason=cat(rs, np.uint8), n_agree=cat(na, np.int16), n_contradict=cat(nc, np.int16))
    return n
