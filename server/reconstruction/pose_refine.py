#!/usr/bin/env python3
"""
pose_refine — E-full: GLOBAL per-frame pose refinement (the stage
docs/pose_refinement.md specs, sized to reality: 210 keyframes x 6 DOF).

WHY (measured, test4 2026-07-11). The omega backbone emits per-frame poses in
one feed-forward pass — no joint optimization ever reconciles them. Phase A
proved every repair stage (elastic/drift/intra/finereg) leaves the core defect
untouched: covisible frames place the SAME surface at slightly different poses
(layering/drift, wavy edges, mispositioned walk end). This stage optimizes ALL
frame poses jointly against exactly that signal.

METHOD — joint point-to-plane optimization of per-frame world CORRECTIONS C_f
(multi-view global ICP with smoothness priors):
  1. Per frame: sample its own cloud points (origins npz gives frame_global +
     pixel per PLY point), rasterize a CONSISTENT depth + world-point grid
     (nearest point per pixel in both), estimate per-pixel normals, mask
     grazing pixels.
  2. Covisibility MATCHES: for frame pairs within ``pair_window``, project i's
     samples into j; a trusted same-pixel hit with compatible depth yields
     (p_i, q_j, n_j) — two placements of the same surface. No per-edge rigid
     fit: 6-DOF per edge would invent the unobservable in-plane components,
     and point-to-POINT drowns the normal signal in in-plane association
     noise (both measured on the synthetic corridor).
  3. SOLVE min over {C_f}:  sum_matches huber( n' . (C_i p - C_j q) )
                          + w_odo   sum_f |Log(C_f^-1 C_{f+1})|^2  (omega's
                            local relative poses are good -> corrections vary
                            smoothly along the walk)
                          + w_leash sum_f |Log(C_f)|^2             (weak)
     with per-edge bounded mass (bias from thousands of sub-mm-skewed matches
     must not out-vote the leash on clean data). C_0 fixes the gauge.
     Gauss-Newton relinearized each iteration; J_j = -J_i = -[p' x n', n'].
     Verified exact on perfect correspondences (residual 0.07 m -> 0.6 mm).
  4. OUTER ICP LOOP (``outer_iters``): measure -> solve -> re-place ->
     re-measure. One-shot association is biased by the very error being
     corrected; re-association compounds the gain (synthetic: 31 -> 14 mm).
  5. GATE (non-negotiable after E1): a FRESH measurement at the final
     placement must show >= ``min_gain`` less inter-frame disagreement than
     the initial one, else NOTHING is applied (identity).
  6. APPLY: every PLY point moves with its frame's correction (origins
     mapping); camera_poses.txt gets C_f @ M_f (backup .preposerefine).

Runs in the fine_register slot (same reproject_chunks contract), pure CPU,
deterministic. Report: pose_refine_report.json.

Author: STAC session 2026-07-11 (E-full).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PoseRefine")

_EPS = 1e-12


# ── SE3 helpers (cv2-free: server env safe) ──────────────────────────

def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < _EPS:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _log_rot(R: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(tr))
    if th < 1e-9:
        return np.zeros(3)
    return (th / (2.0 * np.sin(th))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """xi = [w(3), t(3)] -> 4x4 (small-motion convention: R=exp(w), t as-is)."""
    M = np.eye(4)
    M[:3, :3] = _rodrigues(np.asarray(xi[:3], np.float64))
    M[:3, 3] = np.asarray(xi[3:], np.float64)
    return M


def se3_log(M: np.ndarray) -> np.ndarray:
    return np.concatenate([_log_rot(np.asarray(M[:3, :3], np.float64)),
                           np.asarray(M[:3, 3], np.float64)])


def robust_rigid(src: np.ndarray, dst: np.ndarray, iters: int = 6
                 ) -> Optional[Tuple[np.ndarray, float, int]]:
    """IRLS-Cauchy rigid fit dst ~= R src + t. Returns (T 4x4, med_resid, n)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = len(src)
    if n < 50:
        return None
    w = np.ones(n)
    R, t = np.eye(3), np.zeros(3)
    for _ in range(iters):
        ws = w / max(w.sum(), _EPS)
        cs = ws @ src
        cd = ws @ dst
        H = ((src - cs) * ws[:, None]).T @ (dst - cd)
        U, _, Vt = np.linalg.svd(H)
        S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
        R = Vt.T @ S @ U.T
        t = cd - R @ cs
        r = np.linalg.norm(dst - (src @ R.T + t), axis=1)
        c = max(1.5 * float(np.median(r)), 1e-6)
        w = 1.0 / (1.0 + (r / c) ** 2)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    med = float(np.median(np.linalg.norm(dst - (src @ R.T + t), axis=1)))
    return T, med, n


# ── measurement ──────────────────────────────────────────────────────

def rasterize_depth(pts_cam_z: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    H: int, W: int) -> np.ndarray:
    """Sparse z-buffer of a frame's own points at its native pixel grid."""
    z = np.full((H, W), np.inf, np.float32)
    np.minimum.at(z, (rows, cols), pts_cam_z.astype(np.float32))
    z[~np.isfinite(z)] = 0.0
    return z


def rasterize_frame(P_world: np.ndarray, pts_cam_z: np.ndarray,
                    rows: np.ndarray, cols: np.ndarray, H: int, W: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """CONSISTENT z-buffer + world-point grid: both hold the NEAREST point of
    each pixel. Writing in descending-z order makes the last write the
    smallest z — a min-z raster paired with a last-write world grid would mix
    two different points at colliding pixels and poison every correspondence
    (measured on the synthetic: only 7/123 edges truth-accurate)."""
    order = np.argsort(-np.asarray(pts_cam_z, np.float64), kind="stable")
    r, c = rows[order], cols[order]
    raster = np.zeros((H, W), np.float32)
    raster[r, c] = pts_cam_z[order].astype(np.float32)
    world = np.zeros((H, W, 3), np.float64)
    world[r, c] = P_world[order]
    return raster, world


def grazing_mask(raster: np.ndarray, grad_max: float = 0.12) -> np.ndarray:
    """Pixels whose local depth gradient is small enough to trust a pixel-level
    correspondence. On grazing surfaces (floor/walls parallel to the view ray)
    the depth changes metres per pixel — a same-pixel match there pairs points
    far apart ALONG the surface and the rigid fit reads that slide as a huge
    translation (measured on the synthetic corridor: 1.4m of pure junk)."""
    valid = raster > 1e-6
    z = np.where(valid, raster, np.nan)
    diffs = []
    for ax, sl_a, sl_b in ((0, np.s_[1:, :], np.s_[:-1, :]),
                           (1, np.s_[:, 1:], np.s_[:, :-1])):
        d = np.full(raster.shape, np.nan, np.float32)
        dd = np.abs(z[sl_a] - z[sl_b])          # nan unless BOTH pixels valid
        d[sl_a] = dd
        diffs.append(d)
        d2 = np.full(raster.shape, np.nan, np.float32)
        d2[sl_b] = dd
        diffs.append(d2)
    with np.errstate(all='ignore'):
        g = np.nanmin(np.stack(diffs), axis=0)  # best valid-neighbour gradient
    # grazing = even the SMOOTHEST valid neighbour jumps too much; a pixel
    # with no valid neighbour cannot be judged -> trusted (depth tol covers it)
    return valid & ~(np.isfinite(g) & (g > grad_max))


def normal_grid(world: np.ndarray, valid: np.ndarray, step: int = 2
                ) -> np.ndarray:
    """Per-pixel surface normal from the world grid (cross of the two pixel-
    neighbour tangents). Zero where neighbours are missing."""
    n = np.zeros_like(world)
    a = world[step:, :-step] - world[:-step, :-step]
    b = world[:-step, step:] - world[:-step, :-step]
    cr = np.cross(a, b)
    nn = np.linalg.norm(cr, axis=-1, keepdims=True)
    ok = (valid[step:, :-step] & valid[:-step, step:]
          & valid[:-step, :-step]) & (nn[..., 0] > 1e-9)
    cr = np.where(nn > 1e-9, cr / np.maximum(nn, 1e-12), 0.0)
    n[:-step, :-step][ok] = cr[ok]
    return n


def gather_matches(pts_i: np.ndarray, w2c_j: np.ndarray, K_j: np.ndarray,
                   raster_j: np.ndarray, world_j_grid: np.ndarray,
                   normal_j: np.ndarray, rel_tol: float = 0.15,
                   abs_tol: float = 0.60, max_depth: float = 8.0,
                   near_ref: float = 8.0,
                   min_matches: int = 200, max_matches: int = 400,
                   trust_j: Optional[np.ndarray] = None, seed: int = 0
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray]]:
    """Surface matches between frames i and j: project i's world samples into
    j; where j has a TRUSTED rastered point at that pixel with compatible
    depth and a defined normal, (p_i, q_j, n_j) is one placement pair of the
    same surface. Trust filters: depth cap ``max_depth``, non-grazing pixels,
    depth compatibility min(rel_tol*z, abs_tol). Beyond ``near_ref`` a match's
    weight falls as (near_ref/z)^2 — pixel-match error grows with depth, so
    the near field keeps the authority while FAR landmarks (F1's chimney,
    ~image 704: seen only beyond 8 m, it had NO voice in the consensus and
    drifted when its observers' poses moved 16 cm) still anchor rotations.
    Returns (P, Q, N, W) arrays or None.

    The matches feed a JOINT point-to-plane solve — no per-edge rigid fit:
    a 6-DOF fit per edge would have to invent the in-plane (unobservable)
    components, and point-to-POINT fits drown the normal-direction signal in
    in-plane association noise (measured on the synthetic corridor: median
    edge error 0.36 m for 6 cm of injected noise)."""
    H, W = raster_j.shape
    X = pts_i @ w2c_j[:3, :3].T + w2c_j[:3, 3]
    z = X[:, 2]
    m = (z > 0.2) & (z <= max_depth)
    if m.sum() < min_matches:
        return None
    u = np.round(X[m, 0] / z[m] * K_j[0, 0] + K_j[0, 2]).astype(int)
    v = np.round(X[m, 1] / z[m] * K_j[1, 1] + K_j[1, 2]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inb.sum() < min_matches:
        return None
    u, v = u[inb], v[inb]
    src_w = pts_i[m][inb]
    zj = raster_j[v, u]
    tol = np.minimum(rel_tol * np.maximum(zj, 1e-6), abs_tol)
    nrm = normal_j[v, u]
    ok = ((zj > 1e-6) & (zj <= max_depth)
          & (np.abs(X[m][inb][:, 2] - zj) <= tol)
          & (np.linalg.norm(nrm, axis=1) > 0.5))
    if trust_j is not None:
        ok &= trust_j[v, u]
    if ok.sum() < min_matches:
        return None
    P = src_w[ok]
    Q = world_j_grid[v[ok], u[ok]]
    N = nrm[ok]
    Wt = (float(near_ref) / np.maximum(zj[ok], float(near_ref))) ** 2
    if len(P) > max_matches:
        keep = np.random.default_rng(seed).choice(len(P), max_matches,
                                                  replace=False)
        P, Q, N, Wt = P[keep], Q[keep], N[keep], Wt[keep]
    return P, Q, N, Wt


# ── pose graph solve ─────────────────────────────────────────────────

def _edge_residuals(edge, Cs):
    """Point-to-plane residuals of one edge under corrections Cs.
    Returns (r [M], p' [M,3], n' [M,3]) — p', n' already in corrected world."""
    i, j, P, Q, N = edge[0], edge[1], edge[2], edge[3], edge[4]
    Ci, Cj = Cs[i], Cs[j]
    p = P @ Ci[:3, :3].T + Ci[:3, 3]
    q = Q @ Cj[:3, :3].T + Cj[:3, 3]
    nr = N @ Cj[:3, :3].T
    r = np.sum(nr * (p - q), axis=1)
    return r, p, nr


def solve_pose_graph(n: int, edges: List[tuple], odo_weight: float = 2.0,
                     leash_weight: float = 0.1, iters: int = 12,
                     huber_m: float = 0.05, edge_mass: float = 5.0
                     ) -> List[np.ndarray]:
    """JOINT point-to-plane solve for per-frame world corrections C_f.

    edges: (i, j, P, Q, N) match blocks from gather_matches. Each match
    contributes ONE residual along its surface normal:
        r = n' . (C_i p - C_j q)      J_i = [p' x n', n'],  J_j = -J_i
    (q drops out of the j-Jacobian — standard point-to-plane symmetry).
    Unobservable directions are simply never constrained — the odometry
    smoothness (omega's local relative poses are good) and a weak identity
    leash regularize them, honestly, at 'no correction'. C_0 fixes the gauge.
    Gauss-Newton with IRLS-Huber, relinearized every iteration."""
    xi = np.zeros((n, 6))
    for it in range(iters):
        Cs = [se3_exp(x) for x in xi]
        A = np.zeros((6 * n, 6 * n))
        b = np.zeros(6 * n)
        for edge in edges:
            i, j = edge[0], edge[1]
            r, p, nr = _edge_residuals(edge, Cs)
            w = np.where(np.abs(r) <= huber_m, 1.0,
                         huber_m / np.maximum(np.abs(r), 1e-12))
            if len(edge) > 5 and edge[5] is not None:
                w = w * edge[5]         # per-match depth weight (near_ref/z)^2
            w = w * (edge_mass / max(len(r), 1))   # bounded mass per edge:
            # thousands of sub-mm-biased matches must not out-vote the leash
            # on clean data, yet real signal must beat the priors
            J = np.concatenate([np.cross(p, nr), nr], axis=1)      # [M,6]
            JW = J * w[:, None]
            H6 = JW.T @ J
            g6 = JW.T @ r
            si, sj = 6 * i, 6 * j
            A[si:si + 6, si:si + 6] += H6
            A[sj:sj + 6, sj:sj + 6] += H6
            A[si:si + 6, sj:sj + 6] -= H6
            A[sj:sj + 6, si:si + 6] -= H6
            b[si:si + 6] -= g6
            b[sj:sj + 6] += g6
        for f in range(n - 1):                     # odometry smoothness
            r6 = se3_log(np.linalg.inv(Cs[f + 1]) @ Cs[f])
            sa, sb = 6 * f, 6 * (f + 1)
            b[sa:sa + 6] -= odo_weight * r6
            b[sb:sb + 6] += odo_weight * r6
            A[sa:sa + 6, sa:sa + 6] += odo_weight * np.eye(6)
            A[sb:sb + 6, sb:sb + 6] += odo_weight * np.eye(6)
            A[sa:sa + 6, sb:sb + 6] -= odo_weight * np.eye(6)
            A[sb:sb + 6, sa:sa + 6] -= odo_weight * np.eye(6)
        for f in range(n):                          # weak identity leash
            sf = 6 * f
            b[sf:sf + 6] -= leash_weight * se3_log(Cs[f])
            A[sf:sf + 6, sf:sf + 6] += leash_weight * np.eye(6)
        A[:6, :6] += 1e6 * np.eye(6)                # gauge: C_0 = I
        b[:6] = 0.0
        A += 1e-6 * np.eye(6 * n)
        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logger.warning("pose_refine: singular system at iter %d — stop", it)
            break
        xi = xi + dx.reshape(n, 6)
        if float(np.linalg.norm(dx)) < 1e-9:
            break
    return [se3_exp(x) for x in xi]


def edge_errors(edges, Cs) -> np.ndarray:
    """Median |point-to-plane residual| per edge under corrections Cs."""
    out = []
    for edge in edges:
        r, _, _ = _edge_residuals(edge, Cs)
        out.append(float(np.median(np.abs(r))))
    return np.asarray(out)


# ── orchestration ────────────────────────────────────────────────────

def run(output_dir: Path, pair_window: int = 15, samples_per_frame: int = 4000,
        rel_tol: float = 0.15, max_depth: float = 30.0, near_ref: float = 8.0,
        odo_weight: float = 2.0,
        leash_weight: float = 0.1, min_gain: float = 0.10,
        holdout_frac: float = 0.2, outer_iters: int = 3, seed: int = 0) -> int:
    """Refine all frame poses; returns number of frames actually moved."""
    from plyfile import PlyData, PlyElement
    output_dir = Path(output_dir)
    pp = output_dir / "camera_poses.txt"
    cf = output_dir / "camera_frames.txt"
    ip = output_dir / "intrinsic.txt"
    chunk_files = [p for p in sorted(output_dir.glob("chunk_*.ply"))
                   if not p.stem.startswith(("chunk_997", "chunk_998",
                                             "chunk_999"))]
    if not (pp.exists() and cf.exists() and ip.exists()) or len(chunk_files) < 1:
        logger.info("pose_refine: inputs missing — skipped")
        return 0
    poses = [np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
             for ln in pp.read_text().splitlines() if len(ln.split()) == 16]
    frames = [int(float(x)) for x in cf.read_text().split()]
    Ks = []
    for ln in ip.read_text().splitlines():
        v = [float(x) for x in ln.split()]
        if len(v) == 4:
            Ks.append(np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]]))
    if not (len(poses) == len(frames) == len(Ks)):
        logger.warning("pose_refine: poses/frames/intrinsics mismatch "
                       "(%d/%d/%d) — skipped", len(poses), len(frames), len(Ks))
        return 0
    n = len(frames)
    fidx = {f: k for k, f in enumerate(frames)}

    # ── gather per-frame points (world + pixel) from every chunk ──
    rng = np.random.default_rng(seed)
    pts: Dict[int, list] = {k: [] for k in range(n)}
    pix: Dict[int, list] = {k: [] for k in range(n)}
    plys = []
    HW = None
    for p in chunk_files:
        org = output_dir / f"{p.stem}_origins.npz"
        if not org.exists():
            logger.warning("pose_refine: %s missing — skipped", org.name)
            return 0
        z = np.load(org)
        pd = PlyData.read(str(p))
        v = np.array(pd["vertex"].data)
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        fg = z["frame_global"].astype(np.int64)
        if len(fg) != len(xyz):
            logger.warning("pose_refine: %s points!=origins — skipped", p.name)
            return 0
        if HW is None and "scaled_resolution" in z.files:
            HW = tuple(int(x) for x in z["scaled_resolution"])
        plys.append((p, v, fg))
        rc = np.stack([z["pixel_row"].astype(np.int32),
                       z["pixel_col"].astype(np.int32)], axis=1)
        for f in np.unique(fg):
            k = fidx.get(int(f))
            if k is None:
                continue
            m = fg == f
            pts[k].append(xyz[m])
            pix[k].append(rc[m])
    if HW is None:
        logger.warning("pose_refine: no scaled_resolution in origins — skipped")
        return 0
    H, W = HW
    frame_pts = {}
    frame_pix = {}
    for k in range(n):
        if not pts[k]:
            continue
        P = np.concatenate(pts[k])
        R = np.concatenate(pix[k])
        frame_pts[k] = P
        frame_pix[k] = R
    logger.info("pose_refine: %d frames with points, grid %dx%d, "
                "pair_window %d", len(frame_pts), H, W, pair_window)

    # ── outer ICP loop: measure (associate) → joint solve → re-place →
    # re-measure. One-shot association is biased by the very error being
    # measured; with 2-3 re-associations the matches sharpen as the frames
    # converge (solver verified exact on perfect correspondences). ──
    def _measure(cur_pts, w2c_cur, it):
        rasters, worlds, trusts, normals, samp = {}, {}, {}, {}, {}
        for k, P in cur_pts.items():
            Xc = P @ w2c_cur[k][:3, :3].T + w2c_cur[k][:3, 3]
            rows, cols = frame_pix[k][:, 0], frame_pix[k][:, 1]
            rasters[k], worlds[k] = rasterize_frame(P, Xc[:, 2], rows, cols,
                                                    H, W)
            trusts[k] = grazing_mask(rasters[k])
            normals[k] = normal_grid(worlds[k], rasters[k] > 1e-6)
            take = min(samples_per_frame, len(P))
            samp[k] = P[rng.choice(len(P), take, replace=False)]
        out = []
        for i in sorted(cur_pts):
            for j in sorted(cur_pts):
                if not (i < j <= i + pair_window):
                    continue
                e = gather_matches(samp[i], w2c_cur[j], Ks[j], rasters[j],
                                   worlds[j], normals[j], rel_tol=rel_tol,
                                   max_depth=max_depth, near_ref=near_ref,
                                   trust_j=trusts[j],
                                   seed=seed + it * 999983 + i * 1000 + j)
                if e is not None:
                    out.append((i, j) + e)
        return out

    total = [np.eye(4) for _ in range(n)]
    pre = None
    h0 = None
    for it in range(max(int(outer_iters), 1)):
        cur_pts = {k: (P @ total[k][:3, :3].T + total[k][:3, 3])
                   for k, P in frame_pts.items()}
        w2c_cur = [np.linalg.inv(total[k] @ poses[k]) for k in range(n)]
        edges = _measure(cur_pts, w2c_cur, it)
        if len(edges) < n:
            logger.warning("pose_refine: only %d edges for %d frames — too "
                           "sparse, skipped", len(edges), n)
            return 0
        if it == 0:
            pre = edge_errors(edges, [np.eye(4)] * n)
            h0 = float(np.median(pre))
            logger.info("pose_refine: %d covisibility edges (%d matches) — "
                        "inter-frame surface disagreement median %.1fmm / "
                        "p90 %.1fmm", len(edges),
                        sum(len(e[2]) for e in edges),
                        h0 * 1000, np.percentile(pre, 90) * 1000)
        order = rng.permutation(len(edges))
        n_hold = max(1, int(len(edges) * holdout_frac))
        hold = [edges[t] for t in order[:n_hold]]
        fit = [edges[t] for t in order[n_hold:]]
        Cs_inc = solve_pose_graph(n, fit, odo_weight=odo_weight,
                                  leash_weight=leash_weight)
        total = [Cs_inc[k] @ total[k] for k in range(n)]
        h_it = float(np.median(edge_errors(hold, Cs_inc)))
        logger.info("pose_refine: outer iter %d/%d — held-out %.1fmm", it + 1,
                    outer_iters, h_it * 1000)

    Cs = total
    # GATE on a FRESH measurement at the final placement: "does the cloud now
    # disagree with itself less than it did?" — unbiased by the stale initial
    # association (which carries the very error being corrected)
    cur_pts = {k: (P @ Cs[k][:3, :3].T + Cs[k][:3, 3])
               for k, P in frame_pts.items()}
    w2c_cur = [np.linalg.inv(Cs[k] @ poses[k]) for k in range(n)]
    final_edges = _measure(cur_pts, w2c_cur, outer_iters)
    h1 = (float(np.median(edge_errors(final_edges, [np.eye(4)] * n)))
          if final_edges else float("inf"))
    shifts = np.array([np.linalg.norm(C[:3, 3]) for C in Cs])
    logger.info("pose_refine: fresh inter-frame disagreement %.1fmm -> %.1fmm "
                "| corrections median %.1fmm max %.1fmm", h0 * 1000, h1 * 1000,
                np.median(shifts) * 1000, shifts.max() * 1000)
    accepted = h1 <= h0 * (1.0 - min_gain)
    (output_dir / "pose_refine_report.json").write_text(json.dumps({
        "n_frames": n, "n_edges": len(edges), "grid": [H, W],
        "edge_error_before_m": {"median": float(np.median(pre)),
                                "p90": float(np.percentile(pre, 90))},
        "holdout_before_m": h0, "holdout_after_m": h1,
        "outer_iters": outer_iters,
        "min_gain": min_gain, "accepted": bool(accepted),
        "correction_m": {"median": float(np.median(shifts)),
                         "max": float(shifts.max())},
        "corrections": {str(frames[k]): Cs[k].tolist() for k in range(n)},
    }, indent=1))
    if not accepted:
        logger.warning("pose_refine: held-out gain %.1f%% < %.0f%% required — "
                       "NOTHING APPLIED (identity)",
                       (1 - h1 / max(h0, 1e-12)) * 100, min_gain * 100)
        return 0

    # ── apply: points move with their frame, poses follow ──
    moved = 0
    for p, v, fg in plys:
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        changed = False
        for f in np.unique(fg):
            k = fidx.get(int(f))
            if k is None:
                continue
            C = Cs[k]
            if np.allclose(C, np.eye(4), atol=1e-12):
                continue
            m = fg == f
            xyz[m] = xyz[m] @ C[:3, :3].T + C[:3, 3]
            changed = True
        if changed:
            v["x"] = xyz[:, 0].astype(v["x"].dtype)
            v["y"] = xyz[:, 1].astype(v["y"].dtype)
            v["z"] = xyz[:, 2].astype(v["z"].dtype)
            PlyData([PlyElement.describe(v, "vertex")], text=False,
                    byte_order="<").write(str(p))
    bak = output_dir / "camera_poses.txt.preposerefine"
    if not bak.exists():
        bak.write_text(pp.read_text())
    out_lines = []
    for k, M in enumerate(poses):
        Mn = Cs[k] @ M
        if not np.allclose(Cs[k], np.eye(4), atol=1e-12):
            moved += 1
        out_lines.append(" ".join(f"{x:.9g}" for x in Mn.reshape(-1)))
    pp.write_text("\n".join(out_lines) + "\n")
    logger.info("✅ pose_refine: %d/%d poses refined (backup %s, report "
                "pose_refine_report.json)", moved, n, bak.name)
    return moved


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="E-full global pose refinement")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pair-window", type=int, default=15)
    ap.add_argument("--max-depth", type=float, default=30.0)
    ap.add_argument("--near-ref", type=float, default=8.0)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--rel-tol", type=float, default=0.15)
    ap.add_argument("--odo-weight", type=float, default=2.0)
    ap.add_argument("--leash-weight", type=float, default=0.1)
    ap.add_argument("--min-gain", type=float, default=0.10)
    ap.add_argument("--outer-iters", type=int, default=3)
    a = ap.parse_args()
    run(Path(a.output_dir), pair_window=a.pair_window,
        samples_per_frame=a.samples, rel_tol=a.rel_tol,
        max_depth=a.max_depth, near_ref=a.near_ref,
        odo_weight=a.odo_weight, leash_weight=a.leash_weight,
        min_gain=a.min_gain, outer_iters=a.outer_iters)


if __name__ == "__main__":
    main()
