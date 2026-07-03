"""
Stage 0 — structure-assisted fine registration between chunks (post-BA).

The two-pass bundle adjustment anchors global poses, but mm→cm inter-chunk
bias survives (the source of TSDF "onion layers"). This stage exploits the
scene's dominant structure: the big planes every construction scene has.

  1. per chunk: RANSAC-extract the large planes (floor slab, walls, deck);
  2. match planes across chunk pairs (normal agreement + offset + overlap);
  3. per chunk: plane-constrained point-to-plane alignment — a linearised
     6-DOF rigid correction minimising the distance of this chunk's matched
     plane points to the reference chunks' planes, Tikhonov-regularised so
     unconstrained DOF (along-plane slides a plane can't see) stay put;
  4. chain corrections outward from the largest chunk, iterate, and measure
     the residual per-pair plane separation.

Acceptance (charter): max separation between layers of the same surface must
end < the TSDF truncation δ, so the layers fuse into ONE surface. The report
says pass/fail per pair; failing pairs mean the BA itself needs attention —
this stage refuses to force corrections beyond ``max_correction_m``.

Runs AFTER reconstruction.reproject_chunks and BEFORE CloudComPy, on the same
contract: ``output/chunk_N.ply`` (+ ``chunk_N_origins.npz``). Chunk points
are rewritten in place and camera_poses.txt gets the same per-chunk rigid
correction (backup: camera_poses.txt.prefinereg) so poses and cloud stay
consistent for the TSDF.

    python -m reconstruction.surface_fit.fine_register --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from reconstruction.geometry.primitives import fit_plane_ransac

logger = logging.getLogger("SurfaceFit")

_EPS = 1e-12


@dataclass
class ChunkPlane:
    normal: np.ndarray          # (3,) unit
    d: float                    # n·x + d = 0
    centroid: np.ndarray        # (3,) inlier centroid
    points: np.ndarray          # (M,3) subsampled inlier points
    n_inliers: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray


@dataclass
class PairMatch:
    chunk_a: int
    chunk_b: int
    plane_a: int
    plane_b: int
    separation_m: float         # |signed offset| between the two planes at overlap


@dataclass
class FineRegisterReport:
    corrections: Dict[int, list] = field(default_factory=dict)   # chunk → 4x4 (list)
    sep_before_m: Dict[str, float] = field(default_factory=dict)  # "a-b" → max sep
    sep_after_m: Dict[str, float] = field(default_factory=dict)
    accept_sep_m: float = 0.024
    accepted: bool = True


# ── plane extraction ────────────────────────────────────────────────

def extract_chunk_planes(xyz: np.ndarray, max_planes: int = 8,
                         dist_thresh: float = 0.015,
                         min_frac_of_chunk: float = 0.04,
                         max_points_per_plane: int = 20_000,
                         subsample: int = 300_000,
                         seed: int = 0) -> List[ChunkPlane]:
    """Greedy multi-RANSAC: fit the dominant plane, peel its inliers, repeat.
    Only LARGE planes survive (≥ min_frac_of_chunk of the chunk) — clutter
    planes would corrupt the registration constraint."""
    pts = np.asarray(xyz, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(pts) > subsample:
        pts = pts[rng.choice(len(pts), subsample, replace=False)]
    n0 = len(pts)
    out: List[ChunkPlane] = []
    remaining = pts
    for _ in range(max_planes):
        if len(remaining) < max(int(min_frac_of_chunk * n0), 500):
            break
        pf = fit_plane_ransac(remaining, dist_thresh=dist_thresh, iters=400,
                              min_inlier_frac=min_frac_of_chunk * n0 / len(remaining),
                              measure_curvature=False)
        if pf is None or pf.inliers.sum() < min_frac_of_chunk * n0:
            break
        q = remaining[pf.inliers]
        sel = q if len(q) <= max_points_per_plane else \
            q[rng.choice(len(q), max_points_per_plane, replace=False)]
        out.append(ChunkPlane(normal=pf.normal, d=float(pf.d),
                              centroid=q.mean(0), points=sel,
                              n_inliers=int(pf.inliers.sum()),
                              bbox_min=q.min(0), bbox_max=q.max(0)))
        remaining = remaining[~pf.inliers]
    return out


# ── plane matching across chunks ────────────────────────────────────

def match_planes(pa: List[ChunkPlane], pb: List[ChunkPlane],
                 max_angle_deg: float = 8.0,
                 max_offset_m: float = 0.12,
                 min_overlap_m: float = 0.3) -> List[Tuple[int, int, float]]:
    """(idx_a, idx_b, separation) for plane pairs that are the same physical
    surface seen from two chunks: near-parallel, small offset, bboxes overlap."""
    cos_min = np.cos(np.deg2rad(max_angle_deg))
    out = []
    for i, A in enumerate(pa):
        for j, B in enumerate(pb):
            c = float(A.normal @ B.normal)
            if abs(c) < cos_min:
                continue
            nB, dB = (B.normal, B.d) if c > 0 else (-B.normal, -B.d)
            # separation measured where both surfaces live: B's centroid vs A's plane
            sep = abs(float(A.normal @ B.centroid + A.d))
            if sep > max_offset_m:
                continue
            lo = np.maximum(A.bbox_min, B.bbox_min)
            hi = np.minimum(A.bbox_max, B.bbox_max)
            span = np.sort(hi - lo)
            if span[1] < min_overlap_m:        # need a real 2-D overlap patch
                continue
            out.append((i, j, sep))
    return out


# ── plane-constrained rigid correction ──────────────────────────────

def solve_p2plane(points: np.ndarray, normals: np.ndarray, dists: np.ndarray,
                  lam: float = 1e-2) -> np.ndarray:
    """One linearised point-to-plane step: find small rigid (ω, t) minimising
    Σ (n·(R x + t) + d)² + λ‖(ω,t)‖². Returns a 4×4. The Tikhonov term keeps
    DOF the plane set cannot observe (in-plane slides/rotations) at identity —
    the constraint honesty the charter asks for (planes as restriction)."""
    x = np.asarray(points)
    nrm = np.asarray(normals)
    J = np.hstack([np.cross(x, nrm), nrm])       # (N,6): d/dω via x×n, d/dt = n
    r = np.asarray(dists)
    H = J.T @ J + lam * len(x) * np.eye(6)
    g = J.T @ r
    try:
        sol = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return np.eye(4)
    w, t = sol[:3], sol[3:]
    th = np.linalg.norm(w)
    R = np.eye(3)
    if th > _EPS:                                 # Rodrigues
        k = w / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _apply_to_plane(pl: ChunkPlane, T: np.ndarray) -> ChunkPlane:
    R, t = T[:3, :3], T[:3, 3]
    n = R @ pl.normal
    c = R @ pl.centroid + t
    p = pl.points @ R.T + t
    return ChunkPlane(normal=n, d=float(-n @ c), centroid=c, points=p,
                      n_inliers=pl.n_inliers, bbox_min=p.min(0),
                      bbox_max=p.max(0))


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < _EPS:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def solve_joint(planes: Dict, anchor, matches_fn=None,
                iters: int = 5, lam: float = 1e-2,
                huber_m: float = 0.02,
                sample_per_match: int = 800) -> Dict:
    """JOINT pose-graph solve over all rigid units at once.

    The old per-chunk sequential (Gauss-Seidel) update could not reconcile
    conflicting pairwise constraints — on an 11-chunk hostile scan it left
    pairs at 10 cm while others reached mm (test2, 2026-07-03). Here every
    matched plane pair contributes point-to-plane residuals with Jacobians on
    BOTH units, and all 6·U unknowns (anchor fixed) are solved together,
    Gauss-Newton with Huber IRLS so a WRONG plane match (repeated parallel
    structure) is down-weighted instead of dragging the whole graph.

    ``planes``: {unit_key: [ChunkPlane,...]}. Returns {unit_key: 4×4}.
    """
    if matches_fn is None:
        matches_fn = match_planes
    keys = list(planes.keys())
    idx = {k: i for i, k in enumerate(keys)}
    U = len(keys)
    corr = {k: np.eye(4) for k in keys}
    rng = np.random.default_rng(0)

    for it in range(iters):
        cur = {k: [_apply_to_plane(p, corr[k]) for p in planes[k]] for k in keys}
        rows_i: List[np.ndarray] = []   # jacobian blocks per residual set
        H = np.zeros((6 * U, 6 * U))
        g = np.zeros(6 * U)
        n_res = 0
        for a_i in range(U):
            for b_i in range(a_i + 1, U):
                a, b = keys[a_i], keys[b_i]
                for i_a, i_b, _sep in matches_fn(cur[a], cur[b]):
                    A = cur[a][i_a]
                    B = cur[b][i_b]
                    P = B.points
                    if len(P) > sample_per_match:
                        P = P[rng.choice(len(P), sample_per_match, replace=False)]
                    n = A.normal
                    r = P @ n + A.d                      # signed dist to plane a
                    # Huber IRLS weight per residual
                    w = np.ones_like(r)
                    big = np.abs(r) > huber_m
                    w[big] = huber_m / np.abs(r[big])
                    # J wrt unit b (moves points): [P×n, n]
                    Jb = np.hstack([np.cross(P, np.broadcast_to(n, P.shape)),
                                    np.broadcast_to(n, P.shape)])
                    # J wrt unit a (moves the plane): opposite sign, same lever
                    Ja = -Jb
                    ia, ib = idx[a] * 6, idx[b] * 6
                    JbW = Jb * w[:, None]
                    JaW = Ja * w[:, None]
                    H[ib:ib + 6, ib:ib + 6] += Jb.T @ JbW
                    H[ia:ia + 6, ia:ia + 6] += Ja.T @ JaW
                    Hab = Ja.T @ JbW
                    H[ia:ia + 6, ib:ib + 6] += Hab
                    H[ib:ib + 6, ia:ia + 6] += Hab.T
                    g[ib:ib + 6] += Jb.T @ (w * r)
                    g[ia:ia + 6] += Ja.T @ (w * r)
                    n_res += len(P)
        if n_res == 0:
            break
        # Tikhonov keeps unobservable DOF still; anchor pinned hard
        H += lam * (n_res / max(U, 1)) * np.eye(6 * U)
        ia = idx[anchor] * 6
        H[ia:ia + 6, :] = 0.0
        H[:, ia:ia + 6] = 0.0
        H[ia:ia + 6, ia:ia + 6] = np.eye(6)
        g[ia:ia + 6] = 0.0
        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = float(np.max(np.abs(dx)))
        for k in keys:
            i = idx[k] * 6
            T = np.eye(4)
            T[:3, :3] = _rodrigues(dx[i:i + 3])
            T[:3, 3] = dx[i + 3:i + 6]
            corr[k] = T @ corr[k]
        if step < 1e-6:
            break
    return corr


def register_chunks(chunks: Dict[int, np.ndarray],
                    accept_sep_m: float = 0.024,
                    max_correction_m: float = 0.12,
                    dist_thresh: float = 0.015,
                    max_planes: int = 8,
                    passes: int = 5,
                    lam: float = 1e-2) -> Tuple[Dict[int, np.ndarray], FineRegisterReport]:
    """Estimate one rigid correction per chunk via the JOINT pose-graph solve
    (largest chunk anchored). Returns ({chunk_id: 4×4}, report). Corrections
    larger than ``max_correction_m`` are refused (identity) — a shift that big
    is a pose problem upstream, and forcing it would fake alignment."""
    ids = sorted(chunks, key=lambda k: -len(chunks[k]))
    anchor = ids[0]
    planes = {k: extract_chunk_planes(chunks[k], max_planes=max_planes,
                                      dist_thresh=dist_thresh, seed=k if isinstance(k, int) else 0)
              for k in ids}
    for k in ids:
        logger.info("fine_register: chunk %s → %d planes (%s pts)", k,
                    len(planes[k]), f"{len(chunks[k]):,}")

    report = FineRegisterReport(accept_sep_m=accept_sep_m)

    # before-metric
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            a, b = ids[a_i], ids[b_i]
            m = match_planes(planes[a], planes[b])
            if m:
                report.sep_before_m[f"{a}-{b}"] = max(s for _, _, s in m)

    corr = solve_joint(planes, anchor, iters=passes, lam=lam)

    # refusal: a correction beyond the cap means the upstream poses are off by
    # more than "residual bias" — zero it and flag, never fake alignment.
    for k in ids:
        shift = float(np.linalg.norm(corr[k][:3, 3]))
        if shift > max_correction_m:
            logger.warning("fine_register: chunk %s correction %.1fmm exceeds "
                           "max %.1fmm — REFUSED (check upstream poses)", k,
                           shift * 1000, max_correction_m * 1000)
            corr[k] = np.eye(4)
            report.accepted = False

    # after-metric
    fixed = {k: [_apply_to_plane(p, corr[k]) for p in planes[k]] for k in ids}
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            a, b = ids[a_i], ids[b_i]
            m = match_planes(fixed[a], fixed[b])
            if m:
                key = f"{a}-{b}"
                report.sep_after_m[key] = max(s for _, _, s in m)
                if report.sep_after_m[key] > accept_sep_m:
                    report.accepted = False

    for k in ids:
        report.corrections[k] = corr[k].tolist()
        mm = float(np.linalg.norm(corr[k][:3, 3])) * 1000
        if mm > 0.01:
            logger.info("fine_register: chunk %s correction |t|=%.2fmm", k, mm)
    worst_b = max(report.sep_before_m.values(), default=0.0)
    worst_a = max(report.sep_after_m.values(), default=0.0)
    logger.info("fine_register: worst plane separation %.2fmm → %.2fmm (δ=%.1fmm) %s",
                worst_b * 1000, worst_a * 1000, accept_sep_m * 1000,
                "OK" if report.accepted else "STILL OVER δ")
    return corr, report


# ── piecewise units: intra-chunk drift ──────────────────────────────

def _split_frames(frames_sorted: np.ndarray, pieces: int) -> List[np.ndarray]:
    """Split a chunk's sorted unique frame ids into ~equal contiguous groups."""
    pieces = max(1, min(pieces, len(frames_sorted)))
    return [g for g in np.array_split(frames_sorted, pieces) if len(g)]


def _interp_transforms(centers: List[float], mats: List[np.ndarray],
                       frames: np.ndarray) -> Dict[int, np.ndarray]:
    """Per-frame rigid correction: linear interpolation (rotvec + translation)
    between the chunk's piece transforms — SMOOTH along the trajectory, so
    splitting a chunk never introduces a hard seam inside it."""
    from scipy.spatial.transform import Rotation
    rvecs = [Rotation.from_matrix(M[:3, :3]).as_rotvec() for M in mats]
    ts = [M[:3, 3] for M in mats]
    out: Dict[int, np.ndarray] = {}
    for f in frames:
        x = float(f)
        if len(centers) == 1 or x <= centers[0]:
            rv, tt = rvecs[0], ts[0]
        elif x >= centers[-1]:
            rv, tt = rvecs[-1], ts[-1]
        else:
            j = int(np.searchsorted(centers, x)) - 1
            a = (x - centers[j]) / max(centers[j + 1] - centers[j], 1e-9)
            rv = (1 - a) * rvecs[j] + a * rvecs[j + 1]
            tt = (1 - a) * ts[j] + a * ts[j + 1]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
        T[:3, 3] = tt
        out[int(f)] = T
    return out


# ── session-level entry (chunk PLYs on disk, reproject_chunks contract) ──

def run(output_dir: Path, accept_sep_m: Optional[float] = None,
        max_correction_m: float = 0.12, dist_thresh: float = 0.015,
        max_planes: int = 8, pieces_per_chunk: int = 3) -> int:
    """Fine-register all backbone chunk_N.ply in place; rewrite poses of the
    frames each chunk owns. Returns the number of corrected chunks.

    Each chunk is split into up to ``pieces_per_chunk`` rigid pieces by frame
    ranges (origins npz): a long chunk drifts INTERNALLY, and no single rigid
    transform can align both of its ends (test2: 11 chunks, joint rigid solve
    still left pairs at 10 cm). The joint solve runs over the pieces; the
    correction applied to points and poses is interpolated PER FRAME between
    piece transforms, so pieces never create seams."""
    from plyfile import PlyData, PlyElement
    output_dir = Path(output_dir)
    if accept_sep_m is None:
        try:
            from config import get_param
            accept_sep_m = 2.0 * float(get_param("tsdf.voxel_length", 0.012))
        except Exception:
            accept_sep_m = 0.024

    chunk_files = [p for p in sorted(output_dir.glob("chunk_*.ply"))
                   if not p.stem.startswith(("chunk_997", "chunk_998", "chunk_999"))]
    if len(chunk_files) < 2:
        logger.info("fine_register: %d backbone chunks — nothing to register",
                    len(chunk_files))
        return 0

    plys: Dict[int, tuple] = {}
    clouds: Dict[int, np.ndarray] = {}
    frame_ids: Dict[int, Optional[np.ndarray]] = {}
    for p in chunk_files:
        cid = int(p.stem.split("_")[1])
        pd = PlyData.read(str(p))
        v = np.array(pd["vertex"].data)
        plys[cid] = (p, v)
        clouds[cid] = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        org = output_dir / f"{p.stem}_origins.npz"
        frame_ids[cid] = (np.load(org)["frame_global"].astype(np.int64)
                          if org.exists() else None)

    # ── build rigid units: (chunk, piece) split by frame ranges ──
    unit_pts: Dict[tuple, np.ndarray] = {}
    unit_frames: Dict[tuple, np.ndarray] = {}   # frame ids belonging to a unit
    for cid in clouds:
        fg = frame_ids[cid]
        if fg is None or pieces_per_chunk <= 1:
            unit_pts[(cid, 0)] = clouds[cid]
            unit_frames[(cid, 0)] = (np.unique(fg) if fg is not None
                                     else np.zeros(0, dtype=np.int64))
            continue
        uniq = np.unique(fg)
        groups = _split_frames(uniq, pieces_per_chunk)
        for j, grp in enumerate(groups):
            mask = np.isin(fg, grp)
            unit_pts[(cid, j)] = clouds[cid][mask]
            unit_frames[(cid, j)] = grp
    logger.info("fine_register: %d chunks → %d rigid units (pieces_per_chunk=%d)",
                len(clouds), len(unit_pts), pieces_per_chunk)

    # ── planes per unit + joint solve ──
    planes = {u: extract_chunk_planes(unit_pts[u], max_planes=max_planes,
                                      dist_thresh=dist_thresh, seed=hash(u) % 2**16)
              for u in unit_pts}
    for u in sorted(planes):
        logger.info("fine_register: unit %s → %d planes (%s pts)", u,
                    len(planes[u]), f"{len(unit_pts[u]):,}")
    anchor = max(unit_pts, key=lambda u: len(unit_pts[u]))
    report = FineRegisterReport(accept_sep_m=accept_sep_m)

    def _pair_metric(pl: Dict) -> Dict[str, float]:
        out: Dict[str, float] = {}
        us = sorted(pl.keys())
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a, b = us[i], us[j]
                if a[0] == b[0]:
                    continue     # same chunk: pieces are continuous by design
                m = match_planes(pl[a], pl[b])
                if m:
                    key = f"{a[0]}-{b[0]}"
                    sep = max(s for _, _, s in m)
                    out[key] = max(out.get(key, 0.0), sep)
        return out

    report.sep_before_m = _pair_metric(planes)
    corr = solve_joint(planes, anchor)

    # refusal per unit: beyond the cap = upstream pose problem, don't fake it
    for u in list(corr.keys()):
        shift = float(np.linalg.norm(corr[u][:3, 3]))
        if shift > max_correction_m:
            logger.warning("fine_register: unit %s correction %.1fmm exceeds max "
                           "%.1fmm — REFUSED", u, shift * 1000, max_correction_m * 1000)
            corr[u] = np.eye(4)
            report.accepted = False

    fixed = {u: [_apply_to_plane(p, corr[u]) for p in planes[u]] for u in planes}
    report.sep_after_m = _pair_metric(fixed)
    for k, v in report.sep_after_m.items():
        if v > accept_sep_m:
            report.accepted = False

    (output_dir / "fine_register_report.json").write_text(json.dumps({
        "corrections": {str(k): v.tolist() for k, v in corr.items()},
        "sep_before_m": report.sep_before_m,
        "sep_after_m": report.sep_after_m,
        "accept_sep_m": report.accept_sep_m,
        "accepted": report.accepted,
        "pieces_per_chunk": pieces_per_chunk,
    }, indent=2))

    # ── apply: per-frame interpolated correction (points + poses) ──
    n = 0
    moved_frames: Dict[int, np.ndarray] = {}
    for cid, (path, v) in plys.items():
        units = sorted(u for u in corr if u[0] == cid)
        mats = [corr[u] for u in units]
        if all(np.allclose(M, np.eye(4), atol=1e-12) for M in mats):
            continue
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        fg = frame_ids[cid]
        if fg is None or len(units) == 1:
            T = mats[0]
            xyz = xyz @ T[:3, :3].T + T[:3, 3]
            if fg is not None:
                for f in np.unique(fg):
                    moved_frames[int(f)] = T
        else:
            centers = [float(np.mean(unit_frames[u])) for u in units]
            all_frames = np.unique(fg)
            per_frame = _interp_transforms(centers, mats, all_frames)
            moved_frames.update(per_frame)
            # vectorized per-frame apply: group points by frame id
            order = np.argsort(fg, kind="stable")
            fg_s = fg[order]
            bounds = np.searchsorted(fg_s, all_frames)
            bounds = np.append(bounds, len(fg_s))
            for i, f in enumerate(all_frames):
                sel = order[bounds[i]:bounds[i + 1]]
                T = per_frame[int(f)]
                xyz[sel] = xyz[sel] @ T[:3, :3].T + T[:3, 3]
        v["x"] = xyz[:, 0].astype(v["x"].dtype)
        v["y"] = xyz[:, 1].astype(v["y"].dtype)
        v["z"] = xyz[:, 2].astype(v["z"].dtype)
        PlyData([PlyElement.describe(v, "vertex")], text=False,
                byte_order="<").write(str(path))
        n += 1

    # poses must move WITH their chunk's points or the TSDF re-introduces the bias
    pp = output_dir / "camera_poses.txt"
    cf = output_dir / "camera_frames.txt"
    if moved_frames and pp.exists() and cf.exists():
        bak = output_dir / "camera_poses.txt.prefinereg"
        if not bak.exists():
            bak.write_text(pp.read_text())
        mats = [np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
                for ln in pp.read_text().splitlines() if len(ln.split()) == 16]
        nums = [int(float(x)) for x in cf.read_text().split()]
        if len(nums) == len(mats):
            out_lines = []
            for f, M in zip(nums, mats):
                T = moved_frames.get(f)
                if T is not None:
                    M = T @ M
                out_lines.append(" ".join(f"{x:.9g}" for x in M.reshape(-1)))
            pp.write_text("\n".join(out_lines) + "\n")
            logger.info("fine_register: updated %d poses (backup %s)",
                        sum(1 for f in nums if f in moved_frames), bak.name)
        else:
            logger.warning("fine_register: poses/frames mismatch — poses NOT updated")
    worst_b = max(report.sep_before_m.values(), default=0.0)
    worst_a = max(report.sep_after_m.values(), default=0.0)
    logger.info("fine_register: worst inter-chunk plane separation %.2fmm → %.2fmm "
                "(δ=%.1fmm) %s", worst_b * 1000, worst_a * 1000,
                accept_sep_m * 1000, "OK" if report.accepted else "STILL OVER δ")
    logger.info("✅ fine_register: corrected %d chunks (report: fine_register_report.json)", n)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage-0 structure-assisted chunk fine registration")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--accept-sep", type=float, default=None,
                    help="acceptance: max residual plane separation (m); default 2×tsdf.voxel_length")
    ap.add_argument("--max-correction", type=float, default=0.12)
    ap.add_argument("--pieces", type=int, default=3,
                    help="rigid pieces per chunk (intra-chunk drift); 1 = rigid chunks")
    a = ap.parse_args()
    run(Path(a.output_dir), accept_sep_m=a.accept_sep,
        max_correction_m=a.max_correction, pieces_per_chunk=a.pieces)


if __name__ == "__main__":
    main()
