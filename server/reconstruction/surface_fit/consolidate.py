"""
Stage 1 — per-segment consolidation: collapse the multi-layer, noisy segment
cloud into a thin surface BEFORE model fitting, so the escalation test sees
shape, not registration bias. The stage-4 fidelity report is always computed
against the ORIGINAL cloud — consolidation only feeds the fitter.

Primary: CGAL WLOP via a satellite process in the CloudComPy310 env
(run_wlop.sh → cgal_wlop.py). WLOP projects, it does not invent: every output
point is a local average of measured points.

Fallback (CGAL env unavailable): robust moving-least-squares projection in
the RIMLS spirit — per-point IRLS plane fit over the exact k nearest
neighbours within the radius, with a Gaussian residual weight that lets the
dominant layer win, then projection onto it. torch (CUDA or CPU, one
algorithm), deterministic: see the MLS section and reconstruction.grid_knn.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("SurfaceFit")

_SERVER_DIR = Path(__file__).resolve().parents[2]
_WLOP_LAUNCHER = _SERVER_DIR / "run_wlop.sh"

# PLY property type -> numpy dtype (binary_little_endian)
_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def _read_ply_structured(path: Path):
    """Read a binary_little_endian PLY keeping EVERY property (the cleaned
    cloud carries frame_global/pixel_row/pixel_col/confidence traceability that
    Open3D silently drops — losing it broke Potree and the TSDF cloud mask).
    Returns (header_bytes, structured_array) or None if the layout is not a
    simple vertex-only binary PLY."""
    with open(path, "rb") as f:
        header = b""
        n_pts = 0
        fields: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                return None
            header += line
            s = line.decode("ascii", "replace").strip()
            if s.startswith("format") and "binary_little_endian" not in s:
                return None
            if s.startswith("element"):
                parts = s.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_pts = int(parts[2])
                elif int(parts[2]) > 0:
                    return None  # faces/other elements — not our writer's layout
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                if parts[1] == "list" or parts[1] not in _PLY_TYPES:
                    return None
                fields.append((parts[2], _PLY_TYPES[parts[1]]))
            elif s.startswith("end_header"):
                break
        dtype = np.dtype(fields)
        data = np.fromfile(f, dtype=dtype, count=n_pts)
        if len(data) != n_pts:
            return None
        return header, data


def _write_ply_structured(path: Path, header: bytes, data: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(header)
        data.tofile(f)


def consolidate(xyz: np.ndarray, method: str = "auto",
                neighbor_radius_m: float = 0.06,
                wlop_select_percentage: float = 25.0,
                wlop_iterations: int = 30,
                mls_iterations: int = 2,
                timeout_s: float = 900.0) -> np.ndarray:
    """Consolidate a segment cloud. Returns the thin cloud, or the input
    unchanged when method='none' or everything fails (fitting still works,
    just against the raw layers)."""
    if method == "none":
        return xyz
    if method in ("auto", "wlop"):
        out = consolidate_wlop(xyz, neighbor_radius_m=neighbor_radius_m,
                               select_percentage=wlop_select_percentage,
                               iterations=wlop_iterations, timeout_s=timeout_s)
        if out is not None:
            return out
        if method == "wlop":
            logger.warning("consolidate: WLOP unavailable/failed — falling back to MLS")
    return consolidate_mls(xyz, radius=neighbor_radius_m, iterations=mls_iterations)


# ── WLOP via satellite process ──────────────────────────────────────

def consolidate_wlop(xyz: np.ndarray, neighbor_radius_m: float = 0.06,
                     select_percentage: float = 25.0, iterations: int = 30,
                     timeout_s: float = 900.0,
                     max_points: int = 800_000) -> Optional[np.ndarray]:
    import open3d as o3d
    if not _WLOP_LAUNCHER.exists():
        return None
    xyz = np.asarray(xyz, dtype=np.float64)
    n_in = len(xyz)
    if n_in > max_points:      # WLOP is O(n·k·iter): bound the wall-clock. The
        sel = np.random.default_rng(0).choice(n_in, max_points, replace=False)
        xyz = xyz[sel]         # fitter subsamples to max_fit_points anyway.
        logger.info("consolidate: WLOP input capped %d → %d pts", n_in, max_points)
    with tempfile.TemporaryDirectory(prefix="wlop_") as td:
        inp = Path(td) / "in.ply"
        outp = Path(td) / "out.ply"
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        o3d.io.write_point_cloud(str(inp), pcd, write_ascii=False)
        cmd = ["bash", str(_WLOP_LAUNCHER),
               "--input", str(inp), "--output", str(outp),
               "--select-percentage", str(select_percentage),
               "--neighbor-radius", str(neighbor_radius_m),
               "--iterations", str(iterations)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("consolidate: WLOP subprocess failed: %s", e)
            return None
        result_line = next((l for l in proc.stdout.splitlines()
                            if l.startswith("[WLOP-RESULT]")), "")
        stats_line = next((l for l in proc.stdout.splitlines()
                           if l.startswith("[WLOP]{")), "")
        if proc.returncode != 0 or result_line.endswith("NONE") or not outp.exists():
            logger.warning("consolidate: WLOP failed (rc=%s): %s",
                           proc.returncode, (proc.stderr or proc.stdout)[-400:])
            return None
        out = np.asarray(o3d.io.read_point_cloud(str(outp)).points)
        if len(out) == 0:
            return None
        if stats_line:
            try:
                st = json.loads(stats_line[len("[WLOP]"):])
                logger.info("consolidate: WLOP %d → %d pts (r=%.2gm, %ss)",
                            st["n_in"], st["n_out"], neighbor_radius_m,
                            st["elapsed"])
            except Exception:
                pass
        return out


# ── robust MLS (RIMLS-spirit) on the exact grid kNN ─────────────────
#
# DETERMINISM (2026-09-28): one algorithm on one declared device — the torch
# code below runs identically on CUDA or CPU, the kNN is EXACT
# (reconstruction.grid_knn: whole-cloud grid, one origin, no VRAM-sized tiles,
# no per-cell candidate cap), every per-point sum runs in a fixed order, and an
# MLS pass is a JACOBI update: all projections of a pass read the frozen
# positions of the previous one. The result of a point therefore depends on
# the cloud only — never on how the queries were blocked or on free memory.
# There is no GPU→CPU fallback on an exception: a failure FAILS.

_QUERY_BLOCK = 262144        # BOUND (memory only): queries per kNN block
_CANDIDATE_BUDGET = 1 << 25  # BOUND (memory only): candidate pairs at once


def _rowsum(x):
    """Sum over dim 1 in a FIXED sequential order (elementwise adds) — the same
    bits for a row whatever else is in the batch."""
    s = x[:, 0]
    for j in range(1, x.shape[1]):
        s = s + x[:, j]
    return s


def estimate_oriented_normals(xyz: np.ndarray,
                              cam_centers: Optional[np.ndarray] = None,
                              k: int = 18, device: Optional[str] = None,
                              query_block: int = _QUERY_BLOCK,
                              candidate_budget: int = _CANDIDATE_BUDGET) -> np.ndarray:
    """Per-point normals ORIENTED toward the nearest camera centre.

    Orientation is what lets consolidation tell a ghost layer of the SAME
    surface (same orientation → collapse) from the two REAL faces of a thin
    wall/panel (opposite orientation → keep apart). Cameras are always inside
    the surveyed space, so nearest-camera orientation is consistent.

    Normal = smallest-eigenvalue direction of the point + its EXACT k nearest
    neighbours (float64 PCA, reconstruction.grid_knn) on ``device`` (None =
    CUDA when present, else CPU — the same algorithm on both)."""
    import torch
    from scipy.spatial import cKDTree
    from reconstruction.grid_knn import grid_knn, resolve_device

    dev = resolve_device(device)
    pts64 = np.asarray(xyz, np.float64)
    n = len(pts64)
    nrm = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    if n >= 4:
        kk = int(min(k, n - 1))
        # first-level cell of the exact search: performance only, never the answer
        spacing_probe = pts64[:: max(1, n // 5000)]
        d_nn, _ = cKDTree(spacing_probe).query(spacing_probe, k=2)
        cell = float(max(np.median(d_nn[:, 1]) * 4.0, 0.01))
        tp = torch.from_numpy(pts64).to(dev)
        for q, idx, _d2 in grid_knn(tp, kk, cell, query_block=query_block,
                                    candidate_budget=candidate_budget):
            nb = tp[torch.cat([q[:, None], idx], dim=1)]        # self + k, all found
            c = [_rowsum(nb[..., a]) / (kk + 1) for a in range(3)]
            d = [nb[..., a] - c[a][:, None] for a in range(3)]
            cov = torch.empty((len(q), 3, 3), dtype=torch.float64, device=dev)
            for a in range(3):
                for b in range(a, 3):
                    cov[:, a, b] = cov[:, b, a] = _rowsum(d[a] * d[b])
            nrm[q.cpu().numpy()] = _smallest_eigvec3_torch(cov, torch).cpu().numpy()
        del tp
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    if cam_centers is not None and len(cam_centers):
        tree = cKDTree(np.asarray(cam_centers, dtype=np.float64))
        _, ci = tree.query(xyz, workers=-1)
        to_cam = np.asarray(cam_centers)[ci] - xyz
        flip = np.einsum("ij,ij->i", nrm, to_cam) < 0
        nrm[flip] *= -1.0
    else:  # no cameras: outward from centroid (rooms scanned from inside)
        ctr = pts64.mean(0)
        flip = np.einsum("ij,ij->i", nrm, pts64 - ctr) > 0
        nrm[flip] *= -1.0
    return nrm


def _smallest_eigvec3_torch(A, _torch):
    """Smallest-eigenvalue eigenvector of a batch of symmetric 3x3 matrices,
    CLOSED FORM (trigonometric eigenvalues + row-cross eigenvector) — cusolver's
    batched syev rejects large float64 batches on this stack, and this needs no
    solver at all. (B,3,3) -> (B,3), unit length. Elementwise only (no
    reductions), so a row's result never depends on the batch."""
    a00, a01, a02 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    a11, a12, a22 = A[:, 1, 1], A[:, 1, 2], A[:, 2, 2]
    p1 = a01 ** 2 + a02 ** 2 + a12 ** 2
    q = (a00 + a11 + a22) / 3.0
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = _torch.sqrt(_torch.clamp(p2 / 6.0, min=1e-30))
    # B = (A - qI)/p ; r = det(B)/2 in [-1,1]
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    detB = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = _torch.clamp(detB / 2.0, -1.0, 1.0)
    phi = _torch.acos(r) / 3.0
    lmin = q + 2.0 * p * _torch.cos(phi + 2.0 * _torch.pi / 3.0)
    # eigenvector: null space of (A - lmin I) via the largest cross of two rows
    r0 = _torch.stack([a00 - lmin, a01, a02], dim=1)
    r1 = _torch.stack([a01, a11 - lmin, a12], dim=1)
    r2 = _torch.stack([a02, a12, a22 - lmin], dim=1)
    c01 = _torch.cross(r0, r1, dim=1)
    c02 = _torch.cross(r0, r2, dim=1)
    c12 = _torch.cross(r1, r2, dim=1)

    def _sq(c):
        return c[:, 0] * c[:, 0] + c[:, 1] * c[:, 1] + c[:, 2] * c[:, 2]
    n01, n02, n12 = _sq(c01), _sq(c02), _sq(c12)
    v = _torch.where((n01 >= n02).unsqueeze(1) & (n01 >= n12).unsqueeze(1), c01,
                     _torch.where((n02 >= n12).unsqueeze(1), c02, c12))
    nrm2 = _torch.sqrt(_sq(v)).unsqueeze(1)
    # degenerate (isotropic) neighbourhoods: any unit vector is valid — use +Y
    fallback = _torch.zeros_like(v); fallback[:, 1] = 1.0
    v = _torch.where(nrm2 > 1e-20, v / _torch.clamp(nrm2, min=1e-30), fallback)
    return v


def _project_irls(q, nb, gate, sigma_r: float):
    """IRLS plane through the weighted neighbourhood ``nb`` (B,K,3), then the
    projection of ``q`` (B,3) onto it. ``gate`` (B,K) carries the normal
    agreement and ZERO for the padding of a short neighbourhood. Every sum in
    a fixed order (``_rowsum``)."""
    import torch
    w = gate
    for _irls in range(3):
        wsum = _rowsum(w)
        ctr = [_rowsum(nb[..., a] * w) / wsum for a in range(3)]
        d = [nb[..., a] - ctr[a][:, None] for a in range(3)]
        cov = torch.empty((len(q), 3, 3), dtype=torch.float64, device=q.device)
        for a in range(3):
            for b in range(a, 3):
                cov[:, a, b] = cov[:, b, a] = _rowsum(d[a] * d[b] * w) / wsum
        nrm = _smallest_eigvec3_torch(cov, torch)
        resid = (d[0] * nrm[:, 0:1] + d[1] * nrm[:, 1:2]) + d[2] * nrm[:, 2:3]
        w = gate * torch.exp(-0.5 * (resid / sigma_r) ** 2)
    h = ((q[:, 0] - ctr[0]) * nrm[:, 0] + (q[:, 1] - ctr[1]) * nrm[:, 1]) \
        + (q[:, 2] - ctr[2]) * nrm[:, 2]
    return q - h[:, None] * nrm


def consolidate_mls(xyz: np.ndarray, radius: float = 0.06, k: int = 24,
                    iterations: int = 2, max_points: Optional[int] = 600_000,
                    normals: Optional[np.ndarray] = None,
                    normal_gate: float = 0.25, min_neighbors: int = 5,
                    device: Optional[str] = None,
                    query_block: int = _QUERY_BLOCK,
                    candidate_budget: int = _CANDIDATE_BUDGET) -> np.ndarray:
    """Project each point onto a robust local plane of its neighbourhood: the
    EXACT k nearest neighbours within ``radius`` (plus the point itself).

    IRLS: per-neighbourhood PCA plane, then Gaussian residual reweighting
    (σ = half the radius) so the locally dominant layer wins and the ghost
    layer's pull fades; 2 passes collapse mm-scale double layers. A point with
    fewer than ``min_neighbors`` neighbours within ``radius`` keeps its
    measurement (a plane through fewer is not a surface estimate).

    NORMAL-AWARE mode (``normals`` given): each neighbour is additionally
    weighted by clip(n_i·n_j, 0)² — points on the opposite face of a thin
    structure have opposing oriented normals and get ~zero weight, so the two
    REAL faces consolidate onto themselves instead of collapsing to a
    non-existent mid-surface (the metric-honesty requirement).

    Each pass is a JACOBI update (every projection reads the previous pass's
    frozen positions), so the result does not depend on the order or size of
    the query blocks. ``device``: None = CUDA when present, else CPU — the
    same algorithm on both, chosen once, never on an exception.

    With max_points=None the point count and ORDER are preserved exactly
    (only positions move) — callers may keep index-based references.
    """
    import torch
    from reconstruction.grid_knn import grid_knn, resolve_device

    pts = np.asarray(xyz, dtype=np.float64)
    n = len(pts)
    if n < k + 1:
        return pts
    if max_points is not None and n > max_points:
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        pts = pts[sel]   # segment mode: the fitter subsamples anyway
        if normals is not None:
            normals = normals[sel]
        n = len(pts)

    dev = resolve_device(device)
    sigma_r = max(radius / 2.0, 1e-4)
    cur = torch.from_numpy(np.ascontiguousarray(pts)).to(dev)
    tn = (torch.from_numpy(np.asarray(normals, np.float64)).to(dev)
          if normals is not None else None)
    n_kept = 0
    for _ in range(int(iterations)):
        new = cur.clone()                    # Jacobi: reads stay on `cur`
        n_kept = 0
        for q, idx, _d2 in grid_knn(cur, k, None, radius=radius,
                                    query_block=query_block,
                                    candidate_budget=candidate_budget):
            valid = idx >= 0
            ok = valid.sum(dim=1) >= min_neighbors
            n_kept += int((~ok).sum())       # sparse spots: keep the measurement
            if not bool(ok.any()):
                continue
            q, idx, valid = q[ok], idx[ok], valid[ok]
            nb_idx = torch.cat([q[:, None], idx.clamp(min=0)], dim=1)
            present = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1).double()
            if tn is not None:
                nq, nn = tn[q], tn[nb_idx]
                gate = (nq[:, None, 0] * nn[..., 0] + nq[:, None, 1] * nn[..., 1]) \
                    + nq[:, None, 2] * nn[..., 2]
                gate = torch.clamp(gate, min=0.0) ** 2
                gate = torch.clamp(gate, min=1e-6)               # keep self usable
                if normal_gate > 0:
                    gate = torch.where(gate < normal_gate ** 2,
                                       torch.full_like(gate, 1e-6), gate)
                gate = gate * present
            else:
                gate = present
            new[q] = _project_irls(cur[q], cur[nb_idx], gate, sigma_r)
        cur = new
    out = cur.cpu().numpy()
    del cur, tn
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    logger.info("consolidate: MLS projected %s pts (r=%.2gm, %d passes%s, %s; "
                "%s kept — fewer than %d neighbours within r)",
                f"{n:,}", radius, iterations,
                ", normal-aware" if normals is not None else "", dev,
                f"{n_kept:,}", min_neighbors)
    return out


# ── scene-level consolidation (feeds TSDF / Potree / segmentation) ──

def _load_camera_centers(output_dir: Path) -> Optional[np.ndarray]:
    pp = Path(output_dir) / "camera_poses.txt"
    if not pp.exists():
        return None
    mats = [np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
            for ln in pp.read_text().splitlines() if len(ln.split()) == 16]
    return np.array([M[:3, 3] for M in mats]) if mats else None


def adaptive_radius_m(output_dir: Path, min_radius_m: float = 0.02,
                      max_radius_m: float = 0.06) -> float:
    """Radius driven by stage-0 evidence: 2× the worst residual inter-chunk
    plane separation left by fine_register (its report), clamped. No report →
    conservative max (unknown layering). A report that exists but cannot be
    read FAILS — it never silently becomes "no report"."""
    rep = Path(output_dir) / "fine_register_report.json"
    if not rep.exists():
        return float(max_radius_m)
    seps = json.loads(rep.read_text()).get("sep_after_m", {})
    if not seps:
        return float(max_radius_m)
    return float(np.clip(2.0 * max(seps.values()), min_radius_m, max_radius_m))


def scene_consolidate(output_dir: Path,
                      radius_m: Optional[float] = None,
                      min_radius_m: float = 0.02,
                      max_radius_m: float = 0.06,
                      iterations: int = 2,
                      normal_gate: float = 0.25,
                      k: int = 24,
                      excluded_statuses=None,
                      artifacts_dir: Optional[Path] = None,
                      device: Optional[str] = None,
                      normals_fn=None,
                      min_neighbors: Optional[int] = None,
                      query_block: Optional[int] = None,
                      candidate_budget: Optional[int] = None) -> dict:
    """Stage-1 at SCENE level: consolidate cleaned_cloud.ply IN PLACE with
    normal-aware robust MLS so TSDF masking, Potree, segmentation and every
    fit see the thin surface instead of onion layers.

    - point count and ORDER are preserved (only positions move) → colors and
      segmentation globalIndices stay valid;
    - the untouched measurement is kept as cleaned_cloud_raw.ply — the stage-4
      charter reference (residuals ALWAYS against the original cloud);
    - radius adapts to the fine_register report (adaptive_radius_m);
    - DETERMINISTIC (2026-09-28): exact kNN, Jacobi passes, fixed-order sums,
      on the device postprocessing.scene_consolidate.device declares — the
      same bits for the same cloud whatever the memory bounds or free VRAM.
      Every failure RAISES (no GPU→CPU, no trace→PCA fallback on an
      exception); the only skip is ``enabled: false``, the caller's.

    ``device`` / ``min_neighbors`` / ``query_block`` / ``candidate_budget``
    left None are read from postprocessing.scene_consolidate (a missing key
    fails naming it), and ``excluded_statuses`` left None from
    witness.mls_excluded_statuses (the loops config loader), so every caller —
    pipeline, on-load rebuild, epoch transaction — runs the same configuration.
    """
    from reconstruction.grid_knn import config_section, resolve_device
    cfg = config_section(("postprocessing", "scene_consolidate"),
                         ("device", "min_neighbors", "query_block", "candidate_budget"))
    if excluded_statuses is None:
        # the on-load rebuild and the epoch transaction never passed it, so
        # their MLS moved the single_witness / mask_conflict points the
        # pipeline's leaves alone
        from reconstruction.loops.config import load_loops_config
        excluded_statuses = load_loops_config().witness.mls_excluded_statuses
    dev = resolve_device(device if device is not None else str(cfg["device"]))
    min_nb = int(min_neighbors if min_neighbors is not None else cfg["min_neighbors"])
    knn_kw = {"query_block": int(query_block if query_block is not None
                                 else cfg["query_block"]),
              "candidate_budget": int(candidate_budget if candidate_budget is not None
                                      else cfg["candidate_budget"])}

    output_dir = Path(output_dir)
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        raise FileNotFoundError(f"scene_consolidate: no cleaned_cloud.ply in {output_dir}")
    raw_path = output_dir / "cleaned_cloud_raw.ply"

    r = radius_m if radius_m else adaptive_radius_m(output_dir, min_radius_m,
                                                    max_radius_m)
    # Structure-preserving read: the cleaned cloud carries per-point
    # traceability (frame_global/pixel_row/pixel_col) + confidence that the
    # TSDF mask and Potree DEPEND on. Open3D I/O silently dropped them (and
    # rewrote xyz as float64), which broke both — so we only ever touch the
    # xyz columns and write the file back byte-identical in layout.
    loaded = _read_ply_structured(cloud_path)
    if loaded is None:
        raise ValueError(f"scene_consolidate: {cloud_path} is not a vertex-only "
                         f"binary_little_endian PLY — refusing to guess its layout")
    header, data = loaded
    names = data.dtype.names or ()
    if not {"x", "y", "z"} <= set(names):
        raise ValueError(f"scene_consolidate: {cloud_path} has no x/y/z properties")
    pts = np.column_stack([np.asarray(data["x"], np.float64),
                           np.asarray(data["y"], np.float64),
                           np.asarray(data["z"], np.float64)])
    n = len(pts)

    # keep the raw measurement (charter reference) before touching anything
    if not raw_path.exists():
        import shutil
        shutil.copyfile(cloud_path, raw_path)

    cams = _load_camera_centers(output_dir)
    logger.info("scene_consolidate: %s pts, radius=%.3fm (adaptive), "
                "%s camera centres, normal-aware, %s", f"{n:,}", r,
                len(cams) if cams is not None else 0, dev)
    # traced cloud → normals from the per-frame depth gradient (seconds, camera-
    # oriented for free) instead of kNN-PCA over every point. An EXCEPTION in
    # it fails the stage; only a session that lacks the depth artifacts
    # (normals_from_trace → None, declared in the log and the report) takes
    # the PCA normals.
    normals, normals_source = None, "pca_knn"
    if normals_fn is not None and all(k_ in names for k_ in ("frame_global", "pixel_row", "pixel_col")):
        # the caller's own trace (precision/corrected_cloud: normals from the corrected
        # depth maps, which the session's depth index does not hold)
        normals = normals_fn(pts, np.asarray(data["frame_global"], np.int64),
                             np.asarray(data["pixel_row"], np.int64), np.asarray(data["pixel_col"], np.int64))
        if normals is not None:
            normals_source = "trace"
    elif all(k_ in names for k_ in ("frame_global", "pixel_row", "pixel_col")):
        from reconstruction.trace_normals import normals_from_trace
        # the depth maps, intrinsics and frame list live in the SESSION,
        # not in the staging directory a correction epoch consolidates in
        # (the transaction stages nine geometry artifacts and none of
        # them). Without this the fast path can never fire inside an epoch
        # and every certification pays the kNN-PCA path. The POSES are the
        # ones next to the cloud being consolidated: an epoch stages its
        # warped camera_poses.txt there, and the session's pre-warp rotation
        # would tilt every normal by its keyframe's correction (edge audit
        # 2026-10-01 #10).
        normals = normals_from_trace(
            pts, np.asarray(data["frame_global"], np.int64),
            np.asarray(data["pixel_row"], np.int64),
            np.asarray(data["pixel_col"], np.int64),
            Path(artifacts_dir or output_dir),
            log=lambda m: logger.info("scene_consolidate: %s", m),
            poses_dir=(output_dir if (output_dir / "camera_poses.txt").exists()
                       else None))
        if normals is None:
            logger.warning("scene_consolidate: the session has no depth maps for "
                           "trace normals — PCA normals over the exact kNN")
        else:
            normals_source = "trace"
    if normals is None:
        normals = estimate_oriented_normals(pts, cams, device=dev, **knn_kw)
    # claude_stac.txt §6.3: the MLS never runs on mask_conflict /
    # single_witness points (witness.mls_excluded_statuses) — they neither
    # move nor pull their neighbours; the raw measurement stays theirs
    allowed = np.ones(n, bool)
    n_excluded = 0
    if excluded_statuses and "status" in names:
        from reconstruction.witness.status import status_mask
        allowed = ~status_mask(np.asarray(data["status"]), excluded_statuses)
        n_excluded = int((~allowed).sum())
        logger.info("scene_consolidate: %s pts excluded from the MLS (status in %s)",
                    f"{n_excluded:,}", list(excluded_statuses))
    mls_kw = dict(radius=r, k=k, iterations=iterations, max_points=None,
                  normal_gate=normal_gate, min_neighbors=min_nb, device=dev, **knn_kw)
    if allowed.all():
        moved = consolidate_mls(pts, normals=normals, **mls_kw)
    else:
        moved = pts.copy()
        moved[allowed] = consolidate_mls(pts[allowed], normals=normals[allowed], **mls_kw)
    _bad = ~np.isfinite(moved).all(axis=1)
    if _bad.any():
        raise RuntimeError(
            f"scene_consolidate produced {int(_bad.sum()):,} non-finite positions — "
            f"refusing to write corrupted geometry back (cloud left untouched)")
    disp = np.linalg.norm(moved - pts, axis=1)
    # write ONLY the positions back — every other property (colors, origins,
    # confidence) and the point ORDER stay bit-identical
    out = data.copy()
    out["x"] = moved[:, 0].astype(data.dtype["x"])
    out["y"] = moved[:, 1].astype(data.dtype["y"])
    out["z"] = moved[:, 2].astype(data.dtype["z"])
    _write_ply_structured(cloud_path, header, out)
    stats = {"n_points": int(n), "radius_m": float(r),
             "n_excluded_by_status": int(n_excluded),
             "normals": normals_source, "device": dev,
             "min_neighbors": min_nb,
             "mean_move_mm": float(disp.mean() * 1000.0) if n else 0.0,
             "p95_move_mm": float(np.percentile(disp, 95) * 1000.0) if n else 0.0,
             "raw_backup": raw_path.name}
    (output_dir / "scene_consolidate_report.json").write_text(
        json.dumps(stats, indent=2))
    logger.info("scene_consolidate: done — mean move %.2fmm, p95 %.2fmm "
                "(raw kept as %s)", stats["mean_move_mm"],
                stats["p95_move_mm"], raw_path.name)
    return stats
