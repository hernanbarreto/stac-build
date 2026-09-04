"""
Stage 1 — per-segment consolidation: collapse the multi-layer, noisy segment
cloud into a thin surface BEFORE model fitting, so the escalation test sees
shape, not registration bias. The stage-4 fidelity report is always computed
against the ORIGINAL cloud — consolidation only feeds the fitter.

Primary: CGAL WLOP via a satellite process in the CloudComPy310 env
(run_wlop.sh → cgal_wlop.py). WLOP projects, it does not invent: every output
point is a local average of measured points.

Fallback (CGAL env unavailable): robust moving-least-squares projection in
the RIMLS spirit — per-point IRLS plane fit over k neighbours with a Gaussian
residual weight that lets the dominant layer win, then projection onto it.
Fully vectorized (batched eigh), scipy-only.
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


# ── robust MLS (RIMLS-spirit), scipy-only ───────────────────────────

def estimate_oriented_normals(xyz: np.ndarray,
                              cam_centers: Optional[np.ndarray] = None,
                              k: int = 18) -> np.ndarray:
    """Per-point normals ORIENTED toward the nearest camera centre.

    Orientation is what lets consolidation tell a ghost layer of the SAME
    surface (same orientation → collapse) from the two REAL faces of a thin
    wall/panel (opposite orientation → keep apart). Cameras are always inside
    the surveyed space, so nearest-camera orientation is consistent."""
    from scipy.spatial import cKDTree

    # GPU-first (USER ORDER 2026-09-04): after the streaming cleanup deletes
    # _tmp_results_aligned, the trace-normals fast path is gone and THIS
    # fallback runs on scene-scale clouds — the CPU kNN-PCA burned hours on
    # 255M pts while the GPU idled. Grid-hash kNN + batched smallest-eigvec
    # on CUDA; the exact CPU path stays for CUDA-less boxes.
    nrm = None
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            pts64 = np.asarray(xyz, np.float64)
            n = len(pts64)
            nrm = np.zeros((n, 3), np.float64)
            spacing_probe = pts64[:: max(1, n // 5000)]
            t_probe = cKDTree(spacing_probe)
            d_nn, _ = t_probe.query(spacing_probe, k=2, workers=8)
            cell = float(max(np.median(d_nn[:, 1]) * 4.0, 0.01))
            for qidx, idx, ok in _gpu_grid_knn(pts64, k, cell, _torch,
                                               400_000):
                nb = _torch.from_numpy(pts64[idx]).cuda()
                ctr = nb.mean(dim=1, keepdim=True)
                d = nb - ctr
                cov = _torch.einsum("bkj,bkl->bjl", d, d)
                v = _smallest_eigvec3_torch(cov, _torch).cpu().numpy()
                v[~ok] = np.array([0.0, 0.0, 1.0])
                nrm[qidx] = v
                del nb, ctr, d, cov
            _torch.cuda.empty_cache()
    except Exception as _e:  # noqa: BLE001 — CPU path is the safety net
        logger.warning("estimate_oriented_normals: GPU path failed (%s) — "
                       "falling back to CPU kNN-PCA", _e)
        nrm = None
    if nrm is None:
        from reconstruction.geometry.primitives import estimate_normals
        nrm = estimate_normals(xyz, k=k, orient_outward=False)
    if cam_centers is not None and len(cam_centers):
        tree = cKDTree(np.asarray(cam_centers, dtype=np.float64))
        _, ci = tree.query(xyz, workers=-1)
        to_cam = np.asarray(cam_centers)[ci] - xyz
        flip = np.einsum("ij,ij->i", nrm, to_cam) < 0
        nrm[flip] *= -1.0
    else:  # fallback: outward from centroid (rooms scanned from inside)
        ctr = xyz.mean(0)
        flip = np.einsum("ij,ij->i", nrm, xyz - ctr) > 0
        nrm[flip] *= -1.0
    return nrm


def _smallest_eigvec3_torch(A, _torch):
    """Smallest-eigenvalue eigenvector of a batch of symmetric 3x3 matrices,
    CLOSED FORM (trigonometric eigenvalues + row-cross eigenvector) — cusolver's
    batched syev rejects large float64 batches on this stack, and this needs no
    solver at all. (B,3,3) -> (B,3), unit length."""
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
    n01 = (c01 ** 2).sum(1); n02 = (c02 ** 2).sum(1); n12 = (c12 ** 2).sum(1)
    v = _torch.where((n01 >= n02).unsqueeze(1) & (n01 >= n12).unsqueeze(1), c01,
                     _torch.where((n02 >= n12).unsqueeze(1), c02, c12))
    nrm2 = _torch.linalg.norm(v, dim=1, keepdim=True)
    # degenerate (isotropic) neighbourhoods: any unit vector is valid — use +Y
    fallback = _torch.zeros_like(v); fallback[:, 1] = 1.0
    v = _torch.where(nrm2 > 1e-20, v / _torch.clamp(nrm2, min=1e-30), fallback)
    return v


def _gpu_grid_knn(out_np: np.ndarray, k: int, cell_h: float, _torch,
                  block: int, cand_per_cell: int = 64):
    """GPU grid-hash kNN generator, SIZE-ADAPTIVE (USER ORDER 2026-09-04:
    "debe adaptarse a todo, no se escribe para una nube en particular").

    Yields (qidx (B,) global indices, idx (B,k+1) global neighbour indices,
    valid (B,) bool). When the whole cloud fits the VRAM budget it runs
    resident; otherwise it AUTO-TILES: spatial slabs along the longest axis,
    slab bounds snapped to the cell grid, each processed with a one-cell
    HALO so border points keep their true neighbours — identical results,
    memory bounded per tile, any N."""
    free_b, _tot = _torch.cuda.mem_get_info()
    budget = int(free_b * 0.55)
    n = len(out_np)
    need = n * 40 + block * 27 * cand_per_cell * 12
    n_tiles = max(1, int(np.ceil(need / max(budget, 1))))
    if n_tiles == 1:
        yield from _gpu_grid_knn_resident(
            out_np, np.arange(n, dtype=np.int64), None, k, cell_h, _torch,
            block, cand_per_cell)
        return
    ax = int(np.argmax(out_np.max(0) - out_np.min(0)))
    v = out_np[:, ax]
    lo_all = float(v.min())
    qs = np.quantile(v, np.linspace(0, 1, n_tiles + 1))
    # snap tile bounds to the cell grid so a cell never spans two tiles
    qs = lo_all + np.round((qs - lo_all) / cell_h) * cell_h
    qs[0], qs[-1] = -np.inf, np.inf
    logger.info("gpu_knn: %s pts > VRAM budget — %d spatial tiles (axis %d)",
                f"{n:,}", n_tiles, ax)
    for t in range(n_tiles):
        lo, hi = qs[t], qs[t + 1]
        halo = np.flatnonzero((v >= lo - cell_h) & (v < hi + cell_h))
        if len(halo) == 0:
            continue
        core_mask = (v[halo] >= lo) & (v[halo] < hi)
        yield from _gpu_grid_knn_resident(
            out_np[halo], halo, core_mask, k, cell_h, _torch,
            block, cand_per_cell)
        _torch.cuda.empty_cache()


def _gpu_grid_knn_resident(pts_np: np.ndarray, gmap: np.ndarray,
                           core_mask, k: int, cell_h: float, _torch,
                           block: int, cand_per_cell: int = 64):
    """Resident kNN over one tile. ``gmap`` maps local→global indices;
    ``core_mask`` (or None=all) selects which local points to emit."""
    dev = "cuda"
    tp = _torch.from_numpy(pts_np.astype(np.float32)).to(dev)
    n = len(pts_np)
    tg = _torch.from_numpy(gmap).to(dev)
    tcore = (None if core_mask is None
             else _torch.from_numpy(np.asarray(core_mask)).to(dev))
    key = None
    for ax in range(3):
        v = tp[:, ax]
        c = _torch.floor((v - v.min()) / cell_h).to(_torch.int64)
        dim = int(c.max().item()) + 2
        key = c if key is None else key * dim + c
        if ax == 0:
            dims1 = None
        elif ax == 1:
            dims1 = dim
        else:
            dims2 = dim
    order = _torch.argsort(key)
    skey = key[order]
    spts = tp[order]
    uniq, counts = _torch.unique_consecutive(skey, return_counts=True)
    starts = _torch.cumsum(counts, 0) - counts
    offs = [(dx * dims1 + dy) * dims2 + dz
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    C = cand_per_cell
    ar = _torch.arange(C, device=dev)
    qpos = (_torch.arange(n, device=dev) if tcore is None
            else _torch.nonzero(tcore, as_tuple=True)[0])
    nq = len(qpos)
    for b0 in range(0, nq, block):
        cidx = qpos[b0:b0 + block]
        pk = key[cidx]
        px = tp[cidx]
        B = len(cidx)
        cand = _torch.full((B, 27 * C), -1, device=dev, dtype=_torch.int64)
        for oi, off in enumerate(offs):
            pos = _torch.searchsorted(uniq, pk + off).clamp(max=len(uniq) - 1)
            hit = uniq[pos] == pk + off
            st = starts[pos]
            cnt = counts[pos].clamp(max=C)
            take = ar[None, :] < (cnt * hit)[:, None]
            cand[:, oi * C:(oi + 1) * C] = _torch.where(
                take, st[:, None] + ar[None, :], -1)
        val = cand >= 0
        d2 = ((spts[cand.clamp(min=0)] - px[:, None, :]) ** 2).sum(dim=2)
        d2 = _torch.where(val, d2, _torch.inf)
        k_eff = min(k + 1, d2.shape[1])
        top = _torch.topk(d2, k_eff, dim=1, largest=False)
        finite = _torch.isfinite(top.values)
        n_valid = finite.sum(dim=1)
        sel = _torch.gather(cand, 1, top.indices).clamp(min=0)
        idx = tg[order[sel]]                       # local → GLOBAL indices
        idx = _torch.where(finite, idx, tg[cidx][:, None])
        yield (tg[cidx].cpu().numpy(), idx.cpu().numpy(),
               (n_valid >= 6).cpu().numpy())
        del cand, val, d2, top, sel, idx


def consolidate_mls(xyz: np.ndarray, radius: float = 0.06, k: int = 24,
                    iterations: int = 2, max_points: Optional[int] = 600_000,
                    normals: Optional[np.ndarray] = None,
                    normal_gate: float = 0.25,
                    block: int = 1_500_000) -> np.ndarray:
    """Project each point onto a robust local plane of its k neighbours.

    IRLS: per-neighbourhood PCA plane, then Gaussian residual reweighting
    (σ = half the neighbour radius scale) so the locally dominant layer wins
    and the ghost layer's pull fades; 2 passes collapse mm-scale double
    layers. All heavy math is batched numpy (einsum + batched eigh),
    processed in blocks so scene-scale clouds (20M+) fit in memory.

    NORMAL-AWARE mode (``normals`` given): each neighbour is additionally
    weighted by clip(n_i·n_j, 0)² — points on the opposite face of a thin
    structure have opposing oriented normals and get ~zero weight, so the two
    REAL faces consolidate onto themselves instead of collapsing to a
    non-existent mid-surface (the metric-honesty requirement).

    With max_points=None the point count and ORDER are preserved exactly
    (only positions move) — callers may keep index-based references.
    """
    from scipy.spatial import cKDTree

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

    out = pts.copy()
    sigma_r = max(radius / 2.0, 1e-4)

    # GPU inner loop: the per-block math (weighted centres, 3x3 covariances,
    # batched eigh, IRLS reweighting) is what burned ~an hour of CPU on 35M-pt
    # scenes — on the GPU it is seconds per block. The kNN stays on cKDTree
    # (parallel, minutes); only the dense math moves. Falls back to numpy
    # automatically when CUDA is unavailable.
    _torch = None
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            _torch = None
    except Exception:
        _torch = None

    def _project_block_gpu(q, nb, gate):
        tq = _torch.from_numpy(q).cuda()
        tnb = _torch.from_numpy(nb).cuda()
        tw = (_torch.from_numpy(gate).cuda() if gate is not None
              else _torch.ones(tnb.shape[:2], dtype=tq.dtype, device="cuda"))
        tgate = tw.clone() if gate is not None else None
        for _irls in range(3):
            wsum = tw.sum(dim=1, keepdim=True)
            ctr = (tnb * tw.unsqueeze(-1)).sum(dim=1) / wsum
            d = tnb - ctr.unsqueeze(1)
            cov = _torch.einsum("bkj,bkl,bk->bjl", d, d, tw) / wsum.unsqueeze(-1)
            nrm = _smallest_eigvec3_torch(cov, _torch)
            resid = _torch.einsum("bkj,bj->bk", tnb - ctr.unsqueeze(1), nrm)
            w_res = _torch.exp(-0.5 * (resid / sigma_r) ** 2)
            tw = tgate * w_res if tgate is not None else w_res
        h = _torch.einsum("bj,bj->b", tq - ctr, nrm)
        res = (tq - h.unsqueeze(1) * nrm).cpu().numpy()
        del tq, tnb, tw, ctr, d, cov, nrm, resid, w_res
        return res

    for _ in range(int(iterations)):
        if _torch is not None:
            # GPU-first (USER 2026-09-04): grid-hash kNN replaces the
            # cKDTree build+query — the last CPU burn of this stage
            blocks = _gpu_grid_knn(out, k, max(radius / 2.0, 0.02),
                                   _torch, min(block, 400_000))
            for qidx, idx, ok in blocks:
                q = out[qidx]
                nb = out[idx]
                gate = None
                if normals is not None:
                    gate = np.einsum("bj,bkj->bk", normals[qidx],
                                     normals[idx])
                    gate = np.clip(gate, 0.0, None) ** 2
                    gate = np.maximum(gate, 1e-6)
                    if normal_gate > 0:
                        gate[gate < normal_gate ** 2] = 1e-6
                proj = _project_block_gpu(q, nb, gate)
                proj[~ok] = q[~ok]        # sparse spots: keep the measurement
                out[qidx] = proj
            continue
        tree = cKDTree(out)
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n)
            q = out[b0:b1]
            _, idx = tree.query(q, k=k + 1, workers=-1)   # includes self
            nb = out[idx]                                 # (B, k+1, 3)
            gate = None
            if normals is not None:
                gate = np.einsum("bj,bkj->bk", normals[b0:b1], normals[idx])
                gate = np.clip(gate, 0.0, None) ** 2
                gate = np.maximum(gate, 1e-6)             # keep self usable
                if normal_gate > 0:
                    gate[gate < normal_gate ** 2] = 1e-6
            if _torch is not None:
                out[b0:b1] = _project_block_gpu(q, nb, gate)
                continue
            w = gate if gate is not None else np.ones(idx.shape)
            for _irls in range(3):
                wsum = w.sum(axis=1, keepdims=True)
                ctr = (nb * w[..., None]).sum(axis=1) / wsum      # (B,3)
                d = nb - ctr[:, None, :]
                cov = np.einsum("bkj,bkl,bk->bjl", d, d, w) / wsum[..., None]
                _, vecs = np.linalg.eigh(cov)                     # ascending
                nrm = vecs[:, :, 0]                               # (B,3)
                resid = np.einsum("bkj,bj->bk", nb - ctr[:, None, :], nrm)
                w_res = np.exp(-0.5 * (resid / sigma_r) ** 2)
                if gate is not None:
                    w = gate * w_res
                else:
                    w = w_res
            h = np.einsum("bj,bj->b", q - ctr, nrm)
            out[b0:b1] = q - h[:, None] * nrm
    logger.info("consolidate: MLS projected %s pts (r=%.2gm, %d passes%s)",
                f"{n:,}", radius, iterations,
                ", normal-aware" if normals is not None else "")
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
    conservative max (unknown layering)."""
    rep = Path(output_dir) / "fine_register_report.json"
    try:
        seps = json.loads(rep.read_text()).get("sep_after_m", {})
        worst = max(seps.values()) if seps else None
    except Exception:
        worst = None
    if worst is None:
        return float(max_radius_m)
    return float(np.clip(2.0 * worst, min_radius_m, max_radius_m))


def scene_consolidate(output_dir: Path,
                      radius_m: Optional[float] = None,
                      min_radius_m: float = 0.02,
                      max_radius_m: float = 0.06,
                      iterations: int = 2,
                      normal_gate: float = 0.25,
                      k: int = 24) -> Optional[dict]:
    """Stage-1 at SCENE level: consolidate cleaned_cloud.ply IN PLACE with
    normal-aware robust MLS so TSDF masking, Potree, segmentation and every
    fit see the thin surface instead of onion layers.

    - point count and ORDER are preserved (only positions move) → colors and
      segmentation globalIndices stay valid;
    - the untouched measurement is kept as cleaned_cloud_raw.ply — the stage-4
      charter reference (residuals ALWAYS against the original cloud);
    - radius adapts to the fine_register report (adaptive_radius_m).
    """
    output_dir = Path(output_dir)
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        logger.warning("scene_consolidate: no cleaned_cloud.ply — skipping")
        return None
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
        logger.warning("scene_consolidate: unsupported PLY layout — skipping "
                       "(cloud kept untouched)")
        return None
    header, data = loaded
    names = data.dtype.names or ()
    if not {"x", "y", "z"} <= set(names):
        logger.warning("scene_consolidate: no x/y/z properties — skipping")
        return None
    pts = np.column_stack([np.asarray(data["x"], np.float64),
                           np.asarray(data["y"], np.float64),
                           np.asarray(data["z"], np.float64)])
    n = len(pts)
    if n < 1000:
        logger.warning("scene_consolidate: only %d pts — skipping", n)
        return None

    # keep the raw measurement (charter reference) before touching anything
    if not raw_path.exists():
        import shutil
        shutil.copyfile(cloud_path, raw_path)

    cams = _load_camera_centers(output_dir)
    logger.info("scene_consolidate: %s pts, radius=%.3fm (adaptive), "
                "%s camera centres, normal-aware", f"{n:,}", r,
                len(cams) if cams is not None else 0)
    # traced cloud → normals from the per-frame depth gradient (seconds, camera-
    # oriented for free) instead of KDTree-PCA over every point (many minutes).
    normals = None
    if all(k_ in (data.dtype.names or ()) for k_ in
           ("frame_global", "pixel_row", "pixel_col")):
        try:
            from reconstruction.trace_normals import normals_from_trace
            normals = normals_from_trace(
                pts, np.asarray(data["frame_global"], np.int64),
                np.asarray(data["pixel_row"], np.int64),
                np.asarray(data["pixel_col"], np.int64),
                output_dir, log=lambda m: logger.info("scene_consolidate: %s", m))
        except Exception as _e:  # noqa: BLE001
            logger.info("scene_consolidate: trace normals unavailable (%s)", _e)
    if normals is None:
        normals = estimate_oriented_normals(pts, cams)
    moved = consolidate_mls(pts, radius=r, k=k, iterations=iterations,
                            max_points=None, normals=normals,
                            normal_gate=normal_gate)
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
             "mean_move_mm": float(disp.mean() * 1000.0),
             "p95_move_mm": float(np.percentile(disp, 95) * 1000.0),
             "raw_backup": raw_path.name}
    (output_dir / "scene_consolidate_report.json").write_text(
        json.dumps(stats, indent=2))
    logger.info("scene_consolidate: done — mean move %.2fmm, p95 %.2fmm "
                "(raw kept as %s)", stats["mean_move_mm"],
                stats["p95_move_mm"], raw_path.name)
    return stats
