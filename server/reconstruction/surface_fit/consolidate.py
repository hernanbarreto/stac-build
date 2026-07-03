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
    for _ in range(int(iterations)):
        tree = cKDTree(out)
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n)
            q = out[b0:b1]
            _, idx = tree.query(q, k=k + 1, workers=-1)   # includes self
            nb = out[idx]                                 # (B, k+1, 3)
            w = np.ones(idx.shape)
            if normals is not None:
                gate = np.einsum("bj,bkj->bk", normals[b0:b1], normals[idx])
                gate = np.clip(gate, 0.0, None) ** 2
                gate = np.maximum(gate, 1e-6)             # keep self usable
                if normal_gate > 0:
                    gate[gate < normal_gate ** 2] = 1e-6
                w = gate
            for _irls in range(3):
                wsum = w.sum(axis=1, keepdims=True)
                ctr = (nb * w[..., None]).sum(axis=1) / wsum      # (B,3)
                d = nb - ctr[:, None, :]
                cov = np.einsum("bkj,bkl,bk->bjl", d, d, w) / wsum[..., None]
                _, vecs = np.linalg.eigh(cov)                     # ascending
                nrm = vecs[:, :, 0]                               # (B,3)
                resid = np.einsum("bkj,bj->bk", nb - ctr[:, None, :], nrm)
                w_res = np.exp(-0.5 * (resid / sigma_r) ** 2)
                if normals is not None:
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
    import open3d as o3d
    output_dir = Path(output_dir)
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        logger.warning("scene_consolidate: no cleaned_cloud.ply — skipping")
        return None
    raw_path = output_dir / "cleaned_cloud_raw.ply"

    r = radius_m if radius_m else adaptive_radius_m(output_dir, min_radius_m,
                                                    max_radius_m)
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    pts = np.asarray(pcd.points)
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
    normals = estimate_oriented_normals(pts, cams)
    moved = consolidate_mls(pts, radius=r, k=k, iterations=iterations,
                            max_points=None, normals=normals,
                            normal_gate=normal_gate)
    disp = np.linalg.norm(moved - pts, axis=1)
    pcd.points = o3d.utility.Vector3dVector(moved)   # colors/order untouched
    o3d.io.write_point_cloud(str(cloud_path), pcd, write_ascii=False,
                             compressed=True)
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
