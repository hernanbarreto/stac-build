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


# ── robust MLS fallback (RIMLS-spirit), scipy-only ─────────────────

def consolidate_mls(xyz: np.ndarray, radius: float = 0.06, k: int = 24,
                    iterations: int = 2, max_points: int = 600_000) -> np.ndarray:
    """Project each point onto a robust local plane of its k neighbours.

    IRLS: per-neighbourhood PCA plane, then Gaussian residual reweighting
    (σ = half the neighbour radius scale) so the locally dominant layer wins
    and the ghost layer's pull fades; 2 passes collapse mm-scale double
    layers. All heavy math is batched numpy (einsum + batched eigh).
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(xyz, dtype=np.float64)
    n = len(pts)
    if n < k + 1:
        return pts
    if n > max_points:  # consolidation output feeds the fitter, which
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        pts = pts[sel]  # subsamples anyway — bound the O(n·k) cost
        n = max_points

    out = pts.copy()
    sigma_r = max(radius / 2.0, 1e-4)
    for _ in range(int(iterations)):
        tree = cKDTree(out)
        _, idx = tree.query(out, k=k + 1)          # includes self
        nb = out[idx]                              # (n, k+1, 3)
        w = np.ones(idx.shape)                     # start unweighted
        for _irls in range(3):
            wsum = w.sum(axis=1, keepdims=True)
            ctr = (nb * w[..., None]).sum(axis=1) / wsum         # (n,3)
            d = nb - ctr[:, None, :]
            cov = np.einsum("nkj,nkl,nk->njl", d, d, w) / wsum[..., None]
            _, vecs = np.linalg.eigh(cov)                        # ascending
            nrm = vecs[:, :, 0]                                  # (n,3)
            resid = np.einsum("nkj,nj->nk", nb - ctr[:, None, :], nrm)
            w = np.exp(-0.5 * (resid / sigma_r) ** 2)
        # project each point onto its robust local plane
        h = np.einsum("nj,nj->n", out - ctr, nrm)
        out = out - h[:, None] * nrm
    logger.info("consolidate: MLS projected %d pts (r=%.2gm, %d passes)",
                n, radius, iterations)
    return out
