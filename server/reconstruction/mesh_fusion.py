#!/usr/bin/env python3
"""
Region fusion of two meshes with THE CLOUD as referee.
======================================================

USER DESIGN (2026-08-17, from his test2/test3 observation): each technique is
strong somewhere else — TSDF (and later PGSR→TSDF) nails complex objects (the
train) wherever it reconstructs at all; the Delaunay-visibility mesh nails
large planes (floors/walls) and total coverage. The fusion rule is OBJECTIVE,
never visual:

  Split the scene into cells (~25 cm). In each cell, each candidate mesh gets
  two measured numbers against the CLOUD (the project's validated truth):
    coverage  = fraction of the cell's cloud points with mesh within tau
    fidelity  = fraction of the cell's mesh surface lying within tau of cloud
  WINNER RULE — A-PRIMARY: A (TSDF/PGSR) wins every cell where it has surface
  genuinely tracking the cloud (coverage ≥ 0.5 and fidelity ≥ 0.4); B
  (Delaunay) fills everywhere else. A score CONTEST was tried first and was
  structurally rigged pro-B (its vertices ARE the cloud points — hugging the
  noise beat averaging it, and B stomped the well-reconstructed train).
  Triangles are assigned to cells by centroid; seams are closed by attracting
  B's boundary vertices onto the A surface.

A per-cell decision report (JSON) makes every choice auditable: who won
where, and by how much. No black boxes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("MeshFusion")


def _cell_keys(pts: np.ndarray, origin: np.ndarray, cell: float) -> np.ndarray:
    q = np.floor((pts - origin) / cell).astype(np.int64)
    return (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)


def _mesh_scene(V: np.ndarray, F: np.ndarray):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)),
                     o3d.core.Tensor(F.astype(np.uint32)))
    return sc


def _tri_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def fuse_meshes(V_a: np.ndarray, F_a: np.ndarray,
                V_b: np.ndarray, F_b: np.ndarray,
                cloud_xyz: np.ndarray,
                cell: float = 0.25, tau: float = 0.025,
                report_path: Optional[Path] = None,
                log=logger.info) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fuse mesh A (objects specialist) and mesh B (coverage specialist).
    Returns (V, F, from_a_mask_per_face, stats). Pure geometry — no I/O except
    the optional JSON report."""
    t0 = time.time()
    import open3d as o3d
    from scipy.spatial import cKDTree

    origin = cloud_xyz.min(0) - 1e-6
    ck = _cell_keys(cloud_xyz, origin, cell)
    cells, cinv = np.unique(ck, return_inverse=True)
    n_cells = len(cells)

    # cloud→mesh distances (coverage evidence), one pass per mesh
    sc_a = _mesh_scene(V_a, F_a)
    sc_b = _mesh_scene(V_b, F_b)
    pc32 = cloud_xyz.astype(np.float32)
    d_a = sc_a.compute_distance(o3d.core.Tensor(pc32)).numpy()
    d_b = sc_b.compute_distance(o3d.core.Tensor(pc32)).numpy()
    cnt = np.bincount(cinv, minlength=n_cells).astype(np.float64)
    cov_a = np.bincount(cinv, weights=(d_a < tau), minlength=n_cells) / cnt
    cov_b = np.bincount(cinv, weights=(d_b < tau), minlength=n_cells) / cnt

    # mesh→cloud fidelity per cell (triangle centroids vs cloud)
    tree = cKDTree(cloud_xyz)

    def _fid_rough(V, F):
        cent = V[F].mean(1)
        dd, _ = tree.query(cent.astype(np.float64), k=1,
                           distance_upper_bound=tau * 4)
        fid_pt = (dd < tau).astype(np.float64)
        key = _cell_keys(cent, origin, cell)
        # map to the cloud's cell table (cells without cloud pts → own key set)
        pos = np.searchsorted(cells, key)
        pos_ok = (pos < n_cells) & (cells[np.clip(pos, 0, n_cells - 1)] == key)
        nrm = _tri_normals(V, F)
        return cent, key, pos, pos_ok, fid_pt, nrm

    cent_a, key_a, pos_a, ok_a, fidpt_a, nrm_a = _fid_rough(V_a, F_a)
    cent_b, key_b, pos_b, ok_b, fidpt_b, nrm_b = _fid_rough(V_b, F_b)

    def _cell_stats(pos, ok, fid_pt, nrm):
        fid = np.zeros(n_cells)
        rough = np.zeros(n_cells)
        n = np.bincount(pos[ok], minlength=n_cells).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            fid = np.bincount(pos[ok], weights=fid_pt[ok], minlength=n_cells) / n
            # normal dispersion: 1 - |mean normal| (0 = flat, →1 = chaos)
            mx = np.bincount(pos[ok], weights=nrm[ok][:, 0], minlength=n_cells) / n
            my = np.bincount(pos[ok], weights=nrm[ok][:, 1], minlength=n_cells) / n
            mz = np.bincount(pos[ok], weights=nrm[ok][:, 2], minlength=n_cells) / n
            rough = 1.0 - np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        fid = np.nan_to_num(fid)
        rough = np.nan_to_num(rough, nan=1.0)
        return fid, rough, n

    fid_a, rough_a, ntr_a = _cell_stats(pos_a, ok_a, fidpt_a, nrm_a)
    fid_b, rough_b, ntr_b = _cell_stats(pos_b, ok_b, fidpt_b, nrm_b)

    # WINNER RULE — A-PRIMARY (user rule, 2026-08-17): "lo que el TSDF
    # reconstruye, lo reconstruye muy bien". The earlier coverage×fidelity
    # CONTEST was structurally rigged pro-B: B's vertices ARE the referee's
    # cloud points, so B scored ~perfect even where it was chewed — hugging
    # the noise beat averaging it, and B stomped the train. Now:
    #   A wins every cell where it HAS surface that genuinely tracks the
    #   cloud (cov_a ≥ a_cov_floor AND fid_a ≥ a_fid_floor — the guards only
    #   reject A where its geometry is truly absent or wrong);
    #   B fills everywhere else. No contest, no noise-hugging advantage.
    a_cov_floor, a_fid_floor = 0.5, 0.4
    score_a = cov_a * fid_a * (ntr_a > 0)          # kept for the REPORT only
    score_b = cov_b * fid_b * (ntr_b > 0)
    a_ok = (ntr_a > 0) & (cov_a >= a_cov_floor) & (fid_a >= a_fid_floor)
    winner = np.where(a_ok, 0, 1).astype(np.int8)   # 0=A, 1=B
    # cells where B has nothing either way: keep A if it at least exists
    winner[(~a_ok) & (ntr_b == 0) & (ntr_a > 0)] = 0

    # triangle assignment by centroid cell; faces in cells unknown to the cloud
    # table keep their own mesh only if the OTHER mesh has nothing there
    keep_a = np.ones(len(F_a), bool)
    keep_b = np.ones(len(F_b), bool)
    keep_a[ok_a] = winner[pos_a[ok_a]] == 0
    keep_b[ok_b] = winner[pos_b[ok_b]] == 1
    # off-table faces (no cloud in their cell): B keeps (coverage specialist),
    # A keeps only where B has no face in the same key
    off_a = ~ok_a
    if off_a.any():
        b_keys = set(key_b[~ok_b].tolist())
        keep_a[off_a] = np.array([k not in b_keys for k in key_a[off_a]])

    # merge
    F_a2 = F_a[keep_a]
    F_b2 = F_b[keep_b]
    V = np.vstack([V_a, V_b])
    F = np.vstack([F_a2, F_b2 + len(V_a)])
    from_a = np.zeros(len(F), bool)
    from_a[: len(F_a2)] = True

    # seam closing: B-side vertices in cells ADJACENT to A-won cells are
    # attracted onto the A surface within 1.5×tau (existing mechanism)
    used_b = np.unique(F_b2.reshape(-1))
    if len(used_b) and keep_a.any():
        Vb_used = V_b[used_b].astype(np.float32)
        cp = sc_a.compute_closest_points(
            o3d.core.Tensor(Vb_used))["points"].numpy()
        d = np.linalg.norm(cp - Vb_used, axis=1)
        w = np.clip(1.0 - (d / (1.5 * tau)) ** 2, 0.0, 1.0)
        Vb_new = (Vb_used.astype(np.float64) * (1 - w[:, None])
                  + cp.astype(np.float64) * w[:, None])
        V[len(V_a) + used_b] = Vb_new
        n_seam = int((w > 0.05).sum())
    else:
        n_seam = 0

    stats = {
        "cells": int(n_cells),
        "cells_a": int((winner == 0).sum()), "cells_b": int((winner == 1).sum()),
        "faces_a_kept": int(keep_a.sum()), "faces_a_total": int(len(F_a)),
        "faces_b_kept": int(keep_b.sum()), "faces_b_total": int(len(F_b)),
        "seam_vertices_welded": n_seam,
        "cell_m": cell, "tau_m": tau,
        "elapsed_s": round(time.time() - t0, 1),
    }
    log(f"[fusion] {stats['cells_a']}/{n_cells} cells → A (objetos), "
        f"{stats['cells_b']}/{n_cells} → B (cobertura) | caras: "
        f"A {stats['faces_a_kept']:,}/{len(F_a):,}, "
        f"B {stats['faces_b_kept']:,}/{len(F_b):,} | "
        f"costura: {n_seam:,} vértices soldados ({stats['elapsed_s']}s)")

    if report_path is not None:
        rep = {
            "rule": "A-PRIMARY: A (TSDF/PGSR) wins every cell where it has "
                    "surface tracking the cloud (cov>=0.5, fid>=0.4); B "
                    "(Delaunay) fills the rest; seams welded onto A",
            "stats": stats,
            "cells": [
                {"cell": int(c), "winner": ("A" if winner[i] == 0 else "B"),
                 "score_a": round(float(score_a[i]), 4),
                 "score_b": round(float(score_b[i]), 4),
                 "cov_a": round(float(cov_a[i]), 3),
                 "cov_b": round(float(cov_b[i]), 3),
                 "fid_a": round(float(fid_a[i]), 3),
                 "fid_b": round(float(fid_b[i]), 3)}
                for i, c in enumerate(cells)
            ],
        }
        Path(report_path).write_text(json.dumps(rep))
    return V, F, from_a, stats
