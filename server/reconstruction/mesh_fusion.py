#!/usr/bin/env python3
"""
Region fusion of two meshes with THE CLOUD as referee.
======================================================

USER DESIGN (2026-08-17, from his test2/test3 observation): each technique is
strong somewhere else — the PGSR→TSDF guide nails complex objects (the train)
wherever it reconstructs at all; the Delaunay-visibility mesh nails large
planes (floors/walls) and total coverage. The fusion rule is OBJECTIVE, never
visual:

  Split the scene into cells (~25 cm). In each cell, each candidate mesh gets
  two measured numbers against the CLOUD (the project's validated truth):
    coverage  = fraction of the cell's cloud points with mesh within tau
    fidelity  = fraction of the cell's mesh surface lying within tau of cloud
  WINNER RULE — A-PRIMARY: A (PGSR→TSDF) wins every cell where it has surface
  genuinely tracking the cloud (coverage ≥ 0.5 and fidelity ≥ 0.4); B
  (Delaunay) fills everywhere else. (A symmetric score CONTEST was tried and
  removed: B's vertices ARE the cloud points, so "closeness to the cloud"
  rewarded hugging the noise and stomped the well-reconstructed objects.)

Hard rules added after test4 (sky + layered-noise verdicts):
  * OFF-TABLE = DROPPED: faces whose cell holds NO cloud points are removed
    from BOTH meshes — where the truth has nothing, no mesh may exist. This
    is what keeps PGSR's sky shells (trained without a sky mask) and any
    floater out of the deliverable.
  * MAJORITY-SMOOTHED winner map: each cell's winner is re-voted by its 6
    neighbours (2 passes) — coherent regions instead of a salt-and-pepper
    patchwork of interleaved surfaces.
  * DEDUP: a kept-B face whose three vertices all lie within τ of the kept-A
    surface is the same surface twice — dropped (kills the layered look).

Seams are welded by attracting B's boundary vertices onto the A surface.
Every decision is auditable in fusion_report.json. No black boxes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("MeshFusion")

_SHIFT = 21                      # exact cell code: 3 × 21 bits (±1M cells/axis)


def _mesh_scene(V: np.ndarray, F: np.ndarray):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3c_tensor(V.astype(np.float32)),
                     o3c_tensor(F.astype(np.uint32)))
    return sc


def o3c_tensor(a):
    import open3d as o3d
    return o3d.core.Tensor(a)


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

    # ── exact integer cell codes (collision-free, decodable → neighbours) ──
    origin = cloud_xyz.min(0) - 1e-6
    q_cloud = np.floor((cloud_xyz - origin) / cell).astype(np.int64)

    def _code(q):
        return (q[:, 0] << (2 * _SHIFT)) | (q[:, 1] << _SHIFT) | q[:, 2]

    codes, cinv = np.unique(_code(q_cloud), return_inverse=True)
    n_cells = len(codes)

    def _cell_pos(q):
        """cell index for integer coords q (−1 = off-table: no cloud there)."""
        inb = np.all((q >= 0) & (q < (1 << _SHIFT)), axis=1)
        c = _code(np.clip(q, 0, (1 << _SHIFT) - 1))
        pos = np.searchsorted(codes, c)
        ok = inb & (pos < n_cells) & (codes[np.clip(pos, 0, n_cells - 1)] == c)
        return np.where(ok, pos, -1)

    # ── cloud→mesh coverage per cell ──
    sc_a = _mesh_scene(V_a, F_a)
    sc_b = _mesh_scene(V_b, F_b)
    pc32 = cloud_xyz.astype(np.float32)
    d_a = sc_a.compute_distance(o3c_tensor(pc32)).numpy()
    d_b = sc_b.compute_distance(o3c_tensor(pc32)).numpy()
    cnt = np.bincount(cinv, minlength=n_cells).astype(np.float64)
    cov_a = np.bincount(cinv, weights=(d_a < tau), minlength=n_cells) / cnt
    cov_b = np.bincount(cinv, weights=(d_b < tau), minlength=n_cells) / cnt

    # ── mesh→cloud fidelity per cell (triangle centroids vs cloud) ──
    tree = cKDTree(cloud_xyz)

    def _face_cells(V, F):
        cent = V[F].mean(1)
        dd, _ = tree.query(cent.astype(np.float64), k=1,
                           distance_upper_bound=tau * 4)
        fid_pt = (dd < tau).astype(np.float64)
        q = np.floor((cent - origin) / cell).astype(np.int64)
        pos = _cell_pos(q)
        return pos, fid_pt

    pos_a, fidpt_a = _face_cells(V_a, F_a)
    pos_b, fidpt_b = _face_cells(V_b, F_b)

    def _cell_fid(pos, fid_pt):
        ok = pos >= 0
        n = np.bincount(pos[ok], minlength=n_cells).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            fid = np.bincount(pos[ok], weights=fid_pt[ok], minlength=n_cells) / n
        return np.nan_to_num(fid), n

    fid_a, ntr_a = _cell_fid(pos_a, fidpt_a)
    fid_b, ntr_b = _cell_fid(pos_b, fidpt_b)

    # ── winner: A-PRIMARY with truth guards ──
    a_cov_floor, a_fid_floor = 0.5, 0.4
    score_a = cov_a * fid_a * (ntr_a > 0)          # kept for the REPORT only
    score_b = cov_b * fid_b * (ntr_b > 0)
    a_ok = (ntr_a > 0) & (cov_a >= a_cov_floor) & (fid_a >= a_fid_floor)
    winner = np.where(a_ok, 0, 1).astype(np.int8)   # 0=A, 1=B
    winner[(~a_ok) & (ntr_b == 0) & (ntr_a > 0)] = 0

    # ── MAJORITY smoothing over the 6-neighbourhood (2 passes): coherent
    # regions, no salt-and-pepper interleaving of the two surfaces ──
    deltas = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                       [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.int64)
    d_codes = (deltas[:, 0] << (2 * _SHIFT)) | (deltas[:, 1] << _SHIFT) | deltas[:, 2]
    n_flip = 0
    for _pass in range(2):
        votes_a = np.zeros(n_cells, np.int16)
        votes_n = np.zeros(n_cells, np.int16)
        for dc in d_codes:
            nb = np.searchsorted(codes, codes + dc)
            okn = (nb < n_cells) & (codes[np.clip(nb, 0, n_cells - 1)] == codes + dc)
            votes_a[okn] += (winner[nb[okn]] == 0)
            votes_n[okn] += 1
        # flip only where a clear neighbour majority disagrees AND the flip
        # target actually has faces there (never vote a mesh into an empty cell)
        maj_a = (votes_n >= 3) & (votes_a * 2 > votes_n + 1)
        maj_b = (votes_n >= 3) & ((votes_n - votes_a) * 2 > votes_n + 1)
        flip_to_a = maj_a & (winner == 1) & (ntr_a > 0)
        flip_to_b = maj_b & (winner == 0) & (ntr_b > 0)
        winner[flip_to_a] = 0
        winner[flip_to_b] = 1
        n_flip += int(flip_to_a.sum() + flip_to_b.sum())

    # ── face assignment. OFF-TABLE (pos<0: cell without cloud points) is
    # DROPPED for BOTH meshes — where the truth has nothing, no mesh exists
    # (this removes PGSR's sky shells and any floater). ──
    keep_a = (pos_a >= 0) & (winner[np.clip(pos_a, 0, n_cells - 1)] == 0)
    keep_b = (pos_b >= 0) & (winner[np.clip(pos_b, 0, n_cells - 1)] == 1)
    n_off_a = int((pos_a < 0).sum())
    n_off_b = int((pos_b < 0).sum())

    # ── DEDUP: kept-B faces fully lying on the kept-A surface are the same
    # surface twice (the layered look) — drop them ──
    n_dedup = 0
    if keep_a.any() and keep_b.any():
        sc_ak = _mesh_scene(V_a, F_a[keep_a])
        vb_idx = F_b[keep_b]                      # (M,3) vertex ids
        d_v = sc_ak.compute_distance(
            o3c_tensor(V_b[np.unique(vb_idx)].astype(np.float32))).numpy()
        near_v = np.zeros(len(V_b), bool)
        near_v[np.unique(vb_idx)] = d_v < tau * 0.8
        dup = near_v[vb_idx].all(axis=1)
        kb = np.flatnonzero(keep_b)
        keep_b[kb[dup]] = False
        n_dedup = int(dup.sum())

    # ── merge ──
    F_a2 = F_a[keep_a]
    F_b2 = F_b[keep_b]
    V = np.vstack([V_a, V_b])
    F = np.vstack([F_a2, F_b2 + len(V_a)])
    from_a = np.zeros(len(F), bool)
    from_a[: len(F_a2)] = True

    # ── seam weld: B vertices near the kept-A surface slide onto it ──
    n_seam = 0
    used_b = np.unique(F_b2.reshape(-1))
    if len(used_b) and keep_a.any():
        sc_ak2 = _mesh_scene(V_a, F_a[keep_a])
        Vb_used = V_b[used_b].astype(np.float32)
        cp = sc_ak2.compute_closest_points(o3c_tensor(Vb_used))["points"].numpy()
        d = np.linalg.norm(cp - Vb_used, axis=1)
        w = np.clip(1.0 - (d / (1.5 * tau)) ** 2, 0.0, 1.0)
        V[len(V_a) + used_b] = (Vb_used.astype(np.float64) * (1 - w[:, None])
                                + cp.astype(np.float64) * w[:, None])
        n_seam = int((w > 0.05).sum())

    stats = {
        "cells": int(n_cells),
        "cells_a": int((winner == 0).sum()), "cells_b": int((winner == 1).sum()),
        "majority_flips": n_flip,
        "faces_a_kept": int(keep_a.sum()), "faces_a_total": int(len(F_a)),
        "faces_b_kept": int(keep_b.sum()), "faces_b_total": int(len(F_b)),
        "faces_offtable_dropped_a": n_off_a,
        "faces_offtable_dropped_b": n_off_b,
        "faces_dedup_dropped_b": n_dedup,
        "seam_vertices_welded": n_seam,
        "cell_m": cell, "tau_m": tau,
        "elapsed_s": round(time.time() - t0, 1),
    }
    log(f"[fusion] {stats['cells_a']}/{n_cells} cells → A (objetos), "
        f"{stats['cells_b']}/{n_cells} → B (cobertura) | mayoría: {n_flip} flips "
        f"| caras: A {stats['faces_a_kept']:,}/{len(F_a):,}, "
        f"B {stats['faces_b_kept']:,}/{len(F_b):,} | fuera-de-tabla: "
        f"A −{n_off_a:,}, B −{n_off_b:,} (cielo/flotantes) | dedup: −{n_dedup:,} "
        f"| costura: {n_seam:,} soldados ({stats['elapsed_s']}s)")

    if report_path is not None:
        rep = {
            "rule": "A-PRIMARY (cov>=0.5, fid>=0.4) + majority-smoothed cells; "
                    "OFF-TABLE dropped (no cloud = no mesh: sky/floaters out); "
                    "B faces duplicating the A surface removed; seams welded",
            "stats": stats,
            "cells": [
                {"cell": int(c), "winner": ("A" if winner[i] == 0 else "B"),
                 "score_a": round(float(score_a[i]), 4),
                 "score_b": round(float(score_b[i]), 4),
                 "cov_a": round(float(cov_a[i]), 3),
                 "cov_b": round(float(cov_b[i]), 3),
                 "fid_a": round(float(fid_a[i]), 3),
                 "fid_b": round(float(fid_b[i]), 3)}
                for i, c in enumerate(codes)
            ],
        }
        Path(report_path).write_text(json.dumps(rep))
    return V, F, from_a, stats
