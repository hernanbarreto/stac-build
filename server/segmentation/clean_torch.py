"""The per-object segment cleaning's face detection and voxel pass on the device (USER 2026-11-01: "todo
debe correr en GPU"; the cleaning switched on the same day, DBSCAN left OFF — "este sí borra puntos,
mantenelo apagado").

MEASURED on pccr (analysis/2026-11-01_projection_gpu/clean_profile_numpy.log, the numpy recipe): floor#10
(3.8 M points) 183 s, 174 s of them inside the function itself — the RANSAC's 500 hypotheses each scored by
one pass over every point, and the 5 cm voxel pass that masks the WHOLE point set once per voxel
(22,069 voxels × 3.8 M points); drywall#982 (970 k) 27 s, opening#317 (465 k) 10.5 s, wall#87 (393 k) 8.2 s.
It runs for every instance of the projection.

Here `segmentation.pipeline._faces_and_voxels_numpy`'s steps on torch tensors, float64, one device:

* RANSAC faces — the SAME hypotheses (numpy's seeded `Generator.choice` draws, in the loop's order), every
  one scored on the device in chunks (|p·n + d| < tol), the first maximal count wins (numpy's strict
  `>`), the PCA refit's covariance on the device and its eigenvectors by numpy's `eigh` (the reference's
  routine) — the same faces, inlier for inlier;
* the parallel-face merge (the reference's Python over the handful of faces);
* the nearest-face assignment of every point (the distance matrix on the device, the first minimum);
* the 5 cm voxel pass: voxel ids by the reference's hash, per-voxel centroids and face-id majorities by
  index sums (integer counts, exact), the PCA planarity test of the residual voxels from per-voxel second
  moments (the reference centres the points first: the two forms differ by float64 rounding, a verdict
  flips only at the exact threshold), the flood fill of residual voxels to a compatible assigned neighbour
  (|n·n'| > 0.8) in the reference's offset order — in SYNCHRONOUS rounds (every residual voxel reads the
  state of the previous round), where the reference sweeps its dict in order and lets a voxel adopt a
  neighbour assigned earlier in the same sweep: when two neighbours of different faces are both
  compatible, the two can adopt differently. DECLARED; the snapped centroids then differ by the distance
  between two compatible planes; tests/test_clean_torch.py holds the two equal where one face is
  compatible.

The DBSCAN step is NOT ported: it is off by the user's order and its sklearn path stays as the reference
runs it (declared in the pipeline).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

HYPOTHESIS_CHUNK = 64          # hypotheses scored per pass over the points (a bound on memory)
OFFSETS_26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
              if (dx, dy, dz) != (0, 0, 0)]


def _t(a, device, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), device=device).to(dtype)


def ransac_faces_t(P: torch.Tensor, ransac_tol: float, min_face_pts: int, max_faces: int,
                   log=print) -> List[tuple]:
    """Sequential RANSAC planes over the points ``P`` (n, 3) on the device: the reference's draws
    (``default_rng(0)``, `choice(len(rem), 3, replace=False)` per hypothesis, `n_iters = min(500, max(50,
    n_rem // 10))`), the first maximal inlier count, the PCA refit, the refined inliers claimed.
    Returns [(normal (3,) numpy, d, inlier indices into P (numpy int64)), …]."""
    dev = P.device
    n = int(P.shape[0])
    remaining = torch.ones(n, dtype=torch.bool, device=dev)
    faces: List[tuple] = []
    rng = np.random.default_rng(0)                   # seeded: one cloud, one set of faces
    for face_i in range(int(max_faces)):
        rem_idx = torch.nonzero(remaining).reshape(-1)
        n_rem = int(rem_idx.shape[0])
        if n_rem < int(min_face_pts):
            break
        rem = P[rem_idx]
        n_iters = min(500, max(50, n_rem // 10))
        S = np.stack([rng.choice(n_rem, 3, replace=False) for _ in range(n_iters)]).astype(np.int64)
        tri = rem[_t(S, dev, torch.int64)]                                 # (iters, 3, 3)
        v1 = tri[:, 1] - tri[:, 0]
        v2 = tri[:, 2] - tri[:, 0]
        nrm = torch.cross(v1, v2, dim=1)
        ln = torch.linalg.norm(nrm, dim=1)
        ok = ln >= 1e-10
        nrm = nrm / torch.where(ok, ln, torch.ones_like(ln))[:, None]
        d = -(nrm * tri[:, 0]).sum(1)
        counts = torch.full((n_iters,), -1, dtype=torch.int64, device=dev)
        for a in range(0, n_iters, HYPOTHESIS_CHUNK):
            b = min(a + HYPOTHESIS_CHUNK, n_iters)
            dist = (rem @ nrm[a:b].T + d[a:b]).abs()                          # (n_rem, c)
            counts[a:b] = (dist < float(ransac_tol)).sum(0)
        counts = torch.where(ok, counts, torch.full_like(counts, -1))
        best = int(torch.argmax(counts))                                       # the first maximum
        best_count = int(counts[best])
        if best_count < int(min_face_pts):
            break
        best_n, best_d = nrm[best], d[best]
        inl = (rem @ best_n + best_d).abs() < float(ransac_tol)
        q = rem[inl]
        cen = q.mean(0)
        Qc = q - cen
        cov = (Qc.T @ Qc / float(q.shape[0])).cpu().numpy()
        _, vecs = np.linalg.eigh(cov)
        rn = vecs[:, 0]
        rn_t = _t(rn, dev)
        rd = -float(rn_t @ cen)
        inl_r = (rem @ rn_t + rd).abs() < float(ransac_tol)
        gidx = rem_idx[inl_r]
        faces.append((np.asarray(rn, np.float64), float(rd), gidx.cpu().numpy().astype(np.int64)))
        remaining[gidx] = False
        log(f"[SegPipeline]     face {face_i}: {int(gidx.shape[0])} pts, "
            f"normal=[{rn[0]:.2f},{rn[1]:.2f},{rn[2]:.2f}]")
    return faces


def merge_parallel_faces(faces: List[tuple], points: np.ndarray, log=print) -> List[tuple]:
    """The reference's step 2b: faces whose normals are near-parallel (|dot| > 0.95) merge into the
    dominant one's plane (most inliers), the union of their inliers."""
    if len(faces) <= 1:
        return faces
    merged = []
    used = set()
    for i in range(len(faces)):
        if i in used:
            continue
        n_i, d_i, idx_i = faces[i]
        group_faces = [(n_i, d_i, idx_i)]
        group_idx = [idx_i]
        for j in range(i + 1, len(faces)):
            if j in used:
                continue
            if abs(float(np.dot(n_i, faces[j][0]))) > 0.95:
                group_faces.append(faces[j])
                group_idx.append(faces[j][2])
                used.add(j)
        combined_idx = np.concatenate(group_idx) if len(group_idx) > 1 else idx_i
        dominant = max(group_faces, key=lambda f: len(f[2]))
        merged_normal = dominant[0].copy()
        dom_centroid = np.mean(points[dominant[2]], axis=0)
        merged_d = -float(np.dot(merged_normal, dom_centroid))
        merged.append((merged_normal, merged_d, combined_idx))
    log(f"[SegPipeline]     merged: {len(faces)} → {len(merged)} faces")
    return merged


def assign_faces_t(P: torch.Tensor, faces: List[tuple], max_assign_dist: float = 0.10) -> np.ndarray:
    """The reference's step 3: every point to the face whose plane is nearest (the first minimum); one
    face → every point; several → only within ``max_assign_dist``. Returns the face id per point (int32,
    -1 unassigned)."""
    n = int(P.shape[0])
    if not faces:
        return np.full(n, -1, dtype=np.int32)
    dev = P.device
    N = _t(np.stack([f[0] for f in faces]), dev)                               # (F, 3)
    D = _t(np.array([f[1] for f in faces]), dev)                                # (F,)
    dists = (P @ N.T + D).abs()                                                # (n, F)
    nearest = torch.argmin(dists, dim=1)                                        # the first minimum
    if len(faces) == 1:
        return np.zeros(n, dtype=np.int32)
    nd = dists[torch.arange(n, device=dev), nearest]
    fid = torch.where(nd <= float(max_assign_dist), nearest, torch.full_like(nearest, -1))
    return fid.cpu().numpy().astype(np.int32)


def voxel_pass_t(P: torch.Tensor, face_id: np.ndarray, faces: List[tuple], planarity_thresh: float,
                 mesh_vs: float = 0.05) -> List[List[float]]:
    """The reference's step 6 on the device: 5 cm voxels of the points (the reference's hashed ids,
    processed in ascending id order), per voxel the centroid and the majority face id (ties → the smallest
    id) or, with no face id inside, the PCA normal when λ₀/λ₂ < ``planarity_thresh``; residual voxels
    adopt a compatible assigned 26-neighbour (|n·n'| > 0.8) in the reference's offset order, in
    synchronous rounds until nothing changes; every voxel's centroid snapped onto its face's plane.
    Returns [[cx, cy, cz, nx, ny, nz], …] in the reference's order (first insertion of each centroid cell)."""
    dev = P.device
    n = int(P.shape[0])
    if n < 5 or not faces:
        return []
    fv_keys = torch.floor(P / float(mesh_vs)).to(torch.int64)
    fv_ids = fv_keys[:, 0] * 1_000_003 + fv_keys[:, 1] * 1_000_033 + fv_keys[:, 2]      # the reference's hash
    uniq, inv, cnt = torch.unique(fv_ids, sorted=True, return_inverse=True, return_counts=True)
    m = int(uniq.shape[0])
    # per-voxel sums: centroid, second moments about the origin (the PCA of residual voxels)
    S1 = torch.zeros((m, 3), dtype=torch.float64, device=dev).index_add_(0, inv, P)
    cen = S1 / cnt[:, None].to(torch.float64)
    outer = (P[:, :, None] * P[:, None, :]).reshape(n, 9)
    S2 = torch.zeros((m, 9), dtype=torch.float64, device=dev).index_add_(0, inv, outer).reshape(m, 3, 3)
    cov = S2 / cnt[:, None, None].to(torch.float64) - cen[:, :, None] * cen[:, None, :]
    # the majority face id per voxel (counts of each id; ties → the smallest id = the first maximum)
    fid = _t(face_id, dev, torch.int64)
    F = len(faces)
    has = fid >= 0
    votes = torch.zeros((m, F), dtype=torch.int64, device=dev)
    if bool(has.any()):
        flat = inv[has] * F + fid[has]
        votes.view(-1).index_add_(0, flat, torch.ones(int(flat.shape[0]), dtype=torch.int64, device=dev))
    n_votes = votes.sum(1)
    maj = torch.argmax(votes, dim=1)
    face_of = torch.where(n_votes > 0, maj, torch.full_like(maj, -1))
    # residual voxels: planar by PCA → their smallest eigenvector; else they leave
    enough = cnt >= 3
    cov_np = cov.cpu().numpy()
    vals = np.linalg.eigvalsh(cov_np)
    planar = (np.maximum(vals[:, 0], 1e-12) / np.maximum(vals[:, 2], 1e-12)) < float(planarity_thresh)
    face_np = face_of.cpu().numpy()
    enough_np = enough.cpu().numpy()
    keep = enough_np & ((face_np >= 0) | planar)
    normals = np.zeros((m, 3))
    res = np.nonzero(enough_np & (face_np < 0) & planar)[0]
    if len(res):
        _, vecs = np.linalg.eigh(cov_np[res])
        normals[res] = vecs[:, :, 0]
    fn = np.stack([f[0] for f in faces])
    fd = np.array([f[1] for f in faces])
    assigned = face_np >= 0
    normals[assigned] = fn[face_np[assigned]]
    # the voxel_info dict of the reference: keyed by the CENTROID's cell, in order of first insertion,
    # the value of the last writer (a hash collision puts two cells' points under one id)
    cen_np = cen.cpu().numpy()
    gk = np.floor(cen_np / float(mesh_vs)).astype(np.int64)
    sel = np.nonzero(keep)[0]                                   # ascending id order = insertion order
    if not len(sel):
        return []
    gk_sel = gk[sel]
    _, first_idx, inverse = np.unique(gk_sel, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    last_of_cell = np.full(int(inverse.max()) + 1, -1, dtype=np.int64)
    last_of_cell[inverse] = np.arange(len(sel))               # ascending → the last writer wins
    order_cells = np.argsort(first_idx, kind="stable")        # first insertion order of the cells
    rows = sel[last_of_cell[order_cells]]                      # one voxel row per cell
    cells = gk_sel[last_of_cell[order_cells]]
    v_face = face_np[rows].copy()
    v_normal = normals[rows].copy()
    v_cen = cen_np[rows].copy()
    # flood fill of residual voxels — synchronous rounds, the reference's offset order (declared above)
    M = len(rows)
    base = cells.min(0)
    span = (cells.max(0) - base + 3).astype(np.int64)
    packed = ((cells[:, 0] - base[0] + 1) * span[1] + (cells[:, 1] - base[1] + 1)) * span[2] + (cells[:, 2] - base[2] + 1)
    order = np.argsort(packed, kind="stable")
    packed_sorted = packed[order]
    nb_idx = np.full((M, 26), -1, dtype=np.int64)
    for oi, (ox, oy, oz) in enumerate(OFFSETS_26):
        nk = packed + (ox * span[1] + oy) * span[2] + oz
        pos = np.searchsorted(packed_sorted, nk)
        okp = pos < M
        pos_c = np.where(okp, pos, 0)
        hit = okp & (packed_sorted[pos_c] == nk)
        nb_idx[hit, oi] = order[pos_c[hit]]
    while True:
        unassigned = np.nonzero(v_face < 0)[0]
        if not len(unassigned):
            break
        nb = nb_idx[unassigned]                                             # (u, 26)
        valid = nb >= 0
        nb_c = np.where(valid, nb, 0)
        nb_face = np.where(valid, v_face[nb_c], -1)
        compat = (nb_face >= 0) & (np.abs(np.einsum("ui,uoi->uo", v_normal[unassigned], v_normal[nb_c])) > 0.8)
        any_c = compat.any(1)
        if not any_c.any():
            break
        first = np.argmax(compat, axis=1)                                   # the first compatible offset
        adopt = unassigned[any_c]
        src = nb_c[any_c, first[any_c]]
        v_face[adopt] = v_face[src]
        v_normal[adopt] = fn[v_face[src]]
    # snap assigned voxels onto their face plane
    a = v_face >= 0
    if a.any():
        nrm = v_normal[a]
        dist = np.einsum("ij,ij->i", v_cen[a], nrm) + fd[v_face[a]]
        v_cen[a] = v_cen[a] - dist[:, None] * nrm
    return [[float(c[0]), float(c[1]), float(c[2]), float(nn[0]), float(nn[1]), float(nn[2])]
            for c, nn in zip(v_cen, v_normal)]


def faces_and_voxels_t(points: np.ndarray, indices: np.ndarray, obj_id: int, params: dict,
                       n_original: int, dbscan_removed: int, device: str, log=print) -> tuple:
    """Steps 2–6 of `segmentation.pipeline._clean_segment_subcloud` on ``device``, the same return
    (indices kept, voxel_data, face_normals_summary, face_planes, face id per point)."""
    P = _t(np.asarray(points, np.float64), device)
    faces = ransac_faces_t(P, float(params["ransac_tolerance"]), int(params["min_face_points"]),
                           int(params["max_faces"]), log=log)
    faces = merge_parallel_faces(faces, np.asarray(points, np.float64), log=log)
    local_face_id = assign_faces_t(P, faces)
    n_assigned = int(np.sum(local_face_id >= 0))
    n_residual = int(np.sum(local_face_id < 0))
    log(f"[SegPipeline]   Clean obj {obj_id}: {n_original:,} → {len(indices):,} pts "
        f"(DBSCAN -{dbscan_removed}, {len(faces)} faces, {n_assigned} assigned, {n_residual} residual)")
    voxel_data = voxel_pass_t(P, local_face_id, faces, float(params["planarity_threshold"]))
    face_normals_summary = [(fn, len(fi)) for fn, fd, fi in faces]
    face_planes = [(fn, fd) for fn, fd, fi in faces]
    return indices, voxel_data, face_normals_summary, face_planes, local_face_id
