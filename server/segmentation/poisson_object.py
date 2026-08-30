# STAC-Builder — per-object Poisson meshing from the object's OWN cloud points.
#
# USER DECISION 2026-08-29: the TSDF re-integration inherits every upstream
# error (poses, per-frame depth) and its quality varied wildly across scenes;
# meshing each object straight from the CLEANED CLOUD — the project's validated
# truth — is consistent by construction. Measured on test3: ladder p95 4.9 mm /
# 0% gaps, door 6.3 mm / 0%, wall3 (wall+cone in one piece) 13.2 mm / 0.2%,
# vs the per-object TSDF's 104 mm p95 and 30% missing on the cone.
#
# ENVIRONMENT (load-bearing): Open3D's Poisson thrashes/hangs on this 252-core
# box — TBB ignores OMP_NUM_THREADS. Callers MUST pin CPU affinity (run via
# run_poisson_objects.py, which does os.sched_setaffinity) — verified: hangs
# forever unpinned, 53 s pinned to 8 cores.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


def poisson_from_points(pts: np.ndarray,
                        colors: Optional[np.ndarray],
                        room_center: np.ndarray,
                        depth: Optional[int] = None,
                        density_quantile: float = 0.06,
                        support_crop_m: float = 0.06,
                        cam_positions: Optional[np.ndarray] = None,
                        target_cell_m: float = 0.015,
                        max_depth: int = 11):
    """Screened Poisson over an arbitrary point set (cloud-anchored): normals
    (adaptive radius, oriented per point toward ITS OWN camera when the
    traceability provides one — correct for inside-out rooms AND outside-in
    objects like a train; centroid orientation inverted the train's normals
    and Poisson inflated everything, user 2026-08-29) → Poisson at a depth
    chosen from the object's SIZE (fixed depth 9 gave ~5 cm cells on a 24 m
    train → "gordo") → density trim → support crop → major components →
    vertex colors. Returns (o3d mesh, stats dict) or (None, {})."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # adaptive normal radius: ~3× the median point spacing (fixed 8 cm smeared
    # wheels/cables on dense captures)
    tree = cKDTree(pts)
    sample = pts[:: max(1, len(pts) // 5000)]
    d2, _ = tree.query(sample, k=2)
    spacing = float(np.median(d2[:, 1])) if len(sample) > 10 else 0.01
    n_radius = float(np.clip(3.0 * spacing, 0.02, 0.08))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=n_radius, max_nn=40))
    if cam_positions is not None and len(cam_positions) == len(pts):
        # exact per-point orientation: the surface faces the camera that saw it
        n = np.asarray(pcd.normals)
        to_cam = cam_positions - pts
        flip = np.einsum("ij,ij->i", n, to_cam) < 0
        n[flip] = -n[flip]
        pcd.normals = o3d.utility.Vector3dVector(n)
    else:
        pcd.orient_normals_towards_camera_location(camera_location=room_center)
    # depth from object size: cell ≈ extent·1.1 / 2^depth → pick the depth
    # that reaches ~target_cell_m, clamped for memory
    if depth is None:
        extent = float(np.max(pts.max(0) - pts.min(0)))
        depth = int(np.clip(np.ceil(np.log2(max(extent, 0.5) * 1.1
                                            / max(target_cell_m, 1e-3))),
                            8, max_depth))
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth))
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    v = np.asarray(mesh.vertices)
    if not len(v):
        return None, {}
    tree = cKDTree(pts)
    d, _ = tree.query(v, k=1)
    mesh.remove_vertices_by_mask(d > support_crop_m)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    tri_c, n_c, _ = mesh.cluster_connected_triangles()
    tri_c = np.asarray(tri_c)
    n_c = np.asarray(n_c)
    if len(n_c) > 1:
        keep = n_c >= max(200, 0.05 * int(n_c.max()))
        mesh.remove_triangles_by_mask(~keep[tri_c])
        mesh.remove_unreferenced_vertices()
    v = np.asarray(mesh.vertices)
    if not len(v):
        return None, {}
    if colors is not None and len(colors) == len(pts):
        _, nn = tree.query(v, k=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[nn])
    mesh.compute_vertex_normals()
    d, _ = tree.query(v, k=1)
    d2, _ = cKDTree(v).query(pts[::5], k=1)
    stats = {
        "n_vertices": int(len(v)),
        "n_faces": int(len(mesh.triangles)),
        "mesh_to_points_p95_mm": round(float(np.percentile(d, 95)) * 1000, 1),
        "coverage_gap_frac_5cm": round(float((d2 > 0.05).mean()), 4),
        "depth_used": int(depth),
        "normal_radius_m": round(n_radius, 4),
        "normal_orientation": ("per_point_camera" if cam_positions is not None
                               and len(cam_positions) == len(pts) else "centroid"),
    }
    return mesh, stats


def fill_holes_coherently(mesh, pts: np.ndarray, output_dir: Path,
                          instance_id: int, parts,
                          max_area_no_evidence_m2: float = 0.30,
                          grid_step: float = 0.035):
    """SHAPE-COHERENT, STITCHED hole filling (user 2026-08-29: "repasar la
    malla y rellenar donde debería estar continuo, copiando la forma").

    For every interior boundary loop of the mesh:
      - verdict by image evidence (mask+depth votes over the hole interior):
        covered/occluded → the surface continues → FILL; sees-past majority →
        real opening → LEAVE; no evidence → fill only small holes;
      - the fill COPIES THE LOCAL SHAPE: the loop's own model (a decomposition
        primitive near the ring, else a local plane fit to the ring) gives the
        base surface, and the ring vertices' residuals are IDW-interpolated
        over the interior so the patch passes EXACTLY through the boundary —
        stitched (fill triangles reference the real ring vertices), no seams,
        no floating quads.
    Returns (mesh, n_holes_filled)."""
    import open3d as o3d
    from collections import defaultdict
    from scipy.spatial import Delaunay, cKDTree
    from shapely.geometry import Point, Polygon

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    C = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    if not len(F):
        return mesh, 0

    # ── boundary loops (edges with exactly one face; clean degree-2 chains)
    edges = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    b_edges = uniq[counts == 1]
    nbr = defaultdict(list)
    for a, b in b_edges:
        nbr[int(a)].append(int(b))
        nbr[int(b)].append(int(a))
    used = set()
    loops = []
    for a, b in b_edges:
        a, b = int(a), int(b)
        if (a, b) in used or (b, a) in used:
            continue
        if len(nbr[a]) != 2 or len(nbr[b]) != 2:
            continue
        loop = [a, b]
        used.add((a, b))
        ok = True
        while True:
            cur, prev = loop[-1], loop[-2]
            nxts = [x for x in nbr[cur] if x != prev]
            if len(nxts) != 1 or len(nbr[cur]) != 2:
                ok = False
                break
            nxt = nxts[0]
            used.add((cur, nxt))
            if nxt == loop[0]:
                break
            if nxt in loop[1:]:
                ok = False
                break
            loop.append(nxt)
            if len(loop) > 20000:
                ok = False
                break
        if ok and 6 <= len(loop) <= 20000:
            loops.append(np.asarray(loop, dtype=np.int64))
    if not loops:
        return mesh, 0

    # evidence (may be unavailable — then only small holes fill)
    ev = oid = None
    try:
        from reconstruction.surface_fit.hole_audit import _evidence
        session_dir = Path(output_dir).parent
        e = _evidence(Path(output_dir), session_dir)
        if e.ok:
            sample = V[:: max(1, len(V) // 3000)]
            o = e.calibrate_oid(sample.astype(np.float64), int(instance_id))
            if o is not None:
                ev, oid = e, o
    except Exception as exc:  # noqa: BLE001
        print(f"[Poisson-obj] fill: evidence unavailable ({exc})")

    tree_fc = cKDTree(V[F].mean(axis=1))
    tree_v = cKDTree(V)
    new_v, new_c, new_f = [], [], []
    v_off = len(V)
    n_filled = 0
    for loop in loops[:300]:
        ring = V[loop]
        # local shape: nearest decomposition primitive to the ring, else a
        # local LSQ plane through the ring
        model = None
        best = None
        for kind, m, _mask in parts:
            try:
                d = float(np.median(np.abs(np.asarray(m.signed_distance(ring)))))
            except Exception:  # noqa: BLE001
                continue
            if best is None or d < best[0]:
                best = (d, m)
        if best is not None and best[0] <= 0.08:
            model = best[1]
        if model is None:
            ctr = ring.mean(0)
            Q = ring - ctr
            try:
                n_ = np.linalg.svd(Q, full_matrices=False)[2][2]
            except np.linalg.LinAlgError:
                continue
            b1 = np.linalg.svd(Q, full_matrices=False)[2][0]
            b2 = np.cross(n_, b1)

            class _LocalPlane:
                def to_uv(self, p):
                    q = np.atleast_2d(p) - ctr
                    return np.column_stack([q @ b1, q @ b2])

                def uv_to_world(self, uv):
                    uv = np.atleast_2d(uv)
                    return ctr + uv[:, :1] * b1 + uv[:, 1:2] * b2
            model = _LocalPlane()
        try:
            uv_ring = np.asarray(model.to_uv(ring))
            poly = Polygon(uv_ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            area = float(poly.area)
        except Exception:  # noqa: BLE001
            continue
        if area < 4 * grid_step * grid_step or area > 20.0:
            continue
        # a real HOLE has an empty interior (the outer boundary encloses the
        # whole mesh — skip it)
        try:
            rp = poly.representative_point()
            probe = np.asarray(model.uv_to_world([[rp.x, rp.y]]))[0]
            d_probe, _ = tree_fc.query(probe, k=1)
            if d_probe < 2 * grid_step:
                continue
        except Exception:  # noqa: BLE001
            continue
        # interior sample grid
        minx, miny, maxx, maxy = poly.bounds
        xs = np.arange(minx, maxx + grid_step, grid_step)
        ys = np.arange(miny, maxy + grid_step, grid_step)
        XX, YY = np.meshgrid(xs, ys)
        import shapely
        inside = shapely.contains_xy(poly, XX.ravel(), YY.ravel())
        int_uv = np.column_stack([XX.ravel()[inside], YY.ravel()[inside]])
        # image verdict over the hole interior
        verdict = "no_evidence"
        if ev is not None and len(int_uv):
            probe_w = np.asarray(model.uv_to_world(int_uv[:: max(1, len(int_uv) // 400)]))
            cov, occ, val = ev.vote(probe_w.astype(np.float64), oid)
            seen = val > 0
            if seen.sum() >= 3:
                r_cov = (cov[seen] / val[seen]).mean()
                r_occ = (occ[seen] / val[seen]).mean()
                r_empty = 1.0 - r_cov - r_occ
                if r_empty >= 0.6:
                    verdict = "opening"
                elif (r_cov + r_occ) >= 0.5:
                    verdict = "continuous"
        if verdict == "opening":
            continue
        if verdict == "no_evidence" and area > max_area_no_evidence_m2:
            continue
        # fill COPYING THE SHAPE: base model + IDW of the ring residuals so
        # the patch passes exactly through the boundary
        base_ring = np.asarray(model.uv_to_world(uv_ring))
        resid = ring - base_ring          # (L,3)
        pts2d = np.vstack([uv_ring, int_uv])
        if len(int_uv):
            d_r = np.maximum(
                np.linalg.norm(int_uv[:, None, :] - uv_ring[None, :, :], axis=2),
                1e-6) if len(uv_ring) * len(int_uv) < 4_000_000 else None
            base_int = np.asarray(model.uv_to_world(int_uv))
            if d_r is not None:
                w = 1.0 / d_r ** 2
                w /= w.sum(axis=1, keepdims=True)
                int_w = base_int + w @ resid
            else:
                t_ring = cKDTree(uv_ring)
                dd, jj = t_ring.query(int_uv, k=min(8, len(uv_ring)))
                ww = 1.0 / np.maximum(dd, 1e-6) ** 2
                ww /= ww.sum(axis=1, keepdims=True)
                int_w = base_int + np.einsum("ik,ikj->ij", ww, resid[jj])
        else:
            int_w = np.zeros((0, 3))
        try:
            tri = Delaunay(pts2d)
        except Exception:  # noqa: BLE001
            continue
        cent = pts2d[tri.simplices].mean(axis=1)
        keep_t = shapely.contains_xy(poly, cent[:, 0], cent[:, 1])
        simp = tri.simplices[keep_t]
        if not len(simp):
            continue
        # map: first L indices → existing ring vertices; rest → new vertices
        L = len(uv_ring)
        fmap = np.where(simp < L, loop[np.clip(simp, 0, L - 1)],
                        simp - L + v_off)
        new_f.append(fmap)
        if len(int_w):
            new_v.append(int_w)
            if C is not None:
                _, nn = tree_v.query(int_w, k=1)
                new_c.append(C[nn])
            v_off += len(int_w)
        n_filled += 1
    if not n_filled:
        return mesh, 0
    V2 = np.vstack([V] + new_v) if new_v else V
    F2 = np.vstack([F] + new_f)
    mesh.vertices = o3d.utility.Vector3dVector(V2)
    mesh.triangles = o3d.utility.Vector3iVector(F2.astype(np.int32))
    if C is not None:
        C2 = np.vstack([C] + new_c) if new_c else C
        mesh.vertex_colors = o3d.utility.Vector3dVector(C2)
    print(f"[Poisson-obj] fill: {n_filled} hole(s) filled coherently "
          f"(stitched, shape-copying; openings protected by image evidence)")
    return mesh, n_filled


def _patch_holes_enabled() -> bool:
    """Hole PATCHES are opt-in (config surface_fit.poisson_patch_holes) —
    unstitched patch quads read as floating checkers (user 2026-08-29); they
    return once real stitching exists."""
    try:
        from config import cfg
        return bool((cfg.get("surface_fit") or {}).get("poisson_patch_holes",
                                                       False))
    except Exception:  # noqa: BLE001
        return False


def regularize_poisson_mesh(mesh, pts: np.ndarray, output_dir: Path,
                            instance_id: int,
                            snap_dist_m: float = 0.05,
                            locality_m: float = 0.10,
                            cut_openings: bool = True):
    """REGULARIZED POISSON (user 2026-08-29: "poisson primero, y sobre poisson
    aplicar los ransac y rellenos"). The Poisson mesh provides the complete,
    seamless topology; RANSAC becomes a REGULARIZATION operator on top:

      1. extract the primitives the object contains (plane/cylinder/sphere);
      2. IRON (snap) the mesh vertices of each primitive region onto the
         fitted surface — walls perfectly flat, pipes perfectly round —
         keeping Poisson's connectivity (no seams between regions);
      3. relax the one-ring at region borders (smooth transition);
      4. CUT the faces that Poisson bridged across image-confirmed openings
         (the arch, the doorway) using the mask+depth audit verdicts.

    Returns (mesh, report). Never invents: snapping moves vertices ≤snap_dist
    onto a measured model; cuts follow image evidence; bridges with no
    opening verdict stay."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    report = {"primitives": [], "snapped_vertices": 0, "cut_faces": 0,
              "edge_snapped_vertices": 0, "patched_cells": 0, "mode": None}
    try:
        from reconstruction.surface_fit.decompose import extract_primitives
        from reconstruction.surface_fit.escalate import FITTERS, FitContext
        ctx = FitContext(world_up=np.array([0.0, 1.0, 0.0]),
                         dist_thresh=float(snap_dist_m) * 0.8,
                         ransac_iters=500, min_inlier_frac=0.02)
    except Exception as e:  # noqa: BLE001
        print(f"[Poisson-obj] regularize: surface_fit unavailable ({e})")
        return mesh, report

    # ── 0. SINGLE-SHEET FIRST (user 2026-08-29, after the faceted ceiling:
    # "capas, todo cuadriculado, horrible"): if ONE model — plane, cylinder,
    # sphere or B-SPLINE — explains the whole object, iron everything onto it;
    # fragmenting a curved sheet into piecewise planes creates layered facets.
    # Only objects NO single model explains (the train) are decomposed.
    parts = []
    single_gate = 0.08
    try:
        fit_sub = pts[:: max(1, len(pts) // 150_000)]
        best = None
        for kind in ("plane", "cylinder", "sphere", "bspline"):
            f = FITTERS.get(kind)
            if f is None:
                continue
            try:
                m = f(fit_sub, ctx)
            except Exception:  # noqa: BLE001
                m = None
            if m is None:
                continue
            d = np.abs(np.asarray(m.signed_distance(fit_sub)))
            p95 = float(np.percentile(d, 95))
            if best is None or p95 < best[2]:
                best = (kind, m, p95)
        if best is not None and best[2] <= single_gate:
            kind, m, p95 = best
            mask_all = np.abs(np.asarray(m.signed_distance(pts))) <= snap_dist_m * 2
            parts = [(kind, m, mask_all)]
            report["mode"] = f"single_{kind}"
            print(f"[Poisson-obj] regularize: single-model {kind} explains the "
                  f"object (p95 {p95*1000:.0f}mm) — ironing onto ONE surface")
    except Exception as e:  # noqa: BLE001
        print(f"[Poisson-obj] regularize: single-model test failed ({e})")

    if not parts:
        try:
            parts, rem = extract_primitives(
                pts, ctx, inlier_dist=float(snap_dist_m) * 0.8,
                min_points=max(5000, int(0.02 * len(pts))),
                min_frac_remaining=0.08, max_primitives=10,
                tag=f"regpoisson_{instance_id}")
            parts = list(parts)
            report["mode"] = "decomposed"
        except Exception as e:  # noqa: BLE001
            print(f"[Poisson-obj] regularize: extraction failed ({e}) — "
                  "mesh kept as-is")
            return mesh, report
    if not parts:
        return mesh, report

    # ── STRIP THE BACK SHELL (root fix, user 2026-08-29: "capas sobre capas,
    # agrega volumen"): Poisson closes volumes — every wall/ceiling comes out
    # with a front AND a back skin, and everything built on top (ironing,
    # fills) fought that. Keep ONLY the camera-facing skin: a THIN single
    # sheet with clean hole rings.
    try:
        from scipy.spatial import cKDTree as _KD
        from segmentation.session_io import _load_camera_source
        cam = _load_camera_source(Path(output_dir).parent, Path(output_dir))
        if cam is not None and cam.pose_map:
            cams = np.array([np.asarray(p, np.float64)[:3, 3]
                             for p in cam.pose_map.values()])
            mesh.compute_triangle_normals()
            FN = np.asarray(mesh.triangle_normals)
            F0 = np.asarray(mesh.triangles)
            V0 = np.asarray(mesh.vertices)
            FC0 = V0[F0].mean(axis=1)
            _, ci = _KD(cams).query(FC0, k=1)
            back = np.einsum("ij,ij->i", FN, cams[ci] - FC0) <= 0
            # strip ANY back skin (support-crop already removes most of it
            # where the slab is thick; the leftovers are exactly the local
            # double layers the user sees). Upper guard only: if MOST faces
            # test as back, the winding convention is off — don't butcher.
            if back.any() and back.mean() < 0.70:
                mesh.remove_triangles_by_mask(back)
                mesh.remove_unreferenced_vertices()
                tri_c, n_c, _ = mesh.cluster_connected_triangles()
                tri_c = np.asarray(tri_c)
                n_c = np.asarray(n_c)
                if len(n_c) > 1:
                    keep_c = n_c >= max(150, 0.01 * int(n_c.max()))
                    if (~keep_c).any():
                        mesh.remove_triangles_by_mask(~keep_c[tri_c])
                        mesh.remove_unreferenced_vertices()
                print(f"[Poisson-obj] regularize: stripped back shell — "
                      f"{int(back.sum()):,} faces removed, thin sheet kept")
    except Exception as e:  # noqa: BLE001
        print(f"[Poisson-obj] regularize: shell strip skipped ({e})")

    V = np.asarray(mesh.vertices)
    facing = np.ones(len(V), dtype=bool)
    assigned = np.full(len(V), -1, dtype=np.int64)
    best_d = np.full(len(V), np.inf)
    trees = []
    for k, (kind, model, mask) in enumerate(parts):
        member = pts[mask]
        trees.append(cKDTree(member[:: max(1, len(member) // 60000)]))
        try:
            d_surf = np.abs(np.asarray(model.signed_distance(V)))
        except Exception:  # noqa: BLE001
            continue
        # free-form models iron GENTLY: full projection over 5 cm dragged real
        # local structure onto the spline ("rellenos que deforman la malla
        # correcta", user 2026-08-29). Rigid primitives keep the wider snap —
        # a wall's 5 cm of layered noise SHOULD flatten.
        snap_k = (float(snap_dist_m) if kind in ("plane", "cylinder", "sphere")
                  else min(float(snap_dist_m), 0.03))
        d_loc, _ = trees[k].query(V, k=1)
        ok = (d_surf <= snap_k) & (d_loc <= locality_m) & facing
        upd = ok & (d_surf < best_d)
        assigned[upd] = k
        best_d[upd] = d_surf[upd]

    # 2. iron: project each assigned vertex onto its primitive surface
    V2 = V.copy()
    for k, (kind, model, mask) in enumerate(parts):
        sel = assigned == k
        if not sel.any():
            continue
        try:
            uv = np.asarray(model.to_uv(V[sel]))
            V2[sel] = np.asarray(model.uv_to_world(uv))
        except Exception:  # noqa: BLE001
            assigned[sel] = -1
            continue
        report["primitives"].append({"kind": kind,
                                     "member_points": int(mask.sum()),
                                     "snapped_vertices": int(sel.sum())})
    report["snapped_vertices"] = int((assigned >= 0).sum())

    # 3. relax the transition ring (vertices whose 1-ring mixes regions)
    F = np.asarray(mesh.triangles)
    try:
        import scipy.sparse as sp
        rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2],
                               F[:, 1], F[:, 2], F[:, 0]])
        cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0],
                               F[:, 0], F[:, 1], F[:, 2]])
        A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(V), len(V))).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel()
        fa = assigned[F]
        mixed_face = (fa.max(1) != fa.min(1))
        ring = np.unique(F[mixed_face])
        for _ in range(2):
            nb_mean = A[ring] @ V2 / np.maximum(deg[ring, None], 1)
            V2[ring] = 0.5 * V2[ring] + 0.5 * nb_mean
    except Exception as e:  # noqa: BLE001
        print(f"[Poisson-obj] regularize: border relax skipped ({e})")
    mesh.vertices = o3d.utility.Vector3dVector(V2)

    # 4. openings + holes on the ironed plane regions, image-audited:
    #    - CUT the faces Poisson bridged over image-confirmed openings,
    #      snapping the cut border onto the IDEAL outline (shape ladder) so
    #      the edge is CAD-clean, not "bitten" (user 2026-08-29);
    #    - PATCH the audit-approved cells (support + evidence fills) that the
    #      Poisson mesh left uncovered ("precisar los huecos").
    if cut_openings:
        try:
            from reconstruction.surface_fit.hole_audit import audit_and_fill
            from reconstruction.surface_fit.contours import ideal_open_outlines
            session_dir = Path(output_dir).parent
            FC = V2[F].mean(axis=1)
            drop = np.zeros(len(F), dtype=bool)
            region_evidence = []   # (model, keepg, openg, gu0, gv0, gres, outlines)
            for k, (kind, model, mask) in enumerate(parts):
                if kind != "plane":
                    continue
                uv_sup = np.asarray(model.to_uv(pts[mask]))
                r = audit_and_fill(model, uv_sup, output_dir, session_dir,
                                   int(instance_id), resolution=0.05)
                if r is None:
                    continue
                keepg, openg, gu0, gv0, gres = r[2].get("_grid", (None,) * 5)
                if keepg is None:
                    continue
                outlines = []
                if openg is not None and openg.any():
                    try:
                        outlines = ideal_open_outlines(model, openg, gu0, gv0, gres)
                    except Exception as e:  # noqa: BLE001
                        print(f"[Poisson-obj] regularize: outline fit skipped ({e})")
                region_evidence.append((model, keepg, openg, gu0, gv0, gres,
                                        outlines))
                if openg is None or not openg.any():
                    continue
                # cut candidates by SURFACE distance, not by assignment: the
                # bridge faces over an opening have no measured points nearby
                # (that's why Poisson bridged them) — the open cell already
                # encodes the position.
                try:
                    dfc = np.abs(np.asarray(model.signed_distance(FC)))
                except Exception:  # noqa: BLE001
                    continue
                # a face may be cut by plane k only if it does NOT belong to a
                # DIFFERENT region (user 2026-08-29: distance-only selection
                # let secondary planes cut the main wall — 2,377 cuts, seams
                # and floating shreds). Bridges belong to no region → cuttable.
                fa = assigned[F]
                belongs_other = ((fa >= 0) & (fa != k)).any(axis=1)
                face_sel = (dfc <= max(float(snap_dist_m), 0.08)) & ~belongs_other
                if not face_sel.any():
                    continue
                uv_fc = np.asarray(model.to_uv(FC[face_sel]))
                ci = np.floor((uv_fc[:, 0] - gu0) / gres).astype(np.int64)
                cj = np.floor((uv_fc[:, 1] - gv0) / gres).astype(np.int64)
                inb = ((ci >= 0) & (ci < openg.shape[1])
                       & (cj >= 0) & (cj < openg.shape[0]))
                hit = np.zeros(int(face_sel.sum()), dtype=bool)
                hit[inb] = openg[cj[inb], ci[inb]]
                idxs = np.nonzero(face_sel)[0][hit]
                drop[idxs] = True
            if drop.any():
                # snap the cut border onto the ideal outlines BEFORE removal
                border = np.intersect1d(np.unique(F[drop]), np.unique(F[~drop]))
                n_snapped = 0
                for model, _kg, _og, _u, _v, _g, outlines in region_evidence:
                    if not outlines:
                        continue
                    O = np.vstack(outlines)
                    tree_o = cKDTree(O)
                    try:
                        db = np.abs(np.asarray(model.signed_distance(V2[border])))
                    except Exception:  # noqa: BLE001
                        continue
                    near = db <= max(float(snap_dist_m), 0.08)
                    if not near.any():
                        continue
                    bidx = border[near]
                    uvb = np.asarray(model.to_uv(V2[bidx]))
                    d_o, j_o = tree_o.query(uvb, k=1)
                    move = d_o <= 0.10
                    if move.any():
                        uvb[move] = O[j_o[move]]
                        V2[bidx[move]] = np.asarray(
                            model.uv_to_world(uvb[move]))
                        n_snapped += int(move.sum())
                report["edge_snapped_vertices"] = n_snapped
                mesh.vertices = o3d.utility.Vector3dVector(V2)
                report["cut_faces"] = int(drop.sum())
                mesh.remove_triangles_by_mask(drop)
                mesh.remove_unreferenced_vertices()
                # cutting can leave tiny disconnected shreds — drop them
                tri_c, n_c, _ = mesh.cluster_connected_triangles()
                tri_c = np.asarray(tri_c)
                n_c = np.asarray(n_c)
                if len(n_c) > 1:
                    keep_c = n_c >= max(200, 0.01 * int(n_c.max()))
                    shred = ~keep_c[tri_c]
                    if shred.any():
                        mesh.remove_triangles_by_mask(shred)
                        mesh.remove_unreferenced_vertices()
                        print(f"[Poisson-obj] regularize: dropped "
                              f"{int(shred.sum()):,} shred faces "
                              f"({int((~keep_c).sum())} tiny components)")
                print(f"[Poisson-obj] regularize: cut {int(drop.sum()):,} "
                      f"bridge faces over image-confirmed openings "
                      f"({n_snapped:,} border verts snapped to ideal outlines)")
            # PATCH audit-approved cells with no mesh coverage
            n_patch = 0
            V3 = np.asarray(mesh.vertices)
            F3 = np.asarray(mesh.triangles)
            C3 = (np.asarray(mesh.vertex_colors)
                  if mesh.has_vertex_colors() else None)
            new_v, new_f = [], []
            v_off = len(V3)
            tree_all_fc = (cKDTree(V3[F3].mean(axis=1))
                           if len(F3) else None)
            for model, keepg, _og, gu0, gv0, gres, _ol in (
                    region_evidence if _patch_holes_enabled() else []):
                try:
                    dfc3 = np.abs(np.asarray(
                        model.signed_distance(V3[F3].mean(axis=1))))
                except Exception:  # noqa: BLE001
                    continue
                fc_near = V3[F3].mean(axis=1)[dfc3 <= max(float(snap_dist_m), 0.08)]
                if not len(fc_near):
                    continue
                tree_fc = cKDTree(fc_near)
                jj2, ii2 = np.nonzero(keepg)
                cuv = np.column_stack([gu0 + (ii2 + 0.5) * gres,
                                       gv0 + (jj2 + 0.5) * gres])
                cw = np.asarray(model.uv_to_world(cuv))
                d_m, _ = tree_fc.query(cw, k=1)
                uncovered = d_m > 1.6 * gres
                if tree_all_fc is not None:
                    # no layering: the cell must be uncovered by the WHOLE
                    # mesh, not just by this region's faces
                    d_all, _ = tree_all_fc.query(cw, k=1)
                    uncovered &= d_all > 1.6 * gres
                if not uncovered.any():
                    continue
                # only SMALL uncovered clusters get patched (user 2026-08-29:
                # big flat patch sheets looked layered/checkered) — large
                # uncovered areas are Poisson's honest absence, leave them
                from scipy import ndimage as _ndi
                ug = np.zeros_like(keepg)
                ug[jj2[uncovered], ii2[uncovered]] = True
                lab_u, n_u = _ndi.label(ug)
                if n_u:
                    sizes = np.bincount(lab_u.ravel())
                    ok_ids = np.nonzero(sizes <= 40)[0]
                    ok_ids = ok_ids[ok_ids > 0]
                    small = np.isin(lab_u, ok_ids) & ug
                    uncovered &= small[jj2, ii2]
                if not uncovered.any():
                    continue
                # a fill must be CORRECT (user 2026-08-29): never patch inside
                # an image-confirmed OPENING of ANY region — a secondary
                # plane's patches were re-covering wall3's arch.
                cw_unc = cw[uncovered]
                bad_any = np.zeros(len(cw_unc), dtype=bool)
                for (m_o, _kg_o, og_o, u_o, v_o, g_o, _ol_o) in region_evidence:
                    if og_o is None or not og_o.any():
                        continue
                    try:
                        do = np.abs(np.asarray(m_o.signed_distance(cw_unc)))
                        uvo = np.asarray(m_o.to_uv(cw_unc))
                    except Exception:  # noqa: BLE001
                        continue
                    cio = np.floor((uvo[:, 0] - u_o) / g_o).astype(np.int64)
                    cjo = np.floor((uvo[:, 1] - v_o) / g_o).astype(np.int64)
                    inb = ((cio >= 0) & (cio < og_o.shape[1])
                           & (cjo >= 0) & (cjo < og_o.shape[0]))
                    hito = np.zeros(len(cw_unc), dtype=bool)
                    hito[inb] = og_o[cjo[inb], cio[inb]]
                    bad_any |= hito & (do <= 0.10)
                if bad_any.any():
                    idx_unc = np.nonzero(uncovered)[0]
                    uncovered[idx_unc[bad_any]] = False
                if not uncovered.any():
                    continue
                for uc, wc in zip(cuv[uncovered], cw[uncovered]):
                    quad_uv = np.array([
                        [uc[0] - gres / 2, uc[1] - gres / 2],
                        [uc[0] + gres / 2, uc[1] - gres / 2],
                        [uc[0] + gres / 2, uc[1] + gres / 2],
                        [uc[0] - gres / 2, uc[1] + gres / 2]])
                    qw = np.asarray(model.uv_to_world(quad_uv))
                    new_v.append(qw)
                    new_f.append(np.array([[v_off, v_off + 1, v_off + 2],
                                           [v_off, v_off + 2, v_off + 3]]))
                    v_off += 4
                n_patch += int(uncovered.sum())
            if new_v:
                NV = np.vstack(new_v)
                mesh.vertices = o3d.utility.Vector3dVector(np.vstack([V3, NV]))
                mesh.triangles = o3d.utility.Vector3iVector(
                    np.vstack([F3] + new_f).astype(np.int32))
                if C3 is not None and len(C3):
                    # patch colour = mean of the mesh (the texrecon atlas is
                    # the real appearance; unseen patches stay vertex-coloured)
                    fill_c = C3.mean(axis=0)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(
                        np.vstack([C3, np.tile(fill_c, (len(NV), 1))]))
                report["patched_cells"] = n_patch
                print(f"[Poisson-obj] regularize: patched {n_patch:,} "
                      "audit-approved uncovered cells")
        except Exception as e:  # noqa: BLE001
            print(f"[Poisson-obj] regularize: opening/patch stage skipped ({e})")

    # SHAPE-COHERENT hole filling (user 2026-08-29: "repasar la malla y
    # rellenar donde debería estar continuo, copiando la forma") — stitched,
    # evidence-gated; replaces the floating patch quads entirely.
    try:
        from config import cfg as _cfg
        _fill_on = bool((_cfg.get("surface_fit") or {}).get(
            "poisson_fill_holes", True))
    except Exception:  # noqa: BLE001
        _fill_on = True
    if _fill_on:
        try:
            mesh, nf = fill_holes_coherently(mesh, pts, Path(output_dir),
                                             int(instance_id), parts)
            report["holes_filled"] = nf
        except Exception as e:  # noqa: BLE001
            print(f"[Poisson-obj] regularize: coherent fill skipped ({e})")

    print(f"[Poisson-obj] regularize: {len(report['primitives'])} primitive "
          f"region(s), {report['snapped_vertices']:,} vertices ironed, "
          f"{report['cut_faces']:,} faces cut")
    return mesh, report


def _trace_camera_positions(output_dir: Path):
    """(frame_per_point, posed_frames, cam_positions) from the cloud
    traceability + the session poses — lets every point know WHICH camera saw
    it (nearest posed keyframe), so Poisson normals face the right side for
    inside-out rooms and outside-in objects alike."""
    try:
        from segmentation.pipeline import _load_ply_origins
        from segmentation.session_io import _load_camera_source
        origins = _load_ply_origins(Path(output_dir) / "cleaned_cloud.ply")
        cam = _load_camera_source(Path(output_dir).parent, Path(output_dir))
        if origins is None or cam is None or not cam.pose_map:
            return None
        _xyz, fg, _pr, _pc = origins
        posed = sorted(cam.pose_map.keys())
        pos_arr = np.array([np.asarray(cam.pose_map[f], dtype=np.float64)[:3, 3]
                            for f in posed])
        return fg.astype(np.int64), np.asarray(posed, dtype=np.int64), pos_arr
    except Exception as e:  # noqa: BLE001
        print(f"[Poisson-obj] camera traceability unavailable ({e}) — "
              "centroid normal orientation")
        return None


def _cam_positions_for(trace, indices: np.ndarray) -> Optional[np.ndarray]:
    if trace is None:
        return None
    fg, posed, pos_arr = trace
    fgi = fg[indices]
    k = np.searchsorted(posed, fgi).clip(0, len(posed) - 1)
    kp = np.maximum(k - 1, 0)
    pick = np.where(np.abs(posed[k] - fgi) <= np.abs(posed[kp] - fgi), k, kp)
    return pos_arr[pick]


def export_poisson_objects(output_dir: Path,
                           obj_ids: Optional[List[int]] = None,
                           depth: Optional[int] = None,
                           density_quantile: float = 0.06,
                           support_crop_m: float = 0.06,
                           texture: bool = True,
                           regularize: bool = False,
                           progress_cb: Optional[Callable] = None) -> List[Path]:
    """One Poisson mesh per segmented instance, from its own cloud points.

    Pipeline per object: normals (oriented toward the room interior) →
    screened Poisson → density-quantile trim (kills the balloon) → crop to
    measured support (no extrapolation) → major-component filter → vertex
    colors from the cloud RGB. Output: ``output/tsdf/<label>_<id>_poisson/``
    so the viewer lists it next to the RANSAC mesh for direct comparison.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    output_dir = Path(output_dir)
    result_path = output_dir / "segmentation_result.json"
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not result_path.exists() or not cloud_path.exists():
        print("[Poisson-obj] missing segmentation_result.json / cleaned_cloud.ply")
        return []

    res = json.loads(result_path.read_text())
    pcd_full = o3d.io.read_point_cloud(str(cloud_path))
    cloud = np.asarray(pcd_full.points)
    colors = np.asarray(pcd_full.colors) if len(pcd_full.colors) else None
    room_center = cloud.mean(0)   # fallback orientation only
    trace = _trace_camera_positions(output_dir)   # per-point camera (exact)
    written: List[Path] = []

    for inst in res.get("instances", []):
        iid = int(inst.get("instance_id", inst.get("id")))
        if obj_ids and iid not in obj_ids:
            continue
        label = str(inst.get("label", f"object_{iid}"))
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < len(cloud))]
        if len(gi) < 500:
            print(f"[Poisson-obj] {label}_{iid}: too few points ({len(gi)}) — skipping")
            continue
        t0 = time.time()
        if progress_cb:
            progress_cb(iid, "poisson", None, None)
        pts = cloud[gi]
        mesh, stats = poisson_from_points(
            pts, colors[gi] if colors is not None else None, room_center,
            depth=depth, density_quantile=density_quantile,
            support_crop_m=support_crop_m,
            cam_positions=_cam_positions_for(trace, gi))
        if mesh is None:
            print(f"[Poisson-obj] {label}_{iid}: empty after trimming")
            if progress_cb:
                progress_cb(iid, "error", time.time() - t0, None)
            continue
        reg_rep = None
        if regularize:
            if progress_cb:
                progress_cb(iid, "regularize", time.time() - t0, None)
            mesh, reg_rep = regularize_poisson_mesh(mesh, pts, output_dir, iid)
            mesh.compute_vertex_normals()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label) + f"_{iid}_poisson"
        obj_dir = output_dir / "tsdf" / safe
        obj_dir.mkdir(parents=True, exist_ok=True)
        glb = obj_dir / f"{safe}.glb"
        o3d.io.write_triangle_mesh(str(glb), mesh)
        meta = {
            "method": "poisson_object",
            "instance_id": iid,
            "label": f"{label} (poisson)",
            "glb_file": glb.name,
            "textured": False,
            "vertex_colors": bool(colors is not None),
            "regularized": bool(reg_rep and reg_rep.get("snapped_vertices")),
            "regularize": reg_rep,
            "elapsed_s": round(time.time() - t0, 1),
            **stats,
        }
        # Stage-3: real texrecon atlas from the scan frames (falls back to the
        # vertex colours already baked when texturing fails)
        if texture:
            try:
                from reconstruction.texture_bake import bake_object_glb
                if progress_cb:
                    progress_cb(iid, "texture", time.time() - t0, None)
                if bake_object_glb(glb, output_dir.parent, output_dir):
                    meta["textured"] = True
            except Exception as e:  # noqa: BLE001
                print(f"[Poisson-obj] {label}_{iid}: texture failed ({e}) — "
                      "vertex colours kept")
        (obj_dir / f"{safe}.meta.json").write_text(json.dumps(meta, indent=2))
        written.append(glb)
        print(f"[Poisson-obj] {label}_{iid}: {stats['n_vertices']:,} verts "
              f"p95={stats['mesh_to_points_p95_mm']}mm "
              f"gaps>5cm={stats['coverage_gap_frac_5cm']*100:.1f}% "
              f"({meta['elapsed_s']}s)")
        if progress_cb:
            progress_cb(iid, "done", time.time() - t0, str(glb))
    return written


def poisson_mesh_indices(output_dir: Path, indices_file: Path, out_glb: Path,
                        depth: Optional[int] = None, texture: bool = False) -> bool:
    """Poisson-mesh an arbitrary subset of the cleaned cloud (global indices
    from ``indices_file``) into ``out_glb`` — used for the unexplained
    remainder of a fitted surface (e.g. wall3's attached cone)."""
    import open3d as o3d

    output_dir = Path(output_dir)
    pcd_full = o3d.io.read_point_cloud(str(output_dir / "cleaned_cloud.ply"))
    cloud = np.asarray(pcd_full.points)
    colors = np.asarray(pcd_full.colors) if len(pcd_full.colors) else None
    gi = np.load(indices_file)
    gi = gi[(gi >= 0) & (gi < len(cloud))]
    if len(gi) < 500:
        return False
    trace = _trace_camera_positions(output_dir)
    mesh, stats = poisson_from_points(
        cloud[gi], colors[gi] if colors is not None else None, cloud.mean(0),
        depth=depth, cam_positions=_cam_positions_for(trace, gi))
    if mesh is None:
        return False
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_glb), mesh)
    if texture:
        try:
            from reconstruction.texture_bake import bake_object_glb
            bake_object_glb(out_glb, output_dir.parent, output_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[Poisson-obj] remainder texture failed ({e}) — "
                  "vertex colours kept")
    print(f"[Poisson-obj] remainder {indices_file.name}: {stats['n_vertices']:,} "
          f"verts p95={stats['mesh_to_points_p95_mm']}mm → {out_glb.name}")
    return True
