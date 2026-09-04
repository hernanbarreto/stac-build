"""
Point2CAD per-object B-rep — the CAD path (user "vía 2", 2026-09-03).

Reproduces (now IN-TREE — the 2026-08-31 experiment's glue lived only in a
scratchpad) and extends the validated flow:

    perfect-engine regions → labeled .xyzc → vendor point2cad (fit + sew a
    B-rep, env ``point2cad``) → denormalize → evidence trim + island filter
    → publish ``tsdf/<safe>_p2c/`` for the viewer.

New vs the experiment (per-surface INTENT):
  * regions whose VLM proposal says ``opening`` AND whose measured
    ``interior_void_ratio`` agrees (double lock) are EXCLUDED from the
    input — a hole's fake rim-surface never reaches the B-rep, so the hole
    stays a hole;
  * the proposal's roles/locations ride along into the meta.

Vendor notes (vendor/VENDORS.lock.md): output is NORMALIZED — centered /
minor-PCA-axis-to-x / max-extent-scaled; inverted here with mean/R/scale
recomputed from the SAME input by the vendor's exact float32 math.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_P2C_PY = "/workspace/miniforge3/envs/point2cad/bin/python"


def _vendor_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "point2cad"


# ── vendor normalization, replicated EXACTLY (float32) to invert it ──────

def _vendor_norm_params(points: np.ndarray):
    """(mean, R, scale) such that vendor-normalized = (R @ (p-mean).T).T/scale."""
    eps = np.finfo(np.float32).eps
    pts = points.astype(np.float32)
    mean = np.mean(pts, 0, keepdims=True)
    q = pts - mean
    S, U = np.linalg.eig(q.T @ q)
    S, U = S.real, U.real
    a = U[:, np.argmin(S)].astype(np.float32)
    b = np.array([1, 0, 0], dtype=np.float32)
    cos = float(np.dot(a, b))
    sin = float(np.linalg.norm(np.cross(b, a)))
    u = a
    v = b - np.dot(a, b) * a
    v = v / (np.linalg.norm(v) + eps)
    w = np.cross(b, a)
    w = w / (np.linalg.norm(w) + eps)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    try:
        R = F @ G @ np.linalg.inv(F)
    except Exception:  # noqa: BLE001
        R = np.eye(3, dtype=np.float32)
    rot = (R @ q.T).T
    std = np.max(rot, 0) - np.min(rot, 0)
    scale = float(np.max(std) + eps)
    return mean.astype(np.float64)[0], R.astype(np.float64), scale


def _denormalize(v_norm: np.ndarray, mean, R, scale) -> np.ndarray:
    return (np.linalg.inv(R) @ (v_norm * scale).T).T + mean


# ── evidence trim: keep the B-rep only where the cloud testifies ─────────

def _evidence_trim(V: np.ndarray, F: np.ndarray, cloud: np.ndarray,
                   trim_m: float, log) -> np.ndarray:
    """Face mask: smoothed per-vertex NN-distance to the object's cloud ≤
    trim_m, then island filter (drop tiny disconnected leftovers)."""
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    kd = cKDTree(cloud[:: max(1, len(cloud) // 300_000)])
    d, _ = kd.query(V, k=1)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    for _ in range(5):   # graph-smooth the evidence field (soft iso-contour)
        acc = d.copy()
        cnt = np.ones(len(V))
        np.add.at(acc, edges[:, 0], d[edges[:, 1]])
        np.add.at(acc, edges[:, 1], d[edges[:, 0]])
        np.add.at(cnt, edges[:, 0], 1)
        np.add.at(cnt, edges[:, 1], 1)
        d = acc / cnt
    keep = d[F].mean(axis=1) <= trim_m
    if not keep.any():
        return keep
    # island filter on kept faces
    fidx = np.where(keep)[0]
    loc = np.full(len(F), -1, np.int64)
    loc[fidx] = np.arange(len(fidx))
    fe = {}
    for fi in fidx:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((F[fi, a], F[fi, b])))
            fe.setdefault(key, []).append(loc[fi])
    rows, cols = [], []
    for lst in fe.values():
        for i in range(len(lst) - 1):
            rows.append(lst[i]); cols.append(lst[i + 1])
    g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(len(fidx), len(fidx)))
    _n, lab = sp.csgraph.connected_components(g, directed=False)
    sizes = np.bincount(lab)
    min_faces = max(100, int(0.005 * len(fidx)))
    ok = np.isin(lab, np.where(sizes >= min_faces)[0])
    out = np.zeros(len(F), bool)
    out[fidx[ok]] = True
    log(f"  trim @{trim_m * 100:.0f}cm: {int(keep.sum())}/{len(F)} faces, "
        f"islands removed: {int(keep.sum()) - int(out.sum())}")
    return out


# ── main entry (worker mode 'p2c') ───────────────────────────────────────

def build_p2c_object(output_dir: Path, instance_id: int,
                     cfg: Optional[dict] = None,
                     source: Optional[str] = None,
                     log=print) -> Optional[Path]:
    import open3d as o3d
    import trimesh
    from segmentation.tsdf_export import _safe_label
    from segmentation.perfect_object import (_detect_and_snap,
                                             _load_source_mesh,
                                             _load_shape_proposal,
                                             _match_proposal,
                                             _to_display, _to_raw)

    t0 = time.time()
    cfg = cfg or {}
    trim_m = float(cfg.get("p2c_trim_m", 0.06))
    parallel = int(cfg.get("p2c_parallel", 4))
    max_pts_region = int(cfg.get("p2c_max_pts_region", 4000))
    out = Path(output_dir)

    result = json.loads((out / "segmentation_result.json").read_text())
    inst = next((i for i in result.get("instances", [])
                 if int(i.get("instance_id", i.get("id"))) == int(instance_id)),
                None)
    if inst is None:
        raise ValueError(f"instance {instance_id} not found")
    label = str(inst.get("label", "segment"))
    safe = _safe_label(label, int(instance_id))

    tm, src = _load_source_mesh(out, safe, source)
    V_raw = np.asarray(tm.vertices, np.float64)
    F = np.asarray(tm.faces, np.int64)
    Vd = _to_display(out, V_raw)
    regions, _ = _detect_and_snap(tm, F, Vd, cfg, safe, log)
    if not regions:
        raise RuntimeError(f"{safe}: no regions for point2cad")

    # per-surface intent: openings out of the B-rep (double lock)
    proposal = _load_shape_proposal(out, safe)
    prop_match: Dict[int, dict] = {}
    drop_open: set = set()
    if proposal:
        prop_match = _match_proposal(regions, Vd, proposal)
        for i, e in prop_match.items():
            pr = e.get("proposal") or {}
            if (pr.get("proposed_kind") == "opening"
                    and float(pr.get("confidence", 0.0)) >= 0.5
                    and float(e.get("interior_void_ratio") or 0.0) >= 0.5):
                drop_open.add(i)
        if drop_open:
            log(f"[p2c:{safe}] {len(drop_open)} region(s) excluded as "
                f"OPENINGS: {sorted(drop_open)}")

    # labeled cloud for point2cad (display frame; label = region index)
    rng = np.random.default_rng(5)
    rows: List[np.ndarray] = []
    for i, r in enumerate(regions):
        if i in drop_open:
            continue
        P = Vd[r["v_idx"]]
        if len(P) > max_pts_region:
            P = P[rng.choice(len(P), max_pts_region, replace=False)]
        rows.append(np.column_stack([P, np.full(len(P), i, float)]))
    xyzc = np.vstack(rows)
    work = out / "shape_p2c" / safe
    work.mkdir(parents=True, exist_ok=True)
    xyzc_path = work / "input.xyzc"
    np.savetxt(xyzc_path, xyzc, fmt="%.6f %.6f %.6f %d")
    log(f"[p2c:{safe}] {len(xyzc):,} labeled pts, "
        f"{len(regions) - len(drop_open)} surfaces → point2cad "
        f"(parallel {parallel})")

    cmd = [_P2C_PY, "-m", "point2cad.main",
           "--path_in", str(xyzc_path), "--path_out", str(work),
           "--max_parallel_surfaces", str(parallel)]
    proc = subprocess.Popen(cmd, cwd=str(_vendor_dir()),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line and ("Fitting" in line or "Saving" in line or "%" in line
                     or "Error" in line or "error" in line):
            log(f"[p2c:{safe}] {line}")
    rc = proc.wait()
    clipped = work / "clipped" / "mesh.ply"
    if rc != 0 or not clipped.exists():
        raise RuntimeError(f"{safe}: point2cad failed (rc={rc})")

    # back to metric: invert the vendor normalization of THIS input
    mean, R, scale = _vendor_norm_params(xyzc[:, :3])
    bm = trimesh.load(str(clipped), process=False)
    Vb = _denormalize(np.asarray(bm.vertices, np.float64), mean, R, scale)
    Fb = np.asarray(bm.faces, np.int64)
    colors = None
    try:
        colors = np.asarray(bm.visual.vertex_colors)[:, :3] / 255.0
    except Exception:  # noqa: BLE001
        pass

    evidence = Vd[np.unique(np.concatenate(
        [r["v_idx"] for i, r in enumerate(regions) if i not in drop_open]))]
    keep = _evidence_trim(Vb, Fb, evidence, trim_m, log)
    n_untrimmed = len(Fb)
    Fk = Fb[keep]
    if len(Fk) == 0:
        raise RuntimeError(f"{safe}: evidence trim removed everything")

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(_to_raw(out, Vb)),
        o3d.utility.Vector3iVector(Fk))
    if colors is not None and len(colors) == len(Vb):
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    dst = out / "tsdf" / f"{safe}_p2c"
    dst.mkdir(parents=True, exist_ok=True)
    glb = dst / f"{safe}_p2c.glb"
    o3d.io.write_triangle_mesh(str(glb), mesh)
    meta = {
        "method": "p2c",
        "instance_id": int(instance_id),
        "label": f"{label} (p2c)",
        "glb_file": glb.name,
        "source": "point2cad clipped B-rep",
        "source_mesh": src.parent.name,
        "evidence_trim_m": trim_m,
        "n_surfaces_in": len(regions) - len(drop_open),
        "openings_excluded": sorted(int(i) for i in drop_open),
        "shape_proposal_object": (proposal or {}).get("object"),
        "n_triangles_untrimmed": int(n_untrimmed),
        "n_triangles": int(len(mesh.triangles)),
        "n_vertices": int(len(mesh.vertices)),
        "vertex_colors": bool(colors is not None),
        "textured": False,
        "provenance": "tool_measured",
        "trim": f"smoothed evidence clip @{trim_m * 100:.0f}cm + island filter",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (dst / f"{safe}_p2c.meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[p2c:{safe}] ✅ B-rep: {meta['n_triangles']:,} tris "
        f"({n_untrimmed:,} untrimmed), {len(drop_open)} opening(s) excluded "
        f"→ {glb.name} ({meta['elapsed_s']}s)")
    return glb
