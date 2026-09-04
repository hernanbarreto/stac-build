"""
Per-object geometric UNDERSTANDING and PERFECTION (user 2026-08-31).

Two deliverables share one detection engine:

``diagnose_object`` — the PARTS VIEW ("que el sistema muestre qué entendió"):
    hierarchical connected-region split-and-fit (plane/cylinder/sphere with
    acceptance gates), intent snapping, MIRROR-SYMMETRY detection verified
    against the cloud, published as a flat-colored mesh (one color per part,
    kind-coded) + a parts inventory in the meta. The user critiques the
    UNDERSTANDING here before any rebuild.

``perfect_object`` — mesh IRONING v2: same detection, then a screened-
    Laplacian displacement solve that irons the existing mesh onto the
    perfected surfaces (topology preserved, analytic crease snapping,
    freeform untouched, displacement capped).

Research base: GlobFit '11, Mitra '06/'07 (symmetry), Split-and-Fit '24
(hierarchical partition), CAD-journal 2024 regularity enhancement.
Everything deterministic — provenance ``tool_measured``. The cloud and the
source meshes are never modified.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_UP = np.array([0.0, 1.0, 0.0])   # display (leveled) frame is three.js Y-up


# ── frame helpers ────────────────────────────────────────────────────────

def _floor_transform(output_dir: Path):
    p = Path(output_dir) / "floor_transform.npz"
    if not p.exists():
        return 1.0, np.eye(3), np.zeros(3)
    d = np.load(p)
    return float(d["s"]), np.asarray(d["R"], float), np.asarray(d["t"], float)


def _to_display(output_dir: Path, pts: np.ndarray) -> np.ndarray:
    s, R, t = _floor_transform(output_dir)
    return s * (np.asarray(pts, np.float64) @ R.T) + t


def _to_raw(output_dir: Path, pts: np.ndarray) -> np.ndarray:
    s, R, t = _floor_transform(output_dir)
    return ((np.asarray(pts, np.float64) - t) / s) @ R


# ── constrained re-fits (snap mechanics — validated gates) ───────────────

def _rebuild_plane(P: np.ndarray, n_new: np.ndarray, old_normal: np.ndarray):
    from reconstruction.surface_fit.plane import PlaneModel, _plane_basis
    n = np.asarray(n_new, np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    if float(n @ np.asarray(old_normal)) < 0:
        n = -n
    d = -float(np.mean(P @ n))
    centroid = P.mean(axis=0)
    origin = centroid - (float(centroid @ n) + d) * n
    u, v = _plane_basis(n, _UP)
    res = P @ n + d
    return PlaneModel(normal=n, d=d, origin=origin, u=u, v=v,
                      rms=float(np.sqrt(np.mean(res ** 2))),
                      inlier_frac=1.0, n_points=len(P))


def _rebuild_cylinder(P: np.ndarray, w_new: np.ndarray,
                      radius_override: Optional[float] = None):
    from reconstruction.surface_fit.quadric import CylinderModel
    w = np.asarray(w_new, np.float64)
    w = w / max(np.linalg.norm(w), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(w, ref); e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(w, e1)
    c0 = P.mean(axis=0)
    q = P - c0
    x, y = q @ e1, q @ e2
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    a, b, c = sol
    R_fit = float(np.sqrt(max(c + a * a + b * b, 1e-12)))
    if radius_override is not None:
        R_fit = float(radius_override)
    axis_point = c0 + a * e1 + b * e2
    theta = np.arctan2(y - b, x - a)
    ts = np.sort(theta)
    gaps = np.diff(np.concatenate([ts, ts[:1] + 2 * np.pi]))
    k = int(np.argmax(gaps))
    theta0 = float(ts[(k + 1) % len(ts)]) if len(ts) > 1 else 0.0
    span = float(2 * np.pi - gaps[k]) if len(ts) > 1 else 2 * np.pi
    rho = np.hypot(x - a, y - b)
    return CylinderModel(axis_point=axis_point, axis_dir=w, radius=R_fit,
                         theta_ref=e1, theta0=theta0,
                         theta_span=max(span, 0.1),
                         rms=float(np.sqrt(np.mean((rho - R_fit) ** 2))),
                         inlier_frac=1.0, n_points=len(P))


def _rebuild_sphere_at(P: np.ndarray, center: np.ndarray):
    """Sphere with a FIXED centre (proposal-driven recentre onto a coaxial
    axis): radius = mean radial distance, chart rebuilt as fit_sphere does."""
    from reconstruction.surface_fit.quadric import SphereModel, _orthobasis
    c = np.asarray(center, np.float64)
    q = np.asarray(P, np.float64) - c
    rho = np.linalg.norm(q, axis=1)
    r = float(np.mean(rho))
    mean_dir = P.mean(axis=0) - c
    ln = np.linalg.norm(mean_dir)
    pole = mean_dir / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
    u, _v = _orthobasis(pole)
    phi = np.arccos(np.clip((q @ pole) / np.maximum(rho, 1e-12), -1, 1))
    theta = np.arctan2(q @ np.cross(pole, u), q @ u)
    return SphereModel(center=c, radius=r, pole=pole, theta_ref=u,
                       phi_c=float(np.median(phi)),
                       theta_c=float(np.median(theta)),
                       rms=float(np.sqrt(np.mean((rho - r) ** 2))),
                       inlier_frac=1.0, n_points=len(P))


# ── VLM shape-proposal consumption (proposals NEVER override the gates) ──

def _load_shape_proposal(out: Path, safe: str) -> Optional[dict]:
    p = Path(out) / "shape_proposals" / f"{safe}_proposal.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _match_proposal(regions: List[dict], Vd: np.ndarray,
                    prop: Optional[dict]) -> Dict[int, dict]:
    """Detected region index → proposal entry, matched by display-frame
    centroid + detected kind (region indices drift between RANSAC runs)."""
    entries = [e for e in (prop or {}).get("regions", [])
               if e.get("proposal") and e.get("centroid_m")]
    match: Dict[int, dict] = {}
    used: set = set()
    for i, r in enumerate(regions):
        P = Vd[r["v_idx"]]
        c = P.mean(axis=0)
        diag = float(np.linalg.norm(P.ptp(axis=0)))
        best = None
        for j, e in enumerate(entries):
            if j in used or e.get("detected_kind") != r["kind"]:
                continue
            d = float(np.linalg.norm(np.asarray(e["centroid_m"]) - c))
            if d < max(0.10, 0.25 * diag) and (best is None or d < best[1]):
                best = (j, d)
        if best is not None:
            match[i] = entries[best[0]]
            used.add(best[0])
    return match


def _apply_proposal_intents(regions: List[dict], Vd: np.ndarray,
                            match: Dict[int, dict], log, safe: str) -> int:
    """Relation-driven snaps from the VLM proposal, each guarded by the same
    revert-if-degrading rms gate as the deterministic snaps: parallel plane
    groups share a size-weighted normal; coaxial cylinders share an axis and
    coaxial spheres are recentred onto it."""
    p2d: Dict[int, int] = {}
    for i, e in match.items():
        try:
            p2d[int(e["region"])] = i
        except Exception:  # noqa: BLE001
            continue
    par: set = set()
    coax: set = set()
    for i, e in match.items():
        for t in (e.get("proposal") or {}).get("relations") or []:
            if ":" not in t:
                continue
            k, v = t.split(":", 1)
            try:
                j = p2d.get(int(v))
            except ValueError:
                continue
            if j is None or j == i:
                continue
            pair = (min(i, j), max(i, j))
            if k == "parallel_to":
                par.add(pair)
            elif k in ("coaxial_with", "same_radius_as", "concentric_with"):
                coax.add(pair)

    def _clusters(pairs: set, pred) -> List[List[int]]:
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            if pred(a) and pred(b):
                parent[find(a)] = find(b)
        groups: Dict[int, List[int]] = {}
        for x in list(parent):
            groups.setdefault(find(x), []).append(x)
        return [g for g in groups.values() if len(g) > 1]

    def _rms(model, P) -> float:
        d = np.asarray(model.signed_distance(P))
        return float(np.sqrt(np.mean(d ** 2)))

    applied = 0
    for g in _clusters(par, lambda i: regions[i]["kind"] == "plane"):
        w = np.zeros(3)
        ref = regions[g[0]]["model"].normal
        for i in g:
            n = regions[i]["model"].normal
            if float(n @ ref) < 0:
                n = -n
            w += len(regions[i]["v_idx"]) * n
        w /= max(np.linalg.norm(w), 1e-12)
        for i in g:
            r = regions[i]
            Pm = Vd[r["v_idx"]]
            pre = _rms(r["model"], Pm)
            m2 = _rebuild_plane(Pm, w, r["model"].normal)
            if _rms(m2, Pm) <= max(1.5 * pre, pre + 0.005):
                r["model"] = m2
                r["notes"].append("normal shared with proposed-parallel group")
                applied += 1

    for g in _clusters(coax,
                       lambda i: regions[i]["kind"] in ("cylinder", "sphere")):
        cyls = [i for i in g if regions[i]["kind"] == "cylinder"]
        if not cyls:
            continue
        w = np.zeros(3)
        ref = regions[cyls[0]]["model"].axis_dir
        for i in cyls:
            a = regions[i]["model"].axis_dir
            if float(a @ ref) < 0:
                a = -a
            w += len(regions[i]["v_idx"]) * a
        w /= max(np.linalg.norm(w), 1e-12)
        for i in cyls:
            r = regions[i]
            Pm = Vd[r["v_idx"]]
            pre = _rms(r["model"], Pm)
            m2 = _rebuild_cylinder(Pm, w)
            if _rms(m2, Pm) <= max(1.5 * pre, pre + 0.005):
                r["model"] = m2
                r["notes"].append("axis shared with proposed-coaxial group")
                applied += 1
        # NOTE (user 2026-09-03): sphere recentring onto the proposed axis was
        # REMOVED — it displaced parts, and the 'spheres' were hole rims.
    if applied:
        log(f"[perfect:{safe}] proposal intents: {applied} snap(s) applied "
            f"(gated), {len(par)} parallel + {len(coax)} coaxial relation(s)")
    return applied


def _snap_direction(v: np.ndarray, tol_deg: float) -> Tuple[Optional[np.ndarray], Optional[str]]:
    v = np.asarray(v, np.float64)
    v = v / max(np.linalg.norm(v), 1e-12)
    c = float(abs(v @ _UP))
    if c >= np.cos(np.deg2rad(tol_deg)):
        return (_UP if v @ _UP >= 0 else -_UP), "vertical"
    if c <= np.sin(np.deg2rad(tol_deg)):
        h = v - (v @ _UP) * _UP
        n = np.linalg.norm(h)
        if n > 1e-9:
            return h / n, "horizontal"
    return None, None


def _project_onto(kind: str, model, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, np.float64)
    if kind == "plane":
        return P - np.outer(P @ model.normal + model.d, model.normal)
    if kind == "cylinder":
        w = model.axis_dir
        q = P - model.axis_point
        s = q @ w
        radial = q - np.outer(s, w)
        rho = np.maximum(np.linalg.norm(radial, axis=1), 1e-9)
        return model.axis_point + np.outer(s, w) \
            + radial * (model.radius / rho)[:, None]
    if kind == "sphere":
        q = P - model.center
        r = np.maximum(np.linalg.norm(q, axis=1), 1e-9)
        return model.center + q * (model.radius / r)[:, None]
    return P


# ── shared detection engine ──────────────────────────────────────────────

def _load_source_mesh(out: Path, safe: str, source: Optional[str]):
    import trimesh
    order = ("_pgsr", "_poisson") if source == "pgsr" else ("_poisson", "_pgsr")
    src = None
    for suf in order:
        p = out / "tsdf" / f"{safe}{suf}" / f"{safe}{suf}.glb"
        if p.exists():
            src = p
            break
    if src is None:
        raise FileNotFoundError(f"{safe}: no poisson/pgsr GLB")
    tm = trimesh.load(str(src), force="mesh")
    tm.merge_vertices()
    return tm, src


def _detect_and_snap(tm, F: np.ndarray, Vd: np.ndarray, cfg: dict,
                     safe: str, log) -> Tuple[List[dict], int]:
    """Hierarchical connected split-and-fit + intent snapping.
    Returns (regions, min_faces_eff); each region:
    {kind, model, f_idx, v_idx, notes}."""
    import scipy.sparse as sp
    from reconstruction.surface_fit.escalate import FITTERS, FitContext

    crease0 = float(cfg.get("perfect_crease_deg", 30.0))
    min_region_faces = int(cfg.get("perfect_min_region_faces", 300))
    accept_p95_mm = float(cfg.get("perfect_accept_p95_mm", 40.0))
    max_fit_pts = int(cfg.get("perfect_max_fit_pts", 120_000))

    adj = tm.face_adjacency
    # coarse→fine smoothing levels: a failing region is re-partitioned at the
    # next finer level and each connected piece retried (one threshold either
    # shatters noisy TSDF meshes or merges everything — both observed)
    levels = [(6, crease0), (3, 22.0), (1, 15.0), (0, 10.0)]
    keep_by_level: List[np.ndarray] = []
    FN0 = np.asarray(tm.face_normals, np.float64).copy()
    for n_smooth, cr_deg in levels:
        FN = FN0.copy()
        for _ in range(n_smooth):
            acc = FN.copy()
            np.add.at(acc, adj[:, 0], FN[adj[:, 1]])
            np.add.at(acc, adj[:, 1], FN[adj[:, 0]])
            FN = acc / np.maximum(
                np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
        cosang = np.clip((FN[adj[:, 0]] * FN[adj[:, 1]]).sum(axis=1), -1, 1)
        keep_by_level.append(np.arccos(cosang) < np.deg2rad(cr_deg))

    min_faces_eff = min(min_region_faces, max(100, len(F) // 300))
    ctx = FitContext(world_up=_UP, dist_thresh=0.02, min_inlier_frac=0.02)
    rng = np.random.default_rng(0)

    def _try_ladder(f_idx: np.ndarray):
        v_idx = np.unique(F[f_idx].ravel())
        Pr = Vd[v_idx]
        fit_pts = Pr if len(Pr) <= max_fit_pts else \
            Pr[rng.choice(len(Pr), max_fit_pts, replace=False)]
        for kind in ("plane", "cylinder", "sphere"):
            fitter = FITTERS.get(kind)
            if fitter is None:
                continue
            try:
                model = fitter(fit_pts, ctx)
            except Exception:  # noqa: BLE001
                model = None
            if model is None:
                continue
            d = np.abs(np.asarray(model.signed_distance(Pr)))
            if float(np.percentile(d, 95)) * 1000.0 <= accept_p95_mm:
                return kind, model, v_idx
        return None

    regions: List[dict] = []

    def _components(f_idx: np.ndarray, level: int) -> List[np.ndarray]:
        loc = np.full(len(F), -1, dtype=np.int64)
        loc[f_idx] = np.arange(len(f_idx))
        e = adj[keep_by_level[level]]
        m = (loc[e[:, 0]] >= 0) & (loc[e[:, 1]] >= 0)
        e = e[m]
        gsub = sp.coo_matrix(
            (np.ones(len(e)), (loc[e[:, 0]], loc[e[:, 1]])),
            shape=(len(f_idx), len(f_idx)))
        _, lab = sp.csgraph.connected_components(gsub, directed=False)
        return [f_idx[lab == c] for c in range(lab.max() + 1)
                if (lab == c).sum() >= min_faces_eff]

    def _segment(f_idx: np.ndarray, level: int):
        hit = _try_ladder(f_idx)
        if hit is not None:
            kind, model, v_idx = hit
            regions.append({"kind": kind, "model": model,
                            "f_idx": f_idx, "v_idx": v_idx, "notes": []})
            return
        if level + 1 >= len(levels):
            return
        pieces = _components(f_idx, level + 1)
        if len(pieces) == 1 and len(pieces[0]) == len(f_idx):
            return
        for piece in pieces:
            _segment(piece, level + 1)

    for root in _components(np.arange(len(F)), 0):
        _segment(root, 0)
    log(f"[perfect:{safe}] hierarchical split-and-fit: "
        f"{len(regions)} region(s) accepted (min {min_faces_eff} faces)")

    _snap_regions(regions, Vd, cfg, safe, log)
    return regions, min_faces_eff


def _snap_regions(regions: List[dict], Vd: np.ndarray, cfg: dict,
                  safe: str, log) -> None:
    """Intent snapping with honesty gates (revert if degrading) — shared by
    the mesh and cloud detection paths; mutates ``regions`` in place."""
    tol_deg = float(cfg.get("perfect_snap_deg", 4.0))
    rel_tol_deg = float(cfg.get("perfect_relation_deg", 3.0))
    radius_tol = float(cfg.get("perfect_radius_tol", 0.02))

    dominant_n = None
    plane_regs = [r for r in regions if r["kind"] == "plane"]
    if plane_regs:
        dominant_n = max(plane_regs,
                         key=lambda r: len(r["v_idx"]))["model"].normal.copy()
    for r in regions:
        if r.get("model") is None:   # freeform cloud residue — nothing to snap
            continue
        Pm = Vd[r["v_idx"]]
        model, kind = r["model"], r["kind"]
        d_pre = np.abs(np.asarray(model.signed_distance(Pm)))
        rms_pre = float(np.sqrt(np.mean(d_pre ** 2)))
        model_pre, notes = model, []
        try:
            if kind == "plane":
                n_new, tag = _snap_direction(model.normal, tol_deg)
                if n_new is None and dominant_n is not None \
                        and not np.allclose(model.normal, dominant_n):
                    a = np.rad2deg(np.arccos(np.clip(
                        abs(float(model.normal @ dominant_n)), 0, 1)))
                    if a <= rel_tol_deg:
                        n_new, tag = dominant_n, "parallel-to-dominant"
                    elif abs(a - 90.0) <= rel_tol_deg:
                        h = model.normal - float(model.normal @ dominant_n) * dominant_n
                        if np.linalg.norm(h) > 1e-9:
                            n_new, tag = h / np.linalg.norm(h), "perpendicular-to-dominant"
                if n_new is not None:
                    model = _rebuild_plane(Pm, n_new, model.normal)
                    notes.append(f"normal snapped {tag}")
            elif kind == "cylinder":
                w_new, tag = _snap_direction(model.axis_dir, tol_deg)
                if w_new is not None:
                    model = _rebuild_cylinder(Pm, w_new)
                    notes.append(f"axis snapped {tag}")
        except Exception as e:  # noqa: BLE001
            log(f"[perfect:{safe}] snap failed ({e}) — raw fit kept")
        if notes:
            d_post = np.abs(np.asarray(model.signed_distance(Pm)))
            if float(np.sqrt(np.mean(d_post ** 2))) > max(1.5 * rms_pre,
                                                          rms_pre + 0.005):
                model, notes = model_pre, []
        r["model"], r["notes"] = model, notes

    cyls = [r for r in regions if r["kind"] == "cylinder"]
    if len(cyls) >= 2:
        radii = np.array([r["model"].radius for r in cyls])
        mean_r = float(radii.mean())
        if mean_r > 0 and float(np.max(np.abs(radii - mean_r))) <= radius_tol * mean_r:
            for r in cyls:
                r["model"] = _rebuild_cylinder(Vd[r["v_idx"]],
                                               r["model"].axis_dir,
                                               radius_override=mean_r)
                r["notes"].append(f"radius equalized to {mean_r:.4f} m")


# ── cloud-native detection (USER 2026-09-04: "p2c necesita la nube con
# puntos etiquetados, no el mesh — el mesh ya tiene errores") ─────────────

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8",
              "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "<i2", "int16": "<i2",
              "ushort": "<u2", "uint16": "<u2", "int": "<i4",
              "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def _read_ply_fields(path: Path) -> Dict[str, np.ndarray]:
    """Minimal binary-little-endian PLY vertex reader that PRESERVES custom
    per-point properties (Open3D drops them — ``confidence`` is the point)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY")
        n_vertex, props, fmt = 0, [], None
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            elif line.startswith("element") and n_vertex and props:
                raise ValueError(f"{path}: non-vertex elements after vertex "
                                 "not supported")
            elif line.startswith("property") and n_vertex:
                _, typ, name = line.split()[:3]
                props.append((name, _PLY_TYPES[typ]))
            elif line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: format {fmt} not supported")
        data = np.fromfile(f, dtype=np.dtype(props), count=n_vertex)
    return {name: data[name] for name, _t in props}


def _load_instance_cloud(out: Path, inst: dict, cfg: dict, safe: str,
                         log) -> np.ndarray:
    """The instance's OWN cleaned-cloud points (raw frame), with the
    lowest-confidence ``p2c_conf_trim_pct`` % dropped. The cloud is the
    validated truth — meshes never enter the CAD path."""
    trim_pct = float(cfg.get("p2c_conf_trim_pct", 20.0))
    fields = _read_ply_fields(out / "cleaned_cloud.ply")
    xyz = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(
        np.float64)
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    if len(gi) == 0:
        raise RuntimeError(f"{safe}: instance has no globalIndices into "
                           "cleaned_cloud.ply")
    P = xyz[gi]
    conf = fields.get("confidence")
    if conf is not None and trim_pct > 0:
        c = np.asarray(conf, np.float64)[gi]
        thr = float(np.percentile(c, trim_pct))
        keep = c >= thr
        log(f"[cloud:{safe}] {len(P):,} pts → {int(keep.sum()):,} after "
            f"dropping the {trim_pct:.0f}% lowest-confidence "
            f"(thr {thr:.3f})")
        P = P[keep]
    elif conf is None:
        log(f"[cloud:{safe}] cleaned_cloud.ply has no confidence field — "
            f"no trim applied ({len(P):,} pts)")
    return P


def _detect_and_snap_cloud(Pd: np.ndarray, cfg: dict, safe: str,
                           log) -> Tuple[List[dict], int]:
    """Cloud-native region labeling: RANSAC-first (the validated
    ``decompose.extract_primitives`` machinery — noise-robust where crease
    graphs on raw-point normals are not), each primitive split into connected
    components; large freeform residues become labeled regions of their own
    (kind ``freeform``, no analytic model) so point2cad's INR can fit them.
    ``v_idx`` indexes into ``Pd``. Same snap gates as the mesh detector."""
    import open3d as o3d
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    from reconstruction.surface_fit.escalate import FITTERS, FitContext

    min_region_pts = int(cfg.get("perfect_min_region_faces", 300))
    inlier_dist = float(cfg.get("p2c_inlier_dist_m", 0.02))
    max_prims = int(cfg.get("p2c_max_primitives", 24))
    freeform_min = int(cfg.get("p2c_freeform_min_pts", 2000))
    max_fit_pts = int(cfg.get("perfect_max_fit_pts", 120_000))
    rng = np.random.default_rng(0)

    ctx = FitContext(world_up=_UP, dist_thresh=inlier_dist,
                     min_inlier_frac=0.02)

    # iterative largest-support extraction. Membership is DISTANCE-only —
    # the VGGT cloud's normals are too noisy to veto points (measured on a
    # true plane, rms 11 mm: only 55% pass a 30° gate). Normals instead
    # SCORE the winner (voxel-denoised, orientation-agnostic), so a
    # cylinder's coherent radial field can beat a tangent plane slicing it.
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pd))
    pv = pc.voxel_down_sample(0.03)
    pv.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=24))
    Nv = np.asarray(pv.normals, np.float64)
    _, nn_v = cKDTree(np.asarray(pv.points)).query(Pd, k=1, workers=8)
    N = Nv[nn_v]

    def _members(kind, model, P, Np):
        """(distance-membership mask, mean normal agreement of members)."""
        d = np.abs(np.asarray(model.signed_distance(P)))
        m = d <= inlier_dist
        if not m.any():
            return m, 0.0
        try:
            if kind == "plane":
                ncos = np.abs(Np[m] @ model.normal)
            elif kind == "cylinder":
                q = P[m] - model.axis_point
                q -= np.outer(q @ model.axis_dir, model.axis_dir)
                q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True),
                                1e-12)
                ncos = np.abs((Np[m] * q).sum(axis=1))
            elif kind == "sphere":
                q = P[m] - model.center
                q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True),
                                1e-12)
                ncos = np.abs((Np[m] * q).sum(axis=1))
            else:
                return m, 0.5
            return m, float(ncos.mean())
        except Exception:  # noqa: BLE001 — distance-only fallback
            return m, 0.5

    remaining = np.ones(len(Pd), dtype=bool)
    parts: List[tuple] = []
    min_claim = max(min_region_pts, len(Pd) // 500)
    for it in range(max_prims):
        idx = np.nonzero(remaining)[0]
        if len(idx) < min_claim:
            break
        sub_p, sub_n = Pd[idx], N[idx]
        fit_pts = sub_p if len(sub_p) <= max_fit_pts else \
            sub_p[np.random.default_rng(it).choice(
                len(sub_p), max_fit_pts, replace=False)]
        best = None
        for kind in ("plane", "cylinder", "sphere"):
            fitter = FITTERS.get(kind)
            if fitter is None:
                continue
            try:
                model = fitter(fit_pts, ctx)
            except Exception:  # noqa: BLE001
                model = None
            if model is None:
                continue
            mm, agree = _members(kind, model, sub_p, sub_n)
            n_in = int(mm.sum())
            score = n_in * (0.25 + 0.75 * agree)
            if best is None or score > best[4]:
                best = (kind, model, mm, n_in, score)
        if best is None:
            break
        kind, model, mm, n_in, _score = best
        if n_in < max(min_claim, 0.02 * len(idx)):
            break
        mask = np.zeros(len(Pd), dtype=bool)
        mask[idx[mm]] = True
        parts.append((kind, model, mask))
        remaining &= ~mask

    # connectivity graph: k-NN edges capped at ~3× the median point spacing
    kd = cKDTree(Pd)
    dnn = kd.query(Pd[:: max(1, len(Pd) // 5000)], k=2, workers=8)[0][:, 1]
    radius = max(3.0 * float(np.median(dnn)), 0.01)
    k_conn = 8
    dist, nbr = kd.query(Pd, k=k_conn + 1, workers=8)
    src = np.repeat(np.arange(len(Pd)), k_conn)
    dst = nbr[:, 1:].ravel()
    ok = dist[:, 1:].ravel() <= radius
    E = np.column_stack([src[ok], dst[ok]])

    def _components(mask: np.ndarray) -> List[np.ndarray]:
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return []
        loc = np.full(len(Pd), -1, dtype=np.int64)
        loc[idx] = np.arange(len(idx))
        m = (loc[E[:, 0]] >= 0) & (loc[E[:, 1]] >= 0)
        e = E[m]
        g = sp.coo_matrix((np.ones(len(e)), (loc[e[:, 0]], loc[e[:, 1]])),
                          shape=(len(idx), len(idx)))
        _, lab = sp.csgraph.connected_components(g, directed=False)
        return [idx[lab == c] for c in range(int(lab.max()) + 1)]

    regions: List[dict] = []
    for kind, model, mask in parts:
        for piece in _components(mask):
            if len(piece) < min_region_pts:
                continue
            P = Pd[piece]
            fit_pts = P if len(P) <= max_fit_pts else \
                P[rng.choice(len(P), max_fit_pts, replace=False)]
            refit = None
            try:
                refit = FITTERS[kind](fit_pts, ctx)
            except Exception:  # noqa: BLE001 — the shared model still stands
                refit = None
            regions.append({"kind": kind, "model": refit or model,
                            "f_idx": piece, "v_idx": piece, "notes": []})
    # merge fragments of ONE physical surface (greedy extraction slices a
    # surface into strips): same kind, compatible models, actually adjacent
    # in the point graph — then refit the union
    cos_tol = float(np.cos(np.deg2rad(3.0)))

    def _compatible(a: dict, b: dict) -> bool:
        if a["kind"] != b["kind"] or a["model"] is None or b["model"] is None:
            return False
        ma, mb = a["model"], b["model"]
        try:
            if a["kind"] == "plane":
                if abs(float(ma.normal @ mb.normal)) < cos_tol:
                    return False
                off = float((Pd[b["v_idx"]].mean(0)
                             - Pd[a["v_idx"]].mean(0)) @ ma.normal)
                return abs(off) <= 1.5 * inlier_dist
            if a["kind"] == "cylinder":
                return (abs(float(ma.axis_dir @ mb.axis_dir)) >= cos_tol
                        and abs(ma.radius - mb.radius)
                        <= 0.05 * max(ma.radius, mb.radius))
            if a["kind"] == "sphere":
                return (float(np.linalg.norm(ma.center - mb.center))
                        <= 2.0 * inlier_dist
                        and abs(ma.radius - mb.radius)
                        <= 0.05 * max(ma.radius, mb.radius))
        except Exception:  # noqa: BLE001
            return False
        return False

    if regions:
        lab = np.full(len(Pd), -1, dtype=np.int64)
        for ri, r in enumerate(regions):
            lab[r["v_idx"]] = ri
        la, lb = lab[E[:, 0]], lab[E[:, 1]]
        m = (la >= 0) & (lb >= 0) & (la != lb)
        pairs, counts = np.unique(
            np.sort(np.column_stack([la[m], lb[m]]), axis=1),
            axis=0, return_counts=True)
        parent = list(range(len(regions)))

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for (i, j), c in zip(pairs, counts):
            if c < 10:
                continue
            ri, rj = _find(int(i)), _find(int(j))
            if ri != rj and _compatible(regions[ri], regions[rj]):
                parent[rj] = ri
        groups: Dict[int, List[int]] = {}
        for i in range(len(regions)):
            groups.setdefault(_find(i), []).append(i)
        if len(groups) < len(regions):
            merged: List[dict] = []
            for root, members in groups.items():
                if len(members) == 1:
                    merged.append(regions[members[0]])
                    continue
                piece = np.concatenate(
                    [regions[i]["v_idx"] for i in members])
                kind = regions[root]["kind"]
                P = Pd[piece]
                fit_pts = P if len(P) <= max_fit_pts else \
                    P[rng.choice(len(P), max_fit_pts, replace=False)]
                refit = None
                try:
                    refit = FITTERS[kind](fit_pts, ctx)
                except Exception:  # noqa: BLE001
                    refit = None
                merged.append({"kind": kind,
                               "model": refit or regions[root]["model"],
                               "f_idx": piece, "v_idx": piece, "notes": []})
            log(f"[perfect:{safe}] merged {len(regions)} fragments → "
                f"{len(merged)} surfaces")
            regions = merged

    n_prim = len(regions)
    for piece in _components(remaining):
        if len(piece) >= freeform_min:
            regions.append({"kind": "freeform", "model": None,
                            "f_idx": piece, "v_idx": piece, "notes": []})
    regions.sort(key=lambda r: len(r["v_idx"]), reverse=True)
    covered = sum(len(r["v_idx"]) for r in regions)
    log(f"[perfect:{safe}] cloud RANSAC labeling: {len(regions)} region(s) "
        f"({n_prim} primitive, {len(regions) - n_prim} freeform), coverage "
        f"{covered / max(len(Pd), 1):.0%} of {len(Pd):,} pts")

    _snap_regions(regions, Pd, cfg, safe, log)
    return regions, min_region_pts


# ── mirror-symmetry detection (Mitra '06 lifted to the fitted object) ────

def _detect_mirror_symmetry(Pcloud: np.ndarray, regions: List[dict],
                            log, safe: str,
                            accept_med_mm: float = 20.0) -> Optional[dict]:
    """Vertical mirror-plane candidates from cloud PCA + dominant plane
    normals; each scored by reflecting a cloud subsample and measuring the
    median distance to the nearest original point. Deterministic."""
    from scipy.spatial import cKDTree
    sub = Pcloud[:: max(1, len(Pcloud) // 30000)]
    kd = cKDTree(sub)
    ctr = sub.mean(axis=0)

    cands: List[np.ndarray] = []
    h = sub[:, [0, 2]] - ctr[[0, 2]]
    cov = h.T @ h
    w_, v_ = np.linalg.eigh(cov)
    for k in range(2):
        n = np.array([v_[0, k], 0.0, v_[1, k]])
        cands.append(n / np.linalg.norm(n))
    for r in regions:
        if r["kind"] == "plane":
            nh = r["model"].normal - float(r["model"].normal @ _UP) * _UP
            ln = np.linalg.norm(nh)
            if ln > 0.5:
                cands.append(nh / ln)
    # dedupe by direction
    uniq: List[np.ndarray] = []
    for n in cands:
        if not any(abs(float(n @ u)) > 0.985 for u in uniq):
            uniq.append(n)

    probe = sub[:: max(1, len(sub) // 8000)]
    best = None
    for n in uniq:
        d0 = float(np.median(sub @ n))
        for delta in np.linspace(-0.15, 0.15, 13):
            d = d0 + delta
            refl = probe - 2.0 * ((probe @ n) - d)[:, None] * n[None]
            dist, _ = kd.query(refl, k=1)
            score = float(np.median(dist))
            if best is None or score < best["median_m"]:
                best = {"normal": n.tolist(), "offset": d,
                        "median_m": score}
    if best is None:
        return None
    best["median_mm"] = round(best["median_m"] * 1000, 1)
    best["accepted"] = best["median_m"] * 1000 <= accept_med_mm
    log(f"[perfect:{safe}] symmetry: mirror "
        f"{'FOUND' if best['accepted'] else 'not confirmed'} "
        f"(median reflection error {best['median_mm']} mm, "
        f"normal {np.round(best['normal'], 3).tolist()})")
    return best


def _region_meta(regions: List[dict]) -> List[dict]:
    out = []
    for r in regions:
        e = {"kind": r["kind"], "n_faces": int(len(r["f_idx"])),
             "n_vertices": int(len(r["v_idx"])), "snapped": r["notes"],
             "params": r["model"].params_dict(),
             "provenance": "tool_measured"}
        if r["kind"] == "cylinder":
            e["radius_m"] = round(float(r["model"].radius), 4)
        if r["kind"] == "sphere":
            e["radius_m"] = round(float(r["model"].radius), 4)
        out.append(e)
    return out


# ── PARTS VIEW: show what the system understood ──────────────────────────

def diagnose_object(output_dir: Path, instance_id: int,
                    cfg: Optional[dict] = None,
                    source: Optional[str] = None,
                    log=print) -> Optional[Path]:
    """Flat-colored parts view + inventory + symmetry verdict, published as
    ``tsdf/<label>_<id>_parts/`` — the user validates the UNDERSTANDING."""
    import open3d as o3d
    from segmentation.tsdf_export import _safe_label

    t0 = time.time()
    cfg = cfg or {}
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
    log(f"[perfect:{safe}] parts view from {src.parent.name}: "
        f"{len(Vd):,} verts / {len(F):,} faces")

    regions, _ = _detect_and_snap(tm, F, Vd, cfg, safe, log)

    pc = o3d.io.read_point_cloud(str(out / "cleaned_cloud.ply"))
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    xyz = np.asarray(pc.points)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    Pcloud = _to_display(out, xyz[gi])
    sym = _detect_mirror_symmetry(Pcloud, regions, log, safe)

    # flat color per part: kind-coded base + per-region shade so borders read
    base = {"plane": np.array([0.42, 0.58, 0.85]),
            "cylinder": np.array([0.95, 0.55, 0.15]),
            "sphere": np.array([0.25, 0.75, 0.40])}
    colors = np.full((len(Vd), 3), 0.22)              # freeform = dark gray
    rng = np.random.default_rng(7)
    for ri, r in enumerate(regions):
        tint = base[r["kind"]] * (0.75 + 0.5 * rng.random())
        colors[r["v_idx"]] = np.clip(tint, 0, 1)
        log(f"[perfect:{safe}]   part {ri}: {r['kind']}"
            + (f" r={r['model'].radius:.3f} m" if r["kind"] in ("cylinder", "sphere") else "")
            + f" · {len(r['f_idx']):,} faces"
            + (f" [{'; '.join(r['notes'])}]" if r["notes"] else ""))

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V_raw),
        o3d.utility.Vector3iVector(F))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    dst = out / "tsdf" / f"{safe}_parts"
    dst.mkdir(parents=True, exist_ok=True)
    glb = dst / f"{safe}_parts.glb"
    o3d.io.write_triangle_mesh(str(glb), mesh)
    kinds: Dict[str, int] = {}
    for r in regions:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    meta = {
        "method": "parts",
        "instance_id": int(instance_id),
        "label": f"{label} (parts)",
        "glb_file": glb.name,
        "source_mesh": src.parent.name,
        "inventory": kinds,
        "n_regions": len(regions),
        "parts": _region_meta(regions),
        "symmetry_mirror": sym,
        "freeform_vertices": int((colors[:, 0] == 0.22).sum()),
        "n_vertices": int(len(Vd)),
        "n_triangles": int(len(F)),
        "vertex_colors": True,
        "textured": False,
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }
    (dst / f"{safe}_parts.meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[perfect:{safe}] ✅ parts view: {kinds} · symmetry "
        f"{'FOUND' if (sym or {}).get('accepted') else 'not confirmed'} → "
        f"{glb.name} ({meta['elapsed_s']}s)")
    return glb


# ── PERFECT: iron the mesh onto the perfected surfaces ───────────────────

def perfect_object(output_dir: Path, instance_id: int,
                   cfg: Optional[dict] = None,
                   source: Optional[str] = None,
                   log=print) -> Optional[Path]:
    """Iron ONE segment's chosen mesh onto its perfected primitive regions.
    Publishes ``tsdf/<label>_<id>_perfect/``; returns the GLB path."""
    import open3d as o3d
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    from scipy.spatial import cKDTree
    from segmentation.tsdf_export import _safe_label

    t0 = time.time()
    cfg = cfg or {}
    blend_rings = int(cfg.get("perfect_blend_rings", 4))
    stiffness = float(cfg.get("perfect_iron_lambda", 5.0))

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
    if len(V_raw) < 1000 or len(F) < 1000:
        raise ValueError(f"{safe}: source mesh too small to perfect")
    Vd = _to_display(out, V_raw)
    log(f"[perfect:{safe}] source {src.parent.name}: "
        f"{len(Vd):,} verts / {len(F):,} faces")

    pc = o3d.io.read_point_cloud(str(out / "cleaned_cloud.ply"))
    xyz = np.asarray(pc.points)
    cols = np.asarray(pc.colors) if pc.has_colors() else None
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    Pcloud = _to_display(out, xyz[gi])
    Pcols = cols[gi] if cols is not None else None

    regions, _ = _detect_and_snap(tm, F, Vd, cfg, safe, log)
    if not regions:
        raise RuntimeError(f"{safe}: no primitive region passed the gates — "
                           "nothing to perfect")

    # ── displacement-field ironing
    n_v = len(Vd)
    vert_region = np.full(n_v, -1, dtype=np.int64)
    for ri, r in enumerate(regions):
        vert_region[r["v_idx"]] = ri
    e_all = tm.edges_unique
    Wg = sp.coo_matrix(
        (np.ones(len(e_all)), (e_all[:, 0], e_all[:, 1])),
        shape=(n_v, n_v))
    Wg = (Wg + Wg.T).tocsr()
    diff = np.zeros(n_v, dtype=bool)
    for v0, v1 in e_all:
        if vert_region[v0] != vert_region[v1]:
            if vert_region[v0] >= 0:
                diff[v0] = True
            if vert_region[v1] >= 0:
                diff[v1] = True
    ring = np.full(n_v, blend_rings, dtype=np.int64)
    ring[vert_region < 0] = 0
    frontier = np.nonzero(diff)[0]
    ring[frontier] = 0
    for depth in range(1, blend_rings):
        nxt = np.unique(Wg[frontier].indices)
        nxt = nxt[(ring[nxt] > depth) & (vert_region[nxt] >= 0)]
        if not len(nxt):
            break
        ring[nxt] = depth
        frontier = nxt
    w_v = np.zeros(n_v)
    in_reg = vert_region >= 0
    w_v[in_reg] = ring[in_reg] / float(blend_rings)

    targets = Vd.copy()
    region_cap = np.zeros(len(regions))
    for ri, r in enumerate(regions):
        idx = r["v_idx"]
        proj = _project_onto(r["kind"], r["model"], Vd[idx])
        disp = proj - Vd[idx]
        d_reg = np.abs(np.asarray(r["model"].signed_distance(Vd[idx])))
        cap = max(3.0 * float(np.sqrt(np.mean(d_reg ** 2))), 0.01)
        region_cap[ri] = cap
        mag = np.linalg.norm(disp, axis=1)
        over = mag > cap
        if over.any():
            disp[over] *= (cap / mag[over])[:, None]
        targets[idx] = Vd[idx] + disp

    # sharp creases: boundary verts between two regions → analytic
    # intersection via alternating projections
    crease_v: Dict[int, Tuple[int, int]] = {}
    for v0, v1 in e_all:
        r0, r1 = vert_region[v0], vert_region[v1]
        if r0 >= 0 and r1 >= 0 and r0 != r1:
            crease_v[int(v0)] = (int(min(r0, r1)), int(max(r0, r1)))
            crease_v[int(v1)] = (int(min(r0, r1)), int(max(r0, r1)))
    n_sharp = 0
    for v, (ra, rb) in crease_v.items():
        x = Vd[v].copy()
        A, B = regions[ra], regions[rb]
        for _ in range(6):
            x = _project_onto(A["kind"], A["model"], x[None])[0]
            x = _project_onto(B["kind"], B["model"], x[None])[0]
        cap = max(region_cap[ra], region_cap[rb])
        if np.linalg.norm(x - Vd[v]) <= cap:
            targets[v] = x
            w_v[v] = 1.0
            n_sharp += 1
    log(f"[perfect:{safe}] creases: {n_sharp:,}/{len(crease_v):,} boundary "
        f"verts snapped to analytic intersections")

    deg = np.asarray(Wg.sum(axis=1)).ravel()
    L = sp.diags(deg) - Wg
    Wd = sp.diags(stiffness * w_v)
    Asys = (L + Wd + 1e-6 * sp.identity(n_v)).tocsc()
    rhs = stiffness * w_v[:, None] * (targets - Vd)
    lu = splu(Asys)
    D = np.column_stack([lu.solve(rhs[:, k]) for k in range(3)])
    V_new = Vd + D
    moved = np.linalg.norm(D, axis=1)
    log(f"[perfect:{safe}] ironed: {int((moved > 1e-4).sum()):,} verts moved, "
        f"max {moved.max()*1000:.1f} mm, mean(region) "
        f"{moved[in_reg].mean()*1000:.1f} mm")

    # residuals vs the CLOUD, per region
    kd_cloud = cKDTree(Vd)
    cloud_sub = Pcloud[::max(1, len(Pcloud) // 400_000)]
    _, near_v = kd_cloud.query(cloud_sub, k=1)
    parts_meta = _region_meta(regions)
    for ri, r in enumerate(regions):
        sel = vert_region[near_v] == ri
        if sel.sum() >= 100:
            dc = np.abs(np.asarray(r["model"].signed_distance(cloud_sub[sel])))
            parts_meta[ri]["cloud_residuals"] = {
                "rms_mm": round(float(np.sqrt(np.mean(dc ** 2))) * 1000, 2),
                "p95_mm": round(float(np.percentile(dc, 95)) * 1000, 2),
                "n_cloud_pts": int(sel.sum())}

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(_to_raw(out, V_new)),
        o3d.utility.Vector3iVector(F))
    if Pcols is not None:
        _, nn = cKDTree(Pcloud).query(V_new, k=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(Pcols[nn])
    mesh.compute_vertex_normals()
    dst = out / "tsdf" / f"{safe}_perfect"
    dst.mkdir(parents=True, exist_ok=True)
    glb = dst / f"{safe}_perfect.glb"
    o3d.io.write_triangle_mesh(str(glb), mesh)
    meta = {
        "method": "perfect",
        "version": 2,
        "instance_id": int(instance_id),
        "label": f"{label} (perfect)",
        "glb_file": glb.name,
        "source_mesh": src.parent.name,
        "n_regions": len(regions),
        "parts": parts_meta,
        "freeform_vertices": int((~in_reg).sum()),
        "n_vertices": int(len(V_new)),
        "n_triangles": int(len(F)),
        "vertex_colors": bool(Pcols is not None),
        "textured": False,
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }
    (dst / f"{safe}_perfect.meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[perfect:{safe}] ✅ v2 ironed {len(regions)} region(s) → {glb.name} "
        f"({meta['elapsed_s']}s)")
    return glb


# ── MODEL REBUILD: clean CAD-like model from the validated understanding ──

def _reflect_pts(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return P - 2.0 * ((P @ n) - d)[:, None] * n[None]


def _reflect_dir(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - 2.0 * float(v @ n) * n


def build_model_object(output_dir: Path, instance_id: int,
                       cfg: Optional[dict] = None,
                       source: Optional[str] = None,
                       log=print) -> Optional[Path]:
    """CLEAN MODEL from the validated parts (user 2026-08-31: 'veamos qué
    sale de lo que detectó — que reconstruya'): every detected part is
    re-meshed from its perfect surface with a REGULARIZED outline
    (rectangle/polygon ladder from contours.py — CAD-crisp edges), mirror
    pairs share symmetrized parameters AND color, self-symmetric parts are
    symmetrized against their own reflection, and parts with no scanned
    twin are COMPLETED by mirroring (tagged + tinted — provenance visible).
    Freeform stays OUT of the model: this is the intent, not the scan.
    Published as ``tsdf/<label>_<id>_model/``."""
    import open3d as o3d
    from segmentation.tsdf_export import _safe_label
    from reconstruction.surface_fit.support import support_grid, mesh_on_surface
    from reconstruction.surface_fit.contours import regularize_mesh

    t0 = time.time()
    cfg = cfg or {}
    cell = float(cfg.get("perfect_model_res_m", 0.02))
    sup_r = float(cfg.get("perfect_support_radius_m", 0.06))
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
        raise RuntimeError(f"{safe}: nothing detected to model")

    pc = o3d.io.read_point_cloud(str(out / "cleaned_cloud.ply"))
    xyz = np.asarray(pc.points)
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    Pcloud = _to_display(out, xyz[gi])
    sym = _detect_mirror_symmetry(Pcloud, regions, log, safe)

    # ── symmetry pairing + symmetrization
    pair_of = {}
    self_sym = set()
    if sym and sym.get("accepted"):
        n_s = np.asarray(sym["normal"], np.float64)
        d_s = float(sym["offset"])
        cent = [Vd[r["v_idx"]].mean(axis=0) for r in regions]
        diag = [float(np.linalg.norm(Vd[r["v_idx"]].ptp(axis=0)))
                for r in regions]
        used = set()
        for i, r in enumerate(regions):
            if i in used:
                continue
            ci_r = _reflect_pts(cent[i][None], n_s, d_s)[0]
            if np.linalg.norm(ci_r - cent[i]) < max(0.10, 0.15 * diag[i]):
                self_sym.add(i)
                used.add(i)
                continue
            best = None
            for j, q in enumerate(regions):
                if j == i or j in used or q["kind"] != r["kind"]:
                    continue
                dctr = float(np.linalg.norm(cent[j] - ci_r))
                if dctr > max(0.15, 0.25 * diag[i]):
                    continue
                okp = True
                if r["kind"] == "plane":
                    okp = abs(float(_reflect_dir(r["model"].normal, n_s)
                                    @ q["model"].normal)) > np.cos(np.deg2rad(12))
                elif r["kind"] == "cylinder":
                    okp = (abs(float(_reflect_dir(r["model"].axis_dir, n_s)
                                     @ q["model"].axis_dir)) > np.cos(np.deg2rad(12))
                           and abs(r["model"].radius - q["model"].radius)
                           < 0.15 * max(r["model"].radius, q["model"].radius))
                elif r["kind"] == "sphere":
                    okp = abs(r["model"].radius - q["model"].radius) \
                        < 0.2 * max(r["model"].radius, q["model"].radius)
                if okp and (best is None or dctr < best[1]):
                    best = (j, dctr)
            if best is not None:
                j = best[0]
                pair_of[i], pair_of[j] = j, i
                used.add(i); used.add(j)

        # symmetrize: refit each side on OWN + reflected-partner points
        done = set()
        for i, j in list(pair_of.items()):
            if i in done:
                continue
            done.add(i); done.add(j)
            A, B = regions[i], regions[j]
            Pa, Pb = Vd[A["v_idx"]], Vd[B["v_idx"]]
            comb_a = np.vstack([Pa, _reflect_pts(Pb, n_s, d_s)])
            comb_b = np.vstack([Pb, _reflect_pts(Pa, n_s, d_s)])
            try:
                if A["kind"] == "plane":
                    na = A["model"].normal
                    nb_r = _reflect_dir(B["model"].normal, n_s)
                    if float(nb_r @ na) < 0:
                        nb_r = -nb_r
                    n_avg = (na + nb_r)
                    n_avg /= max(np.linalg.norm(n_avg), 1e-12)
                    A["model"] = _rebuild_plane(comb_a, n_avg, na)
                    B["model"] = _rebuild_plane(
                        comb_b, _reflect_dir(n_avg, n_s), B["model"].normal)
                elif A["kind"] == "cylinder":
                    wa = A["model"].axis_dir
                    wb_r = _reflect_dir(B["model"].axis_dir, n_s)
                    if float(wb_r @ wa) < 0:
                        wb_r = -wb_r
                    w_avg = (wa + wb_r)
                    w_avg /= max(np.linalg.norm(w_avg), 1e-12)
                    A["model"] = _rebuild_cylinder(comb_a, w_avg)
                    B["model"] = _rebuild_cylinder(
                        comb_b, _reflect_dir(w_avg, n_s),
                        radius_override=A["model"].radius)
                A["notes"].append(f"symmetrized with part {j}")
                B["notes"].append(f"symmetrized with part {i}")
            except Exception as e:  # noqa: BLE001
                log(f"[perfect:{safe}] symmetrize {i}↔{j} failed ({e})")
        for i in self_sym:
            r = regions[i]
            P0 = Vd[r["v_idx"]]
            comb = np.vstack([P0, _reflect_pts(P0, n_s, d_s)])
            try:
                if r["kind"] == "plane":
                    r["model"] = _rebuild_plane(comb, r["model"].normal,
                                                r["model"].normal)
                elif r["kind"] == "cylinder":
                    r["model"] = _rebuild_cylinder(comb, r["model"].axis_dir)
                r["notes"].append("self-symmetrized")
            except Exception:  # noqa: BLE001
                pass
        log(f"[perfect:{safe}] symmetry: {len(pair_of)//2} mirror pair(s), "
            f"{len(self_sym)} self-symmetric, "
            f"{len(regions) - len(pair_of) - len(self_sym)} unpaired")

    # ── VLM shape proposal (if one was generated): relation-driven snaps
    # (gated) + part roles into the inventory. The proposal only ever
    # SUGGESTS; every applied snap must survive the same rms gate.
    proposal = _load_shape_proposal(out, safe)
    prop_match: Dict[int, dict] = {}
    intents_applied = 0
    drop_open: set = set()
    if proposal:
        prop_match = _match_proposal(regions, Vd, proposal)
        log(f"[perfect:{safe}] shape proposal found: "
            f"{len(prop_match)}/{len(regions)} region(s) matched "
            f"(object: {(proposal.get('object') or {}).get('identity')})")
        # regions the VLM says are OPENINGS (holes, not material — e.g. a
        # sphere fitted on a hole's rim) are DROPPED from the rebuild; the
        # hole stays a hole in its host plate (user 2026-09-03)
        # drop requires BOTH: the VLM proposes 'opening' AND the measured
        # void evidence agrees (interior_void_ratio ≥ 0.5 — the region is a
        # ring of support around enclosed emptiness). Without the measurement
        # the VLM's guess dropped real plates (observed 2026-09-03) —
        # measurement decides, the VLM only proposes.
        for i, e in prop_match.items():
            pr = e.get("proposal") or {}
            if (pr.get("proposed_kind") == "opening"
                    and float(pr.get("confidence", 0.0)) >= 0.5
                    and float(e.get("interior_void_ratio") or 0.0) >= 0.5):
                drop_open.add(i)
        if drop_open:
            log(f"[perfect:{safe}] {len(drop_open)} region(s) proposed as "
                f"OPENINGS — excluded from the model: {sorted(drop_open)}")
        intent_match = {i: e for i, e in prop_match.items()
                        if i not in drop_open}
        intents_applied = _apply_proposal_intents(regions, Vd, intent_match,
                                                  log, safe)

    # ── clean re-mesh per part (regularized outlines: CAD-crisp)
    final = o3d.geometry.TriangleMesh()
    parts_meta: List[dict] = []
    rng = np.random.default_rng(11)
    pair_color: Dict[int, np.ndarray] = {}
    completed = 0
    for i, r in enumerate(regions):
        Pm = Vd[r["v_idx"]]
        model = r["model"]
        prop_e = (prop_match.get(i) or {}).get("proposal") or {}
        role = str(prop_e.get("part_role") or "") or None
        location = str(prop_e.get("location") or "") or None
        if i in drop_open:
            parts_meta.append({
                "kind": r["kind"], "n_faces_src": int(len(r["f_idx"])),
                "role": role, "location": location,
                "role_provenance": "vlm_proposed",
                "dropped_as_opening": True,
                "provenance": "vlm_proposed"})
            continue
        uv = np.asarray(model.to_uv(Pm))
        V_part = F_part = None
        contour_shape = None
        try:
            grid, u0, v0 = support_grid(uv, cell, sup_r)
            rm = regularize_mesh(model, grid, None, u0, v0, cell)
            if rm is not None:
                verts_uv, faces_uv, reports = rm
                if len(verts_uv) >= 3 and len(faces_uv) >= 1:
                    V_part = np.asarray(model.uv_to_world(verts_uv))
                    F_part = np.asarray(faces_uv, np.int64)
                    if reports:
                        contour_shape = reports[0].get("shape")
        except Exception as e:  # noqa: BLE001
            log(f"[perfect:{safe}] contour regularization failed on part {i} "
                f"({e}) — trimmed grid kept")
        if V_part is None:
            Vt, Ft, _, _ = mesh_on_surface(uv, model.uv_to_world, cell, sup_r)
            if len(Vt) == 0:
                continue
            V_part, F_part = np.asarray(Vt), np.asarray(Ft, np.int64)

        j = pair_of.get(i)
        if j is not None and j in pair_color:
            color = pair_color[j]
        else:
            base = {"plane": np.array([0.72, 0.74, 0.78]),
                    "cylinder": np.array([0.85, 0.62, 0.35]),
                    "sphere": np.array([0.55, 0.75, 0.55])}[r["kind"]]
            color = np.clip(base * (0.8 + 0.4 * rng.random()), 0, 1)
        pair_color[i] = color

        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(_to_raw(out, V_part)),
            o3d.utility.Vector3iVector(F_part))
        m.paint_uniform_color(color)
        m.compute_vertex_normals()
        final += m

        # mirror-complete parts with no scanned twin (not self-symmetric) —
        # ONLY into genuinely UNSCANNED space: if the cloud already covers
        # the mirrored location, the real thing is there (even if detection
        # missed it) and a ghost copy would duplicate geometry
        if sym and sym.get("accepted") and j is None and i not in self_sym:
            from scipy.spatial import cKDTree as _KD
            if "_kd_cloud" not in dir():
                _kd_cloud = _KD(Pcloud[:: max(1, len(Pcloud) // 200_000)])
            probe = _reflect_pts(
                Pm[:: max(1, len(Pm) // 2000)],
                np.asarray(sym["normal"]), float(sym["offset"]))
            dnn, _ = _kd_cloud.query(probe, k=1)
            covered = float((dnn < 0.05).mean())
            if covered > 0.5:
                parts_meta_note = f"twin location already scanned ({covered:.0%}) — not completed"
                d = np.abs(np.asarray(model.signed_distance(Pm)))
                e = {"kind": r["kind"], "n_faces_src": int(len(r["f_idx"])),
                     "role": role, "location": location,
                     "role_provenance": "vlm_proposed" if role else None,
                     "contour": contour_shape, "snapped": r["notes"] + [parts_meta_note],
                     "mirror_pair": None, "self_symmetric": False,
                     "rms_mm": round(float(np.sqrt(np.mean(d ** 2))) * 1000, 2),
                     "p95_mm": round(float(np.percentile(d, 95)) * 1000, 2),
                     "params": model.params_dict(),
                     "provenance": "tool_measured"}
                if r["kind"] in ("cylinder", "sphere"):
                    e["radius_m"] = round(float(model.radius), 4)
                parts_meta.append(e)
                continue
            Vm = _reflect_pts(V_part, np.asarray(sym["normal"]),
                              float(sym["offset"]))
            mc = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(_to_raw(out, Vm)),
                o3d.utility.Vector3iVector(F_part[:, ::-1].copy()))
            mc.paint_uniform_color(np.clip(color * 0.7 + np.array([0.05, 0.15, 0.35]), 0, 1))
            mc.compute_vertex_normals()
            final += mc
            completed += 1

        d = np.abs(np.asarray(model.signed_distance(Pm)))
        e = {"kind": r["kind"], "n_faces_src": int(len(r["f_idx"])),
             "role": role, "location": location,
             "role_provenance": "vlm_proposed" if role else None,
             "contour": contour_shape, "snapped": r["notes"],
             "mirror_pair": j, "self_symmetric": bool(i in self_sym),
             "rms_mm": round(float(np.sqrt(np.mean(d ** 2))) * 1000, 2),
             "p95_mm": round(float(np.percentile(d, 95)) * 1000, 2),
             "params": model.params_dict(), "provenance": "tool_measured"}
        if r["kind"] in ("cylinder", "sphere"):
            e["radius_m"] = round(float(model.radius), 4)
        parts_meta.append(e)

    if len(final.vertices) == 0:
        raise RuntimeError(f"{safe}: model rebuild produced nothing")

    dst = out / "tsdf" / f"{safe}_model"
    dst.mkdir(parents=True, exist_ok=True)
    glb = dst / f"{safe}_model.glb"
    o3d.io.write_triangle_mesh(str(glb), final)
    meta = {
        "method": "model",
        "instance_id": int(instance_id),
        "label": f"{label} (model)",
        "glb_file": glb.name,
        "source_mesh": src.parent.name,
        "n_parts": len(parts_meta),
        "parts": parts_meta,
        "symmetry_mirror": sym,
        "shape_proposal": ({
            "object": proposal.get("object"),
            "matched_regions": len(prop_match),
            "intents_applied": intents_applied,
            "openings_dropped": sorted(int(i) for i in drop_open),
        } if proposal else None),
        "mirror_completed_parts": completed,
        "n_vertices": int(len(final.vertices)),
        "n_triangles": int(len(final.triangles)),
        "vertex_colors": True,
        "textured": False,
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }
    (dst / f"{safe}_model.meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[perfect:{safe}] ✅ model: {len(parts_meta)} clean part(s), "
        f"{completed} mirror-completed → {glb.name} ({meta['elapsed_s']}s)")
    return glb
