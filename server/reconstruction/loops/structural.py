"""Structural constraints — the work as its own control (claude_stac.txt §4.6b).

The case that breaks any loop graph is the out-and-back along a straight
corridor: a closure at the end pins the endpoint, but the lateral bend and the
yaw of the middle stay nearly unobservable and the graph SPREADS them instead
of removing them. Two defences, both mandatory: loop density (§4.6a, measured
by :func:`loop_coverage`) and structural edges on the keyframe graph, all
robust (Huber), all only on instances classified ``structural``, all with a
configurable tolerance because the real work is not perfect, all reported
with their residual:

  * FLOOR DATUM — per-keyframe floor patches onto one plane along the walk
    (the datum = the floor of the first ``reference_span_kf`` keyframes, the
    drift-free start — the correction/floor precedent); a patch farther than
    ``step_demote_m`` from the datum is a REAL step or slope change and gets
    no edge (the step survives, only drift is removed). Attacks the vertical
    drift ("chimney 16 cm apart in y").
  * WALL PLANARITY — a wall segmented as one instance over tens of metres
    constrains the lateral bend directly: every keyframe's patch of the wall
    must land on the wall's plane, itself a NODE of the graph solved jointly
    with the poses (the reference plane of the earliest patches is only its
    start). Tolerated residual ``wall_tol_m`` (Huber width of the offset).
  * COLUMN VERTICALITY and PARALLELISM of repeated elements (beams, rails,
    luminaires): per-keyframe axes must stay vertical / parallel to the
    element's global axis.
  * REGULATED DIMENSIONS — track gauge, platform height, clearance… present
    in every metro scene: a nitidly segmented instance of a regulated class
    yields an ABSOLUTE scale row (§5.2) with σ_regulated
    (``scale_absolute_rows.json``, consumed by the scale graph). Empty by
    default, loaded per project.

Every function here MEASURES on the cloud and returns edge parameters for the
solver (loop_utils.pose_graph); nothing proposes geometry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _cfg(c) -> Dict[str, Any]:
    return asdict(c) if is_dataclass(c) else dict(c)


# ── loop coverage (§4.6a) ────────────────────────────────────────────────────

def loop_coverage(centres: np.ndarray, loop_endpoints: Sequence[Tuple[int, int]],
                  radius_m: float) -> Dict[str, Any]:
    """Fraction of the trajectory length within ``radius_m`` (walked) of an
    accepted loop edge endpoint; the uncovered stretches are listed so the
    kit visual can show where the graph has no witness."""
    C = np.asarray(centres, np.float64)
    n = len(C)
    if n < 2:
        return {"coverage": 0.0, "total_m": 0.0, "covered_m": 0.0, "uncovered": []}
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    chain = np.concatenate([[0.0], np.cumsum(steps)])
    anchors = sorted({int(k) for e in loop_endpoints for k in e if 0 <= int(k) < n})
    covered = np.zeros(n - 1, bool)
    for a in anchors:
        d = np.abs(chain - chain[a])
        covered |= (d[:-1] <= radius_m) | (d[1:] <= radius_m)
    total = float(chain[-1])
    cov_m = float(steps[covered].sum())
    unc = []
    start = None
    for k in range(n - 1):
        if not covered[k] and start is None:
            start = k
        if (covered[k] or k == n - 2) and start is not None:
            end = k if covered[k] else k + 1
            unc.append({"kf_from": int(start), "kf_to": int(end),
                        "length_m": float(chain[end] - chain[start])})
            start = None
    return {"coverage": (cov_m / total) if total > 0 else 0.0, "total_m": total,
            "covered_m": cov_m, "radius_m": float(radius_m), "n_anchors": len(anchors),
            "uncovered": unc}


# ── geometry helpers ─────────────────────────────────────────────────────────

def _plane_svd(P: np.ndarray):
    c = P.mean(0)
    _, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    return n / (np.linalg.norm(n) + 1e-12), float(n @ c), S


def weighted_plane(P: np.ndarray, w: np.ndarray):
    """Weighted LS plane n·p = d (weighted centroid + weighted covariance SVD)."""
    w = np.asarray(w, np.float64)
    w = w / max(float(w.sum()), 1e-12)
    c = (P * w[:, None]).sum(0)
    X = (P - c) * np.sqrt(w)[:, None]
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[2] / (np.linalg.norm(Vt[2]) + 1e-12)
    return n, float(n @ c)


def ransac_plane(P: np.ndarray, tol_m: float, iters: int, rng: np.random.Generator):
    """(n, d, inlier_mask) with n·p = d; SVD refit on the inliers."""
    best_inl, best = None, None
    n_pts = len(P)
    if n_pts < 3:
        return None
    for _ in range(int(iters)):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = P[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        d = float(n @ p0)
        inl = np.abs(P @ n - d) <= tol_m
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best = inl, (n, d)
    if best is None or best_inl.sum() < 3:
        return None
    n, d, _ = _plane_svd(P[best_inl])
    inl = np.abs(P @ n - d) <= tol_m
    return n, d, inl


def _to_local(T: np.ndarray, n_w: np.ndarray, d_w: float):
    """World plane n·p = d → camera-frame plane of pose T (c2w)."""
    R, t = T[:3, :3], T[:3, 3]
    n_l = R.T @ n_w
    d_l = float(d_w - n_w @ t)
    return n_l, d_l


# ── floor datum ──────────────────────────────────────────────────────────────

def floor_datum_edges(session, per_kf_points, up: np.ndarray, cfg, seed: int = 0):
    """Per-keyframe floor patch (low height band, RANSAC plane) → edge onto the
    datum plane fitted on the first ``reference_span_kf`` patches. Returns
    (edges, report); edges = [(kf, n_local, d_local, n_target, d_target)]."""
    c = _cfg(cfg)
    rng = np.random.default_rng(seed)
    up = np.asarray(up, np.float64); up = up / (np.linalg.norm(up) + 1e-12)
    patches: Dict[int, Tuple[np.ndarray, float, int]] = {}
    for k in range(session.n_kf):
        P = per_kf_points(k)
        if len(P) < int(c["min_points"]):
            continue
        h = P @ up
        lo = np.percentile(h, float(c["low_band_pct"]))
        band = P[(h >= lo - float(c["band_m"])) & (h <= lo + float(c["band_m"]))]
        if len(band) < int(c["min_points"]):
            continue
        fit = ransac_plane(band, float(c["ransac_tol_m"]), int(c["ransac_iters"]), rng)
        if fit is None:
            continue
        n, d, inl = fit
        if n @ up < 0:
            n, d = -n, -d
        if float(n @ up) < np.cos(np.radians(float(c["max_tilt_deg"]))):
            continue          # not a floor (a wall band) — the patch is not horizontal
        if inl.sum() < int(c["min_points"]):
            continue
        patches[k] = (n, d, int(inl.sum()))
    if not patches:
        return [], {"n_patches": 0, "reason": "no floor patch on any keyframe"}
    ref_kfs = sorted(k for k in patches if k < int(c["reference_span_kf"]))
    if not ref_kfs:
        ref_kfs = sorted(patches)[:int(c["reference_span_kf"])]
    # datum: weighted mean of the reference patches (normal + offset)
    w = np.array([patches[k][2] for k in ref_kfs], np.float64)
    n_star = np.sum([patches[k][0] * w[i] for i, k in enumerate(ref_kfs)], axis=0)
    n_star = n_star / (np.linalg.norm(n_star) + 1e-12)
    d_star = float(np.sum([patches[k][1] * w[i] for i, k in enumerate(ref_kfs)]) / w.sum())
    edges, demoted = [], []
    for k, (n, d, npts) in sorted(patches.items()):
        # the patch's offset measured along the DATUM normal
        off = abs(d - d_star) if float(n @ n_star) > 0 else abs(-d - d_star)
        if off > float(c["step_demote_m"]):
            demoted.append({"kf": int(k), "offset_m": float(off),
                            "reason": "real step / level change (kept, no edge)"})
            continue
        n_l, d_l = _to_local(session.poses[k], n, d)
        edges.append((int(k), n_l, d_l, n_star, d_star))
    rep = {"n_patches": len(patches), "n_edges": len(edges), "reference_kfs": ref_kfs,
           "datum_normal": n_star.tolist(), "datum_offset_m": d_star,
           "demoted": demoted}
    return edges, rep


# ── wall planarity ───────────────────────────────────────────────────────────

def wall_patches(session, inst_points_idx: np.ndarray, cfg):
    """Per-keyframe patches of one wall instance: {kf: (n_local, d_local,
    n_world, d_world, n_points)} — local in camera coords (from the CURRENT
    poses), world for the plane targets — plus the wall's span along its long
    axis."""
    c = _cfg(cfg)
    P_all = session.xyz[inst_points_idx]
    if len(P_all) < 3:
        return {}, 0.0
    n_g, d_g, S = _plane_svd(P_all)
    c_all = P_all.mean(0)
    _, _, Vt_all = np.linalg.svd(P_all - c_all, full_matrices=False)
    along = (P_all - c_all) @ Vt_all[0]
    span = float(along.max() - along.min())
    ks = session.ks[inst_points_idx]
    patches = {}
    for k in np.unique(ks):
        if k < 0:
            continue
        P = P_all[ks == k]
        if len(P) < int(c["min_points_per_kf"]):
            continue
        n, d, S = _plane_svd(P)
        # a keyframe's patch votes only when it is a well-conditioned PLANE of
        # some extent: a sliver seen at the corridor's end has a noisy normal
        # that would ask the node for a rotation the wall never observed
        ev = (S ** 2) / max(len(P) - 1, 1)
        if ev[2] > float(c["planar_ratio"]) * max(ev[0], 1e-12):
            continue
        cP = P.mean(0)
        _, _, VtP = np.linalg.svd(P - cP, full_matrices=False)
        ext2 = (P - cP) @ VtP[1]
        if float(ext2.max() - ext2.min()) < float(c["min_patch_extent_m"]):
            continue
        if float(n @ n_g) < 0:
            n, d = -n, -d
        n_l, d_l = _to_local(session.poses[int(k)], n, d)
        patches[int(k)] = (n_l, d_l, n, d, int(len(P)))
    return patches, span


def reference_plane(patches: Dict[int, tuple], span_kf: int):
    """The wall's datum: the plane of its EARLIEST ``span_kf`` keyframe patches
    (weighted by their point count) — the least-drifted observation of the
    wall, the same principle as the floor datum (reference_span_kf). A plane
    fitted over the whole drifted wall would follow the bend and pull the
    poses to a wrong, averaged plane."""
    ks = sorted(patches)
    ref = ks[:max(1, int(span_kf))]
    w = np.array([patches[k][4] for k in ref], np.float64)
    n = np.sum([patches[k][2] * w[i] for i, k in enumerate(ref)], axis=0)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(np.sum([patches[k][3] * w[i] for i, k in enumerate(ref)]) / w.sum())
    return n, d, ref


def wall_edges(session, instances: List[dict], classes: Dict[int, str], cfg,
               targets: Optional[Dict[int, tuple]] = None):
    """[(kf, n_local, d_local, n_target, d_target, wall_id)] for every long
    structural wall + a report. Targets: the reference planes of the walls
    (earliest patches) unless the caller passes re-estimated ones."""
    c = _cfg(cfg)
    labels = set(str(x).lower() for x in c["labels"])
    edges, rep = [], []
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        if classes.get(iid, "movable") != "structural":
            continue
        if str(inst.get("label", "")).lower() not in labels:
            continue
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < session.n_points)]
        patches, span = wall_patches(session, gi, c)
        if span < float(c["min_span_m"]) or len(patches) < 2:
            rep.append({"instance_id": iid, "span_m": span, "n_patches": len(patches),
                        "used": False})
            continue
        if targets and iid in targets:
            n_t, d_t = targets[iid]
            ref = None
        else:
            n_t, d_t, ref = reference_plane(patches, int(c["reference_span_kf"]))
        # the target normal must point like the patches' (a re-fitted plane
        # comes back with an arbitrary sign; n_w − n_t on opposite normals
        # would ask for a half turn)
        n_mean = np.sum([patches[k][2] * patches[k][4] for k in patches], axis=0)
        if float(n_mean @ n_t) < 0:
            n_t, d_t = -np.asarray(n_t), -float(d_t)
        for k, (n_l, d_l, _n, _d, npts) in patches.items():
            edges.append((k, n_l, d_l, n_t, d_t, iid))
        rep.append({"instance_id": iid, "span_m": span, "n_patches": len(patches),
                    "used": True, "reference_kfs": ref})
    return edges, rep


# ── axes: columns vertical, repeated elements parallel ──────────────────────

def axis_edges(session, instances: List[dict], classes: Dict[int, str], cfg,
               up: np.ndarray, vertical: bool):
    """[(kf, axis_local, target_axis, iid)]: per-keyframe principal axis of a
    column (target = up) or of a repeated element (target = its global axis)."""
    c = _cfg(cfg)
    labels = set(str(x).lower() for x in c["labels"])
    up = np.asarray(up, np.float64); up = up / (np.linalg.norm(up) + 1e-12)
    edges, rep = [], []
    ratio = float(c["axis_ratio"])
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        if classes.get(iid, "movable") != "structural":
            continue
        if str(inst.get("label", "")).lower() not in labels:
            continue
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < session.n_points)]
        if len(gi) < 3:
            continue
        P_all = session.xyz[gi]
        _, S_all, Vt_all = np.linalg.svd(P_all - P_all.mean(0), full_matrices=False)
        a_g = Vt_all[0]
        if vertical:
            target = up
        else:
            target = a_g / (np.linalg.norm(a_g) + 1e-12)
        if float(target @ a_g) < 0:
            a_g = -a_g
        ks = session.ks[gi]
        n_e = 0
        for k in np.unique(ks):
            if k < 0:
                continue
            P = P_all[ks == k]
            if len(P) < int(c["min_points_per_kf"]):
                continue
            _, S, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
            if S[1] > ratio * max(S[0], 1e-12):
                continue          # the patch has no axis in this view (a partial face)
            a = Vt[0]
            if float(a @ target) < 0:
                a = -a
            a_l = session.poses[int(k)][:3, :3].T @ a
            edges.append((int(k), a_l, target, iid))
            n_e += 1
        rep.append({"instance_id": iid, "n_edges": n_e, "vertical": vertical})
    return edges, rep


# ── regulated dimensions → absolute scale rows (§5.2) ───────────────────────

def _circle_fit(xy: np.ndarray):
    """Algebraic (Kåsa) circle fit x² + y² + a·x + b·y + c = 0 → (centre, r).
    Works on a partial arc — a column is seen from one side."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -sol[0] / 2.0, -sol[1] / 2.0
    r2 = cx * cx + cy * cy - sol[2]
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return np.array([cx, cy]), float(np.sqrt(r2))


def _instance_dimension(P: np.ndarray, dimension: str, up: np.ndarray,
                        pct_lo: float, pct_hi: float) -> Optional[float]:
    """Model dimension of one instance from its points: height = supported
    extent along up; diameter = 2·r of the circle fitted to the horizontal
    projection (a partial arc is fine — a column is seen from one side, so a
    centroid-based radius would halve it); width / gauge = supported extent
    across the horizontal long axis."""
    if len(P) < 10:
        return None
    c = P.mean(0)
    X = P - c
    if dimension == "height":
        h = X @ up
        return float(np.percentile(h, pct_hi) - np.percentile(h, pct_lo))
    horiz = X - np.outer(X @ up, up)
    _, S, Vt = np.linalg.svd(horiz, full_matrices=False)
    if dimension == "diameter":
        e1 = Vt[0]
        e2 = np.cross(up, e1)
        xy = np.column_stack([horiz @ e1, horiz @ e2])
        fit = _circle_fit(xy)
        return None if fit is None else float(2.0 * fit[1])
    across = horiz @ Vt[1]
    return float(np.percentile(across, pct_hi) - np.percentile(across, pct_lo))


def regulated_rows(session, instances: List[dict], classes: Dict[int, str],
                   dims, sigma_regulated: float, up: np.ndarray,
                   chunk_ranges: Optional[List[List[int]]] = None,
                   dims_pct: Tuple[float, float] = (0.0, 100.0)) -> List[dict]:
    """Absolute scale rows from regulated dimensions: log(value/measured) with
    σ_regulated (inflated by the dimension tolerance), attributed to the
    chunk(s) the instance was measured in (all chunks when no plan)."""
    up = np.asarray(up, np.float64); up = up / (np.linalg.norm(up) + 1e-12)
    rows = []
    for rd in dims:
        d = _cfg(rd)
        for inst in instances:
            iid = int(inst.get("instance_id", inst.get("id")))
            if classes.get(iid, "movable") != "structural":
                continue
            if str(inst.get("label", "")).lower() != str(d["label"]).lower():
                continue
            gi = np.asarray(inst.get("globalIndices") or [], np.int64)
            gi = gi[(gi >= 0) & (gi < session.n_points)]
            meas = _instance_dimension(session.xyz[gi], d["dimension"], up,
                                       float(dims_pct[0]), float(dims_pct[1]))
            if meas is None or meas <= 0:
                continue
            log_s = float(np.log(float(d["value_m"]) / meas))
            sig = float(np.sqrt(sigma_regulated ** 2 + (float(d["tol_m"]) / float(d["value_m"])) ** 2))
            ks = np.unique(session.ks[gi]); ks = ks[ks >= 0]
            chunks = []
            if chunk_ranges:
                mid = int(np.median(ks)) if len(ks) else 0
                chunks = [ci for ci, (a, b) in enumerate(chunk_ranges) if a <= mid < b]
            for ck in (chunks or [None]):
                rows.append({"chunk": ck, "log_s": log_s, "sigma": sig,
                             "source": f"regulated:{d['label']}:{d['dimension']}",
                             "instance_id": iid, "measured_m": meas,
                             "value_m": float(d["value_m"]), "provenance": "tool_measured"})
    return rows


def write_absolute_rows(output_dir, rows: List[dict], chunk_ranges: Optional[List[List[int]]]):
    """scale_absolute_rows.json — every row needs a chunk to enter the chunk
    scale graph; rows without a plan are attributed to every chunk."""
    out = []
    for r in rows:
        if r.get("chunk") is None:
            for ck in range(len(chunk_ranges or [])):
                out.append(dict(r, chunk=int(ck)))
        else:
            out.append(dict(r, chunk=int(r["chunk"])))
    p = Path(output_dir) / "scale_absolute_rows.json"
    p.write_text(json.dumps({"version": 1, "rows": out}, indent=1))
    return p
