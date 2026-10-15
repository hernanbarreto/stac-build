"""Manhattan-world regularization of the fitted planes (USER 2026-10-15: "manhattan world para que
piso, paredes, puerta y todo lo que se indique como manhattan queden bien rectos, BIM perfecto").

A modelling constraint, never a reconstruction one: the cloud and the poses are untouched (the
gravity prior in the pose graph bent the chain on pccr and was retired). What it does, on the
planes of the objects the user left marked Manhattan (segmentation.manhattan, default all):

1. THE FRAME, measured on the scene: up = the normal of the heaviest marked plane (the floor on
   every scene shot so far); the two horizontal axes = the dominant direction of the marked
   walls (weighted circular mean of their azimuths folded modulo 90°) and its perpendicular.
   The tilt of the measured up against the hint and every wall's residual to its axis are
   recorded.
2. THE SNAP: each marked plane's normal goes EXACTLY to the nearest frame axis when it lies
   within ``snap_deg`` of it (a BOUND, declared: beyond it the surface is not a Manhattan
   surface — it keeps its measured normal and says so); its offset is re-measured as the
   robust median of its own points along the snapped normal; its in-plane basis follows the
   frame (floors: u = X, walls: v = up), so the contour regularization's rectangles come out
   axis-aligned.
3. COPLANAR marked planes on one axis within ``coplanar_tol_mm`` become ONE plane (weighted
   offset) — a wall split by SAM3 into pieces is one wall.

Every number the snap moved — degrees of normal, millimetres of offset — is in the report and
next to the model; the residual report against the ORIGINAL points says what the straight model
does not explain."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.geometry.primitives import _orthobasis
from .plane import PlaneModel

logger = logging.getLogger("SurfaceFit")
_EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), _EPS)


def scene_frame(normals: Sequence[np.ndarray], weights: Sequence[float], up_hint: np.ndarray,
                snap_deg: float) -> dict:
    """The orthonormal frame (up, X, Z) measured on the marked planes."""
    up_hint = _unit(up_hint)
    n = [_unit(v) for v in normals]
    w = np.asarray(weights, np.float64)
    if not len(n):
        e1, e2 = _orthobasis(up_hint)
        return {"up": up_hint, "X": e1, "Z": e2, "up_from": None, "up_tilt_deg": 0.0,
                "azimuth_deg": 0.0, "n_walls": 0}
    # up = the heaviest marked plane, sign along the hint
    h = int(np.argmax(w))
    up = n[h] * (1.0 if n[h] @ up_hint >= 0 else -1.0)
    tilt = float(np.rad2deg(np.arccos(np.clip(abs(up @ up_hint), 0.0, 1.0))))
    e1, e2 = _orthobasis(up)
    # the walls: normals within snap_deg of horizontal (relative to the measured up)
    s, c, n_walls = 0.0, 0.0, 0
    heaviest = None                      # (weight, horizontal normal) of the heaviest wall
    for ni, wi in zip(n, w):
        ang_to_horizontal = float(np.rad2deg(np.arcsin(np.clip(abs(ni @ up), 0.0, 1.0))))
        if ang_to_horizontal > snap_deg:
            continue
        hz = ni - (ni @ up) * up
        if np.linalg.norm(hz) < _EPS:
            continue
        phi = float(np.arctan2(hz @ e2, hz @ e1))
        s += wi * np.sin(4.0 * phi)
        c += wi * np.cos(4.0 * phi)
        n_walls += 1
        if heaviest is None or wi > heaviest[0]:
            heaviest = (float(wi), _unit(hz))
    theta = float(np.arctan2(s, c) / 4.0) if n_walls else 0.0
    X = _unit(np.cos(theta) * e1 + np.sin(theta) * e2)
    Z = _unit(np.cross(up, X))
    # X is the axis of the HEAVIEST wall (sign along its normal), Z = up × X — a named,
    # reproducible convention instead of whichever fold of the azimuth came first
    if heaviest is not None:
        if abs(float(heaviest[1] @ Z)) > abs(float(heaviest[1] @ X)):
            X = Z
        X = X * (1.0 if float(heaviest[1] @ X) >= 0 else -1.0)
        Z = _unit(np.cross(up, X))
    return {"up": up, "X": X, "Z": Z, "up_from": h, "up_tilt_deg": tilt,
            "azimuth_deg": float(np.rad2deg(theta)), "n_walls": int(n_walls)}


def _nearest_axis(n: np.ndarray, frame: dict) -> Tuple[str, np.ndarray, float]:
    """(axis name, signed axis vector, angle in degrees) of the frame axis nearest to ``n``."""
    best = None
    for name in ("up", "X", "Z"):
        a = frame[name]
        d = float(n @ a)
        ang = float(np.rad2deg(np.arccos(np.clip(abs(d), 0.0, 1.0))))
        if best is None or ang < best[2]:
            best = (name, a * (1.0 if d >= 0 else -1.0), ang)
    return best


def _basis_for(n: np.ndarray, frame: dict) -> Tuple[np.ndarray, np.ndarray]:
    """In-plane (u, v) from the frame: horizontal surfaces u = X, v = n × u (Z on a floor);
    everything else v = up projected (height), u = v × n (horizontal) — the walls' rectangles
    and the floor's outline come out axis-aligned in UV."""
    up = frame["up"]
    if abs(float(n @ up)) > 0.5:
        u = _unit(frame["X"] - (frame["X"] @ n) * n)
        v = _unit(np.cross(n, u))
        return u, v
    v = _unit(up - (up @ n) * n)
    u = _unit(np.cross(v, n))
    return u, v


def regularize_manhattan(models: Sequence[PlaneModel], weights: Sequence[float],
                         points: Sequence[np.ndarray], up_hint: np.ndarray,
                         snap_deg: float, coplanar_tol_mm: float,
                         ids: Optional[Sequence[int]] = None
                         ) -> Tuple[List[PlaneModel], dict]:
    """Snap the marked planes to the scene frame. Returns NEW models (originals untouched) and
    the report: the frame, and per plane what moved, by how much, or why it did not."""
    ids = list(ids) if ids is not None else list(range(len(models)))
    frame = scene_frame([m.normal for m in models], weights, up_hint, snap_deg)
    out: List[PlaneModel] = []
    objs: List[dict] = []
    axis_of: List[Optional[str]] = []
    for m, pts, iid in zip(models, points, ids):
        name, axis, ang = _nearest_axis(m.normal, frame)
        pts = np.asarray(pts, np.float64)
        if ang <= float(snap_deg):
            n_new = axis
            d_new = float(-np.median(pts @ n_new)) if len(pts) else float(-n_new @ m.origin)
            ctr = pts.mean(0) if len(pts) else m.origin
            origin = ctr - (ctr @ n_new + d_new) * n_new
            u, v = _basis_for(n_new, frame)
            out.append(PlaneModel(normal=n_new, d=d_new, origin=origin, u=u, v=v, rms=m.rms,
                                  inlier_frac=m.inlier_frac, n_points=m.n_points))
            shift_mm = 1000.0 * abs(float(origin @ m.normal + m.d))
            objs.append({"instance_id": int(iid), "snapped": True, "axis": name,
                         "moved_deg": round(ang, 4), "offset_shift_mm": round(shift_mm, 2)})
            axis_of.append(name)
        else:
            out.append(PlaneModel(normal=m.normal.copy(), d=float(m.d), origin=m.origin.copy(),
                                  u=m.u.copy(), v=m.v.copy(), rms=m.rms,
                                  inlier_frac=m.inlier_frac, n_points=m.n_points))
            objs.append({"instance_id": int(iid), "snapped": False, "axis": name,
                         "off_manhattan_deg": round(ang, 4),
                         "reason": f"{ang:.1f}° from the nearest frame axis, beyond "
                                   f"manhattan_snap_deg {float(snap_deg):g} — kept as measured"})
            axis_of.append(None)
    # coplanar marked planes on one axis → one plane
    w = np.asarray(weights, np.float64)
    ctol = float(coplanar_tol_mm) / 1000.0
    groups: List[List[int]] = []
    used: set = set()
    for i in range(len(out)):
        if i in used or axis_of[i] is None:
            continue
        grp = [i]
        for j in range(i + 1, len(out)):
            if j in used or axis_of[j] != axis_of[i]:
                continue
            si = out[i].d * (1.0 if out[i].normal @ frame[axis_of[i]] > 0 else -1.0)
            sj = out[j].d * (1.0 if out[j].normal @ frame[axis_of[j]] > 0 else -1.0)
            if abs(si - sj) <= ctol:
                grp.append(j)
        if len(grp) > 1:
            dm = float(np.average(
                [out[g].d * (1.0 if out[g].normal @ frame[axis_of[g]] > 0 else -1.0) for g in grp],
                weights=[w[g] for g in grp]))
            for g in grp:
                sign = 1.0 if out[g].normal @ frame[axis_of[g]] > 0 else -1.0
                out[g].d = float(sign * dm)
                out[g].origin = out[g].origin - (out[g].origin @ out[g].normal + out[g].d) * out[g].normal
                objs[g]["coplanar_group"] = [int(ids[k]) for k in grp]
            used.update(grp)
            groups.append([int(ids[k]) for k in grp])
    report = {
        "frame": {"up": frame["up"].tolist(), "X": frame["X"].tolist(), "Z": frame["Z"].tolist(),
                  "up_from_instance": (int(ids[frame["up_from"]]) if frame["up_from"] is not None else None),
                  "up_tilt_deg": round(float(frame["up_tilt_deg"]), 4),
                  "azimuth_deg": round(float(frame["azimuth_deg"]), 4),
                  "n_walls": int(frame["n_walls"])},
        "snap_deg": float(snap_deg), "coplanar_tol_mm": float(coplanar_tol_mm),
        "objects": objs, "coplanar_groups": groups,
        "provenance": "human_validated mark (segmentation.manhattan); the snap is a modelling "
                      "decision — the measured deviation of every surface is recorded here and "
                      "in its residual report, the cloud and the poses are untouched",
    }
    n_snap = sum(1 for o in objs if o["snapped"])
    logger.info("manhattan: frame up tilt %.2f°, azimuth %.2f° from %d wall(s); %d/%d plane(s) "
                "snapped, %d coplanar group(s)", frame["up_tilt_deg"], frame["azimuth_deg"],
                frame["n_walls"], n_snap, len(objs), len(groups))
    return out, report
