# STAC-Builder — Phase 5: deterministic spatial tools over the R.8 store.
#
# THE MEASURING LAYER. The VLM proposes/orchestrates; these pure functions do
# ALL the measuring, over the canonical instance store (points, OBBs, class,
# vote/onion metrics). Every tool: typed, SI units, a confidence estimate, and
# returns {"insufficient_data": true} rather than extrapolating when it lacks
# support. Each has a synthetic-geometry unit test.
#
# FRAME: every tool operates and answers in the DISPLAY frame — the exact frame
# the viewer renders (floor-aligned via output/floor_transform.npz, metric
# scale). Store geometry is kept raw and converted ON READ here, so gizmo /
# level-floor edits are picked up automatically and chat numbers, overlays and
# the rendered cloud always agree (this fixed the misplaced-measurement bug).
#
# PROVENANCE:
#   Ported/adapted from R3D vendor/r3d/r3d/pipeline/eval/tool_use.py:
#     list_objects, get_position, get_object_size, get_object_volume,
#     get_distance (OBB closest-point math _closest_point_on_obb/_invert_rigid),
#     get_my_position, get_distance_from_me (ego tools — "me" is the LAST
#     camera pose of the scan, persisted by the Phase R builder).
#   Ours (construction-supervision tools): get_clearance, get_plumb, get_level,
#     get_span, count_objects, measure_between, fits_through,
#     get_height_profile, get_flatness_report (surface_fitting bridge),
#     get_alignment_health, get_onion_report, get_instance_history,
#     get_findings, define/evaluate/objects_in/fits_in_volume.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phase_r.geometry import invert_rigid, obb_local_coords
from phase_r.instance_store import InstanceStore


def load_display_transform(store_path: str | Path) -> np.ndarray | None:
    """4x4 similarity mapping RAW store coordinates -> the viewer's DISPLAY
    frame, from output/floor_transform.npz (s, R, t) next to the store.
    None when the scene has no floor alignment (identity)."""
    p = Path(store_path).parent / "floor_transform.npz"
    if not p.exists():
        return None
    try:
        d = np.load(p)
        s, R, t = float(d["s"]), np.asarray(d["R"], float), np.asarray(d["t"], float)
        M = np.eye(4)
        M[:3, :3] = s * R
        M[:3, 3] = t
        return M
    except Exception:
        return None


def _insufficient(reason: str) -> dict[str, Any]:
    return {"insufficient_data": True, "reason": reason}


def _closest_point_on_obb(point: np.ndarray, T: np.ndarray, aabb: np.ndarray) -> np.ndarray:
    """World point -> nearest point on the OBB surface/volume (R3D-ported)."""
    Tinv = invert_rigid(T)
    local = (Tinv @ np.append(point, 1.0))[:3]
    lo = np.array([aabb[0], aabb[2], aabb[4]])
    hi = np.array([aabb[1], aabb[3], aabb[5]])
    clipped = np.clip(local, lo, hi)
    return (T @ np.append(clipped, 1.0))[:3]


class SpatialTools:
    """Bound to one scene's InstanceStore. Methods are the tool implementations;
    `openai_schemas()` returns the tool-calling specs for the orchestrator.
    All geometry is served in the DISPLAY frame (see module docstring)."""

    def __init__(self, store: InstanceStore, display_transform: np.ndarray | None = None):
        self.store = store
        if display_transform is None:
            # Stores built by segmentation/pipeline.py are ALREADY in the
            # display frame (meta built_from=*_display_frame) — applying the
            # floor transform again DOUBLE-transformed every point/OBB the
            # moment the npz stopped being identity (2026-08-29: door OBB drawn
            # tilted and low, volumes placed off-floor). Only legacy RAW-frame
            # stores (no such meta) still need the npz mapping.
            built_from = None
            try:
                built_from = store.get_meta("built_from")
            except Exception:  # noqa: BLE001
                pass
            if not (built_from and "display_frame" in str(built_from)):
                display_transform = load_display_transform(store.path)
        self._M = np.asarray(display_transform, float) if display_transform is not None else None
        # similarity scale of the display transform (uniform by construction)
        self._s = (float(np.cbrt(abs(np.linalg.det(self._M[:3, :3]))))
                   if self._M is not None else 1.0)
        self._pts_cache: dict[int, np.ndarray | None] = {}

    # ── frame helpers ────────────────────────────────────────────────
    def _to_display(self, pts: np.ndarray) -> np.ndarray:
        if self._M is None or pts is None or not len(pts):
            return pts
        ph = np.concatenate([np.atleast_2d(pts), np.ones((len(np.atleast_2d(pts)), 1))], 1)
        return (self._M @ ph.T).T[:, :3]

    def _obb(self, iid: int, window_id: str = "global"):
        """(T, aabb, pos) in the DISPLAY frame (rigid T, extents ×s)."""
        obb = self.store.get_obb(iid, window_id)
        if obb is None or self._M is None:
            return obb
        T, aabb, _pos = obb
        R_m = self._M[:3, :3] / self._s
        T2 = np.asarray(T, float).copy()
        T2[:3, :3] = R_m @ T2[:3, :3]
        T2[:3, 3] = self._to_display(T2[:3, 3][None])[0]
        return T2, np.asarray(aabb, float) * self._s, T2[:3, 3]

    # ── helpers ─────────────────────────────────────────────────────
    def _instance(self, iid: int) -> dict | None:
        for i in self.store.list_instances():
            if i["instance_id"] == iid:
                return i
        return None

    def _points(self, iid: int) -> np.ndarray | None:
        if iid not in self._pts_cache:
            pts = self.store.get_points(iid)
            self._pts_cache[iid] = self._to_display(pts) if pts is not None else None
        return self._pts_cache[iid]

    def _density_confidence(self, iid: int) -> float:
        pts = self._points(iid)
        if pts is None or len(pts) < 30:
            return 0.2
        return float(min(1.0, 0.4 + len(pts) / 5000.0))

    def _label_of(self, iid: int) -> str:
        inst = self._instance(iid)
        return inst["label"] if inst else str(iid)

    # ── discovery (R3D) ─────────────────────────────────────────────
    def list_objects(self, label: str | None = None) -> dict:
        """List tracked objects (id + class + Phase 2 enrichment), optionally
        filtered by class/label substring."""
        insts = self.store.list_instances()
        if label:
            insts = [i for i in insts if label.lower() in i["label"].lower()]
        out = []
        for i in insts:
            entry = {"id": i["instance_id"], "label": i["label"], "status": i["status"]}
            c = self.store.get_classification(i["instance_id"])
            if c:
                entry["class"] = c["class_final"]
                if c.get("material"):
                    entry["material"] = c["material"]
                if c.get("state"):
                    entry["state"] = c["state"]
            out.append(entry)
        return {"objects": out}

    def count_objects(self, label: str | None = None) -> dict:
        """Count objects, optionally of a given class."""
        return {"count": len(self.list_objects(label)["objects"]), "unit": "count"}

    # ── position / size / volume (R3D) ──────────────────────────────
    def get_position(self, id: int) -> dict:
        """World position (OBB centre) of an object, metres (x,y,z)."""
        obb = self._obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, _a, pos = obb
        return {"position_m": [round(float(v), 4) for v in pos], "unit": "m",
                "confidence": self._density_confidence(id), "source": "tool_measured"}

    def get_object_size(self, id: int) -> dict:
        """Object bounding-box size width/height/depth, metres (OBB extents)."""
        obb = self._obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, aabb, _p = obb
        return {"width_m": round(float(aabb[1] - aabb[0]), 4),
                "height_m": round(float(aabb[3] - aabb[2]), 4),
                "depth_m": round(float(aabb[5] - aabb[4]), 4), "unit": "m",
                "confidence": self._density_confidence(id), "source": "tool_measured"}

    def get_object_volume(self, id: int) -> dict:
        """Bounding-box volume of an object, cubic metres."""
        obb = self._obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, aabb, _p = obb
        v = float((aabb[1] - aabb[0]) * (aabb[3] - aabb[2]) * (aabb[5] - aabb[4]))
        return {"bbox_volume_m3": round(v, 5), "unit": "m^3",
                "confidence": self._density_confidence(id), "source": "tool_measured"}

    # ── ego tools (R3D get_my_position / get_distance_from_me) ──────
    def _camera_positions(self) -> dict[int, np.ndarray] | None:
        raw = self.store.get_meta("camera_positions")
        if not raw:
            return None
        try:
            d = json.loads(raw)
            return {int(k): self._to_display(np.asarray(v, float)[None])[0]
                    for k, v in d.items()}
        except Exception:
            return None

    def get_my_position(self) -> dict:
        """Position of the camera at the END of the scan ("me"), metres.
        Ported from R3D; ego = last recorded camera pose."""
        cams = self._camera_positions()
        if not cams:
            return _insufficient("no camera trajectory recorded in the store")
        last = cams[max(cams)]
        return {"position_m": [round(float(v), 4) for v in last], "unit": "m",
                "frame": max(cams), "confidence": 0.9, "source": "tool_measured",
                "note": "'me' = camera position at the last frame of the scan"}

    def get_distance_from_me(self, id: int) -> dict:
        """Distance from the scan-end camera position to an object's surface,
        metres (R3D-ported ego tool)."""
        cams = self._camera_positions()
        if not cams:
            return _insufficient("no camera trajectory recorded in the store")
        obb = self._obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        me = cams[max(cams)]
        T, aabb, _p = obb
        d = float(np.linalg.norm(_closest_point_on_obb(me, T, aabb) - me))
        return {"distance_m": round(d, 4), "unit": "m",
                "confidence": self._density_confidence(id), "source": "tool_measured",
                "note": "'me' = camera position at the last frame of the scan"}

    # ── distances (R3D get_distance + our clearance) ────────────────
    def get_distance(self, id1: int, id2: int) -> dict:
        """Approx. surface distance between two objects (OBB-to-OBB), metres."""
        o1, o2 = self._obb(id1), self._obb(id2)
        if o1 is None or o2 is None:
            return _insufficient("missing OBB for one of the objects")
        T1, a1, p1 = o1
        T2, a2, p2 = o2
        d = min(np.linalg.norm(_closest_point_on_obb(p1, T2, a2) - p1),
                np.linalg.norm(_closest_point_on_obb(p2, T1, a1) - p2))
        return {"distance_m": round(float(d), 4), "unit": "m",
                "confidence": min(self._density_confidence(id1), self._density_confidence(id2)),
                "source": "tool_measured"}

    def get_clearance(self, id1: int, id2: int) -> dict:
        """Minimum real clearance between two objects using their point clouds,
        metres, with the closest point pair (construction supervision)."""
        p1, p2 = self._points(id1), self._points(id2)
        if p1 is None or p2 is None or len(p1) < 30 or len(p2) < 30:
            return _insufficient("insufficient points for a reliable clearance")
        from scipy.spatial import cKDTree
        tree = cKDTree(p2)
        dists, idx = tree.query(p1, k=1)
        j = int(np.argmin(dists))
        return {"clearance_m": round(float(dists[j]), 4), "unit": "m",
                "point_a_m": [round(float(v), 4) for v in p1[j]],
                "point_b_m": [round(float(v), 4) for v in p2[idx[j]]],
                "confidence": min(self._density_confidence(id1), self._density_confidence(id2)),
                "source": "tool_measured"}

    def _feat_point(self, iid: int, feature: str) -> np.ndarray | None:
        """Representative point of a NAMED PART of an object, from its REAL
        point cloud (2026-08-29: OBB face centres were wrong for curved /
        composite shapes — a curved ceiling has a lower AND an upper part, a
        ladder a base and a top; bands of the actual points capture them)."""
        pts = self._points(iid)
        if pts is not None and len(pts) >= 30:
            y = pts[:, 1]
            y_min, y_max = float(y.min()), float(y.max())
            band = max(0.05, 0.05 * (y_max - y_min))
            if feature in ("top", "upper"):
                return pts[y >= y_max - band].mean(0)
            if feature in ("bottom", "base", "lower"):
                return pts[y <= y_min + band].mean(0)
            if feature == "highest":
                return pts[int(np.argmax(y))]
            if feature == "lowest":
                return pts[int(np.argmin(y))]
            if feature == "centroid":
                return pts.mean(0)
        # OBB fallback (few points): centre / vertical face centres
        obb = self._obb(iid)
        if obb is None:
            return None
        T, aabb, pos = obb
        if feature == "centroid":
            return pos
        off = aabb[3] if feature in ("top", "upper", "highest") else aabb[2]
        return pos + np.asarray(T)[:3, 1] * off

    _FEATURES = ("centroid", "top", "bottom", "base", "upper", "lower",
                 "highest", "lowest", "closest")

    def measure_between(self, id1: int, id2: int,
                        feature1: str = "centroid", feature2: str = "centroid",
                        axis: str = "straight") -> dict:
        """Distance between NAMED PARTS of two objects, metres, with the
        endpoint pair. Parts come from the REAL point cloud: centroid | top /
        upper | bottom / base / lower (surface bands) | highest / lowest
        (extreme points) | closest (nearest surface pair). axis="vertical"
        returns the height difference |Δy| (e.g. free height floor → ceiling
        UPPER part), axis="straight" the 3-D distance."""
        if feature1 not in self._FEATURES or feature2 not in self._FEATURES:
            return _insufficient(f"unknown feature; pick from {sorted(set(self._FEATURES))}")
        if feature1 == "closest" or feature2 == "closest":
            c = self.get_clearance(id1, id2)
            if c.get("insufficient_data"):
                return c
            a = np.asarray(c["point_a_m"], float)
            b = np.asarray(c["point_b_m"], float)
            d = abs(float(a[1] - b[1])) if axis == "vertical" else float(c["clearance_m"])
            return {"distance_m": round(d, 4), "unit": "m", "axis": axis,
                    "point_a_m": c["point_a_m"], "point_b_m": c["point_b_m"],
                    "feature1": "closest", "feature2": "closest",
                    "confidence": c["confidence"], "source": "tool_measured"}

        a, b = self._feat_point(id1, feature1), self._feat_point(id2, feature2)
        if a is None or b is None:
            return _insufficient("missing points/OBB for one of the objects")
        d = abs(float(a[1] - b[1])) if axis == "vertical" \
            else float(np.linalg.norm(a - b))
        return {"distance_m": round(d, 4), "unit": "m", "axis": axis,
                "point_a_m": [round(float(v), 4) for v in a],
                "point_b_m": [round(float(v), 4) for v in b],
                "feature1": feature1, "feature2": feature2,
                "confidence": min(self._density_confidence(id1), self._density_confidence(id2)),
                "source": "tool_measured"}

    def get_extent(self, id: int) -> dict:
        """True point-cloud extent of an object: axis-aligned min/max (m),
        height, and the centres of its LOWEST and HIGHEST surface bands — a
        curved ceiling has both a lower and an upper part; this says exactly
        where each one is."""
        pts = self._points(id)
        if pts is None or len(pts) < 30:
            return _insufficient("insufficient points for an extent")
        mn, mx = pts.min(0), pts.max(0)
        y = pts[:, 1]
        band = max(0.05, 0.05 * float(mx[1] - mn[1]))
        low_c = pts[y <= mn[1] + band].mean(0)
        high_c = pts[y >= mx[1] - band].mean(0)
        return {"min_m": [round(float(v), 4) for v in mn],
                "max_m": [round(float(v), 4) for v in mx],
                "size_m": [round(float(v), 4) for v in (mx - mn)],
                "height_m": round(float(mx[1] - mn[1]), 4),
                "lowest_band_center_m": [round(float(v), 4) for v in low_c],
                "highest_band_center_m": [round(float(v), 4) for v in high_c],
                "unit": "m", "confidence": self._density_confidence(id),
                "source": "tool_measured"}

    def fits_through(self, item_size: list, opening_id: int | None = None,
                     id1: int | None = None, id2: int | None = None,
                     margin_m: float = 0.05) -> dict:
        """Can an item of size [w,h,d] metres pass THROUGH an opening (door /
        window / hatch: its OBB face excluding the thickness axis) OR through
        the gap between two objects? Deterministic, with a safety margin."""
        if item_size is None or len(item_size) != 3:
            return _insufficient("item_size must be [w,h,d] in metres")
        item = np.sort(np.abs(np.asarray(item_size, float)))  # ascending
        if opening_id is not None:
            obb = self._obb(opening_id)
            if obb is None:
                return _insufficient(f"no OBB for opening {opening_id}")
            _T, aabb, _p = obb
            ext = np.array([aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]])
            passage = np.sort(ext)[1:]  # drop the thickness (smallest) axis
            need = item[:2]             # item's two smallest dims must fit
            ok = bool(np.all(need + margin_m <= passage))
            return {"fits": ok, "passage_m": [round(float(v), 4) for v in passage],
                    "item_cross_section_m": [round(float(v), 4) for v in need],
                    "margin_m": margin_m, "mode": "opening", "unit": "m",
                    "confidence": self._density_confidence(opening_id),
                    "source": "tool_measured"}
        if id1 is not None and id2 is not None:
            c = self.get_clearance(id1, id2)
            if c.get("insufficient_data"):
                return c
            gap = c["clearance_m"]
            ok = bool(item[0] + margin_m <= gap)  # smallest item dim through the gap
            return {"fits": ok, "gap_m": gap,
                    "item_min_dim_m": round(float(item[0]), 4),
                    "margin_m": margin_m, "mode": "gap", "unit": "m",
                    "gap_point_a_m": c["point_a_m"], "gap_point_b_m": c["point_b_m"],
                    "confidence": c["confidence"], "source": "tool_measured",
                    "note": "gap mode checks the closest pinch point only; "
                            "use fits_in_volume for full-path planning"}
        return _insufficient("pass opening_id OR id1+id2")

    # ── plumb / level (ours, vs gravity) ────────────────────────────
    def get_plumb(self, id: int) -> dict:
        """Out-of-plumb of a vertical element: tilt of its principal vertical
        axis from true vertical, degrees + mm-per-metre. Vertical = +Y of the
        DISPLAY frame (floor-aligned)."""
        pts = self._points(id)
        if pts is None or len(pts) < 50:
            return _insufficient("insufficient points for plumb")
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        axis = vt[0]
        if axis[1] < 0:
            axis = -axis
        vertical = np.array([0.0, 1.0, 0.0])
        cos = abs(float(np.dot(axis, vertical)))
        tilt_deg = float(np.degrees(np.arccos(min(1.0, cos))))
        return {"plumb_deviation_deg": round(tilt_deg, 3),
                "deviation_mm_per_m": round(float(np.tan(np.radians(tilt_deg)) * 1000), 2),
                "unit": "deg", "confidence": self._density_confidence(id),
                "source": "tool_measured"}

    def get_level(self, id: int) -> dict:
        """Out-of-level of a horizontal element: tilt of its plane from
        horizontal, degrees + mm-per-metre (DISPLAY frame, floor-aligned)."""
        pts = self._points(id)
        if pts is None or len(pts) < 50:
            return _insufficient("insufficient points for level")
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        normal = vt[2]  # smallest-variance direction = plane normal
        if normal[1] < 0:
            normal = -normal
        vertical = np.array([0.0, 1.0, 0.0])
        cos = abs(float(np.dot(normal, vertical)))
        tilt_deg = float(np.degrees(np.arccos(min(1.0, cos))))
        return {"level_deviation_deg": round(tilt_deg, 3),
                "deviation_mm_per_m": round(float(np.tan(np.radians(tilt_deg)) * 1000), 2),
                "unit": "deg", "confidence": self._density_confidence(id),
                "source": "tool_measured"}

    def get_span(self, id: int) -> dict:
        """Largest horizontal extent (span) of an object, metres."""
        obb = self._obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, aabb, _p = obb
        return {"span_m": round(float(max(aabb[1] - aabb[0], aabb[5] - aabb[4])), 4),
                "unit": "m", "confidence": self._density_confidence(id),
                "source": "tool_measured"}

    def get_height_profile(self, id: int, n_stations: int = 10,
                           floor_id: int | None = None) -> dict:
        """Height profile (gálibos) along an object's longest horizontal axis:
        per station, the object's min/max height — and, when floor_id is given,
        the free clearance between the floor and the object's underside."""
        pts = self._points(id)
        obb = self._obb(id)
        if pts is None or len(pts) < 50 or obb is None:
            return _insufficient("insufficient points for a height profile")
        T, aabb, _p = obb
        ext_x, ext_z = aabb[1] - aabb[0], aabb[5] - aabb[4]
        axis_i = 0 if ext_x >= ext_z else 2
        local = obb_local_coords(pts, T)
        s_lo, s_hi = (aabb[0], aabb[1]) if axis_i == 0 else (aabb[4], aabb[5])
        span = max(s_hi - s_lo, 1e-9)
        station_of = np.clip(((local[:, axis_i] - s_lo) / span * n_stations).astype(int),
                             0, n_stations - 1)
        floor_pts = self._points(floor_id) if floor_id is not None else None
        stations = []
        for k in range(n_stations):
            sel = station_of == k
            if int(sel.sum()) < 10:
                stations.append({"station": k, "insufficient_data": True})
                continue
            ys = pts[sel][:, 1]
            entry = {"station": k,
                     "position_along_m": round(float(s_lo + (k + 0.5) / n_stations * span), 3),
                     "y_min_m": round(float(ys.min()), 4),
                     "y_max_m": round(float(ys.max()), 4)}
            if floor_pts is not None and len(floor_pts) >= 30:
                # clearance = object's underside minus the floor level below it
                xz = pts[sel][:, [0, 2]].mean(0)
                near = floor_pts[np.linalg.norm(floor_pts[:, [0, 2]] - xz, axis=1) < max(span / n_stations, 0.5)]
                if len(near) >= 10:
                    entry["clearance_m"] = round(float(ys.min() - float(np.median(near[:, 1]))), 4)
            stations.append(entry)
        valid = [st for st in stations if "y_min_m" in st]
        if not valid:
            return _insufficient("no station had enough points")
        out = {"stations": stations, "n_stations": n_stations, "unit": "m",
               "confidence": self._density_confidence(id), "source": "tool_measured"}
        clearances = [st["clearance_m"] for st in valid if "clearance_m" in st]
        if clearances:
            out["min_clearance_m"] = round(min(clearances), 4)
        return out

    def get_flatness_report(self, id: int, tolerance_mm: float = 5.0) -> dict:
        """Flatness report for a surface: BRIDGE to surface_fitting artifacts
        (output/surface_fit/<label>_<id>/residuals.json) when they exist;
        deterministic plane-fit residuals over the instance's points otherwise.
        All residuals in millimetres."""
        # 1) surface_fitting artifact (authoritative when present)
        base = Path(self.store.path).parent / "surface_fit"
        if base.is_dir():
            for d in sorted(base.glob(f"*_{id}")):
                rj = d / "residuals.json"
                if rj.exists():
                    try:
                        rep = json.loads(rj.read_text())
                        st = rep.get("stats", rep)
                        return {"source": "tool_measured", "backend": "surface_fitting",
                                "rms_mm": st.get("rms_mm"), "p95_mm": st.get("p95_mm"),
                                "max_mm": st.get("max_mm"),
                                "flatness_worst_mm": st.get("flatness_worst_mm"),
                                "flatness_span_m": st.get("flatness_span_m"),
                                "flatness_pass": st.get("flatness_pass"),
                                "n_findings": len(rep.get("findings", [])),
                                "artifact": str(rj), "unit": "mm",
                                "confidence": 0.9}
                    except Exception:
                        pass
        # 2) deterministic fallback: plane fit + residual stats on store points
        pts = self._points(id)
        if pts is None or len(pts) < 100:
            return _insufficient("no surface_fit artifact and too few points "
                                 "for a plane-fit flatness estimate")
        from phase_r.depth_regularization import fit_plane
        n, d0 = fit_plane(pts)
        res_mm = np.abs(pts @ n + d0) * 1000.0
        return {"source": "tool_measured", "backend": "plane_fit_fallback",
                "rms_mm": round(float(np.sqrt((res_mm ** 2).mean())), 2),
                "p95_mm": round(float(np.percentile(res_mm, 95)), 2),
                "max_mm": round(float(res_mm.max()), 2),
                "flatness_pass": bool(np.percentile(res_mm, 95) <= tolerance_mm),
                "tolerance_mm": tolerance_mm, "unit": "mm",
                "confidence": self._density_confidence(id),
                "note": "plane-fit fallback — run surface_fitting for the "
                        "measurement-grade report"}

    # ── Phase R alignment health (ours) ─────────────────────────────
    def get_alignment_health(self, id: int) -> dict:
        """Alignment health of an object: vote entropy + onion metric (Phase R)."""
        m = self.store.get_metrics(id)
        if not m.get("vote") and not m.get("onion"):
            return _insufficient(f"no alignment metrics for object {id}")
        return {"vote": m.get("vote"), "onion": m.get("onion"),
                "confidence": self._density_confidence(id), "source": "tool_measured"}

    def get_onion_report(self, id: int, seam: str | None = None) -> dict:
        """Onion (doubled-surface) report for an object: mode separation in
        metres and whether it is bimodal (a registration-error signal). Pass a
        seam ("wa|wb") for the per-seam metric; includes the per-instance
        heatmap when available."""
        rows = self.store.list_onion_seams(id)
        if not rows:
            return _insufficient(f"no onion metric for object {id}")
        target = seam or "global"
        row = next((r for r in rows if r["seam"] == target), None)
        if row is None:
            return _insufficient(f"no onion metric for object {id} at seam '{target}'; "
                                 f"available: {[r['seam'] for r in rows]}")
        state = "doubled_surface" if row["bimodal"] else "clean"
        out = {"separation_m": round(float(row["separation_m"]), 4),
               "bimodal": bool(row["bimodal"]), "state": state, "seam": row["seam"],
               "seams_available": [r["seam"] for r in rows], "unit": "m",
               "confidence": self._density_confidence(id), "source": "tool_measured"}
        if row.get("heatmap"):
            try:
                out["heatmap"] = json.loads(row["heatmap"])
            except Exception:
                pass
        return out

    def get_instance_history(self, id: int) -> dict:
        """Windows an object was seen in and the residuals applied (Phase R)."""
        windows = self.store.list_obb_windows(id)
        inst = self._instance(id)
        if inst is None:
            return _insufficient(f"unknown object {id}")
        return {"windows": windows, "n_views": inst.get("n_views"),
                "first_frame": inst.get("first_frame"), "last_frame": inst.get("last_frame"),
                "confidence": 0.9, "source": "tool_measured"}

    def get_findings(self, id: int | None = None, region: str | None = None) -> dict:
        """Visual findings (cracks/moisture/…) for an object/region (Phase 3).
        Returns insufficient_data when no findings have been recorded."""
        finds = self.store.list_findings(instance_id=id)
        if not finds:
            scope = f"object {id}" if id is not None else "the scene"
            return _insufficient(f"no findings recorded for {scope}")
        out = [{"finding_id": f["finding_id"], "type": f["type"], "severity": f["severity"],
                "description": f["description"], "confidence": round(f["confidence"], 2),
                "instance_id": f["instance_id"],
                "point3d": (self._to_display(np.asarray(f["point3d"], float)[None])[0]
                            .round(4).tolist() if f["point3d"] else None),
                "status": f["status"], "correlated_residual": f["correlated_residual"],
                "provenance": ("human_validated" if f["status"] == "human_validated"
                               else "vlm_proposed")}
               for f in finds]
        n_validated = sum(1 for f in out if f["provenance"] == "human_validated")
        return {"findings": out, "count": len(out),
                # envelope provenance reflects the strongest state present —
                # per-item provenance is authoritative
                "source": ("human_validated" if n_validated == len(out)
                           else "mixed" if n_validated else "vlm_proposed"),
                "n_validated": n_validated,
                "note": "unvalidated findings are proposals pending human validation"}

    # ── user-defined evaluation volumes (space evaluation) ──────────
    def _all_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Pool every instance's display-frame points once, with a parallel
        array of the owning instance id (occupancy / space queries)."""
        if getattr(self, "_pooled", None) is None:
            clouds, owners = [], []
            for i in self.store.list_instances():
                pts = self._points(i["instance_id"])
                if pts is not None and len(pts):
                    clouds.append(pts)
                    owners.append(np.full(len(pts), i["instance_id"], np.int64))
            self._pooled = (np.vstack(clouds), np.concatenate(owners)) if clouds \
                else (np.empty((0, 3)), np.empty((0,), np.int64))
        return self._pooled

    def _resolve_volume(self, center, size, yaw_deg, volume_id):
        """Return (center, size, yaw_deg) from explicit args or a saved volume.
        User volumes are stored in the DISPLAY frame (what the user picked)."""
        if volume_id is not None:
            v = self.store.get_user_volume(int(volume_id))
            if v is None:
                return None
            return np.asarray(v["center"], float), np.asarray(v["size"], float), float(v["yaw_deg"])
        if center is None or size is None:
            return None
        return np.asarray(center, float), np.asarray(size, float), float(yaw_deg or 0.0)

    @staticmethod
    def _volume_frame(center: np.ndarray, size: np.ndarray, yaw_deg: float):
        half = np.abs(size) / 2.0
        yaw = np.radians(yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # about world up (+Y)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = center
        return T, half

    def _inside_mask(self, pts: np.ndarray, T: np.ndarray, half: np.ndarray) -> np.ndarray:
        if not len(pts):
            return np.zeros(0, bool)
        local = obb_local_coords(pts, T)
        return np.all(np.abs(local) <= half[None, :] + 1e-9, axis=1)

    def _floor_instance(self) -> int | None:
        for i in self.store.list_instances():
            if "floor" in str(i.get("label", "")).lower():
                return int(i["instance_id"])
        return None

    def define_volume(self, name: str, size: list | None = None,
                      center: list | None = None, yaw_deg: float = 0.0,
                      volume_m3: float | None = None,
                      anchor_id: int | None = None,
                      on_floor: bool = True) -> dict:
        """Persist a user-defined evaluation box (DISPLAY frame). PLACEMENT
        RULES (user 2026-08-29 — volumes were landing anywhere):
        - size = [w,h,d] m, or volume_m3 → a cube of that volume.
        - x/z default to the anchor object's centroid (anchor_id; DEFAULT
          anchor = the floor → 'centre of the floor'). An explicit center
          overrides.
        - on_floor=true (DEFAULT): the box RESTS ON the floor — base at the
          floor surface, centre y = floor_top + h/2. Only on_floor=false uses
          an explicit center y."""
        if size is None and volume_m3:
            e = float(volume_m3) ** (1.0 / 3.0)
            size = [e, e, e]
        if size is None or len(size) != 3:
            return _insufficient("pass size [w,h,d] in metres, or volume_m3")
        size = [abs(float(s)) for s in size]
        # x/z: explicit center wins; else anchor centroid (default: the floor)
        anchor = anchor_id if anchor_id is not None else self._floor_instance()
        ax, az = 0.0, 0.0
        if anchor is not None:
            cpt = self._feat_point(int(anchor), "centroid")
            if cpt is not None:
                ax, az = float(cpt[0]), float(cpt[2])
        cx = float(center[0]) if center and len(center) >= 1 and center[0] is not None else ax
        cz = float(center[2]) if center and len(center) == 3 and center[2] is not None else az
        # y: resting on the floor unless explicitly overridden
        if on_floor or not (center and len(center) == 3 and center[1] is not None):
            floor_top = 0.0   # display frame: leveled floor sits at y=0
            fid = self._floor_instance()
            if fid is not None:
                tpt = self._feat_point(fid, "top")
                if tpt is not None:
                    floor_top = float(tpt[1])
            cy = floor_top + size[1] / 2.0
        else:
            cy = float(center[1])
        vid = self.store.add_user_volume(name or "volume", [cx, cy, cz],
                                         size, yaw_deg or 0.0)
        self.store.set_meta("volumes_frame", "display")
        return {"volume_id": vid, "name": name, "center": [round(cx, 4), round(cy, 4), round(cz, 4)],
                "size": [round(s, 4) for s in size], "yaw_deg": float(yaw_deg or 0.0),
                "on_floor": bool(on_floor), "anchor_id": anchor,
                "source": "user_defined"}

    def list_volumes(self) -> dict:
        """The user-defined evaluation volumes saved in this scene."""
        return {"volumes": self.store.list_user_volumes(), "source": "tool_measured"}

    def measure_volume(self, volume_id: int, id: int | None = None,
                       axis: str = "straight") -> dict:
        """Measure FROM a user-defined volume: clearance from the box surface
        to an object's real points (0 with points_inside>0 = they intersect),
        plus the centre-to-centroid distance. axis='vertical' → |Δy| of the
        nearest pair."""
        r = self._resolve_volume(None, None, 0.0, volume_id)
        if r is None:
            return _insufficient("unknown volume_id")
        c, s, yaw = r
        T, half = self._volume_frame(np.asarray(c, float), np.asarray(s, float), float(yaw))
        if id is None:
            return _insufficient("pass the object id to measure against")
        pts = self._points(int(id))
        if pts is None or len(pts) < 30:
            return _insufficient("insufficient points on the object")
        local = obb_local_coords(pts, T)
        outside = np.maximum(np.abs(local) - half[None, :], 0.0)
        d = np.linalg.norm(outside, axis=1)
        j = int(np.argmin(d))
        n_inside = int((d <= 1e-9).sum())
        # nearest object point and its counterpart on the box surface
        p_obj = pts[j]
        clipped = np.clip(local[j], -half, half)
        p_box = (np.asarray(T)[:3, :3] @ clipped) + np.asarray(T)[:3, 3]
        dist = abs(float(p_obj[1] - p_box[1])) if axis == "vertical" else float(d[j])
        return {"clearance_m": round(dist, 4), "unit": "m", "axis": axis,
                "intersects": bool(n_inside > 0),
                "points_inside": n_inside,
                "point_a_m": [round(float(v), 4) for v in p_box],
                "point_b_m": [round(float(v), 4) for v in p_obj],
                "center_distance_m": round(float(np.linalg.norm(
                    np.asarray(c, float) - pts.mean(0))), 4),
                "confidence": self._density_confidence(int(id)),
                "source": "tool_measured"}

    def evaluate_volume(self, volume_id: int | None = None, center: list | None = None,
                        size: list | None = None, yaw_deg: float = 0.0,
                        voxel_m: float = 0.1) -> dict:
        """Evaluate a user-defined space: which objects fall inside, how much of
        the box is occupied vs free (voxel occupancy), and the free volume in m³.
        Pass a saved volume_id OR an explicit center+size (+yaw)."""
        r = self._resolve_volume(center, size, yaw_deg, volume_id)
        if r is None:
            return _insufficient("provide volume_id or center+size")
        c, s, yaw = r
        T, half = self._volume_frame(c, s, yaw)
        box_vol = float(np.prod(np.abs(s)))
        pts, owners = self._all_points()
        if not len(pts):
            return _insufficient("no reconstructed points to evaluate the volume")
        inside = self._inside_mask(pts, T, half)
        n_in = int(inside.sum())
        # per-object occupancy
        objs = []
        for iid in np.unique(owners[inside]) if n_in else []:
            cnt = int((owners[inside] == iid).sum())
            objs.append({"id": int(iid), "label": self._label_of(int(iid)),
                         "points_inside": cnt})
        objs.sort(key=lambda o: -o["points_inside"])
        # voxel occupancy of the box interior
        voxel_m = max(0.02, float(voxel_m))
        n_vox = np.maximum(1, np.floor(np.abs(s) / voxel_m)).astype(int)
        total_vox = int(np.prod(n_vox))
        occ_frac = 0.0
        if n_in:
            local = obb_local_coords(pts[inside], T) + half  # 0..size
            keys = np.floor(local / voxel_m).astype(np.int64)
            keys = np.clip(keys, 0, n_vox - 1)
            occ_vox = len(np.unique(keys, axis=0))
            occ_frac = min(1.0, occ_vox / total_vox)
        free_frac = float(max(0.0, 1.0 - occ_frac))
        return {"volume_id": (int(volume_id) if volume_id is not None else None),
                "explicit_box": volume_id is None,
                "box_volume_m3": round(box_vol, 4),
                "occupied_fraction": round(occ_frac, 4),
                "free_fraction": round(free_frac, 4),
                "free_volume_m3": round(free_frac * box_vol, 4),
                "points_inside": n_in, "objects_inside": objs,
                "voxel_m": voxel_m, "unit": "m", "source": "tool_measured",
                "confidence": 0.7 if n_in else 0.3}

    def objects_in_volume(self, volume_id: int | None = None, center: list | None = None,
                          size: list | None = None, yaw_deg: float = 0.0) -> dict:
        """List the objects that intersect a user-defined volume, with the
        fraction of each object's points that lie inside."""
        r = self._resolve_volume(center, size, yaw_deg, volume_id)
        if r is None:
            return _insufficient("provide volume_id or center+size")
        c, s, yaw = r
        T, half = self._volume_frame(c, s, yaw)
        out = []
        for i in self.store.list_instances():
            pts = self._points(i["instance_id"])
            if pts is None or not len(pts):
                continue
            inside = self._inside_mask(pts, T, half)
            frac = float(inside.mean())
            if inside.any():
                out.append({"id": i["instance_id"], "label": i["label"],
                            "fraction_inside": round(frac, 3),
                            "points_inside": int(inside.sum())})
        out.sort(key=lambda o: -o["fraction_inside"])
        return {"objects": out, "count": len(out), "source": "tool_measured"}

    def fits_in_volume(self, item_size: list, volume_id: int | None = None,
                       center: list | None = None, size: list | None = None,
                       yaw_deg: float = 0.0, voxel_m: float = 0.1) -> dict:
        """Can an axis-aligned item of given size (w,h,d metres) fit in the FREE
        space of a user-defined volume? Voxelizes the box, marks voxels occupied
        by any scene point, and slides the item box over free voxels. Returns
        whether it fits and a candidate placement (box-local, metres)."""
        r = self._resolve_volume(center, size, yaw_deg, volume_id)
        if r is None or item_size is None or len(item_size) != 3:
            return _insufficient("provide item_size [w,h,d] and volume_id or center+size")
        c, s, yaw = r
        T, half = self._volume_frame(c, s, yaw)
        item = np.abs(np.asarray(item_size, float))
        if np.any(item > np.abs(s) + 1e-9):
            return {"fits": False, "reason": "item larger than the volume",
                    "source": "tool_measured"}
        voxel_m = max(0.02, float(voxel_m))
        n_vox = np.maximum(1, np.floor(np.abs(s) / voxel_m)).astype(int)
        occ = np.zeros(tuple(n_vox), bool)
        pts, _o = self._all_points()
        inside = self._inside_mask(pts, T, half) if len(pts) else np.zeros(0, bool)
        if inside.any():
            local = obb_local_coords(pts[inside], T) + half
            keys = np.clip(np.floor(local / voxel_m).astype(int), 0, n_vox - 1)
            occ[keys[:, 0], keys[:, 1], keys[:, 2]] = True
        # integral image for O(1) empty-box tests
        free = (~occ).astype(np.int64)
        ii = free.cumsum(0).cumsum(1).cumsum(2)
        need = np.maximum(1, np.ceil(item / voxel_m).astype(int))
        if np.any(need > n_vox):
            return {"fits": False, "reason": "item larger than the volume",
                    "source": "tool_measured"}
        placement = self._find_empty_box(ii, n_vox, need)
        if placement is None:
            return {"fits": False, "reason": "no free region large enough",
                    "free_volume_m3": round(float(free.sum()) * voxel_m ** 3, 4),
                    "source": "tool_measured"}
        pos_local = (np.asarray(placement) * voxel_m).tolist()
        return {"fits": True, "placement_box_local_m": [round(v, 3) for v in pos_local],
                "item_size_m": item.tolist(), "voxel_m": voxel_m,
                "source": "tool_measured", "confidence": 0.6}

    @staticmethod
    def _find_empty_box(ii: np.ndarray, n_vox: np.ndarray, need: np.ndarray):
        """First origin where a `need`-sized box has zero occupied voxels, via a
        3D summed-area table. Returns (i,j,k) origin or None."""
        nx, ny, nz = n_vox
        dx, dy, dz = need
        # pad ii by one at the front for clean differencing
        P = np.zeros((nx + 1, ny + 1, nz + 1), np.int64)
        P[1:, 1:, 1:] = ii
        for i in range(0, nx - dx + 1):
            for j in range(0, ny - dy + 1):
                for k in range(0, nz - dz + 1):
                    a, b, c2 = i + dx, j + dy, k + dz
                    tot = (P[a, b, c2] - P[i, b, c2] - P[a, j, c2] - P[a, b, k]
                           + P[i, j, c2] + P[i, b, k] + P[a, j, k] - P[i, j, k])
                    if tot == dx * dy * dz:  # all-free box
                        return (i, j, k)
        return None

    # ── session / scene awareness (2026-08-29, user: the assistant must know
    #    WHAT it is looking at, not only measure) ──────────────────────
    def get_session_info(self) -> dict:
        """WHICH session/scene is open: session + project name, frame count,
        cloud size, object count, floor leveling state, and the cached scene
        description when one exists."""
        out_dir = Path(self.store.path).parent
        session_dir = out_dir.parent
        parts = session_dir.resolve().parts
        project = None
        if "projects" in parts:
            k = parts.index("projects")
            if k + 1 < len(parts):
                project = parts[k + 1]
        n_frames = len(list((session_dir / "frames").glob("*.jpg"))) \
            if (session_dir / "frames").is_dir() else 0
        n_cloud = None
        cp = out_dir / "cleaned_cloud.ply"
        if cp.exists():
            try:
                with open(cp, "rb") as fp:
                    for ln in fp:
                        if ln.startswith(b"element vertex"):
                            n_cloud = int(ln.split()[-1])
                        if ln.startswith(b"end_header"):
                            break
            except Exception:  # noqa: BLE001
                pass
        insts = self.store.list_instances()
        return {"project": project, "session": session_dir.name,
                "scan_frames": n_frames, "cloud_points": n_cloud,
                "n_objects": len(insts),
                "object_labels": [i["label"] for i in insts],
                "scene_type": self.store.get_meta("scene_type"),
                "scene_description": self.store.get_meta("scene_description"),
                "floor_leveled": (out_dir / "floor_transform.npz").exists()
                                 or (out_dir / ".orientation_applied").exists(),
                "source": "tool_measured"}

    def describe_scene(self, refresh: bool = False) -> dict:
        """WHAT the scene is: the VLM LOOKS at sampled scan keyframes and
        writes a short description (kind of space, structures, materials,
        anything notable). Cached in the scene db (provenance vlm_proposed);
        refresh=true re-runs. INTERPRETATION, never measurement — figures
        still come only from the measuring tools."""
        cached = self.store.get_meta("scene_description")
        if cached and not refresh:
            return {"description": cached, "cached": True,
                    "source": "vlm_proposed"}
        session_dir = Path(self.store.path).parent.parent
        frames = sorted((session_dir / "frames").glob("*.jpg"))
        if not frames:
            return _insufficient("no scan frames on disk to look at")
        step = max(1, len(frames) // 6)
        pick = frames[::step][:6]
        try:
            from semantic.client import get_semantic_client
            from semantic.types import system as _sys, user as _usr
            client = get_semantic_client(consumer="phase5.describe_scene")
            resp = client.chat([
                _sys("You describe 3D-scanned construction scenes from their "
                     "scan frames, for a site engineer."),
                _usr("Describe this scanned scene: what kind of space it is, "
                     "the main structures and materials, and anything notable. "
                     "Concrete and concise (max 120 words). Answer in Spanish.",
                     images=[str(p) for p in pick])])
        except Exception as e:  # noqa: BLE001
            return _insufficient(f"semantic service unavailable ({e}) — try "
                                 "again once the model is loaded")
        desc = (resp.content or "").strip()
        if desc:
            self.store.set_meta("scene_description", desc)
            self.store.set_meta("scene_description_origin", "vlm_proposed")
        return {"description": desc or "(no answer)", "cached": False,
                "frames_used": [p.name for p in pick],
                "source": "vlm_proposed"}

    def remember_note(self, key: str, note: str) -> dict:
        """Persist a conclusion/observation about this scene in its db so later
        conversations can use it. Notes are vlm_proposed context, NEVER
        measurements."""
        import json as _json
        import time as _time
        try:
            blob = _json.loads(self.store.get_meta("chat_notes") or "{}")
        except Exception:  # noqa: BLE001
            blob = {}
        blob[str(key)] = {"note": str(note),
                          "ts": _time.strftime("%Y-%m-%d %H:%M")}
        self.store.set_meta("chat_notes", _json.dumps(blob, ensure_ascii=False))
        return {"ok": True, "stored": str(key), "n_notes": len(blob),
                "source": "vlm_proposed"}

    def describe_object(self, id: int) -> dict:
        """Detailed dossier of one segmented object (que_es, características,
        materiales, estado, función, interacción con el entorno) generated by
        the shape proposer's deep-analysis pass and cached in the store.
        vlm_proposed context — measurements still come from the measure
        tools."""
        import json as _json
        raw = self.store.get_meta(f"object_analysis_{int(id)}")
        analysis = None
        if raw:
            try:
                analysis = _json.loads(raw)
            except Exception:  # noqa: BLE001
                analysis = None
        if analysis is None:
            # USER 2026-09-04: the dossier must exist ALWAYS — never depend
            # on the CAD pass. Generate it right here (the chat only runs
            # while the VLM is up).
            try:
                from pathlib import Path as _P
                from segmentation.object_analysis import analyze_object
                analysis = analyze_object(_P(self.store.path).parent, int(id),
                                          log=lambda *_a: None)
            except Exception as e:  # noqa: BLE001
                return _insufficient(f"could not analyze object {id}: {e}")
        if not analysis:
            return _insufficient(f"analysis of object {id} produced nothing")
        return {"id": int(id), "label": self._label_of(int(id)),
                "analysis": analysis, "source": "vlm_proposed"}

    def recall_notes(self, key: str | None = None) -> dict:
        """Read the notes previously stored about this scene (all, or one key)."""
        import json as _json
        try:
            blob = _json.loads(self.store.get_meta("chat_notes") or "{}")
        except Exception:  # noqa: BLE001
            blob = {}
        if key is not None:
            return {"note": blob.get(str(key)), "key": key,
                    "source": "vlm_proposed"}
        return {"notes": blob, "n_notes": len(blob), "source": "vlm_proposed"}

    # ── tool-calling schemas for the orchestrator ───────────────────
    def impls(self) -> dict:
        return {
            "list_objects": self.list_objects, "count_objects": self.count_objects,
            "get_position": self.get_position, "get_object_size": self.get_object_size,
            "get_object_volume": self.get_object_volume, "get_distance": self.get_distance,
            "get_my_position": self.get_my_position,
            "get_distance_from_me": self.get_distance_from_me,
            "get_clearance": self.get_clearance, "measure_between": self.measure_between,
            "get_extent": self.get_extent,
            "get_session_info": self.get_session_info,
            "describe_scene": self.describe_scene,
            "describe_object": self.describe_object,
            "remember_note": self.remember_note,
            "recall_notes": self.recall_notes,
            "fits_through": self.fits_through, "get_plumb": self.get_plumb,
            "get_level": self.get_level, "get_span": self.get_span,
            "get_height_profile": self.get_height_profile,
            "get_flatness_report": self.get_flatness_report,
            "get_alignment_health": self.get_alignment_health,
            "get_onion_report": self.get_onion_report,
            "get_instance_history": self.get_instance_history,
            "get_findings": self.get_findings,
            "define_volume": self.define_volume,
            "list_volumes": self.list_volumes,
            "measure_volume": self.measure_volume,
            "evaluate_volume": self.evaluate_volume,
            "objects_in_volume": self.objects_in_volume,
            "fits_in_volume": self.fits_in_volume,
        }

    def openai_schemas(self) -> list[dict]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props or {},
                               "required": required or []}}}
        idp = {"type": "integer", "description": "object id from list_objects"}
        arr = {"type": "array", "items": {"type": "number"}}
        return [
            fn("list_objects", "List tracked objects (id + class + material/state). Call this first.",
               {"label": {"type": "string", "description": "optional class filter"}}),
            fn("count_objects", "Count objects, optionally of a class.",
               {"label": {"type": "string"}}),
            fn("get_position", "World position (metres) of an object.", {"id": idp}, ["id"]),
            fn("get_object_size", "Object size w/h/d in metres.", {"id": idp}, ["id"]),
            fn("get_object_volume", "Bounding-box volume in cubic metres.", {"id": idp}, ["id"]),
            fn("get_my_position", "Position of the camera at the end of the scan ('me')."),
            fn("get_distance_from_me", "Distance from the scan-end camera to an object (m).",
               {"id": idp}, ["id"]),
            fn("get_distance", "Approx surface distance between two objects (m).",
               {"id1": idp, "id2": idp}, ["id1", "id2"]),
            fn("get_clearance", "Minimum real clearance between two objects (m) with the closest point pair.",
               {"id1": idp, "id2": idp}, ["id1", "id2"]),
            fn("measure_between",
               "Distance between NAMED PARTS of two objects (m), from their real "
               "point clouds. Parts: centroid | top/upper | bottom/base/lower "
               "(surface bands — a curved ceiling has BOTH) | highest/lowest "
               "(extreme points) | closest. axis='vertical' → height difference "
               "|Δy| (e.g. free height floor→ceiling upper part); 'straight' → 3-D.",
               {"id1": idp, "id2": idp,
                "feature1": {"type": "string",
                             "enum": ["centroid", "top", "upper", "bottom", "base",
                                      "lower", "highest", "lowest", "closest"]},
                "feature2": {"type": "string",
                             "enum": ["centroid", "top", "upper", "bottom", "base",
                                      "lower", "highest", "lowest", "closest"]},
                "axis": {"type": "string", "enum": ["straight", "vertical"]}},
               ["id1", "id2"]),
            fn("get_extent",
               "True point-cloud extent of an object: min/max XYZ, height, and "
               "the centres of its lowest and highest surface bands (where the "
               "lower and upper parts of a curved/composite object are).",
               {"id": idp}, ["id"]),
            fn("get_session_info",
               "Which session/scene is open: project + session name, scan frame "
               "count, cloud size, objects, floor leveling, cached scene "
               "description. Call for any 'what am I looking at' question."),
            fn("describe_scene",
               "LOOK at sampled scan keyframes (VLM) and describe what the scene "
               "is — kind of space, structures, materials, notable items. Cached "
               "in the scene db; refresh=true re-runs. Interpretation "
               "(vlm_proposed), NEVER measurements.",
               {"refresh": {"type": "boolean"}}),
            fn("describe_object",
               "Detailed dossier of ONE segmented object: qué es, descripción, "
               "características, materiales, estado, función, interacción con "
               "el entorno (vlm_proposed, cached by the CAD/propose pass). Use "
               "when asked what an object is / its materials / condition.",
               {"id": idp}, ["id"]),
            fn("remember_note",
               "Persist a conclusion/observation about this scene in its db for "
               "future conversations (key + note). Never store figures that did "
               "not come from a tool.",
               {"key": {"type": "string"}, "note": {"type": "string"}},
               ["key", "note"]),
            fn("recall_notes",
               "Read notes previously stored about this scene (all, or one key).",
               {"key": {"type": "string"}}),
            fn("fits_through",
               "Can an item [w,h,d] m pass through an opening (door/window: "
               "opening_id) or through the gap between two objects (id1+id2)?",
               {"item_size": {**arr, "description": "[w,h,d] metres"},
                "opening_id": idp, "id1": idp, "id2": idp,
                "margin_m": {"type": "number"}}, ["item_size"]),
            fn("get_plumb", "Out-of-plumb (tilt from vertical) of an element, degrees + mm/m.",
               {"id": idp}, ["id"]),
            fn("get_level", "Out-of-level (tilt from horizontal) of a surface, degrees + mm/m.",
               {"id": idp}, ["id"]),
            fn("get_span", "Largest horizontal extent (span) of an object, metres.", {"id": idp}, ["id"]),
            fn("get_height_profile",
               "Height/clearance profile (gálibos) along an object's longest axis; "
               "pass floor_id for floor-to-underside clearances.",
               {"id": idp, "n_stations": {"type": "integer"}, "floor_id": idp}, ["id"]),
            fn("get_flatness_report",
               "Flatness report (mm): surface_fitting residuals when available, "
               "plane-fit fallback otherwise.",
               {"id": idp, "tolerance_mm": {"type": "number"}}, ["id"]),
            fn("get_alignment_health", "Alignment health: vote entropy + onion metric.", {"id": idp}, ["id"]),
            fn("get_onion_report",
               "Doubled-surface report: mode separation (m) + state (+ per-seam via seam).",
               {"id": idp, "seam": {"type": "string"}}, ["id"]),
            fn("get_instance_history", "Windows seen + residuals for an object.", {"id": idp}, ["id"]),
            fn("get_findings", "Visual findings for an object/region (Phase 3).",
               {"id": idp, "region": {"type": "string"}}),
            fn("define_volume",
               "Place a user-defined evaluation box. DEFAULT: it RESTS ON the "
               "floor, centred on the floor (or on anchor_id's x/z). Pass size "
               "[w,h,d] m or volume_m3 (cube). Only pass center to override "
               "x/z; center y is honoured only with on_floor=false.",
               {"name": {"type": "string"},
                "size": {**arr, "description": "[w,h,d] full extents, metres"},
                "volume_m3": {"type": "number", "description": "cube of this volume instead of size"},
                "center": {**arr, "description": "optional [x,y,z] override, metres"},
                "anchor_id": {"type": "integer", "description": "object whose centroid gives x/z (default: the floor)"},
                "on_floor": {"type": "boolean", "description": "default true: base sits ON the floor"},
                "yaw_deg": {"type": "number", "description": "rotation about vertical"}},
               ["name"]),
            fn("list_volumes", "The evaluation volumes saved in this scene."),
            fn("measure_volume",
               "Measure FROM a saved volume to an object: clearance box-surface → "
               "object points (intersects=true when points are inside), plus "
               "centre distance. axis='vertical' → |Δy|.",
               {"volume_id": {"type": "integer"}, "id": idp,
                "axis": {"type": "string", "enum": ["straight", "vertical"]}},
               ["volume_id", "id"]),
            fn("evaluate_volume",
               "Evaluate a space: objects inside, occupied vs free fraction, and "
               "free volume (m³). Pass volume_id OR center+size(+yaw).",
               {"volume_id": {"type": "integer"}, "center": arr, "size": arr,
                "yaw_deg": {"type": "number"}}),
            fn("objects_in_volume",
               "List objects intersecting a user-defined volume with each one's "
               "fraction inside. Pass volume_id OR center+size(+yaw).",
               {"volume_id": {"type": "integer"}, "center": arr, "size": arr,
                "yaw_deg": {"type": "number"}}),
            fn("fits_in_volume",
               "Check whether an item of size [w,h,d] metres fits in the FREE space "
               "of a user-defined volume; returns a candidate placement.",
               {"item_size": {**arr, "description": "[w,h,d] metres"},
                "volume_id": {"type": "integer"}, "center": arr, "size": arr,
                "yaw_deg": {"type": "number"}},
               ["item_size"]),
        ]
