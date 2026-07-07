# STAC-Builder — Phase 5: deterministic spatial tools over the R.8 store.
#
# THE MEASURING LAYER. The VLM proposes/orchestrates; these pure functions do
# ALL the measuring, over the canonical instance store (points, OBBs, class,
# vote/onion metrics). Every tool: typed, SI units, a confidence estimate, and
# returns {"insufficient_data": true} rather than extrapolating when it lacks
# support. Each has a synthetic-geometry unit test.
#
# PROVENANCE:
#   Ported/adapted from R3D vendor/r3d/r3d/pipeline/eval/tool_use.py:
#     list_objects, get_position, get_object_size, get_object_volume,
#     get_distance (OBB closest-point math _closest_point_on_obb/_invert_rigid).
#   Ours (construction-supervision tools): get_clearance, get_plumb, get_level,
#     get_span, count_objects, get_alignment_health, get_onion_report,
#     get_instance_history, get_findings.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from typing import Any

import numpy as np

from phase_r.geometry import invert_rigid, obb_local_coords
from phase_r.instance_store import InstanceStore


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
    `openai_schemas()` returns the tool-calling specs for the orchestrator."""

    def __init__(self, store: InstanceStore):
        self.store = store
        self._pts_cache: dict[int, np.ndarray | None] = {}

    # ── helpers ─────────────────────────────────────────────────────
    def _instance(self, iid: int) -> dict | None:
        for i in self.store.list_instances():
            if i["instance_id"] == iid:
                return i
        return None

    def _points(self, iid: int) -> np.ndarray | None:
        if iid not in self._pts_cache:
            self._pts_cache[iid] = self.store.get_points(iid)
        return self._pts_cache[iid]

    def _density_confidence(self, iid: int) -> float:
        pts = self._points(iid)
        if pts is None or len(pts) < 30:
            return 0.2
        return float(min(1.0, 0.4 + len(pts) / 5000.0))

    # ── discovery (R3D) ─────────────────────────────────────────────
    def list_objects(self, label: str | None = None) -> dict:
        """List tracked objects (id + class), optionally filtered by class."""
        insts = self.store.list_instances()
        if label:
            insts = [i for i in insts if label.lower() in i["label"].lower()]
        return {"objects": [{"id": i["instance_id"], "label": i["label"],
                             "status": i["status"]} for i in insts]}

    def count_objects(self, label: str | None = None) -> dict:
        """Count objects, optionally of a given class."""
        return {"count": len(self.list_objects(label)["objects"]), "unit": "count"}

    # ── position / size / volume (R3D) ──────────────────────────────
    def get_position(self, id: int) -> dict:
        """World position (OBB centre) of an object, metres (x,y,z)."""
        obb = self.store.get_obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, _a, pos = obb
        return {"position_m": [round(float(v), 4) for v in pos], "unit": "m",
                "confidence": self._density_confidence(id), "source": "tool_measured"}

    def get_object_size(self, id: int) -> dict:
        """Object bounding-box size width/height/depth, metres (OBB extents)."""
        obb = self.store.get_obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, aabb, _p = obb
        return {"width_m": round(float(aabb[1] - aabb[0]), 4),
                "height_m": round(float(aabb[3] - aabb[2]), 4),
                "depth_m": round(float(aabb[5] - aabb[4]), 4), "unit": "m",
                "confidence": self._density_confidence(id), "source": "tool_measured"}

    def get_object_volume(self, id: int) -> dict:
        """Bounding-box volume of an object, cubic metres."""
        obb = self.store.get_obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, aabb, _p = obb
        v = float((aabb[1] - aabb[0]) * (aabb[3] - aabb[2]) * (aabb[5] - aabb[4]))
        return {"bbox_volume_m3": round(v, 5), "unit": "m^3",
                "confidence": self._density_confidence(id), "source": "tool_measured"}

    # ── distances (R3D get_distance + our clearance) ────────────────
    def get_distance(self, id1: int, id2: int) -> dict:
        """Approx. surface distance between two objects (OBB-to-OBB), metres."""
        o1, o2 = self.store.get_obb(id1), self.store.get_obb(id2)
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

    # ── plumb / level (ours, vs gravity) ────────────────────────────
    def get_plumb(self, id: int) -> dict:
        """Out-of-plumb of a vertical element: tilt of its principal vertical
        axis from true vertical, degrees + mm-per-metre (needs gravity)."""
        obb = self.store.get_obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        T, aabb, _p = obb
        # OBB column 1 is the up axis (gravity-aligned fit); if gravity was known
        # the tilt is 0 by construction, so we measure the point cloud's actual
        # principal axis vs vertical instead.
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
                "source": "tool_measured",
                "note": "vs assumed +Y up; supply gravity for georeferenced plumb"}

    def get_level(self, id: int) -> dict:
        """Out-of-level of a horizontal element: tilt of its plane from
        horizontal, degrees + mm-per-metre."""
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
        obb = self.store.get_obb(id)
        if obb is None:
            return _insufficient(f"no OBB for object {id}")
        _T, aabb, _p = obb
        return {"span_m": round(float(max(aabb[1] - aabb[0], aabb[5] - aabb[4])), 4),
                "unit": "m", "confidence": self._density_confidence(id),
                "source": "tool_measured"}

    # ── Phase R alignment health (ours) ─────────────────────────────
    def get_alignment_health(self, id: int) -> dict:
        """Alignment health of an object: vote entropy + onion metric (Phase R)."""
        m = self.store.get_metrics(id)
        if not m.get("vote") and not m.get("onion"):
            return _insufficient(f"no alignment metrics for object {id}")
        return {"vote": m.get("vote"), "onion": m.get("onion"), "source": "tool_measured"}

    def get_onion_report(self, id: int) -> dict:
        """Onion (doubled-surface) report for an object: mode separation in
        metres and whether it is bimodal (a registration-error signal)."""
        m = self.store.get_metrics(id).get("onion")
        if m is None:
            return _insufficient(f"no onion metric for object {id}")
        state = "doubled_surface" if m["bimodal"] else "clean"
        return {"separation_m": round(float(m["separation_m"]), 4),
                "bimodal": bool(m["bimodal"]), "state": state, "unit": "m",
                "source": "tool_measured"}

    def get_instance_history(self, id: int) -> dict:
        """Windows an object was seen in and the residuals applied (Phase R)."""
        windows = self.store.list_obb_windows(id)
        inst = self._instance(id)
        if inst is None:
            return _insufficient(f"unknown object {id}")
        return {"windows": windows, "n_views": inst.get("n_views"),
                "first_frame": inst.get("first_frame"), "last_frame": inst.get("last_frame"),
                "source": "tool_measured"}

    def get_findings(self, id: int | None = None, region: str | None = None) -> dict:
        """Visual findings (cracks/moisture/…) for an object/region (Phase 3).
        Returns insufficient_data until Phase 3 has populated findings."""
        return _insufficient("no findings store yet (Phase 3 not populated)")

    # ── tool-calling schemas for the orchestrator ───────────────────
    def impls(self) -> dict:
        return {
            "list_objects": self.list_objects, "count_objects": self.count_objects,
            "get_position": self.get_position, "get_object_size": self.get_object_size,
            "get_object_volume": self.get_object_volume, "get_distance": self.get_distance,
            "get_clearance": self.get_clearance, "get_plumb": self.get_plumb,
            "get_level": self.get_level, "get_span": self.get_span,
            "get_alignment_health": self.get_alignment_health,
            "get_onion_report": self.get_onion_report,
            "get_instance_history": self.get_instance_history,
            "get_findings": self.get_findings,
        }

    def openai_schemas(self) -> list[dict]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props or {},
                               "required": required or []}}}
        idp = {"type": "integer", "description": "object id from list_objects"}
        return [
            fn("list_objects", "List tracked objects (id + class). Call this first.",
               {"label": {"type": "string", "description": "optional class filter"}}),
            fn("count_objects", "Count objects, optionally of a class.",
               {"label": {"type": "string"}}),
            fn("get_position", "World position (metres) of an object.", {"id": idp}, ["id"]),
            fn("get_object_size", "Object size w/h/d in metres.", {"id": idp}, ["id"]),
            fn("get_object_volume", "Bounding-box volume in cubic metres.", {"id": idp}, ["id"]),
            fn("get_distance", "Approx surface distance between two objects (m).",
               {"id1": idp, "id2": idp}, ["id1", "id2"]),
            fn("get_clearance", "Minimum real clearance between two objects (m) with the closest point pair.",
               {"id1": idp, "id2": idp}, ["id1", "id2"]),
            fn("get_plumb", "Out-of-plumb (tilt from vertical) of an element, degrees + mm/m.",
               {"id": idp}, ["id"]),
            fn("get_level", "Out-of-level (tilt from horizontal) of a surface, degrees + mm/m.",
               {"id": idp}, ["id"]),
            fn("get_span", "Largest horizontal extent (span) of an object, metres.", {"id": idp}, ["id"]),
            fn("get_alignment_health", "Alignment health: vote entropy + onion metric.", {"id": idp}, ["id"]),
            fn("get_onion_report", "Doubled-surface report: mode separation (m) + state.", {"id": idp}, ["id"]),
            fn("get_instance_history", "Windows seen + residuals for an object.", {"id": idp}, ["id"]),
            fn("get_findings", "Visual findings for an object/region (Phase 3).",
               {"id": idp, "region": {"type": "string"}}),
        ]
