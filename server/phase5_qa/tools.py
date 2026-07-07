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
        Returns insufficient_data when no findings have been recorded."""
        finds = self.store.list_findings(instance_id=id)
        if not finds:
            scope = f"object {id}" if id is not None else "the scene"
            return _insufficient(f"no findings recorded for {scope}")
        out = [{"finding_id": f["finding_id"], "type": f["type"], "severity": f["severity"],
                "description": f["description"], "confidence": round(f["confidence"], 2),
                "instance_id": f["instance_id"], "point3d": f["point3d"],
                "status": f["status"], "correlated_residual": f["correlated_residual"],
                "provenance": ("human_validated" if f["status"] == "human_validated"
                               else "vlm_proposed")}
               for f in finds]
        return {"findings": out, "count": len(out), "source": "vlm_proposed",
                "note": "findings are proposed and require human validation"}

    # ── user-defined evaluation volumes (space evaluation) ──────────
    def _all_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Pool every instance's world points once, with a parallel array of the
        owning instance id (for occupancy / space-evaluation queries)."""
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
        """Return (center, size, yaw_deg) from explicit args or a saved volume."""
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

    def define_volume(self, name: str, center: list, size: list, yaw_deg: float = 0.0) -> dict:
        """Persist a user-defined evaluation box (world centre + full extents in
        metres, yaw about vertical). Returns its volume_id for later queries."""
        if center is None or size is None or len(center) != 3 or len(size) != 3:
            return _insufficient("center and size must each be [x,y,z] in metres")
        vid = self.store.add_user_volume(name or f"volume", center, size, yaw_deg or 0.0)
        return {"volume_id": vid, "name": name, "center": list(map(float, center)),
                "size": list(map(float, size)), "yaw_deg": float(yaw_deg or 0.0),
                "source": "user_defined"}

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
            inst = self._instance(int(iid))
            objs.append({"id": int(iid), "label": inst["label"] if inst else str(iid),
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
        return {"volume_id": volume_id, "box_volume_m3": round(box_vol, 4),
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

        def s(i, j, k):  # inclusive-exclusive box sum on the padded integral image
            return int(ii[i, j, k])
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
            "define_volume": self.define_volume,
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
            fn("define_volume",
               "Persist a user-defined evaluation box (world centre + full extents "
               "in metres, yaw about vertical). Use to mark a space to evaluate.",
               {"name": {"type": "string"},
                "center": {"type": "array", "items": {"type": "number"},
                           "description": "[x,y,z] world centre, metres"},
                "size": {"type": "array", "items": {"type": "number"},
                         "description": "[w,h,d] full extents, metres"},
                "yaw_deg": {"type": "number", "description": "rotation about vertical"}},
               ["name", "center", "size"]),
            fn("evaluate_volume",
               "Evaluate a space: objects inside, occupied vs free fraction, and "
               "free volume (m³). Pass volume_id OR center+size(+yaw).",
               {"volume_id": {"type": "integer"},
                "center": {"type": "array", "items": {"type": "number"}},
                "size": {"type": "array", "items": {"type": "number"}},
                "yaw_deg": {"type": "number"}}),
            fn("objects_in_volume",
               "List objects intersecting a user-defined volume with each one's "
               "fraction inside. Pass volume_id OR center+size(+yaw).",
               {"volume_id": {"type": "integer"},
                "center": {"type": "array", "items": {"type": "number"}},
                "size": {"type": "array", "items": {"type": "number"}},
                "yaw_deg": {"type": "number"}}),
            fn("fits_in_volume",
               "Check whether an item of size [w,h,d] metres fits in the FREE space "
               "of a user-defined volume; returns a candidate placement.",
               {"item_size": {"type": "array", "items": {"type": "number"},
                              "description": "[w,h,d] metres"},
                "volume_id": {"type": "integer"},
                "center": {"type": "array", "items": {"type": "number"}},
                "size": {"type": "array", "items": {"type": "number"}},
                "yaw_deg": {"type": "number"}},
               ["item_size"]),
        ]
