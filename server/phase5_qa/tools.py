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
import re
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


_CHAT_CONFIG_CACHE: dict[str, Any] = {}
_DEVICE_OVERRIDE: str | None = None


def use_device(dev: str | None) -> str | None:
    """Set the device of every tool set built without one (None = back to
    ``semantic.chat.tools_device``); returns the previous override. tests/conftest.py runs
    the suite on 'cpu'."""
    global _DEVICE_OVERRIDE
    prev = _DEVICE_OVERRIDE
    if dev is not None:
        from . import tools_torch as _tt
        dev = _tt.resolve_device(dev)
    _DEVICE_OVERRIDE = dev
    return prev


def effective_device() -> str:
    """The device the chat's point tools run on: the override, else the configured key."""
    return _DEVICE_OVERRIDE if _DEVICE_OVERRIDE is not None else chat_config()["tools_device"]


def chat_config() -> dict[str, Any]:
    """``semantic.chat`` of config.yaml, typed: ``list_limit`` (the objects a listing names one
    by one; beyond it the listing is by label, with counts and id ranges), ``history_turns``
    (the earlier turns of the conversation a question carries) and ``tools_device`` (numpy |
    cpu | cuda — where the measurements over the points run, tools_torch). A missing key fails
    naming it."""
    from semantic import semantic_config as _sc
    # read once per config.yaml on disk (its mtime): the file is 4,700 lines of YAML and a
    # question's tools ask for these keys several times
    path = getattr(_sc, "_CONFIG_PATH", None)
    try:
        stamp = (str(path), path.stat().st_mtime_ns) if path is not None else None
    except OSError:
        stamp = None
    if stamp is not None and _CHAT_CONFIG_CACHE.get("stamp") == stamp:
        return dict(_CHAT_CONFIG_CACHE["value"])
    chat = (_sc.load_semantic_config() or {}).get("chat")
    if not isinstance(chat, dict):
        raise RuntimeError("config.yaml: semantic.chat is missing")
    out = {}
    for k, lo in (("list_limit", 1), ("history_turns", 0), ("max_tokens", 1), ("command_candidates", 1)):
        v = chat.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise RuntimeError(f"config.yaml: semantic.chat.{k} is missing or not an integer >= {lo}")
        out[k] = v
    from . import tools_torch as _tt
    dev = chat.get("tools_device")
    if not isinstance(dev, str) or dev not in _tt.DEVICES:
        raise RuntimeError(f"config.yaml: {_tt.KEY} is missing or not one of {list(_tt.DEVICES)}")
    out["tools_device"] = dev
    if stamp is not None:
        _CHAT_CONFIG_CACHE.update(stamp=stamp, value=dict(out))
    return out


def norm_label(text: Any) -> str:
    """A label as it is compared: lower case, '_' and '-' as spaces, single spaces — the list
    writes fire_extinguisher, a person writes "fire extinguisher"."""
    return " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())


def id_ranges(ids) -> str:
    """Sorted ids as ranges: [2, 3, 4, 9] -> '2-4, 9'."""
    v = sorted({int(i) for i in ids})
    out, k = [], 0
    while k < len(v):
        j = k
        while j + 1 < len(v) and v[j + 1] == v[j] + 1:
            j += 1
        out.append(str(v[k]) if j == k else f"{v[k]}-{v[j]}")
        k = j + 1
    return ", ".join(out)


# label#id as the Instances list writes it (sign#797, fire_extinguisher#425), a bare #797,
# or "id 797" / "id=797"
_REF_LABELLED = re.compile(r"([A-Za-zÀ-ÿ][\w\-]*)\s*#\s*(\d+)")
_REF_BARE = re.compile(r"(?:#\s*|\bids?\s*[=:]?\s*)(\d+)", re.IGNORECASE)


# the tools that read an object's POINTS (everything else is a lookup in the store): with
# semantic.chat.tools_device = cuda they run in the chat's own tools process
# (phase5_qa.tools_worker) — the backend never touches the card
POINT_TOOLS = frozenset({
    "get_clearance", "measure_between", "get_extent", "fits_through", "get_plumb", "get_level",
    "get_height_profile", "get_flatness_report", "define_volume", "measure_volume",
    "evaluate_volume", "objects_in_volume", "fits_in_volume"})


# the arguments of the tools that name ONE object
ID_ARGS = ("id", "id1", "id2", "floor_id", "opening_id", "anchor_id", "from_id", "to_id")


class SpatialTools:
    """Bound to one scene's InstanceStore. Methods are the tool implementations;
    `openai_schemas()` returns the tool-calling specs for the orchestrator.
    All geometry is served in the DISPLAY frame (see module docstring).

    ``device``: numpy (the path of record) | cpu | cuda — where the measurements over the
    points run (tools_torch); None = ``semantic.chat.tools_device``."""

    def __init__(self, store: InstanceStore, display_transform: np.ndarray | None = None,
                 device: str | None = None):
        from . import tools_torch as _tt
        self._tt = _tt
        self._dev = _tt.resolve_device(device if device is not None else effective_device())
        self._tpts_cache: dict[int, Any] = {}
        self._band_cache: dict[int, dict] = {}
        self._pooled = None
        self._pooled_t = None
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
        self._inst_cache: dict[int, dict] | None = None

    # ── frame helpers ────────────────────────────────────────────────
    def _to_display(self, pts: np.ndarray) -> np.ndarray:
        if self._M is None or pts is None or not len(pts):
            return pts
        ph = np.concatenate([np.atleast_2d(pts), np.ones((len(np.atleast_2d(pts)), 1))], 1)
        return (self._M @ ph.T).T[:, :3]

    def _obb(self, iid: int, window_id: str = "global"):
        """(T, aabb, pos) in the DISPLAY frame (rigid T, extents ×s)."""
        obb = self.store.get_obb(iid, window_id)
        if obb is None:
            return None
        # float64 like the points (2026-11-13): the store's float32 values, promoted, so the
        # box's inverse and every coordinate in it are computed once, the same on both paths
        obb = tuple(np.asarray(v, np.float64) for v in obb)
        if self._M is None:
            return obb
        T, aabb, _pos = obb
        R_m = self._M[:3, :3] / self._s
        T2 = np.asarray(T, float).copy()
        T2[:3, :3] = R_m @ T2[:3, :3]
        T2[:3, 3] = self._to_display(T2[:3, 3][None])[0]
        return T2, np.asarray(aabb, float) * self._s, T2[:3, 3]

    def all_obbs(self, window_id: str = "global") -> dict[int, tuple]:
        """Every object's (T, aabb, pos) in the DISPLAY frame from ONE query — the same
        values as ``_obb`` object by object."""
        raw = {iid: tuple(np.asarray(v, np.float64) for v in obb)
               for iid, obb in self.store.list_obbs(window_id).items()}
        if self._M is None:
            return raw
        R_m = self._M[:3, :3] / self._s
        out = {}
        for iid, (T, aabb, _pos) in raw.items():
            T2 = np.asarray(T, float).copy()
            T2[:3, :3] = R_m @ T2[:3, :3]
            T2[:3, 3] = self._to_display(T2[:3, 3][None])[0]
            out[iid] = (T2, np.asarray(aabb, float) * self._s, T2[:3, 3])
        return out

    # ── helpers ─────────────────────────────────────────────────────
    def _instances(self) -> dict[int, dict]:
        """The store's instances by id, read ONCE per tool set (a question's tools named
        list_instances dozens of times; nothing in the chat adds or removes an instance)."""
        if self._inst_cache is None:
            self._inst_cache = {int(i["instance_id"]): i for i in self.store.list_instances()}
        return self._inst_cache

    def _instance(self, iid: int) -> dict | None:
        try:
            return self._instances().get(int(iid))
        except (TypeError, ValueError):
            return None

    def _match_label(self, label: str | None) -> list[dict]:
        """The instances a label asks for: the normalised text inside the normalised label
        (fire extinguisher = fire_extinguisher), and when nothing matches a plural, its
        singular (signs -> sign)."""
        insts = list(self._instances().values())
        want = norm_label(label)
        if not want:
            return insts
        hit = [i for i in insts if want in norm_label(i["label"])]
        if not hit:
            for suffix in ("es", "s"):
                if want.endswith(suffix) and len(want) > len(suffix) + 1:
                    hit = [i for i in insts if want[:-len(suffix)] in norm_label(i["label"])]
                    if hit:
                        break
        return hit

    def labels_summary(self, insts: list[dict] | None = None) -> list[dict]:
        """Every label with its count and its ids as ranges, the most numerous first."""
        by: dict[str, list[int]] = {}
        for i in (insts if insts is not None else self._instances().values()):
            by.setdefault(i["label"], []).append(int(i["instance_id"]))
        n_pts = self.store.list_point_counts()
        out = []
        for lab, ids in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            row = {"label": lab, "count": len(ids), "ids": id_ranges(ids)}
            if len(ids) > 1:
                # the label's largest object (most points; the lowest id among equals): "the
                # floor", "the ceiling" of a scene with several is this one
                row["largest_id"] = min(ids, key=lambda k: (-n_pts.get(k, 0), k))
            out.append(row)
        return out

    def resolve_references(self, text: str) -> list[dict]:
        """The objects a question names by id — ``sign#797``, ``#797``, ``id 797`` — looked up
        in THIS scan's store, deterministically, before the model reads the question: each
        with whether it exists, its label, and whether the label written beside it is its own."""
        refs: dict[int, str | None] = {}
        for m in _REF_LABELLED.finditer(text or ""):
            refs.setdefault(int(m.group(2)), m.group(1))
        for m in _REF_BARE.finditer(text or ""):
            refs.setdefault(int(m.group(1)), None)
        out = []
        for iid, said in refs.items():
            inst = self._instance(iid)
            entry: dict[str, Any] = {"id": iid, "exists": inst is not None}
            if inst is None:
                same = self._match_label(said) if said else []
                if same:
                    entry["label_asked"] = said
                    entry["ids_with_that_label"] = id_ranges(i["instance_id"] for i in same)
            else:
                entry["label"] = inst["label"]
                if said and norm_label(said) not in norm_label(inst["label"]) \
                        and norm_label(inst["label"]) not in norm_label(said):
                    entry["label_asked"] = said
                obb = self._obb(iid)
                if obb is not None:
                    _T, aabb, pos = obb
                    entry["position_m"] = [round(float(v), 3) for v in pos]
                    entry["size_m"] = [round(float(aabb[1] - aabb[0]), 3),
                                       round(float(aabb[3] - aabb[2]), 3),
                                       round(float(aabb[5] - aabb[4]), 3)]
            out.append(entry)
        return out

    def _points(self, iid: int) -> np.ndarray | None:
        """An object's points in the display frame, FLOAT64. The store keeps float32; until
        2026-11-13 a display-frame store was measured in float32 arithmetic. The same values
        are now promoted first, so the numpy path and the device path (float64, the GPU ports'
        rule) agree to 1e-9 — a figure can differ from before in its last printed digit."""
        if iid not in self._pts_cache:
            pts = self.store.get_points(iid)
            self._pts_cache[iid] = (np.asarray(self._to_display(pts), np.float64)
                                    if pts is not None else None)
        return self._pts_cache[iid]

    def _tp(self, iid: int):
        """The object's points on the device (float64), kept for the life of this tool set."""
        if iid not in self._tpts_cache:
            self._tpts_cache[iid] = self._tt.tensor(self._points(iid), self._dev)
        return self._tpts_cache[iid]

    def _bands(self, iid: int) -> dict:
        if iid not in self._band_cache:
            self._band_cache[iid] = self._tt.band_stats(self._tp(iid))
        return self._band_cache[iid]

    def release(self) -> None:
        """Drop what this tool set holds on the device and hand the cached blocks back."""
        self._tpts_cache.clear()
        self._band_cache.clear()
        self._pooled_t = None
        if self._dev != "numpy":
            self._tt.release(self._dev)

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
        """List tracked objects (id + class + Phase 2 enrichment), optionally filtered by
        label. Up to ``semantic.chat.list_limit`` objects are named one by one; beyond it the
        answer is BY LABEL — each label's count and its ids as ranges — so a scene of
        thousands of objects still answers whole (pccr 2408's 8,777 objects were 486 kB of
        JSON: more than the model's context, the question failed with HTTP 400)."""
        insts = self._match_label(label)
        limit = chat_config()["list_limit"]
        if len(insts) > limit:
            return {"count": len(insts), "listed": "by_label", "labels": self.labels_summary(insts),
                    "note": f"more than {limit} objects: listed by label with id ranges — "
                            f"filter with a label, or use the ids directly"}
        cls = self.store.list_classifications()
        n_pts = self.store.list_point_counts()
        out = []
        for i in insts:
            entry = {"id": i["instance_id"], "label": i["label"], "status": i["status"],
                     "points": n_pts.get(int(i["instance_id"]), 0)}
            c = cls.get(int(i["instance_id"]))
            if c:
                entry["class"] = c["class_final"]
                if c.get("material"):
                    entry["material"] = c["material"]
                if c.get("state"):
                    entry["state"] = c["state"]
            out.append(entry)
        return {"count": len(out), "objects": out}

    def count_objects(self, label: str | None = None) -> dict:
        """Count objects of a label; with no label, the total AND the count of every label."""
        insts = self._match_label(label)
        res: dict[str, Any] = {"count": len(insts), "unit": "count"}
        if not norm_label(label):
            res["by_label"] = {r["label"]: r["count"] for r in self.labels_summary(insts)}
        else:
            res["label"] = label
            res["matched_labels"] = sorted({i["label"] for i in insts})
        return res

    def get_object(self, id: int) -> dict:
        """One object in one call: label, position, size and vertical extent."""
        inst = self._instance(id)
        if inst is None:
            return _insufficient(f"no object with id {id} in this scan")
        res: dict[str, Any] = {"id": int(inst["instance_id"]), "label": inst["label"],
                               "status": inst["status"]}
        obb = self._obb(int(inst["instance_id"]))
        if obb is not None:
            _T, aabb, pos = obb
            res["position_m"] = [round(float(v), 4) for v in pos]
            res["size_m"] = {"width": round(float(aabb[1] - aabb[0]), 4),
                             "height": round(float(aabb[3] - aabb[2]), 4),
                             "depth": round(float(aabb[5] - aabb[4]), 4)}
        res["confidence"] = self._density_confidence(int(inst["instance_id"]))
        res["unit"] = "m"
        res["source"] = "tool_measured"
        return res

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
        if self._dev == "numpy":
            from scipy.spatial import cKDTree
            tree = cKDTree(p2)
            dists, idx = tree.query(p1, k=1)
            j = int(np.argmin(dists))
            dist, i2 = float(dists[j]), int(idx[j])
        else:
            dist, j, i2 = self._tt.closest_pair(self._tp(id1), self._tp(id2))
        return {"clearance_m": round(float(dist), 4), "unit": "m",
                "point_a_m": [round(float(v), 4) for v in p1[j]],
                "point_b_m": [round(float(v), 4) for v in p2[i2]],
                "confidence": min(self._density_confidence(id1), self._density_confidence(id2)),
                "source": "tool_measured"}

    def _feat_point(self, iid: int, feature: str) -> np.ndarray | None:
        """Representative point of a NAMED PART of an object, from its REAL
        point cloud (2026-08-29: OBB face centres were wrong for curved /
        composite shapes — a curved ceiling has a lower AND an upper part, a
        ladder a base and a top; bands of the actual points capture them)."""
        pts = self._points(iid)
        if pts is not None and len(pts) >= 30 and self._dev != "numpy":
            st = self._bands(iid)
            name = {"upper": "top", "base": "bottom", "lower": "bottom"}.get(feature, feature)
            if name in ("top", "bottom", "highest", "lowest", "centroid"):
                return st[name]
        elif pts is not None and len(pts) >= 30:
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
        if self._dev == "numpy":
            mn, mx = pts.min(0), pts.max(0)
            y = pts[:, 1]
            band = max(0.05, 0.05 * float(mx[1] - mn[1]))
            low_c = pts[y <= mn[1] + band].mean(0)
            high_c = pts[y >= mx[1] - band].mean(0)
        else:
            st = self._bands(id)
            mn, mx, low_c, high_c = st["min"], st["max"], st["bottom"], st["top"]
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
    def _principal(self, id):
        """(vt, extents) of an object's points, or None under 50 points: vt's rows are its
        principal directions, the largest extent first; extents = the standard deviation of the
        points along each, metres."""
        pts = self._points(id)
        if pts is None or len(pts) < 50:
            return None
        if self._dev == "numpy":
            _u, sv, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
        else:
            _c, vt, sv = self._tt.principal_axes_values(self._tp(id))
        return vt, np.asarray(sv, float) / np.sqrt(float(len(pts)))

    @staticmethod
    def _is_face(extents) -> bool:
        """Is the element measured on its FACE (a wall, a door, a plate) or on its AXIS (a column,
        a post, a beam)? A face when its middle extent is nearer the largest than the smallest
        on a ratio scale: e1 / e0 >= e2 / e1. No threshold: the element's own three extents."""
        e0, e1, e2 = (float(v) for v in extents)
        return e1 * e1 >= e0 * e2

    def get_plumb(self, id: int) -> dict:
        """Out-of-plumb of an element, degrees + mm-per-metre. Vertical = +Y of the DISPLAY frame
        (floor-aligned). What is measured follows the element's own shape (:meth:`_is_face`):

          - a FACE (wall, door, panel): the tilt of its plane from the plumb line — the angle
            between the plane and the vertical, asin|n·y| with n the plane's normal;
          - an AXIS (column, post, riser): the tilt of its long axis from the vertical.

        Until 2026-11-14 the long axis was taken for every element: a wall's longest direction is
        horizontal, so its "out of plumb" read 89°. ``vertical_element`` False = the face / axis
        is nearer the horizontal than the vertical (plumb does not apply; mm/m not given)."""
        pr = self._principal(id)
        if pr is None:
            return _insufficient("insufficient points for plumb")
        vt, ext = pr
        vertical = np.array([0.0, 1.0, 0.0])
        face = self._is_face(ext)
        if face:
            sin = abs(float(np.dot(vt[2], vertical)))            # the plane's normal
            tilt_deg = float(np.degrees(np.arcsin(min(1.0, sin))))
        else:
            cos = abs(float(np.dot(vt[0], vertical)))            # the long axis
            tilt_deg = float(np.degrees(np.arccos(min(1.0, cos))))
        out = {"plumb_deviation_deg": round(tilt_deg, 3),
               "measured_on": "face" if face else "axis",
               "vertical_element": bool(tilt_deg < 45.0),
               "extents_m": [round(float(e), 4) for e in ext],
               "unit": "deg", "confidence": self._density_confidence(id),
               "source": "tool_measured"}
        if out["vertical_element"]:
            out["deviation_mm_per_m"] = round(float(np.tan(np.radians(tilt_deg)) * 1000), 2)
        return out

    def get_level(self, id: int) -> dict:
        """Out-of-level of an element, degrees + mm-per-metre (DISPLAY frame, floor-aligned). As
        :meth:`get_plumb`, by the element's own shape: a FACE (floor, slab, table top) — the tilt
        of its plane from the horizontal; an AXIS (beam, pipe run) — the tilt of its long axis
        from the horizontal. ``horizontal_element`` False = nearer the vertical (level does not
        apply; mm/m not given)."""
        pr = self._principal(id)
        if pr is None:
            return _insufficient("insufficient points for level")
        vt, ext = pr
        vertical = np.array([0.0, 1.0, 0.0])
        face = self._is_face(ext)
        if face:
            cos = abs(float(np.dot(vt[2], vertical)))            # the plane's normal
            tilt_deg = float(np.degrees(np.arccos(min(1.0, cos))))
        else:
            sin = abs(float(np.dot(vt[0], vertical)))            # the long axis
            tilt_deg = float(np.degrees(np.arcsin(min(1.0, sin))))
        out = {"level_deviation_deg": round(tilt_deg, 3),
               "measured_on": "face" if face else "axis",
               "horizontal_element": bool(tilt_deg < 45.0),
               "extents_m": [round(float(e), 4) for e in ext],
               "unit": "deg", "confidence": self._density_confidence(id),
               "source": "tool_measured"}
        if out["horizontal_element"]:
            out["deviation_mm_per_m"] = round(float(np.tan(np.radians(tilt_deg)) * 1000), 2)
        return out

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
        if self._dev == "numpy":
            ext_x, ext_z = aabb[1] - aabb[0], aabb[5] - aabb[4]
            axis_i = 0 if ext_x >= ext_z else 2
            local = obb_local_coords(pts, T)
            s_lo, s_hi = (aabb[0], aabb[1]) if axis_i == 0 else (aabb[4], aabb[5])
            span = max(s_hi - s_lo, 1e-9)
            station_of = np.clip(((local[:, axis_i] - s_lo) / span * n_stations).astype(int),
                                 0, n_stations - 1)
            floor_pts = self._points(floor_id) if floor_id is not None else None
            rows = []                       # per station: (points, y_min, y_max, clearance)
            for k in range(n_stations):
                sel = station_of == k
                if int(sel.sum()) < 10:
                    rows.append((int(sel.sum()), None, None, None))
                    continue
                ys = pts[sel][:, 1]
                clearance = None
                if floor_pts is not None and len(floor_pts) >= 30:
                    # clearance = object's underside minus the floor level below it
                    xz = pts[sel][:, [0, 2]].mean(0)
                    near = floor_pts[np.linalg.norm(floor_pts[:, [0, 2]] - xz, axis=1) < max(span / n_stations, 0.5)]
                    if len(near) >= 10:
                        clearance = float(ys.min() - float(np.median(near[:, 1])))
                rows.append((int(sel.sum()), float(ys.min()), float(ys.max()), clearance))
        else:
            floor_t = (self._tp(floor_id)
                       if floor_id is not None and self._points(floor_id) is not None else None)
            axis_i, s_lo, span, rows = self._tt.height_profile(self._tp(id), T, aabb,
                                                               n_stations, floor_t)
        stations = []
        for k, (cnt, y_lo, y_hi, clearance) in enumerate(rows):
            if cnt < 10:
                stations.append({"station": k, "insufficient_data": True})
                continue
            entry = {"station": k,
                     "position_along_m": round(float(s_lo + (k + 0.5) / n_stations * span), 3),
                     "y_min_m": round(float(y_lo), 4),
                     "y_max_m": round(float(y_hi), 4)}
            if clearance is not None:
                entry["clearance_m"] = round(float(clearance), 4)
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
        if self._dev == "numpy":
            from phase_r.depth_regularization import fit_plane
            n, d0 = fit_plane(pts)
            res_mm = np.abs(pts @ n + d0) * 1000.0
            rms, p95, worst = (float(np.sqrt((res_mm ** 2).mean())),
                               float(np.percentile(res_mm, 95)), float(res_mm.max()))
        else:
            st = self._tt.plane_residuals_mm(self._tp(id))
            rms, p95, worst = st["rms"], st["p95"], st["max"]
        return {"source": "tool_measured", "backend": "plane_fit_fallback",
                "rms_mm": round(rms, 2),
                "p95_mm": round(p95, 2),
                "max_mm": round(worst, 2),
                "flatness_pass": bool(p95 <= tolerance_mm),
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

    def _pool_t(self):
        """Every instance's points on the device with their owner, from ONE read of the store
        (float32 as stored, float64 on the device) — the pool of the volume tools, kept for
        the life of this tool set. MEASURED 2026-11-13: pccr 2026-08-31's 18.6 M points took
        17.5 s through 1,360 per-instance reads, 0.2 s in one query (warm cache)."""
        if self._pooled_t is None:
            torch = self._tt._torch()
            ids, clouds = [], []
            for iid, pts in self.store.iter_points():
                if pts is None or not len(pts):
                    continue
                if self._M is not None:                    # a legacy raw-frame store
                    pts = self._to_display(pts)
                ids.append(np.full(len(pts), iid, np.int64))
                clouds.append(pts)
            if clouds:
                P = torch.as_tensor(np.concatenate(clouds), device=self._dev).to(torch.float64)
                owners = torch.as_tensor(np.concatenate(ids), device=self._dev)
            else:
                P = torch.zeros((0, 3), dtype=torch.float64, device=self._dev)
                owners = torch.zeros((0,), dtype=torch.int64, device=self._dev)
            uid, cnt = self._tt.counts_by_owner(owners)
            self._pooled_t = (P, owners, {int(a): int(b) for a, b in zip(uid, cnt)})
        return self._pooled_t

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
        if self._dev == "numpy":
            local = obb_local_coords(pts, T)
            outside = np.maximum(np.abs(local) - half[None, :], 0.0)
            d = np.linalg.norm(outside, axis=1)
            j = int(np.argmin(d))
            n_inside = int((d <= 1e-9).sum())
            d_j, local_j, centroid = float(d[j]), local[j], pts.mean(0)
        else:
            r = self._tt.box_clearance(self._tp(int(id)), T, half)
            j, n_inside, d_j, local_j, centroid = (r["j"], r["n_inside"], r["distance"],
                                                   r["local_j"], r["centroid"])
        # nearest object point and its counterpart on the box surface
        p_obj = pts[j]
        clipped = np.clip(local_j, -half, half)
        p_box = (np.asarray(T)[:3, :3] @ clipped) + np.asarray(T)[:3, 3]
        dist = abs(float(p_obj[1] - p_box[1])) if axis == "vertical" else d_j
        return {"clearance_m": round(dist, 4), "unit": "m", "axis": axis,
                "intersects": bool(n_inside > 0),
                "points_inside": n_inside,
                "point_a_m": [round(float(v), 4) for v in p_box],
                "point_b_m": [round(float(v), 4) for v in p_obj],
                "center_distance_m": round(float(np.linalg.norm(
                    np.asarray(c, float) - centroid)), 4),
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
        voxel_m = max(0.02, float(voxel_m))
        n_vox = np.maximum(1, np.floor(np.abs(s) / voxel_m)).astype(int)
        total_vox = int(np.prod(n_vox))
        occ_vox = 0
        if self._dev == "numpy":
            pts, owners = self._all_points()
            if not len(pts):
                return _insufficient("no reconstructed points to evaluate the volume")
            inside = self._inside_mask(pts, T, half)
            n_in = int(inside.sum())
            # per-object occupancy
            per_owner = [(int(iid), int((owners[inside] == iid).sum()))
                         for iid in (np.unique(owners[inside]) if n_in else [])]
            # voxel occupancy of the box interior
            if n_in:
                local = obb_local_coords(pts[inside], T) + half  # 0..size
                keys = np.floor(local / voxel_m).astype(np.int64)
                keys = np.clip(keys, 0, n_vox - 1)
                occ_vox = len(np.unique(keys, axis=0))
        else:
            P, owners_t, _totals = self._pool_t()
            if not int(P.shape[0]):
                return _insufficient("no reconstructed points to evaluate the volume")
            inside = self._tt.inside_mask(P, T, half)
            n_in = int(inside.sum())
            uid, cnt = self._tt.counts_by_owner(owners_t[inside])
            per_owner = [(int(a_), int(b_)) for a_, b_ in zip(uid, cnt)]
            if n_in:
                occ_vox = self._tt.occupied_voxels(
                    self._tt.voxel_keys(P[inside], T, half, voxel_m, n_vox), n_vox)
        objs = [{"id": iid, "label": self._label_of(iid), "points_inside": cnt}
                for iid, cnt in per_owner]
        objs.sort(key=lambda o: -o["points_inside"])
        occ_frac = min(1.0, occ_vox / total_vox) if n_in else 0.0
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
        if self._dev == "numpy":
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
        else:
            P, owners_t, totals = self._pool_t()
            uid, cnt = self._tt.counts_by_owner(owners_t[self._tt.inside_mask(P, T, half)])
            n_inside = {int(a_): int(b_) for a_, b_ in zip(uid, cnt)}
            for i in self.store.list_instances():          # the record's order
                k = n_inside.get(int(i["instance_id"]))
                if k:
                    out.append({"id": i["instance_id"], "label": i["label"],
                                "fraction_inside": round(k / totals[int(i["instance_id"])], 3),
                                "points_inside": k})
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
        need = np.maximum(1, np.ceil(item / voxel_m).astype(int))
        if self._dev == "numpy":
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
            if np.any(need > n_vox):
                return {"fits": False, "reason": "item larger than the volume",
                        "source": "tool_measured"}
            placement = self._find_empty_box(ii, n_vox, need)
            n_free = int(free.sum())
        else:
            if np.any(need > n_vox):
                return {"fits": False, "reason": "item larger than the volume",
                        "source": "tool_measured"}
            P, _owners, _totals = self._pool_t()
            inside = self._tt.inside_mask(P, T, half)
            keys = self._tt.voxel_keys(P[inside], T, half, voxel_m, n_vox)
            placement, n_free = self._tt.free_placement(keys, n_vox, need)
        if placement is None:
            return {"fits": False, "reason": "no free region large enough",
                    "free_volume_m3": round(float(n_free) * voxel_m ** 3, 4),
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
                "object_labels": {r["label"]: r["count"] for r in self.labels_summary(insts)},
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
    # ── VIEWER CONTROL (USER 2026-10-09: "el chat debe manejar en todo momento la nube … mostrar,
    #    apagar, encender objetos, rotar, viajar de un lugar a otro") ──────────────────────────
    # These tools MEASURE nothing: each returns a ``viewer`` action the viewer performs as the
    # answer arrives (AssistantPanel → App). Object ids are checked against the store.
    _VIEWS = ("top", "front", "back", "left", "right", "bottom", "iso", "home")
    _COLOR_MODES = {"rgb": "rgb", "color": "rgb", "status": "status", "confidence": "status",
                    "votes": "mv_votes", "mv_votes": "mv_votes", "deviation": "deviation"}

    def _known_ids(self, ids) -> tuple[list[int], list]:
        have = set(self._instances())
        ok, bad = [], []
        for x in (ids if isinstance(ids, (list, tuple)) else [ids]):
            try:
                v = int(x)
            except (TypeError, ValueError):
                bad.append(x)
                continue
            (ok if v in have else bad).append(v)
        return ok, bad

    def focus_object(self, id: int) -> dict:
        ok, _ = self._known_ids([id])
        if not ok:
            return _insufficient(f"no object with id {id}")
        return {"viewer": {"action": "focus", "ids": ok}, "label": self._label_of(ok[0]),
                "done": "the object is selected: its OBB and dimensions are shown and the "
                        "camera travels to it"}

    def _ids_or_label(self, ids, label) -> tuple[list[int], list]:
        """The objects a viewer command is about: the ids listed, plus EVERY object of a label
        ("show only the doors" is show_only(label="door") — no list of ids to write)."""
        ok, bad = self._known_ids(ids if ids is not None else [])
        if label:
            for i in self._match_label(label):
                if int(i["instance_id"]) not in ok:
                    ok.append(int(i["instance_id"]))
        return ok, bad

    def show_objects(self, ids: list | None = None, visible: bool = True,
                     label: str | None = None) -> dict:
        ok, bad = self._ids_or_label(ids, label)
        if not ok:
            return _insufficient(f"no known object among {ids if ids else label}")
        return {"viewer": {"action": "visibility", "ids": ok, "visible": bool(visible)},
                "count": len(ok), "labels": sorted({self._label_of(i) for i in ok}),
                "unknown_ids": bad}

    def show_only(self, ids: list | None = None, label: str | None = None) -> dict:
        ok, bad = self._ids_or_label(ids, label)
        if not ok:
            return _insufficient(f"no known object among {ids if ids else label}")
        return {"viewer": {"action": "show_only", "ids": ok},
                "count": len(ok), "labels": sorted({self._label_of(i) for i in ok}),
                "unknown_ids": bad}

    def show_all_objects(self) -> dict:
        return {"viewer": {"action": "show_all"}}

    def set_obb(self, ids: list | None = None, on: bool = True, label: str | None = None) -> dict:
        ok, bad = self._ids_or_label(ids, label)
        if not ok:
            return _insufficient(f"no known object among {ids if ids else label}")
        return {"viewer": {"action": "obb", "ids": ok, "on": bool(on)}, "count": len(ok),
                "unknown_ids": bad}

    def set_view(self, view: str) -> dict:
        v = str(view or "").lower()
        if v not in self._VIEWS:
            return _insufficient(f"unknown view {view!r}; one of {list(self._VIEWS)}")
        return {"viewer": {"action": "view", "view": v}}

    def rotate_view(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> dict:
        return {"viewer": {"action": "orbit", "yaw_deg": float(yaw_deg or 0.0),
                           "pitch_deg": float(pitch_deg or 0.0)}}

    def travel(self, from_id: int, to_id: int) -> dict:
        ok, _ = self._known_ids([from_id, to_id])
        if len(ok) < 2:
            return _insufficient(f"both objects must exist (got {from_id}, {to_id})")
        return {"viewer": {"action": "travel", "ids": ok},
                "labels": [self._label_of(i) for i in ok]}

    def color_cloud(self, mode: str) -> dict:
        m = self._COLOR_MODES.get(str(mode or "").lower())
        if m is None:
            return _insufficient(f"unknown colour mode {mode!r}; one of {sorted(self._COLOR_MODES)}")
        return {"viewer": {"action": "color", "mode": m}}

    def clear_view(self) -> dict:
        return {"viewer": {"action": "clear"}}

    # ── THE METHOD (USER 2026-10-09: "debe saber explicar el método completo del pipeline, cómo
    #    se reconstruyó, qué confianza tienen los puntos") ─────────────────────────────────────
    def explain_reconstruction(self) -> dict:
        """How THIS session was reconstructed, segmented and certified: the pipeline's stages and
        the session's own records (keyframes, chunks, epochs, segmentation rounds, coverage,
        certification, the per-point confidence the cloud carries). Facts are read from the
        session's files (tool_measured); the stage descriptions are the method's own."""
        import json as _json
        out = Path(self.store.path).parent
        sd = out.parent

        def _read(rel):
            q = out / rel
            try:
                return _json.loads(q.read_text()) if q.is_file() else None
            except (OSError, ValueError):
                return None

        facts: dict = {}
        frames = sd / "frames"
        facts["video_frames"] = len(list(frames.glob("*.jpg"))) if frames.is_dir() else None
        covis = None
        for cand in (sd / "intake" / "covis.json",):
            try:
                covis = _json.loads(cand.read_text()) if cand.is_file() else None
            except (OSError, ValueError):
                covis = None
        if isinstance(covis, dict):
            ch = covis.get("chunks") or covis.get("chunk_ranges") or []
            facts["omega_chunks"] = len(ch) if isinstance(ch, list) else None
        ep = _read("geometry_epoch.json")
        if isinstance(ep, dict):
            facts["geometry_epoch"] = {k: ep.get(k) for k in ("epoch", "kind", "parent_epoch")}
        meta = _read("potree/metadata.json")
        if isinstance(meta, dict):
            facts["cloud_points"] = meta.get("points")
            conf = next((a for a in meta.get("attributes", []) if a.get("name") == "confidence"), None)
            if conf:
                facts["point_confidence_range"] = {"min": conf.get("min"), "max": conf.get("max")}
        rounds = _read("seg_rounds.json")
        if isinstance(rounds, dict):
            facts["segmentation_rounds"] = [
                {"round": r.get("round"), "new_prompts": len(r.get("new_prompts") or [])}
                for r in rounds.get("rounds", [])]
        seg = _read("segmentation_view.json")
        if isinstance(seg, dict) and isinstance(seg.get("data"), dict):
            d = seg["data"]
            facts["segmentation"] = {"objects": len(d.get("instances") or []),
                                     "coverage": d.get("coverage"),
                                     "prompts": len(d.get("prompts") or [])}
        acta = _read("certify_acta.json")
        if isinstance(acta, dict):
            facts["certification"] = {k: acta.get(k) for k in
                                      ("verdict", "epoch_final", "stopped_reason", "n_iters",
                                       "deliverable_only") if k in acta}
        cm = _read("precision/cloud_metrics.json")
        if isinstance(cm, dict):
            fl = cm.get("floor") if isinstance(cm.get("floor"), dict) else None
            if fl:
                facts["floor_metrics"] = {k: fl.get(k) for k in
                                          ("thickness_median_m", "thickness_p90_m",
                                           "undulation_m", "tilt_deg") if k in fl}
        method = [
            "1. INTAKE: the video's keyframes are chosen by parallax; DA3 metric depth measures the scale; the walk is measured.",
            "2. RECONSTRUCTION (VGGT-Omega): poses and dense depth per keyframe, in chunks planned by co-visibility (50 % overlap), each chunk metric-locked by DA3 anchors.",
            "3. PRECISION CORE F0–F6: gauge of scale along the walk, camera refinement, pose correction (loop closures judged by held-out pairs), Omega's depth bent to the refined landmarks — the cloud is re-projected from it.",
            "4. CLOUD: confidence gate per chunk, voxel + outlier filtering, witness filter (points seen consistently by several keyframes), consolidation; every point keeps its keyframe, pixel and confidence.",
            "5. SEGMENTATION IN ROUNDS: the VLM names the objects on every keyframe; SAM3 segments and tracks each name over all keyframes; the VLM looks again with what is segmented painted grey, until nothing new is named. Masks are projected on the cloud (each point gets its object).",
            "6. CERTIFICATION: closures (objects seen twice) correct depth per chunk, the floor is aligned, the mask filter removes points inside other objects' masks; a final epoch is published.",
            "7. DESCRIPTIONS: the VLM describes each object (vlm_proposed); every figure comes from tools over the geometry (tool_measured).",
            "Point confidence: Omega's own per-pixel confidence, normalised per chunk; points under the 10 % floor never enter the cloud; colour the cloud by 'confidence' (status) to see it.",
        ]
        return {"method": method, "session": facts, "source": "tool_measured"}

    def impls(self, remote=None) -> dict:
        """Tool implementations. Every returned measurement is stamped with
        the session's geometry epoch + human-directed-corrections count
        (USER 2026-09-08: a supervision measurement never hides that the
        geometry was corrected by a person).

        ``remote(tool, kwargs)``: when given, the tools that read points (POINT_TOOLS) are
        run by it — the chat's tools process on the card (tools_worker) — and this tool set
        only looks things up in the store."""
        from pathlib import Path as _Path
        from correction.epoch import stamp as _stamp_epoch
        _out_dir = _Path(self.store.path).parent

        def _wrap(fn):
            def _inner(*a, **k):
                res = fn(*a, **k)
                if isinstance(res, dict):
                    _stamp_epoch(res, _out_dir)
                return res
            return _inner

        raw = self._impls_raw()
        if remote is not None:
            for name in POINT_TOOLS:
                raw[name] = (lambda _n: (lambda **kw: remote(_n, kw)))(name)

        def _with_labels(fn):
            # "the floor", "the largest wall": the model may pass a LABEL where an id is asked —
            # the tool layer takes that label's largest object and says which one it took.
            # Every object a call is about is named in its result (`about`): the answer is
            # written from it (phase5_qa/answers.py), never from the model's wording
            def _inner(*a, **k):
                took, named = {}, {}
                for key in ID_ARGS:
                    if key in k and k[key] is not None:
                        k[key], used = self.resolve_id(k[key])
                        if used:
                            took[key] = used
                        inst = self._instance(k[key])
                        if inst is not None:
                            named[key] = {"id": int(inst["instance_id"]), "label": inst["label"]}
                res = fn(*a, **k)
                if isinstance(res, dict):
                    if named:
                        res["about"] = named
                    if took:
                        res["objects_taken_by_label"] = took
                return res
            return _inner

        return {name: _wrap(_with_labels(fn)) for name, fn in raw.items()}

    def resolve_id(self, value):
        """(id, None) for an id — an integer or its digits — and (the id of the label's LARGEST
        object, {"label", "id"}) for a label; the value itself when nothing matches (the tool
        then says the object does not exist)."""
        if isinstance(value, bool) or value is None:
            return value, None
        if isinstance(value, (int, float)):
            return int(value), None
        text = str(value).strip()
        m = re.search(r"#\s*(\d+)\s*$", text)          # "sign#797", "#797": the id is the number
        if m:
            return int(m.group(1)), None
        if text.isdigit():
            return int(text), None
        hit = self._match_label(text)
        if not hit:
            return value, None
        n_pts = self.store.list_point_counts()
        best = min(hit, key=lambda i: (-n_pts.get(int(i["instance_id"]), 0), int(i["instance_id"])))
        return int(best["instance_id"]), {"label": best["label"], "id": int(best["instance_id"])}

    def _impls_raw(self) -> dict:
        return {
            "list_objects": self.list_objects, "count_objects": self.count_objects,
            "get_object": self.get_object,
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
            "focus_object": self.focus_object, "show_objects": self.show_objects,
            "show_only": self.show_only, "show_all_objects": self.show_all_objects,
            "set_obb": self.set_obb, "set_view": self.set_view, "rotate_view": self.rotate_view,
            "travel": self.travel, "color_cloud": self.color_cloud, "clear_view": self.clear_view,
            "explain_reconstruction": self.explain_reconstruction,
        }

    @staticmethod
    def openai_schemas() -> list[dict]:
        """The scene's tools as the engine reads them — THE DATABASE OF COMMANDS' rows that run
        on the scene (server/commands/catalog/*.yaml: a tool's description and arguments are
        declared there, once), the same for every scan."""
        import commands
        return commands.database().tool_schemas(runs_on="scene")
