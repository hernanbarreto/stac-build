"""THE SCAN'S SIDE OF A BIM COMPARISON — its cloud, its segmented objects, and the key each object is
matched to an IFC element by.

The comparison (bim/comparison.py, bim/registration.py) was written in March for sessions whose
segments were LABELLED by hand with the number of the BIM element they are ("292127" ↔ the IFC
element named "…:292127"). A scan of today is segmented by concept: its objects are named "floor",
"wall", "rack" — many with the same label — and each has an INSTANCE ID. An IFC that STAC wrote
from such a scan (bim/ifc_from_mesh.py, bim/ifc_from_scan.py) names its elements
``<label>:<label>:<instance id>``: the id is the key. Both ways are read here, told apart by the IFC:

    an IFC STAC wrote from a scan   the element's number is an INSTANCE ID of that segmentation
    any other IFC                   the element's number is matched to a segment LABELLED with it
                                    (several objects with that label are one element's points)

The scan's data is read the way the rest of the app reads it: the cloud as a memory map of
``cleaned_cloud.ply``, the list of objects from the instance store (USER 2026-11-13: "la lista debe
ser tomada de la base de datos"), each object's points from the per-point ids (``instance_ids.npy``).
A session with no store or no per-point ids is read from ``segmentation_result.json`` /
``segmentation.json`` (each instance's own index list), as in March.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_STAC_IFC = re.compile(r"^(?P<session>.+)_(?P<day>\d{4}-\d{2}-\d{2})_(?P<source>.+)_from_(?:mesh|scan)\.ifc$")


def stac_source(ifc_file) -> Optional[Tuple[str, str]]:
    """(session, scan key) of the scan an IFC was written from by STAC — None for any other IFC.
    Read from the name the file carries INSIDE (its header): a renamed file keeps it."""
    try:
        name = str(ifc_file.header.file_name.name or "")
    except Exception:  # noqa: BLE001 — an IFC with no readable header is simply not one of ours
        return None
    m = _STAC_IFC.match(name)
    return (m.group("session"), f"{m.group('day')}/{m.group('source')}") if m else None


def element_keys(ifc_file) -> Dict[str, Dict[str, Any]]:
    """{key: {element_key, ifc_type, ifc_name, ifc_id}} of the IFC's products whose Name ends in a
    number after its last colon (Revit's element id; a STAC IFC's instance id) — the matching key."""
    out: Dict[str, Dict[str, Any]] = {}
    for el in ifc_file.by_type("IfcProduct"):
        name = getattr(el, "Name", None)
        if not name:
            continue
        name = str(name)
        key = name.split(":")[-1].strip()
        if key.isdigit():
            out[key] = {"element_key": key, "ifc_type": el.is_a(), "ifc_name": name, "ifc_id": el.id()}
    return out


class ScanError(ValueError):
    """The scan cannot be compared — the message says what is missing."""


class Scan:
    """One scan's cloud and segmented objects (``output_dir`` = the scan's output directory)."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self._xyz: Optional[np.ndarray] = None
        self._order: Optional[np.ndarray] = None                  # the points sorted by instance id …
        self._bounds: Dict[int, Tuple[int, int]] = {}             # … and each id's slice of that order
        self._lists: Dict[int, np.ndarray] = {}                   # (a session read from the JSON)
        self.instances: Dict[int, Dict[str, Any]] = {}            # id -> {id, label, points}
        self.source = ""
        self._read_instances()

    # ── the objects ──────────────────────────────────────────────────────────────────────────
    def _read_instances(self) -> None:
        ids_path = self.output_dir / "instance_ids.npy"
        store = self.output_dir / "scene_r.db"
        if ids_path.exists() and store.exists():
            from phase_r.instance_store import read_instances
            listed = read_instances(store) or {}
            ids = np.load(ids_path, mmap_mode="r")
            order = np.argsort(ids, kind="stable")
            sorted_ids = np.asarray(ids)[order]
            values, starts = np.unique(sorted_ids, return_index=True)
            ends = np.append(starts[1:], len(sorted_ids))
            self._order = order
            for value, a, b in zip(values.tolist(), starts.tolist(), ends.tolist()):
                if value and value in listed:                     # (0 = unsegmented)
                    self._bounds[value] = (a, b)
                    self.instances[value] = {"id": value, "label": str(listed[value]["label"]), "points": b - a}
            self.points_total = int(len(sorted_ids))
            self.source = "the instance store and the per-point ids"
            return
        for name in ("segmentation_result.json", "segmentation.json"):
            path = self.output_dir / name
            if path.exists():
                data = json.loads(path.read_text())
                for n, inst in enumerate(data.get("instances", [])):
                    iid = int(inst.get("id", inst.get("instance_id", n + 1)))
                    idx = np.asarray(inst.get("globalIndices", inst.get("point_indices", [])), dtype=np.int64)
                    self._lists[iid] = idx
                    self.instances[iid] = {"id": iid, "label": str(inst.get("label", "")), "points": int(len(idx))}
                self.points_total = int(data.get("total_points", 0))
                self.source = name
                return
        raise ScanError(f"{self.output_dir}: no segmentation (neither the instance store with instance_ids.npy "
                        f"nor segmentation_result.json)")

    def indices(self, instance_id: int) -> np.ndarray:
        """The cloud indices of one object's points, ascending."""
        if instance_id in self._bounds:
            a, b = self._bounds[instance_id]
            return np.sort(np.asarray(self._order[a:b], dtype=np.int64))
        return self._lists.get(instance_id, np.empty(0, dtype=np.int64))

    # ── the cloud ────────────────────────────────────────────────────────────────────────────
    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) float64 — the cloud as it is on disk (display frame on a scan whose orientation is baked)."""
        if self._xyz is None:
            from exporter.cloud import open_ply
            path = self.output_dir / "cleaned_cloud.ply"
            if not path.exists():
                raise ScanError(f"{path}: the scan has no cloud")
            cloud = open_ply(path)
            self._xyz = np.stack([np.asarray(cloud[a], dtype=np.float64) for a in ("x", "y", "z")], axis=1)
            if self.points_total and self.points_total != len(self._xyz) and self._order is not None:
                raise ScanError(f"{self.output_dir}: the cloud has {len(self._xyz)} points and the per-point ids "
                                f"{self.points_total} — the segmentation is of another cloud")
        return self._xyz

    # ── the matching ─────────────────────────────────────────────────────────────────────────
    def match(self, elements: Dict[str, Dict[str, Any]], by_instance_id: bool) -> List[Dict[str, Any]]:
        """The scan's objects paired with the IFC's elements: one match per element that has a
        counterpart — {segment_label, segment_ids, element_key, ifc_type, ifc_name, ifc_id}."""
        out: List[Dict[str, Any]] = []
        if by_instance_id:
            for iid in sorted(self.instances):
                el = elements.get(str(iid))
                if el is not None:
                    out.append({"segment_label": self.instances[iid]["label"], "segment_ids": [iid], **el})
            return out
        by_label: Dict[str, List[int]] = {}
        for iid in sorted(self.instances):
            by_label.setdefault(self.instances[iid]["label"], []).append(iid)
        for label in sorted(by_label):
            el = elements.get(label)
            if el is not None:
                out.append({"segment_label": label, "segment_ids": by_label[label], **el})
        return out

    def indices_of_match(self, match: Dict[str, Any]) -> np.ndarray:
        """The points a match names: its ``segment_ids`` (what match() gives), else — a match written
        by hand with a label only — every object labelled ``segment_label``."""
        ids = match.get("segment_ids")
        if ids is None:
            label = str(match.get("segment_label", ""))
            ids = [iid for iid in sorted(self.instances) if self.instances[iid]["label"] == label]
        parts = [self.indices(int(i)) for i in ids if int(i) in self.instances]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return parts[0] if len(parts) == 1 else np.unique(np.concatenate(parts))


def first_ifc(ifcs_dir: Path, session_dir: Optional[Path] = None) -> Optional[Path]:
    """The IFC a comparison reads: the first of the project's ifcs/ by name (then, a session of the
    old layout, the first beside it)."""
    for d in (ifcs_dir, session_dir):
        if d is not None and Path(d).is_dir():
            found = sorted(Path(d).glob("*.ifc"))
            if found:
                return found[0]
    return None
