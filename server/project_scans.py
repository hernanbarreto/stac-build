"""Multi-scan projects — scans list + composition state (USER DESIGN 2026-09-06).

A project is ONE place holding N scans from different days
(`projects/<slug>/scans/<date>/src_<source>/`). Each scan keeps its own
artifacts untouched (cloud, Potree, segmentation, floor transform, approved
chunk corrections). The PROJECT adds:

  * the scan list with labels,
  * the composition: which scan is the REFERENCE (chosen by the user) and,
    for every other scan, a 4x4 transform expressed in the reference frame
    (identity when never composed; a similarity — scale + rigid — once
    registered on invariant objects, the scale split symmetrically between
    both scans; or a manual gizmo placement).

Everything lives in project.json under "scans" / "composition". Changing the
reference re-expresses the stored transforms (no data loss).

See docs/design/multi_scan_project.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from project_paths import ProjectPaths

IDENTITY = [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def scan_key(date: str, source: str) -> str:
    return f"{date}/{source}"


def split_key(key: str):
    parts = key.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "default")


def discover_scans(paths: ProjectPaths) -> List[Dict]:
    """Every scan present on disk, oldest first."""
    out = []
    for date in paths.list_scan_days():
        for source in paths.list_sources(date):
            out.append({"key": scan_key(date, source), "date": date,
                        "source": source})
    return out


def load_meta(paths: ProjectPaths) -> dict:
    meta = paths.load_project_meta() or {}
    meta.setdefault("scans", [])
    comp = meta.setdefault("composition", {})
    comp.setdefault("reference", None)
    comp.setdefault("transforms", {})
    return meta


def sync_scans(paths: ProjectPaths) -> dict:
    """Make project.json's scan list match the disk (adds new scans with a
    default label, keeps existing labels, drops entries whose folder is
    gone). Picks a reference if none is set: the OLDEST scan (the user
    changes it at will)."""
    meta = load_meta(paths)
    on_disk = discover_scans(paths)
    known = {s["key"]: s for s in meta["scans"]}
    merged = []
    for s in on_disk:
        entry = known.get(s["key"], {"key": s["key"],
                                     "label": f"scan {s['date']}",
                                     "added_at": time.strftime(
                                         "%Y-%m-%dT%H:%M:%S")})
        entry.update({"date": s["date"], "source": s["source"]})
        merged.append(entry)
    meta["scans"] = merged
    keys = {s["key"] for s in merged}
    comp = meta["composition"]
    if comp.get("reference") not in keys:
        comp["reference"] = merged[0]["key"] if merged else None
    comp["transforms"] = {k: v for k, v in comp["transforms"].items()
                          if k in keys}
    paths.save_project_meta(meta)
    return meta


def get_reference(paths: ProjectPaths) -> Optional[str]:
    return load_meta(paths)["composition"].get("reference")


def get_transform(paths: ProjectPaths, key: str) -> np.ndarray:
    """4x4 composition transform of a scan in the reference frame
    (identity for the reference itself and for never-composed scans)."""
    comp = load_meta(paths)["composition"]
    if key == comp.get("reference"):
        return np.eye(4)
    t = comp["transforms"].get(key)
    if not t or not t.get("matrix"):
        return np.eye(4)
    return np.array(t["matrix"], dtype=np.float64).reshape(4, 4)


def set_transform(paths: ProjectPaths, key: str, matrix: np.ndarray,
                  method: str, extra: Optional[dict] = None) -> dict:
    """Store a scan's composition transform (row-major 4x4)."""
    meta = load_meta(paths)
    if key == meta["composition"].get("reference"):
        raise ValueError("the reference scan cannot be transformed — pick "
                         "another reference first")
    entry = {"matrix": [float(x) for x in np.asarray(matrix).reshape(-1)],
             "method": method,
             "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        entry.update(extra)
    meta["composition"]["transforms"][key] = entry
    paths.save_project_meta(meta)
    return entry


def set_reference(paths: ProjectPaths, new_ref: str) -> dict:
    """Change the reference scan. Every stored transform T_k (scan k → old
    reference frame) is re-expressed in the new reference's frame:
    T_k' = T_new⁻¹ · T_k, and the old reference gets T_old' = T_new⁻¹."""
    meta = load_meta(paths)
    keys = {s["key"] for s in meta["scans"]}
    if new_ref not in keys:
        raise ValueError(f"unknown scan {new_ref}")
    old_ref = meta["composition"].get("reference")
    if old_ref == new_ref:
        return meta
    T_new = get_transform(paths, new_ref)
    inv = np.linalg.inv(T_new)
    new_transforms = {}
    for key in keys:
        if key == new_ref:
            continue
        T_k = get_transform(paths, key)      # identity for old_ref
        Tp = inv @ T_k
        prev = meta["composition"]["transforms"].get(key, {})
        new_transforms[key] = {
            **prev,
            "matrix": [float(x) for x in Tp.reshape(-1)],
            "method": prev.get("method", "reexpressed"),
            "reexpressed_from": old_ref,
        }
    meta["composition"]["reference"] = new_ref
    meta["composition"]["transforms"] = new_transforms
    paths.save_project_meta(meta)
    return meta


def scan_potree_url(project: str, key: str) -> str:
    date, source = split_key(key)
    return f"/potree_scan/{project}/{date}/{source}/"
