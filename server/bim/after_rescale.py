"""AFTER A CALIBRATION'S RESCALE, WHAT THE SCAN'S NEW SCALE LEFT STALE IS RENEWED (USER 2026-11-28: "cuando se
reescala, debe borrar el genrecon, el ifc y las comparaciones que hayan" → "cual es el paso correcto para el genrecon el
ifc y la comparacion" → "perfecto, dale").

The rescale (correction/rescale.py) scales the cloud, the cameras, the segments, the boxes and the GenRecon mesh
exactly — the mesh by a node at its root, which the IFC generator now reads (ifc_from_mesh.mesh_placements). What it
does not carry, and is renewed here, in the rescale's own process, once its epoch is in place:
  · THE IFCs WRITTEN FROM THIS SCAN (told by the name inside their header, bim/scan_data.stac_source) are written
    again from the scan as it now is, under the same name, with what they were made with: the share of the mesh kept
    (their header's "Triangles kept: A of B"), the texture when it lay beside them, the cut's neighbours and large
    faces of config.yaml export.ifc (his values). An IFC that came from outside — a design model — is left as it is:
    it is in real metres. Their viewer tiles follow by themselves (bim/ifc_tiles.py: a new version of the file).
  · THE STORED BIM COMPARISON of this scan (the project's bim_comparison/, the scan its meta names) is removed: it
    measured the old scale; the next comparison makes it again.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_FILE_NAME = re.compile(rb"FILE_NAME\('((?:[^']|'')*)'")
_KEPT = re.compile(rb"Triangles kept: ([\d,]+) of ([\d,]+)")
_STAC_IFC = re.compile(r"^(?P<session>.+)_(?P<day>\d{4}-\d{2}-\d{2})_(?P<source>.+)_from_(?P<kind>mesh|scan)\.ifc$")
HEAD_BYTES = 1 << 16


def stac_ifc(path: Path) -> Optional[Dict[str, Any]]:
    """What STAC wrote this IFC from — {session, scan, kind: mesh | scan, keep} — read from its header alone (an IFC
    can be hundreds of MB: it is never opened whole here); None for an IFC STAC did not write from a scan."""
    with open(path, "rb") as f:
        head = f.read(HEAD_BYTES)
    m = _FILE_NAME.search(head)
    if not m:
        return None
    s = _STAC_IFC.match(m.group(1).decode("utf-8", "replace").replace("''", "'"))
    if not s:
        return None
    keep = 1.0
    k = _KEPT.search(head)
    if k:
        kept, total = (int(x.replace(b",", b"")) for x in k.groups())
        keep = 1.0 if total <= 0 or kept >= total else round(kept / total, 4)
    return {"session": s.group("session"), "scan": f"{s.group('day')}/{s.group('source')}", "kind": s.group("kind"),
            "keep": keep}


def remove_sheet(bim_dir: Path) -> bool:
    """The stored comparison removed — its meta first (without it there is no sheet: comparison.stored_sheet), then its
    files and octree. False when there was none."""
    import shutil
    from bim.comparison import NEW_SHEET_DIR, SHEET_FILES, SHEET_META, SHEET_OCTREE
    bim_dir = Path(bim_dir)
    meta = bim_dir / SHEET_META
    if not meta.is_file():
        return False
    meta.unlink()
    for name in SHEET_FILES:
        (bim_dir / name).unlink(missing_ok=True)
    shutil.rmtree(bim_dir / SHEET_OCTREE, ignore_errors=True)
    shutil.rmtree(bim_dir / NEW_SHEET_DIR, ignore_errors=True)
    return True


def renew(session: str, scan_key: str, ifcs_dir: Path, bim_dir: Path,
          log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """The scan's stale products after its rescale: its IFCs written again, its stored comparison removed. Returns what
    was done; an IFC that could not be written again is listed with its error (the rescale itself stands)."""
    from bim.comparison import stored_sheet
    out: Dict[str, Any] = {"ifcs": [], "errors": [], "comparison_removed": False}
    meta = stored_sheet(bim_dir)
    if meta and meta.get("scan") == scan_key:
        out["comparison_removed"] = remove_sheet(bim_dir)
        log(f"[rescale] the stored BIM comparison of {scan_key} measured the old scale: removed (compare again)")
    for path in sorted(Path(ifcs_dir).glob("*.ifc")):
        src = stac_ifc(path)
        if not src or src["session"] != session or src["scan"] != scan_key:
            continue
        log(f"[rescale] {path.name}: written again from the scan at its new scale (keep {src['keep']})")
        try:
            if src["kind"] == "mesh":
                from bim import ifc_from_mesh
                from reconstruction.grid_knn import config_section
                how = config_section(("export", "ifc"), ("neighbours", "max_edge_voxels"))
                rep = ifc_from_mesh.write(session, scan_key, keep=src["keep"], neighbours=int(how["neighbours"]),
                                          max_edge_voxels=float(how["max_edge_voxels"]), out=path,
                                          with_textures=(path.parent / f"{path.stem}_textures").is_dir(),
                                          log=lambda m: log(f"  {m}"))
                out["ifcs"].append({"name": path.name, "elements": sum(rep.get("by_class", {}).values()),
                                    "faces": rep.get("faces"), "bytes": rep.get("bytes")})
            else:
                from bim import ifc_from_scan
                rep = ifc_from_scan.write(session, scan_key, out=path)
                out["ifcs"].append({"name": path.name, "elements": rep.get("elements") if isinstance(rep, dict) else None})
        except Exception as e:  # noqa: BLE001 — listed in the result and in the log, never hidden
            out["errors"].append({"name": path.name, "error": f"{type(e).__name__}: {e}"})
            log(f"[rescale] ⚠ {path.name} could NOT be written again: {type(e).__name__}: {e}")
            continue
        try:
            from bim import ifc_tiles
            ifc_tiles.ensure(Path(ifcs_dir), path.name)        # its viewer tiles, now
        except Exception as e:  # noqa: BLE001
            log(f"[rescale] ⚠ {path.name}: its tiles were not started ({e}) — the viewer starts them when it asks")
    return out
