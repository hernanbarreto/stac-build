"""WHERE A WORK STANDS ON THE GLOBE (USER 2026-11-24: "vamos a agregarle además la ubicación … gps de cada
proyecto, necesita el punto gps real y la orientación verdad?, en el panel izquierdo debe haber una opción con icono
de mundo, y lo mismo cada sesión debe poder agregarle la información del pin gps … guardamos y ya queda indicado que
ese proyecto tiene asignado el punto gps" / "cesium siempre sin cuenta").

ONE GEOREFERENCE PER PROJECT, ``<project>/georef.json`` (it travels with the session's ZIP, it goes with a rename or
a deletion of the project):

  lat, lon      where the scan's ORIGIN stands: WGS84 degrees
  height_m      the origin's ellipsoidal height, metres (a pin placed by hand: the terrain's height there)
  heading_deg   the plan's orientation: the azimuth, clockwise from true north, of the plan's "up" — the scan's
                frame written Z-up (x, y, z)_display -> (x, -z, y), as every export of STAC writes it: its +Y.
                0 = that +Y points north and +X east. A point (X, Y) of the plan stands, east and north of the
                origin, at  e = X cos h + Y sin h,  n = -X sin h + Y cos h
  scan          the scan whose frame the heading is of (the project's reference scan when it was set)
  source        manual (a click on the imagery) | video (the GPS the video recorded) | points (points of known
                coordinates)
  accuracy_m    as its source states it — None when no source states one (a click: as good as the imagery and the
                hand; never presented as measured)
  set_by, set_at  who, when (UTC)

provenance: ``human_stated`` for a click, ``device_measured`` for a video's GPS, ``tool_measured`` for a fit to
points — nothing of it is the VLM's.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

FILE = "georef.json"
SOURCES = {"manual": "human_stated", "video": "device_measured", "points": "tool_measured"}


class GeoError(ValueError):
    pass


def _number(body: Dict[str, Any], key: str, lo: float, hi: float, *, required: bool = True) -> Optional[float]:
    v = body.get(key)
    if v is None and not required:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise GeoError(f"{key}: a number is needed, got {v!r}") from None
    if not math.isfinite(x) or not lo <= x <= hi:
        raise GeoError(f"{key}: {x} is outside {lo} … {hi}")
    return x


def validate(body: Dict[str, Any], *, set_by: str, scan: Optional[str]) -> Dict[str, Any]:
    """The georeference to store, from what the interface sent; GeoError names what is wrong."""
    source = str(body.get("source") or "")
    if source not in SOURCES:
        raise GeoError(f"source: one of {sorted(SOURCES)}, got {source!r}")
    heading = _number(body, "heading_deg", -360.0, 360.0, required=False)
    note = " ".join(str(body.get("note") or "").split())[:200]
    return {
        "version": 1,
        "lat": _number(body, "lat", -90.0, 90.0),
        "lon": _number(body, "lon", -180.0, 180.0),
        "height_m": _number(body, "height_m", -500.0, 9000.0),
        "heading_deg": round((heading or 0.0) % 360.0, 6),
        "scan": scan,
        "source": source,
        "provenance": SOURCES[source],
        "accuracy_m": _number(body, "accuracy_m", 0.0, 100000.0, required=False),
        "set_by": set_by,
        "set_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }


def load(project_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(project_dir) / FILE
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GeoError(f"{p}: unreadable ({e})") from e
    return d if isinstance(d, dict) else None


def save(project_dir: Path, georef: Dict[str, Any]) -> Dict[str, Any]:
    """Written whole, then put in place (a reader never sees half of it)."""
    p = Path(project_dir) / FILE
    tmp = p.with_name(f".{FILE}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(georef, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return georef


def remove(project_dir: Path) -> bool:
    p = Path(project_dir) / FILE
    if p.is_file():
        p.unlink()
        return True
    return False


def plan_to_enu(x: float, y: float, heading_deg: float) -> tuple[float, float]:
    """A point of the plan (Z-up frame: X, Y) -> (east, north) of the origin, metres."""
    h = math.radians(heading_deg)
    return x * math.cos(h) + y * math.sin(h), -x * math.sin(h) + y * math.cos(h)


# ── what a video recorded of where it was taken ────────────────────────────────────────────────────────────────────
# ISO 6709 as the phones write it in the container ("+40.4168-003.7038/", "+40.4168-003.7038+650.000/"): the tags
# Android writes ("location") and Apple's (com.apple.quicktime.location.ISO6709, with its stated accuracy). Measured
# 2026-11-24: none of the six videos of the project carries one (grupo_electrogeno: a Samsung, location off).
_ISO6709 = re.compile(r"^\s*([+-]\d{1,2}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?\s*/?\s*$")
_LOCATION_TAGS = ("com.apple.quicktime.location.iso6709", "location", "location-eng", "com.android.location")


def parse_iso6709(text: str) -> Optional[Dict[str, float]]:
    m = _ISO6709.match(str(text or ""))
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    out = {"lat": lat, "lon": lon}
    if m.group(3) is not None:
        out["altitude_m"] = float(m.group(3))
    return out


def video_location(video: Path) -> Optional[Dict[str, Any]]:
    """The place the video's own metadata records, or None: {lat, lon, altitude_m?, accuracy_m?, tag}."""
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
                            str(video)], capture_output=True, text=True, timeout=30)
        info = json.loads(r.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    tags: Dict[str, str] = {}
    for block in [info.get("format") or {}] + list(info.get("streams") or []):
        for k, v in (block.get("tags") or {}).items():
            tags.setdefault(str(k).lower(), str(v))
    for key in _LOCATION_TAGS:
        place = parse_iso6709(tags.get(key, ""))
        if place:
            acc = tags.get("com.apple.quicktime.location.accuracy.horizontal")
            try:
                place["accuracy_m"] = float(acc) if acc is not None else None
            except ValueError:
                place["accuracy_m"] = None
            place["tag"] = key
            return place
    return None


# ── the plan: the scan seen from above, as a satellite sees it ─────────────────────────────────────────────────────
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def _ply_points(path: Path):
    """A binary little-endian PLY's vertices, mapped (not read into memory)."""
    import numpy as np
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise GeoError(f"{path}: no end_header")
            head += line
    lines = head.decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise GeoError(f"{path}: not a binary little-endian PLY")
    count, fields, in_vertex = 0, [], False
    for ln in lines:
        p = ln.split()
        if p[:2] == ["element", "vertex"]:
            count, in_vertex = int(p[2]), True
        elif p[:1] == ["element"]:
            in_vertex = False
        elif p[:1] == ["property"] and in_vertex:
            if p[1] == "list" or p[1] not in _PLY_TYPES:
                raise GeoError(f"{path}: vertex property {' '.join(p[1:])} is not read here")
            fields.append((p[2], _PLY_TYPES[p[1]]))
    return np.memmap(path, dtype=np.dtype(fields), mode="r", offset=len(head), shape=(count,))


def plan_raster(output_dir: Path, *, max_points: int = 8_000_000, max_side: int = 2048,
                min_cell_m: float = 0.05) -> Dict[str, Any]:
    """The scan's published cloud from above, in the frame the georeference speaks of (the display frame — its
    floor adjustment applied — written Z-up: X = x, Y = -z): each cell the colour of its HIGHEST point (a roof, a
    canopy: what the imagery shows), transparent where no point is. Returns the PNG and where it lies:
    {png, cell_m, x_min, x_max, y_min, y_max, width, height, points_used, points}. At most ``max_points`` points,
    every k-th (the same ones at every call)."""
    import io
    import numpy as np
    from PIL import Image
    cloud = Path(output_dir) / "cleaned_cloud.ply"
    if not cloud.is_file():
        raise GeoError(f"{cloud}: no published cloud")
    v = _ply_points(cloud)
    n = len(v)
    step = max(1, -(-n // max_points))
    pts = np.stack([np.asarray(v["x"][::step], np.float64), np.asarray(v["y"][::step], np.float64),
                    np.asarray(v["z"][::step], np.float64)], axis=1)
    rgb = np.stack([np.asarray(v[c][::step]) for c in ("red", "green", "blue")], axis=1)
    ft = Path(output_dir) / "floor_transform.npz"
    if ft.is_file():
        d = np.load(ft)
        pts = float(d["s"]) * pts @ np.asarray(d["R"], np.float64).T + np.asarray(d["t"], np.float64)
    X, Y, H = pts[:, 0], -pts[:, 2], pts[:, 1]
    x0, x1, y0, y1 = float(X.min()), float(X.max()), float(Y.min()), float(Y.max())
    cell = max(min_cell_m, max(x1 - x0, y1 - y0) / max_side)
    w, h = int((x1 - x0) // cell) + 1, int((y1 - y0) // cell) + 1
    col = ((X - x0) // cell).astype(np.int64)
    row = ((y1 - Y) // cell).astype(np.int64)           # row 0 = the plan's top (+Y)
    pix = row * w + col
    order = np.lexsort((H, pix))                          # by cell, then height: each cell's last is its highest
    last = order[np.r_[np.flatnonzero(np.diff(pix[order])), len(order) - 1]]
    img = np.zeros((h * w, 4), np.uint8)
    img[pix[last], :3] = rgb[last]
    img[pix[last], 3] = 255
    buf = io.BytesIO()
    Image.fromarray(img.reshape(h, w, 4), "RGBA").save(buf, format="PNG", optimize=False)
    return {"png": buf.getvalue(), "cell_m": cell, "x_min": x0, "x_max": x0 + w * cell, "y_min": y1 - h * cell,
            "y_max": y1, "width": w, "height": h, "points_used": int(len(X)), "points": int(n)}
