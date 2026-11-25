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
