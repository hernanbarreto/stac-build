"""THE GLOBE'S ROUTES (geo.py: one georeference per project — USER 2026-11-24).

    GET    /api/geo                    every project: its name, its scans, its reference scan, its georeference (or None)
    GET    /api/geo/{session_id}       the project's georeference (or None)
    PUT    /api/geo/{session_id}       {lat, lon, height_m, heading_deg?, source, accuracy_m?, note?} — signed in
    DELETE /api/geo/{session_id}       signed in

Anyone who can open the session reads it; setting or removing it needs a signed-in user (it says who placed the
work). Wired by main.py through :func:`configure` (the session resolver, the projects' folder and the audit log are
the server's; this module never imports main).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import geo as G

router = APIRouter(prefix="/api/geo", tags=["geo"])
_security = HTTPBearer(auto_error=False)
_resolve_ctx: Optional[Callable] = None
_projects_dir: Optional[Callable[[], Path]] = None
_audit_log: Optional[Callable] = None


def configure(resolve_ctx: Callable, projects_dir: Callable[[], Path], audit_log: Optional[Callable] = None) -> None:
    global _resolve_ctx, _projects_dir, _audit_log
    _resolve_ctx, _projects_dir, _audit_log = resolve_ctx, projects_dir, audit_log


def _user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)) -> Optional[Dict[str, Any]]:
    if not credentials:
        return None
    from auth import decode_token
    return decode_token(credentials.credentials)


def _name(user: Optional[Dict[str, Any]]) -> str:
    if user is None:
        raise HTTPException(401, "Sign in to place a work on the globe: the place records who set it")
    return str(user.get("username") or user.get("sub") or "unknown")


def _project_dir(session_id: str) -> Path:
    if _resolve_ctx is None or _projects_dir is None:
        raise HTTPException(500, "geo router not configured")
    d = Path(_projects_dir()) / session_id
    if "/" in session_id or "\\" in session_id or session_id in (".", "..") or not (d / "project.json").is_file():
        raise HTTPException(404, f"project {session_id} does not exist")
    return d


def _project_row(d: Path) -> Dict[str, Any]:
    try:
        meta = json.loads((d / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    scans: List[str] = []
    scans_dir = d / "scans"
    if scans_dir.is_dir():
        for day in sorted(p for p in scans_dir.iterdir() if p.is_dir()):
            for src in sorted(p for p in day.iterdir() if p.is_dir() and p.name.startswith("src_")):
                scans.append(f"{day.name}/{src.name[4:]}")
    reference = (meta.get("composition") or {}).get("reference") or (scans[0] if scans else None)
    try:
        georef = G.load(d)
    except G.GeoError:
        georef = None
    # its name is the session's, the folder's, as everywhere in the app (a project.json's own "name" can be an old
    # one: zaragoza's says "test5")
    ifc_dir = d / "ifcs"
    ifcs = sorted(p.name for p in ifc_dir.glob("*.ifc")) if ifc_dir.is_dir() else []
    return {"session_id": d.name, "name": d.name, "scans": scans, "reference": reference, "georef": georef,
            "ifcs": ifcs}


@router.get("")
def geo_projects():
    if _projects_dir is None:
        raise HTTPException(500, "geo router not configured")
    root = Path(_projects_dir())
    rows = [_project_row(d) for d in sorted(root.iterdir()) if (d / "project.json").is_file()] if root.is_dir() else []
    return {"ok": True, "projects": rows}


@router.get("/{session_id}")
def geo_get(session_id: str):
    return {"ok": True, "georef": G.load(_project_dir(session_id))}


@router.put("/{session_id}")
def geo_put(session_id: str, body: Dict[str, Any], user: Optional[Dict[str, Any]] = Depends(_user)):
    who = _name(user)
    d = _project_dir(session_id)
    scan = body.get("scan") or _project_row(d)["reference"]
    try:
        georef = G.save(d, G.validate(body, set_by=who, scan=scan))
    except G.GeoError as e:
        raise HTTPException(400, str(e)) from e
    if _audit_log is not None:
        _audit_log("geo.set", session_id, user=who, role=(user or {}).get("role", ""),
                   detail=f"{georef['lat']:.6f},{georef['lon']:.6f} heading={georef['heading_deg']} "
                          f"source={georef['source']}")
    return {"ok": True, "georef": georef}


@router.delete("/{session_id}")
def geo_delete(session_id: str, user: Optional[Dict[str, Any]] = Depends(_user)):
    who = _name(user)
    removed = G.remove(_project_dir(session_id))
    if removed and _audit_log is not None:
        _audit_log("geo.remove", session_id, user=who, role=(user or {}).get("role", ""), detail="")
    return {"ok": True, "removed": removed}


def _scan_of(session_id: str, scan: Optional[str]) -> str:
    """The scan a place is about: the one asked, else the one the georeference speaks of, else the reference."""
    d = _project_dir(session_id)
    if scan:
        return scan
    g = G.load(d)
    return (g or {}).get("scan") or _project_row(d)["reference"] or ""


def _scan_ctx(session_id: str, scan: str):
    if _resolve_ctx is None:
        raise HTTPException(500, "geo router not configured")
    ctx = _resolve_ctx(session_id, scan) if scan else _resolve_ctx(session_id)
    if not Path(ctx.output_dir).is_dir():
        raise HTTPException(404, f"{session_id} {scan}: no such scan")
    return ctx


@router.get("/{session_id}/video")
def geo_from_video(session_id: str, scan: Optional[str] = None):
    """Where the scan's video says it was taken (its container's GPS tag), or None."""
    key = _scan_of(session_id, scan)
    ctx = _scan_ctx(session_id, key)
    src = Path(getattr(ctx, "source_dir", None) or Path(ctx.output_dir).parent)
    videos = [p for p in sorted(src.iterdir()) if p.suffix.lower() in (".mp4", ".mov", ".m4v")] if src.is_dir() else []
    video = next((p for p in videos if p.stem == "source_video"), videos[0] if videos else None)
    return {"ok": True, "scan": key, "video": video.name if video else None,
            "location": G.video_location(video) if video else None}


_PLANS: Dict[tuple, Dict[str, Any]] = {}


@router.get("/{session_id}/plan")
def geo_plan(session_id: str, scan: Optional[str] = None):
    """The scan from above (geo.plan_raster) as a PNG data URL with where it lies in the plan's frame. Kept in this
    process's memory for the cloud as it is (its size and date) — nothing is written in the session."""
    import base64
    key = _scan_of(session_id, scan)
    out = Path(_scan_ctx(session_id, key).output_dir)
    cloud, ft = out / "cleaned_cloud.ply", out / "floor_transform.npz"
    if not cloud.is_file():
        raise HTTPException(404, f"{session_id} {key}: no published cloud")
    st = cloud.stat()
    stamp = (str(cloud), st.st_size, st.st_mtime_ns, ft.stat().st_mtime_ns if ft.is_file() else 0)
    plan = _PLANS.get(stamp)
    if plan is None:
        try:
            r = G.plan_raster(out)
        except G.GeoError as e:
            raise HTTPException(500, str(e)) from e
        plan = {k: v for k, v in r.items() if k != "png"}
        plan["image"] = "data:image/png;base64," + base64.b64encode(r["png"]).decode("ascii")
        if len(_PLANS) >= 8:
            _PLANS.pop(next(iter(_PLANS)))
        _PLANS[stamp] = plan
    return {"ok": True, "scan": key, **plan}


@router.get("/{session_id}/content")
def geo_content(session_id: str, scan: Optional[str] = None):
    """What the globe can show of a work, in its scan: the GenRecon mesh's tileset when it has one, else the cloud's
    (a few points, geo.cloud_points_glb); the scan's floor adjustment. URLs name their scan (stac_scan / scan)."""
    from urllib.parse import quote
    key = _scan_of(session_id, scan)
    out = Path(_scan_ctx(session_id, key).output_dir)
    sid = quote(session_id, safe="")
    stac_scan = quote(f"{session_id}|{key}", safe="")
    mesh = None
    tsdf = out / "tsdf"
    folders = sorted(p.parent.parent.name for p in tsdf.glob("*/tiles/tileset.json")) if tsdf.is_dir() else []
    folder = "genrecon_photo" if "genrecon_photo" in folders else (folders[0] if folders else None)
    if folder:
        mesh = f"/api/segmentation/tsdf/tiles/{sid}/{quote(folder, safe='')}/tileset.json?stac_scan={stac_scan}"
    cloud = (f"/api/geo/{sid}/cloud/tileset.json?scan={quote(key, safe='')}"
             if (out / "cleaned_cloud.ply").is_file() else None)
    # the project's IFC models, as the BIM panel loads them (the page turns one into glTF for the globe on demand)
    ifcs = [{"name": n, "url": f"/api/sessions/{sid}/bim/{quote(n, safe='')}"}
            for n in _project_row(_project_dir(session_id))["ifcs"]]
    return {"ok": True, "scan": key, "georef": G.load(_project_dir(session_id)), "floor": G.floor_adjustment(out),
            "mesh_tileset": mesh, "mesh_folder": folder, "cloud_tileset": cloud, "ifcs": ifcs}


_CLOUDS: Dict[tuple, Dict[str, Any]] = {}


def _cloud(session_id: str, scan: Optional[str]) -> Dict[str, Any]:
    key = _scan_of(session_id, scan)
    out = Path(_scan_ctx(session_id, key).output_dir)
    cloud, ft = out / "cleaned_cloud.ply", out / "floor_transform.npz"
    if not cloud.is_file():
        raise HTTPException(404, f"{session_id} {key}: no published cloud")
    st = cloud.stat()
    stamp = (str(cloud), st.st_size, st.st_mtime_ns, ft.stat().st_mtime_ns if ft.is_file() else 0)
    got = _CLOUDS.get(stamp)
    if got is None:
        try:
            got = {**G.cloud_points_glb(out), "scan": key}
        except G.GeoError as e:
            raise HTTPException(500, str(e)) from e
        if len(_CLOUDS) >= 2:                                 # ~15 MB each
            _CLOUDS.pop(next(iter(_CLOUDS)))
        _CLOUDS[stamp] = got
    return got


@router.get("/{session_id}/cloud/tileset.json")
def geo_cloud_tileset(session_id: str, scan: Optional[str] = None):
    from urllib.parse import quote
    c = _cloud(session_id, scan)
    return {"asset": {"version": "1.1", "generator": "STAC geo_api"}, "geometricError": 0,
            "root": {"boundingVolume": {"box": c["box"]}, "geometricError": 0, "refine": "ADD",
                     "content": {"uri": f"cloud.glb?scan={quote(c['scan'], safe='')}"}}}


@router.get("/{session_id}/cloud/cloud.glb")
def geo_cloud_glb(session_id: str, scan: Optional[str] = None):
    from fastapi.responses import Response
    return Response(_cloud(session_id, scan)["glb"], media_type="model/gltf-binary")
