"""THE COMMON POINTS' ROUTES (fusion/common_points.py — USER 2026-11-17: "deberían marcarse dos
puntos y decir estos son los mismos en ambas … para que se puedan posar uno sobre otro y comparar").

    GET    /api/project/{project}/common_points                    the points, each with the scans it is marked on
    POST   /api/project/{project}/common_points                    {name?} — a new point (P<n>)
    PUT    /api/project/{project}/common_points/{point}/marks      {scan, position} — mark it on a scan
    DELETE /api/project/{project}/common_points/{point}/marks?scan=
    DELETE /api/project/{project}/common_points/{point}
    GET    /api/project/{project}/common_points/solve?scan=&reference=&scale=
                                                                   the similarity scan → reference the points
                                                                   marked on both give, with each one's residual

``solve`` REPORTS: nothing of the project is changed by it (where the factor is applied is the
user's decision). Wired by main.py through :func:`configure`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from fusion import common_points as CP

router = APIRouter(prefix="/api/project", tags=["common points"])
_security = HTTPBearer(auto_error=False)
_resolve_ctx: Optional[Callable] = None
_audit_log: Optional[Callable] = None


def configure(resolve_ctx: Callable, audit_log: Optional[Callable] = None) -> None:
    global _resolve_ctx, _audit_log
    _resolve_ctx, _audit_log = resolve_ctx, audit_log


def _user_name(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)) -> str:
    if not credentials:
        return ""
    from auth import decode_token
    payload = decode_token(credentials.credentials)
    return str(payload.get("username") or payload.get("sub") or "")


def _ctx(project: str, scan: Optional[str] = None):
    if _resolve_ctx is None:
        raise HTTPException(500, "common points router not configured")
    try:
        ctx = _resolve_ctx(project, scan) if scan else _resolve_ctx(project)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
    if not Path(ctx.session_dir).is_dir():
        raise HTTPException(404, f"{project}{' (' + scan + ')' if scan else ''} does not exist")
    return ctx


def _project_dir(project: str) -> Path:
    ctx = _ctx(project)
    project_dir = getattr(ctx, "project_dir", None)
    if project_dir is None:
        raise HTTPException(409, f"{project} is a session of the old layout: it holds one scan, nothing to lay on another")
    return Path(project_dir)


def _epochs(project: str, project_dir: Path) -> Dict[str, int]:
    """{scan key: its geometry epoch now} for every scan a point is marked on (a scan that is gone is not judged)."""
    from correction.epoch import current_epoch
    out: Dict[str, int] = {}
    for p in CP.load(project_dir)["points"]:
        for scan in p["marks"]:
            if scan not in out:
                try:
                    out[scan] = current_epoch(_ctx(project, scan).output_dir)
                except Exception:  # noqa: BLE001
                    pass
    return out


def _points(project: str, project_dir: Path) -> Dict[str, Any]:
    try:
        return {"ok": True, "points": CP.listing(project_dir, _epochs(project, project_dir))}
    except CP.CommonPointsError as e:
        raise HTTPException(500, str(e)) from e


@router.get("/{project}/common_points")
def common_points_list(project: str):
    return _points(project, _project_dir(project))


@router.get("/{project}/common_points/solve")
def common_points_solve(project: str, scan: str = Query(...), reference: str = Query(...), scale: Optional[float] = Query(default=None)):
    project_dir = _project_dir(project)
    try:
        return {"ok": True, **CP.solve(project_dir, scan, reference, _epochs(project, project_dir), scale)}
    except CP.CommonPointsError as e:
        raise HTTPException(409, str(e)) from e


@router.post("/{project}/common_points")
def common_points_add(project: str, body: Dict[str, Any]):
    project_dir = _project_dir(project)
    try:
        point = CP.add(project_dir, body.get("name"))
    except CP.CommonPointsError as e:
        raise HTTPException(400, str(e)) from e
    return {**_points(project, project_dir), "point": point["id"]}


@router.put("/{project}/common_points/{point}/marks")
def common_points_mark(project: str, point: str, body: Dict[str, Any], author: str = Depends(_user_name)):
    project_dir = _project_dir(project)
    scan = body.get("scan")
    if not isinstance(scan, str) or not scan:
        raise HTTPException(400, "scan (the scan the point is marked on) is required")
    from correction.epoch import current_epoch
    epoch = current_epoch(_ctx(project, scan).output_dir)        # (404 when the project has no such scan)
    try:
        CP.mark(project_dir, point, scan, body.get("position"), geometry_epoch=epoch, author=author)
    except KeyError:
        raise HTTPException(404, f"no common point {point} in {project}") from None
    except CP.CommonPointsError as e:
        raise HTTPException(400, str(e)) from e
    if _audit_log is not None:
        _audit_log("common_point.mark", project, user=author or "unknown", detail=f"{point} on {scan}")
    return _points(project, project_dir)


@router.delete("/{project}/common_points/{point}/marks")
def common_points_unmark(project: str, point: str, scan: str = Query(...)):
    project_dir = _project_dir(project)
    try:
        CP.unmark(project_dir, point, scan)
    except KeyError:
        raise HTTPException(404, f"common point {point} is not marked on {scan}") from None
    return _points(project, project_dir)


@router.delete("/{project}/common_points/{point}")
def common_points_delete(project: str, point: str):
    project_dir = _project_dir(project)
    try:
        CP.delete(project_dir, point)
    except KeyError:
        raise HTTPException(404, f"no common point {point} in {project}") from None
    return _points(project, project_dir)
