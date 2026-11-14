"""REST of a scan's calibration measurements taken on the 3D model (calibration3d.py).

    GET    /api/calibration/{session_id}?scan_key=                  the document + its summary
    POST   /api/calibration/{session_id}/measurements               add one (two picked points
                                                                    + the real length)
    PATCH  /api/calibration/{session_id}/measurements/{mid}         what the user states about it
    DELETE /api/calibration/{session_id}/measurements/{mid}?scan_key=
    POST   /api/calibration/{session_id}/planes                     add a plane (three or more
                                                                    picked points)
    PATCH  /api/calibration/{session_id}/planes/{pid}               its label / note
    DELETE /api/calibration/{session_id}/planes/{pid}?scan_key=
    PUT    /api/calibration/{session_id}/mode                       global | axes
    POST   /api/calibration/{session_id}/apply                      THE REAL RESCALE: the scan's
                                                                    geometry rewritten at the
                                                                    calibrated scale (a new epoch)
    GET    /api/calibration/{session_id}/apply?scan_key=            how that is going

Wired by ``main.py`` through :func:`configure` (the session resolver and the audit log are the
server's; this module never imports main), as the evaluation router is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

import calibration3d as C

router = APIRouter(prefix="/api/calibration", tags=["calibration"])
_resolve_ctx: Optional[Callable] = None
_audit_log: Optional[Callable] = None
_busy: Optional[Callable] = None                 # (session_id) -> why the scan cannot be rewritten now, or None
_on_cloud_changed: Optional[Callable] = None     # (session_id, output_dir): tell the open viewers


def configure(resolve_ctx: Callable, audit_log: Optional[Callable] = None,
              busy: Optional[Callable] = None, on_cloud_changed: Optional[Callable] = None) -> None:
    global _resolve_ctx, _audit_log, _busy, _on_cloud_changed
    _resolve_ctx = resolve_ctx
    _audit_log = audit_log
    _busy = busy
    _on_cloud_changed = on_cloud_changed


def _output_dir(session_id: str, scan_key: Optional[str]) -> Path:
    if _resolve_ctx is None:
        raise HTTPException(500, "calibration router not configured")
    try:
        ctx = _resolve_ctx(session_id, scan_key) if scan_key else _resolve_ctx(session_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    if not Path(ctx.session_dir).is_dir():
        # the resolver returns a to-be-created context for an unknown id: never create a scan here
        raise HTTPException(404, f"scan {session_id} ({scan_key or 'active'}) does not exist")
    return Path(ctx.output_dir)


def _epoch(out: Path) -> int:
    from correction.epoch import current_epoch
    return int(current_epoch(out))


def _answer(out: Path, doc: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    return {"ok": True, "path": str(C.path_of(out)), "document": doc,
            "summary": C.summary(doc, _epoch(out)), **extra}


def _load(out: Path) -> Dict[str, Any]:
    try:
        return C.load(out)
    except C.CalibrationError as e:
        raise HTTPException(409, detail={"error": str(e)}) from e


def _log(action: str, session_id: str, detail: Dict[str, Any]) -> None:
    if _audit_log is not None:
        _audit_log(action, session_id, detail=json.dumps(detail))


@router.get("/{session_id}")
def get_calibration(session_id: str, scan_key: Optional[str] = Query(default=None)):
    out = _output_dir(session_id, scan_key)
    return _answer(out, _load(out))


@router.post("/{session_id}/measurements")
def add_measurement(session_id: str, body: Dict[str, Any]):
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        m = C.add(doc, body, _epoch(out))
        C.save(out, doc)
    except C.CalibrationError as e:
        raise HTTPException(400, detail={"error": str(e)}) from e
    _log("calibration.add", session_id, {"id": m["id"], "role": m["role"], "value_m": m["value_m"],
                                         "measured_m": m["measured_m"]})
    return _answer(out, doc, measurement=m)


@router.patch("/{session_id}/measurements/{mid}")
def update_measurement(session_id: str, mid: str, body: Dict[str, Any]):
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        m = C.update(doc, mid, body)
        C.save(out, doc)
    except KeyError:
        raise HTTPException(404, f"no calibration measurement {mid!r}")
    except C.CalibrationError as e:
        raise HTTPException(400, detail={"error": str(e)}) from e
    _log("calibration.update", session_id, {"id": mid})
    return _answer(out, doc, measurement=m)


@router.delete("/{session_id}/measurements/{mid}")
def delete_measurement(session_id: str, mid: str, scan_key: Optional[str] = Query(default=None)):
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        C.remove(doc, mid)
        C.save(out, doc)
    except KeyError:
        raise HTTPException(404, f"no calibration measurement {mid!r}")
    _log("calibration.delete", session_id, {"id": mid})
    return _answer(out, doc)


@router.post("/{session_id}/planes")
def add_plane(session_id: str, body: Dict[str, Any]):
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        pl = C.add_plane(doc, body, _epoch(out))
        C.save(out, doc)
    except C.CalibrationError as e:
        raise HTTPException(400, detail={"error": str(e)}) from e
    _log("calibration.plane.add", session_id, {"id": pl["id"], "label": pl["label"],
                                               "inclination_deg": pl["inclination_deg"]})
    return _answer(out, doc, plane=pl)


@router.patch("/{session_id}/planes/{pid}")
def update_plane(session_id: str, pid: str, body: Dict[str, Any]):
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        pl = C.update_plane(doc, pid, body)
        C.save(out, doc)
    except KeyError:
        raise HTTPException(404, f"no calibration plane {pid!r}")
    except C.CalibrationError as e:
        raise HTTPException(400, detail={"error": str(e)}) from e
    _log("calibration.plane.update", session_id, {"id": pid})
    return _answer(out, doc, plane=pl)


@router.delete("/{session_id}/planes/{pid}")
def delete_plane(session_id: str, pid: str, scan_key: Optional[str] = Query(default=None)):
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        C.remove_plane(doc, pid)
        C.save(out, doc)
    except KeyError:
        raise HTTPException(404, f"no calibration plane {pid!r}")
    _log("calibration.plane.delete", session_id, {"id": pid})
    return _answer(out, doc)


@router.put("/{session_id}/mode")
def set_mode(session_id: str, body: Dict[str, Any]):
    """The type of calibration: 'global' (one factor) or 'axes' (one per axis — refused unless a
    gauge measurement runs along each of the three axes)."""
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        mode = C.set_mode(doc, body.get("mode"), _epoch(out))
        C.save(out, doc)
    except C.CalibrationError as e:
        raise HTTPException(400, detail={"error": str(e)}) from e
    _log("calibration.mode", session_id, {"mode": mode})
    return _answer(out, doc)


# ── THE REAL RESCALE: the calibration applied to the scan's geometry ─────────────────────────
# USER 2026-11-13: "el reescalado debe ser real … debe generarse la nube real escalada".
# correction/rescale.py does it, in its own process (the cloud is rewritten and the octree
# rebuilt: minutes on a large scan, CPU only — the chat's engines stay up); this route decides
# whether it may run, launches it, and tells the open viewers when the new epoch is live.
_APPLY: Dict[str, Dict[str, Any]] = {}
_SERVER_DIR = Path(__file__).resolve().parent


def apply_refusal(summary: Dict[str, Any]) -> Optional[str]:
    """Why this calibration cannot be applied to the geometry (None when it can)."""
    if summary.get("state") == "uncalibrated" or summary.get("scale_factor") is None:
        return "the scan has no calibration measurement on its present geometry"
    if summary.get("mode_used") == "axes":
        return ("the calibration is per axis: only a GLOBAL factor is applied to the geometry — a "
                "different factor on each axis is not a similarity and the cameras cannot follow "
                "it (set the calibration to global to apply it)")
    s = float(summary["scale_factor"])
    if not (s > 0.0) or s != s:
        return f"the factor {s!r} is not a scale"
    if s == 1.0:
        return "the factor is exactly 1: the scan is already at the calibrated scale"
    return None


def kept_epoch_bytes(out: Path) -> int:
    """What a rescale ADDS to the disk: the present epoch is kept beside the new one — its cloud,
    its raw cloud and its octree are the bulk of it. (The volume's quota is invisible to the
    system: the user is told the figure before confirming.)"""
    total = 0
    for rel in ("cleaned_cloud.ply", "cleaned_cloud_raw.ply"):
        p = out / rel
        if p.exists():
            total += p.stat().st_size
    potree = out / "potree"
    if potree.is_dir():
        total += sum(f.stat().st_size for f in potree.iterdir() if f.is_file())
    return int(total)


@router.get("/{session_id}/apply")
def apply_status(session_id: str, scan_key: Optional[str] = Query(default=None)):
    out = _output_dir(session_id, scan_key)
    st = _APPLY.get(str(out))
    return {"ok": True, "keeps_bytes": kept_epoch_bytes(out), **(st or {"status": "idle"})}


@router.post("/{session_id}/apply")
async def apply_calibration(session_id: str, body: Dict[str, Any]):
    import asyncio
    import os
    import sys
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    key = str(out)
    if (_APPLY.get(key) or {}).get("status") == "running":
        raise HTTPException(409, detail={"error": "this scan is being rescaled already"})
    why = _busy(session_id) if _busy is not None else None
    if why:
        raise HTTPException(409, detail={"error": why, "busy": True})
    doc = _load(out)
    epoch = _epoch(out)
    summ = C.summary(doc, epoch)
    refusal = apply_refusal(summ)
    if refusal:
        raise HTTPException(409, detail={"error": refusal})
    scale = float(summ["scale_factor"])
    source = {"mode": summ.get("mode_used"), "scale_factor": scale, "state": summ.get("state"),
              "n_gauge": summ.get("n_gauge"), "n_test": summ.get("n_test"),
              "verification": summ.get("verification"), "epoch": int(epoch)}
    argv = [sys.executable, "-m", "correction.rescale_worker", "--session", session_id,
            "--scale", repr(scale), "--source", json.dumps(source, default=float)]
    if scan_key:
        argv += ["--scan", scan_key]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")       # the pod grants ~26 CPUs and shows 252
    state: Dict[str, Any] = {"status": "running", "scale": scale, "epoch_from": int(epoch), "progress": 0,
                             "message": "starting", "log": []}
    _APPLY[key] = state
    _log("calibration.apply", session_id, {"scale": scale, "epoch_from": int(epoch), "scan": scan_key})

    async def _run():
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=str(_SERVER_DIR), env=env, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            assert proc.stdout is not None
            report = None
            error = None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip()
                if line.startswith("RESULT "):
                    report = json.loads(line[7:])
                elif line.startswith("ERROR "):
                    error = line[6:]
                elif line.startswith("PROGRESS "):
                    _p, _, msg = line[9:].partition(" ")
                    state.update(progress=int(_p) if _p.isdigit() else state["progress"], message=msg)
                else:
                    print(f"[rescale] {line}")
                    state["log"] = (state["log"] + [line])[-40:]
            code = await proc.wait()
            if code == 0 and report is not None:
                state.update(status="done", progress=100, message="applied", report=report,
                             epoch_to=report.get("epoch_to"))
                if _on_cloud_changed is not None:
                    await _on_cloud_changed(session_id, out)
            else:
                state.update(status="failed", error=error or f"the rescale ended with code {code}")
        except Exception as e:  # noqa: BLE001 — declared in the state the interface polls
            state.update(status="failed", error=f"{type(e).__name__}: {e}")

    asyncio.get_running_loop().create_task(_run())
    return {"ok": True, **state}
