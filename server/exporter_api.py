"""THE EXPORTER'S ROUTES — what a scan can export, the export in its own process, its state and
the download (USER 2026-11-17: "que la persona lo pueda descargar en su maquina").

    GET  /api/export/{session_id}/plan?scan_key=        what can be exported and how large each item is
    POST /api/export/{session_id}                        {scan_key, items, confidence, lang} — launch it
    GET  /api/export/{session_id}/status?scan_key=      how it is going
    GET  /api/export/{session_id}/download?scan_key=    the ZIP of the scan's last export

THE SESSION'S EXPORT (USER 2026-11-19 — the Export section of the session's download window: "genera
otro zip pero no a nivel sesion … baja en un zip las nubes que se seleccionen, los mesh, los mesh
convertidos a IFC, los IFC que hay cargados"): ONE ZIP with what was ticked of each scan and the
project's loaded IFCs.

    GET  /api/export/{session_id}/session_plan          each scan's items and the loaded IFCs
    POST /api/export/{session_id}/session               {scans: {scan key: [items]}, loaded_ifcs, confidence, lang}
    GET  /api/export/{session_id}/session_status
    GET  /api/export/{session_id}/session_download      the ZIP of the session's last export

The export runs in a child process (CPU only — the chat and the voice stay up) that reads the
scan's files: it is refused while a pipeline job of the session is running or queued (its stages
rewrite those files). One export of a scan at a time; a new one replaces the scan's previous ZIP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from exporter import cloud as C
from exporter import package as P

router = APIRouter(prefix="/api/export", tags=["export"])
_busy: Optional[Callable] = None                 # (session_id) -> why the scan's files cannot be read now, or None
_audit_log: Optional[Callable] = None
_RUNS: Dict[str, Dict[str, Any]] = {}
_SERVER_DIR = Path(__file__).resolve().parent


def configure(busy: Optional[Callable] = None, audit_log: Optional[Callable] = None) -> None:
    global _busy, _audit_log
    _busy, _audit_log = busy, audit_log


def _key(session_id: str, scan_key: Optional[str]) -> str:
    return f"{session_id}|{scan_key or ''}"


def _plan(session_id: str, scan_key: Optional[str]) -> Dict[str, Any]:
    try:
        return P.plan(session_id, scan_key)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"error": str(e)}) from e
    except (P.ExportError, C.CloudError) as e:
        raise HTTPException(409, detail={"error": str(e)}) from e


@router.get("/{session_id}/plan")
def export_plan(session_id: str, scan_key: Optional[str] = Query(default=None)):
    return {"ok": True, **_plan(session_id, scan_key), "run": _RUNS.get(_key(session_id, scan_key)) or {"status": "idle"}}


@router.get("/{session_id}/status")
def export_status(session_id: str, scan_key: Optional[str] = Query(default=None)):
    return {"ok": True, **(_RUNS.get(_key(session_id, scan_key)) or {"status": "idle"})}


def _checked(body: Dict[str, Any]) -> tuple:
    confidence = body.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) < 1.0):
        raise HTTPException(400, detail={"error": "confidence must be in [0, 1)"})
    lang = body.get("lang", "es")
    if lang not in P.LANGS:
        raise HTTPException(400, detail={"error": f"lang must be one of {list(P.LANGS)}"})
    return float(confidence), lang


def _launch(key: str, argv: list, state: Dict[str, Any]) -> None:
    """Run the export's child process (CPU only) and keep ``state`` — what the interface polls — current."""
    import asyncio
    import os
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")       # the pod grants ~26 CPUs and shows 252
    env["CUDA_VISIBLE_DEVICES"] = ""             # an export never touches the card
    _RUNS[key] = state

    async def _run():
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=str(_SERVER_DIR), env=env, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            assert proc.stdout is not None
            report, error, tail = None, None, []
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip()
                if line.startswith("RESULT "):
                    report = json.loads(line[7:])
                elif line.startswith("ERROR "):
                    error = line[6:]
                elif line.startswith("PROGRESS "):
                    pct, _, msg = line[9:].partition(" ")
                    state.update(progress=int(pct) if pct.isdigit() else state["progress"], message=msg)
                elif line.strip():
                    print(f"[export] {line}")
                    tail = (tail + [line])[-5:]
            code = await proc.wait()
            if code == 0 and report is not None:
                state.update(status="done", progress=100, message="done", report=report)
            else:
                state.update(status="failed", error=error or f"the export ended with code {code}: {' | '.join(tail)}")
        except Exception as e:  # noqa: BLE001 — declared in the state the interface polls
            state.update(status="failed", error=f"{type(e).__name__}: {e}")

    asyncio.get_running_loop().create_task(_run())


@router.post("/{session_id}")
async def export_run(session_id: str, body: Dict[str, Any]):
    import sys
    scan_key = body.get("scan_key") or body.get("scan") or None
    items = body.get("items")
    if not isinstance(items, list) or not items or any(k not in P.ITEMS for k in items):
        raise HTTPException(400, detail={"error": f"items must be a list among {list(P.ITEMS)}"})
    confidence, lang = _checked(body)
    key = _key(session_id, scan_key)
    if (_RUNS.get(key) or {}).get("status") == "running":
        raise HTTPException(409, detail={"error": "this scan is being exported already"})
    why = _busy(session_id) if _busy is not None else None
    if why:
        raise HTTPException(409, detail={"error": why, "busy": True})
    available = _plan(session_id, scan_key)["items"]
    missing = [f"{k}: {available[k].get('why')}" for k in items if not available[k]["available"]]
    if missing:
        raise HTTPException(409, detail={"error": "cannot export " + "; ".join(missing)})
    argv = [sys.executable, "-m", "exporter.package", "--session", session_id, "--items", ",".join(items),
            "--confidence", repr(confidence), "--lang", lang]
    if scan_key:
        argv += ["--scan", scan_key]
    state: Dict[str, Any] = {"status": "running", "items": list(items), "confidence": confidence,
                             "progress": 0, "message": "starting"}
    if _audit_log is not None:
        _audit_log("export.run", session_id, detail=json.dumps({"scan": scan_key, "items": items, "confidence": confidence}))
    _launch(key, argv, state)
    return {"ok": True, **state}


@router.get("/{session_id}/download")
def export_download(session_id: str, scan_key: Optional[str] = Query(default=None)):
    if (_RUNS.get(_key(session_id, scan_key)) or {}).get("status") == "running":
        raise HTTPException(409, detail={"error": "the export is still running"})
    try:
        path = P.zip_path(P.context(session_id, scan_key), session_id, scan_key)
    except (FileNotFoundError, P.ExportError) as e:
        raise HTTPException(404, detail={"error": str(e)}) from e
    if not path.exists():
        raise HTTPException(404, detail={"error": "this scan has no export yet"})
    return FileResponse(str(path), media_type="application/zip", filename=path.name)


# ── the session's export: several scans and the loaded IFCs in one ZIP ───────────────────────────
def _session_key(session_id: str) -> str:
    return f"{session_id}|*"


def _session_plan(session_id: str) -> Dict[str, Any]:
    try:
        return P.plan_all(session_id)
    except FileNotFoundError as e:
        raise HTTPException(404, detail={"error": str(e)}) from e
    except (P.ExportError, C.CloudError) as e:
        raise HTTPException(409, detail={"error": str(e)}) from e


@router.get("/{session_id}/session_plan")
def export_session_plan(session_id: str):
    return {"ok": True, **_session_plan(session_id), "run": _RUNS.get(_session_key(session_id)) or {"status": "idle"}}


@router.get("/{session_id}/session_status")
def export_session_status(session_id: str):
    return {"ok": True, **(_RUNS.get(_session_key(session_id)) or {"status": "idle"})}


@router.post("/{session_id}/session")
async def export_session_run(session_id: str, body: Dict[str, Any]):
    import sys
    scans = body.get("scans") or {}
    if (not isinstance(scans, dict)
            or any(not isinstance(v, list) or any(k not in P.ITEMS for k in v) for v in scans.values())):
        raise HTTPException(400, detail={"error": f"scans must be {{scan key: [items among {list(P.ITEMS)}]}}"})
    scans = {k: list(v) for k, v in scans.items() if v}
    loaded = bool(body.get("loaded_ifcs"))
    confidence, lang = _checked(body)
    key = _session_key(session_id)
    if (_RUNS.get(key) or {}).get("status") == "running":
        raise HTTPException(409, detail={"error": "this session is being exported already"})
    why = _busy(session_id) if _busy is not None else None
    if why:
        raise HTTPException(409, detail={"error": why, "busy": True})
    plan = _session_plan(session_id)
    unknown = [k for k in scans if k not in plan["scans"]]
    if unknown:
        raise HTTPException(404, detail={"error": f"no scan {unknown} in {session_id}"})
    missing = [f"{k} {item}: {plan['scans'][k]['items'][item].get('why')}"
               for k, items in scans.items() for item in items if not plan["scans"][k]["items"][item]["available"]]
    if missing:
        raise HTTPException(409, detail={"error": "cannot export " + "; ".join(missing)})
    if not scans and not (loaded and plan["loaded_ifcs"]):
        raise HTTPException(400, detail={"error": "nothing to export: tick something of a scan, or the loaded IFCs"})
    selection = {"scans": scans, "loaded_ifcs": loaded}
    argv = [sys.executable, "-m", "exporter.package", "--session", session_id, "--selection", json.dumps(selection),
            "--confidence", repr(confidence), "--lang", lang]
    state: Dict[str, Any] = {"status": "running", "selection": selection, "confidence": confidence,
                             "progress": 0, "message": "starting"}
    if _audit_log is not None:
        _audit_log("export.run", session_id, detail=json.dumps({**selection, "confidence": confidence}))
    _launch(key, argv, state)
    return {"ok": True, **state}


@router.get("/{session_id}/session_download")
def export_session_download(session_id: str):
    if (_RUNS.get(_session_key(session_id)) or {}).get("status") == "running":
        raise HTTPException(409, detail={"error": "the export is still running"})
    try:
        path = P.session_zip_path(P.context_of_project(session_id), session_id)
    except (FileNotFoundError, P.ExportError) as e:
        raise HTTPException(404, detail={"error": str(e)}) from e
    if not path.exists():
        raise HTTPException(404, detail={"error": "this session has no export yet"})
    return FileResponse(str(path), media_type="application/zip", filename=path.name)
