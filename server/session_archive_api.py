"""THE SESSION ARCHIVE'S ROUTES — a session downloaded as one ZIP, and a ZIP uploaded as a session
(USER 2026-11-18: "una función en cada sesión para descargarla como zip … y para subirla … en nueva
sesión … crear una vacía o subir un archivo zip que va a comprobar que tenga la estructura correcta
… la sube y la descomprime, elimina el zip y se queda con la sesión perfectamente utilizable").

    GET    /api/session_archive/{session_id}/plan        what the ZIP will hold (files, bytes, scans), or why not now
    POST   /api/session_archive/{session_id}/ticket      the link to the download (60 s to start it)
    GET    /api/session_archive/{session_id}/download    the ZIP (?ticket=) — whole, or the RANGE asked for

    POST   /api/session_import                           {filename, size, name?} — open an upload → {job, part_bytes}
    PUT    /api/session_import/{job}/part?offset=N       the next part of the file (raw body)
    POST   /api/session_import/{job}/finish              the file is whole: validate it, extract it, delete it
    GET    /api/session_import/{job}                     how it is going
    DELETE /api/session_import/{job}                     give it up (the partial file is removed)

Admin and manager only, as creating a session is. THE DOWNLOAD is a browser download (the browser
writes it to the disk as it arrives — a 100 GB session never sits in the page's memory), and a link
cannot carry the login header: the ticket is asked for WITH the login and names the session.
THE ZIP IS NEVER ON THE VOLUME: its bytes are read from the session as they are sent
(session_archive.Layout). ITS SIZE IS KNOWN before the first byte (Content-Length: the browser shows
what is left) and ANY RANGE of it can be asked for again (Accept-Ranges + ETag): a download cut by
the network is RESUMED by the browser — or by `curl -C -` / `wget -c` on the same link — where it
stopped. The link starts within 60 s and then stays good for that download and its resumptions
(RESUME_SECONDS; shorter than a login). While a session is being read no pipeline job of it may
start, and it is not read while one runs.

THE UPLOAD comes in parts (32 MB): a proxy's body limit or a cut connection costs one part, not the
file; each part is appended only at the offset the server already holds (a repeated or lost part is
answered with that offset). The file is written beside the sessions, under a hidden name, and
removed whatever the outcome. The session's manifest is the ZIP's first entry: as soon as it has
arrived the upload is refused for what the manifest alone tells (not a session, a name that exists)
— not after the last of 100 GB.
"""
from __future__ import annotations

import asyncio
import re
import secrets
import threading
import time
from email.utils import formatdate
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import session_archive as A

router = APIRouter(tags=["session archive"])
_security = HTTPBearer(auto_error=False)

PART_BYTES = 32 * 1024 * 1024
TICKET_SECONDS = 60                                                # to START the download
RESUME_SECONDS = 12 * 3600                                         # … and then to resume it (a login lasts 24 h)
MAX_TICKETS = 8                                                    # (each holds its session's manifest)
_projects_dir: Optional[Path] = None
_busy: Optional[Callable[[str], Optional[str]]] = None            # (session_id) -> why its files cannot be read now
_audit_log: Optional[Callable[..., None]] = None
_commit = ""
_TICKETS: Dict[str, Dict[str, Any]] = {}                           # ticket -> {session_id, layout, expires, used, user, …}
_READING: Dict[str, int] = {}                                      # session_id -> downloads in progress
_IMPORTS: Dict[str, Dict[str, Any]] = {}                           # job -> state
_lock = threading.Lock()


def configure(projects_dir: Path, busy: Optional[Callable] = None, audit_log: Optional[Callable] = None,
              commit: str = "") -> None:
    global _projects_dir, _busy, _audit_log, _commit
    _projects_dir, _busy, _audit_log, _commit = Path(projects_dir), busy, audit_log, commit


def reading(session_id: str) -> Optional[str]:
    """Why the session's files must not be rewritten now: it is being downloaded."""
    return f"the session {session_id} is being downloaded as a ZIP: wait for it to end" if _READING.get(session_id) else None


def _manager(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)) -> Dict[str, Any]:
    from auth import decode_token
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload.get("role") not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Admin or Manager only")
    return payload


def _project(session_id: str) -> Path:
    if not A.ID_RE.match(session_id) or _projects_dir is None:
        raise HTTPException(status_code=404, detail="Session not found")
    d = _projects_dir / session_id
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found (only sessions of the project layout can be archived)")
    return d


def _free(session_id: str) -> Path:
    d = _project(session_id)
    why = _busy(session_id) if _busy is not None else None
    if why:
        raise HTTPException(status_code=409, detail=why)
    return d


def _layout(session_id: str) -> A.Layout:
    d = _free(session_id)
    try:
        return A.Layout(d, session_id, A.manifest_of(d, session_id, _commit))
    except A.ArchiveError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


def _summary(layout: A.Layout) -> Dict[str, Any]:
    m = layout.manifest
    return {"session_id": m["session_id"], "files": m["files"], "bytes": m["bytes"], "scans": m["scans"],
            "zip_bytes": layout.total, "downloading": bool(_READING.get(m["session_id"]))}


# ── the download ───────────────────────────────────────────────────────────────────────────────
@router.get("/api/session_archive/{session_id}/plan")
def archive_plan(session_id: str, _user: Dict[str, Any] = Depends(_manager)):
    return {"ok": True, **_summary(_layout(session_id))}


@router.post("/api/session_archive/{session_id}/ticket")
def archive_ticket(session_id: str, user: Dict[str, Any] = Depends(_manager)):
    layout = _layout(session_id)
    now = time.time()
    ticket = secrets.token_urlsafe(32)
    with _lock:
        for t in [t for t, v in _TICKETS.items() if v["expires"] < now]:
            del _TICKETS[t]
        for t in sorted(_TICKETS, key=lambda t: _TICKETS[t]["created"])[:max(0, len(_TICKETS) - MAX_TICKETS + 1)]:
            del _TICKETS[t]
        _TICKETS[ticket] = {"session_id": session_id, "layout": layout, "created": now, "expires": now + TICKET_SECONDS,
                            "used": False, "failed": None,
                            "user": user.get("username", user.get("sub", "unknown")), "role": user.get("role", "")}
    return {"ok": True, **_summary(layout), "url": f"/api/session_archive/{session_id}/download?ticket={ticket}",
            "filename": f"{session_id}_stac_session.zip", "expires_in": TICKET_SECONDS, "resume_seconds": RESUME_SECONDS}


_RANGE = re.compile(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$")


def _range(header: Optional[str], total: int) -> Optional[Tuple[int, int]]:
    """The ONE range a request asks for, as [start, stop) — None when it asks for none (or for
    several, or in a form this does not read: the whole ZIP is then the answer, which is always
    right); a range that lies outside the ZIP raises 416."""
    m = _RANGE.match(header or "")
    if not m or not (m.group(1) or m.group(2)):
        return None
    if m.group(1):
        start = int(m.group(1))
        stop = min(int(m.group(2)) + 1, total) if m.group(2) else total
    else:
        start, stop = max(0, total - int(m.group(2))), total      # (the last n bytes)
    if start >= stop:
        raise HTTPException(status_code=416, detail="No such range", headers={"Content-Range": f"bytes */{total}"})
    return start, stop


@router.api_route("/api/session_archive/{session_id}/download", methods=["GET", "HEAD"])
def archive_download(session_id: str, request: Request, ticket: str = Query(...)):
    now = time.time()
    first = False
    with _lock:
        t = _TICKETS.get(ticket)
        if t is not None and (t["session_id"] != session_id or t["expires"] < now):
            _TICKETS.pop(ticket, None)
            t = None
        if t is not None and not t["used"]:
            first = True
            t.update(used=True, expires=t["created"] + RESUME_SECONDS)
    if t is None:
        raise HTTPException(status_code=403, detail="This download link is not valid any more: ask for the download again")
    if t["failed"]:
        raise HTTPException(status_code=409, detail=t["failed"])
    _free(session_id)
    layout: A.Layout = t["layout"]
    etag = f'"{layout.etag}"'
    headers = {"Content-Disposition": f'attachment; filename="{session_id}_stac_session.zip"',
               "Accept-Ranges": "bytes", "ETag": etag, "Last-Modified": formatdate(t["created"], usegmt=True),
               "X-STAC-Session-Bytes": str(layout.manifest["bytes"]), "X-STAC-Session-Files": str(layout.manifest["files"]),
               "Cache-Control": "private, no-store"}
    start, stop, status = 0, layout.total, 200
    # (If-Range: "only if it is still that ZIP" — another one, and the whole of this one is sent)
    if request.headers.get("if-range", etag).strip() == etag:
        asked = _range(request.headers.get("range"), layout.total)
        if asked is not None:
            start, stop = asked
            status = 206
            headers["Content-Range"] = f"bytes {start}-{stop - 1}/{layout.total}"
    headers["Content-Length"] = str(stop - start)
    if request.method == "HEAD":
        return Response(status_code=status, headers=headers, media_type="application/zip")
    if first and _audit_log is not None:
        _audit_log("archive.download", session_id, user=t["user"], role=t["role"],
                   detail=f"files={layout.manifest['files']} bytes={layout.manifest['bytes']} zip={layout.total}")
    return StreamingResponse(_stream(session_id, layout, start, stop, t), status_code=status,
                             media_type="application/zip", headers=headers)


def _stream(session_id: str, layout: A.Layout, start: int, stop: int, ticket: Optional[Dict[str, Any]] = None) -> Iterator[bytes]:
    """The ZIP's bytes [start, stop) as they are read from the session. While they are read the
    session counts as READ (reading()); when the reader leaves — the end, or a client that went
    away — it is free again. A session that is no longer the one the download began with ends the
    response short of its declared length (the browser reports the download as failed, never a cut
    ZIP as whole) and the link answers no more."""
    # (counted HERE, at the first byte asked for: a response that is never read leaves nothing counted)
    with _lock:
        _READING[session_id] = _READING.get(session_id, 0) + 1
    try:
        yield from layout.stream(start, stop)
    except A.ArchiveError as e:
        print(f"[archive] ❌ {session_id}: {e}", flush=True)
        if ticket is not None:
            ticket["failed"] = f"{e} — ask for the download again"
        raise RuntimeError(str(e)) from e
    finally:
        with _lock:
            _READING[session_id] = max(0, _READING.get(session_id, 1) - 1)
            if not _READING[session_id]:
                _READING.pop(session_id, None)


# ── the upload ─────────────────────────────────────────────────────────────────────────────────
def _incoming() -> Path:
    if _projects_dir is None:
        raise HTTPException(status_code=503, detail="The session archive is not configured")
    d = _projects_dir / ".incoming"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job(job: str) -> Dict[str, Any]:
    st = _IMPORTS.get(job)
    if st is None:
        raise HTTPException(status_code=404, detail="This upload is not known (the server was restarted?): start it again")
    return st


def _public(st: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in st.items() if k not in ("path", "cancel")}


@router.post("/api/session_import")
async def import_open(body: Dict[str, Any], user: Dict[str, Any] = Depends(_manager)):
    size = body.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise HTTPException(status_code=400, detail="size must be the file's size in bytes")
    name = (body.get("name") or "").strip() or None
    if name is not None and not A.ID_RE.match(name):
        raise HTTPException(status_code=400, detail="A session's name can only contain letters, numbers, dashes and underscores")
    if name is not None and _projects_dir is not None and (_projects_dir / name).exists():
        raise HTTPException(status_code=409, detail=f"A session named '{name}' already exists")
    job = secrets.token_hex(8)
    path = _incoming() / f"{job}.zip"
    path.write_bytes(b"")
    _IMPORTS[job] = {"job": job, "phase": "uploading", "filename": str(body.get("filename") or "session.zip")[:200],
                     "size": size, "received": 0, "name": name, "pct": 0.0, "error": None, "session_id": None,
                     "head": 30,                                  # bytes needed to read the manifest; 0 = said, or not sayable
                     "path": str(path), "cancel": threading.Event(),
                     "user": user.get("username", user.get("sub", "unknown")), "role": user.get("role", ""), "started_at": time.time()}
    return {"ok": True, "job": job, "part_bytes": PART_BYTES}


@router.put("/api/session_import/{job}/part")
async def import_part(job: str, request: Request, offset: int = Query(..., ge=0), _user: Dict[str, Any] = Depends(_manager)):
    st = _job(job)
    if st["phase"] != "uploading":
        raise HTTPException(status_code=409, detail=f"This upload is {st['phase']}: it takes no more parts")
    if offset != st["received"]:
        # (a repeated part, or one that never arrived: the client goes on from what the server holds)
        raise HTTPException(status_code=409, detail={"error": "offset", "received": st["received"]})
    written = 0
    try:
        with open(st["path"], "ab") as out:
            async for chunk in request.stream():
                if st["received"] + written + len(chunk) > st["size"]:
                    raise HTTPException(status_code=400, detail="More bytes than the size the upload was opened with")
                out.write(chunk)
                written += len(chunk)
    except OSError as e:
        _fail(st, "the volume is full: the upload was removed" if e.errno in (28, 122) else f"{type(e).__name__}: {e}")
        raise HTTPException(status_code=507, detail=st["error"]) from e
    except BaseException:
        # a part cut on its way: what it wrote is taken back, the client sends that part again
        with open(st["path"], "ab") as out:
            out.truncate(st["received"])
        raise
    st["received"] += written
    st["pct"] = round(100.0 * st["received"] / st["size"], 1)
    if st["head"] and st["received"] >= min(st["head"], st["size"]):
        # the manifest is the ZIP's first entry: what it alone tells is told NOW, not after the last part
        try:
            manifest, need = A.manifest_at_head(Path(st["path"]))
            if manifest is not None:
                report = A.check_head(manifest, _projects_dir, st["name"])
                st.update(session_id=report["session_id"], files=report["files"], bytes=report["bytes"])
            st["head"] = 0 if manifest is not None or st["received"] >= st["size"] else need
        except A.ArchiveError as e:
            _fail(st, str(e))
            raise HTTPException(status_code=422, detail=str(e)) from e
    return {"ok": True, "received": st["received"], "session_id": st["session_id"]}


def _fail(st: Dict[str, Any], why: str) -> None:
    st.update(phase="failed", error=why)
    try:
        Path(st["path"]).unlink()
    except OSError:
        pass


@router.post("/api/session_import/{job}/finish")
async def import_finish(job: str, _user: Dict[str, Any] = Depends(_manager)):
    st = _job(job)
    if st["phase"] != "uploading":
        return {"ok": True, **_public(st)}
    if st["received"] != st["size"]:
        raise HTTPException(status_code=409, detail={"error": "incomplete", "received": st["received"], "size": st["size"]})
    st.update(phase="validating", pct=0.0)

    def work() -> None:
        path = Path(st["path"])
        try:
            report = A.validate(path, _projects_dir, st["name"])
            st.update(phase="extracting", session_id=report["session_id"], files=report["files"], bytes=report["bytes"])

            def progress(done: int, total: int, _what: str) -> None:
                st["pct"] = round(100.0 * done / total, 1) if total else 100.0

            A.extract(path, _projects_dir, st["name"], progress=progress, cancelled=st["cancel"].is_set)
            st.update(phase="done", pct=100.0, finished_at=time.time())
            if _audit_log is not None:
                _audit_log("archive.import", report["session_id"], user=st["user"], role=st["role"],
                           detail=f"from={report['source_id']} files={report['files']} bytes={report['bytes']}")
            print(f"[archive] ✅ session '{report['session_id']}' imported from {st['filename']} "
                  f"({report['files']} files, {report['bytes'] / 1e9:.2f} GB)", flush=True)
        except A.ArchiveError as e:
            st.update(phase="failed", error=str(e))
        except Exception as e:  # noqa: BLE001 — declared in the state the interface polls
            import traceback
            traceback.print_exc()
            st.update(phase="failed", error=f"{type(e).__name__}: {e}")
        finally:
            try:
                path.unlink()                                      # (the ZIP is removed whatever the outcome)
            except OSError:
                pass

    asyncio.get_running_loop().run_in_executor(None, work)
    return {"ok": True, **_public(st)}


@router.get("/api/session_import/{job}")
def import_status(job: str, _user: Dict[str, Any] = Depends(_manager)):
    return {"ok": True, **_public(_job(job))}


@router.delete("/api/session_import/{job}")
def import_cancel(job: str, _user: Dict[str, Any] = Depends(_manager)):
    st = _job(job)
    st["cancel"].set()
    if st["phase"] == "uploading":
        _fail(st, "cancelled")
    return {"ok": True, **_public(st)}


def discard_leftovers() -> int:
    """At the server's start: what an upload or an extraction a restart cut short left beside the
    sessions (their state died with the process)."""
    if _projects_dir is None or not _projects_dir.is_dir():
        return 0
    import shutil
    n = 0
    inc = _projects_dir / ".incoming"
    for f in list(inc.glob("*.zip")) if inc.is_dir() else []:
        f.unlink()
        n += 1
    for d in _projects_dir.glob(".incoming-*"):
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n
