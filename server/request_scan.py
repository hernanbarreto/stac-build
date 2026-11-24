"""EACH DEVICE WORKS ON ITS OWN SCAN (USER 2026-11-24: "debe mantener el contexto y la conversación para cada
sesión de usuario que está trabajando, OJO con eso!!!" — he works from several devices at once).

What was shared, traced that day: a route that does not name its scan resolves the project's ACTIVE scan, a
field of project.json that any device rewrites (opening a project sets it to the composition reference, a tab
switch to that tab) — so with two devices on two scans of one project, the octree a viewer goes on fetching
(/potree/<session>/…, no scan in it), the segmentation list, the sheet … came from the other device's scan; and
the viewer socket kept ONE current session for the whole server (the last any device opened).

The page now says which scan it is on with every request — the header ``X-STAC-Scan: <session>|<scan>``
(URL-encoded), or ``stac_scan=<session>|<scan>`` in the query of what a browser loads by URL (an <img>, a
<video>) — and project_paths.resolve_session takes it when the route names no scan and the session is the one
the header names. Nothing else changes: a route that names its scan keeps it; a request with no header (a
worker, a tool, a test) resolves as before.
"""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from urllib.parse import parse_qs, unquote

HEADER = b"x-stac-scan"
QUERY = "stac_scan"
_CURRENT: contextvars.ContextVar[Optional[Tuple[str, str]]] = contextvars.ContextVar("stac_request_scan",
                                                                                     default=None)


def parse(value: Optional[str]) -> Optional[Tuple[str, str]]:
    """``"<session>|<scan>"`` (each URL-encoded) → (session, scan); None when it is not one. A scan is
    "<date>/<source>"; neither part may climb out of its folder."""
    if not value or "|" not in value:
        return None
    sid, key = (unquote(p).strip() for p in value.split("|", 1))
    parts = key.split("/")
    if not sid or len(parts) != 2 or any(not p or p in (".", "..") or "\\" in p for p in parts) \
            or "/" in sid or "\\" in sid or sid in (".", ".."):
        return None
    return sid, key


def set_scan(sid: str, key: str) -> contextvars.Token:
    return _CURRENT.set((sid, key))


def reset(token: contextvars.Token) -> None:
    _CURRENT.reset(token)


def scan_for(session_id: str) -> Optional[str]:
    """The scan the current request (or viewer connection) is on, when it is on ``session_id``."""
    v = _CURRENT.get()
    return v[1] if v and v[0] == session_id else None


class RequestScanMiddleware:
    """Pure ASGI (the route runs in the same task, so it sees what is set here): the header, else the query."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        said = None
        for k, v in scope.get("headers") or []:
            if k == HEADER:
                said = parse(v.decode("latin-1"))
                break
        if said is None:
            q = parse_qs((scope.get("query_string") or b"").decode("latin-1")).get(QUERY)
            said = parse(q[0]) if q else None
        if said is None:
            return await self.app(scope, receive, send)
        token = set_scan(*said)
        try:
            return await self.app(scope, receive, send)
        finally:
            reset(token)


class CopyContextExecutor(ThreadPoolExecutor):
    """The event loop's default executor: what a route hands to a thread (``run_in_executor``) runs in the
    route's context — the scan it is on included — as a sync route already does."""

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)
