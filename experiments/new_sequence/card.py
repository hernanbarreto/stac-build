"""The card for work launched OUTSIDE the backend (the experiment's steps): the engine lease is taken — so neither the
chat's engine nor a pipeline job starts under it —, the voice and the chat's engine are stopped, and at the end the
lease is released and both are asked to come back. Nothing of the pipeline is changed: these are the backend's own
functions and routes (semantic.service, workers.base, /api/voice/stop)."""
import contextlib
import json
import ssl
import sys
import urllib.request

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)
BASE = "https://127.0.0.1:8765"
TOKEN = "new-sequence"


def _http(path: str, method: str = "GET", timeout: float = 90.0):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(BASE + path, method=method, data=(b"{}" if method == "POST" else None),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        body = r.read().decode("utf-8", "replace")
    try:
        return json.loads(body)
    except ValueError:
        return body


@contextlib.contextmanager
def card(label: str, session: str = "bufferStop_cubos", log=print):
    from semantic.service import JOB_PERMIT_OWNER, acquire_engine_lease, engine_lease_holder, release_engine_lease
    holder = engine_lease_holder()
    if holder is not None:
        raise RuntimeError(f"the card's lease is held by {holder.get('owner')} / {holder.get('stage')} "
                           f"(pid {holder.get('pid')}) — nothing is started")
    acquire_engine_lease(JOB_PERMIT_OWNER, stage=label, session=session, token=TOKEN)
    log(f"[card] the lease is this process's: {label}")
    try:
        try:
            _http("/api/voice/stop", "POST", timeout=60.0)
            log("[card] the voice was asked to stop")
        except Exception as e:  # noqa: BLE001 — declared: the backend may be down, the lease still decides
            log(f"[card] the voice route did not answer ({e})")
        from workers.base import stop_semantic_service_verified
        stop_semantic_service_verified(None, stage=label, log=log)
        yield
    finally:
        release_engine_lease(JOB_PERMIT_OWNER, TOKEN)
        log("[card] the lease is released")
        for path in ("/api/semantic/status?warmup=true", "/api/voice/status?start=true"):
            try:
                _http(path, timeout=30.0)
            except Exception as e:  # noqa: BLE001 — declared
                log(f"[card] {path} did not answer ({e}) — the chat / voice come back at the backend's next start")
