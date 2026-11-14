"""What the interface calls to speak with STAC.

    GET  /api/voice/status[?start=true]     up | loading | down | busy | disabled (+ the engines)
    POST /api/voice/transcribe?session_id=&scan=     body = the recorded audio
                                             → {text, language, probabilities, audio_s, seconds}
    POST /api/voice/speak                   {"text", "lang"} → audio/wav

The backend only relays: the engines live in the voice service's own process (voice/serve.py).
A transcription is given the system's name and the labels of the scan on screen as VOCABULARY
(the recogniser then writes "STAC" and "fire extinguisher", not "stack" and something close).
Never beside a job: while a pipeline holds the card the service is down and the answer is 'busy'.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import service, voice_config

router = APIRouter(prefix="/api/voice", tags=["voice"])
_VOCAB: Dict[str, Any] = {}
_VOCAB_LOCK = threading.Lock()


def _busy() -> Optional[str]:
    try:
        from phase5_qa.api import _engine_busy
        return _engine_busy()
    except Exception:  # noqa: BLE001 — no chat module (a test): nothing to wait for
        return None


def vocabulary(store_path: Optional[str], cfg: Dict[str, Any]) -> str:
    """The recogniser's vocabulary for the scan on screen: the system's name, then the scan's
    labels as people say them, the most frequent first (``voice.stt.prompt_max_labels``). Read
    once per store on disk."""
    name = cfg["name"]
    head = f"{name}, {name} Build."
    n = int(cfg["stt"]["prompt_max_labels"])
    if not store_path or n <= 0:
        return head
    import os
    try:
        st = os.stat(store_path)
    except OSError:
        return head
    key = (store_path, st.st_ino, st.st_size, st.st_mtime_ns, n, name)
    with _VOCAB_LOCK:
        if _VOCAB.get("key") == key:
            return _VOCAB["value"]
    try:
        from phase_r.instance_store import read_instances
        counts: Dict[str, int] = {}
        for inst in (read_instances(store_path) or {}).values():      # {id: {label, points}}
            label = str(inst.get("label") or "").replace("_", " ").strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
    except Exception as e:  # noqa: BLE001 — declared: the vocabulary is then the name alone
        print(f"[Voice] the scan's labels could not be read for the vocabulary ({e})")
        return head
    labels = [l for l, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]
    value = head + (" " + ", ".join(labels) + "." if labels else "")
    with _VOCAB_LOCK:
        _VOCAB.update(key=key, value=value)
    return value


def _post(path: str, data: bytes, content_type: str, timeout_s: float):
    req = urllib.request.Request(voice_config.base_url() + path, data, {"Content-Type": content_type})
    return urllib.request.urlopen(req, timeout=timeout_s)


def _relay_error(e: Exception) -> JSONResponse:
    if isinstance(e, urllib.error.HTTPError):
        try:
            return JSONResponse(json.loads(e.read()), status_code=e.code)
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": f"voice service: HTTP {e.code}"}, status_code=502)
    return JSONResponse({"status": "down", "error": f"the voice service does not answer ({e})"},
                        status_code=503)


@router.get("/status")
async def voice_status(start: bool = False):
    import asyncio

    def _do():
        st = service.status()
        if st["status"] in ("down",) and start and not _busy():
            service.start()
            st = service.status()
        if st["status"] != "disabled" and _busy():
            st = {"status": "busy", "detail": _busy()}
        return st
    return JSONResponse(await asyncio.get_running_loop().run_in_executor(None, _do))


@router.post("/transcribe")
async def voice_transcribe(request: Request, session_id: Optional[str] = None, scan: Optional[str] = None):
    import asyncio
    cfg = voice_config.load()
    if not cfg["enabled"]:
        return JSONResponse({"status": "disabled", "error": "voice.enabled is false"}, status_code=503)
    busy = _busy()
    if busy:
        return JSONResponse({"status": "busy", "error": busy}, status_code=503)
    data = await request.body()
    ctype = request.headers.get("content-type") or "application/octet-stream"

    def _do():
        if not service.is_alive():
            service.start()
            return JSONResponse({"status": "loading", "error": "the voice engines are loading"}, status_code=503)
        store = None
        if session_id:
            try:
                from phase5_qa.api import _resolve_store
                store = _resolve_store({"session_id": session_id, "scan": scan})
            except Exception:  # noqa: BLE001 — no store: the vocabulary is the name alone
                store = None
        prompt = vocabulary(store, cfg)
        try:
            with _post("/transcribe?" + urllib.parse.urlencode({"prompt": prompt}), data, ctype,
                       cfg["service"]["request_timeout_s"]) as r:
                return JSONResponse(json.loads(r.read()))
        except Exception as e:  # noqa: BLE001 — relayed as what it is
            return _relay_error(e)
    return await asyncio.get_running_loop().run_in_executor(None, _do)


@router.post("/speak")
async def voice_speak(body: Dict[str, Any]):
    import asyncio
    cfg = voice_config.load()
    if not cfg["enabled"]:
        return JSONResponse({"status": "disabled", "error": "voice.enabled is false"}, status_code=503)
    busy = _busy()
    if busy:
        return JSONResponse({"status": "busy", "error": busy}, status_code=503)
    text, lang = str((body or {}).get("text") or ""), str((body or {}).get("lang") or "")
    if lang not in cfg["languages"]:
        return JSONResponse({"error": f"lang must be one of {cfg['languages']}"}, status_code=400)

    def _do():
        if not service.is_alive():
            service.start()
            return JSONResponse({"status": "loading", "error": "the voice engines are loading"}, status_code=503)
        try:
            with _post("/speak", json.dumps({"text": text, "lang": lang}).encode(), "application/json",
                       cfg["service"]["request_timeout_s"]) as r:
                headers = {k: r.headers[k] for k in ("X-Audio-Seconds", "X-Synthesis-Seconds", "X-Spoken-Text")
                           if r.headers.get(k)}
                return Response(r.read(), media_type="audio/wav", headers=headers)
        except Exception as e:  # noqa: BLE001
            return _relay_error(e)
    return await asyncio.get_running_loop().run_in_executor(None, _do)


@router.post("/stop")
async def voice_stop():
    """End the voice service (its CUDA context leaves the card) — for GPU work launched OUTSIDE
    the backend, whose exclusive-GPU check would find it. The pipeline manager and the server's
    GPU tasks do this by themselves; the next status?start=true starts it again."""
    import asyncio
    gone = await asyncio.get_running_loop().run_in_executor(
        None, lambda: service.stop("asked through the API"))
    return JSONResponse({"stopped": bool(gone), **service.status()})
