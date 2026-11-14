"""The voice service's life, from the backend: started with it, stopped while a job holds the
card, started again when the card frees (USER 2026-11-14: "tanto el motor de voz y habla del
sistema como el chat vlm deberan iniciar cuando levante el backend porque son el core del
funcionamiento de stac build").

The service holds a CUDA context: a compute process on the card that every exclusive-GPU check
of a stage would refuse — so it ENDS, verified, when the engine lease is taken (the pipeline
manager and the server's GPU tasks call :func:`stop`), exactly as the chat's tools process does.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import voice_config

REPO = Path(__file__).resolve().parent.parent.parent
LAUNCHER = REPO / "scripts" / "serve_voice.sh"
LOG = REPO / "logs" / "voice_latest.log"
_STATE: Dict[str, Any] = {"proc": None, "log": None}
_LOCK = threading.Lock()


def health(timeout_s: float = 2.0) -> Optional[Dict[str, Any]]:
    """The service's own /health, or None when it does not answer."""
    try:
        with urllib.request.urlopen(voice_config.base_url() + "/health", timeout=timeout_s) as r:
            doc = json.loads(r.read())
        return doc if isinstance(doc, dict) and doc.get("status") == "ok" else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def is_alive(timeout_s: float = 2.0) -> bool:
    return health(timeout_s) is not None


def is_starting() -> bool:
    """A launcher of this backend is running and the service does not answer yet."""
    p = _STATE["proc"]
    return p is not None and p.poll() is None and not is_alive(1.0)


def _die_with_parent() -> None:
    """The child dies with the backend (a hard death of the backend included)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)      # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 — the shutdown hook stops it too
        pass
    os.setsid()                                   # its own group: stop() ends the whole tree


def _busy() -> Optional[str]:
    """Why the voice may not be on the card now: a job (or a GPU task of the server) holds the
    engine lease. None when it may."""
    try:
        from semantic.service import engine_available_to
        ok, why = engine_available_to(owner="voice")
        return None if ok else why
    except Exception:  # noqa: BLE001 — no lease module (a test): nothing holds the card
        return None


def start(log: Callable[[str], Any] = print) -> bool:
    """Launch the service unless it is disabled, alive, being launched, or the card is a job's.
    Returns True when it is alive or was launched (it loads in the background, ~40 s)."""
    cfg = voice_config.load()
    if not cfg["enabled"]:
        return False
    with _LOCK:
        if is_alive():
            return True
        p = _STATE["proc"]
        if p is not None and p.poll() is None:
            return True                                           # loading
        why = _busy()
        if why:
            log(f"[Voice] not started — {why}")
            return False
        if not LAUNCHER.exists():
            log(f"[Voice] the launcher is missing ({LAUNCHER})")
            return False
        LOG.parent.mkdir(parents=True, exist_ok=True)
        logf = open(LOG, "w")
        try:
            proc = subprocess.Popen(["bash", str(LAUNCHER)], cwd=str(REPO), stdout=logf,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    preexec_fn=_die_with_parent)
        except Exception as e:  # noqa: BLE001 — declared
            logf.close()
            log(f"[Voice] could not launch the service: {e}")
            return False
        _STATE.update(proc=proc, log=logf)
        log(f"[Voice] starting (pid {proc.pid}; log {LOG.name})")
        return True


def wait_alive(timeout_s: Optional[float] = None) -> bool:
    cfg = voice_config.load()
    deadline = time.time() + (cfg["service"]["startup_timeout_s"] if timeout_s is None else timeout_s)
    while time.time() < deadline:
        if is_alive():
            return True
        p = _STATE["proc"]
        if p is None or p.poll() is not None:
            return is_alive()                                     # the launcher ended: up or dead
        time.sleep(1.0)
    return False


def stop(reason: str, log: Callable[[str], Any] = print, wait_s: float = 15.0) -> bool:
    """End the service and VERIFY it is gone (its CUDA context leaves the card). True when no
    service of this backend is left running."""
    with _LOCK:
        p = _STATE["proc"]
        if p is None or p.poll() is not None:
            _close_log()
            _STATE["proc"] = None
            return True
        log(f"[Voice] stopping (pid {p.pid}) — {reason}")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(p.pid, sig)                              # the launcher's whole group
            except ProcessLookupError:
                break
            except Exception as e:  # noqa: BLE001
                log(f"[Voice] signal {sig.name} failed: {e}")
            try:
                p.wait(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        gone = p.poll() is not None
        if gone:
            _close_log()
            _STATE["proc"] = None
        else:
            log(f"[Voice] the service (pid {p.pid}) did NOT end")
        return gone


def _close_log() -> None:
    f = _STATE.get("log")
    if f is not None:
        try:
            f.close()
        except Exception:  # noqa: BLE001
            pass
        _STATE["log"] = None


def status() -> Dict[str, Any]:
    cfg = voice_config.load()
    if not cfg["enabled"]:
        return {"status": "disabled"}
    h = health()
    if h is not None:
        return {**h, "status": "up"}
    why = _busy()
    if why:
        return {"status": "busy", "detail": why}
    return {"status": "loading" if is_starting() else "down"}


def log_tail(n: int = 12) -> str:
    try:
        return "\n".join(LOG.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no voice log)"
