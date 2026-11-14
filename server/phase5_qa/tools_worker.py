"""The chat's measurements in their OWN process (USER 2026-11-13: "el vlm y todas sus
herramientas deben funcionar en GPU").

WHY A PROCESS: the backend never touches the card. A CUDA context in the server process is a
compute process on the GPU for the backend's whole life, and every GPU step after it refuses
the card (``repro.require_exclusive_gpu`` — found 2026-10-07, see
``semantic.service.release_cached_vram``). So with ``semantic.chat.tools_device: cuda`` the
tools that read an object's points (``phase5_qa.tools.POINT_TOOLS``) run here: ONE child per
store, spawned the first time the chat needs it, keeping the instance store open and the points
it has measured on the device between questions (the pool of every instance is read from the
disk once, not once per question).

It ENDS — joined, its context gone, the card free:
  * when a pipeline job or a server GPU task takes the engine lease (``stop`` is called by
    pipeline_manager and main right after the lease is theirs; a question in flight fails,
    declared, and nothing is started while the lease is held);
  * when a question is about another store, or the scan's segmentation was written again;
  * with the backend: PR_SET_PDEATHSIG for a hard death, the lifespan shutdown for a clean one.

Protocol over a multiprocessing Pipe (spawn): {"type": "call", "tool", "kwargs"} →
{"type": "result", "result"} | {"type": "error", "error"}; {"type": "stop"}.

This module imports nothing heavy: the child sets its environment (the stage threads and the
deterministic variables of ``precision.runner.step_env``) BEFORE numpy and torch load.
"""
from __future__ import annotations

import os
import threading
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, Optional

CHAT_OWNER = "chat"                    # the chat's name at the engine lease (phase5_qa.api)
_POLL_S = 0.25


class ToolsWorkerError(RuntimeError):
    """The chat's tools process could not run the tool (not started, ended, bound exceeded) or
    the tool itself raised."""


class ToolsWorkerBusy(ToolsWorkerError):
    """The card is someone else's now: the tools process is not started beside a job."""


# ── the child ───────────────────────────────────────────────────────────────────────────────
def _serve(conn, store_path: str, device: str, env: Dict[str, str], seed: int) -> None:
    try:                                # die with the backend, however it dies
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    except Exception:  # noqa: BLE001 — Linux only; the daemon flag and the shutdown remain
        pass
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        t0 = time.time()
        from phase_r.instance_store import InstanceStore
        from phase5_qa.tools import POINT_TOOLS, SpatialTools
        if device != "numpy":
            import repro
            repro.enable_deterministic_torch(int(seed))          # STRICT: raise, never warn
        store = InstanceStore(store_path)
        tools = SpatialTools(store, device=device)
        impls = {k: v for k, v in tools._impls_raw().items() if k in POINT_TOOLS}
        if device == "cuda":
            import torch
            torch.zeros(1, device="cuda")                        # the context, before a question
        conn.send({"type": "ready", "pid": os.getpid(), "device": device,
                   "startup_s": round(time.time() - t0, 2)})
    except Exception as e:  # noqa: BLE001 — told to the parent, which raises it
        try:
            conn.send({"type": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if not isinstance(msg, dict) or msg.get("type") == "stop":
            break
        if msg.get("type") != "call":
            continue
        try:
            tools._inst_cache = None        # labels are read again: a rename is seen at once
            fn = impls.get(msg.get("tool"))
            if fn is None:
                raise KeyError(f"'{msg.get('tool')}' is not a point tool")
            out = {"type": "result", "result": fn(**(msg.get("kwargs") or {}))}
        except Exception as e:  # noqa: BLE001 — the tool's own failure, reported as its result
            out = {"type": "error", "error": f"{type(e).__name__}: {e}"}
        try:
            conn.send(out)
        except (EOFError, OSError):
            break
    try:
        store.close()
    except Exception:  # noqa: BLE001
        pass


# ── the parent ──────────────────────────────────────────────────────────────────────────────
def store_signature(store_path: str) -> tuple:
    """What tells one segmentation of a scan from the next: the store's path and the size and
    time of the scan's segmentation result (written by the projection and by every edit of
    the instances). The store's own file is not read for this — the chat writes notes and
    volumes into it."""
    p = Path(store_path)
    res = p.parent / "segmentation_result.json"
    try:
        st = res.stat()
        return (str(p), int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return (str(p), None, None)


def _settings() -> Dict[str, Any]:
    """The worker's bound, threads and seed from config.yaml — every one an existing key:
    ``semantic.service.request_timeout_s`` (the bound of one chat request: the bound of one
    tool call and of the start), ``reconstruction.precision.runner.threads`` (the fixed CPU
    threads of every stage) and ``semantic.generation.seed``. A missing key fails naming it."""
    import yaml
    from semantic import semantic_config as _sc
    with open(_sc._CONFIG_PATH, "r") as f:
        full = yaml.safe_load(f) or {}
    sem = full.get("semantic") or {}
    timeout = (sem.get("service") or {}).get("request_timeout_s")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ToolsWorkerError("config.yaml: semantic.service.request_timeout_s is missing or not positive")
    threads = (((full.get("reconstruction") or {}).get("precision") or {}).get("runner") or {}).get("threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ToolsWorkerError("config.yaml: reconstruction.precision.runner.threads is missing "
                               "or not a positive integer")
    seed = (sem.get("generation") or {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ToolsWorkerError("config.yaml: semantic.generation.seed is missing or not an integer")
    return {"timeout_s": float(timeout), "threads": int(threads), "seed": int(seed)}


class _Worker:
    def __init__(self, store_path: str, device: str, signature: tuple,
                 log: Callable[[str], Any]):
        cfg = _settings()
        from precision.runner import step_env
        full = step_env(cfg["threads"], base=os.environ)
        env = {k: v for k, v in full.items() if os.environ.get(k) != v}
        self.store_path, self.device, self.signature = str(store_path), device, signature
        self.timeout_s = cfg["timeout_s"]
        self._io = threading.Lock()
        ctx = get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, name="stac-chat-tools", daemon=True,
                                 args=(child, self.store_path, device, env, cfg["seed"]))
        t0 = time.time()
        self._proc.start()
        child.close()
        msg = self._wait(self.timeout_s, "start")
        if msg.get("type") != "ready":
            self.stop()
            raise ToolsWorkerError(f"the chat's tools process did not start: {msg.get('error')}")
        self.pid = int(msg["pid"])
        log(f"[ChatTools] tools process up on {device} in {time.time() - t0:.1f} s "
            f"(pid {self.pid}, store {self.store_path})")

    def alive(self) -> bool:
        return self._proc.is_alive()

    def _wait(self, timeout_s: float, what: str) -> dict:
        t0 = time.time()
        while True:
            try:
                if self._conn.poll(_POLL_S):
                    return self._conn.recv()
            except (EOFError, OSError):
                raise ToolsWorkerError(f"the chat's tools process ended during the {what} — "
                                       f"a job took the GPU, or it failed (the backend's log)")
            if not self._proc.is_alive():
                raise ToolsWorkerError(f"the chat's tools process ended during the {what} — "
                                       f"a job took the GPU, or it failed (the backend's log)")
            if time.time() - t0 > timeout_s:
                self.stop()
                raise ToolsWorkerError(f"the chat's tools process did not answer the {what} "
                                       f"within semantic.service.request_timeout_s = "
                                       f"{timeout_s:.0f} s")

    def call(self, tool: str, kwargs: dict) -> Any:
        with self._io:
            if not self.alive():
                raise ToolsWorkerError("the chat's tools process is not running")
            try:
                self._conn.send({"type": "call", "tool": tool, "kwargs": kwargs})
            except (EOFError, OSError, ValueError) as e:
                raise ToolsWorkerError(f"the chat's tools process is gone ({e})")
            msg = self._wait(self.timeout_s, f"tool {tool}")
        if msg.get("type") == "error":
            raise ToolsWorkerError(str(msg.get("error")))
        return msg.get("result")

    def stop(self) -> bool:
        """End the process — verified: True when it is gone."""
        if self._proc.is_alive() and self._io.acquire(blocking=False):
            try:                                     # idle: asked to leave
                self._conn.send({"type": "stop"})
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._io.release()
            self._proc.join(3.0)
        if self._proc.is_alive():                    # measuring, or it did not leave: killed
            self._proc.terminate()
            self._proc.join(5.0)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(5.0)
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        return not self._proc.is_alive()


_LOCK = threading.RLock()
_WORKER: Optional[_Worker] = None


def _lease_refusal() -> Optional[str]:
    """Why the tools process may not start or measure now: someone other than the chat holds
    the engine lease (a job, a server GPU task) — the card is theirs."""
    try:
        from semantic.service import engine_available_to
        ok, why = engine_available_to(owner=CHAT_OWNER)
    except Exception:  # noqa: BLE001 — no lease machinery (a test, a CLI): nobody to wait for
        return None
    return None if ok else str(why)


def call(store_path: str, tool: str, kwargs: dict, *, device: str = "cuda",
         log: Callable[[str], Any] = print) -> Any:
    """One point tool on ``store_path`` in the chat's tools process, started when needed.
    Raises :class:`ToolsWorkerBusy` while the engine lease is someone else's."""
    global _WORKER
    sig = store_signature(store_path)
    with _LOCK:                                      # the check and the start are one step
        why = _lease_refusal()
        if why:
            raise ToolsWorkerBusy(f"the GPU is in use — {why}")
        w = _WORKER
        if w is not None and (not w.alive() or w.signature != sig or w.device != device):
            w.stop()
            _WORKER = w = None
        if w is None:
            w = _WORKER = _Worker(store_path, device, sig, log)
    return w.call(tool, kwargs)


def warm(store_path: str, *, device: str = "cuda", log: Callable[[str], Any] = print) -> bool:
    """Start the tools process of this store ahead of the first question (the chat was opened).
    False while the card is someone else's."""
    global _WORKER
    sig = store_signature(store_path)
    with _LOCK:
        if _lease_refusal():
            return False
        w = _WORKER
        if w is not None and (not w.alive() or w.signature != sig or w.device != device):
            w.stop()
            _WORKER = w = None
        if w is None:
            _WORKER = _Worker(store_path, device, sig, log)
    return True


def stop(reason: str = "", log: Callable[[str], Any] = print) -> bool:
    """End the tools process if there is one (the card goes to whoever asked). True when no
    process of it is left."""
    global _WORKER
    with _LOCK:
        w, _WORKER = _WORKER, None
        if w is None:
            return True
        gone = w.stop()
        log(f"[ChatTools] tools process stopped ({reason or 'n/a'})"
            + ("" if gone else " — STILL ALIVE"))
        return gone


def status() -> dict:
    with _LOCK:
        w = _WORKER
        if w is None or not w.alive():
            return {"running": False}
        return {"running": True, "pid": w.pid, "device": w.device, "store": w.store_path}
