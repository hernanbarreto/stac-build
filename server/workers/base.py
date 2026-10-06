# STAC-Builder: Base Worker Protocol
# Defines the IPC message protocol and base helper used by all workers.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import signal
import time
import traceback
from multiprocessing.connection import Connection
from typing import Optional


class WorkerPipe:
    """Helper wrapper around a multiprocessing.Pipe connection.
    
    Provides typed send methods for the IPC protocol and
    a check_cancel() method workers should call periodically.
    """

    def __init__(self, conn: Connection):
        self._conn = conn
        self._cancelled = False

    # ── Send helpers (Worker → Server) ───────────────────────

    def send_progress(self, pct: float, msg: str, stage: str = ""):
        """Send a progress update.  pct is 0-100."""
        self._conn.send({
            "type": "progress",
            "stage": stage,
            "pct": round(pct, 1),
            "msg": msg,
        })

    def send_log(self, msg: str, level: str = "info"):
        self._conn.send({"type": "log", "level": level, "msg": msg})

    def send_done(self, success: bool, elapsed: float, detail: str = ""):
        self._conn.send({
            "type": "done",
            "success": success,
            "elapsed": round(elapsed, 2),
            "detail": detail,
        })

    def send_error(self, msg: str, tb: Optional[str] = None):
        self._conn.send({
            "type": "error",
            "msg": msg,
            "traceback": tb or "",
        })

    # ── Receive helpers (Server → Worker) ────────────────────

    def check_cancel(self) -> bool:
        """Non-blocking check if server sent a cancel signal."""
        if self._cancelled:
            return True
        while self._conn.poll():
            try:
                msg = self._conn.recv()
                if isinstance(msg, dict) and msg.get("type") == "cancel":
                    self._cancelled = True
                    return True
            except (EOFError, OSError):
                self._cancelled = True
                return True
        return False

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def die_with_parent(poll_s: float = 2.0) -> None:
    """USER 2026-10-06: "cuando mato el proceso debe matar todo lo que tenía corriendo,
    omega, da3, vlm, sam3, pointdit, todo". A daemon thread watches the parent; the
    moment it is gone (the server restarted, killed, crashed) the WHOLE process group
    of this stage dies with SIGKILL — the worker, the inline child, Omega's
    vggt_long.py, DA3, SAM3, the precision steps, PointDiT: everything launched under
    it. PR_SET_PDEATHSIG alone kills only the direct child, and the stage tree used to
    survive a backend restart (pccr 2408: an orphan Omega kept 58 GB of RAM and the
    CPU for an hour; zaragoza: an orphan SAM3 held 21 GB of VRAM for four hours).
    Deliberately persistent services (vLLM) run in their own session and are not
    touched."""
    import os
    import signal
    import threading
    ppid = os.getppid()

    def _watch():
        while True:
            time.sleep(poll_s)
            if os.getppid() != ppid:
                try:
                    os.killpg(os.getpgid(0), signal.SIGKILL)
                except OSError:
                    os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=_watch, name="die-with-parent", daemon=True).start()


def run_worker_safe(worker_fn, conn: Connection, *args, **kwargs):
    """Execute a worker function with standard error handling and cleanup.

    This is the top-level entry point called as a multiprocessing target.
    It wraps the actual worker logic in try/except, sends done/error
    messages, and ensures the pipe is always closed.
    """
    # Become our own process-group leader so a pipeline cancel can kill the
    # WHOLE stage subtree (bash scripts, DA3/CloudCompy children) with one
    # killpg — a plain terminate() on the worker orphaned its children, which
    # kept running and holding GPU memory. Deliberately-persistent services
    # (vLLM) detach with start_new_session and are unaffected.
    # A worker HOSTED by another worker (workers/inline_child.py, env
    # STAC_INLINE_CHILD) stays in its host's group, so that same killpg reaches it.
    try:
        import os
        if not os.environ.get("STAC_INLINE_CHILD"):
            os.setsid()
    except Exception:
        pass
    die_with_parent()          # the stage tree never outlives the server (USER 2026-10-06)
    pipe = WorkerPipe(conn)
    t0 = time.time()
    try:
        worker_fn(pipe, *args, **kwargs)
        pipe.send_done(success=True, elapsed=time.time() - t0)
    except Exception as exc:
        pipe.send_error(str(exc), traceback.format_exc())
        pipe.send_done(success=False, elapsed=time.time() - t0, detail=str(exc))
    finally:
        pipe.close()
        # Cleanup GPU if torch available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass  # CUDA not available or not initialized in this process
        import gc
        gc.collect()


# ── Exclusive-GPU helpers (shared by reconstruction / SAM3 workers) ──────────

def gpu_total_gb() -> Optional[float]:
    """TOTAL VRAM (GB) of GPU 0 via nvidia-smi — a property of the card, unlike the
    free memory; None when it can't be read."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def gpu_free_gb() -> Optional[float]:
    """Free VRAM (GB) of GPU 0 via nvidia-smi; None when it can't be read."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def stop_semantic_service(pipe: Optional["WorkerPipe"] = None, stage: str = "",
                          log=None) -> None:
    """EXCLUSIVE GPU for a heavy stage: stop the vLLM semantic service (its ~40 GB
    resident VRAM starves Omega single passes and long SAM3 sessions). Any later
    consumer restarts it (semantic.service.ensure_service — the VLM worker AND the
    spatial-Q&A route), so this is a stage-scoped handover, not a shutdown. No-op
    when vLLM isn't running.

    ``log`` is for callers that are not workers and have no pipe — the epoch
    transaction's re-consolidation is one (correction/apply.py).
    """
    _say = (pipe.send_log if pipe is not None else (log or (lambda m, **k: None)))
    try:
        if not vllm_pids():
            return
        _say(f"[gpu] stopping vLLM semantic service — {stage or 'this stage'} "
             f"gets the whole GPU (it auto-restarts on next VLM use)")
        # the launcher chain (serve_semantic.sh → semantic.serve → vllm serve) is killed as a
        # whole and the kill is REPEATED until nothing is left: a vLLM still starting turns into
        # a new 'vllm serve' process right after the first signal (zaragoza 2026-10-05, launched
        # one second after the backend came up)
        for attempt in range(6):
            kill_vllm_pids(signal.SIGTERM if attempt < 3 else signal.SIGKILL)
            for _ in range(10):
                time.sleep(2)
                if not vllm_pids():
                    break
            if not vllm_pids():
                break
        free = gpu_free_gb()
        if free is not None:
            _say(f"[gpu] vLLM stopped — {free:.0f} GB VRAM free")
    except Exception as e:  # noqa: BLE001
        _say(f"[gpu] could not stop vLLM ({e}) — continuing with shared GPU")


VLLM_PROCESS_PATTERN = "vllm serve"      # what stop_semantic_service kills and pgrep looks for
# the whole launcher chain, each ANCHORED to the start of the command line so a shell whose
# command text merely mentions these words (a terminal, a tool, a test) is never matched:
#   bash /…/scripts/serve_semantic.sh  →  /…/envs/semantic/bin/python -m semantic.serve
#   →  /…/envs/semantic/bin/python /…/bin/vllm serve …  →  its engine (proctitle VLLM::EngineCore)
VLLM_PATTERNS = (r"^\S+/python[0-9.]* \S+/bin/vllm serve",
                 r"^\S+/python[0-9.]* -m semantic\.serve",
                 r"^(\S*/)?bash \S*/scripts/serve_semantic\.sh",
                 r"^VLLM::")


def kill_vllm_pids(sig) -> list:
    """Signal every vLLM process by PID (the `vllm serve` command lines pgrep finds), never
    the backend or any shell that merely mentions the pattern. Returns the PIDs signalled."""
    import os
    pids = [p for p in vllm_pids() if p != os.getpid()]
    for p in pids:
        try:
            os.kill(p, sig)
        except ProcessLookupError:
            pass
    return pids


def vllm_pids() -> list:
    """PIDs of every process whose command line matches ``vllm serve`` (pgrep -f),
    [] when none. RuntimeError when pgrep itself cannot run — a check that could
    not look is not a check that found nothing."""
    import os
    import subprocess
    pids = set()
    for pat in VLLM_PATTERNS:
        try:
            out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        except OSError as e:
            raise RuntimeError(f"cannot verify the GPU handover: pgrep failed to run ({e})") from e
        if out.returncode not in (0, 1):
            raise RuntimeError(f"cannot verify the GPU handover: pgrep -f '{pat}' "
                               f"exited {out.returncode} ({out.stderr.strip()})")
        pids.update(int(p) for p in out.stdout.split() if p.strip().isdigit())
    pids.discard(os.getpid())
    return sorted(pids)


def stop_semantic_service_verified(pipe: Optional["WorkerPipe"] = None, stage: str = "",
                                   log=None) -> dict:
    """:func:`stop_semantic_service`, then VERIFY no ``vllm serve`` process is left
    (stop_semantic_service swallows its own failures and, after its wait, logs
    "vLLM stopped" without looking again). Raises RuntimeError naming the PIDs
    still alive — a stage that needs the whole GPU must not start on a shared one.
    Returns the check for the caller's report: ``{"service_stopped": True,
    "check": "pgrep -f 'vllm serve'", "remaining_pids": [], "free_gb": float|None}``."""
    # the backend kicks vLLM at its own boot through an HTTP probe that takes seconds: a pipeline
    # launched right after a restart stops nothing, then the launcher appears (zaragoza 2026-10-05,
    # twice) — so stop + verify is REPEATED until a verification finds nothing
    left = []
    for _ in range(6):
        stop_semantic_service(pipe, stage=stage, log=log)
        left = vllm_pids()
        if not left:
            break
        time.sleep(2)
    if left:
        raise RuntimeError(
            f"{stage or 'this stage'} needs the GPU without the semantic service, but "
            f"{len(left)} '{VLLM_PROCESS_PATTERN}' process(es) are still running after the "
            f"stop (PIDs {left}) — stop them (pkill -f '{VLLM_PROCESS_PATTERN}') and re-run")
    return {"service_stopped": True, "check": f"pgrep -f '{VLLM_PROCESS_PATTERN}'",
            "remaining_pids": [], "free_gb": gpu_free_gb()}


# ── Running one worker inside another (the reconstruction stage hosts the
# semantics and the precision core — USER 2026-09-28) ────────────────────────

def _terminate(proc, timeout: float = 5.0) -> None:
    """SIGTERM the hosted child, wait, SIGKILL what is left. The child shares the
    host's process group (no setsid, STAC_INLINE_CHILD), so the pipeline's own
    killpg on a cancelled stage reaches it and its subprocesses as well."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except Exception:  # noqa: BLE001 — subprocess.TimeoutExpired
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            pass


def run_stage_inline(pipe: "WorkerPipe", module_name: str, session_dir: str, config: dict, *,
                     label: str, stage: str = "reconstruction",
                     pct_range: tuple = (0.0, 100.0), cancel_grace_s: float = 30.0) -> None:
    """Run another worker module's ``run(conn, session_dir, config)`` as a child of
    THIS worker and relay what it says: logs verbatim (prefixed with ``label``),
    progress rescaled into ``pct_range`` of ``stage``, a cancel forwarded to the
    child (which stops its own subprocesses and reports) and enforced after
    ``cancel_grace_s``. The child's ``error`` / failed ``done`` — or a child that
    dies without one — raise a RuntimeError naming the label and the reason.

    The child is a plain subprocess (``python -m workers.inline_child``) speaking
    the stage protocol over a multiprocessing Pipe handed down by fd: the pipeline's
    stage workers are daemonic multiprocessing processes and Python forbids those
    to spawn multiprocessing children (pccr 2026-09-28: "daemonic processes are not
    allowed to have children" was the first thing the hosted VLM stage said)."""
    import os
    import subprocess
    import sys
    from multiprocessing import Pipe
    from pathlib import Path

    server_dir = Path(__file__).resolve().parent.parent
    host_conn, child_conn = Pipe()
    fd = child_conn.fileno()
    env = dict(os.environ)
    env["STAC_INLINE_CHILD"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "workers.inline_child", module_name, session_dir, str(fd)],
        cwd=str(server_dir), env=env, pass_fds=(fd,))
    child_conn.close()
    host_conn.send(config)
    lo, hi = float(pct_range[0]), float(pct_range[1])
    error: Optional[dict] = None
    done: Optional[dict] = None
    cancel_sent_at: Optional[float] = None
    try:
        while True:
            if cancel_sent_at is None and pipe.check_cancel():
                try:
                    host_conn.send({"type": "cancel"})
                except Exception:  # noqa: BLE001 — the child may be gone already
                    pass
                cancel_sent_at = time.time()
            if cancel_sent_at is not None and time.time() - cancel_sent_at > cancel_grace_s:
                _terminate(proc)
                raise RuntimeError(f"cancelled during {label}")
            if not host_conn.poll(0.25):
                if proc.poll() is not None and not host_conn.poll(0.0):
                    break
                continue
            try:
                msg = host_conn.recv()
            except EOFError:
                break
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "log":
                pipe.send_log(f"[{label}] {msg.get('msg', '')}", level=msg.get("level", "info"))
            elif kind == "progress":
                frac = max(0.0, min(100.0, float(msg.get("pct", 0.0)))) / 100.0
                pipe.send_progress(lo + (hi - lo) * frac, f"{label}: {msg.get('msg', '')}",
                                   stage=stage)
            elif kind == "error":
                error = msg
                if msg.get("traceback"):
                    pipe.send_log(str(msg["traceback"]), level="error")
            elif kind == "done":
                done = msg
                break
    finally:
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            _terminate(proc)
        host_conn.close()
    if cancel_sent_at is not None:
        raise RuntimeError(f"cancelled during {label}")
    if error is not None or done is None or not done.get("success", False):
        why = ((error or {}).get("msg") or (done or {}).get("detail")
               or f"the worker died without reporting (exit code {proc.returncode})")
        raise RuntimeError(f"{label} failed: {why}")
