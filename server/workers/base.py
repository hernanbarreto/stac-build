# STAC-Builder: Base Worker Protocol
# Defines the IPC message protocol and base helper used by all workers.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

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
    try:
        import os
        os.setsid()
    except Exception:
        pass
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
    import subprocess
    _say = (pipe.send_log if pipe is not None else (log or (lambda m, **k: None)))
    try:
        if subprocess.run(["pgrep", "-f", "vllm serve"],
                          capture_output=True).returncode != 0:
            return
        _say(f"[gpu] stopping vLLM semantic service — {stage or 'this stage'} "
             f"gets the whole GPU (it auto-restarts on next VLM use)")
        subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
        for _ in range(30):
            time.sleep(2)
            if subprocess.run(["pgrep", "-f", "vllm serve"],
                              capture_output=True).returncode != 0:
                break
        free = gpu_free_gb()
        if free is not None:
            _say(f"[gpu] vLLM stopped — {free:.0f} GB VRAM free")
    except Exception as e:  # noqa: BLE001
        _say(f"[gpu] could not stop vLLM ({e}) — continuing with shared GPU")


VLLM_PROCESS_PATTERN = "vllm serve"      # what stop_semantic_service kills and pgrep looks for


def vllm_pids() -> list:
    """PIDs of every process whose command line matches ``vllm serve`` (pgrep -f),
    [] when none. RuntimeError when pgrep itself cannot run — a check that could
    not look is not a check that found nothing."""
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", VLLM_PROCESS_PATTERN], capture_output=True,
                             text=True)
    except OSError as e:
        raise RuntimeError(f"cannot verify the GPU handover: pgrep failed to run ({e})") from e
    if out.returncode not in (0, 1):
        raise RuntimeError(f"cannot verify the GPU handover: pgrep -f '{VLLM_PROCESS_PATTERN}' "
                           f"exited {out.returncode} ({out.stderr.strip()})")
    return [int(p) for p in out.stdout.split() if p.strip().isdigit()]


def stop_semantic_service_verified(pipe: Optional["WorkerPipe"] = None, stage: str = "",
                                   log=None) -> dict:
    """:func:`stop_semantic_service`, then VERIFY no ``vllm serve`` process is left
    (stop_semantic_service swallows its own failures and, after its wait, logs
    "vLLM stopped" without looking again). Raises RuntimeError naming the PIDs
    still alive — a stage that needs the whole GPU must not start on a shared one.
    Returns the check for the caller's report: ``{"service_stopped": True,
    "check": "pgrep -f 'vllm serve'", "remaining_pids": [], "free_gb": float|None}``."""
    stop_semantic_service(pipe, stage=stage, log=log)
    left = vllm_pids()
    if left:
        raise RuntimeError(
            f"{stage or 'this stage'} needs the GPU without the semantic service, but "
            f"{len(left)} '{VLLM_PROCESS_PATTERN}' process(es) are still running after the "
            f"stop (PIDs {left}) — stop them (pkill -f '{VLLM_PROCESS_PATTERN}') and re-run")
    return {"service_stopped": True, "check": f"pgrep -f '{VLLM_PROCESS_PATTERN}'",
            "remaining_pids": [], "free_gb": gpu_free_gb()}


# ── Running one worker inside another (the reconstruction stage hosts the
# semantics and the precision core — USER 2026-09-28) ────────────────────────

def _kill_process_tree(process, timeout: float = 5.0) -> None:
    """SIGTERM the child's process group (it setsid()'d in run_worker_safe, so its
    own children — DA3, SAM3, the precision steps — share it), wait, SIGKILL what
    is left. Mirrors PipelineManager._kill_stage_tree."""
    import os
    import signal
    pid = process.pid
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        pgid = None
    try:
        if pgid is not None and pgid != os.getpgid(0):
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError):
        process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        try:
            if pgid is not None and pgid != os.getpgid(0):
                os.killpg(pgid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.join(timeout=timeout)


def run_stage_inline(pipe: "WorkerPipe", module_name: str, session_dir: str, config: dict, *,
                     label: str, stage: str = "reconstruction",
                     pct_range: tuple = (0.0, 100.0)) -> None:
    """Run another worker module's ``run(conn, session_dir, config)`` as a child
    process of THIS worker and relay what it says: logs verbatim (prefixed with
    ``label``), progress rescaled into ``pct_range`` of ``stage``, a cancel
    forwarded to the child and its whole process group killed. The child's
    ``error`` / failed ``done`` — or a child that dies without one — raise a
    RuntimeError naming the label and the reason (nothing fails silently). The
    child is a 'spawn' process like every pipeline stage (CUDA-safe)."""
    import importlib
    from multiprocessing import get_context

    mod = importlib.import_module(module_name)
    ctx = get_context("spawn")
    server_conn, worker_conn = ctx.Pipe()
    proc = ctx.Process(target=mod.run, args=(worker_conn, session_dir, config),
                       name=f"inline-{label}")
    proc.start()
    worker_conn.close()
    lo, hi = float(pct_range[0]), float(pct_range[1])
    error: Optional[dict] = None
    done: Optional[dict] = None
    try:
        while True:
            if pipe.check_cancel():
                try:
                    server_conn.send({"type": "cancel"})
                except Exception:  # noqa: BLE001 — the child may be gone already
                    pass
                _kill_process_tree(proc)
                raise RuntimeError(f"cancelled during {label}")
            if not server_conn.poll(0.25):
                if not proc.is_alive() and not server_conn.poll(0.0):
                    break
                continue
            try:
                msg = server_conn.recv()
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
        proc.join(timeout=10)
        if proc.is_alive():
            _kill_process_tree(proc)
        server_conn.close()
    if error is not None or done is None or not done.get("success", False):
        why = ((error or {}).get("msg") or (done or {}).get("detail")
               or f"the worker died without reporting (exit code {proc.exitcode})")
        raise RuntimeError(f"{label} failed: {why}")
