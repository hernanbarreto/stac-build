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


def stop_semantic_service(pipe: "WorkerPipe", stage: str = "") -> None:
    """EXCLUSIVE GPU for a heavy stage: stop the vLLM semantic service (its ~40 GB
    resident VRAM starves Omega single passes and long SAM3 sessions). Any later
    consumer auto-restarts it (_ensure_semantic_service), so this is a stage-scoped
    handover, not a shutdown. No-op when vLLM isn't running."""
    import subprocess
    try:
        if subprocess.run(["pgrep", "-f", "vllm serve"],
                          capture_output=True).returncode != 0:
            return
        pipe.send_log(f"[gpu] stopping vLLM semantic service — {stage or 'this stage'} "
                      f"gets the whole GPU (it auto-restarts on next VLM use)")
        subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
        for _ in range(30):
            time.sleep(2)
            if subprocess.run(["pgrep", "-f", "vllm serve"],
                              capture_output=True).returncode != 0:
                break
        free = gpu_free_gb()
        if free is not None:
            pipe.send_log(f"[gpu] vLLM stopped — {free:.0f} GB VRAM free")
    except Exception as e:  # noqa: BLE001
        pipe.send_log(f"[gpu] could not stop vLLM ({e}) — continuing with shared GPU",
                      level="warning")
