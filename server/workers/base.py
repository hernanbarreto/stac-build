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
