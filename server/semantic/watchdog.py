"""vLLM dies with the backend (USER 2026-10-06: "todo lo que haya debe morir", the VLM included).

`semantic.serve` exec()s into `vllm serve`, so no thread of ours lives inside it, and
PR_SET_PDEATHSIG fires on the death of the THREAD that launched it and reaches only the
launcher. This small process watches the BACKEND's pid; the moment the backend is gone it
kills the whole vLLM session (launcher → semantic.serve → vllm serve → EngineCore) —
SIGTERM, then SIGKILL — and exits. It also exits by itself when vLLM was stopped normally.

    python -m semantic.watchdog <backend_pid> <vllm_session_pgid>
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except OSError:
        return False


def watch(backend_pid: int, pgid: int, poll_s: float = 2.0, grace_s: float = 10.0) -> str:
    while True:
        if not _group_alive(pgid):
            return "vllm gone"
        if not _alive(backend_pid):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                return "vllm gone"
            t0 = time.time()
            while _group_alive(pgid) and time.time() - t0 < grace_s:
                time.sleep(0.5)
            if _group_alive(pgid):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
            return "backend gone — vllm killed"
        time.sleep(poll_s)


if __name__ == "__main__":
    sys.exit(0 if watch(int(sys.argv[1]), int(sys.argv[2])) else 1)
