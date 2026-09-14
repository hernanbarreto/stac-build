"""Cross-process exclusive lock over one session's artifacts.

The correction module already refuses a second operation on a session, but it
does so with an in-process dict: it only sees what the BACKEND runs. A
certification runs in a worker subprocess (and, when investigating, as a
standalone script), so the dict never knew about it and the UI stayed free to
rewrite the very files the run was producing.

pccr 2026-09-14: the floor-levelling endpoint rewrote ``segmentation_result.json``
while the certification's split was writing its own copy of the same file. The
result was a corrupt document that killed the run and the instance store. Atomic
writes (``atomic_io``) keep a reader from ever seeing a seam; they do not stop
two writers from racing. This does.

The lock is a file in the session's ``output/`` — the one thing every
participant shares regardless of which process it lives in. It records who
holds it and the OS pid, so a holder killed without releasing (``detené todo``,
a crash, a reboot) does not leave the session locked forever: the next
acquirer sees the pid is gone and takes it over, saying so.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

LOCK_NAME = ".session_lock.json"

_guard = threading.Lock()


class SessionBusy(RuntimeError):
    """Raised when the session is held by someone else."""

    def __init__(self, holder: dict):
        self.holder = holder
        super().__init__(
            f"session busy: {holder.get('label') or holder.get('owner')} "
            f"(pid {holder.get('pid')}, since {holder.get('started_at')})")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # alive, owned by another user
        return True
    except Exception:            # noqa: BLE001 — unreadable pid is treated as gone
        return False
    return True


def holder(output_dir: Path | str) -> Optional[dict]:
    """The live holder of the session, or None (a dead holder is not one)."""
    p = Path(output_dir) / LOCK_NAME
    try:
        info = json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except Exception:            # noqa: BLE001 — an unreadable lock is a dead lock
        return None
    if not _pid_alive(info.get("pid", -1)):
        return None
    return info


def acquire(output_dir: Path | str, label: str, owner: str = "") -> dict:
    """Take the session. Raises SessionBusy when a live holder exists."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / LOCK_NAME
    info = {"label": label, "owner": owner or "", "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    payload = json.dumps(info, indent=2).encode("utf-8")
    with _guard:
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            live = holder(output_dir)
            if live is not None:
                raise SessionBusy(live)
            # the recorded pid is gone: the previous holder died without
            # releasing, so the lock is stale and we take it over.
            stale = None
            try:
                stale = json.loads(p.read_text())
            except Exception:    # noqa: BLE001
                pass
            print(f"[SessionLock] stale lock taken over "
                  f"(dead pid {(stale or {}).get('pid')}, "
                  f"was '{(stale or {}).get('label')}')", flush=True)
            fd = os.open(str(p), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o666)
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    return info


def release(output_dir: Path | str) -> None:
    """Release the session if THIS process holds it."""
    p = Path(output_dir) / LOCK_NAME
    with _guard:
        try:
            info = json.loads(p.read_text())
        except FileNotFoundError:
            return
        except Exception:        # noqa: BLE001 — unreadable: drop it, it blocks everyone
            info = {}
        if info.get("pid") in (os.getpid(), None):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def session_lock(output_dir: Path | str, label: str, owner: str = ""):
    """Hold the session for the duration of the block."""
    acquire(output_dir, label, owner)
    try:
        yield
    finally:
        release(output_dir)
