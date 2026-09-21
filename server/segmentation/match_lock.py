"""One matching per session state, across PROCESSES — never two.

USER 2026-09-21, after watching a full reconstruction: *"todo el pipeline debe
ser correcto, coherente y no ejecutarse nada de forma innecesaria — más tiempo,
más errores... nada de parches ni conformarnos con guardas que hacen que no se
rompa"*.

What happened that day: the pipeline's SAM3 worker was matching masks against
22.7 M points when the viewer opened the session. The viewer asked for the
scene, found no cached result, and launched a SECOND full matching — in the
API process, while the first ran in a `multiprocessing.spawn` worker. The
existing guard was a `threading.Lock`, which coordinates threads inside ONE
process and therefore saw nothing.

The cost was not hypothetical:

  * ~10 minutes of duplicated work over the whole cloud
  * three passes computing 121 / 113 / 119 instances and a fourth writing 125
    — the session's object count decided by arrival order
  * the viewer showing 119 while the file said 125
  * FOUR Potree octrees built over a cloud that never changed
  * a fusion round that only existed because a second matching ran
  * a record left naming a masklet another pass had already folded

None of it corrupted anything, because the fusion refuses and is idempotent.
That is precisely what this module exists to make unnecessary: the guards were
never the fix.

So: an OS-level lock on the session directory, honoured by every path that
matches — `_match_and_save_result` and `apply_segmentation_to_cloud` alike. A
second caller does not compete and does not queue up a redundant pass: it
WAITS for the one in flight and reads what that one wrote.
"""

from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

LOCK_NAME = ".matching.lock"


class MatchingBusy(RuntimeError):
    """Another process is matching this session and the caller would not wait."""


@contextmanager
def matching_lock(output_dir, *, timeout_s: Optional[float] = None,
                  log: Callable[[str], None] = print):
    """Hold the session's matching lock for the body.

    `timeout_s=None` waits as long as it takes (the right default: the other
    pass is producing exactly the result this caller wants). A finite timeout
    raises `MatchingBusy` so a request path can answer instead of hanging.

    The lock file carries the holder's pid and start time, so a stale one from
    a killed process is visible and can be reported rather than guessed at.
    """
    import fcntl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LOCK_NAME
    fh = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    t0 = time.time()
    waited = False
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if not waited:
                    holder = _holder(fh)
                    log(f"  matching already in flight{holder} — waiting for "
                        f"it instead of starting a second pass")
                    waited = True
                if timeout_s is not None and (time.time() - t0) > timeout_s:
                    raise MatchingBusy(
                        f"another process has been matching {output_dir.name} "
                        f"for {time.time() - t0:.0f} s{_holder(fh)}")
                time.sleep(0.5)
        os.ftruncate(fh, 0)
        os.write(fh, f"pid={os.getpid()} since={time.time():.0f}\n".encode())
        os.fsync(fh)
        if waited:
            log(f"  the other matching finished after {time.time() - t0:.1f} s")
        yield waited
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            os.close(fh)


def _holder(fh: int) -> str:
    try:
        os.lseek(fh, 0, os.SEEK_SET)
        txt = os.read(fh, 200).decode(errors="replace").strip()
        return f" ({txt})" if txt else ""
    except Exception:  # noqa: BLE001 — the label is a courtesy, never a gate
        return ""
