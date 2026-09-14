"""Atomic file writes for the session artifacts.

Every artifact under ``output/`` is read by other processes while it is being
written: the backend serves the viewer from the same files a worker or a
standalone run is producing. Writing straight onto the live path — the
``open(p, "w") + json.dump`` / ``Path.write_text`` pattern this module
replaces — is not a write but a truncate followed by a stream, and anything
that reads (or writes) during the stream sees an incomplete file.

pccr 2026-09-14 lost a certification to exactly that: the UI's floor levelling
rewrote a 235 MB ``segmentation_result.json`` while the certification's split
was streaming its own copy over the same path. The shorter of the two landed
first and the longer one's tail survived past its end, leaving 398 bytes of a
second document glued to a valid one — ``json.JSONDecodeError: Extra data``,
and both the run and the instance store died on it.

Writing to a sibling temporary and ``os.replace``-ing it is atomic on POSIX:
a reader opens either the whole old file or the whole new one, never a seam.
It does not make concurrent WRITERS correct — two of them still race, the
loser's version simply disappears instead of corrupting the winner's. Use
``session_lock`` for that.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Replace ``path`` with ``text`` (UTF-8) atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, obj: Any, **dumps_kwargs) -> None:
    """Replace ``path`` with ``obj`` serialized as JSON, atomically.

    The document is serialized in full before the file is touched, so a
    serialization error leaves the previous artifact intact instead of a
    half-written one.
    """
    atomic_write_bytes(path, json.dumps(obj, **dumps_kwargs).encode("utf-8"))
