"""Access-log noise filter — the UI polls, the pipeline speaks, only one of
the two is worth a line.

USER 2026-09-16: *"que no aparezcan estos INFO: 127.0.0.1:xxxxx - GET /api/"*.
Measured on the first minutes of a run: 102 access lines, 60 of them two polls
(`/api/certify/state` 34, `/api/correction/state` 26) and 24 more the kit's
periodic set — the reconstruction's own output was the minority of its own log.

Two rules, and the second is the important one:
  * a line whose ``METHOD /path`` starts with one of the configured prefixes
    (``server.access_log_quiet`` in config.yaml) is dropped;
  * UNLESS its status is 4xx/5xx — an endpoint that fails is exactly what the
    log is for, however often it is polled.

Also drops 206 Partial Content (Potree octree nodes and video range requests:
hundreds per cloud load).

Console and file get the same treatment because they are the same stream:
``scripts/start.sh`` tees stdout into ``logs/server_<ts>.log``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence, Tuple

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Only the fallback for a config that cannot be read — the real list is YAML.
FALLBACK_QUIET: Tuple[str, ...] = (
    "GET /health",
    "GET /api/tasks/",
    "GET /api/semantic/status",
)

_STATUS_RE = re.compile(r'"\s+(\d{3})\b')


def load_quiet_prefixes(config_path: Path = CONFIG_PATH) -> Tuple[str, ...]:
    """``server.access_log_quiet`` from config.yaml (fallback on any error —
    a broken config must never stop the server from starting)."""
    try:
        import yaml
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
        quiet = ((raw.get("server") or {}).get("access_log_quiet")) or []
        return tuple(str(q) for q in quiet) or FALLBACK_QUIET
    except Exception:
        return FALLBACK_QUIET


def _status_of(record: logging.LogRecord, msg: str) -> int:
    """The HTTP status of an access record (0 when it is not one).

    uvicorn logs ``'%s - "%s %s HTTP/%s" %d'`` with the status as the last
    positional arg; the regex is the fallback for a pre-formatted message.
    """
    args = record.args
    if isinstance(args, (tuple, list)) and args:
        try:
            return int(args[-1])
        except (ValueError, TypeError):
            pass
    m = _STATUS_RE.search(msg)
    return int(m.group(1)) if m else 0


class PollingNoiseFilter(logging.Filter):
    """Drops the polls, never an error."""

    def __init__(self, quiet: Sequence[str] | None = None):
        super().__init__()
        self.quiet = tuple(quiet) if quiet is not None else load_quiet_prefixes()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        status = _status_of(record, msg)
        if status >= 400:
            return True                      # a failure is never silenced
        if any(p in msg for p in self.quiet):
            return False
        return status != 206                 # range-request spam


def install(logger_name: str = "uvicorn.access",
            quiet: Sequence[str] | None = None) -> bool:
    """Attach the filter once. Returns True if it was added now.

    Called at import time AND from the lifespan: uvicorn's startup dictConfig
    can drop a filter installed before it ran.
    """
    logger = logging.getLogger(logger_name)
    if any(isinstance(f, PollingNoiseFilter) for f in logger.filters):
        return False
    logger.addFilter(PollingNoiseFilter(quiet))
    return True
