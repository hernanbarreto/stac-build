# STAC-Builder — Semantic Service: centralized JSONL call logging.
#
# Every call through semantic_client is appended as one JSON line recording:
#   consumer, backend/model, the (image-byte-free) prompt, image refs+hashes,
#   tools offered, the response text / tool calls, latency and token usage.
# This is the single audit trail of "what did the VLM propose, when, for whom"
# and underpins the vlm_proposed provenance rule. Mirrors the repo's existing
# _audit_log JSONL convention (server/main.py). Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .semantic_config import load_semantic_config, resolve_path

_lock = threading.Lock()


def _log_path() -> Path | None:
    cfg = load_semantic_config().get("logging", {})
    if not cfg.get("enabled", True):
        return None
    p = resolve_path(cfg.get("jsonl_path", "logs/semantic_calls.jsonl"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log_call(entry: dict[str, Any]) -> None:
    """Append one call record as a JSON line (best-effort, never raises)."""
    try:
        path = _log_path()
        if path is None:
            return
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(), **entry}
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _lock:
            with open(path, "a") as f:
                f.write(line + "\n")
    except Exception:
        # Logging must never break a consumer.
        pass
