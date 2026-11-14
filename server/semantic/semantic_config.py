# STAC-Builder — Semantic Service: config loader.
#
# Loads the `semantic:` block from server/config.yaml. Deliberately standalone
# (locates config.yaml relative to this file, does NOT import server/config.py)
# so the client works from ANY conda env / cwd — the vLLM `semantic` env, the
# backend `da3` env, the `r3d` env, or a bare CLI. Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# server/semantic/semantic_config.py -> parents[1] == server/
_SERVER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVER_DIR.parent
_CONFIG_PATH = _SERVER_DIR / "config.yaml"


# Fallback defaults — used only if the `semantic:` block is missing from
# config.yaml, so the client never hard-crashes on a fresh checkout.
_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "service": {
        "host": "127.0.0.1",
        "port": 8799,
        "api_key": "EMPTY",
        "startup_timeout_s": 900,
        "request_timeout_s": 600,
    },
    "default_backend": "qwen_local",
    "generation": {"temperature": 0.0, "max_tokens": 2048},
    "tool_calling": {"max_iterations": 8},
    "logging": {"jsonl_path": "logs/semantic_calls.jsonl", "enabled": True},
    "backends": {
        "qwen_local": {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "served_model_name": "qwen_local",
            "weights_path": "weights/qwen3vl/8b-instruct",
            "gpu_memory_utilization": 0.35,
            "max_model_len": 32768,
            "max_images_per_prompt": 8,
            "tool_call_parser": "hermes",
            "reasoning": False,
        }
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# config.yaml parsed ONCE per file on disk: {"stamp": (path, inode, size, mtime_ns), "value": merged}
_CACHE: dict[str, Any] = {}


def _config_stamp() -> tuple | None:
    """What identifies config.yaml as it is on disk now; None when it does not exist."""
    try:
        st = _CONFIG_PATH.stat()
    except OSError:
        return None
    return (str(_CONFIG_PATH), st.st_ino, st.st_size, st.st_mtime_ns)


def load_semantic_config() -> dict[str, Any]:
    """Return the merged `semantic:` config block (defaults <- config.yaml).

    The file is parsed once per (inode, size, mtime) and every call gets its OWN copy of the
    block: config.yaml is ~4,700 lines and its parse took 0.32 s — paid twice by every question
    of the chat (the client and the call log, measured 2026-11-14), 0.65 s of an answer asked in
    1-2 s. An edit of the file is read at the next call, as before."""
    import copy
    stamp = _config_stamp()
    if stamp is not None and _CACHE.get("stamp") == stamp:
        return copy.deepcopy(_CACHE["value"])
    data: dict[str, Any] = {}
    if stamp is not None:
        with open(_CONFIG_PATH, "r") as f:
            full = yaml.safe_load(f) or {}
        data = full.get("semantic", {}) or {}
    merged = _deep_merge(_DEFAULTS, data)
    if stamp is not None and _config_stamp() == stamp:      # not rewritten while it was read
        _CACHE.update(stamp=stamp, value=copy.deepcopy(merged))
    return merged


def base_url(cfg: dict[str, Any] | None = None) -> str:
    """OpenAI-compatible base URL of the running vLLM service."""
    cfg = cfg or load_semantic_config()
    svc = cfg["service"]
    # Env override lets a consumer point at a non-default host/port.
    override = os.environ.get("SEMANTIC_BASE_URL")
    if override:
        return override.rstrip("/")
    return f"http://{svc['host']}:{svc['port']}/v1"


def backend_config(name: str | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve one backend's config by name (default: cfg['default_backend'])."""
    cfg = cfg or load_semantic_config()
    name = name or cfg.get("default_backend", "qwen_local")
    backends = cfg.get("backends", {})
    if name not in backends:
        raise KeyError(
            f"Unknown semantic backend '{name}'. Known: {sorted(backends)}"
        )
    b = dict(backends[name])
    b.setdefault("name", name)
    # Resolve a relative weights_path against the repo root.
    wp = b.get("weights_path")
    if wp and not os.path.isabs(wp):
        b["weights_path_abs"] = str((_REPO_ROOT / wp).resolve())
    else:
        b["weights_path_abs"] = wp
    return b


def resolve_path(p: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (_REPO_ROOT / pp)


def repo_root() -> Path:
    return _REPO_ROOT
