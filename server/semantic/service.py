# STAC-Builder — semantic service lifecycle (health + auto-start).
#
# The reconstruction and SAM3 stages STOP vLLM to get the whole GPU
# (workers/base.py::stop_semantic_service). Every later consumer therefore has
# to be able to bring it back: the user never starts it by hand. This module is
# that single entry point, shared by the VLM worker (subprocess) and by the
# FastAPI routes (spatial Q&A), which previously had no way to restart it and
# answered 500 ConnectionRefused after any pipeline run.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

_STAC_ROOT = Path(__file__).resolve().parents[2]
_LAUNCHER = _STAC_ROOT / "scripts" / "serve_semantic.sh"


def _service_cfg(config: Optional[dict] = None) -> dict:
    if config is None:
        from config import cfg as config  # lazy: keeps this importable from workers
    return (config.get("semantic", {}) or {}).get("service", {}) or {}


def is_alive(config: Optional[dict] = None, timeout_s: float = 3.0) -> bool:
    """True when the vLLM endpoint answers /health."""
    import requests
    svc = _service_cfg(config)
    url = f"http://{svc.get('host', '127.0.0.1')}:{svc.get('port', 8799)}/health"
    try:
        return requests.get(url, timeout=timeout_s).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def is_starting() -> bool:
    """True when a vLLM process exists but is not serving yet (weights loading)."""
    try:
        return subprocess.run(["pgrep", "-f", "vllm serve"],
                              capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def ensure_service(config: Optional[dict] = None,
                   log: Optional[Callable[[str], Any]] = None,
                   cancelled: Optional[Callable[[], bool]] = None,
                   timeout_s: Optional[float] = None) -> bool:
    """Healthcheck the semantic service; if down, start it and wait until it
    serves. `scripts/serve_semantic.sh` has its own double-start guard, so
    concurrent callers cannot spawn a second vLLM.

    Returns True once the endpoint is healthy, False on timeout / launch error.
    Loading Qwen3-VL weights takes minutes — pass a short `timeout_s` from
    request handlers that must not block, and report the wait to the user.
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    if is_alive(config):
        return True
    if not _LAUNCHER.exists():
        _log(f"Semantic service down and launcher missing ({_LAUNCHER})")
        return False

    _log("Semantic service down — starting vLLM (Qwen3-VL)...")

    # Give back the VRAM THIS process is holding before vLLM asks for its own.
    # A GPU stage that ran here (the epoch re-consolidation is the one that
    # bites) leaves PyTorch's caching allocator sitting on tens of GB it no
    # longer uses, and vLLM refuses to start below its
    # `--gpu-memory-utilization` share. pccr 2026-09-19: the certification
    # stopped vLLM to take the whole GPU, consolidated, asked for it back, and
    # vLLM died with "Free memory on device cuda:0 (16.27/47.41 GiB) … is less
    # than desired GPU memory utilization (0.5, 23.71 GiB)" — then the
    # certification blocked FOREVER waiting for a service that could not start
    # while the certification itself held the memory. A deadlock, not a crash:
    # nothing in the log, 1 % CPU, S (sleeping), for as long as anyone waited.
    try:
        import torch  # noqa: PLC0415 — optional here, the caller may be CPU-only
        if torch.cuda.is_available():
            free_before = torch.cuda.mem_get_info()[0] / 2 ** 30
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free_after = torch.cuda.mem_get_info()[0] / 2 ** 30
            if free_after - free_before > 0.1:
                _log(f"[gpu] released {free_after - free_before:.1f} GB of cached "
                     f"VRAM before starting vLLM ({free_after:.1f} GB free)")
    except Exception as e:  # noqa: BLE001 — declared, never fatal
        _log(f"[gpu] could not release cached VRAM ({e}) — starting anyway")

    def _die_with_parent():
        # USER ORDER 2026-09-04: the chat must DIE with the backend — twice
        # today orphaned vLLMs (start_new_session) kept 20-45 GB of VRAM after
        # the backend was killed and OOM'd the GPU stages. PR_SET_PDEATHSIG
        # delivers SIGTERM to the (exec-chained bash→python→vllm) child the
        # moment its parent exits, however the parent died; vLLM shuts its
        # EngineCore down on SIGTERM.
        try:
            import ctypes
            import signal as _sig
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, _sig.SIGTERM)
        except Exception:  # noqa: BLE001 — best effort, non-Linux fallback
            pass

    try:
        subprocess.Popen(["bash", str(_LAUNCHER)], cwd=str(_STAC_ROOT),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         preexec_fn=_die_with_parent)
    except Exception as e:  # noqa: BLE001
        _log(f"Could not launch semantic service: {e}")
        return False

    if timeout_s is None:
        timeout_s = float(_service_cfg(config).get("startup_timeout_s", 900))
    if timeout_s <= 0:
        # main.py kicks the launcher at boot ON PURPOSE without waiting, so the
        # server never blocks on a model load. Saying "did not come up within
        # 0s" made a deliberate non-wait read as a failure (USER 2026-09-16).
        _log("Semantic service launched — not waiting for it (loads in the "
             "background; the chat reloads when it is ready)")
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if cancelled and cancelled():
            return False
        if is_alive(config):
            _log("Semantic service is up")
            return True
        time.sleep(5)
    _log(f"Semantic service did not come up within {timeout_s:.0f}s")
    return False
