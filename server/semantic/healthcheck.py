# STAC-Builder — Semantic Service: healthcheck + VRAM probe.
#
#   python -m semantic.healthcheck [--wait SECONDS] [--backend NAME]
# Polls the vLLM /models endpoint until the expected model is served (or
# timeout), and reports GPU memory used (nvidia-smi) so we can log the vLLM
# resting / under-load budget the spec asks for. Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from semantic.backends import make_backend  # noqa: E402
from semantic.semantic_config import load_semantic_config  # noqa: E402


def gpu_mem_used_mib() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to poll before giving up")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    cfg = load_semantic_config()
    backend = make_backend(args.backend, cfg)
    deadline = time.time() + args.wait
    while True:
        h = backend.health()
        if h["ok"]:
            mem = gpu_mem_used_mib()
            print(f"[healthcheck] OK — serving '{backend.model}'; served={h['served_models']}")
            if mem:
                print(f"[healthcheck] GPU mem used (MiB): {mem}")
            return 0
        if time.time() >= deadline:
            print(f"[healthcheck] NOT READY — {h.get('error')}; served={h.get('served_models')}")
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
