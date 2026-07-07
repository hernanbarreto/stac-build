# STAC-Builder — Semantic Service: vLLM launcher.
#
# Single source of truth: reads the `semantic:` block from config.yaml and
# exec's `vllm serve` with the matching flags. Run inside the `semantic` conda
# env:
#     python -m semantic.serve --backend qwen_local
# (or via scripts/serve_semantic.sh). Ours (launcher). vLLM is external.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python semantic/serve.py` and `python -m semantic.serve` alike.
_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from semantic.semantic_config import backend_config, load_semantic_config  # noqa: E402


def build_argv(backend: str, overrides: dict) -> list[str]:
    cfg = load_semantic_config()
    svc = cfg["service"]
    b = backend_config(backend, cfg)

    # Prefer a local weights path if present, else the HF model id (vLLM will
    # pull to HF_HOME on the network volume). Local path keeps it fully offline.
    model_ref = b["model_id"]
    wp = b.get("weights_path_abs")
    if wp and Path(wp).exists() and any(Path(wp).iterdir()):
        model_ref = wp

    gpu_util = overrides.get("gpu_memory_utilization") or b.get("gpu_memory_utilization", 0.35)
    max_len = overrides.get("max_model_len") or b.get("max_model_len", 32768)

    argv = [
        "vllm", "serve", str(model_ref),
        "--served-model-name", b["served_model_name"],
        "--host", str(svc["host"]),
        "--port", str(svc["port"]),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_len),
        "--limit-mm-per-prompt", json.dumps({"image": int(b.get("max_images_per_prompt", 8))}),
        "--trust-remote-code",
    ]
    dtype = b.get("dtype")
    if dtype:
        argv += ["--dtype", str(dtype)]
    # Native Qwen3 tool-calling.
    parser = b.get("tool_call_parser", "hermes")
    argv += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
    extra = b.get("extra_serve_args") or []
    argv += [str(x) for x in extra]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch the STAC semantic vLLM service")
    ap.add_argument("--backend", default=None, help="backend name (default: config default_backend)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--print-only", action="store_true", help="print the vllm command and exit")
    args = ap.parse_args()

    cfg = load_semantic_config()
    backend = args.backend or cfg.get("default_backend", "qwen_local")
    overrides = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    argv = build_argv(backend, overrides)
    print("[semantic.serve] " + " ".join(argv), flush=True)
    if args.print_only:
        return
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
