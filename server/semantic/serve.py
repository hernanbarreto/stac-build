# STAC-Builder — Semantic Service: vLLM launcher.
#
# Single source of truth: reads the `semantic:` block (of server/config.yaml, or of the
# job's FROZEN output/run_config.yaml — `--config`) and exec's `vllm serve` with the
# matching flags. Run inside the `semantic` conda env:
#     python -m semantic.serve --backend qwen_local [--config <run_config.yaml>]
# (or via scripts/serve_semantic.sh). Ours (launcher). vLLM is external.
#
# DETERMINISM (docs/plan_determinismo.md points 80, 94, 155 — 2026-10-08): every answer of
# the engine must be a pure function of its request, whatever ran before and whatever else
# is on the machine. The launcher therefore
#   * serves WITHOUT prefix caching, ONE sequence at a time, a FIXED token batch of the
#     model's own length (so no prefill is ever split by a card-dependent default — vLLM 0.19
#     picks 8192 or 2048 by the card's memory, arg_utils.py:2053), WITHOUT compilation
#     (enforce-eager: Inductor's benchmark_combo_kernel picks kernels by TIMING and caches
#     them in /root/.cache/vllm, outside /workspace), with vLLM's OWN generation defaults
#     (the model's generation_config.json would inject top_k 20 / top_p 0.8 / temperature
#     0.7 as defaults) and a fixed seed; the attention backend is PINNED to FLASH_ATTN and
#     VLLM_BATCH_INVARIANT=1 is set — vLLM 0.19 refuses batch invariance with an automatic
#     backend (batch_invariant.py:1008: None is not a supported backend) and installs its
#     batch-invariant matmul on sm_80 / sm_89 / sm_100 and the cuBLAS-workspace form on any
#     other card (batch_invariant.py:938-957): which form applies is RECORDED, never guessed;
#   * loads ONLY the local weights (:data:`PINNED_WEIGHTS`): their directory must exist and
#     every pinned file must hash to its pinned sha256 (a missing directory used to fall back
#     to the Hugging Face id at an unpinned revision);
#   * writes the IDENTITY of the service it starts (logs/semantic_service.json: backend,
#     served model, weights, the exact argv, the versions, the card, the sha256 of the
#     semantic configuration block) BEFORE exec — the pid is the same after exec, so a
#     reader can tell whether the vLLM answering /health is the one this launch described
#     (semantic.service.service_identity). The VLM stage copies that identity into
#     vlm_analysis.json and refuses an engine whose identity is not the frozen one.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Allow `python semantic/serve.py` and `python -m semantic.serve` alike.
_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

import repro  # noqa: E402
from semantic.semantic_config import (_DEFAULTS, _deep_merge, backend_config,  # noqa: E402
                                      load_semantic_config, repo_root, resolve_path)

IDENTITY_NAME = "semantic_service.json"      # logs/<this>: the identity of the vLLM serving now
IDENTITY_VERSION = 1
# the attention backend every launch pins (vLLM 0.19's default on Ampere; batch invariance
# requires it to be named explicitly — batch_invariant.py:990-1017)
ATTENTION_BACKEND = "FLASH_ATTN"
# the compute capabilities on which vLLM 0.19 installs its batch-invariant matmul kernels
# (model_executor/layers/batch_invariant.py:938-948: family 10x, 8.0, 8.9); any other card
# gets the cuBLAS-workspace form of invariance (:950-957). Recorded, never a decision.
BATCH_INVARIANT_MM_CAPABILITIES = ("8.0", "8.9")
BATCH_INVARIANT_MM_FAMILY_MAJOR = 10
# the distributions whose versions change what the engine computes (recorded in the identity)
SERVICE_LIBS = ("vllm", "torch", "transformers", "tokenizers", "triton", "flash-attn",
                "flashinfer-python", "xformers", "pillow", "safetensors", "huggingface-hub")

# The LOCAL weights every backend serves, pinned (point 94 / 155): the Hugging Face revision
# the directory was downloaded from (.cache/huggingface/download/*.metadata, commit line) and
# the sha256 of every file vLLM reads, measured with sha256sum on 2026-10-08 (the safetensors
# shards' values equal the LFS etags in the same metadata files). A backend whose weights are
# not pinned here cannot be served: an unpinned model is an unknown input.
PINNED_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3-VL-8B-Instruct": {
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "sha256": {
            "chat_template.json": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
            "config.json": "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
            "generation_config.json": "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
            "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
            "model-00001-of-00004.safetensors": "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
            "model-00002-of-00004.safetensors": "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
            "model-00003-of-00004.safetensors": "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
            "model-00004-of-00004.safetensors": "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
            "model.safetensors.index.json": "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
            "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
            "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
            "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        },
    },
}


class ServeError(RuntimeError):
    """The service cannot be launched as the plan requires (weights missing or not the pinned
    ones, an unpinned backend, a configuration that cannot be read). Never a fallback."""


def identity_path() -> Path:
    """Where the launcher writes the identity of the service it starts (and where
    semantic.service reads it): ``logs/semantic_service.json`` of the repo."""
    return resolve_path("logs") / IDENTITY_NAME


# ── configuration ────────────────────────────────────────────────────────────────────────

def semantic_config_from(path: Optional[os.PathLike]) -> Dict[str, Any]:
    """The merged ``semantic:`` block: of ``path`` (a YAML holding a ``semantic`` section —
    the job's frozen output/run_config.yaml, point 155) when given, of server/config.yaml
    otherwise (:func:`load_semantic_config`). An unreadable file or one without the
    section RAISES: a launch never falls back to another configuration."""
    if path is None:
        return load_semantic_config()
    import yaml
    p = Path(path)
    try:
        full = yaml.safe_load(p.read_text()) or {}
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise ServeError(f"semantic configuration {p} cannot be read ({e})") from e
    block = full.get("semantic") if isinstance(full, dict) else None
    if not isinstance(block, dict):
        raise ServeError(f"{p} has no 'semantic' section — nothing to serve from")
    return _deep_merge(_DEFAULTS, block)


def semantic_config_sha256(cfg: Mapping[str, Any]) -> str:
    """The identity of a semantic configuration block (canonical JSON sha256) — what the
    stage compares against the frozen run configuration it was given."""
    return repro.sha256_json(dict(cfg))


# ── the weights ──────────────────────────────────────────────────────────────────────────

def weights_identity(b: Mapping[str, Any], log=print) -> Dict[str, Any]:
    """The local weights of backend ``b`` VERIFIED against :data:`PINNED_WEIGHTS`: the
    directory exists, every pinned file is there and hashes to its sha256 (the whole file is
    read — the same check server/da3_weights.py runs on DA3's weights). RAISES otherwise:
    no download, no other revision, no unpinned model (point 94). Returns the record a
    product made with these weights carries: model id, revision, repo-relative path and the
    per-file sha256."""
    model_id = str(b.get("model_id") or "")
    pin = PINNED_WEIGHTS.get(model_id)
    if pin is None:
        raise ServeError(f"backend '{b.get('name')}' serves '{model_id}', which has no pinned "
                         f"revision (semantic.serve.PINNED_WEIGHTS: {sorted(PINNED_WEIGHTS)}) — "
                         f"pin its revision and file sha256s first")
    wp = b.get("weights_path_abs")
    if not wp:
        raise ServeError(f"backend '{b.get('name')}' declares no weights_path — the local "
                         f"weights are the only ones served")
    root = Path(wp)
    if not root.is_dir() or not any(root.iterdir()):
        raise ServeError(f"the local weights of '{model_id}' are not at {root} — the service "
                         f"never downloads them (point 94): put the pinned revision "
                         f"{pin['revision'][:12]} there")
    want = dict(pin["sha256"])
    t0 = time.time()
    for name in sorted(want):
        f = root / name
        if not f.is_file():
            raise ServeError(f"{f} is missing — the pinned weights of '{model_id}' "
                             f"(revision {pin['revision'][:12]}) are incomplete")
        got = repro.sha256_file(f)
        if got != want[name]:
            raise ServeError(f"{f} hashes to sha256 {got[:12]}…, the pinned file is "
                             f"{want[name][:12]}… — these are not the weights every validated "
                             f"run served")
    try:
        rel = root.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        rel = root.as_posix()
    log(f"[semantic.serve] weights of '{model_id}' @ {pin['revision'][:12]} verified "
        f"({len(want)} file(s) by sha256, {time.time() - t0:.0f} s) in {rel}")
    return {"model_id": model_id, "revision": str(pin["revision"]), "path": rel,
            "sha256": dict(sorted(want.items()))}


# ── the command line ─────────────────────────────────────────────────────────────────────

def deterministic_serve_args(max_model_len: int, seed: int) -> List[str]:
    """The vLLM flags of point 80 (every launch, never a switch): no prefix caching, one
    sequence at a time, a token batch of the model's own length, no compilation, vLLM's
    generation defaults, a fixed seed and the pinned attention backend."""
    return ["--no-enable-prefix-caching",
            "--max-num-seqs", "1",
            "--max-num-batched-tokens", str(int(max_model_len)),
            "--enforce-eager",
            "--generation-config", "vllm",
            "--seed", str(int(seed)),
            "--attention-backend", ATTENTION_BACKEND]


def build_argv(backend: str, overrides: Optional[dict] = None,
               cfg: Optional[Mapping[str, Any]] = None, *, model_ref: Optional[str] = None,
               ) -> List[str]:
    """The ``vllm serve`` command line of ``backend`` under ``cfg`` (default: the merged
    config.yaml block). ``model_ref`` is the verified local weights directory
    (:func:`weights_identity`); without it the backend's declared ``weights_path_abs`` is
    used as written — the launcher always verifies first, ``--print-only`` shows the line."""
    cfg = dict(cfg) if cfg is not None else load_semantic_config()
    overrides = overrides or {}
    svc = cfg["service"]
    b = backend_config(backend, cfg)
    ref = model_ref or b.get("weights_path_abs")
    if not ref:
        raise ServeError(f"backend '{backend}' declares no weights_path — only local weights "
                         f"are served (point 94)")

    gpu_util = overrides.get("gpu_memory_utilization") or b.get("gpu_memory_utilization", 0.35)
    max_len = int(overrides.get("max_model_len") or b.get("max_model_len", 32768))
    gen = cfg.get("generation", {}) or {}
    seed = gen.get("seed")
    if seed is None or isinstance(seed, bool) or int(seed) != seed:
        raise ServeError("semantic.generation.seed must be an integer — the engine's seed is "
                         "part of every answer's identity (point 80)")

    argv = [
        "vllm", "serve", str(ref),
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
    # Thinking-variant support: -Instruct models set reasoning:false (no flag);
    # a thinking Qwen3 backend flips it on and vLLM parses <think> blocks.
    if b.get("reasoning"):
        argv += ["--reasoning-parser", "qwen3"]
    argv += deterministic_serve_args(max_len, int(seed))
    extra = b.get("extra_serve_args") or []
    argv += [str(x) for x in extra]
    return argv


def batch_invariance_record(capability: str) -> Dict[str, Any]:
    """Which form of vLLM 0.19's batch invariance THIS card gets (see the module header):
    the Triton batch-invariant matmul on sm_80 / sm_89 / sm_10x, the cuBLAS-workspace form
    elsewhere. Always enabled; the form is recorded."""
    cap = str(capability).strip()
    major = int(cap.split(".")[0]) if cap.split(".")[0].isdigit() else -1
    mm = cap in BATCH_INVARIANT_MM_CAPABILITIES or major == BATCH_INVARIANT_MM_FAMILY_MAJOR
    return {"enabled": True, "env": "VLLM_BATCH_INVARIANT=1", "capability": cap,
            "form": "batch_invariant_matmul" if mm else "cublas_workspace"}


def service_environment(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment ``vllm serve`` runs with: :func:`repro.deterministic_env` (the cuBLAS
    workspace, PYTHONHASHSEED=0), batch invariance ON, Hugging Face OFFLINE (the weights are
    local and verified; nothing is ever fetched), the one HF cache."""
    env = repro.deterministic_env(base)
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env.setdefault("HF_HOME", "/workspace/hf_cache")
    return env


def _versions() -> Dict[str, Optional[str]]:
    return {name: repro._dist_version(name) for name in SERVICE_LIBS}


def identity_record(backend: str, cfg: Mapping[str, Any], argv: List[str],
                    weights: Mapping[str, Any], env: Mapping[str, str],
                    config_path: Optional[os.PathLike]) -> Dict[str, Any]:
    """What the service about to start IS — the stable part (``identity``: what a product
    made with it records and stamps) and the volatile part (``process``: pid, times, the
    file the configuration came from). The card is read through repro.card_identity (a
    torch probe subprocess that exits before vLLM starts — this launcher never holds a CUDA
    context itself)."""
    b = backend_config(backend, cfg)
    card = repro.card_identity(0)
    repo = repo_root().resolve()

    def _rel(s: str) -> str:
        try:
            return Path(s).resolve().relative_to(repo).as_posix()
        except (ValueError, OSError):
            return s

    identity = {
        "identity_version": IDENTITY_VERSION,
        "backend": backend,
        "served_model_name": b["served_model_name"],
        "weights": dict(weights),
        "argv": [_rel(a) if os.path.isabs(a) else a for a in argv],
        "deterministic_args": deterministic_serve_args(
            int(b.get("max_model_len", 32768)), int((cfg.get("generation") or {}).get("seed"))),
        "generation": dict(cfg.get("generation") or {}),
        "batch_invariance": batch_invariance_record(card["capability"]),
        "env": {k: env.get(k) for k in ("VLLM_BATCH_INVARIANT", "PYTHONHASHSEED",
                                        repro.CUBLAS_WORKSPACE_ENV, "HF_HUB_OFFLINE", "HF_HOME")},
        "versions": _versions(),
        "python": sys.version.split()[0],
        "card": {"key": card["key"], "name": card["name"], "capability": card["capability"],
                 "memory_total_mib": card["memory_total_mib"]},
        "semantic_config_sha256": semantic_config_sha256(cfg),
    }
    identity["sha256"] = repro.sha256_json(identity)
    return {"identity": identity,
            "process": {"pid": os.getpid(), "started_unix": time.time(),
                        "python_executable": sys.executable,
                        "config_path": (str(config_path) if config_path is not None
                                        else "server/config.yaml"),
                        "card_uuid": card["uuid"]}}


def write_identity(doc: Mapping[str, Any], path: Optional[os.PathLike] = None) -> Path:
    """Atomic write of the identity file (temporary file in the same directory, then rename)."""
    p = Path(path) if path is not None else identity_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, p)
    return p


# ── entry point ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Launch the STAC semantic vLLM service")
    ap.add_argument("--backend", default=None, help="backend name (default: config default_backend)")
    ap.add_argument("--config", default=None,
                    help="a YAML with a 'semantic' section to serve from — the job's frozen "
                         "output/run_config.yaml (default: server/config.yaml)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--print-only", action="store_true", help="print the vllm command and exit")
    ap.add_argument("--verify-weights-only", action="store_true",
                    help="verify the backend's local weights against the pinned sha256s and exit")
    args = ap.parse_args()

    cfg = semantic_config_from(args.config)
    backend = args.backend or cfg.get("default_backend", "qwen_local")
    overrides = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    if args.print_only:
        print("[semantic.serve] " + " ".join(build_argv(backend, overrides, cfg)), flush=True)
        return
    b = backend_config(backend, cfg)
    weights = weights_identity(b, log=lambda m: print(m, flush=True))
    if args.verify_weights_only:
        print(json.dumps(weights, indent=1), flush=True)
        return
    argv = build_argv(backend, overrides, cfg, model_ref=b["weights_path_abs"])
    env = service_environment()
    doc = identity_record(backend, cfg, argv, weights, env, args.config)
    p = write_identity(doc)
    print(f"[semantic.serve] identity {doc['identity']['sha256'][:12]} written to {p} "
          f"(card {doc['identity']['card']['key']}, batch invariance "
          f"{doc['identity']['batch_invariance']['form']})", flush=True)
    print("[semantic.serve] " + " ".join(argv), flush=True)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
