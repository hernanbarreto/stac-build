# `server/semantic/` — Semantic Service (Phase 0)

The single, persistent VLM backend for the whole pipeline: one **vLLM**
OpenAI-compatible endpoint serving **Qwen3-VL**, reached only through the
`semantic_client` in this package. Every downstream consumer (Phases 1–6)
imports `get_semantic_client(...)`; nobody talks to vLLM directly and nobody
loads the model in-process.

## Why a service, not in-process

The backend runs in conda env `da3` (pins `transformers<4.48`); vLLM needs a
newer stack. So vLLM lives in its **own** env `semantic` and is exposed as an
HTTP service. Consumers in *any* env (`da3`, `r3d`, a bare CLI) use the
dependency-light client (`requests` + `Pillow` only). One GPU-resident model,
many consumers.

## Architectural rule it enforces

> The VLM proposes, describes, detects, classifies, orchestrates. **It never
> measures.** Every metric comes from deterministic tools over geometry (or
> `surface_fitting`).

Provenance of every VLM output is tracked: outputs are `vlm_proposed` until a
tool or a human upgrades them (`tool_measured` / `human_validated`). The JSONL
call log (`logs/semantic_calls.jsonl`) is the audit trail behind that rule.

## Layout

| File | Role |
|------|------|
| `types.py` | Message / ToolCall / LLMResponse; image encoding (PIL/numpy/path/bytes/data-URL) |
| `semantic_config.py` | Loads the `semantic:` block from `server/config.yaml`, env-agnostic |
| `backends.py` | `LLMBackend` interface + `QwenLocalBackend` (HTTP). `make_backend()` factory |
| `_parse.py` | Tolerant OpenAI-response / tool-argument parsing |
| `call_log.py` | Centralized JSONL logging of every call |
| `client.py` | `SemanticClient` (logging + bounded tool-loop) and `get_semantic_client()` |
| `serve.py` | vLLM launcher — builds `vllm serve` argv from config (single source of truth) |
| `healthcheck.py` | Polls `/models`, reports VRAM |
| `smoke_test.py` | text / image+text / tool-calling smoke tests |

## Backends (config: `semantic.backends`)

- `qwen_local` (default) — `Qwen/Qwen3-VL-8B-Instruct`, the **exact** variant
  R3D uses (`vendor/r3d/.../generate_responses.py`, `rgb_overlay_evals.py`).
- `qwen_local_large` — `Qwen/Qwen3-VL-32B-Instruct-FP8`, the larger candidate
  for the Phase 7 comparison. A single vLLM process serves one model at a time.

## Run

```bash
# 1. weights (network volume)
bash setup_weights.sh semantic            # 8B baseline
bash setup_weights.sh semantic-large      # 32B-FP8 candidate (optional, ~32 GB)

# 2. serve (env `semantic`)
bash scripts/serve_semantic.sh            # default backend
bash scripts/serve_semantic.sh qwen_local_large

# 3. verify (any env with requests+pyyaml)
conda activate semantic && cd server
python -m semantic.healthcheck --wait 900
python -m semantic.smoke_test
```

`init_pod.sh` starts the service automatically (tmux session `semantic`);
`START_SEMANTIC=0` skips it.

## Client usage (any consumer)

```python
from semantic import get_semantic_client, system, user

sc = get_semantic_client(consumer="phase2.classify")     # default backend
resp = sc.chat([system("..."), user("classify this", images=[crop])])

# tool loop (Phase 5): the model calls deterministic tools; it never measures
out = sc.run_tool_loop(messages, tools=SPEC, tool_impls=IMPLS, max_iterations=8)
```

## Provenance (ours vs external)

- **Ours:** everything in this directory (client, service launcher, config,
  logging, smoke tests). The design (persistent shared service) deliberately
  diverges from R3D's in-process `LLM` client.
- **External / from R3D:** only the *choice* of baseline model
  (`Qwen/Qwen3-VL-8B-Instruct`) and the confirmation that `vllm==0.19.0` +
  `hermes` tool parser fit this family. The reference in-process client we did
  **not** copy lives at `vendor/r3d/r3d/pipeline/eval/vllm_client.py`.
- **External:** `vllm`, `Qwen3-VL` (weights on the network volume).
