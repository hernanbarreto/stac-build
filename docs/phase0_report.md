# Phase 0 — Semantic Service: Report

**Status: complete, validated end-to-end on the A6000.** Persistent vLLM
service serving Qwen3-VL-8B-Instruct, shared by all consumers through
`semantic_client`.

## What was built

| Component | Path |
|-----------|------|
| conda env `semantic` | `/workspace/miniforge3/envs/semantic` (vllm 0.19.0, torch 2.10.0+cu128, transformers 4.57.6) |
| Client package | `server/semantic/` (types, config, backends, parser, call-log, client+tool-loop, launcher, healthcheck, smoke tests, README) |
| Config block | `server/config.yaml` → `semantic:` |
| Service launcher | `scripts/serve_semantic.sh` → `python -m semantic.serve` |
| Stop helper | `scripts/stop_semantic.sh` (kills EngineCore child; frees GPU) |
| Weights downloader | `setup_weights.sh semantic` / `semantic-large` |
| Pod integration | `init_pod.sh` (tmux `semantic`, `START_SEMANTIC` flag, healthcheck hint) |
| R3D reuse catalog | `docs/r3d_catalog.md` |

## Model selection (spec 0.a / 0.b)

- **Baseline `qwen_local` = `Qwen/Qwen3-VL-8B-Instruct`** — the EXACT variant
  R3D uses (`vendor/r3d/r3d/pipeline/scripts/generate_responses.py:11`,
  `rgb_overlay_evals.py:14`). No deviation from spec.
- **Candidate `qwen_local_large` = `Qwen/Qwen3-VL-32B-Instruct-FP8`** —
  configured, download deferred to Phase 7. FP8 block-wise runs on Ampere via
  vLLM's FP8-Marlin. Weights ≈32 GiB → cannot coexist with reconstruction;
  usable only served alone for the Phase 7 comparison. Coexistence-over-accuracy
  (spec 0.b cutoff) already argues for the 8B as the production default.

## Architecture decision

vLLM runs in its **own** env `semantic` as a standalone OpenAI-compatible
service; the `da3` backend (pins transformers<4.48) and every other consumer
reach it over HTTP via `semantic_client` (requests + Pillow only). This
diverges deliberately from R3D's in-process `LLM` client
(`vendor/r3d/.../vllm_client.py`) so one GPU-resident model serves many
consumers across many conda envs.

## Serving config (native tool-calling)

```
vllm serve weights/qwen3vl/8b-instruct --served-model-name qwen_local
  --gpu-memory-utilization 0.50 --max-model-len 24576
  --limit-mm-per-prompt {"image":8} --dtype bfloat16
  --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code
```

## VRAM budget (MEASURED, A6000 48 GiB) — spec requirement

The spec says "start at gpu_memory_utilization ≈ 0.35 and adjust by measuring".
Measured findings:

| util | KV cache avail | max usable ctx | verdict |
|------|----------------|----------------|---------|
| 0.35 | ~0 (weights ≈16 GiB ≈ 0.34·48) | — | **infeasible** for the 8B |
| 0.45 | 2.33 GiB | ~16.9k tokens | too tight for 24k+ |
| **0.50** (chosen) | **4.7 GiB** | **34,208 tokens** (max_model_len 24,576, 1.39x concurrency) | **works, with margin** |

- **vLLM resting VRAM: 23.0 GiB** (23,532 MiB / 49,140 MiB). vLLM pre-allocates
  the KV pool, so **under-load ≈ resting** (23,534 MiB after the smoke load).
- **Headroom for the rest of the pipeline: ~25.6 GiB.** Coexistence with a heavy
  reconstruction (VGGT/DA3 peak can reach 30–40 GiB) is therefore **tight**; the
  likely production answer is temporal partitioning (run masklets/segmentation
  early, Q&A after the heavy fusion) or dropping util during reconstruction
  peaks via `START_SEMANTIC=0`. The full A6000 coexistence test is Phase 7 (R.9
  / spec Phase 7 "prueba de convivencia").

## Smoke tests — ALL PASS

| # | Modality | Result | Latency |
|---|----------|--------|---------|
| 1 | text-only | `"OK"` | 2231 ms (incl. cold path) |
| 2 | image + text | `"Red"` (identified the red column) | 386 ms |
| 3 | tool-calling (dummy `get_distance`) | called the tool, answered **"2.5 meters"** — did NOT invent the number | 1115 ms |

The tool-calling test confirms the inviolable rule mechanically: the VLM
**used the deterministic tool** for the number rather than guessing.

## Centralized logging

Every call is appended to `logs/semantic_calls.jsonl`: `ts, pid, consumer,
backend, model, messages (image bytes replaced by sha1 refs), tools_offered,
response{content,tool_calls,usage,finish_reason}, latency_ms, error`. This is
the audit trail behind the `vlm_proposed` provenance rule.

## Disk footprint (Phase 0 additions)

| Item | Size |
|------|------|
| `semantic` conda env | 17 GB |
| Qwen3-VL-8B weights | 17 GB |
| `vendor/r3d` | 25 MB |
| **Phase 0 total** | **≈ 34 GB** |

(32B-FP8 candidate would add ≈32 GB if downloaded for Phase 7.)

## How to run

```bash
bash setup_weights.sh semantic                 # already done (17 GB)
bash scripts/serve_semantic.sh                 # start (tmux via init_pod.sh)
conda activate semantic && cd server
python -m semantic.healthcheck --wait 900
python -m semantic.smoke_test
bash scripts/stop_semantic.sh                  # stop + free GPU
```
