# CLAUDE.md — working rules for this repo

## INVIOLABLE: no phase left with pending items
When working through the multi-phase plan (`claude_stac.txt`: Phase 0 → 1 → R →
5 → 2 → 3 → 4 → 6 → 7):

- **Finish each phase 100% before advancing to the next. Never advance while the
  previous phase has ANY pending item.** If Phase 1 has leftovers, finish Phase 1
  before touching Phase R; if Phase R has leftovers, finish Phase R before Phase
  5; and so on.
- **Never leave things pending.** Every sub-item of a phase must be implemented,
  wired, and tested. The only acceptable open item is a genuine EXTERNAL data
  dependency the user must provide (e.g. a hand-labeled ground-truth set, a
  multi-window reconstruction) — and even then all code + synthetic/unit tests
  for it must be complete, and the dependency stated explicitly.
- At the close of each phase: summary + metrics, then wait for the user's OK
  (as `claude_stac.txt` mandates).

## Provenance rule (architectural, inviolable)
The VLM proposes/describes/detects/classifies/orchestrates. It NEVER measures.
Every metric comes from deterministic tools over geometry or `surface_fitting`.
Every VLM output entering a deliverable is tagged `vlm_proposed` /
`tool_measured` / `human_validated`.

## Segment everything, understand the scene
The auto-prompter comprehends what it is seeing (no domain assumption) and
segments EVERYTHING. The construction vocabulary is a canonicalization/routing
overlay, never a detection filter.

## Environment notes
- Backend: env `da3`, `bash scripts/start.sh` (FastAPI/uvicorn, port 8765).
- Semantic service (Phase 0): env `semantic`, `bash scripts/serve_semantic.sh`
  (vLLM/Qwen3-VL on 127.0.0.1:8799); clients use `server/semantic/`.
- Phase R geometry reuses R3D (`vendor/r3d`) ported into `server/phase_r/`.
- All code / YAML / docstrings / comments in ENGLISH.
- No paid external APIs; everything local. GPU: RTX A6000 48 GB (sm_86).
