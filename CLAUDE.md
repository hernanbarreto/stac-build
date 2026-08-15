# CLAUDE.md — working rules for this repo

## Precision task status (claude_stac.txt, phases A–F) — updated 2026-08-12
Phases A–D are CLOSED with pre-registered A/B verdicts (docs/scale_ab_results.md,
docs/phase_bc_ab_results.md); E (external scorecard vs COLMAP/OpenMVS +
RealityScan import) and F (final matrix + precision_report.md) are DEFERRED by
the user. Current production defaults (all evidence-backed, do not "improve"
them blindly):
- backend `vggtomega_pgsr` (user decision: +86% mesh coverage, ~2 h/scene PGSR
  stage; `vggtomega` = fast mode), `pipeline.auto_tsdf: true` (every run ends
  with the mesh, never Potree-only).
- `tsdf.mv_consistency: true` (won its A/B), `tsdf.depth_source: auto` (prefers
  the session's PGSR renders), voxel 12 mm (8/6 mm lost), `native_depth_method`
  off (lost: doubles the double-surface stat).
- scale: `global_median`, 12 anchors (structured models + more anchors + depth
  top-up all lost or neutral); VIO source auto-detected when present.
- `reconstruction.pose_refine.enabled: true` (point-to-plane, SELF-GATED);
  `pgsr.pose_refine: false` (photometric variant LOST: RMS +11%).
- PGSR trains with the vendor's published max-quality regime (r2, ncc 0.5,
  outdoor thresholds, exposure comp) in env `pgsr`; `torch.set_num_threads(8)`
  is LOAD-BEARING (without it the multi-view stage is ~10× slower on many-core
  boxes — GPU idles, CPU thrashes).
- Keyframe quantum: 80 since 2026-08-15 (USER DECISION after a visual A/B on
  bufferStop: markedly more complete, less ghosting; caveat — the visual
  baseline was the pre-08-12-pipeline run, so quantum and pipeline upgrades are
  confounded). Denser keyframes give PGSR ~3× more training views; the measured
  trade-offs on bufferStop were scale MAD 3.5%→11.1% (confidence 0.82→0.50),
  probe walk over-measured (28.8 vs 12.7 m real → chunked mode fires), runtime
  1h34→2h50. Status: UNDER EVALUATION across more scenes; do not flip it back
  or "re-validate" without the user's word. (History: 250 had won the
  2026-08-11 A/B on pose-proxy metrics.)

## Operating lessons (user feedback, hard-earned)
- NEVER launch a long GPU run without a performance checkpoint in the first
  minutes (compare measured rate vs expectation; abort on anomaly, not hours in).
- One GPU job at a time; A/B timing measured under contention is INVALID.
- Report progress UNPROMPTED during long runs (Monitor tick relayed to the user
  ~every 20 min) — silence reads as a hang.
- When the user says "detené todo": kill EVERYTHING immediately, confirm with
  the process list, and wait. No new launches without their word.

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
