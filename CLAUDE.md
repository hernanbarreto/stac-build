# CLAUDE.md — working rules for this repo

## ⭐ BEST CONFIGURATION TO DATE — USER-VERIFIED 2026-08-19 (do NOT change)
The user validated this exact config visually as **the best reconstruction
configuration we have** ("quedó la mejor configuración... de momento no tocamos
más"). It is the restored **2026-08-11 recipe**: everything added between 08-12
and 08-18 was tried, judged worse by the user, and REVERTED. Never re-apply
those reverted knobs without an explicit new decision from him.

    backend vggtomega_pgsr          (full PGSR precision stage)
    simple.conf_percentile 10       (cloud filter: the 08-11 value; 20 and 50
                                     were tried 08-18 — the mesh recipe was the
                                     problem, not the cloud filter)
    pgsr: max_abs_split_points 50000 (vendor default), use_depth_filter false,
          sky_mask false, cloud_anchor false, uniform seed, resolution 1
    tsdf: rasterize_cloud_depth false → integrate the PGSR renders directly,
          mv_consistency true, tsdf_weight_thresh 2.0 (≥2 cameras must agree —
          1.0 produced floating parts "en cualquier lado"), voxel 12 mm,
          texture true (texrecon)

REVERTED / DISCARDED on the user's visual verdict (all 2026-08-18/19 — do not
resurrect blindly): full-frustum cloud raster as the mesh source (`rasterize_
cloud_depth: true`, "muy ruidoso"), confidence-hierarchical raster, band-average
raster, per-pixel PGSR↔cloud blend (`pgsr_blend_tau_m`), cloud-anchored PGSR
training (`pgsr.cloud_anchor` + confidence-weighted anchor/seed — displaced
floating parts), NVIDIA **NKSR** (built in env `nksr` from source, wired as
`tsdf.mesh_method: "nksr"`, license NC — "no dio buenos resultados"), and the
**DA3-streaming cloud** (single 600-frame chunk, vendor conf 0.75×mean — "no es
mejor ni cerca que vggt"; the vendor single-chunk crash IS fixed in
vendor/depth-anything-3, keep the patch). The code for all of these stays in
the repo, selectable, OFF by default.

## ⭐ FLOW CHANGE — USER DECISION 2026-08-28 (supersedes auto-mesh mandate)
The automatic end-of-pipeline mesh worked on some scenes and not others, so:
- **Reconstruction ends at the CLEANED CLOUD** (`pipeline.auto_tsdf: false`);
  the cloud is pushed to the viewer the moment CloudCompy finishes. The PGSR
  and TSDF stages no longer run in the pipeline (the 08-19 mesh recipe above
  stays the recipe FOR WHEN a mesh is requested).
- **Closing the Segmentation Manager runs ONLY DBSCAN + matching + OBBs** —
  `/api/segmentation/refresh` no longer auto-carves per-object meshes.
- **Individual meshing (`/api/segmentation/tsdf/export`) is TSDF + texrecon
  ONLY**: if `output/tsdf/scene/scene.glb` doesn't exist it is baked on demand
  (config `tsdf:` recipe with `mesh_method=tsdf`, `texture_mode=texrecon`
  forced), then instances are carved from it. The legacy untextured
  `export_tsdf_meshes` fallback is no longer called. Without PGSR renders,
  `depth_source: auto` falls back to the backend's native depth (artifact-based
  — sessions that DID run PGSR still integrate its renders).
- **Chat always available**: vLLM (Qwen3-VL) starts at server boot (lifespan),
  is unloaded by the reconstruction workers for exclusive GPU (unchanged), and
  `_semantic_reload_if_idle` reloads it when the pipeline finishes, fails, or
  is cancelled (skipped while another pipeline is running/queued; multi-scan
  reloads only after the last scan).

## Precision task status (claude_stac.txt, phases A–F) — updated 2026-08-19
Phases A–D are CLOSED with pre-registered A/B verdicts (docs/scale_ab_results.md,
docs/phase_bc_ab_results.md); E (external scorecard vs COLMAP/OpenMVS +
RealityScan import) and F (final matrix + precision_report.md) are DEFERRED by
the user. Current production defaults (all evidence-backed, do not "improve"
them blindly):
- DOCTRINE (user, standing): the VGGT-Ω cloud is the truth and nothing modifies
  it (`pgsr.consistent_cloud: false` — the viewer cloud/Potree is never
  overwritten). `pipeline.auto_tsdf: false` since 2026-08-28 (run ends at the
  cloud; mesh on demand only — see FLOW CHANGE block above).
- `tsdf.mv_consistency: true` (won its A/B), `tsdf.depth_source: auto`, voxel
  12 mm (8/6 mm lost), `native_depth_method` off (lost: doubles the
  double-surface stat). `cloud_delaunay` (Delaunay+fusion) kept as alternative
  — superseded: never reached the required quality.
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
