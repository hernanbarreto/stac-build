# Correction module — Phase 2 plan

Date: 2026-09-08. Follows `docs/correction_inventory.md`. Decisions the prompt
left open are marked **DECISION** with a one-line justification.

---

## 1. Package layout — `server/correction/`

Replaces `segmentation/correction_analysis.py` (deleted). One responsibility per
module; `main.py` only mounts the router.

```
server/correction/
  __init__.py       # public surface: run_objects, run_floor, approve, undo, state
  config.py         # CorrectionConfig dataclasses ← config.yaml `correction:`;
                    # missing/invalid key → CorrectionConfigError naming the key
  epoch.py          # geometry epoch: read/bump/stamp/check (json+pathlib only,
                    # importable from any module without cycles)
  session.py        # immutable CorrectionSession: cloud (+raw), provenance,
                    # keyframes, poses, per-frame camera centres. PLY I/O lives
                    # here; pose/frame parsing reuses segmentation/session_io.
  units.py          # keyframe as the atomic unit; visits = contiguous keyframe
                    # runs; chunk plan ONLY from output/chunk_plan.json
  evidence.py       # copies per marked instance (curated OBB + margin), visits,
                    # per-visit curated evidence, shoot distance
  observability.py  # PCA classes, baselines → observable DOF per visit
  diagnose.py       # internal fingerprint → pose | depth+pose; analytic DA3
                    # anchor cross-check from scale_diagnostics.json
  solve.py          # trimmed yaw-planar ICP + per-ray depth expansion +
                    # DOF projection (unobservable DOF forced to identity)
  distribute.py     # per-keyframe R,t (slerp+lerp between anchors) and k
                    # (step function over the displaced visit's keyframes)
  gates.py          # pure gate functions → {name, passed, measured, limits}
  floor.py          # floor alignment: per-keyframe anchors vs a reference
                    # floor MODEL (level | plane | profile)
  apply.py          # transactional apply/undo/approve (§4)
  invalidate.py     # instance-store in-place update, findings re-anchor,
                    # derived-artifact staleness enumeration
  ledger.py         # append-only corrections.jsonl + epoch_<N>.npz + mirror
  replay.py         # python -m correction.replay --session <dir> --to-epoch N
  report.py         # structured per-run report assembly
  api.py            # FastAPI router (+configure() DI for viewer notify / ctx)
```

**DECISION (DI):** `api.py` receives `resolve_ctx` and `notify_viewer` via a
`configure()` call from `main.py` (router import would be circular). `main.py`
keeps exactly two correction lines: `configure(...)` + `include_router(...)`.

## 2. Units, visits, chunks

- Atomic unit: **keyframe** (line index of `camera_frames.txt`), reached from
  every point through `frame_global`. All transforms are defined, persisted and
  applied per keyframe.
- **Visit** = maximal run of keyframes observing a copy, with gaps ≤
  `units.visit_gap_kf` bridged (evidence keyframes are sparse; a 1–2 kf hole in a
  run is the same visit). Reference = earliest visit group (USER 2026-09-06).
- **Chunks**: read from `output/chunk_plan.json` when present; used ONLY to pool
  evidence for reporting. No plan (phase-1 single pass) → keyframes only.
  `workers/map_worker.py` starts persisting the plan in both chunked paths:

```json
// output/chunk_plan.json  (written by map_worker next to vggt_omega_config.yaml)
{ "version": 1, "phase": "chunked-metric" | "direct-chunked",
  "n_keyframes": 312, "chunk_size": 45, "overlap": 22,
  "chunk_ranges": [[0,45],[23,68], ...],       // exactly chunk_plan.chunk_ranges
  "walk_m": 86.4, "chunk_walk_m": 12.0 }
```

## 3. Solve pipeline (kind = objects)

`mark → evidence → observability → diagnose → solve → gates → distribute →
apply(tx) → invalidate/regenerate → report → pending → approve | undo`.
Every stage returns structured data that lands in the report — also on rejection.

- Solve is **per displaced visit** (not per fake chunk): evidence = the visit's
  curated instance points (MEJORAS §3.4 — bbox points only locate copies), init
  from centroid offsets, trimmed yaw-planar ICP against the reference copies'
  points (planar=True doctrine, USER 2026-09-06), then **DOF projection**:
  components the visit does not observe are removed (single plane → t along its
  normal only, yaw 0; single elongated object → t ⊥ axis, yaw 0 unless axis
  vertical and a second constraint exists; compact asymmetric object or ≥2
  objects with baseline ≥ `min_baseline_m` → yaw + full t).
- Depth `k` only with ≥ `min_objects_for_depth` objects with baseline ≥
  `min_baseline_m`; expansion along each point's own camera ray.
- Visits with < `min_object_points_solve` own points inherit the nearest solved
  visit's transform (declared in the report; nothing inside a marked bbox is
  skipped).
- Distribution: identity up to the reference visit's last keyframe; solved
  transform anchored at each displaced visit's first evidence keyframe;
  slerp(yaw)+lerp(t) between anchors; last anchor extends to the end.
  **DECISION (k distribution):** `k` is a step function over the displaced
  visit's keyframe span, 1.0 elsewhere — depth error is a per-frame acquisition
  error (matches today's per-chunk behaviour); the continuity gate governs the
  rigid part only.

## 4. Transactionality

- `apply` stages EVERYTHING under `output/_tx_epoch_<N>/`: cloud, raw cloud,
  canonical poses + copies (`maplong_run/`, `omega_run/`, `da3_run/` when
  present), `segmentation_result.json` (OBBs recomputed), cumulative
  `depth_correction.json`, regenerated `scale_diagnostics.json`,
  `geometry_epoch.json`, `corrections/epoch_<N>.npz`, and the **Potree octree
  built inside the tx dir** (`potree_converter` gains an optional
  `potree_dir_override`/`ply_override` pair — backwards compatible).
- Integrity checks before swap: point count identical, provenance columns
  byte-identical, poses parseable and row-count identical, epoch file present,
  Potree `metadata.json` present (when rebuild enabled).
- Swap: per-artifact rename current → `output/_epoch_<N-1>/`, rename tx →
  current, journaled (`_tx_swap_journal.json`) and rolled back on any failure
  mid-swap. Any failure BEFORE the swap deletes the tx dir; the session is
  untouched. State `pending` is written at swap time (fixes MEJORAS §2.7).
- Undo = inverse swap from `_epoch_<N-1>/` (kept until approve). Approve deletes
  `_epoch_<N-1>/` and appends the verdict to the ledger.
- **Per-session lock** (in-memory registry in `api.py`, like the Potree build
  lock): a second correction / floor-align / undo / approve while one runs →
  409 with the blocking task id. `pending` state additionally blocks new runs.
- No `except Exception: pass` anywhere in the package: corrupt state/ledger →
  explicit error with the file path and what to do.

## 5. Scale rule vs DA3 anchors

- A fingerprint-derived `k` is a LOCAL depth correction, never a session
  re-scale. After solving, `diagnose.scale_check` recomputes the DA3 agreement
  of every anchor keyframe inside the corrected span **analytically**:
  `agree_f = s_f_current / k_f` (per-frame depth ratio scales exactly by 1/k;
  rigid per-keyframe warps cannot change camera-frame depth). Inputs:
  `scale_diagnostics.json` `anchors.frames[].s_f` (+ current epoch history) and
  `s_applied` — no depth `.npz` needed, so freed sessions still gate.
- Gate fails if any affected anchor's |log agreement| worsens beyond
  `scale_agree_tol` or the anchor MAD grows beyond `scale_mad_tol`.
- `override_scale_check: true` applies anyway; the override (operator, numbers,
  timestamp) is recorded in the ledger and surfaces in every downstream report.
- After apply, `scale_diagnostics.json` is regenerated with an `epochs` history:
  `{epoch, correction_id, anchors: {frame: s_f_current}}` — never silently stale.

## 6. Floor alignment (kind = floor)

- Reference model chosen by the user in the UI (ledger-recorded); default from
  `floor.model_default` (= `plane`):
  - `level`: horizontal plane at y=0 (today's behaviour, now declared).
  - `plane`: single RANSAC plane over the anchor keyframes' floor points — a
    real slope survives; only drift (per-keyframe deviation from the plane) is
    removed.
  - `profile`: longitudinal slope — per-anchor floor height regressed against
    trajectory chainage (robust linear fit y = a + b·s); for ramps/platform
    drainage where one global plane underfits.
- Anchors are **keyframes** whose local floor RANSAC passes the guards
  (`max_tilt_deg`, `min_inliers`, `ransac_tol_m`); failed ones are demoted and
  reported. Non-anchor keyframes interpolate (slerp+lerp). Real steps between
  anchors are preserved.
- Passes the same gates (continuity + scene exam) and the same transactional
  apply; `floor_transform.npz` is reset per model inside the tx (old one rides
  the epoch dir).

## 7. Geometry epoch & downstream

- `output/geometry_epoch.json`: `{"epoch": N, "created_at", "correction_id",
  "parent_epoch"}`. Missing file = epoch 0 (every existing session).
- `correction/epoch.py` provides `current_epoch(output_dir)`,
  `stamp(meta: dict, output_dir)` (adds `geometry_epoch`,
  `human_directed_corrections`, `corrections_overridden`) and
  `check(artifact_meta, output_dir)`.
- Stamped writers (same delivery): `tsdf_export` metas (object/scene/poisson/
  crop), `surface_fit` meta/report/hole_audit/silhouette, `poisson_object`,
  `bim/comparison` `sabana_meta.json` (+ 409-style stale flag before reuse),
  `coverage_store` writes, `phase3_findings` store rows, `phase5_qa` tool
  outputs (`_finalize` adds epoch + corrections count), phase6 report header.
- Consumers: mesh/artifact listing endpoints add `stale: true` when artifact
  epoch ≠ current (UI badge + Regenerate); BIM comparison refuses to serve a
  cross-epoch sabana without the stale flag; measurements always carry epoch.
- Pipeline Replace of RECONSTRUCTION/CLOUDCOMPY deletes `geometry_epoch.json`,
  `depth_correction.json`, `corrections/epoch_*.npz`, `_epoch_*`/`_tx_epoch_*`
  leftovers (new geometry = epoch 0). `corrections.jsonl` is **never** deleted
  (history); replay onto a re-reconstruction re-keys by `frame_global`.

## 8. Depth sidecar

**DECISION (sidecar vs rewrite): `sidecar`.** Rewriting GBs of vendor depth per
correction is unaffordable, unauditable and unreplayable; a per-keyframe scalar
is exact. `rewrite` stays a config enum value that raises `NotImplementedError`
loudly if selected (no silent half-support).

```json
// output/depth_correction.json (cumulative product across epochs)
{ "version": 1, "epoch": N, "k": { "<frame_global>": 1.0374, ... } }  // only ≠1
```

- Accessor in `segmentation/session_io.py`:
  `load_depth_correction(output_dir)` (mtime-cached) and
  `correct_depth(depth, frame_global, output_dir)`.
- Applied to every **post-reconstruction** consumer of reconstruction-derived
  per-keyframe depth — the ones that can actually observe k ≠ 1: `tsdf_export`
  (all resolvers, incl. Stray/PGSR when used as INTEGRATION sources),
  `mv_consistency` (through the tsdf resolver), `autoprompt/session_builder`,
  `phase3_findings/detect`, `nvdiffrast_bake` (texture-bake occlusion).
  **DECISION (boundary):** strictly intra-reconstruction readers
  (`scale_align`/`scale_model` estimation, `vggt_tracks`, `densify_fillers`,
  `dense_pose_fusion`, `run_bundle_adjust`, `run_colmap_ba`, `trace_normals`
  at consolidate time, map_worker DA3 priors) are NOT wrapped: the sidecar
  cannot exist while they run — pipeline Replace deletes it before the stage
  starts (new geometry = epoch 0) — and touching the user-validated
  reconstruction recipe for a structural no-op adds risk with zero effect.
  `pgsr_cloud`/`pgsr_render` artifacts are instead epoch-stamped and badge
  stale after a correction (regenerating them re-reads corrected inputs).
- **Never** applied to DA3 anchor depth (`da3_run/results_output` in
  scale_align/scale_model) or Stray LiDAR PNGs read as external measurements —
  those are the independent witnesses the scale gate checks against. (When DA3
  depth is used as a TSDF *integration source*, it IS corrected — the mesh must
  land on the corrected cloud.)

## 9. Ledger & replay

- `output/corrections.jsonl`, append-only. Two record types:
  - `{"type":"run", "correction_id", "epoch_from", "epoch_to", "kind":
    "objects"|"floor", "operator", "created_at", "instance_ids", "visits",
    "observability", "diagnosis", "anchors":[{kf, frame, rot_deg, t, k}],
    "gates":[...], "overrides":{...}, "verdict":"pending",
    "algorithm_version"}`
  - `{"type":"verdict", "correction_id", "verdict":"approved"|"undone",
    "operator", "at"}`
- Exact transform per epoch: `output/corrections/epoch_<N>.npz` with `R_kf
  (n,3,3)`, `t_kf (n,3)`, `k_kf (n,)`, `frames (n,)` (real frame numbers → replay
  re-keyable onto a re-reconstruction).
- Mirror: after each append, the full ledger tail is written to the instance
  store `scene_meta["corrections_ledger"]` (survives store rebuilds by being
  re-mirrored on the next append; the jsonl is authoritative).
- `python -m correction.replay --session <session_dir> --to-epoch N` rebuilds
  cloud+poses from `_epoch_0`-equivalent inputs by applying approved epochs in
  order; test asserts float-tolerance reproduction.
- **DECISION (operator):** decoded from the JWT the UI now attaches; a missing
  token is recorded as the explicit literal `"unauthenticated"` — visible, never
  guessed, never blocking local use.

## 10. Configuration (`config.yaml` → `correction:`)

As specified in the prompt §6, loaded into typed dataclasses; **no default in
code** — a missing mandatory key fails at load naming the key. Current code
values become the YAML defaults (inventory §4); new gates calibrated on the
synthetic suite: `heldout_floor_tol_m: 0.05`, `heldout_floor_abs_m: 0.08`,
`heldout_object_tol_m: 0.10`, `max_step_mm: 25`, `max_step_deg: 0.5`,
`scale_agree_tol: 0.05`, `scale_mad_tol: 0.03`, `max_rot_deg: 10`,
`max_translation_m: 3.0` (MEJORAS §3 plausibility caps),
`pca_ratio_planar: 0.05`, `pca_ratio_cylindrical: 0.15`, `visit_gap_kf: 2`.
`runtime.workers: auto` = `min(8, len(os.sched_getaffinity(0)))` (CLAUDE.md
environment lesson).

## 11. API changes

Removed (with their UI callers): `POST /api/segmentation/chunks/apply_transform`,
`GET /api/segmentation/chunks/boxes/{sid}`, `POST /api/segmentation/chunks/
align_floor`, `POST/GET /api/segmentation/correction/{run,undo,approve,status}`
(handlers `main.py:4937-5142`).

New router (all task-backed, all send/decode Authorization, all 409 on
lock/pending conflicts):

| Route | Purpose |
|---|---|
| `POST /api/correction/run` | `{session_id, instance_ids, override_scale_check?}` → staged run, returns report (status pending or rejected) |
| `POST /api/correction/floor` | `{session_id, model: level\|plane\|profile, keyframes?: [..] \| "auto"}` |
| `POST /api/correction/approve` / `POST /api/correction/undo` | verdicts (executor-run, task-tracked) |
| `GET  /api/correction/state/{sid}` | epoch, pending status, latest report |
| `GET  /api/correction/ledger/{sid}` | full ledger for the history view |
| `GET  /api/correction/artifacts/{sid}` | derived artifacts + their epoch + stale flag |

## 12. UI changes (`ui/src`)

- **Delete**: chunk panel (`App.tsx:4084-4230`), 📦 button (`:2596-2606`),
  chunk states (`:127-133,180`), `loadChunkBoxes` + push effect (`:318-330`),
  Viewport chunk gizmo (`Viewport.tsx:152-159,498-597,2166-2223`), the
  `onChunkSelected`/`onChunkDelta` props and wiring (`App.tsx:3202-3203`).
  Evaluation-volume gizmo (`selVolume`/`volTcRef`) and placed-object gizmo
  untouched.
- **Correction panel** (single flow): select marked instances → *Analizar y
  corregir* → per-stage progress from `/api/tasks/{sid}` (`task_type:
  "correction"`) → report panel (visits per object, diagnosis, observable DOF,
  residuals before/after, scene exam, scale vs DA3, continuity, every gate's
  verdict; on rejection: the reason + "marcá además …" suggestion). Approve /
  Deshacer only when `pending`.
- **Floor alignment**: model selector (level/plane/profile, one explanation
  line), anchor scope ("all qualifying keyframes" default), same report panel.
- **History view**: ledger table (epoch, kind, operator, verdict, overrides).
- **Stale badges**: `GET /api/correction/artifacts` drives a per-artifact
  `stale` badge (meshes, surface_fit, BIM sabana, coverage) + Regenerate where
  an endpoint exists (per-object meshes, surface_fit); BIM offers re-register.
- Viewer reloads Potree/poses only on the post-swap `potree_ready` broadcast.

## 13. Tests (`server/tests/`)

Shared generator `server/tests/synth_correction.py`: keyframed trajectory,
floor (flat/ramp/step), objects (plane wall, cylinder, asymmetric box), two
visits, provenance-complete PLY + poses + frames + `segmentation_result.json` +
synthetic `scale_diagnostics.json` (+ optional `omega_run` npz and
`chunk_plan.json`); injectors for progressive rigid drift, depth compression k,
and both. Potree rebuild monkeypatched (separate failure test).

| File | Covers (§10 of the prompt) |
|---|---|
| `test_correction_recover.py` | rigid / k / both recovery within tolerance |
| `test_correction_observability.py` | planar-only DOF, cylinder axis, no k without evidence — unobservable DOF untouched and declared |
| `test_correction_gates.py` | scene-exam veto leaves every file untouched; plausibility caps (box1 replay); continuity ≤ max_step |
| `test_correction_scale.py` | k vs DA3 anchors: reject w/o override; override applies + ledger record |
| `test_correction_downstream.py` | corrected depth accessor reprojects onto corrected cloud (~0 residual); raw==cleaned same transform; globalIndices stable; epoch in derived metas |
| `test_correction_tx.py` | Potree failure → zero file changes; undo/approve exact states; concurrent run → 409 |
| `test_correction_ledger_replay.py` | N corrections incl. one undone → replay reproduces final epoch; undone entry preserved |
| `test_correction_floor.py` | plane keeps ramp slope; level flattens (declared); real step preserved |
| `test_correction_config.py` | incomplete/out-of-range config → load error names the key; static scan: no decision literal outside config.py (whitelist: 0/1/identity/indices) |
| `test_correction_units.py` | no plan → keyframes; with `chunk_plan.json` → real ranges/overlap; no `// 30` anywhere in the package |

Existing suites (`test_pose_refine`, `test_scale_v2`, `test_surface_fit_*`,
`test_mesh_export`, `test_native_depth`, `test_mv_consistency`) must stay green;
`session_io` additions are additive.

## 14. Delivery order (Phase 3)

1. `map_worker` chunk-plan persistence + `session_io` depth accessor (additive).
2. `correction/` package core (config→session→units→evidence→observability→
   diagnose→solve→distribute→gates→floor→report).
3. `apply`/`ledger`/`replay`/`invalidate`/`epoch` + `potree_converter` override.
4. `api.py` + `main.py` swap (delete old endpoints + module) + epoch stamping in
   downstream writers + depth-accessor migration.
5. UI: delete gizmo, new panel/history/badges.
6. Tests + docs (README, CLAUDE.md decision entry, ARCHITECTURE, MEJORAS).

Commits per functional unit, messages citing the findings (H1…H10) they close.
