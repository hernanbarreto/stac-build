# Final reconstruction pipeline — STATE OF 2026-11-08 (read this first in a new session)

THE PIPELINE IS FINAL. Validated by the user on pccr 2026-08-31 (main `8cef74f`): *"la cobertura es muy buena, sin
errores, sin duplicados, muy muy buen resultado, excelente, que quede este como pipeline final"*. Nothing in it
changes without his word. This file says what each stage does and what is still open; how it got here is in the git
history of this file and of CLAUDE.md (the 2026-10-17 version of this file: `git show 2f13136:docs/pipeline_final.md`).

## 1 · The pipeline ("Reconstruir"), stage by stage

`pipeline_manager.DEFAULT_STAGE_ORDER`: RECONSTRUCTION → CLOUDCOMPY → GENRECON → VLM → SAM3 → PROJECTION → FLOOR →
CERTIFY. GenRecon runs only when ordered from the Autosegment window; the certification is OFF (§1.9).

### 1.1 Intake (`intake/`, inside the reconstruction stage, `workers/map_worker.py`)
- **I0 quality** — per-frame sharpness / luma / inter-frame difference (8 CPU threads).
- **I1 parallax keyframes** — pyramidal LK + homography RANSAC + rotation fit; a keyframe every 12 px of parallax
  (quantum) and one by ROTATION every (1 − τ)·FOV_h / 4 (pccr 10.8°); witness frames in between.
- **I3 DA3 to measure** — the focal probe (16 frames → the session K) and DA3 windows (metric depth for the gauge
  and the anchors; deleted after the chain, regenerated on demand). I2 (content tags / exclusion masks) is REMOVED.
- Frames are SEALED by `frames/manifest.json` (sha256 of every frame); every step is stamped and reused only on an
  identical stamp (frames' bytes, code, parameters, environment).

### 1.2 Structure (inside the reconstruction stage)
- **AMB3R-SLAM over the WHOLE video at stride 1** (`reconstruction/amb3r_probe.py` → `intake/amb3r_video.npz`; env
  `amb3r`, fork `--stac_light_map`): AMB3R decides poses only — its backend reconstructs ~1 frame in 4 with what each
  saw (depth every 4 px, K, c2w through its submap's Sim(3)); the rest are interpolated. It is the coverage map.
- **Coverage fill** (`reconstruction/coverage_fill.py`): AMB3R frames are added as keyframes (`closed_by:
  coverage`) while a frame brings points no keyframe sees, or sees from one view only (a second view ≥
  `refine.min_tri_deg`, 2°), beyond the depth test's own miss rate. pccr: 295 + 18 = 313 keyframes.
- **Turn frames** (`intake/turn_frames.py`): between consecutive keyframes the turn is cut into steps ≤ θ_q/3·1.25,
  the sharpest frame per step — Omega CONTEXT views (slots dropped) and F6 witnesses.
- **Cubes** (`reconstruction/cubes.py`, version 9): along the walk, units while their cameras fit
  `structure.cube_m` (5 m) AND ≤ `structure.chunk_view_cap` (64) Omega views, at least two units per cube,
  neighbours sharing HALF their keyframes. `kf_covisibility` (in `intake/cubes.json`): every pair of keyframes where
  one sees ANY part of the other's view on one surface (c > 0, τ 0) — read by F4's revisits and F6's witnesses.
- **Omega per cube** (fork `vendor/VGGT-Long`, `loop_utils/cube_run.py`): one VGGT-Omega pass per cube at the
  largest patch-aligned resolution at which the largest cube fits the card (`max_resolution` 1024); the DA3 metric
  lock per cube; exact Sim(3) seams on the shared frames; ONE Sim(3) graph over the cubes on the SEAMS ALONE
  (`robust_seams=False`, no DA3 prior: the priors broke seams by 16–33 cm). Then `scale_align` + orientation.

### 1.3 The precision chain (`precision/runner.py`, inside the reconstruction stage)
| Step | Module | What it does |
|---|---|---|
| F0 | `precision.camera` | the session camera (K + lens) |
| F2 | `precision.gauge` | the continuous metric gauge: scale s along the walk from the best instrument by held-out (DA3 windows vs mono) — no step at a cube transition |
| F4 | `precision.tracks` | native-pixel tracks over keyframe windows, REVISIT windows (pairs of `kf_covisibility` the walk does not tie) and the witnesses; held-out split |
| F3 | `precision.omega_probe` | Omega resolution probe — report only |
| F5 | `precision.refine` | joint refinement (torch LM on the card): poses, camera, landmarks; rungs R0 → R3, each TAKEN only when the held-out improves beyond 1.1 × its own error with ≥ 5 judges (`metric_lock.decide_change`) |
| F6 | `precision.depth_on_f5` (`f6_bend`) | THE PRODUCT — see 1.4 |
| check | `precision.chunk_check` | floor / ceiling / camera per keyframe and per cube — report only |

### 1.4 F6 — the cloud (`precision/depth_on_f5.py`, `precision/f6_torch.py`, `cloud.source: omega_bent`)
1. **Bend**: each keyframe's Omega depth × k(u,v) = c0 + c1·u + c2·v fitted to F5's landmarks (10-step Huber IRLS;
   c1/c2 only ≥ 2× their fit error; c0 verified on the keyframe's own held-out rows).
2. **PointDiT detail at the EDGES only** (`precision.mono_detail`, `detail_scope: edges`): within 12 px of a
   discontinuity and within τ; mixed edge pixels resolved to the front or back surface; surfaces stay Omega.
3. **The vote**: witnesses = the walk neighbours (±3/6/12), the turn frames, and EVERY keyframe that sees any part
   of the keyframe's view (no cap, any distance but the offsets, the widest baseline first). A vote counts only when
   it TESTS the depth — the witness sees the point from ≥ 2° away from the keyframe's camera (`two_sided_vote
   (min_parallax_deg)`). Contradicted pixels leave or are repaired to the witnesses' median.
4. **The publication floor**: a pixel enters only with ≥ `precision.cloud.min_support_views` (10) depth-testing
   views (USER 2026-11-07: *"prefiero poco y bueno que mucho y mierda"* — never loosened for coverage). The cloud's
   `confidence` column is that support. Plus the ONE confidence floor (`conf_min_norm`, min-max per cube) and sky.
5. The cloud stage's cleaner (voxel 5 mm + SOR), published as an epoch with its octree (transactional).

### 1.5 CLOUDCOMPY
The published cloud goes to the viewer (`cleaned_cloud.ply`, Potree; PotreeConverter on ONE thread for byte
identity). A run without *Segment when done* ends here (plus the floor, §1.8).

### 1.6 GENRECON — manual only (`workers/genrecon_worker.py`, `server/tools/genrecon/`)
Ordered from the Autosegment window with the viewer's confidence value (`pipeline.auto_genrecon: false`, USER
2026-11-05; with the final pipeline's cloud he orders it at 0 %: *"ya todo viene muy filtrado"*). Mode `cloud`: every
3.7 m chunk's active voxels (11.6 cm) are the published cloud's — a voxel is active from FIVE points
(`genrecon.min_points_per_voxel: 5`, USER 2026-11-08 *"vayamos con 5 puntos"*: against one point, 5 drops 4.5 / 9.9 /
6.7 % of the voxels of pccr / observatorio / zaragoza and the old 10 dropped 7.2 / 16.1 / 10.9 %, 97–99 % of them at
a surface's edge); the cubes sit on the
segmentation's floor / ceiling levels; the keyframes are BAKED into the generated mesh's own UV atlas
(`photo_albedo.py`) → the product `genrecon_photo` (USER 2026-10-17: *"la mejor que he visto hasta ahora"*). The
generative variant is OFF (*"inventó cualquier cosa"*); modes `cloud+hull` / `cloud+objects` / `cloud+unseen` exist,
never default.

### 1.7 VLM → SAM3 → rounds → PROJECTION
- **VLM** (`workers/vlm_worker.py`, `segmentation/autoprompt/`): Qwen3-VL in the job's OWN vLLM (frozen config,
  deterministic), ONE call per keyframe, prompts per CONCEPT; true synonyms merged; every phrase's fate in
  `output/segmentation_census.json`; at most `autoprompt.max_sam3_prompts` (150).
- **SAM3** (`workers/sam3_worker.py`, `segmentation/sam3_group.py`): concepts in groups sharing each chunk's vision
  backbone; strict deterministic; an out-of-memory or the vendor tracker's own stop ("No points are provided") is a
  STOP continued in a fresh session, never a failure.
- **Rounds** (`segmentation/autoprompt/rounds.py`, USER 2026-10-08/09 *"las pasadas son recursivas"*): the VLM sees
  every keyframe with what is segmented painted grey and names what is left; SAM3 appends those prompts; until a
  round names nothing new. pccr 2026-11-08: 76 concepts + 3 + … over four rounds.
- **Projection** (`segmentation/pipeline.py`): the masks onto the published cloud ONCE, after the last round; OBBs;
  32-bit instance ids. pccr 2026-11-08: 491 segments.
- VLM and SAM3 are KEPT through Replace unless the keyframes changed (`pipeline_manager.
  _semantics_keyframes_change`).

### 1.8 FLOOR — automatic, the last step of every reconstruction (USER 2026-11-05)
`correction/floor.py solve_floor_session` (`correction.floor.model_default: session`): ONE rigid motion for the whole
session, measured on the SEGMENTED support surfaces (labels whose head noun is in `autoprompt.support_surfaces`):
levels searched along the support's own normal, THE floor = the level under the most cameras, other real levels keep
their step. Without segmentation: the points below each keyframe's camera, declared. pccr 2026-11-08: tilt 3.27°,
height −26.0 cm → epoch 4.

### 1.9 CERTIFY — OFF
`certify.auto_after_segmentation: false` (USER 2026-11-05: *"nube genrecon vlm+sam3 fin, así tiene que ser"*). The
code (closures, depth per chunk, mask filter, chunk check on the certified depth) stays, unused.

## 2 · Measured on pccr 2026-08-31 (the validated run, 2026-11-08)
313 keyframes (295 I1 + 18 coverage), 2,111 witnesses, 53 turn frames; 11 cubes of 40–64 keyframes, Omega at 832
(grid 464 × 832); walk 17.5 m (I3 17.7 m). F2: s 0.915–1.226 along the walk. F4: 16.5 M observations, 2.69 M
tracks, σ 1.27 px (23.7 min). F5: R2 applied, held-out 0.462 px, witnesses 1,399 / 1,790 within 2.14 px (19.3 min).
F6: 19,540 co-visible witnesses, a median of 53 per keyframe (max 109); vote kept 62.4 % / contradicted 16.0 % /
repaired 15.7 % / rescued 6.0 %; COVERAGE 63.6 %; 76.9 M raw points → 11,462,333 (12.5 min). Floor thickness median
2.9 cm (p90 8.9), cameras 1.35 m above it. Segmentation 491 objects; floor epoch 4. From F5 to the end: 2 h 38 min.

## 3 · Rules the pipeline lives by
- **Determinism** (USER 2026-10-07: *"STAC Build es una herramienta de ingeniería, debe dar siempre lo mismo"*):
  bit-for-bit; the plan is `docs/plan_determinismo.md` (166 points, cited by the code). Mechanisms: `server/repro.py`
  (strict deterministic torch, `require_exclusive_gpu` — a co-tenant FAILS the GPU step, `stamp`/`check_stamp`,
  `environment_record`), per-card constants committed in `server/card_table.json`, DA3 weights pinned by revision +
  sha256, one HF cache, PotreeConverter on one thread, clocks only in `*.timing.json`.
- **The user's decision rule** (one function, `metric_lock.decide_change`): a change is applied only when (a) the
  95 % CI of the paired change lies entirely on the improving side, (b) ≥ 5 judges, (c) the median improvement ≥
  `correction_graph.graph.improvement_error_factor` (1.1) × the measured error.
- **Resume**: Replace OFF resumes — every step reuses its products only on its stamp; Replace ON wipes to the frames
  and the video. The job's configuration is frozen in `output/run_config.yaml`.
- **Storage**: a session keeps what the next reader needs (aligned chunks, bridges, I3 windows deleted once read).
- **The floor is geometric** (USER 2026-10-08): the support the camera walks over, whatever its name; a session may
  hold several levels, a real change of level is kept as a step.

## §10 · Epochs
The epoch NUMBER is a counter (F2, F5, F6 and the floor each take one: pccr 2026-11-08 published epoch 3 and the
floor epoch 4). `certify.single_final_epoch: false`: the published cloud's epoch stays stored and selectable next to
the floor's. Renumbering to 0 / 1 is pending (list A).

## 4 · PENDING
### A · Reconstruction pipeline
1. **Capture distance** (USER 2026-11-07, *"me lo van a preguntar"*): the recommended minimum and maximum distance to
   the target (he expects the maximum ≤ 10 m) — per session F6's held-out error in metres by range, the support
   floor's pass rate by range, the published points by distance to their birth camera. Measure when he says.
2. **Weighted vote** (his question 2026-11-08): answered, not built — his word.
3. **Smarter segmentation** (USER 2026-10-04): one physical object still comes out as several segments under
   different names; the consolidation before SAM3 and/or after the projection is open.
4. **Epoch renumbering** to 0 / 1 (§10).
5. **GPU** (USER 2026-11-01): the intake I0 / I1 on the card (a GPU LK / RANSAC is not cv2's arithmetic — the
   keyframes can change: needs an A/B of the selection or his OK); the SAM3 group A/B (masks equal, time) and the
   pose graph's card equality still to measure; the Potree builds off the critical path.

### B · General
1. `docs/MEJORAS_OBLIGATORIAS.md` (USER 2026-09-06, *"obligatorio todo"*): item 1 — the per-point membership out of
   `segmentation_result.json` (`membership.py` never built; `globalIndices` still in 73 files; zaragoza's file was
   1 GB and dropped the viewer socket) — and item 2 (I/O and Potree speed) are NOT done.
2. Lock the brush during an epoch publish (`correction.apply.publish_in_flight` exists; the erase endpoint must
   refuse while it returns a transaction).
3. BIM comparison: register to the BIM on the structure; per element deviation of position / dimensions / flatness /
   plumbness with its margin; objects present / missing / displaced; a deviation below the session's own
   repeatability reported as not measurable.
4. GenRecon: closing the gaps of 1–2 voxels surrounded by surface that NO keyframe saw through (doors and windows
   stay open) — to test on his word (USER 2026-11-08: *"lo probamos luego"*); the model's own presence stays out
   (`cloud+unseen`: *"no"*). The projected texture (`scene_texture.py`) is parked; outdoor sessions get a painted
   "ceiling" at the cube's top (declared).
5. Viewer: the 2026-10-01 additions to validate by eye (gravity sandbox, reference human, rename / delete of placed
   objects, search, fly-to); shadows step 2 (the cloud casting shadows — measure the fps cost first).
6. Pre-existing broken test `segmentation/autoprompt/tests/test_associate.py::
   test_adaptive_sampling_scales_with_camera_path` (its fixture lacks `autoprompt.merge_synonyms`).
