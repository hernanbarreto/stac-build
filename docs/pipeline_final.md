# Reconstruction pipeline — STATE OF 2026-11-09 (read this first in a new session)

The pipeline validated by the user on 2026-11-09 on two sessions: zaragoza (*"salió bien, lo único la columna
partida"*) and pccr 2026-08-31 (*"bien, pccr validado visualmente, ok"*). README.md holds the mathematics of every stage;
this file says what each stage does with today's settings, the numbers of the reference runs, and what is still open.
How it got here is in the git history of this file and of CLAUDE.md (the 2026-11-08 version: `git show
04c58bc:docs/pipeline_final.md`).

## 1 · The pipeline, stage by stage

`pipeline_manager.DEFAULT_STAGE_ORDER`: RECONSTRUCTION → CLOUDCOMPY → VLM → SAM3 → PROJECTION → FLOOR → CERTIFY →
GENRECON. A **Reconstruir** order runs RECONSTRUCTION, CLOUDCOMPY and FLOOR (the semantic stages when *Segment when done*
is ticked); the **Autosegment** window orders any subset of VLM, SAM3 and GENRECON; CERTIFY is off.

**One range for every view** (`reconstruction.view_range_m` = 15 m, `server/view_range.py`, USER 2026-11-09: *"lo
dejamos en 15 m y vemos cómo se comporta"*): a point farther than it from the camera that measured it is born by no
keyframe and weighs in no fit, pairing, scale, closure, co-visibility or statistic of any stage.
**Angles, not pixels**: the parallax, F4 and F6 thresholds are angles read in each stage's pixels (fx · tan θ).

### 1.1 Intake (`intake/`, inside the reconstruction stage, `workers/map_worker.py`)
- **Frames** extracted with their long side ≤ 1024 px (`frames_max_long_side`), sealed by `frames/manifest.json`.
- **I0 quality** — per-frame sharpness / luma / inter-frame difference; only exposure rejects.
- **Focal probe** — 16 frames through DA3 at native resolution → the session K; its depth maps are kept
  (`intake/focal_depth.npz`) as AMB3R's metre.
- **I1 parallax keyframes** — pyramidal LK + rotation fit; a keyframe every 1.774° of parallax (`parallax_quantum_deg`,
  12 px at the calibration focal) or every (1 − 0.3)·FOV_h / 4 of rotation, the sharpest frame of the ±25 % band;
  **never more than 8 frames without a keyframe** (`max_window_frames` 8, AMB3R's own completeness rule:
  `parallax.stride_fill`); witness frames every 0.222° of parallax.
- Every step is stamped and reused only on an identical stamp (frames' bytes, code, parameters, environment).

### 1.2 Structure (inside the reconstruction stage)
- **AMB3R-SLAM over the WHOLE video at stride 1** (`reconstruction/amb3r_probe.py` → `intake/amb3r_video.npz`, env
  `amb3r`, heavy model VGGT-Ω): every frame's pose; ~1 frame in 4 reconstructed with what it saw. **Put in metres** once
  by DA3 on the focal probe (`metric_scale`: per frame the median DA3/AMB3R depth ratio over its nearest 25 %, the
  median of the frames in the 10–90 % band; pccr 2026-08-31 2.638 m/unit, MAD 6.3 %).
- **Coverage fill** (`reconstruction/coverage_fill.py`): frames are added as keyframes while one brings points no
  keyframe sees, or sees from one view only (a second view ≥ 2°), beyond the test's own miss rate; points beyond the
  range do not count.
- **Turn frames** (`intake/turn_frames.py`): VGGT-Ω context views through the turns, and F6 witnesses.
- **I3** — DA3 windows (≤ 1024 px) chained into the walk; the metric anchors per keyframe.
- **Cubes** (`reconstruction/cubes.py`, version 10, `structure.cut: covis`): a cube takes units along the walk while
  Σ 1/ℓ ≤ 13.67 (ℓ the keyframe's co-visibility length) and ≤ 64 keyframes (`chunk_view_cap`; turn frames ride along);
  neighbours share half their keyframes; a cube under 32 keyframes (`min_chunk_keyframes`) merges into its neighbour.
  `kf_covisibility`: every pair of keyframes where one sees any part of the other's view on one surface.
- **VGGT-Ω per cube** (fork `vendor/VGGT-Long`, branch `gpu-main`, `loop_utils/cube_run.py`): one pass per cube at
  the largest patch-aligned resolution ≤ 1024 at which the largest cube fits the card; the DA3 metric lock per cube
  (within the range); **seams** = exact Sim(3) of the shared frames, always free scale; the session's scale bound
  (1.1 × the 95 % interval of the seams' log-scales); **revisit bridges**: F4's revisit pairs between cubes that share
  no frame, one joint VGGT-Ω pass over F4's loop window at each end, fitted at once onto the two locked cubes — only
  the fits are kept (`bridge_fit_*.npy`); closures within the bound enter **one Sim(3) graph** under Cauchy, the seams
  squared. Then `scale_align` (residual metre check) and the upright orientation.

### 1.3 The precision chain (`precision/runner.py`, inside the reconstruction stage)
| Step | Module | What it does |
|---|---|---|
| F0 | `precision.camera` | the session camera (K + lens), frames checked against the extraction cap |
| F2 | `precision.gauge` | the continuous metric gauge along the walk (DA3 windows vs mono, by held-out); refuses unless epoch 0 |
| F4 | `precision.tracks` | native-pixel tracks: keyframe windows, revisit windows, witness windows; 20 % held out |
| F3 | `precision.omega_probe` | VGGT-Ω resolution probe — report only |
| F5 | `precision.refine` | joint refinement (torch LM): poses, camera, landmarks within the range; rungs R0 → R3, each taken only when the held-out improves beyond 1.1 × its own error with ≥ 5 judges |
| F6 | `precision.depth_on_f5` | THE PRODUCT — see 1.4 |
| check | `precision.chunk_check` | floor / ceiling / camera per keyframe and per cube — report only |

When the chain must start again from F0 and no finished step has moved the geometry, the runner first returns to epoch
0 from the lineage the publish kept in `output/_restart_epochs/` (`correction/restart.py`, `runner.needs_epoch0`).

### 1.4 F6 — the cloud (`precision/depth_on_f5.py`, `precision/f6_torch.py`, `cloud.source: omega_bent`)
1. **Bend**: each keyframe's VGGT-Ω depth × k(u,v) = c0 + c1·ũ + c2·ṽ fitted to F5's landmarks (Huber IRLS; c0
   verified on held-out rows; landmark rows within the range).
2. **PointDiT detail at the edges only** (`mono_detail`): within 1.807° of a discontinuity (12 px at the calibration
   focal), within τ; surfaces stay VGGT-Ω's.
3. **The vote — COLMAP's fusion rule** (`cloud.fusion_rule: colmap`). Witnesses: the walk neighbours (±3/6/12), the
   turn and interval frames, and every keyframe that sees any part of the view. A witness confirms when the
   forward-backward reprojection is within 0.301° (`fusion_max_reproj_deg`), the depths within the session's measured
   τ, the triangulation angle in [1.5°, 30°] (`fusion_tri_min_deg` 1.5 since 2026-11-09) and the point within the range
   of it. **Kept**: born within the range, ≥ 5 confirmations (`min_support_views`), more confirm than see through it
   (`fusion_majority`), not hidden behind a better-supported surface (`fusion_occlusion_veto`).
4. **In place** (`fusion_in_place`, 2026-11-09): a pixel not kept whose in-range witnesses' surfaces agree (≥ 5 within
   τ of their median) with its OWN depth within τ is published where it is — never moved. **No repair**
   (`fusion_repair: false`: moved pixels painted their birth colour elsewhere).
5. The cloud's `confidence` column is the support (confirmations, or the agreeing witnesses for in place).
6. **The cleaner** (`reconstruction/stream_clean.py`, by pieces, byte-identical to the in-memory path): voxel 5 mm
   (nearest real point to each cell centre) + SOR (8 neighbours, σ 3, no neighbour within 1.5 cm → out); published as
   an epoch with its Potree octree (deterministic for any thread count).

### 1.5 CLOUDCOMPY
The published cloud goes to the viewer (`cleaned_cloud.ply`, Potree).

### 1.6 VLM → SAM3 → rounds → PROJECTION
- **VLM** (`workers/vlm_worker.py`, `segmentation/autoprompt/`): Qwen3-VL in the job's own vLLM (frozen config,
  deterministic), one call per keyframe, prompts per concept; every phrase's fate in `segmentation_census.json`; at
  most `autoprompt.max_sam3_prompts` (150).
- **SAM3** (`workers/sam3_worker.py`): concepts in groups sharing each chunk's vision backbone; strict deterministic.
- **Rounds**: the VLM sees every keyframe with what is segmented painted grey and names what is left, until a round
  names nothing new.
- **Projection** (`segmentation/pipeline.py`): the masks onto the published cloud once, after the last round; OBBs.

### 1.7 FLOOR — after every projection, and at the end of every Reconstruir (`pipeline.auto_floor: true`)
`correction/floor.py solve_floor_session`: ONE rigid motion for the session, measured on the segmented support surfaces
(head noun in `autoprompt.support_surfaces`); the floor = the level under the most cameras; other real levels keep
their step. Without segmentation: the points below each keyframe's camera, declared.

### 1.8 CERTIFY — OFF
`certify.auto_after_segmentation: false`. The code stays, unused.

### 1.9 GENRECON — ordered only (`workers/genrecon_worker.py`, `server/tools/genrecon/`)
`pipeline.auto_genrecon: false`; ordered from the Autosegment window or the GenRecon button with the confidence
threshold of that order (0 % on the reference runs). Mode `cloud`: cubes of 3.7 m (`genrecon.cube_size_m`), voxels of
11.56 cm; a voxel is active from **10** points (`min_points_per_voxel`, back to the original on 2026-11-09); one cube
base on the lowest measured support level, the support levels' columns filled solid down to the lattice base; the
keyframes baked into the generated mesh's own atlas (`photo_albedo.py`) → `genrecon_photo`. The keyframes' presence
vote is off (`presence_vote: false`). **Known issue** (CLAUDE.md §7): on zaragoza 36.5 % of the mesh area has no cloud
point within one voxel, almost all at floor level; main suspect, the solid fill.

## 2 · Measured on the reference runs (2026-11-09)

| | zaragoza (2026-06-03) | pccr (2026-08-31) |
|---|---|---|
| keyframes / turn frames | 279 / 17 | 547 / 53 |
| walk (I3) | 12.2 m | 17.8 m |
| cubes | 8 of 55–64 kf | 17 of 35–64 kf |
| VGGT-Ω grid (w × h) | 1024 × 576 | 464 × 832 |
| seams before the solve | σ 2.5–16.5 cm | σ 1.0–3.7 cm |
| revisits → bridges → closures | 134 → 115 → 82 | 75 → 49 → 43 |
| seams after the solve | 3.5 cm median, 11.2 max | 1.7 cm median, 4.8 max |
| F5 | R1, held-out 0.398 px, witness bound 42.24 px | R2, held-out 0.534 px, bound 2.26 px |
| F6 τ | 0.92 % | 1.43 % |
| F6 kept / in place | 36.4 % / 11.1 % | 55.3 % / 6.7 % |
| F6 coverage | 42.2 % | 59.9 % |
| raw → published points | 69.5 M → 23.3 M | 126.5 M → 18.8 M |
| core (F0 → published) | 51.6 min | 94.4 min |
| GenRecon (10 points per voxel) | 58 406 active voxels, 8.37 M faces, 60.9 % texels photographed | — |

## 3 · Rules the pipeline lives by
- **Determinism** (USER 2026-10-07: *"STAC Build es una herramienta de ingeniería, debe dar siempre lo mismo"*):
  `server/repro.py` (strict deterministic torch, `require_exclusive_gpu`, `stamp` / `check_stamp`,
  `environment_record`), per-card constants in `server/card_table.json`, pinned weights, one HF cache, the canonical
  octree, clocks only in `*.timing.json`. On 2026-11-09 the re-runs of zaragoza and pccr gave the same seams,
  closures and graph moves to the last printed digit.
- **The decision rule** (one function, `metric_lock.decide_change`): a change is applied only when (a) the 95 % CI of
  the paired change lies on the improving side, (b) ≥ 5 judges, (c) the median improvement ≥ 1.1 × the measured error.
- **Configuration**: read once when the order is given and frozen in `output/run_config.yaml`; a change to
  `config.yaml` reaches only later orders.
- **Resume**: Replace OFF resumes on stamps (a changed module re-runs every step whose code stamp includes it); Replace
  ON empties the scan to its frames, video and capture inputs.
- **Storage**: a session keeps what the next reader needs (aligned chunks, bridge predictions, I3 windows deleted once
  read; only the bridges' fits stay). The quota of the volume is 450 GB.
- **Failures**: stop, fix the cause in code, resume at the failed step (Replace only when what advanced cannot be
  recovered).

## 4 · PENDING
### A · Reconstruction pipeline
1. **GenRecon's invented floor-level surface** (above, CLAUDE.md §7): the test without the solid fill.
2. **MapAnything as the metric source** in place of DA3 (licence first; decided by measuring against known dimensions).
3. **Capture distance**: F6's error by range and the published points by distance to their birth camera, per session.
4. **Smarter segmentation**: one physical object still comes out as several segments.
5. **Epoch numbering** to 0 / 1.
6. **GPU** (USER 2026-11-01): the intake I0 / I1 on the card (needs an A/B of the selection or his OK).
7. **Station's heavy residual tail** (witness bound 37.04 px): split the held-out residuals by keyframe type, cube and
   region when he says.
8. The F6 test `test_in_place_publishes_a_surface_its_witnesses_agree_on_without_moving_it` takes minutes on the CPU;
   make it fast.

### B · Next fronts (USER 2026-11-09, CLAUDE.md §7)
1. **Fusion**: merge (scans of different areas into one cloud and mesh) and compare (two dates aligned on shared zones,
   position and scale).
2. **BIM comparison** against an IFC.
3. **Tickets**: review items and non-conformities placed anywhere in the project.
