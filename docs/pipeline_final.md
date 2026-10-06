# Final reconstruction pipeline — STATE OF 2026-10-04 (read this first in a new session)

This file is the single source of truth for: what the reconstruction pipeline IS today, what each
stage does, the user decisions behind it, and EVERY pending item (two lists at the end: A = the
reconstruction pipeline, B = general). Validated by the user on 2026-10-01 ("así está bien, todo lo
que describiste es correcto").

## Decisions (USER, 2026-09-30 / 2026-10-01 / 2026-10-04 / 2026-10-05)

**2026-10-05 (night) — SEGMENTATION ON DEMAND (USER: "un check para correr o no el vlm sam etc luego de la
reconstrucción … en la lista de segmentación un botón para autosegment … una ventana donde se puede ver el prompt
para vlm y segmentación, luego checkbox para el resto de etapas").** "Reconstruir" carries a per-scan check
*Segment when done* (OFF by default, USER): off, the run ends with the cloud in the viewer (`run_pipeline`
`segment`: the scan keys whose chain runs; `pipeline_manager.select_stages`). The Instances panel's **Autosegment**
button (also in the command palette) opens a window with the VLM prompt (scene understanding — the SAM3
vocabulary) and the session's SAM3 prompts, both editable, and a checkbox per stage: VLM, SAM3 + mask projection,
certification, object descriptions. The VLM prompt is SAVED IN THE SESSION (`output/autosegment.json`, USER) and
every later VLM pass of that session uses it (`AutoPrompter.understand_prompt` → `understand_frame(prompt=)`);
edited SAM3 prompts go into `vlm_analysis.json` `prompt` (what the SAM3 stage reads; a VLM run replaces them —
declared in the window). Run = `run_pipeline` with `autosegment` over the viewer socket: the chosen stages FORCED to
run (`start_pipeline(force=True)`: the resume probes are not consulted), nothing wiped, captions off through the
run's config. REST: `GET/POST /api/autosegment/{session}` (state / save prompts). Module
`segmentation/autoprompt/autosegment.py`, UI `features/AutosegmentDialog.tsx`; tests `tests/test_autosegment.py`.

**2026-10-05 (evening) — THE ORDER: "intake, da3 para medir, omega, f0 a f6, octree, época publicada, vlm, sam3,
máscaras, correcciones, época 1"; "chunks de 15 m siempre"; "olvidate de la exclusión de personas y objetos en
movimiento"; "como mucho la original y la final, ninguna intermedia".** Zaragoza (17.8 m) at 5 m chunks came out
in 6 chunks (metric lock spread ×1.39, pose graph held-out 12.5 cm and rejected, floor tilted 2.5–5.8° per chunk,
the certification then applied 2.64 m of translation) — "muchísimo peor que Omega en un solo chunk". Wired:
`chunk_walk_m: 15`; `intake.content.enabled: false` (I2 selectable, off); the VLM and SAM3 are pipeline STAGES after
the cloud stage again (`pipeline_manager.DEFAULT_STAGE_ORDER`: reconstruction → cloudcompy → vlm → sam3 → certify;
the reconstruction stage hosts no semantics; the cloud stage hands the published cloud to the viewer at once and its
resume probe no longer keys on the segmentation; the SAM3 stage projects its masks on the cloud on disk; the
certification follows); `certify.single_final_epoch: false` = the published cloud's epoch stays stored and selectable
next to the certification's (the reconstruction stage always discards its pose-only epochs, builds no Omega
comparison cloud). F6's mixed-pixel snap reads `seg_masks.npz` when present; without it (the new order) the mixed
pixels go to the vote as they are (declared in its log). ALSO SEEN on zaragoza: `segmentation_result.json` is 1 GB
at 111 M points and the viewer socket dropped while it was sent (the viewer showed no segmentation) — pending list B.

**2026-10-05 — STORAGE: a session keeps what the next reader needs, nothing else ("que no almacene al pedo … debe ir
limpiando"), after zaragoza (1080p) died at the second loop bridge with the /workspace quota full.** Measured: pccr held
18 GB, 12 GB of it dead (Omega's aligned chunks 6.5 GB + bridges 1.9 GB + uncertainty maps 0.3 GB kept by
`certify.keep_aligned_chunks: true`, the I3 gauge windows 2.6 GB read by F0 only); zaragoza 42 GB (24 GB of chunk
predictions, 8.5 GB of windows, 6.5 GB of two bridges). Wired: `certify.keep_aligned_chunks: false` (aligned chunks
deleted right after the omega-depth/scale step, bridges + `uncert/` at the end of the reconstruction — existing
cleanup code, now on); a bridge is saved and held WITHOUT its images (never read; a third of its 3.6 GB at 1080p);
`gauge.delete_windows_after_chain: true` (the chain's last step deletes `output/da3_windows/window_*.npz`; the plan,
walk.json and the anchors stay; `run_gauge` regenerates the files for a re-run from F0); the focal probe and the VRAM
probe delete their window depth once K / the footprint is written. Epochs are NOT the bloat: one live potree (0.8 GB),
`corrections/epoch_N.npz` ≤ 1.4 MB each. Kept on purpose: `omega_run` + `da3_run` records (the core's inputs),
`origins.npz`, `cleaned_cloud_raw.ply` (epoch transactions), `precision/tracks.npz`, `scene_r.db`, `seg_broadcast.json`.
**The bend's grid (zaragoza's second failure, 2026-10-05):** F5's ladder chose R2 (lens k1 −0.0035, held-out 0.784 →
0.777 px) and `f6_bend` refused a camera with distortion; Omega's records were also 1920×1088 for 1920×1080 frames. Fixed
at the root, not hidden: the bend works on the UNDISTORTED NATIVE frame (the chain's convention: corrected_cloud,
silhouette_filter, `pixel_u_und/v_und`); Omega's record, F5's landmark pixels and the SAM3 label maps are carried onto it
through the lens + record grid (`record_on_native`, nearest; `undistort_points`) when they differ; `f6_check` does the same
before unprojecting with K (its old guard checked x only). pccr (no lens, record on the camera grid) takes the old path bit
for bit. Report: `depth_on_f5.json` `grid_of_the_bend`. Measured on zaragoza: lens shift ≤ 3 px at the corners, 0 % uncovered.

**"Reconstruir" with the replace box OFF now RESUMES** (no wipe: the intake's step markers, DA3 depth on disk, the
fork's `[STAC resume]` chunk predictions / `loop_closures.txt` / `metric_lock.json` stamps, the core's step records
decide what is reused; everything derived is recomputed) — since 2026-07-11 a requested reconstruction wiped output/
unconditionally. The dialog defaults the box OFF when a scan shows cached Omega chunks (`maplong_run/
_tmp_results_unaligned`, now detected). With the box ON a reconstruction still leaves only the frames and the video.

**2026-10-04 — VALIDATED: epoch 8's recipe + PointDiT at the edges ("la reconstrucción está perfecta").** pccr epoch 5 of the
2026-10-04 run: f6_bend with `mono_detail.enabled: true`, `detail_scope: edges` (PointDiT's detail only within 12 px of a
discontinuity and within τ; surfaces stay Omega; mixed edge pixels resolved to front / back), floor levelled per chunk
(undulation 13.1 → 7.9 cm), 158 segments. The first variant (detail on every surface) was REJECTED by eye the same day:
bent walls, points shed off a rack. The second VLM pass over the unsegmented points is OFF (it added prompts and
segments). Next sessions run this recipe unchanged; the user is reconstructing a second scene with it.

0. **pccr EPOCH 8 is the best cloud and IS the pipeline** (*"valido el resultado de la época 8, es la
   mejor, incorporar al pipeline"*). Epoch 8 = epoch 7's depth adjustment EXACTLY (the bend) + the
   edge-keeping vote. History: epoch 7 (*"piso perfecto, voladores perfectos, la corrección de
   profundidad de una sutileza excepcional"*) → a first epoch 8 with a different bend window (±16, by
   consecutive-keyframe agreement) had a WORSE floor and was discarded (*"tenías que dejarlo como la 7"*)
   → epoch 8 rebuilt with epoch 7's exact bend + the edge vote = the validated one.
1. **Priority: object DEFINITION** — *"los objetos deben tener mucha definición, corte en los filos,
   las aristas"*. Old rules may be replaced where something measures better (*"olvidate de las reglas
   que pusimos en su momento"*).
2. **The cloud is Omega's depth BENT to F5 per keyframe + the edge-keeping vote — no plane sweep, no
   DA3** (DA3 layered objects: *"capas duplicadas y puntos voladores"*). F5 rung R1 (fx 392); the
   camera TRAVELS with the cloud (intrinsic.txt is an epoch artifact).
3. **The certification** de-duplicates objects and collapses the floor (*"no debe haber cebolla en el
   piso"*), respecting real steps and slopes; floor motion BLENDED across chunk overlaps (no camera step).
4. **ONE epoch: the final one** (*"debe quedar una sola época que es la final"*).
5. **Work one task at a time**, never several agents at once on the user's session; never change what
   works on a hypothesis — measure first, and reproduce validated numbers exactly when porting.

## The pipeline ("Reconstruir"), stage by stage

| # | Stage | Code | What it does |
|---|---|---|---|
| 1 | Intake I0, I1, I3 | `intake/`, `workers/map_worker.py` | frames, parallax keyframes, DA3 to measure (K, walk). I2 (content tags + exclusion masks) is OFF (`intake.content.enabled: false`, USER 2026-10-05) |
| 2 | VLM + SAM3 — **AFTER the cloud stage** (USER 2026-10-05) | `workers/vlm_worker.py`, `workers/sam3_worker.py` (pipeline stages `vlm`, `sam3`), `segmentation/autoprompt/` | ONCE, on EVERY keyframe (+2×2 crops); the same VLM call names the SAM3 prompts AND describes each kind for ShapeR (`vlm_analysis.json` `shape_descriptions`); the SAM3 stage projects its masks on the published cloud (stage 8) |
| 3 | Omega | `workers/map_worker.py` `_run_vggtomega`, planner `reconstruction/chunk_covis.py`, resolution `reconstruction/chunk_plan.omega_resolution_for` | chunks = the **co-visibility plan** (USER 2026-10-06, always, no switch): ONE pass when the whole walk is ≤ H = 13.67 co-visibility lengths, else variable-size chunks at 50 % overlap (each seam its own block), handed to the fork as `Model.chunk_ranges`, persisted in `output/chunk_plan.json` (+ covis report); Omega's resolution = the largest patch-aligned one at which the LARGEST chunk fits the card (native when it fits); poses, metric scale, per-keyframe depth + conf (`omega_run/results_output/frame_<f>.npz`) |
| 4 | F0, F2, F4 (F3 report) | `precision/` runner steps | session camera, continuous gauge, native-pixel tracks |
| 5 | F5 | `precision/refine.py` | corrected poses + camera (rung chosen by held-out; pccr R1, fx 364→392) |
| 6 | **Depth on F5 (`f6_bend`)** | `precision/depth_on_f5.py`, `precision.cloud.source: omega_bent`, config `precision.bend`; **mono detail** `precision/mono_detail.py` + `precision/pointdit_runner.py`, config `precision.mono_detail` (flag `enabled`) | (a) F5 FIT tracks triangulated with F5 poses + camera; (b) per keyframe Omega depth × k(u,v)=c0+c1·u+c2·v, fitted EXACTLY as epoch 7: 10-step Huber IRLS (`bend.irls_iterations`), landmark rows on depth > `bend.min_depth_m` (0.05), < `bend.min_rows` (20) rows → k = 1, window by half A of the HELD-OUT tracks (pccr ±0, 7.25 % → 2.80 %); (c) validity = the ONE confidence floor (`conf_min_norm`, min-max per Omega chunk) + not sky; (d) epoch 8's EDGE-KEEPING VOTE: τ = p75 of the neighbour disagreement on INTERIOR pixels (pccr 1.93 %), mixed pixels at contours snapped to their SAM3 mask's side, two-sided vote over ±3/±6/±12, contradicted pixels repaired to the neighbours' median when ≥ 2 agree, below-floor edge pixels admitted when confirmed; confidence column = agree count. pccr: 63.1 % kept, 15.1 % contradicted, 8.5 % repaired, 0.4 % admitted, coverage 71.0 %. **(c2) MONO DETAIL (USER 2026-10-04, `mono_detail.enabled`)** between the bend and the vote: PointDiT-H (DINOv3 ViT-H+/16, 16 Euler steps from zeros, deterministic) on native 512-px tiles with feathered overlaps → per tile z_cal ≈ s·z_mono + b (Huber IRLS weighted by the calibrated confidence, outside the band; tiles rejected by support or by the session's p95 residual) → z = lowpass(z_bent) + (z_al − lowpass(z_al)) with σ from Omega's patch; the discontinuity band (1-px jump > τ, dilated by context_scale) resolves Omega's MIXED pixels to the front or back surface by PointDiT's side, or marks them mixed_unresolved (out of the cloud). Metric and gauge are never touched (DA3's). Per-pixel provenance → cloud column `source` (1 omega_bent, 2 mono_detail, 3 band_front, 4 band_back); report `depth_on_f5.json` `mono_detail`; viewer layers View → PointDiT detail / Mixed unresolved. Flag off = epoch 8 bit for bit (tested) |
| 7 | Cloud publish | `precision/corrected_cloud.publish` | voxel + SOR, octree, transactional epoch; `corrected_cloud.json` carries per-frame s_k + bend; intrinsic.txt inside the transaction |
| 8 | Mask projection + OBBs | `segmentation/pipeline.py` (cloud stage) | class byte = `_encode_classification` + `class_map.json` (`class_byte` on every viewer payload); co-visible split (camera-return rule + free-space test); split children listed (`mask_fates.created_by_projection`); OBB yaw by RANSAC; every instance inherits its concept's `shape_caption` |
| 9 | Certification | `workers/certify_worker.py`, `reconstruction/certify/run.py`, `correction/` | depth per chunk; floor per chunk BLENDED across overlaps; mask filter with the cloud's own camera (`camera.json`), rule 4 by majority of views; NO MLS (`precision.cloud.consolidate: false`); chunk check on the CERTIFIED depth (composes the transform epochs) |
| 10 | Two epochs: ORIGINAL + FINAL | `certify.single_final_epoch: false` (USER 2026-10-05) | the reconstruction stage discards its pose-only epochs and Omega's chunks and builds no comparison cloud; the published cloud (stage 7) is the ORIGINAL and stays stored + selectable when the certification publishes the FINAL. The epoch NUMBER is a counter (F2, F5, F6 each take one: zaragoza published 3, certified 4) — the user wants to read them as 0 and 1; renumbering is pending list A |
| 11 | Object descriptions | `segmentation/object_captioner.caption_session_objects` (end of certify) | ONE Qwen3-VL call per object over its largest SAM3-mask views → `shape_caption` source `object` (config `segmentation.object_captions`) |

**On demand (Meshing dialog; search + select / deselect all, nothing preselected):** **Object** = ShapeR on every selected instance (keyed by `instance_id`; every face oriented OUTWARD by ray parity — `reconstruction/orient_outward.py` — before and after the texture bake; drawn in the viewer in its segment colour) (`/api/segmentation/
shape/export` → `segmentation/shaper_export.py` → `run_shaper_batch.py`, env `shaper`, preset max):
PKL views = every posed keyframe that SEES the object (occlusion test with the published depth +
`loops.witness.occlusion_tol_rel`), max 32; caption = manual > VLM object > VLM concept > label;
texture baked from the scan frames (`shaper.texture`); progress per object in the dialog.
**Mesh** = RANSAC + Poisson today → to become point2cad (pending B2).

**Run checks (automatic since 2026-10-04):** chunk check; `precision/cloud_metrics.py` = the floor metric (layer thickness + tilt-removed undulation per 1 m cell, every number in `precision.cloud_metrics`) after f6_bend (`chunk_check.json` → `cloud_metrics`) and on the certified cloud in the acta, where the crease-profile edge metric (`precision/edge_metric.py`, largest `edge_max_objects` objects) runs too; `output/precision/cloud_metrics.json`. pccr epoch 7 reference: 18.5 M pts, floor 5.2 cm thickness / 7.6 cm undulation. Diagnostics by hand: `python -m precision.flyers --session <dir>` (flyer classes a-d, viewer layer View → Flyers), `python -m precision.mono_ab --session <dir> --out <dir>` (the PointDiT A/B, Phase 8).

**Viewer fixes of 2026-10-01** (in main): the session load and every epoch's potree_ready send the
LIVE poses + intrinsic.txt (`main.py` `_camera_poses_payload`; maplong_run/ holds Omega's original
poses); segment visibility keyed by `class_byte`, reset when the class map changes between epochs;
the objects the projection creates are listed (their points could never be hidden).

## PENDING — A. Reconstruction pipeline (clean list)

| # | Stage | Status |
|---|---|---|
| 1 | Intake | done |
| 2 | VLM + SAM3 | done — fallback with category (2026-10-04). The second pass over the UNSEGMENTED points (`segmentation/second_pass.py`) is **OFF** (USER 2026-10-04: it ADDED 16 prompts and segments on pccr and the floor ended inside an `electrical_panel` segment). **PENDING — smarter segmentation (USER 2026-10-04): the same physical object comes out as 3-4 segments under different names (not the same pixels, the same object: 10 'door', 10 'panel', 11 'metal_frame' on pccr). The consolidation must happen BEFORE SAM3 (one concept = one prompt, `merge_synonyms` is not enough) and/or AFTER the projection (merge the segments that are one physical object, the VLM deciding on the views). One VLM pass per image stays the rule (SAM3 prompt + ShapeR description in the same call).** |
| 3 | Omega (co-visibility chunks, 50 %) | wired 2026-10-06 (USER: "el plan de chunk no cambia … para que encaje adaptamos la resolución") — pccr 2026-08-31 plans 5 chunks (lengths 42–139), zaragoza and observatorio ONE pass; zaragoza's 183 kf run at 1376 on a 48 GB card / 1808 on 80 GB. **Pending:** the user's eye on the first run (the walk-sized 15 m chunks are deleted) |
| 4 | F0, F2, F4 (F3) | done |
| 5 | F5 (R1) | done |
| 6 | Depth on F5 (`f6_bend` = epoch 8) | done — **reproduced 2026-10-04**: the product code on pccr's F5 files gives epoch 8's numbers exactly (held-out 2.80 % at ±0, τ 1.93 %, kept 63.1 / contradicted 15.1 / repaired 8.5 / admitted 0.4 %, coverage 71.0 %) — **pending:** it aborts when F5's camera carries lens distortion or Omega's grid is not the native one (pccr has neither); no fallback |
| 7 | Cloud publish | done |
| 8 | Mask projection + OBB + class byte + co-visible split | done — **pending:** desk #174's 3.14 m piece may still hold 2 desks |
| 9 | Certification | done — **pending:** the chunk check flags a floor drift INSIDE chunk 0 (+16 cm, kf 0–62) and chunk 1 undecided (+5 cm); no intra-chunk correction exists |
| 10 | One final epoch | done |
| 11 | Per-object VLM description for ShapeR | done |
| — | Run checks | done 2026-10-04 (`precision/cloud_metrics.py`, floor after f6_bend + floor and edges in the acta) |
| — | PointDiT (claude_stac.txt 2026-10-04) | Phases 0-8 built (flyers, runner, tiles, affine, detail/band, mixed pixels, provenance + layers, A/B). **`mono_detail.enabled: true` since 2026-10-04 by the USER's verdict on the maps** ("la nitidez de los bordes de pointdit es abrumadoramente superior a omega"). A/B pccr off → on: held-out 2.48 → 2.55 % (edge band 2.54 → 2.70 %, p90 29 → 39 %), contradicted 15.1 → 15.7 %, coverage 71.0 → 70.3 %, 18.42 → 19.29 M pts, mixed-edge flyers −30 %, +17 min (analysis/2026-10-04_pointdit/ab). First end-to-end run (2026-10-04): the detail over every surface bent walls and shed points off a rack while ducts and cables came out → **`mono_detail.detail_scope: edges` (in the pipeline since dc1f509)**: PointDiT's detail only within `detail_zone_px` (12) of a discontinuity and within τ; surfaces stay Omega. pccr: detail on 17.9 % of the pixels (was 85.5 %). **Pending:** the user's eye on this cloud |
| — | End-to-end validation | **pending:** the user relaunches pccr from scratch (restart the backend first) |

**VLM refinement (stage 2):** prompts per CONCEPT, never per object (the understanding prompt asks one entry per
kind; pccr's 43 prompts are all concepts); 100 % of the points segmented (a second VLM pass looks at what stayed
unsegmented and proposes the concepts it sees there — MEASURED 2026-10-04 on pccr's live epoch: 14.6 % of the points
have NO mask at their source pixel; today's 99.4 % coverage is the geometric growth of `_attach_unsegmented`, which
gives them a neighbour's label); the fallback retry carries the category (DONE 2026-10-04:
`session_builder.with_category`, "white tiled floor", never "white tiled").

## PENDING — B. General

1. **Meshing → Object (ShapeR):** validate on the fresh reconstruction (the texture frame on the
   ShapeR mesh, the view counts; pccr read-only check: backpack 15 views (was 12), monitor 271: 20).
2. **Meshing → Mesh → point2cad (NEW, USER 2026-10-01):** pressing Mesh (instead of RANSAC +
   Poisson) runs point2cad on the structure — floor, walls, pipes, columns, beams, ducts, chosen by
   GEOMETRY, never by label. To DEFINE before building: the full pipe SAM3 instance → surface regions
   (`segmentation/perfect_object.py`) → point2cad (`segmentation/p2c_object.py`, env `point2cad`;
   needs `.xyzc` = points + surface id; fits plane/sphere/cylinder/cone/INR, intersects into edges and
   corners, output normalized — inverted in p2c_object) → metric B-rep. License CC-BY-NC (internal
   use approved; a commercial deliverable needs a licence). Validated later by the user.
3. **BIM comparison** (§11): register to the BIM on the structure; per element deviation of position /
   dimensions / flatness / plumbness with its margin; objects present / missing / displaced; a deviation
   below the session's own repeatability (pccr ~5 cm) is reported as not measurable.
4. **Lock the brush during an epoch publish:** a brush reassignment while a publish ran left the
   viewer with no segmentation (pccr 2026-10-01). `correction.apply.publish_in_flight` exists; the
   erase endpoint must refuse while it returns a transaction.
5. **Pre-existing broken test:** `segmentation/autoprompt/tests/test_associate.py::
   test_adaptive_sampling_scales_with_camera_path` — its fixture lacks `autoprompt.merge_synonyms`.
6. **Shadows, step 2 (optional):** the point cloud CASTS shadows too (a second pass over millions of points
   from the light — measure the fps cost first). Step 1 is done: meshes cast + receive, points receive
   (`ui/src/components/shadows.ts`, View → Shadows).
7. **Viewer additions of 2026-10-01 to validate by eye on pccr:** the gravity sandbox (place with the mouse,
   grab and throw, the cloud is solid, spheres with the STAC Build logo), the reference human in Add
   object, ShapeR objects drawn in their segment colour, rename / delete of generated and placed objects,
   one search + show / hide all over every list, fly-to on every row, fixed-size dialogs.
8. Keep this file updated with every change of the above.

## §11 · Deliverables for the BIM comparison (USER 2026-09-30)

*"usar point2cad en lugar de ransac … y ShapeR para objetos, y comparamos lo que podemos"*. Raw points
are not a reliable basis (flyers, layers); the comparison uses reconstructed elements with their own
error. Structure → point2cad B-rep (B2); objects → ShapeR + texture baked from the scan frames
(unseen parts stay untextured). ShapeR generates what no camera saw: trusted for presence, position,
orientation and overall extent, not for detailed dimensions of unseen sides; every mesh carries
measured vs generated (`caption_source`, view counts in its `.meta.json`).

## Environment notes

- Pod: cgroup 117 GB RAM / ~30 CPUs — cap every job (`ulimit -v`, `OMP_NUM_THREADS`, `taskset`,
  `nice`); never fan out agents on the user's session. `memory.usage_in_bytes` counts the page cache
  (83 GB of it on 2026-10-05 with 4 GB of RSS) — read `memory.stat` `total_rss` for the real pressure.
- /workspace is a MooseFS volume with a RunPod quota (400 GB since 2026-10-05, was 300): `df` reports the
  cluster, not the quota, and no `mfs*` tool is installed — a full quota fails every write with
  `Errno 122 Disk quota exceeded` and the worker dies without a traceback (its log write fails too).
  Big tenants: miniforge3 180 GB, hf_cache 44 GB (`t5-v1_1-xl` 22 GB is ShapeR's text encoder — keep),
  vendor 29 GB, the sessions (see the 2026-10-05 storage decision).
- ShapeR on the A100 (sm_80): `torchsparse` in env `shaper` rebuilt from `nihalsid/torchsparse@20ccc92`
  with `TORCH_CUDA_ARCH_LIST="8.0;8.6"`, `SPHASH_INCLUDE`, `CUDA_INCLUDE`, `CUDA_LIB`.
- One Hugging Face cache: `HF_HOME=/workspace/hf_cache`.
- texrecon (`vendor/mvs-texturing`, the texture bake of every object mesh) was built against libtiff 5;
  the pod now has libtiff 6 only ("libtiff.so.5: cannot open shared object file", 2026-10-01). Relinked
  with its own `apps/texrecon/CMakeFiles/texrecon.dir/link.txt`, `libtbb.so.12.19` → `libtbb.so.12`
  (the vendored oneTBB holds 12.15; a full `cmake ..` fails on that mismatch).
- PointDiT: submodule `vendor/pointdit` (@ 11f53a3) with `third_party/dinov3` cloned inside and the weights under
  `pretrained/` (H and L 512 checkpoints, DINOv3 ViT-H+/16 and ViT-L/16; sha256 verified), runs in env `da3`
  (`torchmetrics` + `lightning-utilities` added `--no-deps`). Viewer: points drawn as cubes (View → Points as cubes).
- The pccr session on disk (2026-10-01): live epoch 8 (certified, 18.2 M pts, 110 objects), stored
  epochs 0 / 6 / 7 — hand-built, not what the code produces; a "Reconstruir" wipes it.
- Analysis scripts of the depth-source study: `analysis/2026-09-30_depth_sources/` (epoch 7 / 8
  recipes, floor metric, diagnostics).
