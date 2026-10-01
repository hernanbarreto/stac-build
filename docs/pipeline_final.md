# Final reconstruction pipeline — STATE OF 2026-10-01 (read this first in a new session)

This file is the single source of truth for: what the reconstruction pipeline IS today, what each
stage does, the user decisions behind it, and EVERY pending item (two lists at the end: A = the
reconstruction pipeline, B = general). Validated by the user on 2026-10-01 ("así está bien, todo lo
que describiste es correcto").

## Decisions (USER, 2026-09-30 / 2026-10-01)

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
| 1 | Intake I0–I4 | `intake/`, `workers/map_worker.py` | frames, parallax keyframes, content exclusion |
| 2 | VLM + SAM3 | `workers/vlm_worker.py`, `workers/sam3_worker.py`, `segmentation/autoprompt/` | ONCE, on EVERY keyframe (+2×2 crops); the same VLM call names the SAM3 prompts AND describes each kind for ShapeR (`vlm_analysis.json` `shape_descriptions`) |
| 3 | Omega | `workers/map_worker.py` `_run_vggtomega` | chunks of 5 m of walk, 50 % overlap (pccr: 6); poses, metric scale, per-keyframe depth + conf (`omega_run/results_output/frame_<f>.npz`) |
| 4 | F0, F2, F4 (F3 report) | `precision/` runner steps | session camera, continuous gauge, native-pixel tracks |
| 5 | F5 | `precision/refine.py` | corrected poses + camera (rung chosen by held-out; pccr R1, fx 364→392) |
| 6 | **Depth on F5 (`f6_bend`)** | `precision/depth_on_f5.py`, `precision.cloud.source: omega_bent`, config `precision.bend` | (a) F5 FIT tracks triangulated with F5 poses + camera; (b) per keyframe Omega depth × k(u,v)=c0+c1·u+c2·v, fitted EXACTLY as epoch 7: 10-step Huber IRLS (`bend.irls_iterations`), landmark rows on depth > `bend.min_depth_m` (0.05), < `bend.min_rows` (20) rows → k = 1, window by half A of the HELD-OUT tracks (pccr ±0, 7.25 % → 2.80 %); (c) validity = the ONE confidence floor (`conf_min_norm`, min-max per Omega chunk) + not sky; (d) epoch 8's EDGE-KEEPING VOTE: τ = p75 of the neighbour disagreement on INTERIOR pixels (pccr 1.93 %), mixed pixels at contours snapped to their SAM3 mask's side, two-sided vote over ±3/±6/±12, contradicted pixels repaired to the neighbours' median when ≥ 2 agree, below-floor edge pixels admitted when confirmed; confidence column = agree count. pccr: 63.1 % kept, 15.1 % contradicted, 8.5 % repaired, 0.4 % admitted, coverage 71.0 % |
| 7 | Cloud publish | `precision/corrected_cloud.publish` | voxel + SOR, octree, transactional epoch; `corrected_cloud.json` carries per-frame s_k + bend; intrinsic.txt inside the transaction |
| 8 | Mask projection + OBBs | `segmentation/pipeline.py` (cloud stage) | class byte = `_encode_classification` + `class_map.json` (`class_byte` on every viewer payload); co-visible split (camera-return rule + free-space test); split children listed (`mask_fates.created_by_projection`); OBB yaw by RANSAC; every instance inherits its concept's `shape_caption` |
| 9 | Certification | `workers/certify_worker.py`, `reconstruction/certify/run.py`, `correction/` | depth per chunk; floor per chunk BLENDED across overlaps; mask filter with the cloud's own camera (`camera.json`), rule 4 by majority of views; NO MLS (`precision.cloud.consolidate: false`); chunk check on the CERTIFIED depth (composes the transform epochs) |
| 10 | One final epoch | `certify.single_final_epoch: true`, `correction.apply.keep_only_live_epoch` | no Omega comparison cloud (`_epoch_0`) is built; after the certification every `_epoch_<N>/` is deleted (ledger + `corrections/epoch_<N>.npz` stay) |
| 11 | Object descriptions | `segmentation/object_captioner.caption_session_objects` (end of certify) | ONE Qwen3-VL call per object over its largest SAM3-mask views → `shape_caption` source `object` (config `segmentation.object_captions`) |

**On demand (Meshing dialog; search + select / deselect all, nothing preselected):** **Object** = ShapeR on every selected instance (keyed by `instance_id`; every face oriented OUTWARD by ray parity — `reconstruction/orient_outward.py` — before and after the texture bake; drawn in the viewer in its segment colour) (`/api/segmentation/
shape/export` → `segmentation/shaper_export.py` → `run_shaper_batch.py`, env `shaper`, preset max):
PKL views = every posed keyframe that SEES the object (occlusion test with the published depth +
`loops.witness.occlusion_tol_rel`), max 32; caption = manual > VLM object > VLM concept > label;
texture baked from the scan frames (`shaper.texture`); progress per object in the dialog.
**Mesh** = RANSAC + Poisson today → to become point2cad (pending B2).

**Run checks:** chunk check (automatic, in the acta). The crease-profile edge metric
(`precision/edge_metric.py`) and the floor metric (`analysis/2026-09-30_depth_sources/floor_metric.py`:
layer thickness + tilt-removed undulation per 1 m cell) exist but run by hand (pending A-checks).
pccr epoch 7 reference: 18.5 M pts, floor 5.2 cm thickness / 7.6 cm undulation.

**Viewer fixes of 2026-10-01** (in main): the session load and every epoch's potree_ready send the
LIVE poses + intrinsic.txt (`main.py` `_camera_poses_payload`; maplong_run/ holds Omega's original
poses); segment visibility keyed by `class_byte`, reset when the class map changes between epochs;
the objects the projection creates are listed (their points could never be hidden).

## PENDING — A. Reconstruction pipeline (clean list)

| # | Stage | Status |
|---|---|---|
| 1 | Intake | done |
| 2 | VLM + SAM3 | done — **pending: VLM refinement** (below) |
| 3 | Omega (5 m chunks, 50 %) | done |
| 4 | F0, F2, F4 (F3) | done |
| 5 | F5 (R1) | done |
| 6 | Depth on F5 (`f6_bend` = epoch 8) | done — **pending:** it aborts when F5's camera carries lens distortion or Omega's grid is not the native one (pccr has neither); no fallback |
| 7 | Cloud publish | done |
| 8 | Mask projection + OBB + class byte + co-visible split | done — **pending:** desk #174's 3.14 m piece may still hold 2 desks |
| 9 | Certification | done — **pending:** the chunk check flags a floor drift INSIDE chunk 0 (+16 cm, kf 0–62) and chunk 1 undecided (+5 cm); no intra-chunk correction exists |
| 10 | One final epoch | done |
| 11 | Per-object VLM description for ShapeR | done |
| — | Run checks | **pending:** the edge metric and the floor metric run by hand, not as automatic reports of every run |
| — | End-to-end validation | **pending:** the user relaunches pccr from scratch (restart the backend first) |

**VLM refinement (stage 2):** prompts per CONCEPT, never per object; 100 % of the points segmented (a
second VLM pass looks at what stayed unsegmented and proposes the concepts it sees there); the fallback
retry carries the category, not bare adjectives (pccr: a 'window' fallback got a door's description).

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
  `nice`); never fan out agents on the user's session.
- ShapeR on the A100 (sm_80): `torchsparse` in env `shaper` rebuilt from `nihalsid/torchsparse@20ccc92`
  with `TORCH_CUDA_ARCH_LIST="8.0;8.6"`, `SPHASH_INCLUDE`, `CUDA_INCLUDE`, `CUDA_LIB`.
- One Hugging Face cache: `HF_HOME=/workspace/hf_cache`.
- texrecon (`vendor/mvs-texturing`, the texture bake of every object mesh) was built against libtiff 5;
  the pod now has libtiff 6 only ("libtiff.so.5: cannot open shared object file", 2026-10-01). Relinked
  with its own `apps/texrecon/CMakeFiles/texrecon.dir/link.txt`, `libtbb.so.12.19` → `libtbb.so.12`
  (the vendored oneTBB holds 12.15; a full `cmake ..` fails on that mismatch).
- The pccr session on disk (2026-10-01): live epoch 8 (certified, 18.2 M pts, 110 objects), stored
  epochs 0 / 6 / 7 — hand-built, not what the code produces; a "Reconstruir" wipes it.
- Analysis scripts of the depth-source study: `analysis/2026-09-30_depth_sources/` (epoch 7 / 8
  recipes, floor metric, diagnostics).
