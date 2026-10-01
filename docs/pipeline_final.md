# Final reconstruction pipeline — USER DECISIONS 2026-09-30

Status of each stage: **KEEP** (in the code today, unchanged), **CHANGE** (in the code, to be modified),
**NEW** (to be written), **DECIDE** (settled by a test that is pending). Every threshold that decides
something is either a user criterion or measured by the session itself — no invented numbers.

## Decisions (USER, 2026-09-30 / 2026-10-01)

0. **2026-10-01 — epoch 7 of pccr is the best so far** (*"esta epoch 7 es la mejor hasta el momento"*;
   floor, flyers and the depth correction judged excellent). Its recipe IS the pipeline below. The
   priority is object DEFINITION: *"lo más importante es que los objetos deben tener mucha definición,
   corte en los filos, las aristas"*. Old rules may be replaced where something measures better:
   *"olvidate de las reglas que pusimos en su momento, estamos en otra etapa"*.
1. **The cloud is Omega's depth BENT to F5 per keyframe, then a multi-view vote — no plane sweep, no DA3**
   (stage 6 below). Superseded: F6's sweep + `f7_cloud` (§7, the 2026-09-30 decision) and the DA3 fill
   (§7b; it layered objects — "capas duplicadas y puntos voladores, probablemente de DA3").
2. **F5 rung**: R1 (the applied rung, fx 392) — epoch 7 used it. The camera TRAVELS with the cloud.
3. **The certification** de-duplicates objects and collapses the floor (*"no debe haber cebolla en el
   piso"*), respecting real steps and slopes; floor motion blended across chunk overlaps (no camera step).
4. **ONE epoch: the final one.**

## Stages (2026-10-01 — the epoch-7 recipe)

| # | Stage | Status | What it does |
|---|---|---|---|
| 1 | Intake (I0–I4) | KEEP | frames, parallax keyframes, content exclusion |
| 2 | VLM + SAM3 | KEEP (+ pending VLM refinement) | once, on EVERY keyframe; prompts per CONCEPT |
| 3 | Omega | KEEP | chunks of 5 m of walk, 50 % overlap; initial poses, metric scale, per-keyframe depth + conf |
| 4 | F0, F2, F4 (F3 report) | KEEP | session camera, continuous gauge, native-pixel tracks |
| 5 | F5 joint refinement | KEEP | corrected poses + camera (R1) |
| 6 | **Depth on F5** | NEW — **only in `analysis/2026-09-30_depth_sources/omega_bent_epoch7.py`**, to port into `precision/` as the runner step replacing `f6_sweep` + `f7_cloud` | (a) F5 FIT tracks triangulated with F5 poses + camera; (b) per keyframe Omega depth × k(u,v)=c0+c1u+c2v fitted robustly on them, window chosen on the HELD-OUT tracks (pccr: 7.25 % → 2.80 %); (c) validity = the ONE confidence floor (`conf_min_norm`, min-max per Omega chunk) + not sky; (d) multi-view vote over ±3/±6/±12: leaves when more neighbours see free space through it than agree, else depth = median of the agreeing views; τ = measured p75 (pccr 1.96 %). pccr: contradicted 10.2 % (F6 on the unbent depth: 32 %), coverage 69.2 % |
| 7 | Cloud | CHANGE | voxel + SOR, octree; the publish writes `intrinsic.txt` + `camera.json` of the camera it used as epoch artifacts |
| 8 | Mask projection + OBBs | KEEP (fixed 2026-09-30/10-01, uncommitted) | class byte = `_encode_classification` + `class_map.json`, `class_byte` on every viewer payload; co-visible split (camera-return rule + free-space test); OBB yaw by RANSAC |
| 9 | Certification | CHANGE | depth per chunk; floor per chunk BLENDED across overlaps (done); mask filter with the cloud's camera, rule 4 fixed + majority (in progress); NO MLS re-consolidation (in progress); chunk check report |
| 10 | Publish | CHANGE | ONE final epoch |
| 11 | Deliverables for BIM | NEW | structure → point2cad B-rep, objects → ShapeR (textured); see §11 |

**Run checks (every reconstruction):** the crease-profile edge metric (`precision/edge_metric.py`, in
progress), the floor metric (layer thickness + tilt-removed undulation), and camera-trajectory continuity
against F5 (no new step between consecutive keyframes). pccr epoch 7: 18.5 M pts, floor layer 5.2 cm,
undulation 7.6 cm, kf 62→63 7.0 cm (F5 6.9; Omega had 47 cm with a 39 cm drop).

**Epoch 8 = epoch 7 + the edge fixes** (bend window by consecutive-keyframe agreement, two-sided vote,
confidence floor narrowed at edges with SAM3 snapping of mixed pixels, agree count per point, mask-filter
camera + rule 4). They enter the pipeline only if the edge metric says 8 improves on 7
(`heldout_change` at 0.95); otherwise stage 6 stays as in epoch 7.

### 7 · (superseded 2026-10-01) Corrected Omega cloud (`f7_cloud`, F6 depth + repair)

point = c2w_F5 · (z_F6 · K_F5⁻¹ [u v 1]), z_F6 = F6's depth (tier 0 sweep / tier 1 Omega × s_k).
- `precision.cloud.confidence_gate: false` — no gate of this step's own. F6 already dropped what is
  under the ONE confidence floor (`conf_min_norm`, its DISCARD_LOW_CONF: 7.6 % of pccr); the
  `conf_percentile` 20 on top cut up to 20 % more per chunk (weak-texture floors and walls).
- `precision.cloud.repair_contradicted: true` — F6 contradicted 32.1 % of pccr's pixels (another view
  sees free space through Omega's depth: Omega and F5 disagree there). The pixel still sees a surface:
  the neighbours' (`repair_neighbors` ±3, ±6, ±12) F6 surfaces are z-buffered onto its ray; when
  `repair_min_views` (2) agree within τ of their median it takes that median, otherwise it leaves.
  τ = `repair_tau_quantile` (p75) of the session's own neighbour disagreement, measured every run.
  Code: `precision/corrected_cloud.py` `repair_contradicted`, `measured_tau`; test
  `tests/test_corrected_cloud_repair.py`.
- `witness_filter`, `silhouette_filter`, `consolidate`: false (epoch 4's recipe; the certify stage's
  mask filter judges points against the other objects' masks).
- Measured on pccr before the repair (epoch 4): 60.6 M raw → 18.4 M after voxel + SOR — "muy
  erosionada" (USER), the reason for the two changes above.

### 7b · (superseded 2026-10-01) DA3 conditioned cloud / DA3 fill — layered objects

Kept documented: DA3-streaming 120/60, native resolution, conditioned on F5's poses + camera through
`_stac_chunk_camera_priors`, no confidence gate, multi-view fusion (contradiction vote + median of the
agreeing views, τ = measured p75). pccr: 9 % flyers out, final poses within 3 cm of F5. Floor per
variant (thickness / undulation cm): F5-R0+364 5.4/21.0, F5+392 5.8/17.3, F5+DA3 focal 6.1/15.0.
Scripts: `analysis/2026-09-30_depth_sources/`.

### 9 · Certification (CHANGE)

Order: closures measured → object duplicates corrected → floor collapsed → mask filter → checks.

**Objects — no duplicates.** The visit-drift instrument (orthogonal-view silhouettes) measures every
object two separated visits saw. On the DA3 cloud of pccr it found 7 determined closures of 0.5–1.5 m
(door, desk, beam…), several with large tangential parts. Required:
- the correction removes them — the correction's UNIT must be the DA3 cloud's own structure (its
  keyframes / its chunks), not Omega's six chunks, which the DA3 cloud was not built from;
- after the correction the same instrument re-measures every object; a remaining separation above the
  session's repeatability is reported per object as a failure (and the loop iterates while it improves).

**Floor — collapse, no "onion".** The per-chunk rigid levelling (model `chunk`) puts each chunk's floor
at y = 0 but does not remove the parallel layers INSIDE a chunk, which come from per-keyframe depth
offsets. Required: each keyframe's segmented floor points are brought onto the CONSENSUS floor surface
(per-cell robust median over all keyframes that see the cell), which keeps real steps and slopes
because it is measured per cell and never forced to a plane. Measured before / after and reported:
- floor layer thickness per 1 m cell (p90 − p10 around the cell's own plane, median over cells),
- undulation between cells with the global tilt removed (`floor_metric.py`, v2).
Reference numbers, pccr before collapse: thickness 5.4–5.8 cm, undulation 14–21 cm.

**Chunk check** stays as the final report (floor / ceiling / camera height per chunk and keyframe).
On the corrected DA3 cloud of pccr it still flagged: an intra-chunk floor drift up to +21.9 cm at
keyframe 61 (the same stretch where Omega's depth also failed), and the last chunk 19 cm low.

### 10 · Publish (CHANGE)

The session keeps ONE epoch: the final corrected cloud, with its octree, poses, segmentation, OBBs,
`classification.npy` + `class_map.json`. No epoch 0 / intermediate epochs left on disk.

## Fixed on 2026-09-30 (tests green, not yet committed)

- `segmentation/pipeline.py` — the mask projection wrote the class byte as `min(obj id, 255)`: 50 objects
  with instance id ≥ 255 shared byte 255 and ONE viewer toggle switched all of them (drywalls, backpack,
  beams, windows…). Now `_encode_classification` (the republish encoder + `class_map.json`, incremental
  bytes translated). Test: `tests/test_projection_class_byte.py`.
- `correction/invalidate.py` — an epoch select refit every OBB from `globalIndices` of ANOTHER cloud
  (walk-sized cubes). Now a segmentation that does not index the live cloud is left as it is and declared
  stale. Test: `tests/test_instance_store_foreign_indices.py`.
- `segmentation/pipeline.py` `_split_covisible_components` (+ `segmentation.split_covisible`) — an
  instance whose components (apart by more than `fragment_gap_m`, each ≥ `visit_drift.min_points`)
  are seen TOGETHER in ≥ `dedupe_overlap` of the smaller one's keyframes AND whose gap is free space the
  cameras see through (`_gap_is_free_space`: nothing lies behind a hole) is split: pccr desk #174 was
  three desks 2 m apart (one SAM3 mask over neighbouring desks + the space dedupe). Components never
  seen together (a drift duplicate) stay for the certification. Test: `tests/test_split_covisible.py`.
- `precision/corrected_cloud.py` + `precision/config.py` — §7 (no own confidence gate, repair of the
  contradicted pixels). Test: `tests/test_corrected_cloud_repair.py`.
- `correction/floor.py` — the chunk floor model's anchor keyframe written as an explicit median (the
  package forbids fixed divisors).

## Environment notes

- ShapeR on the A100 (sm_80): `torchsparse` in env `shaper` was built for sm_86 only ("no kernel image").
  Rebuild from `nihalsid/torchsparse@20ccc92` with `TORCH_CUDA_ARCH_LIST="8.0;8.6"` and the fork's own
  variables `SPHASH_INCLUDE`, `CUDA_INCLUDE`, `CUDA_LIB` (an empty `CUDA_LIB` leaves a bare `-L`).
- One Hugging Face cache: `HF_HOME=/workspace/hf_cache` (the backend's). Scripts run from a shell that
  used `~/.cache/huggingface` downloaded T5-XL + CLIP a second time (23 GB).
- Cap every job (pod: 117 GB / ~30 CPUs): `taskset`, `OMP_NUM_THREADS`, `nice`.

## Pending

- Validate §7 on pccr through the pipeline's own code, then the certify stage on it.
- F5 rung R0 vs R1 for this recipe: R1 (the applied rung, fx 392) is what epoch 4 used.
- **VLM refinement (USER 2026-09-30, after the cloud recipe and the certification are closed):**
  - prompts per CONCEPT, never per object — SAM3 returns every instance a concept prompt matches;
  - the goal is 100 % of the points segmented (pccr today 96.9–99.3 %): a second, targeted VLM pass
    looks at what stayed unsegmented and proposes the concepts it sees there;
  - the fallback retry carries the category, not bare adjectives (pccr: a 'window' fallback got a
    door's description);
  - one ShapeR description per object: after the mask projection the VLM describes each object's SHAPE
    from its best isolated views (`segmentation/object_captioner.caption_object_qwen`), short and
    geometric, stored on the instance as `vlm_proposed`; the ShapeR PKL export uses it by default
    (a manual caption from the UI still wins). Today the PKL carries the SAM3 label (`auto_caption: false`).

### 11 · Deliverables for the BIM comparison (USER 2026-09-30)

*"usar point2cad en lugar de ransac … y ShapeR para objetos, y comparamos lo que podemos"*. Raw points are
not a reliable basis (flyers, layers); the comparison uses reconstructed elements with their own error.
- **Structure** (walls, floor, ceiling, columns, beams, pipes, ducts — chosen by GEOMETRY, never by label):
  SAM3 instance → surface regions (the in-tree split-and-fit engine, `segmentation/perfect_object.py`) →
  point2cad (`segmentation/p2c_object.py`, env `point2cad`). Point2CAD does NOT partition a cloud into
  surfaces (its paper uses ParseNet/HPNet): it takes `.xyzc` = points + a SURFACE id, fits per surface
  plane / sphere / cylinder / cone / INR freeform and keeps the lowest error, intersects neighbours into
  edges and corners, clips — a metric B-rep with per-surface parameters and residuals. Output is normalized
  (inverted in `p2c_object.py`). `--max_parallel_surfaces 4` (each worker holds a CUDA context).
  **License CC-BY-NC** (approved for internal use; a commercial deliverable needs a licence).
- **Objects** (furniture, equipment): ShapeR (preset max, fitted to the segment's cloud) + a texture
  baked from the scan frames (existing `bake_object_glb`); unseen parts stay untextured. ShapeR generates
  what no camera saw: trusted for presence, position, orientation and overall extent — not for detailed
  dimensions of unseen sides; every mesh carries measured vs generated.
- **Comparison**: register to the BIM on the structure; per element deviation of position / dimensions /
  flatness / plumbness with its margin; installations routing; objects present / missing / displaced.
  A deviation below the session's own repeatability (pccr ~5 cm) is reported as not measurable.
- **The camera travels with the cloud (found 2026-10-01).** The viewer frames every camera with
  `intrinsic.txt` (`server/main.py` ~1303 and ~7490); after F5 that file still held Omega's camera
  (fx ≈ 363) while the cloud was built with F5's (fx 391.9) — standing at a camera, the scene did not
  match the image. pccr fixed in data (each epoch carries its `intrinsic.txt` in its manifest). Code fix:
  the cloud publish (`precision/corrected_cloud.publish`) writes `intrinsic.txt` from the camera it used
  and lists it (and `camera.json`) as epoch artifacts.

