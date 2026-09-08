# Correction module — Phase 1 inventory (read-only)

Date: 2026-09-08. Produced for the correction-module redesign (prompt_claude.txt).
Every entry below was verified by grep/read against the working tree at commit
`d25e715`; nothing is from memory. Scope: `server/` + `ui/src` (vendor/, archive/,
`__pycache__`, logs, static excluded).

---

## 1. Current corrector — verified findings (H1–H10 confirmed)

Source: `server/segmentation/correction_analysis.py` (1197 lines),
endpoints `server/main.py:4937–5142`, UI `ui/src/App.tsx` + `ui/src/components/Viewport.tsx`.

| # | Confirmed at | Notes beyond the prompt |
|---|---|---|
| H1 | `correction_analysis.py:233-234, 420, 567, 796-797, 878-879, 907, 984-985` — `ks // 30` and `(len(frames)+29)//30` in `run_correction`, `compute_chunk_boxes`, `apply_manual_chunk`, `align_floor_y0` | The REAL chunk plan (`reconstruction/chunk_plan.py:86-119`) sizes chunks by walked meters, clamp [24,150], 50% overlap — and is **never persisted** (see §5). `compute_chunk_boxes` also ignores `chunk_*_meta.json:chunk_step`. |
| H2 | `apply_manual_chunk` `:837-939`; endpoints `main.py:5047-5093`; UI `Viewport.tsx:498-597, 2166-2223` (`chunkTcRef`, `chunkBoxByIdRef`, `selChunk`, `emitChunkDelta`), `App.tsx:127-133, 318-330, 2596-2606, 4084-4230` | Moves one rigid 30-kf bucket with no distribution — the exact failure `run_correction`'s 4b comment documents. |
| H3 | `run_correction` step 4 `:510-519` — `floor_heldout_cm` computed, logged, **never compared to any threshold** | The only veto is the object residual (`:494-502`). MEJORAS_OBLIGATORIAS §3 (box1: 92°/9 m applied while floor went +0.1 → −22.7 cm) is the measured consequence. |
| H4 | `run_correction` step 5 `:605-656` rewrites only `cleaned_cloud.ply`, `camera_poses.txt`, OBBs, Potree | Untouched: `cleaned_cloud_raw.ply` (surface_fit residual reference, `surface_fit/runner.py:299-304`), every depth `.npy/.npz` (§4.6), `scale_diagnostics.json` (§4.7), `scene_r.db` geometry (§4.9), pose copies (`omega_run/`, `da3_run/`, `maplong_run/` — `scale_align.py:322-324` proves multi-copy writes are the repo norm). |
| H5 | `undo_correction:706`, `approve_correction:738`, `_save_state:61` | One-level undo dir; `correction_report.json` overwritten each run; `correction_history.json` keeps only `{approved_at, chunks, points_moved}` — no operator, no transforms, no replay. |
| H6 | `align_floor_y0:942-1161` — every anchor chunk's floor → y=0 with vertical normal | Ramps/drainage slopes are flattened by construction. Also resets `floor_transform.npz` to identity (`:1146-1150`). |
| H7 | Literals inventoried in §6 below (34 decision literals) | `config.yaml` has **no** `correction:` section (verified: top-level keys are server, pipeline, paths, scene_analysis, segmentation, autoprompt, semantic, frame_selection, models, alignment, storage, fusion, visualization, bim, tsdf, poisson, reconstruction, postprocessing, coverage_engine, surface_fit, meshflow). |
| H8 | `load_state:51-58` (`except Exception: pass` → `{"status": "none"}`); Potree failure after cloud rewrite `:668-671` ("undo is available") | Also `approve_correction` runs synchronously ON the event loop (`main.py:5031-5044`), `undo` has no task/progress, and MEJORAS §2.7: pending state is written only after the multi-minute Potree rebuild. |
| H9 | Cloud+pose load, kf mapping, backup, rewrite, Potree rebuild each appear 3× (`run_correction`, `apply_manual_chunk`, `align_floor_y0`) with divergences (e.g. only `align_floor_y0` handles `floor_transform.npz`) | |
| H10 | `run_correction` step 4 `:440-454` applies `k_scale` with no DA3 cross-check; `scale_diagnostics.json` is **write-only** in the whole repo (no reader anywhere) and never regenerated | |

Additional verified facts the redesign must absorb:

- The pending-guard *is* enforced inside `apply_manual_chunk` (`:847-850`) and
  `align_floor_y0` (`:959-961`), but there is **no session lock**: a correction and
  an export/erase can interleave freely.
- `main.py:4988` duplicates the pending check for `/correction/run` only.
- Correction endpoints never call `rebuild_instance_store`; only `/alignment`
  (`main.py:2569-2574`) and `/level_floor` (`main.py:2814-2819`) do → after a
  correction the chat tools (`phase5_qa/tools.py`) serve **stale cached XYZ**.
- None of the 7 correction/chunk fetches in the UI send `Authorization`, although
  `useAuth()` exposes the token in the same component (`App.tsx:446`).
- The server registers `task_type="correction"` progress but the UI never polls it;
  all three mutating calls `await` the raw POST for minutes.

---

## 2. Readers/writers of the primary artifacts (grep-based, exhaustive)

Legend: R read, W write, D delete/invalidate. Only decision-relevant sites listed
individually; mechanical existence-probes are grouped.

### 2.1 `cleaned_cloud.ply`

Writers: `workers/cloudcompy_worker.py:20,198-202` (producer; MLS rewrites in
place), `main.py:247` (legacy postprocess), `fuse_scans.py:468`,
`segmentation/erase.py:651` (physical delete, **re-indexes points**),
`reconstruction/surface_fit/consolidate.py:529-544`,
`segmentation/correction_analysis.py:641,611,714,889,899,1099,1113`.
Deleters: `pipeline_manager.py:317,429`, `main.py:327-338,7163`.

Readers (server): `project_paths.py:273,366` · `main.py:458,671,1516,1913,2638,
2705,4337,4945,5924,5971,7173,7198,7249,7278,7593` · `potree_converter.py:253` ·
`pipeline_manager.py:799-812` · `workers/{cloudcompy(28,41,391),pgsr(85),tsdf(35),
instance_cleaner(75-102),map(2129)}_worker.py` · `segmentation/pipeline.py:106,
1711,2378,2439` · `segmentation/tsdf_export.py:1042,1429,2214,2303,2433,3559,3903`
· `segmentation/{mesh_export(146),perfect_object(520,877,973,1167),
shape_proposer(605),object_analysis(101),poisson_object(917,964,1076),
erase(226,322,794),dn_splatter_export(195,275)}.py` ·
`segmentation/correction_analysis.py:202,774,864,970` ·
`reconstruction/{pgsr_export(187,235),nksr_scene(58),cloud_mesh(210,616)}.py` ·
`reconstruction/surface_fit/{runner(291),hole_audit(55)}.py` ·
`reconstruction_runner.py:316` · `bim/{registration(398),comparison(861,1012)}.py`
· `phase5_qa/tools.py:885` · `ar_api.py:70,124,226` · `fuse_scans.py:198,216` ·
`run_pgsr_object.py:146`, `run_surface_fit.py:96`.
UI: never the file — binary WS stream + Potree octree only.

### 2.2 `cleaned_cloud_raw.ply`

W: `surface_fit/consolidate.py:535,544` (pre-MLS metric reference).
R: `surface_fit/runner.py:283,299-304` (residuals vs raw; warns+ignores on size
mismatch), `pgsr_export.py:187`, `tools/scale_ab.py:359`,
`tests/test_surface_fit_consolidate.py`. D: `pipeline_manager.py:317`.
Same point order/count as `cleaned_cloud.ply` (copy taken before in-place MLS) →
**carries the same provenance fields** and must be transformed identically.

### 2.3 `camera_poses.txt` + pose copies

Canonical: `output/camera_poses.txt`. Copies: `maplong_run/`, `omega_run/`,
`da3_run/`, plus the **stale pre-scale** `camera_poses_mapanything.json` /
`mapanything_poses.json` (documented stale at `session_io.py:306-314`, still in
the fallback chain `:332-333`).

Writers that rewrite EVERY copy (the precedent to follow): `scale_align.py:322-324`
(`.prescale` backup), `orient.py:301` (`.preorient`), `run_bundle_adjust.py:226`
(`.preba`), `run_colmap_ba.py:167-189` (`.precolmapba`), `pose_refine.py:353,537`
(`.preposerefine`), `surface_fit/fine_register.py:892-895` (`.prefinereg`),
`dense_pose_fusion.py:439-461` (appends filler poses+frames to every copy).
`correction_analysis.py:654,911,1124` rewrites ONLY the canonical file (gap).

Readers: `session_io.py:296-347` (the canonical `CameraSource` resolver, keys by
real frame via `camera_frames.txt`) · `tsdf_export.py:797-866` (own parser
`_load_da3_refined_poses`, **replaces** `cam.pose_map` at `:1848`) ·
`main.py:1230,7300-7392` (flythrough + viewer frusta) · `scale_align.py:84-90` ·
`orient.py:85` · `chunk_plan.py:21-28` · `pose_refine.py:353` ·
`dense_pose_fusion.py:72` · `trace_normals.py:99` · `nksr_run.py:49` ·
`cloud_mesh.py:206` · `mv_consistency.py:122` · `pgsr_{export,cloud}.py` ·
`native_depth.py:279` · `surface_fit/{consolidate(498),fine_register(892)}.py` ·
`map_worker.py:1040,1364,1852,1896,2131` · `bim/occlusion_raycaster.py:56,97` ·
`run_{bundle_adjust,colmap_ba}.py` · `tools/scale_ab.py:144`.
UI: `Viewport.tsx:4220-4255` renders `msg.cameraPoses` frusta.

### 2.4 `camera_frames.txt` / `frame_list.json`

W: `map_worker.py:2198-2214` (real frame numbers), `stray_da3_streaming.py:226`,
`dense_pose_fusion.py:461` (append). R: `session_io.py:232-289`,
`tsdf_export.py:290,817-850,1864`, `correction_analysis.py:217,789,870,976`,
`scale_align.py:84`, `vggt_tracks.py:98`, `run_bundle_adjust.py:42`,
`run_colmap_ba.py:189`, `reproject_chunks.py:39`, `pose_refine.py:354`,
`trace_normals.py:60,101`, `nksr_run.py:53`, `fine_register.py:893`,
`dense_pose_fusion.py:75`, `map_worker.py:1364,1895,2423-2438,2578`,
`main.py:1243`.

### 2.5 `segmentation_result.json`

Writers/mutators: `segmentation/pipeline.py:2487,2557` (producer),
`main.py:2559-2580` (/alignment OBB re-projection; unlink on failure),
`main.py:2704-2812` (/level_floor OBB recompute), `main.py:3247-3288,3477-3529,
3556-3673,6346-6379,7163`, `correction_analysis.py:1164-1194` (OBBs in place),
`erase.py:323,790`, `object_analysis.py:193`, `fuse_scans.py:471`.
D: `pipeline_manager.py:321,328`.

Readers: 13 `main.py` endpoint sites (2666…6717) · `pipeline.py:2430,2593-2609
("load NEVER invalidates" doctrine — trusted blindly)` ·
`propagation_resume.py:65,139` · `tsdf_export.py:1014,1135,3840,3918` ·
`mesh_export.py:152` · `poisson_object.py:963` · `p2c_object.py:151` ·
`shape_proposer.py:567` · `perfect_object.py:859,955,1150` ·
`object_analysis.py:75` · `correction_analysis.py:243` ·
`surface_fit/runner.py:290` · `reconstruction_runner.py:331` ·
`bim/{registration(384),comparison(960),occlusion_raycaster(408)}.py` ·
`workers/{sam3(132),tsdf(112)}_worker.py` · `phase5_qa/api.py:44-53` ·
`run_perfect_objects.py:84`, `run_pgsr_object.py:140`, `fuse_scans.py:53`.
`globalIndices` stay valid after a correction **iff point order is preserved**
(erase.py physical delete is the only order-changing writer).

### 2.6 Per-keyframe depth (`.npy`/`.npz`)

Layout: `da3_run/results_output/frame_<N>.npz` (DA3 metric anchors; kept while
`tsdf.depth_source ∈ {da3, da3_frames, auto}` — `map_worker.py:2048-2056`) ·
`omega_run/results_output/frame_<N>.npz` (Ω depth, **kept** for texture bake —
`map_worker.py:2061-2099`) · `maplong_run/_tmp_results_aligned/chunk_K.npy`
(deleted after scale pass / TSDF — `map_worker.py:1866`, `tsdf_worker.py:143`) ·
`da3_run/{stem}_depth.npy` · `pgsr_render/frame_<num>.npz` ·
`da3_run/hires_{output,raw}`.

Producers: `map_worker.py:1341-1353,1457-1483`, `extract_da3_{depth,full}.py`,
`stray_da3_streaming.py:358,483`, `convert_stray_to_da3.py:273`,
`calibrate_depth_hybrid.py:150`, `pgsr_train.py:568`, `native_depth.py:190-216`,
`tools/extract_da3_anchors.py`, **`scale_align.py:268-320` (multiplies aligned
chunk depth by `s` in place — precedent for depth rescale)**, `orient.py:231`.

Readers (= the depth-accessor migration list): `tsdf_export.py:97-132,167-187,
214-229,277-292,430-460` (the four resolvers + PGSR) ·
`autoprompt/session_builder.py:185-189` · `scale_align.py:53,67,158` ·
`scale_model.py:71,460` · `mv_consistency.py:126` · `trace_normals.py:57` ·
`run_bundle_adjust.py:57,118` · `run_colmap_ba.py:70` · `densify_fillers.py:27,44`
· `dense_pose_fusion.py:89,241` · `nvdiffrast_bake.py:53,117,268` ·
`pgsr_cloud.py:55` · `pgsr_export.py:69` · `native_depth.py:279` ·
`vggt_tracks.py:54` · `map_worker.py:215,1199,1478,1565,1594` ·
`phase3_findings/detect.py:146` · `bridge_tum_format.py:157`,
`stray_direct_ply.py:75`.
(`hole_audit.py` and `texture_bake.py` read NO depth files — hole_audit z-buffers
the cloud itself; texrecon is photometric.)

### 2.7 `scale_diagnostics.json` / scale_align

W: `scale_align.py:379,524-527` (schema: `mode_used, model, s_da3, scale_source,
vio, s_applied, anchors{count, mad_rel, spread, frames[{num, s_f, n_px,
z_median_omega, d_median_m}]}, jackknife, residual_vs_depth, scale_confidence,
dry_run`). **No reader anywhere in server/ or ui/** — write-only, never deleted on
Replace, silently stale after any correction.
Marker: `output/.metric_scale_applied` records the applied `s=` (`:510-519`).
Key property for the scale gate: per-anchor `s_f` = median(DA3/Ω) **before**
apply_scale; the current agreement of anchor frame f is `s_f / s_applied`, and a
per-keyframe depth correction `k_f` changes it analytically to
`s_f / (s_applied · k_f)` — the gate and the per-epoch regeneration need **no
depth files**, only `scale_diagnostics.json` + the applied `k_kf`.

### 2.8 `floor_transform.npz`

W: `cloudcompy_worker.py:285` (producer; unlinked when `.orientation_applied` —
`:253-258`), `main.py:2540,2802`, `correction_analysis.py:1146-1150` (reset to
identity), `fuse_scans.py:470`. D: `pipeline_manager.py:318,429`.
R: display-frame definition `segmentation/pipeline.py:1760-1780,2454`,
`perfect_object.py:40`, `erase.py:25`, `phase5_qa/tools.py:43-56,903`,
`bim/{registration(40),comparison(983)}.py`, `ar_api.py:62-132`,
`main.py:400,438,1926,2594,4957,5983,7233-7420,7578`, `correction_analysis.py:1177`,
`tools/glb_to_usdz.py:172`. UI: `Viewport.tsx:4098,4168` + XR engines.

### 2.9 SQLite instance store — `output/scene_r.db`

Schema `phase_r/instance_store.py:34-140`. Geometry-caching tables (stale if the
cloud moves): `instance_points` (raw float32 world/**display** XYZ + frame_ids),
`instance_obb` (4×4 + AABB + position), `user_volumes` (user geometry — must NOT
move), `findings` (`point3d` + `frame_id` → re-transformable per keyframe),
`vote_regions`/`onion_metrics` (coordinate-derived, medium).
Rebuilders: `pipeline.py:2387-2419` (`_write_instance_store` **deletes the whole
db** incl. findings/volumes/notes), `pipeline.py:2423-2468`
(`rebuild_instance_store`, guards index range), callers `main.py:2572,2817,5956`,
`phase5_qa/api.py:42-53`, `propagation_resume.py:161`, `erase.py:751,881`.
Chat consumers: `phase5_qa/tools.py` (positions/extents/bands/volumes/findings —
all from the store, converted to display via `floor_transform.npz`).
**Gap confirmed: no correction path touches the store.**

### 2.10 Chunk plan persistence

`reconstruction/chunk_plan.py` is pure/in-memory; sole production caller
`map_worker.py:1584-1588,1725,1768,1920,2004-2019`. **No `chunk_plan.json`
exists.** What IS persisted: `chunk_<i>_meta.json` (`chunk_id, source_chunk,
frame_count, chunk_step, frame_global_start/end…` — `map_worker.py:2486,2607`),
`vggt_omega_config.yaml` (`chunk_size, overlap, frame_ownership, metric_lock…` —
`:1680`), `coverage_trim.json` (`:1928`), `scale_anchor_frames.json` (`:1540`),
vendor `maplong_run/chunk_health.json`. Readers of chunk meta:
`tsdf_export.py:308-327`, `trace_normals.py:75-85`, `occlusion_raycaster.py:41`,
`pipeline.py:843,1856`.
→ `map_worker.py` must start persisting `output/chunk_plan.json`
(`chunk_ranges`, `overlap`, sizing inputs); phase-1 single-pass sessions have no
plan and the corrector works purely per keyframe.

---

## 3. Consumer → action matrix (final, every reader has a row)

Rigid part: per-keyframe (R_kf, t_kf). Depth part: per-keyframe scalar k_kf along
each point's own camera ray. "tx" = inside the atomic transaction; "regen" =
regenerated right after the swap; "invalidate" = epoch-stamped stale, user
regenerates on demand (USER 2026-08-29: never bake the scene because one object
needs a mesh).

| Artifact / consumer | Action on apply | Why |
|---|---|---|
| `cleaned_cloud.ply` | **Transform (tx)** per keyframe: k along ray, then R,t | The deliverable cloud. Point order unchanged. |
| `cleaned_cloud_raw.ply` | **Transform (tx)** with the SAME per-keyframe transform | Same order + provenance; surface_fit residuals stay coherent. Missing file (pre-consolidate session) → skip, recorded in report. |
| `camera_poses.txt` + copies (`maplong_run/`, `omega_run/`, `da3_run/`) | **Transform (tx)**, all copies, backup pattern like `.prescale` | Precedent: scale_align/orient rewrite every copy. `camera_poses_mapanything.json` is documented-stale → left untouched (stale before, stale after). |
| `camera_frames.txt` | No change | Frame identity is untouched. |
| Per-keyframe depth (`omega_run/`, `da3_run/*.npy|npz`, `maplong_run/`, `pgsr_render/`, hires) | **Sidecar (tx)**: `output/depth_correction.json` (k per keyframe, epoch-stamped); ALL readers load depth through a single `session_io` accessor that applies k. Rigid part needs nothing (camera and its points move together → camera-frame depth is invariant). | Rewriting GBs of vendor artifacts per correction is unaffordable and unauditable; a sidecar is exact, reversible and replayable. Mode `rewrite` reserved in config, not implemented silently. |
| `segmentation_result.json` | **Recompute OBBs (tx)**; `globalIndices` untouched (order preserved — asserted by test) | Same as today's `_update_result_obbs`, now inside the tx. |
| SAM3 masks (`seg_masks.npz`) | No change | 2-D. Any 3-D cache derived from them is epoch-invalidated. |
| `classification.npy` (when MEJORAS §1 lands) | No change | Per-point membership, order preserved. |
| Instance store `scene_r.db` | **Update geometry in place (regen)**: `set_points`/`set_obb` from the corrected cloud; findings `point3d` re-transformed via their `frame_id`; `user_volumes` untouched; epoch written to `scene_meta` | Full `_write_instance_store` rebuild deletes findings/volumes/notes — unacceptable. User volumes are user geometry (prompt §5.7). |
| Potree | **Rebuild inside the tx dir**, atomic swap | No "cloud changed but Potree failed" state. |
| `output/tsdf/*`, `output/surface_fit/*`, poisson, perfect_object, p2c, shape, per-object meshes | **Invalidate** (artifact epoch ≠ current → UI "stale" badge + Regenerate) | Costly; user decides (USER 2026-08-29). |
| BIM registration / comparison (`sabana/*`, `bim_comparison/`) | **Invalidate**; offer re-register; session report declares comparison epoch | Registration transform was fitted against the old cloud. |
| Coverage (`coverage/*.npz`, timeline) | **Invalidate** (epoch stamp on new writes) | Derived from poses+cloud. |
| `phase3_findings` (store `findings` table) | **Transform (regen)** `point3d` per origin `frame_id`; findings without resolvable frame → epoch-invalidated, declared | They persist `frame_id` + normalized box (`detect.py:75-88,319-322`). |
| Evaluation volumes (`user_volumes`) | **No move**; UI/chat warned the scene changed epoch | User geometry, not measurement. |
| `scale_diagnostics.json` | **Regenerate (tx)**: per-epoch history; anchor ratios updated analytically `s_f → s_f / k_f` | §2.7. Never silently stale again. |
| `floor_transform.npz` | Floor-align kind: reset per model (in tx, old one preserved in epoch dir); object kind: untouched | Today's behaviour, made transactional. |
| Phase-5 tools / phase-6 report / measure exports | **Stamp**: every measurement carries `geometry_epoch` + `human_directed_corrections` (+ `corrections_overridden`) | Prompt §5.7/§8. |
| `chunk_boxes.json`, `correction_state.json`, `correction_report.json`, `correction_history.json` | **Deleted** with the gizmo; replaced by ledger + per-run reports | H2/H5. |
| `ar_upright.json` | Self-invalidates on cloud mtime (`ar_api.py:73-80`) | Already safe. |
| Viewer (Potree + `potree_ready` + poses payload) | Reload **only after** the swap | Same notify helper, moved post-tx. |
| `flythrough`, viewer camera frusta | Nothing extra — they re-read `camera_poses.txt` per request | Transformed file is coherent. |

---

## 4. Numeric literals in `correction_analysis.py` → proposed config keys

| Literal (site) | Meaning | Proposed key |
|---|---|---|
| `OBB_MARGIN = 0.03` (:251) | curated-OBB margin when collecting copy evidence | `correction.evidence.obb_margin_m` |
| `300` (:384, :414) | min object points to solve a visit | `correction.evidence.min_object_points_solve` |
| `500` (:431) | min points for a copy to enter the fingerprint | `correction.evidence.min_object_points_fingerprint` |
| `dref > 0.3` (:435) | min inter-object baseline for the depth fingerprint | `correction.evidence.min_baseline_m` |
| implicit `2` objects for k (:434-438) | min objects for a depth diagnosis | `correction.evidence.min_objects_for_depth` |
| `DEPTH_COMPRESS_TOL = 0.03` (:41) | \|k−1\| beyond this → depth correction | `correction.solve.depth_compress_tol` |
| `ICP_ITERS = 100` (:42) | trimmed-ICP iterations | `correction.solve.icp_iters` |
| `ICP_TRIM = 0.7` (:43) | trim fraction | `correction.solve.icp_trim` |
| `50000` (:471) | ICP source sample | `correction.solve.icp_sample` |
| `20000` (:484-487) | residual evaluation sample | `correction.solve.eval_sample` |
| `max(100, …)` (:143) | min trimmed correspondences | `correction.solve.icp_min_corr` |
| `1e-3° / 1e-5 m` (:170) | ICP convergence epsilons | `correction.solve.icp_converge_{deg,m}` |
| `_fit_plane`: `tol=0.02`, `300` iters, `60000` sample (:111-115) | plane RANSAC for the no-offset init | `correction.solve.plane_ransac_{tol_m,iters,sample}` |
| `default_rng(0)` (:193, :799, :962) | determinism seed | `correction.solve.seed` |
| `MAX_RESIDUAL_M = 0.15` (:44, :496) | copies must collapse under this median NN | `correction.gates.max_object_residual_m` |
| `0.5 × before` (:496) | required residual improvement | `correction.gates.residual_improvement_ratio` |
| floor band `±0.5` (:407, :510) | held-out floor band half-height | `correction.floor.band_m` |
| `workers=8` (:142, :484-487) | KD-tree query workers | `correction.runtime.workers` (auto) |
| `// 30`, `c*30+15` (:233-234, :420, :567, :796, :878, :907, :984) | fake chunk stride | **eliminated** (keyframe unit + persisted plan) |
| `MAX_TILT_DEG = 10` (:989) | floor-anchor max normal tilt | `correction.floor.max_tilt_deg` |
| `MIN_INLIERS = 5000` (:990) | floor-anchor min inliers | `correction.floor.min_inliers` |
| floor RANSAC `400` iters, `0.02` tol, `0.03` refit band, `120000` sample, `bn < MIN_INLIERS*0.5`, `percentile 5` + `+0.5` band (:1003-1036) | floor-anchor RANSAC internals | `correction.floor.ransac_{iters,tol_m,refit_band_m,sample}`, `correction.floor.min_inlier_ratio`, `correction.floor.low_band_{pct,m}` |
| `compute_chunk_boxes`: `100`, `200000`, percentiles 1/99 (:800-817) | chunk-box stats | **deleted with the gizmo** |
| `1e-3` det tolerance (:853) | manual-gizmo scale guard | **deleted with the gizmo** |
| max step (reported only, :590-599) | continuity | `correction.gates.max_step_{mm,deg}` (now enforced) |
| MEJORAS §3 caps (not yet coded): rot > 10°, \|t\| > 3 m | plausibility caps | `correction.gates.max_rot_deg`, `correction.gates.max_translation_m` |

---

## 5. Real gaps the implementation must create (not blockers — in scope)

1. `output/chunk_plan.json` does not exist → `map_worker.py` starts persisting it
   (`chunk_size`, `overlap`, `chunk_ranges`, sizing inputs, keyframe count) in both
   the direct-chunked and phase-2 paths; absent plan = phase-1 single pass = no
   chunks, keyframe-only operation.
2. No epoch concept anywhere (`grep -rn epoch server/ ui/` → 0 hits) →
   `output/geometry_epoch.json` is new, epoch 0 implicit for existing sessions.
3. `scale_diagnostics.json` has no reader → the scale gate becomes its first
   consumer; regeneration is analytic (§2.7), no depth files needed.
4. Operator identity: JWT machinery exists (`auth/core.py:74-88`) and the UI holds
   the token (`App.tsx:446`) but correction fetches don't send it → UI attaches
   `Authorization`; the ledger records the decoded username, or the explicit
   literal `"unauthenticated"` when no token is presented (recorded, never
   guessed).
5. No session lock for mutations → new per-session correction lock (409 + blocking
   task id), also guarding floor-align and export collisions.

## 6. Blockers requiring the user (none hard; stated for the record)

- **None for implementation.** All information needed exists in the repo.
- Validation on a REAL long-walk session with genuine revisit duplicates (e.g.
  pccr_v1) is an external step only the user can run/judge visually; the synthetic
  suite (§10 of the prompt) covers everything mechanically verifiable.
