# R3D Reuse Catalog

Vendored at `/workspace/stac-build/vendor/r3d`. Paths are relative to that root
unless absolute. Line numbers are from the current checkout
(`git 9669cacd`). This is the reuse map for **Phase R** (semantic anchoring of
poses/depth) and **Phase 5** (spatial Q&A). Provenance note for the future:
anything ported/adapted from here is tagged `# from R3D` in our modules.

> Scope correction: **`gap_fit` is NOT a runtime spatial tool.** It is an
> offline QA `QuestionType` / question generator
> (`r3d/data_gen/generators/multihop.py:57` `generate_gap_fit`, enum
> `annotation_schema.py:79`). The LLM-callable spatial tools are exactly the 7
> defined in `tool_use.py`.

## 1. Spatial tools catalog (LLM-callable)

Source: `r3d/pipeline/eval/tool_use.py`. Tools dispatched by string name in
`_execute_tool` (`tool_use.py:206-232`). The model emits
`TOOL_CALL: name(arg1, arg2)`, parsed by regex `_TOOL_CALL_PATTERN`
(`tool_use.py:52-54`): `TOOL_CALL\s*:\s*(\w+)\(\s*(.*?)\s*\)`. Args split on
commas, stripped of quotes (`_parse_args`, `tool_use.py:91-92`). Object IDs are
**numeric ints**; names are rejected (`_resolve_id`, `tool_use.py:101-109`).

Axis convention: internally positions are scene coords `(x, y, z)` with **y up**.
But `get_position`/`get_my_position` output strings swap to `(x, z, y)` so the
3rd printed coordinate is vertical, and the docs tell the model "z is the
vertical (up) axis" (`tool_use.py:37-38`, formatting `:137`, `:221`). OBB AABB
layout is `[xmin,xmax, ymin,ymax, zmin,zmax]`, y vertical.

| Tool | Signature | Impl (file:line) | Returns / units | Description |
|------|-----------|------------------|-----------------|-------------|
| `list_objects` | `list_objects()` | `_tool_list_objects` `tool_use.py:195-203` | `Tracked objects (N):\n  ID <id>: <query_name>` | Discovery. Objects whose visibility window overlaps the query window via `scene.get_tracked_object_ids_in_window(start_ns,end_ns)`. Call first. |
| `get_distance` | `get_distance(id1, id2)` | `_tool_get_distance` `tool_use.py:112-125` | `"<d>.2f meters"` | OBB-to-OBB surface distance (not centroid). `_obb_distance`→`_closest_point_on_obb`. 0.0 if a center is inside the other's OBB. |
| `get_position` | `get_position(id)` | `_tool_get_position` `tool_use.py:128-137` | `"(x, z, y) meters"` (vertical last) | Object position = best reconstruction's OBB center (`recon.position`). |
| `get_my_position` | `get_my_position()` | inline `tool_use.py:219-221` | `"(x, z, y) meters"` | Camera position at `timestamp_ns_end`. `frame.T_scene_device[:3,3]` via `_get_camera_position` (`:83-88`). |
| `get_distance_from_me` | `get_distance_from_me(id)` | `_tool_get_distance_from_me` `tool_use.py:140-157` | `"<d>.2f meters"` | Camera position to nearest point on object OBB (`_closest_point_on_obb`). |
| `get_object_size` | `get_object_size(id)` | `_tool_get_object_size` `tool_use.py:160-171` | `"width=<w>m, height=<h>m, depth=<d>m"` | OBB extents `aabb[1]-aabb[0]`, `aabb[3]-aabb[2]`, `aabb[5]-aabb[4]`. |
| `get_object_volume` | `get_object_volume(id)` | `_tool_get_object_volume` `tool_use.py:174-192` | `"estimated_volume=<v> cubic meters (<L> liters), bounding_box_volume=<bv> cubic meters"` | SAM3D mesh cavity/functional volume rescaled to OBB (`volume.mesh_rescaled_volume`) + `bbox_volume(obb_aabb)`. Needs `MeshStore`. |

Unknown tool → error string (`:232`). Only first `TOOL_CALL` per turn honored.

### Core geometry helpers (reusable for anchoring)
- `_invert_rigid(T)` `tool_use.py:57-61` — SE(3) inverse (`R.T`, `-R.T@t`).
- `_closest_point_on_obb(point, recon)` `tool_use.py:64-71` — world point → OBB
  frame (`_invert_rigid(recon.obb_transform)`), clip to `[lo,hi]` from
  `obb_aabb`, back to world. Primitive for point-to-box distance.
- `_obb_distance(r1, r2)` `tool_use.py:74-80` — symmetric closest-point distance,
  0 on penetration.

## 2. Scene representation

**`SceneState`** (`r3d/pipeline/scene_state.py:27-176`) — read-only scene API
over a `SceneStore` (+ optional `MeshStore`), scoped to one `sequence_id`.
Thread-safe reads; caches `object_id → SceneObject` (`_get_object_map`, `:45`).

Key methods: `get_object_ids/get_object/get_all_objects` (`:57`),
`get_best_reconstruction(id, policy="best_psnr"|"latest")` (`:66`),
`get_object_position` (`:96`), `get_object_bbox_3d`→`(obb_aabb, obb_transform)`
(`:103`), `get_initial_object_bbox_3d` (`:112`), `get_visible_objects_at` /
`get_frame_visibility` (`:121`), `resolve_by_name` (`:136`),
`get_tracked_object_ids_in_window` (`:146`), `get_object_volume` (`:153`).

**Per-object data model** (Pydantic, `r3d/pipeline/stores/base.py`):
- `SceneObject` (`base.py:26-35`, frozen): `sequence_id, object_id:int,
  query_name, first_seen_ns, last_seen_ns`. Identity record, no geometry.
- `ObjectReconstruction` (`base.py:38-58`): geometry — `reconstruction_id,
  sequence_id, object_id, time_range_start/end_ns, obb_aabb(6), obb_transform(4,4),
  position(3), initial_obb_*, num_gaussians, psnr, ssim, lpips, created_ns`.
- `FrameVisibility` (`base.py:61-71`): `sequence_id, timestamp_ns, object_id,
  bbox_2d(4), mask_rle:dict|None, sam3_score`.
- `ObjectCoverage` (`base.py:74-84`): `num_views, angular_span_deg,
  num_distinct_viewpoints, mean_visibility_ratio`.
- `ObjectMesh` (`base.py:87-109`): SAM3D mesh metadata keyed `(sequence_id,
  object_name)`; `mesh_path`, `metric_scale_{x,y,z}`.
- `ObjectPoints` (`base.py:112-120`): `points(N,3), num_points` — raw per-object
  point cloud.

Masks stored RLE (COCO `{counts,size}`); decode `r3d/utils/rle.py:decode_rle_to_mask`.

## 3. build_scene pipeline (depth-lift → OBB)

- **`scene_builder.py`** — metadata compilation (`build_scene`, `:27`):
  `_build_scene_objects` (`:46`), `_build_frame_visibility` (`:80`),
  `_build_object_coverage` (`:199`; angular coverage `_compute_angular_coverage`
  `:161` = 30° azimuth bins mod 12 + 30° elevation, span = max pairwise view-dir
  angle).
- **`scripts/build_scene.py`** — the 3D reconstruction driver. **Port this.**

Per sequence (`_process_sequence`, `build_scene.py:360-448`):

**A. Depth-lift mask→3D.** `_collect_object_points` (`:323-357`) × timestamps ×
query_names, decode mask, optional erode, lift. `_lift_depth_to_3d` (`:47-79`):
```python
vs, us = np.mgrid[0:h:stride, 0:w:stride]        # stride subsample (default 4)
valid = mask_sub & (depth_sub > 0)
x_cam = (us - cx)/fx * d ; y_cam = (vs - cy)/fy * d ; z_cam = d
pts_world = (T_scene_device @ T_device_camera @ [x,y,z,1]^T)   # :76-77
```
`(ts,obj_id)` dedup via `seen` set.

**Mask erosion:** `_erode_mask` (`:316-320`) — `cv2.MORPH_ELLIPSE` kernel
`(2*e+1)`, 1 iteration. `--mask-erosion` default **0** (script overrides the
function's own default of 8). Applied before lifting.

**B. Subsample cap.** `> --max-points-per-object` (default 50000) → random
subsample seeded by `obj_id` (`:390`).

**C. 3-stage filtering** (`:396-406`):
1. `_filter_outliers_knn` (init-KNN) `:82-94` — k=6, keep points with
   mean-kNN-dist `< 3.0 × median_nn_dist`. No-op if ≤ max(k,20) points.
2. `_multiview_consensus_filter` (**plurality / identity vote**) `:97-188` —
   reproject each 3D point into every frame where the object is segmented; keep
   iff inside the object's SAM3 mask union in **strictly >50%** of in-bounds views
   (`keep = visible & (obj_votes > total_visible/2)`, `:186`). Geometry-only pose
   load (`load_frame_pose`, `sqlite_store.py:300`). No depth/occlusion test.
3. `_filter_outliers_knn` again (bbox-KNN) `:402`.

**D. Gravity-aligned OBB fit.** `_fit_gravity_aligned_obb` (`:191-257`), see §6.
Objects with `<3` points skipped. Points persisted via `points_store.write_points`
(`:408`); OBB written as `ObjectReconstruction` with `initial_*` mirroring final
(`:423-441`).

CLI (`_build_parser`, `:260`): `--frames-dir --dataset --output-dir --knn-k(6)
--max-points-per-object(50000) --min-points(6) --depth-stride(4)
--mask-erosion(0)`. Segmentation source: HF parquet (`ParquetSegmentationStore`).

## 4. Store schema (SQLite)

`stores/sqlite_store.py`. WAL, `synchronous=NORMAL`, numpy arrays as **float64
`tobytes()` blobs** (`_ndarray_to_blob`/`_blob_to_ndarray`, `:106-111`);
read-only via `mode=ro` URI.

**scene.db** — `SQLiteSceneStore` (`:641-916`):
```
scene_objects(sequence_id TEXT, object_id INT, query_name TEXT,
              first_seen_ns INT, last_seen_ns INT, PK(sequence_id,object_id))       # :649
object_reconstructions(reconstruction_id INT PK AUTOINC, sequence_id, object_id,
   time_range_start_ns, time_range_end_ns,
   obb_aabb BLOB(6), obb_transform BLOB(4x4), position BLOB(3),
   initial_obb_aabb, initial_obb_transform, initial_position,
   num_gaussians INT, psnr REAL, ssim REAL, lpips REAL, created_ns INT)             # :657
frame_visibility(sequence_id, timestamp_ns, object_id, bbox_2d BLOB(4),
   mask_rle TEXT(JSON), sam3_score REAL, PK(sequence_id,timestamp_ns,object_id))    # :677
object_coverage(sequence_id, object_id, num_views INT, angular_span_deg REAL,
   num_distinct_viewpoints INT, mean_visibility_ratio REAL, PK(...))                # :686
```
Row→model reconstructor `_row_to_recon` (`:840`) decodes blob shapes.

**object_points.db** — `SQLiteObjectPointsStore` (`:1277-1323`):
```
object_points(sequence_id TEXT, object_id INT,
  points_blob BLOB(float32 (N,3)), num_points INT, PK(sequence_id,object_id))
```
Points are **float32** here (unlike float64 elsewhere). Natural table to extend
for our per-instance store (Phase R.8).

**frames.db** — `SQLiteFrameStore` (`:119-338`): `frames(sequence_id,
timestamp_ns, rgb_path, depth_path, fx,fy,cx,cy, img_width, img_height,
T_scene_device BLOB(4x4), T_device_camera BLOB(4x4), gravity_world BLOB(3)|NULL,
depth_source TEXT, PK(...))`. Depth = uint16 **mm** PNG (range [0, 65.535 m]).
`load_frame_pose` (`:300`) = geometry-only fast path used by the vote.

**segmentations.db** — `SQLiteSegmentationStore` (`:454`): `segmentations(...,
object_id, bbox_2d BLOB, mask_rle TEXT, score REAL, obj_ptr BLOB, min_depth_m REAL,
PK(...,object_id))`.

**mesh.db** — `SQLiteMeshStore` (`:346`). **annotations.db** —
`SQLiteAnnotationStore` (`:924`), incl. `gt_object_bboxes(annotation_id,
object_position, timestamp_ns, obb_aabb BLOB(6), obb_transform BLOB(4x4))` and
`get_nearest_gt_bbox` (`:1249`).

**Parquet mirror** (`stores/parquet_store.py`, read-only): loads HF
`facebook/r3d-bench`; all writers `NotImplementedError`.

## 5. VLM eval / tool-use loop

Entry `run_tool_use(annotation, images, scene, frame_store, sequence_id, vlm,
model, max_turns=10, ...)` (`tool_use.py:352-381`).
1. `_build_initial_messages` (`:235`) → `[SYSTEM(TOOL_DESCRIPTIONS),
   USER(question + base64 JPEG attachments)]`. `image_to_base64` JPEG q90
   (`vlm.py:101`).
2. `_run_tool_loop` (`:309`): initial `vlm.query_multiturn`, then ≤ max_turns of
   `_handle_tool_turn`.
3. `_handle_tool_turn` (`:262`): regex-match a `TOOL_CALL`; none → done; else
   `_execute_tool`, append AI response + `USER("Tool result: <r>")`, re-query.
4. Returns `(history_text, tool_log[{tool,args,result}])`.

**Text-protocol, not native function-calling.** `VLMClient` ABC (`vlm.py:29`);
backends `VLLMClient` (`eval/vllm_client.py`, in-process offline `LLM`) and
`HFVLMClient` (`eval/hf_vlm.py`). System prompt = `TOOL_DESCRIPTIONS`
(`tool_use.py:21-50`, natural-language tool list, no JSON schema) +
`ANSWER_FORMAT_INSTRUCTIONS` (`eval/prompts.py:5`, final answer `ANSWER: <value>`
no units).

> Our Phase 5 differs: we use Qwen3 **native** tool-calling (JSON tool schemas
> via `--tool-call-parser hermes`) over the persistent service, not R3D's
> `TOOL_CALL:` text protocol. The tool *semantics* and `SceneState` port; the
> loop is `SemanticClient.run_tool_loop`.

## 6. Gravity / OBB (pose anchoring)

`_fit_gravity_aligned_obb(points)` — `build_scene.py:191-257`. Gravity along
world **Y** (up):
```python
y_min, y_max = points[:,1].min(), points[:,1].max()   # vertical extent from Y
xz = points[:, [0,2]]                                  # ground-plane projection
cov = np.cov((xz - xz.mean(0)).T)
eigvals, eigvecs = np.linalg.eigh(cov)
axes_2d = eigvecs[:, argsort(-eigvals)]                # PCA principal axes in XZ
proj = (xz - xz.mean(0)) @ axes_2d
extent = [p_max[0]-p_min[0], y_max-y_min, p_max[1]-p_min[1]]
# 8 corners (2 heights × 4 in-plane) → world; position = corners.mean(0)
# frame axes from corner edges (normalized); obb_transform 4x4 (cols=axes)
obb_aabb = [-ex/2, ex/2, -ey/2, ey/2, -ez/2, ez/2]     # centered, y up
```
Returns `(obb_transform(4,4), obb_aabb(6), position(3))`. Properties:
- Up axis (col 1) is exactly world +Y — gravity alignment enforced by PCA-ing
  only XZ + taking vertical from Y. Yaw data-driven; roll/pitch zero.
- `obb_aabb` centered (`±extent/2`); placement lives in `obb_transform`. Pairs
  with `_closest_point_on_obb`/`_invert_rigid`.
- Gravity is **implicit** (world Y). `frames.gravity_world` is captured
  (`sqlite_store.py:141`, `populate_frames.py:188`) but NOT consumed by the
  fitter. For our poses (which are not guaranteed Y-up) we must rotate points
  into a gravity frame first — which is exactly what Phase R.4 estimates from the
  floor/slab plane + ChArUco.

## Reuse pointers
- **Phase R (anchoring):** `_lift_depth_to_3d` + `_fit_gravity_aligned_obb`
  (`build_scene.py:47,191`) → per-instance OBBs from mask+depth;
  `_multiview_consensus_filter` (`:97`) → point→instance plurality vote;
  `_closest_point_on_obb`/`_invert_rigid` (`tool_use.py:57-71`) → point↔box math.
  Persist via a table modeled on `object_points`/`object_reconstructions`.
- **Phase 5 (Q&A):** the 7 tools + `SceneState` port directly; swap the text
  protocol for Qwen native tool schemas over `SemanticClient`.
