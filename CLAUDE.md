# CLAUDE.md — working rules for this repo

## ⭐ BEST CONFIGURATION TO DATE — USER-VERIFIED 2026-08-19 (do NOT change)
The user validated this exact config visually as **the best reconstruction
configuration we have** ("quedó la mejor configuración... de momento no tocamos
más"). It is the restored **2026-08-11 recipe**: everything added between 08-12
and 08-18 was tried, judged worse by the user, and REVERTED. Never re-apply
those reverted knobs without an explicit new decision from him.

    backend vggtomega_pgsr          (full PGSR precision stage)
    simple.conf_percentile 10 — PINNED (USER FINAL 2026-08-30 after the
                                     full sweep 10→25→35→20 in one day: every
                                     value above 10 hollowed weak-texture
                                     surfaces on test3, even vendor-default 20
                                     — VGGT confidence is systematically low on
                                     legitimate flat/textureless surface, so
                                     the gate trades noise for holes. Noise is
                                     handled downstream: SOR, brush, per-mesh
                                     steps. Do not raise without his word.)
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
- **Individual meshing (`/api/segmentation/tsdf/export`) is hybrid
  surface_fit → TSDF** (second decision, same day): with
  `surface_fit.export_first: true`, architectural instances
  (`surface_fit.fitted_roles`) FIRST get a fitted smooth surface — the
  existing `reconstruction/surface_fit` module (RANSAC plane/cylinder/… with
  escalation, `min_inlier_frac` 0.30 as the coverage gate, scene
  regularization, support trimming) — published into `output/tsdf/<name>/`
  (meta `method: "surface_fit"`, untextured) so the viewer picks it up
  unchanged; the full deliverable (residuals/heatmap) stays in
  `output/surface_fit/`. 2026-08-29 refinements (user): per-ROLE model ladder
  (`surface_fit.role_models` — wall/floor/etc → plane ONLY, column/beam →
  plane+cylinder, vault/tunnel keep curved models; the generic escalation took
  wall1 to a 99-DOF b-spline blanket — "espantoso") and border snap in
  `support.py` (mesh borders pulled onto the measured point extent — no more
  ~10 cm overhang past the cloud). Everything else — and every rejected fit —
  gets a PER-OBJECT TSDF integration (USER 2026-08-29: NEVER bake the whole
  scene because one object needs a mesh; the scene mesh is only reused via
  crop when it already exists). `export_tsdf_meshes` depth chain: Stray →
  PGSR renders → backend chunk depth (maplong_run) → DA3 npy; default
  depth_trunc raised 5→12 m (a door 5.3–7 m from every camera integrated
  ZERO frames at 5 m). Same-day refinements (all USER 2026-08-29):
  - GEOMETRY decides, not the label (labels will be BIM names/bare codes):
    unknown-role instances are try-fitted with the generic ladder and accepted
    only if p95 ≤ `unknown_accept_p95_mm` (60); `escalate_rms_gate_mm` (35)
    stops escalation once a model is within construction tolerance — a flat
    wall never becomes a b-spline blanket regardless of its name.
  - `ceiling` ladder includes the curved models (curved ceilings were falling
    to broken TSDF; flat ones still stop at plane via the rms gate).
  - Support = ON-SURFACE points only (`support_dist_m` 0.04): off-surface
    points no longer fake support (wall3's opening stayed filled).
  - Unexplained remainder (`unexplained_dist_m` 0.10): points the model can't
    explain (wall3's attached cone) are TSDF'd per object and FUSED into the
    same GLB (`fused_rest_points` in meta) — the whole object is delivered.
  `run_surface_fit.py --instance-id` now routes through `fit_scene` (roles +
  regularization), same as `--all`. Without PGSR renders, `depth_source: auto`
  falls back to the backend's native depth (artifact-based — sessions that DID
  run PGSR still integrate its renders).
- **Chat = spatial intelligence** (USER 2026-08-29): the assistant must KNOW
  what it is looking at, not only measure. phase5_qa additions: session header
  + per-object y-ranges in the system prompt; `get_session_info`;
  `describe_scene` (VLM looks at sampled scan frames, description cached in
  the store as `scene_description`, provenance vlm_proposed);
  `remember_note`/`recall_notes` (persistent conclusions in the store db);
  part-aware measuring — `measure_between` features top/upper, bottom/base/
  lower (REAL point bands, not OBB faces), highest/lowest, closest, plus
  `axis: vertical`; `get_extent` (lowest/highest band centres — a curved
  ceiling has both). Validated on test3: floor→ceiling lower 1.28 m vs upper
  3.35 m; ladder base→top 2.43 m.
- **Meshing = RANSAC + Poisson, TSDF out of the automatic chain** (USER
  2026-08-29, evaluation mode): the per-object TSDF re-integration was 5×
  worse than Poisson from the object's OWN cloud points (wall3 cone: p95
  104 mm / 30% uncovered vs 20 mm / 1.4%; ladder 4.9 mm, door 6.3 mm, 0%
  gaps) — cloud-anchored meshing is consistent because the cloud is the
  validated truth. `/tsdf/export` now publishes BOTH per instance: the
  surface_fit mesh (`<label>_<id>/`, remainder fused via Poisson) and a
  Poisson mesh with cloud vertex colors (`<label>_<id>_poisson/`) so the
  user compares them in the viewer. New: `segmentation/poisson_object.py` +
  `run_poisson_objects.py` (subprocess, `os.sched_setaffinity` to 8 cores —
  **ENVIRONMENT LESSON: Open3D Poisson hangs UNPINNED on this 252-core box;
  TBB ignores OMP_NUM_THREADS**; verified: infinite hang unpinned, ~1 min/
  object pinned). UI: one `🧩 Meshing` modal — segment selection + two
  buttons only: `Object` (MeshFlow generative) and `Mesh` (ransac+poisson);
  whole-scene buttons and TSDF sliders removed.
- **Stage-1 hole audit — "understand what we reconstruct"** (USER CONCEPT
  2026-08-29): a hole in a fitted surface is either a REAL opening or a
  reconstruction gap, and the SCAN FRAMES are the witness.
  `reconstruction/surface_fit/hole_audit.py`: every unsupported UV cell is
  projected into the SAM3 mask keyframes (poses+K from session_io; K lives on
  the TRACE grid 384×688, NOT the RGB grid; mask npz key `f<frame>_o<id>` with
  **oid = instance_id − 1**, self-calibrated per instance by projecting its own
  points — validated 90–100% hit on test3) and voted: covered → filled
  (`image_supported`), uncovered → real opening (border follows the mask at
  cell resolution), ambiguous → open (never invent). Wired into
  `fit_scene` export (config `surface_fit.hole_audit` + ratio/votes gates);
  residual reports still use measured points only. test3 results: wall3
  +3.96 m² image-supported fill (wall continues behind the attached cone),
  wall2 8.3 m² confirmed open (door opening preserved), fills ≤0.03 m².
  Stages 2 and 3 shipped same day (USER: "incluso con razonamiento de
  ocluido, importantísimo"):
  - OCCLUSION REASONING: three-way vote per hole cell (own mask = direct
    witness / OTHER instance's mask = occluded / no mask = sees past). No
    direct witness but occluded in ≥`hole_occluded_ratio` of views → filled
    as `occlusion_inferred` (floor behind the ladder: 2313 cells, 5.8 m²).
  - Stage 2: the audit runs on CURVED surfaces too (bspline ceiling: 2177
    fills), plus `silhouette_report` — mesh footprint vs own mask per
    keyframe (precision/recall/IoU, tool_measured; high precision ≈0.86 =
    never where the images say nothing; recall marks what remains to cover).
  - Stage 3 (`texture_objects: true`): `bake_object_glb` (texture_bake)
    bakes a texrecon atlas from the scan frames onto every per-object mesh —
    fitted+audited surfaces, Poisson meshes, fused remainders. Regions no
    camera saw stay vertex-coloured (`unseen_vertexcolor` submesh) — texture
    is never invented either.
  - v2 fixes after the user's first full run (2026-08-29 evening — masks are
    2-D, they needed DEPTH): (1) votes are Z-BUFFER-verified (per-frame
    z-buffer from the full cloud at mask res) — occluded only when measured
    geometry sits ≥15 cm in front; a coplanar in-fill object (door leaf in
    wall2) is NOT an occluder → the doorway stays open (fills went
    3188→7 on wall2, 612→9 on wall1); (2) occlusion-inferred fills only in
    ENCLOSED holes (support on all four grid sides — ladder shadow on the
    floor yes, phantom extension past a wall edge no); (3) curved models cap
    fills to ≤3 cells from support (bspline extrapolation spiked the
    ceiling); (4) texrecon SEGFAULTED on audited grid meshes (duplicate
    verts/sliver faces from the border snap) — bake_object_glb welds +
    drops degenerates first. Ceiling precision 0.87→0.97, wall2 IoU
    0.63→0.67. Poisson meshes deliberately do NOT get audit fills (they are
    the "as measured" deliverable). Open item: wall1_poisson texture looked
    slightly displaced to the user once (not reproduced later).
  - v3 fixes after the user's second run (2026-08-29 night): (1) 'covered'
    votes require DEPTH CONSISTENCY — measured geometry BEHIND the surface
    through a cell (>15 cm) means the camera sees PAST it → OPEN (wall3's
    access ARCH was filled because the attached cone — same instance, same
    mask — was visible through it; now 3.18 m² open); (2) face WINDING
    toward the nearest camera before texrecon — arbitrary grid winding made
    texrecon label whole surfaces back-facing/unseen → untextured (wall3
    went 807 textured/12036 unseen → 2997/150); (3) 3-D spike crop for
    curved models (mesh verts >25 cm from measurement are spline behaviour —
    ceiling dropped its 50 spike verts); (4) `hole_interp_max_cells` (30):
    tiny ENCLOSED no-verdict gaps are interpolated across the fitted surface
    (provenance 'interpolated') — the bounded bridging Poisson does
    implicitly, so ransac no longer loses on small holes.
- **Multi-primitive decomposition** (USER 2026-08-29, the train: "donde se
  pueda aplicar ransac — conos, circunferencias, planos — debe aplicarse, y
  poisson donde no hay manera"): `surface_fit/decompose.py` — iterative
  largest-support-first plane/cylinder/sphere extraction (`extract_primitives`)
  over (a) instances no single model explains (unknown-role rejects / no fit)
  and (b) large unexplained remainders of accepted fits. Each primitive is
  meshed with the fitted machinery; parts merge into the instance's
  surface.glb (`parts` in results, `forced_leftover` keeps Poisson from
  re-meshing what primitives claimed); only the residue goes to Poisson.
  Config `surface_fit.decompose*`. Validated: wall3's attached structure →
  sphere quadric, 40k pts, rms 19 mm, residue 7.5%. Fix (same night): the
  train extracted ZERO primitives — the fitter's INTERNAL gate (10%) equalled
  the first plane's real share (~10.5% at 1.2 cm) so acceptance flipped on
  the RANSAC seed; decomposition now hands the fitters permissive gates
  (min_inlier_frac 0.02, dist_thresh = decompose_inlier_dist_m) and OUR
  acceptance decides → train: 6 planes, 2.13 M pts (61%), 39% Poisson residue.
- **Contour regularization** (USER CONCEPT 2026-08-29: "las formas tienden a
  ser perfectas — detectar la tendencia para perfeccionarlas"):
  `surface_fit/contours.py` — every boundary/opening of a fitted PLANE is
  vectorized (cv2) and tried against a 2-D shape ladder (circle, rectangle,
  rounded rect, arch = rect+circular cap, direction-snapped polygon, raw);
  among templates passing the `contour_tol_m` p95 gate, the LOWEST-DEVIATION
  one wins (tie → lowest DOF). The OUTER outline is fitted against the
  morphologically CLOSED support (the boundary's intent); OPENINGS are carved
  only from the audit's image-confirmed open cells. Mesh rebuilt on a fine
  grid clipped to the shapely region with boundary vertices PROJECTED exactly
  onto the ideal outline (CAD-crisp edges; no earcut/triangle deps needed).
  Shape parameters land in `contours` of the results + hole_audit.json
  (tool_measured). Also same-day: arch leak fixed via 5-px minimum-filter
  Z-buffer (background seen through the arch had no measured point on the
  exact pixel), and wall3's remainder now decomposes into sphere+plane with
  only ~4k residue pts to Poisson.
- **Chat runs in a worker thread** (fix 2026-08-29): SpatialQA.ask ran inline
  in the async endpoint and froze the WHOLE backend event loop for the 30–120 s
  of the tool loop (/health 000 → UI hung). Both chat paths now run_in_executor.
- **Evaluation volumes are first-class objects** (USER 2026-08-29): chat
  `define_volume` RESTS the box ON the floor by default, centred on the floor
  (or `anchor_id`); accepts `volume_m3` (cube). The volume appears in the
  viewer IMMEDIATELY (panel refresh on define_volume in the trace). In the
  viewer: click a volume (navigate tool) → gizmo toolbar (Move / Rotate[yaw] /
  Resize / Solid / Delete); edits persist via `/api/scene/volumes/update`
  (new `update_user_volume` in the store) and the box is tinted by collision
  state vs the scene (evaluate: violet=free, amber=touching, red=colliding).
  Volumes are raycast targets of the measure tools, and the chat measures
  from them with `measure_volume` (box-surface clearance, intersects flag).
- **Chat must interact with the 3D model**: text-only answers with no animated
  measurements mean the session resolved NO instance store. Fixed 2026-08-28:
  `_resolve_store` (phase5_qa/api.py) rebuilds `scene_r.db` on the fly from
  `segmentation_result.json` via `segmentation.pipeline.rebuild_instance_store`
  (validated bit-identical to the matcher-built store on test3). The tool-less
  general-chat fallback is ONLY for sessions with no segmentation at all.
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
- Keyframe quantum: 60 SINCE 2026-08-30 (USER: "muy muy pocas" — 250 gave 12
  views/300 frames; PGSR out of the flow changed the old densify-worse
  verdict; watch for drift). Coverage trim (rotation/static ends) OFF same
  day by user order (simple.coverage_trim: false). (History: 250 on 08-16;
  80 on 2026-08-15 — USER DECISION after a visual A/B on
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
