# CLAUDE.md — working rules for this repo

## ⭐ BEST CONFIGURATION TO DATE — USER-VERIFIED 2026-08-19 (do NOT change)
The user validated this exact config visually as **the best reconstruction
configuration we have** ("quedó la mejor configuración... de momento no tocamos
más"). It is the restored **2026-08-11 recipe**: everything added between 08-12
and 08-18 was tried, judged worse by the user, and REVERTED. Never re-apply
those reverted knobs without an explicit new decision from him.

    backend vggtomega_pgsr          (full PGSR precision stage)
    simple.conf_percentile 20 — USER 2026-09-04 (10→20, his word, with the
                                     DINOv3 fase-2 two-witness filter now
                                     active downstream). (Previous: 10 —
                                     PINNED USER FINAL 2026-08-30 after the
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

## ⭐ CORRECTION MODULE — USER DECISION 2026-09-08 (redesign, supersedes the
## 2026-09-06 chunk-gizmo corrector)
"El gizmo se elimina; queda solo la corrección por objetos marcados; el sistema
resuelve de forma consistente con toda la escena; nada hardcodeado; época
geométrica obligatoria en todo artefacto derivado."
- `server/correction/` replaces `segmentation/correction_analysis.py` (deleted,
  with `apply_manual_chunk`, `compute_chunk_boxes`, `align_floor_y0`, the
  `/api/segmentation/chunks/*` + `/api/segmentation/correction/*` endpoints and
  the UI chunk gizmo — the evaluation-VOLUMES gizmo stays, it is another
  feature). New API: `/api/correction/*` (run/floor/approve/undo/state/ledger/
  artifacts), per-session lock (second op → 409 + blocking task id).
- The atomic unit is the KEYFRAME (per-point provenance `frame_global`); the
  fake `// 30` chunk bucket is gone. Real chunks only from
  `output/chunk_plan.json`, now persisted by map_worker on every chunked run.
- The flow: mark → evidence (visits = keyframe runs in the curated OBB;
  reference = earliest visit) → DOF observability (PCA: a lone plane/column
  never gets DOF it cannot observe; rejection says what else to mark) →
  diagnose pose|depth+pose (internal fingerprint) → solve (trimmed yaw-planar
  ICP; depth k along each point's own ray) → GATES (ALL veto: object collapse
  incl. per-object centroids, plausibility rot≤10°/|t|≤3 m, scene exam =
  floor + unmarked witnesses, continuity per keyframe, scale vs DA3 anchors —
  analytic over scale_diagnostics.json, override recorded in the ledger) →
  distribute (identity to ref end, slerp+lerp anchors) → TRANSACTIONAL apply
  (everything staged in `_tx_epoch_<N>/` incl. the Potree build, verified,
  journaled atomic swap; previous epoch in `_epoch_<N-1>/` until
  Approve/Undo) → in-place instance-store refresh (findings re-anchored per
  keyframe; user volumes NEVER move).
- Geometry epoch (`geometry_epoch.json`, epoch 0 = original reconstruction):
  every derived artifact (tsdf/surface_fit/poisson/perfect/mesh_export/
  hole_audit/sábana/coverage/phase-5 measurements/phase-6 report) is stamped
  `geometry_epoch` + `human_directed_corrections` + `corrections_overridden`;
  stale ones get a UI badge + Regenerate (never auto-regenerated). Depth
  corrections live in the `depth_correction.json` sidecar served ONLY through
  `segmentation/session_io.correct_depth` (DA3-anchor/Stray witness depth is
  never corrected). Ledger `corrections.jsonl` is append-only (undone runs
  keep their verdict; a new reconstruction resets the epoch but NEVER deletes
  the ledger); `corrections/epoch_<N>.npz` + `python -m correction.replay`
  reproduce any epoch bit-faithfully (keyed by frame_global → re-appliable to
  a re-reconstruction).
- Floor alignment is the same flow (kind=floor): per-keyframe anchors vs an
  explicit model — level | plane (default; real slopes survive) | profile —
  with step-demotion (real level changes preserved), anchor-normal smoothing
  (patch-noise rotations at 15 m lever arms became 75 mm steps) and
  `min_tilt_deg` (below it only the height lands).
- **USER 2026-09-09 (first real run, pccr): "no debes rechazar correcciones
  por umbrales arbitrarios... siempre debe aplicarse la corrección de piso y
  de duplicados, no importa lo mucho que haya que corregir."** →
  `correction.gates.mode: advisory` (default): every gate is still MEASURED
  and lands in the report as a ⚠ warning, the correction is APPLIED and the
  visual Approve/Undo is the verdict (`veto` keeps the blocking behaviour for
  evaluation). Same day, after two more real runs:
  - **Closure distribution = DRIFT-RATE model (USER's formulation)**: the
    measured position of every keyframe carries the accumulated error
    E(d) = ε·d of the distance WALKED since the start (E(0) = 0, the start is
    exact). A duplicate pins the line (ε = closure / Δchainage between the
    two copies, each attributed to the median chainage of its evidence
    points); more duplicates pin a piecewise curve. Every keyframe gets
    −E(d_k): little near the start, growing along the walk, extrapolated
    forward — the REFERENCE copy moves too (C(d_j) = C(d_ref) ∘ T_j, exact
    for the separable model). Removed (both smeared the closure over
    keyframes that were RIGHT and duplicated correct sectors): the
    linear-in-keyframes spread with identity up to the reference visit, and
    the per-chunk seam-weighted blocks. Declared limit: one duplicate
    observes the net translation/yaw; heading curvature needs a second.
  - The "floor as a constraint" of the object correction was REMOVED (USER:
    the per-keyframe low band is not a validated floor curve — "es
    mentira"). Floor alignment (kind=floor) stays: reference = floor of
    the first `reference_span_kf` keyframes (a plane over the whole drifted
    floor followed the drift, 3° tilt, invisible fix), anchors follow the
    trend (moving median heights + mean normals over `smooth_window_kf`,
    off-trend anchors = furniture → demoted); ONE floor button in the UI
    (plane; level/profile stay API-selectable).
  - A BOUNDED object whose two copies share extents observes the full
    translation (`observability.bounded_extent_tol`; desk1 was solved along
    its normal only). Segment OBBs appeared ROTATED against the cloud:
    `_display_matrix` composed the project composition transform into the
    active scan's display — the main cloud now shows in the scan's own
    floor frame only (composition is a fusion concern). The 🔧 modal closes
    itself when a run is applied so the viewer + collapsible verdict box
    are free.
- ALL parameters in `config.yaml` `correction:` (typed dataclasses; a missing
  key fails at load naming it; zero decision literals in the package —
  enforced by test). Tests: `server/tests/test_correction_*.py` + shared
  generator `tests/synth_correction.py` (30 tests, no GPU).

## ⭐ CORRECTION ALGORITHM — USER DECISION 2026-09-18 (visit drift replaces the
## greedy loop; it IS the correction now)
**"esto agregalo completo como algoritmo de correccion en lugar del que
tenemos actualmente porque el piso quedo perfecto, hasta las lineas
perfectamente alineadas … que se generen tantas epocas automaticamente hasta
que no mejore mas … ademas antes aplicale a la nube el filtrado que te dije en
cada etapa, por revisitas que no aportan etc … ademas se debe aplicar floor
transform en cada ajuste tambien"**

`correction/visit_drift.py` (the measurement) + `correction/visit_drift_run.py`
(the epoch loop). One epoch =

1. **FILTER the cloud for real** — a visit contributing < `min_visit_share` of
   an object never observed it, it grazed it; an object under `min_points`
   cannot be measured. Both are DELETED (like `single_witness`), the instance
   index is reindexed, and an instance left with no points stops existing.
2. **MEASURE** every object two separated visits saw, by aligning the
   SILHOUETTES of its two copies in the three orthogonal views of its OWN OBB
   (plan L-W, side L-U, front W-U) by normalised FFT cross-correlation. Each
   view fixes two components, so **every component is measured twice and their
   disagreement is the determination test** — it replaced 3-DOF-vs-6-DOF ICP.
   3-D ICP SLIDES on flat/symmetric surfaces (pccr epoch 2: ICP said 1.4 cm of
   residual drift on `desk#203` while the plan view showed the two tops
   plainly offset); the centroid hides it too (3.1 cm), compensating what is
   missing on one side with what is extra on the other.
   Filters before it, in order: 2+ visits · `min_points` · visit share ·
   `min_walk_m` of WALK between visits · `min_comparability` of IQR extent
   (never max−min: one flyer stretches the OBB).
3. **CORRECT** — every DETERMINED closure (disagreement ≤ 2× the session's own
   repeatability) becomes a knot; closures at the same moment of the walk, or
   agreeing within that same bar, are ONE knot (weighted mean). All of them go
   to `distribute` TOGETHER: feeding one closure per epoch and iterating made
   the loop oscillate (pccr, monitor 76.5→0→48.6 while the door went
   64.7→49→0.5).
4. **PUBLISH** a selectable epoch — cloud, poses, segmentation, octree, epoch
   record, per-keyframe npz — and **RE-LEVEL THE FLOOR** on the new geometry.
   Every correction tilts the scene a little and the display frame has to
   follow or the floor walks off y=0 epoch after epoch.

It **stops before publishing** when what the closures AGREE on falls under the
session's own repeatability: once the survivors point in opposite directions
their weighted sum is ~0, and what each still shows is disagreement BETWEEN
objects, which no motion of the walk removes. Judged after publishing, this
cost two epochs that moved the cloud by 2 mm and rebuilt a 22 M-point octree
to do it. Second stop: an epoch that corrects no less than the previous one.

Measured on pccr 2026-08-31 (19.3 m walk, 22.1 M points, 82 objects):
67.0 cm applied → 8.2 cm → converged, 3 objects testifying together
(76.5/70.9/64.7 cm → 23.4/18.6/24.8 cm), floor +334.9 mm/2.90° → 0.

**UNHOOKED AND DELETED the same day** (USER: *"luego desenganchamos el anterior
y lo eliminamos asi no queda basura"*): `reconstruction/certify/iterate.py`
(the §9 GREEDY loop and `greedy_is_the_correction`), its `certify.greedy:`
config block and `GreedyConfig`, and its two test files. On the pccr run that
produced the session's acta the greedy chain was the ONLY stage that moved
geometry and it REGRESSED the objective by 4.5 %. The keyframe pose graph
still RUNS inside the certification and its loop residual reaches the acta —
it is MEASURED, never applied (`stages.poses.verdict = "MEASURED"`). Scale and
depth still solve their own degrees of freedom, which visit-drift never
touches.

Consequences wired the same day:
- `certify_session` measures `metrics_initial` BEFORE the correction. It used
  to be iteration 0's own starting state, which was the same thing only while
  the correction happened inside iteration 0 — after the move it compared the
  corrected session with itself and every §9 gate passed by construction.
  `metrics_final` is the last state MEASURED, not the last one applied.
- An epoch carries what it DELETED (`dropped` in `corrections/epoch_<N>.npz`)
  and `correction.replay` honours it, so an epoch that both moved 22 M points
  and deleted eight thousand still replays exactly.
- `geometry_epoch.json` is an epoch ARTIFACT: it travels with the geometry.
  `select_epoch` swaps the union of the manifests and nothing else, so a
  record left live is a record that lies — the session showed epoch 0 while
  the file said 3 and the next run numbered itself from that. `run_select`
  repairs a record that did not travel. `floor_transform.npz` /
  `floor_level.json` are artifacts too, for the same reason.
- `distribute` extrapolates past the LAST KNOT along the screw measured FROM
  THE START (E(d) = ε·d, the model's own sentence), not along the last
  segment: past the last closure there is nothing to measure and the rate with
  the longest lever arm behind it is the one to continue with. Two knots 3.7 m
  apart whose 15.8 and 15.2 cm closures point ~35° apart moved the walk's end
  33 cm for a 15 cm closure. With ONE knot — the common case — the two
  readings are identical.
- DECLARED LIMIT, measured by the §10.11 envelope: the drift-rate model
  represents error that ACCUMULATES with the distance walked. A step kink
  injected into one chunk (what the known-answer experiment injects) is not
  that shape; the instrument recovers part of a small one, fails on a large
  one, and the §9 gates say so instead of hiding it.

## ⭐ THE DRIFT IS DEPTH, NOT POSE — USER DECISION 2026-09-19 (pccr)
**The duplicates are separated ALONG THE LINE OF SIGHT, not sideways.** Five of
pccr's six surviving closures are 97-99 % RADIAL (`desk#203`: 67.2 cm of 67.8).
A translation is the same everywhere; a DEPTH error grows with distance — which
is why the desk closed at 0.7 cm at 3.6 m while the tile lines 8 m away stayed
18.1 cm off, and why no two objects ever agreed on a vector: each sits at a
different BEARING, so ONE depth error becomes a different world vector for each.

Two independent instruments agree on the size: the SAM3 silhouettes
(`k(d) = 1 + eps*d` fits the closures with ONE parameter better than the
per-keyframe TRANSLATION with three — rms 16.7 vs 22.8 cm, and beats it across
the whole regularisation sweep) and the DA3 anchors of `scale_diagnostics.json`,
which never see a silhouette (per-frame scale median 0.8784 first half → 1.0015
second, ratio **1.1401**, bootstrap 90 % CI [1.008, 1.256], Spearman rho +0.459
p 0.0038 over 38 frames). `scale_align` ran in `global_median` and applied ONE
factor (0.9845): it corrects the average and leaves the drift. The session had
already written the evidence and nobody read it — `mad_rel` 10.3 %,
`scale_confidence` 0.486, `residual_vs_depth` +21 % at 0.59 m to −12 % at 2.13 m.

RULED OUT BY MEASUREMENT, do not re-try blind: global rigid rotation (residual
27 cm, axis swings 67° on leave-one-out), per-object rotation (peak flat over
±6°, |t| moves ≤1.2 cm), uniform scale of a copy (the three OBB axis ratios were
1.183 / 0.849 / 1.138 — not one number), and camera rotation about its own
centre, which is impossible by construction: it moves a point PERPENDICULAR to
its ray and these closures lie ALONG it.

Wired the same day (USER: *"arreglemos la escala en origen"* → B first):
- `visit_drift.scale_rows()` reads every closure as the depth ratio it measures
  (`k_b = 1 + (t·u)/D_b`, `s_ab = 1/k_b`) and `visit_drift_run` publishes them as
  `scale_loop_rows.json`, STAMPED with the epoch measured on — this correction
  changes the very depths the rows are made of, so `scale_epoch` re-measures
  when the stamp is stale instead of compounding a correction already applied.
  NOTHING IS VETOED: the TANGENTIAL part becomes the row's residual, so a
  sideways closure widens its own error bar. pccr's one false identity,
  `glass_door#121` (a TRIPLICATE — three parallel doors in one masklet), is 91 %
  tangential and priced itself out at σ 0.267 vs the desk's 0.054.
- `certify/scale_stage` had ZERO loop rows on pccr while `visit_drift` measured
  five good ones every pass and spent them all on a translation solver. It now
  reads them, plus `da3_trend_rows()` — the SHAPE of the DA3 drift as RELATIVE
  rows between consecutive chunks, never absolute pins (NOT the 2026-09-13 move:
  that fed each chunk's absolute median, ±8-15 % monocular noise; a relative row
  carries no opinion about the metre, only about how it is DISTRIBUTED, and
  already-applied corrections are subtracted via the lock-relative agreements so
  the row goes to zero when the drift is gone). σ is measured from the session's
  own `anchors.mad_rel` over √n per chunk.
- An anchor row of a chunk that has NOT moved since the lock is 1.0 BY
  CONSTRUCTION — it restates the gauge, it does not measure whether the lock was
  right. With no other evidence it is exactly the row to keep (2026-09-13). With
  loop rows it is not harmless: seven such rows at σ 0.03 plus six seam rows at
  0.02 outvoted the one loop row 13 to 1 and turned a measured +17.2 % into
  +2.3 %. They now STAND DOWN when something else can speak, and the report says
  which and why (`anchor_rows_stood_down`). WHAT HOLDS THE GAUGE then: anchors
  and absolute rows are the only ABSOLUTE rows; seams and loops are relative, so
  `solve_scale_graph`'s lstsq returns the MINIMUM-NORM solution — geometric mean
  of the factors = 1. That is correct here and must not be "fixed": the lock
  already set the session's size from all 38 anchors at once; what stands down is
  each chunk's individual pin. The drift is redistributed, the total size is not.
- `visit_drift_run.scale_epoch()` applies it: `depth × r_k about each keyframe's
  OWN camera` + the translation that keeps the walk continuous (the existing
  `scale_transforms` + `warp_subset`, unchanged). The CHUNK is the unit because
  each chunk carries its own gauge; a per-FRAME depth change would break the
  multi-view consistency the reconstruction still has INSIDE a chunk.
  pccr epoch 1: r 0.9586→1.0432, cameras move ≤31.3 cm, the cloud ≤66.9 cm —
  the order of the duplicate separation, reached WITHOUT measuring a single
  translation.
- **DECLARED LIMIT**: the DA3 anchors show a CONTINUOUS drift and seven chunks
  can only spell a STAIRCASE. With `sigma_seam_log` 0.02 over six seams the
  model tops out near 12 %; pccr needs ~14 %. Do not "fix" this by retuning the
  seam σ — the overlap frames really do measure relative scale to 0.1-1 %, and
  the cap is structural. The fix is A: `scale_align` learning a mode along the
  WALK (the axis its 2026-08-11 A/B never tested — it tested DEPTH, `s·z+b` and
  `a0·z+a1·z²`, and correctly found no structure there).

## ⭐ ARBITRARY NUMBERS LEDGER — USER 2026-09-14 ("en algún momento nos va a
## joder seguro")
Every value below GATES a decision — it accepts, rejects, stops or caps
something. The list exists because most of them were INVENTED when the code
around them was written, and an invented number that decides is a bug waiting
for the scene that exposes it. Two already did, both on 2026-09-14: the
known-answer tolerance (5 cm, see below) turned a declaration into a verdict
against §10.10, and `mask_filter.visit_gap_kf: 20` asserted a continuity
nothing observed. **When one of these misbehaves, the fix is to replace it
with a measurement, not to retune it.**

- **DICTATED BY THE USER — his criteria, not thresholds someone chose**
  (`correction.visit_drift`, 2026-09-18, written down in the session's
  `output/reproject/CRITERIOS.md`): `min_points`, `min_walk_m` (two
  visits under a metre of WALK apart are one pass with an occlusion in the
  middle), and `min_visit_share`, RAISED 0.01 -> **0.25** on 2026-09-19
  ("podemos subirlo al 25%") after the orthogonal views of pccr's six
  survivors: at 1 % four entered on a visit worth 6-10 % of the object —
  `black_office_chair#199`'s was a patch of FLOOR — and the silhouette peak
  falls with the share (0.86/0.76 above a third; 0.72/0.51/0.34 below).
  (`min_comparability` and `fallback_best_n` were REMOVED with the old step 3;
  they configure nothing today.)
- **DERIVED — leave alone, the evidence is in the YAML comment**
  - `segmentation.dedupe_overlap: 0.5` — from all 2,926 instance pairs of pccr
    (max mutual overlap in the whole session is 61 %; at 0.5 the mutual test
    merges 5 pairs, the old containment test 54).
  - `segmentation.mask_filter.visit_gap_kf: 1` — the USER's definition, not a
    threshold: a visit ends at ANY discontinuity.
  - `correction.visit_drift.silhouette_cell_m: 0.01` + `silhouette_close_px: 5`
    + `silhouette_blur_px: 2.0` — MEASURED on pccr: at 5 mm and unfilled the
    projection of a cloud is nearly empty and the correlation follows the
    noise (peaks 0.013–0.032, the two views measuring one component
    disagreeing by 24 and 30 cm); closed and blurred at 1 cm, peaks 0.70–0.91
    and agreement within 5 cm.
  - `correction.visit_drift.default_repeatability_m: 0.0477` — a FALLBACK, not
    a decision: the session measures its own repeatability
    (`certify.repeatability.session_repeatability`: uncertainty.json → elastic
    seam residual → intra-chunk agreement) and that always wins. The value is
    what pccr measured, kept for a session that wrote no evidence at all.
- **INVENTED AND IT DECIDES — the ones to replace first**
  - `correction_graph.graph.drift_min_gain: 0.2` — the only one that REJECTS a
    correction by a made-up threshold. The replacement already exists: the
    It is now unreachable from the session's correction path (the greedy
    loop that fell through to it was DELETED 2026-09-18 and the keyframe graph
    no longer applies); it still gates `correction_graph` when that module is
    called directly.
  - `segmentation.mask_filter.min_inside_frac: 0.5` (a point is noise below
    this share of frames), `min_votes: 2` (frames needed to judge at all),
    `max_drop_frac: 0.5` (refuses the whole filter).
  - `loops.reprojection.min_self_recall / min_cross_recall: 0.4`,
    `min_agreeing_frac: 0.6` — decide `same_object | distinct | unusable`.
  - `loops.duplicate_min_sep_m: 0.20` — what counts as a duplicate at all.
  - `loops.spatial.identity_reject_factor: 3.0` — separation above
    factor × δ(L) → split instead of drift.
- **INVENTED BUT ONLY WARNS (gates.mode advisory everywhere — they are
  MEASURED and declared, the correction is applied)**
  - `graph.min_loop_gain: 0.5`, `authority.pose_graph_max_m: 1.0` /
    `_max_deg: 5.0` / `saturation_warn: 0.8`, the drift budget
    (`spatial.drift_floor_m: 0.3`, `drift_rate_m_per_m: 0.013`), and the five
    §9 iteration gates (`certify.gates.*`). None of them clamps anything —
    verified 2026-09-14. The drift budget line in the log reads
    "144 cm > 75 cm": the closure MEASURED vs what drift over that walk would
    predict, applied anyway.
- **INVENTED, BOUNDS NOT DECISIONS** — `drift_prior_rot_deg: 30` /
  `drift_prior_trans_m: 5` (deliberately generous: pccr demands ~3 m and the
  blow-up they stop measured 458°), `visit_drift.max_epochs: 12` and
  `visit_drift.search_margin_m: 1.20`, `certify.max_iters: 3`, `scale.max_correction_log: 0.2`,
  `visit_loops.sigma_floor_m: 0.01`, `outlier_mad_k: 3.0`,
  `outlier_max_sigma_factor: 10.0`, `outlier_overlap_frac: 0.6`.
- **REMOVED, do not bring back** — `certify.known_answer.tol_t_m/tol_deg/
  tol_scale` (the 5 cm and friends: §10.10 asks to REPORT the recovery error,
  §10.11 says "grado industrial no es 'no falla': es saber cuándo está fuera
  de especificación y decirlo"; the envelope now stops where the §9 gates
  fail, as the spec names). `correction_graph.graph.min_gain` of the greedy
  loop (USER rejected 5 % as arbitrary; the fixed sample makes any threshold
  unnecessary and a test fails if it returns). The whole `certify.greedy:`
  block with the module it configured (`reconstruction/certify/iterate.py`) —
  deleted 2026-09-18, replaced by `correction.visit_drift`.

## ⭐ ONE COMMAND — USER DECISION 2026-09-13/14 ("Reconstruir" delivers the
## corrected cloud; nothing manual, nothing OFF left in the code)
- Pipeline: RECONSTRUCTION → VLM → SAM3 → CLOUDCOMPY → CERTIFY
  (`pipeline.auto_segment: true`, `workers/certify_worker.py`, resume probe +
  cascade in `pipeline_manager`). The certification applies its epochs
  (transactional, own octree) and the viewer reloads the corrected cloud when
  the acta shows a new epoch; the kit opens by itself with pending epochs —
  Approve/Undo is the only human verdict. No "Run certification" button, no
  `/api/certify/run`.
- EVERY loop closure is MEASURED inside the pipeline and none is discarded:
  SALAD candidates (fork `LoopModel` searches top-k among NON-local keyframes;
  thresholds from `loops.salad`), exact bridges + in-run keyframe graph
  (fork), SAM3 instance copies (`instance_edges`) + geometric revisits
  (`visit_edges`) in the post-hoc graph. "nunca debe descartarse un duplicado
  detectado por SAM3": any VLM class except dynamic proposes; a non-structural
  proposer only inflates σ (`loops.semantic.nonstructural_sigma_factor`).
  SINCE 2026-09-18 the IN-RUN closures still move the reconstruction (fork);
  the POST-HOC keyframe graph is measured and reported, not applied — the
  session's correction is the visit-drift loop (see the CORRECTION ALGORITHM
  block above), and the graph's loop residual is how the acta says what it
  left behind.
- GATES ARE ADVISORY everywhere (`correction.gates.mode`,
  `correction_graph.graph.gate_mode`, `certify.gates.mode` = advisory): drift
  budget, loop gain, held-out pairs, authority, the §9 iteration gates are
  MEASURED and declared (acta `gate_warnings`, kit ⚠, attention list), the
  correction is applied. `veto` exists for evaluation only. No gravity prior,
  no floor datum in the pose graphs (both bent the chain on pccr).
- Post-hoc scale rows are RELATIVE to the metric lock (anchor agreement now /
  at lock): an untouched session is identity, an injected/accumulated scale
  error is recovered. Raw DA3 medians are never re-solved post-hoc.
- User rules: I never launch the backend, vLLM, a pipeline, NOR TESTS — I
  write the code and tell him what to restart/test; he runs everything.

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
