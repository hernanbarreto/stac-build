# Final reconstruction pipeline — STATE OF 2026-10-07 (read this first in a new session)

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

## ⭐ DETERMINISM — USER 2026-10-07 ("STAC Build es una herramienta de ingeniería, debe dar siempre lo mismo")
The whole pipeline must be BIT-FOR-BIT deterministic: the same input gives the same bytes, whatever ran before,
whatever else is on the machine. Trigger: two pccr 2026-08-31 runs with identical input diverged at the in-run pose
graph (a code change between them), F5 then took rung R2 (a false lens, p2 −0.007) on a 0.0005 px "improvement"
over 128k tracks, F6 resampled every Omega map through it: floor undulation 14.6 → 45.1 cm.
- THE PLAN: `docs/plan_determinismo.md` — 166 points (1-63 Omega→F6, 64-166 intake / VLM+SAM3 / cloud+segmentation /
  corrections+certification / orchestration), each with its fix or the user's decision (DECIDIDO). Wave 1 = the whole
  RECONSTRUCTION (1-79 + 102, 138, 140, 146) implemented 2026-10-07; wave 2 (after the cloud) follows while the user
  tests reconstructions.
- THE USER'S RULE (point 1, delegated to all judges): a change is APPLIED only if (a) significant — 95 % CI of the
  paired change entirely on the improving side (cluster bootstrap when judges share a keyframe), (b) ≥ 5 judges
  (`loop_judge.min_judge_closures(0.95)`), (c) the median improvement ≥ `correction_graph.graph.improvement_error_factor`
  (2.0, USER) × the MEASURED error. One function: `loop_utils/metric_lock.decide_change`. Users: the in-run pose graph
  (leave-one-out judges, every closure judges once; < 5 closures → declared before solving, not applied), the depth
  graph, the F2 instrument switch, the F5 rung ladder (error = the solver's own continuation error; R2 warm-started
  from R1; R3 report-only), the F6 per-keyframe bend (c1/c2 enter only ≥ 2× their fit error; c0 verified on the
  keyframe's own held-out rows, else the pooled neighbours' fit), the chunk check verdicts, the zoom exclusion and the
  scale-drift gate. argmins (gauge λ, bend window) take the SMOOTHEST option within 2× the measured error of the best.
- SHARED MECHANISMS: `server/repro.py` — `deterministic_torch` / `enable_deterministic_torch` (STRICT deterministic
  algorithms, TF32 off, cudnn benchmark off, seeds, cuBLAS workspace), `require_exclusive_gpu` (any co-tenant on the
  card FAILS the GPU step: no resolution step-down, no window halving — ever), `card_identity` (torch, no 'unknown'),
  `environment_record` (card, driver, torch/cuDNN, CPU, BLAS core, libs, git of repo+forks), `stamp`/`check_stamp`
  (inputs + code + config sha256 on every reused file), `stable_id` (no uuid in artifacts), exact float64 writers
  for every pose/intrinsics text file. Per-card constants COMMITTED in `server/card_table.json` (DA3 window sizing,
  focal-probe layout, Omega footprint factor) — nothing learned from OOMs at run time (`weights/omega_footprint.json`
  gone). DA3 weights pinned by revision + sha256 (`server/da3_weights.py`), ONE HF cache (`/workspace/hf_cache`).
  OPENBLAS_CORETYPE=HASWELL pinned in the chain's step env. PotreeConverter pinned to 1 thread + total-order sorts
  (rebuilt; measured byte-identical). Timings/clock → `*.timing.json`, never in compared artifacts.
- RESUME: every chain step stamps what it consumed (`chain_state.json` STATE_VERSION 2: old files are ignored → re-run
  from F0); Omega has a stamped completion marker (`maplong_run/omega_complete.json`, `fork_stamp.json`); every reused
  file (windows, walk, covis, loop closures, scale rows, calibration, masks, gauge.json) is taken only on a matching
  stamp, otherwise recomputed or refused with the reason.
- INTAKE (points 64-66, 69-71, 74, 75, 78, 79, 84 — package E, 2026-10-07): the I0/I1/I2 marker (`intake/intake_state.json`
  v2) keys every step on a `repro.stamp` — every frame's bytes under `frames/<name>`, the intake + DA3-extractor code
  (`intake.stamps.INTAKE_CODE_FILES`), the step's parameters, K + the focal probe's stamp, the CPU environment
  (`intake.stamps.cpu_environment_record`: numpy/scipy/OpenCV/Pillow versions, CPU, BLAS core, OpenCV dispatch + IPP,
  BOTH JPEG decoders with their libjpeg-turbo builds, numpy's FFT dtype) — never a version bumped by hand; the log
  names what differs. No time, epoch or absolute path in any intake product (epochs = 0 by construction: the intake
  feeds the reconstruction's epoch 0; the session's epochs at run time, times and absolute paths go to
  `intake/intake_state.timing.json`). I0's FFT in explicit float64 (bit-identical to the numpy-1.26 float32 path);
  numpy / scipy / opencv-python / pillow PINNED in requirements.txt + environment.yml to env da3's versions. The job's
  configuration is FROZEN once by the pipeline manager after the replace wipe (`output/run_config.yaml` +
  `run_config.sha256`, `intake.run_config`); the intake refuses an intake configuration that is not the frozen one and
  records the sha (not a key: a change elsewhere in config.yaml re-runs nothing); the CLIs freeze their own. Every I1
  window records its decision margins in `selected_frames.json` (`windows[].margins`: closing p − hi, band-edge
  distance, top-2 sharp_rank gap, worst fb margin, tracks margin, TRF nfev vs the bound). I2 (off): the disabled branch
  deletes leftover `intake/exclusion_masks/*.png` and `content_tags.json` carries the stamp of exactly the valid masks
  (root `stamp` + `exclusion_masks.stamp`; readers: `precision.tracks.exclusion_mask_paths` / `intake.content.
  valid_exclusion_masks`); a VLM answer that does not parse FAILS the stage (`VLMParseError`), never 'nothing'. The
  focal probe is reused on its stamp only (frames' bytes, committed layout, card, weights, dtype, versions —
  `intake.focal_probe`); the SALAD revisit bars carry their bootstrap error and pairs within 2× it are in neither
  class; the non-local band is min_walk_m of the FROZEN walk with its margin recorded.
- FRAMES SEALED + CAPTURE IN inputs/ (points 67, 73, 76, 77 — 2026-10-07): every frame writer (video upload,
  Stray `rgb.mp4` — `ingestors/stray_scanner.py`, `convert_stray_to_da3.py` —, the WebXR `/ws/scan` socket) fills
  `<scan>/frames.extracting/` through `intake.frames_manifest.FrameSealer` (duplicate frame numbers refused) and
  renames it to `frames/` only when complete, `frames/manifest.json` written LAST: video name/size/sha256, decoder
  (OpenCV version, FFMPEG requested explicitly, the Video I/O build lines, `CAP_PROP_ORIENTATION_AUTO` SET to 0 —
  measured to be OpenCV 4.11.0's default: bufferStop's / fosa_pan's −90° videos have unrotated frames on disk —
  and the container's orientation), encoder (`cv2.imencode` with every JPEG option explicit, the same bytes as the
  old `imwrite` q95 — tested), reported vs decoded count, every frame's sha256; deterministic bytes, no clock. An
  upload into a scan WITH frames → 409 naming the count (the UI has no replace-video action). While a writer works
  (in-process claim) or the temp dir exists (a restart mid-extraction) the pipeline fails the job with the reason,
  the wipe and the intake refuse; the next upload removes the temp dir. The legacy `/ws/camera` stream (no client)
  is refused. The intake checks the frames against the manifest (names + count, then the sha256 of its OWN I0 stamp)
  and fails naming the first difference; a manifest-less `frames/` (all five sessions today) is ADOPTED once
  (origin `adopted`, decoder/encoder null, DECLARED in the log). `quality.list_frames` refuses two files with one
  frame number. CAPTURE: `inputs/stray/`, `inputs/vio_trajectory.*` / `inputs/vio/`, `inputs/webxr/`
  (`ingestors/capture_inputs.py`); every Stray / VIO reader reads `inputs/` first, then the scan's own legacy
  places (never a sibling); the replace wipe keeps `frames/manifest.json` + `inputs/` and MOVES legacy capture data
  into `inputs/` (two different copies of one file refuse the wipe before anything is touched). `camera.json`
  `report.k_source` = source + file + sha256; `scale_diagnostics.json` `scale_source_file` / `vio_sha256`;
  `gauge.json` (+ its v2 block) `scale_source` / `scale_inputs`. No session has VIO/Stray: a no-op for them but
  the adoption (and F0+ re-runs once: `frames/` now holds the manifest the chain's `frames` stamp hashes).
- DECLARED for the first run after this: F5 solves each rung twice (continuation = solver error); PointDiT and DA3
  run STRICT — an op without a deterministic CUDA kernel RAISES (fail, never degrade); Potree on one thread ≈ 0.55 M
  pts/s; F2's algebra on Haswell kernels (last-bit changes vs earlier runs); F6 numbers change by design (points 48/49).
- VERIFICATION: run the same session twice (replace ON) and compare the sha256 of every product; any difference is a
  bug of this plan, not noise.
- 2026-10-07 (evening, USER — after the first deterministic pccr run showed the turn at frames 558–723 torn by 39 cm and
  the table doubled): (1) I1 also makes a keyframe by ROTATION — the quantum is a QUARTER of the co-visibility bound,
  (1 − τ)·FOV_h/4 (pccr 10.8°: co-visible with the next three keyframes, ℓ ≥ 4); measured on pccr, half the bound
  (21.6°) still left 19–27° jumps ("huecos") and the session's own 7° stacked 30 keyframes 2–3 cm apart ("solapados");
  pccr 289 → 295 keyframes, the turn 16 → 21, largest frame gap 70 → 51. (2) THE CHUNK PLAN IS MEASURED ON THE
  PARALLAX KEYFRAMES ONLY (covis.json plan_index; COVIS_VERSION 4): counted, the rotation keyframes made turns cheap
  (D_total 34.7 → 31.9, 5 chunks → 4 longer ones spanning the turn, Omega drifted −54 cm inside them — floor undulation
  13.5 → 32.7 cm). They fill the turns inside the chunks; every range maps to full keyframe indices. (3) Ownership of a
  shared block = split at its midpoint (the frame at the midpoint to the earlier chunk — bit-identical to the old
  nearest-centre rule on uniform layouts); the centre rule gave the whole block to the smaller chunk (pccr: 63/63 vs
  50/139). (4) TRIED AND REMOVED the same evening: a top-view (silhouette correlation) witness corroborating loop closures
  — on real bridge windows it read false shifts (67 cm on a local closure, 1.7–5.1 m start↔end) and the graph applied
  metres; ownership by co-visibility company (untested on its own); F6 landmark-precision weights (measured: do not move
  s_k at the turn) and nearest-side edge pixels (a recipe change). The start↔end closure (~1 m, bridge 284↔8) stays
  measured, not applied: the certification is the validated corrector of that loop.
- 2026-10-07 (night): the ZOOM rule's error was each chunk's own per-frame focal scatter (2–5 px); Omega gives the
  SAME frames focals that differ by up to 21.8 px between two chunks' inferences (pccr seams: +1.4, +11.8, −1.9,
  +21.8 px), so its per-inference noise read as a zoom on 4 of 5 chunks, their DA3 anchors were dropped and the scale
  verification failed by 18.6 %. The error is now max(own scatter, Omega's focal error between inferences measured on
  the shared frames = RMS of the seams' median offsets / √2, pccr 8.8 px) — metric_lock.inference_focal_error.
- 2026-10-08 — WAVE 1 VALIDATED and COMMITTED (main 1c466f4, fork 67f30e9) on pccr 2026-08-31 (floor undulation
  10.0 cm, thickness 4.8 cm, worst in-chunk drift 8 cm; USER: "pccr se ve bien, con menos errores que hoy a la mañana")
  and zaragoza 2026-06-03 (ONE pass, D_total 5.01; floor 27.8 cm, as before). Hashes of the run and a full copy of the
  session under /workspace/stac-keep/ (docs/determinism_work/verification_runs.md). The bit-for-bit two-run comparison
  of the reconstruction was NOT run (USER: wave 2 first).
- 2026-10-08 — F6 ON THE CARD, STRICT (USER: "todos los pasos que son CPU y pueden ser GPU estricto deben ser gpu
  estricto … la premisa es que cada paso sea lo más veloz posible, manteniendo la calidad y el determinismo"; bit
  identity with the numpy recipe NOT required, run-to-run identity IS). `precision/f6_torch.py`: every map operation of
  mono_detail (per-tile affine fit in closed form, blend, Gaussian lowpass as matrix products with explicit 'nearest'
  boundaries, band, sides) and of the edge-keeping vote (splat by stable sort — no scatter, no atomics —, dense
  projections, sort-based medians, snap, repair) on torch tensors on ONE device: the stage entries (`depth_on_f5.main`,
  `corrected_cloud.main`, `mono_ab.main`) call `f6_torch.require_cuda(seed)` (no card → fail; strict deterministic mode
  for the whole process; a CUDA run with the mode off FAILS), the synthetic tests run the same code on the CPU
  (`use_device`). PointDiT runs a keyframe's tiles as ONE batch (`PointDiTRunner.depth_batch`, `mono_detail.tile_batch`
  0 = all; a card that cannot hold it fails, never halves). `depth_on_f5.json` carries `maps_numerics` (device, card,
  torch flags). VERIFIED 2026-10-08 on two fresh copies of pccr (precision.mono_ab, variant 'on'): cleaned cloud
  byte-identical (sha f548f2f985b9a566), reports identical but the timings; F6 compute 16–18 min vs 32 in the
  pipeline (PointDiT 411–479 s vs 819 at batch 2; the map phases 26 s vs 186). Still CPU inside F6: the bend
  (landmark IRLS + bootstrap, ~9 min on pccr) — the next port. Expected on zaragoza (15 tiles/keyframe): PointDiT 56
  min → to be measured.
- 2026-10-08 — SEGMENTATION IN ROUNDS (USER DECISION, to implement after wave 2's P2: "si si, es la manera correcta,
  porque sino el tiempo que se ahorra ahora lo pierde después el humano fusionando o diciendo que 5 cosas son lo
  mismo"): round k — the VLM sees every keyframe with everything already segmented GREYED OUT and names only what is
  still visible, with the most specific common name of ONE object (desk, chair — never 'furniture'); the synonym merge
  over the new names; SAM3 runs the new prompts over all keyframes into the same (staged) store; the unsegmented
  remainder per keyframe is recomputed; stop when a round names nothing new (plus a declared bound of rounds). Then the
  VLM describes every segment (ShapeR + chat). Duplicates are prevented by construction, completeness is measured; the
  after-the-fact dedupes stay as safety nets. Cost ~2–3 h on pccr vs ~2 h, plus one vLLM start per round (never
  vLLM and SAM3 together on the card).
- 2026-10-08 — THE FLOOR IS GEOMETRIC, AND THERE MAY BE SEVERAL (USER): "piso" = the horizontal SUPPORT surface the
  camera walks over — the dominant horizontal plane under the cameras, nearest to y=0 or parallel to it — whatever it
  is called (floor, platform, andén, walkway, slab, deck); labels are only a hint (plan point 128). A session can hold
  several floors at different heights, each a segmented object; the floor of a keyframe/chunk is the surface under the
  camera at that moment and a real change of level (stepping down from the platform) is PRESERVED as a step, never
  corrected as drift; a second level with significant support by the user's rule is a real floor, not noise. Applies
  to the certification's floor alignment, the re-level after each epoch, the chunk check and the cloud metrics.
- 2026-10-08 — WAVE 2 (after the cloud), packages P2–P5 of docs/determinism_work/impl_spec2.md, running/landed the same
  day (reports in the session log; summary here when all four are in):
  P2 VLM + SAM3 (landed): each VLM stage launches its OWN vLLM from the frozen config (no prefix cache, one sequence,
  batch invariance, eager, seed, FLASH_ATTN pinned) and stops it at the end (`semantic.service.job_engine`; identity
  verified and written to vlm_analysis.json); weights pinned by sha256 (`semantic.serve.PINNED_WEIGHTS`); engine LEASE
  (`logs/semantic_engine_lease.json`): chat/intel wait while a job holds it; `vlm_analysis.json` stamped over every
  input and reused WHOLE only on an identical stamp; per-call records (image sha1, finish_reason, salvage); captions
  stamped, never carried by id, all-or-nothing; SAM3 strict deterministic (TF32 off, repro flags after the build), NO
  object cap (vendor/sam31 patched: no 320 GB pin), any error/OOM fails the stage; InternVL3 fallback gone; the cloud
  stage never projects masks (a rebuilt cloud deletes the previous projection products).
  P3 cloud + segmentation geometry (landed): the raw SAM3 store is IMMUTABLE and sealed (staged, swapped atomically,
  `reconstruction_id` + stamp), the fusion lives in `fusion_map.json` tied to the raw store's sha (readers:
  `fuse_parent.fused_parent` / `FusedMasks`), `seg_masks.npz` canonical (object, frame) order; mask identity, space
  dedupe, label-fragment contiguity, co-visibility splits and OBB core/yaw all by the user's rule with margins
  recorded (`decisions` in the result); the projection is PURE (no previous result; error or empty → fails and deletes
  the old one), stamped over every input, reused only on an identical stamp; octree built only from the cloud the masks
  were projected on, `potree_stamp.json`; PotreeConverter canonical octree layout (byte-identical conversions, no
  log.txt; POTREE_NUM_THREADS=1 still required); leveling seeded and cache-keyed by the cloud's sha; 16-bit instance ids
  (`instance_ids.npy`, LAS extra dim `instance`; the viewer still keys on the uint8 class — pending); caches keyed by
  content; no clock in any compared artifact (`segmentation_timing.json`). Mask audit projects through the cloud
  camera's LENS (`session_io.CameraSource.pixels`) — a lens-refined session (pccr R2) is no longer refused.
  DECLARED: octrees change bytes and rebuild once; sessions projected before 2026-10-08 re-project once; legacy
  sessions without camera.json cannot be projected; `segmentation.obb_orientation.min_plane_frac` removed.

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
