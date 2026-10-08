# Implementation spec — docs/plan_determinismo.md (63 points), 2026-10-07
Plan point N ↔ audit id (details: mechanism/evidence/skeptic in audit_items.json, same dir):
1 omega-01 | 2 omega-02 | 3 omega-03 | 4 omega-04 | 5 omega-05 | 6 omega-06 | 7 omega-07 (refuted, no action) | 8 omega-08
9 omega-09 | 10 omega-10 | 11 omega-11 | 12 omega-12 | 13 omega-13 | 14 omega-14 | 15 omega-15 | 16 omega-16 | 17 omega-17
18 omega-18 | 19 omega-19 | 20 omega-20 | 21 omega-21 | 22 omega-22 | 23 M-05 | 24 M-06 | 25 f0f4-01 | 26 f0f4-02
27 f0f4-03 | 28 f0f4-04 | 29 f0f4-05 | 30 f0f4-06 | 31 f0f4-07 | 32 f0f4-08 | 33 f0f4-09 | 34 f0f4-10 | 35 f0f4-11
36 f0f4-12 | 37 f0f4-13 | 38 f0f4-14 | 39 f0f4-15 | 40 f0f4-16 | 41 M-01 | 42 M-02 | 43 M-03 | 44 M-04 | 45 X-01
46 f5f6-01 | 47 f5f6-02 (refuted, no own action) | 48 f5f6-03 | 49 f5f6-04 | 50 f5f6-05 | 51 f5f6-06 | 52 f5f6-07 (kept)
53 f5f6-08 (kept, record bars) | 54 f5f6-09 | 55 f5f6-10 | 56 f5f6-11 | 57 f5f6-12 | 58 f5f6-13 | 59 f5f6-14 | 60 f5f6-15
61 M-07 | 62 M-08 | 63 M-09
The text of each point and its DECIDIDO line in /workspace/stac-build/docs/plan_determinismo.md is the requirement.

## Shared mechanisms (built ONCE by the foundation step, reused by everyone)
- THE USER'S RULE (point 1), one function in vendor/VGGT-Long/loop_utils/metric_lock.py, e.g.
  decide_change(before, after, *, error_m, confidence, min_judges, clusters=None) -> dict:
  improves only if (a) the bootstrap CI (fixed seed) of the paired change lies entirely on the improving side
  (cluster bootstrap when clusters given), (b) n judges >= min_judges (= min_judge_closures(confidence), 5 at 0.95),
  (c) the median paired improvement >= 2 x error (the factor 2 is the USER's decision 2026-10-07: a named config key
  with that provenance, read where the judges' config lives). Returns verdict + every margin for the logs.
  Users: 1, 2, 18 (scale drift), 19, 30, 46, 48 (c0 check). The old heldout_change stays for reports only if still used.
- server/repro.py (new): deterministic_torch() (use_deterministic_algorithms(True) STRICT, cudnn.benchmark False,
  cudnn.deterministic True, allow_tf32 False for cudnn and matmul, fixed seeds, CUBLAS_WORKSPACE_CONFIG),
  require_exclusive_gpu(log) (no other compute process on the card — nvidia-smi --query-compute-apps or NVML; this
  process excluded; FAIL with the list, never degrade), card_identity() via torch (name, total memory, capability)
  with no 'unknown' sentinel (fail), environment_record() (card, driver, torch, cuDNN, CUDA, numpy/OpenBLAS core,
  CPU model, python libs, git commit of repo and fork), stamp(inputs=[paths], code=[module files], config=dict) ->
  sha256 dict and check_stamp(saved, now) -> list of differences, stable_id(*parts) -> deterministic id (replaces
  uuid ids). Reuse what already exists (tracks.py has a deterministic context; intake has stamps) — unify, do not duplicate.
- Lossless poses (45): every writer of camera_poses.txt / intrinsic.txt / witness poses (server AND fork) writes
  float64 round-trip exact (repr or '%.17g'); readers unchanged. Covered by a test that write->read is bit-exact.

## Assignment (disjoint file sets)
FOUNDATION: the shared mechanisms above + their unit tests.
A  FORK vendor/VGGT-Long (vggt_long.py, loop_utils/*): 1, 2 (leave-one-out: n solves each holding one closure out,
   judged by decide_change with error = max bridge sigma; final solve with all closures; < 5 closures -> declared
   before solving, not applied), 8, 9, 11, 12 (fork runtime: fixed dtype, TF32 off, no CPU fallback — fail), 17, 18,
   19, 20 (sky-mask cache keyed on sha of skyseg.onnx; pinned/vendored model), fork-side writers for 45.
B  SERVER reconstruction + intake + DA3 windows (workers/map_worker.py, reconstruction/chunk_plan.py,
   reconstruction/chunk_covis.py, reconstruction/loops/spatial_gate.py + calibration, reconstruction/scale_align*,
   intake/vram.py, intake/walk.py, reconstruction extract_da3_depth.py / DA3 window code, scripts/start.sh env):
   3, 4 (map_worker + walk: exclusive GPU before DA3/Omega, OOM -> fail, no step-down/halving), 5/25/41 (I3 window
   size from a COMMITTED per-card table of measured per-token constants — A100 values from da3_vram.json measurements
   in the logs: 1.194 MiB/token at 840, 0.903 at 1932, weights 6.4 GiB — formula deterministic; unknown card -> fail
   naming the calibration CLI that writes the table entry), 6, 10, 12 (record environment in chunk_plan.json),
   13, 14 (resolution persisted per session and reused while the plan is unchanged), 15, 16 (margins logged),
   21, 22, 23 (stamped completion marker that cleanup never deletes; map_worker skips the fork only on a matching
   stamp), 24, 26 (with C: F2 regeneration goes through B's walk API with the exclusive-GPU check), 27 (DA3 extractor:
   strict deterministic, TF32 off, environment in windows.json), 32 (pin DA3 revision + weights sha), 42 (record
   each window's reference view; regeneration must reproduce it or fail), 43 (one HF cache for every launcher:
   backend, runner, by-hand), 44 (per-window stamp; any mismatch clears all windows), scale_align/orient writers for 45.
C  PRECISION F0-F4 + chain (precision/camera.py, gauge.py, tracks.py, omega_probe.py, runner.py, poses_epoch.py,
   epoch*.py, provenance.py, correction/session.py write_poses, correction/ledger.py ids, correction/epoch.py):
   26 (f2 marked gpu when it must regenerate; regenerated plan must equal walk.json or fail), 28 (track split by a
   hash of the track's stable key), 29, 30, 31 (per-step stamp of inputs+config+code; resume from the first changed
   step), 33, 34, 35, 36 (stable ids; timings to *.timing.json outside compared artifacts), 37 (covered by the GPU
   check; add the record), 38 (pin OPENBLAS_CORETYPE in step_env, record CPU), 39, 40, 45 (write_poses lossless),
   58 (step_env for f5: OMP/OPENBLAS threads 1 unless a test proves the 8-thread solve bit-identical), 4 (runner:
   require_exclusive_gpu before every gpu step).
D  F5/F6 (precision/refine.py, colmap_ba.py, depth_on_f5.py, mono_detail.py, pointdit_runner.py, chunk_check.py,
   segmentation primitives used by chunk_check, corrected_cloud.py, potree_converter.py, cloud_metrics.py):
   46 (decide_change with keyframe clusters; error from 59), 48, 49, 50, 51 (masks only when stamped for this
   reconstruction), 52 (no change; a test pins it), 53 (record bars + margins), 54 (PointDiT inside
   deterministic_torch, card recorded), 55, 56 (stable ids, no wall clock in compared artifacts), 57 (prove with two
   conversions in a test; pin threads if they differ), 59 (continuation solve from the end state; its held-out change
   = solver error), 60, 61 (R2 warm-started from R1's solution), 62 (R3 report-only), 63.

## Rules for every implementer
- Read CLAUDE.md rules; code/comments in English; no invented numbers (every constant measured, derived or the
  user's, with provenance in a comment); determinism is not a config switch.
- CPU-only tests, capped: (ulimit -v 30000000; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 900
  taskset -c 0-7 nice -n 15 /workspace/miniforge3/envs/da3/bin/python -m pytest -x -q <tests>) from /workspace/stac-build/server.
  The pod dies over 117 GB RAM. Never run the pipeline, the backend, GPU work. Never touch server/projects/.
- Do not commit. Only edit files of your assignment (shared helpers: only FOUNDATION writes them; others import).
- Report per plan number: implemented / not, files+functions changed, tests added, test result.
