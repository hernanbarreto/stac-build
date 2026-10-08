# Implementation spec, wave 2 — plan points 64-166 (docs/plan_determinismo.md), 2026-10-07
Details per point (mechanism, evidence, fix_idea, skeptic corrections): audit2_items.json (same dir), field n = plan number.
The plan line (and its DECIDIDO, when present) is the requirement. Wave 1 (points 1-63) is already implemented: REUSE its
shared mechanisms, never re-implement them — server/repro.py (deterministic_torch, require_exclusive_gpu, card_identity,
environment_record, stamp/check_stamp, stable_id, lossless pose writer) and decide_change in
vendor/VGGT-Long/loop_utils/metric_lock.py (the user's rule). Read them first (and `git diff` to see wave 1's changes).

## Packages (disjoint files; a point listed in two packages: each does only its file)
P1 INTAKE — server/intake/* (content, focal, parallax, quality, run, vram, walk), server/precision/tracks.py,
   server/reconstruction/scale_align.py, vendor/VGGT-Long/LoopModels/calibration.py.
   Points 64-66, 68-72 (72 refuted: no action), 74, 75, 78, 79, and 84 (content.py side).
P2 VLM + SAM3 — server/semantic/serve.py, server/semantic/service.py, server/segmentation/autoprompt/*,
   server/segmentation/sam3_wrapper.py, vendor/sam31/**, server/segmentation/object_captioner.py, server/phase5_qa/api.py,
   server/reconstruction/loops/semantic_classes.py, server/workers/{cloudcompy_worker,sam3_worker,vlm_worker}.py,
   scripts/serve_semantic.sh. Points 80-82, 85-92, 94-98, 141, 154-156, 159, 163-165 (and their in-package share of others).
P3 CLOUD + SEGMENTATION GEOMETRY — server/segmentation/{pipeline,fuse_parent,republish,session_io,mask_space,mask_filter}.py,
   server/reconstruction/gpu_cloud_clean.py, vendor/PotreeConverter/** (C++: rebuild the binary on CPU, capped, and verify
   two conversions of the same PLY are byte-identical), server/alignment_manager.py, server/reconstruction/surface_fit/hole_audit.py,
   server/phase_r/instance_store.py. Points 83, 93, 99-105, 107-110, 113-125, 143, 145.
P4 CORRECTIONS + CERTIFICATION — server/correction/*, server/precision/{chunk_check,cloud_metrics,gauge}.py,
   server/reconstruction/certify/{run,scale_stage,api}.py, server/workers/certify_worker.py.
   Points 106, 126-140, 142, 144, 146-148, 150, 160, 166.
P5 ORCHESTRATION — server/main.py, server/pipeline_manager.py, server/run_mapanything.sh, scripts/start.sh, ui/src/App.tsx.
   Points 67, 73, 76, 77, 111, 112, 149, 151-153, 157, 158, 161, 162.

## Rules (same as wave 1)
- Code/comments in English; no invented numbers; determinism is not a config switch; decisions = the plan's DECIDIDO.
- CPU-only capped tests from /workspace/stac-build/server:
  (ulimit -v 30000000; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 900 taskset -c <your cores>
   nice -n 15 /workspace/miniforge3/envs/da3/bin/python -m pytest -x -q <tests>). UI: type-check only (npx tsc --noEmit).
- Never run the pipeline, the backend, vLLM, SAM3 or GPU work; never touch server/projects/; do not commit.
- Only edit your package's files. Report per point: implemented / no_action / not_implemented (+ exact reason), files,
  tests, result.

## ORDER OF EXECUTION (user 2026-10-07): the user tests RECONSTRUCTIONS while wave 2 runs
The user launches reconstructions (intake → Omega → F0-F6 → cloud) on the backend as soon as wave 1 is closed. Wave 2
therefore runs in TWO sub-waves so nothing it edits is a file the running reconstruction imports:
- SUB-WAVE 2a (runs while the user tests): packages P2 (VLM+SAM3), P3 (cloud+segmentation geometry), P4 (corrections +
  certification), P5 (orchestration) — EXCLUDING any point whose file is in the reconstruction path:
  server/intake/*, server/precision/*, server/reconstruction/* (except certify/ and surface_fit/ and loops/semantic_classes.py),
  server/workers/map_worker.py, server/extract_da3_depth.py, vendor/VGGT-Long/**. Points moved OUT of 2a for that reason:
  68 (tracks.py), 84 (intake/content.py), 102 (gpu_cloud_clean.py — check: is it in the reconstruction's cloud stage? if the
  cloud stage after F6 imports it, it is reconstruction → sub-wave 2b), 138, 140, 146 (precision/*), 72 (refuted anyway).
  pipeline_manager.py / main.py edits (P5) are allowed but must be import-safe: the running backend is restarted by the user
  only when told; keep every edit backward compatible so a restart mid-test does not break the queue.
- SUB-WAVE 2b (after the user's reconstruction tests, or in a worktree): P1 INTAKE (64-66, 69-71, 74, 75, 78, 79) + 68, 84,
  102 (if reconstruction), 138, 140, 146.

## REVISED (user 2026-10-07): the RECONSTRUCTION closes in wave 1, entirely
Wave 1 now = points 1-63 (packages A-D) + package E (intake: 64, 65, 66, 69, 70, 71, 74, 75, 78, 79, 84) + the
precision/reconstruction points of wave 2: 68 (tracks.py), 102 (gpu_cloud_clean.py), 138 (cloud_metrics.py), 140
(chunk_check.py), 146 (gauge.py) — assigned to the package that owns each file (C: 68, 146; D: 138, 140; B: 102) as a
follow-up once their first pass reports. Only then does the user test reconstructions. Wave 2 (everything after the
cloud: VLM, SAM3, cloud/segmentation geometry, corrections, certification, orchestration) runs while he tests and never
edits a reconstruction file.
