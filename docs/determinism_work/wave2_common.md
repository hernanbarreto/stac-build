# Wave 2 — common brief (read fully)
You are implementing ONE work package of the determinism plan of the STAC pipeline (repo /workspace/stac-build). The user
requires the WHOLE pipeline to be BIT-FOR-BIT deterministic: the same input always gives the same bytes, whatever ran
before, whatever else is on the machine. STAC Build is an engineering tool. Wave 1 (the reconstruction: intake → Omega →
F0–F6 → cloud, plan points 1–79 + 102/138/140/146) is DONE and the user is TESTING reconstructions on the backend RIGHT
NOW. Wave 2 (yours) = everything after the cloud. HARD CONSTRAINT: never edit a reconstruction file — server/intake/*,
server/precision/*, server/reconstruction/* (except certify/, surface_fit/ and loops/semantic_classes.py), server/workers/
map_worker.py, server/extract_da3_depth.py, vendor/VGGT-Long/**, server/repro.py, server/card_table.*, server/da3_weights.py.
Edits to server/main.py / server/pipeline_manager.py must be import-safe and backward compatible (the backend may be
restarted mid-test).

READ FIRST: /workspace/stac-build/docs/plan_determinismo.md (the plan; your points' lines and their DECIDIDO are binding),
/tmp/claude-0/-workspace-stac-build/448cd311-7766-4e9d-b6c6-746bda2501e6/scratchpad/audit2_items.json (audit details per
point: field n = plan number; mechanism, evidence, fix_idea, skeptic corrections — use the skeptic's corrections),
docs/pipeline_final.md section "DETERMINISM" (what wave 1 built), and the shared mechanisms you MUST reuse:
- server/repro.py: deterministic_torch / enable_deterministic_torch (strict), require_exclusive_gpu, card_identity,
  environment_record(gpu=...), stamp(inputs, code, config) / check_stamp, stable_id(*parts, n_hex=None), exact writers.
- decide_change(before, after, *, error, error_factor, confidence, min_judges=None, clusters=None) in
  vendor/VGGT-Long/loop_utils/metric_lock.py (import with the fork on sys.path, as precision/refine.py does) = THE USER'S
  RULE: a change is applied only if (a) 95 % CI of the paired change entirely on the improving side, (b) ≥ 5 judges,
  (c) median improvement ≥ error_factor × the measured error. error_factor = correction_graph.graph.improvement_error_factor
  (accessor reconstruction.loops.config.improvement_error_factor(raw_config)). For argmins: the simplest/smoothest option
  within error_factor × the measured error of the best.
- correction.epoch.reconstruction_id(output_dir) (the identity of a reconstruction; stamps of segmentation / masks /
  certification products must carry it — F6 already refuses seg_masks.npz without it, plan point 51).
- The user's own criteria stay unchanged (CLAUDE.md ledger: min_walk_m 1 m, conf_min_norm 0.10, the four mask-filter rules,
  min_points 1000, dedupe 0.5 …): record their margins, never retune them. No invented numbers: every new constant measured,
  derived or the user's, with provenance in a comment. Determinism is not a config switch. Deleted config keys: a leftover
  fails the load (repo convention).

RULES: code/comments in English. Tests: CPU-only, under server/tests, that FAIL on regression; update existing ones. Run
capped from /workspace/stac-build/server: (ulimit -v 30000000; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2 timeout 1500 taskset -c <YOUR CORES> nice -n 15 /workspace/miniforge3/envs/da3/bin/python -m pytest -q
-p no:cacheprovider <tests>). The pod dies above 117 GB RAM — never uncapped. The user's reconstruction is using the GPU
and cores 0-7 may be busy: NEVER run the pipeline, the backend, vLLM, SAM3 or any GPU work; NEVER touch
/workspace/stac-build/server/projects/; do NOT commit. UI: only `npx tsc --noEmit` (no dev server). vendor/sam31 and
vendor/PotreeConverter are separate git repos: edit in place, do not commit. Every point of your package must end
implemented (or no_action where the plan says so); if one truly cannot, say exactly why.
FINAL REPORT (your last message): per point — status, files/functions, tests, exact pytest commands with counts; plus
anything DECLARED the user must know before the next run.
