# Wave 2 — common brief v2 (2026-10-08; read fully, it replaces wave2_common.md)

You are implementing ONE work package of the determinism plan of the STAC pipeline (repo /workspace/stac-build). The user
requires the WHOLE pipeline to be BIT-FOR-BIT deterministic: the same input always gives the same bytes, whatever ran
before, whatever else is on the machine. STAC Build is an engineering tool.

STATE: wave 1 (the reconstruction: intake → Omega → F0–F6 → cloud; plan points 1–79 + 102/138/140/146, and the
orchestration points 67/73/76/77) is DONE, validated by the user on real runs (pccr 2026-08-31, zaragoza 2026-06-03) and
COMMITTED (main 1c466f4, fork vendor/VGGT-Long 67f30e9). Wave 2 (yours) = everything after the cloud, plan points 80–166
minus the ones above. The backend is RUNNING and idle (the user opens sessions in the viewer); it is restarted by the user
only when told, at the end — so every edit to server/main.py / server/pipeline_manager.py / server/workers/*.py must be
import-safe and backward compatible.

HARD CONSTRAINTS
- NEVER edit a reconstruction file: server/intake/*, server/precision/*, server/reconstruction/* (except certify/,
  surface_fit/ and loops/semantic_classes.py), server/workers/map_worker.py, server/extract_da3_depth.py,
  vendor/VGGT-Long/**, server/repro.py, server/card_table.*, server/da3_weights.py, server/potree_converter.py.
- Only edit the files of YOUR package (impl_spec2.md "Packages"). A point listed in two packages: each does only its own
  file. If a point needs a change in another package's file, write exactly what in your report; do not make it.
- NEVER run the pipeline, the backend, vLLM, SAM3 or any GPU work (CUDA_VISIBLE_DEVICES= on every command). NEVER touch
  /workspace/stac-build/server/projects/ (the user's sessions). Do NOT commit (vendor/sam31 and vendor/PotreeConverter
  included: edit in place).
- The pod dies above 117 GB RAM (cgroup v1; `free` lies) and has a 30-CPU quota. EVERY python/pytest/build you run is
  capped, from /workspace/stac-build/server:
  (ulimit -v 20000000; CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
   OPENCV_NUM_THREADS=2 timeout 1500 taskset -c <YOUR CORES> nice -n 15 /workspace/miniforge3/envs/da3/bin/python -m pytest
   -q -p no:cacheprovider <tests>)
  Cores: P2 8-13, P3 14-19, P4 20-25, P5 26-29. Check /sys/fs/cgroup/memory/memory.usage_in_bytes before a heavy job.
  UI: only `npx tsc --noEmit -p /workspace/stac-build/ui` (no dev server). C++ (P3 only): rebuild
  vendor/PotreeConverter in its existing build dir (vendor/PotreeConverter/build, cmake --build . -j4 under the same caps);
  the binary the pipeline runs is vendor/PotreeConverter/build/PotreeConverter.

READ FIRST
- docs/plan_determinismo.md — the plan. Your points' lines and their DECIDIDO are BINDING (the user's decisions). A point
  whose line says "Corrección:" without DECIDIDO: implement that correction; where it leaves a choice open, take the
  simplest option and DECLARE it in your report (the user decides; never a config switch, never a default he did not ask).
- docs/determinism_work/audit2_items.json — audit per point (field n = plan number: mechanism, evidence, fix_idea, the
  skeptic's corrections — use the skeptic's corrections). Read only your points (jq / python, by n).
- docs/pipeline_final.md section "DETERMINISM" (what wave 1 built and how it is wired).
- Shared mechanisms you MUST reuse, never re-implement:
  - server/repro.py: deterministic_torch / enable_deterministic_torch (strict), require_exclusive_gpu, card_identity,
    environment_record(gpu=...), stamp(inputs, code, config) / check_stamp, stable_id(*parts, n_hex=None), the exact
    (lossless) writers, atomic JSON/npz writes.
  - decide_change(before, after, *, error, error_factor, confidence, min_judges=None, clusters=None) in
    vendor/VGGT-Long/loop_utils/metric_lock.py (import with the fork on sys.path, as server/precision/refine.py does) = THE
    USER'S RULE: a change is applied only if (a) the 95 % CI of the paired change lies entirely on the improving side,
    (b) ≥ 5 judges, (c) the median improvement ≥ error_factor × the measured error. error_factor =
    correction_graph.graph.improvement_error_factor (1.1 — USER 2026-10-07 "1.1 en todos los casos"; accessor
    reconstruction.loops.config.improvement_error_factor(raw_config)). Where the plan says "2 × el error" it means this
    factor. For argmins: the simplest/smoothest option within error_factor × the measured error of the best.
  - correction.epoch.reconstruction_id(output_dir): the identity of a reconstruction; stamps of segmentation / masks /
    certification products carry it (F6 already refuses a seg_masks.npz without it, plan point 51).
  - intake/run_config.py: the frozen per-run config (output/run_config.yaml + sha) — every stage reads its config from it.
- The user's own criteria stay UNCHANGED (CLAUDE.md ledger: min_walk_m 1 m, conf_min_norm 0.10, the four mask-filter
  rules, min_points 1000, dedupe 0.5, …): record their margins, never retune them. No invented numbers: every new constant
  is measured, derived or the user's, with its provenance in a comment. Determinism is not a config switch. A deleted
  config key: a leftover fails the load (repo convention). Code and comments in ENGLISH.

TESTS (USER RULE, hard-earned 2026-10-07: "deja de modificar test")
- Never edit an existing test to make it pass: fix the code. An existing test may change ONLY when the plan's DECIDIDO
  changes the behaviour it asserts — then list that test and the plan point in your report. Never weaken an assertion.
- New tests: CPU-only, synthetic, under server/tests, exercising the REAL code path (the function the pipeline calls, not
  a copy), failing on regression. Keep them few and fast. Nothing enters the pipeline untested: the user runs ONE
  verification run per change on the real sessions and judges by eye.

FINAL REPORT (your last message, the user reads it): per point — implemented / no_action (plan says so) / not_implemented
(exact reason); files and functions; tests with the exact pytest command and counts (passed/failed); plus DECLARED: every
choice the plan left open that you made, every behaviour change the user must know before the next run, every existing
test you changed and why. Short, factual, no narrative.
