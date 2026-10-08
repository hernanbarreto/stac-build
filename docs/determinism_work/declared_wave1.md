# Declared by the wave-1 packages (to tell the user before he launches; to fold into docs)
## Package C (F0-F4, chain)
- chain_state STATE_VERSION 1→2: an existing chain_state.json is ignored on the next resume → re-run from F0 (pccr/zaragoza outputs are wiped anyway).
- Code-closure boundary: an edit to a boundary module (segmentation/pipeline.py, certify/run.py …) re-runs from F2.
- reconstruction_id hashes omega_run/results_output (GBs): tens of seconds once per chain + once in F2 and F4.
- OPENBLAS_CORETYPE=HASWELL pinned in step_env (measured: da3 numpy dispatched Zen, mapanything/pycolmap Haswell): F2 algebra moves to Haswell kernels — last-bit changes vs earlier runs.
- Point 58 verdict: KEEP 8 threads for F5 (measured bit-identical to 1 thread; pycolmap's OpenBLAS runs CHOLMOD on 1 thread).
- Point 39: knots at k·knot_walk_m, λ normalised per reached span — slightly different knot positions than the linspace recipe.
- known_dimensions.json has no writer in the repo; needs the reconstruction_id stamp when written by hand.
- Still scanning sibling scans for Stray (point 35): workers/map_worker._find_stray_dir (patch → package B), segmentation/shaper_export._find_stray_dir (→ C follow-up).
- Random correction ids remain outside the chain: corrected_cloud.py (D fixed), fuse.py (F7, not default), visit_drift_run.py, correction/run.py, certify/run.py, witness/*, loops/kf_graph.py (wave 2).
## Package D (F5/F6)
- STAC_REFINE_RUNGS kept as a declared diagnostic (recorded in refine.json "ladder_restricted_by").
- New config keys: bend.heldout_confidence 0.95, bend.bootstrap 2000, bend.seed 0, mono_detail.seed 0.
- PointDiT runs STRICT deterministic: if an op has no deterministic CUDA kernel, f6_bend RAISES on the first real run (fail, never degrade) — user must know.
- PotreeConverter pinned to 1 thread (POTREE_THREADS=1): ~0.55 M pts/s → pccr 22 M ≈ 40 s; 128 M ≈ 4 min. Rebuilt binary (sorted chunk files, total-order Poisson sort).
- F5 runs each rung twice (continuation = solver error): R0–R2 time roughly doubles.
- Points 48/49 change F6's numbers by design (per-keyframe fits that do not verify take the pooled neighbours' fit).
- origins.npz geometry_epoch column kept = live epoch (load_origins keys on it); reconstruction_id added alongside.
- precision/mono_ab.py reads footprint.vram_peak_gb (now None) — A/B tool, declared.
- chunk_check: an invented 0.001 acceptance in the report-only best-plane call → 0.0.
## Package B (reconstruction + DA3 windows) — done, 317 passed
- weights/omega_footprint.json physically still there; the code deletes it on the next run (declared).
- `reconstruction.simple.exclusive_gpu` still stops vLLM, but require_exclusive_gpu fails on any co-tenant regardless.
- Fork stamp (A, point 9) STOPS on products of another stamp; map_worker wipes only when the omega_complete marker is stale; a crashed first run (no marker) resumes under A's rules.
- NOTE for A: vggt_long.py gate_frame_pair call can pass salad={similarity, threshold} to log the SALAD margin.
## CROSS-PACKAGE BREAKS TO FIX BEFORE THE FINAL TEST RUN (from B's whole-suite run)
- test_certify_f3 ×7, test_depth_f3 ×3, test_kit_f4 ×2 errors: `depth_graph_verdict() missing 'held_err'/'error_factor'` — A's fork signature change; the server callers (certify/kf_graph/depth stages) must pass them.
- test_precision_runner::test_the_code_closure_stops_at_the_precision_core: gauge.py now imports precision.runner → chain gauge → runner → tracks (C).
- tests/test_precision_camera.py ×3: `chunk_check.plane_min_inlier_frac no longer exists` (D deleted the key; the camera tests' fixture config still carries it).
- test_intake_run.py ×22: run_quality() got 'geometry_epoch' — E's in-flight edit (should resolve when E finishes).
- test_viewer_socket_lock: unrelated, check.
## Package D final additions
- New key cloud_metrics.edge_max_points_per_object 380000 (derived from 900 s / 20 objects on this box's measured rate); removed chunk_check.plane_min_inlier_frac, cloud_metrics.edge_timeout_s (leftover fails load).
- chunk_check verdicts now by the user's rule (≥ 5 keyframe judges, 95 %, ≥ 2× band resolution); 'chunk vs others pooled' reference unchanged (declared contaminated when many chunks are off).
- SAM3 writer (wave 2, points 83/96) must stamp seg_masks.npz / segmentation.json with reconstruction_id or F6 snaps nothing.
