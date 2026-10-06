# Variable-size chunking (co-visibility planned) — DESIGN, not implemented

Generated 2026-10-06 from three code maps; nothing changed in the code. Needs the user's OK on the items listed at the end.

VARIABLE-SIZE CHUNKING: CO-VISIBILITY-PLANNED OMEGA CHUNKS
(Design only. No files changed, nothing run. Every default listed under \"NEEDS THE USER'S OK\" waits for his OK.)

Core idea in one line: measure, before Omega, how many keyframes each keyframe stays co-visible for (the VGG-T3 depth-consistency test). Add up the inverse of that length along the walk as a \"motion budget\" (VGGT-Motion). Cut the walk into half-chunk blocks with an equal budget, so each chunk is two consecutive blocks and the 50 % overlap is kept. Place the cuts where a seam is best conditioned.

How the parts map to the two papers:
- Static stretches cost almost no budget. This is VGGT-Motion's \"static redundancy\", priced instead of pruned.
- Turns and near-field stretches cost a lot of budget.
- Open spaces cost almost nothing, so they collapse to a single pass.
- The design has one free parameter, H, calibrated on pccr, zaragoza and observatorio.

======================================================================
1. ALGORITHM
======================================================================
Index space: i = 0..n-1, the selected keyframes in walk order. This is the same index space as the fork's post-stride chunk_indices.

1.1 Inputs per keyframe (all produced before Omega by I3)
- c2w_i: the I3 chained metric pose, taken from the SAME DA3 window that the anchor depth came from.
  - Today depth and pose come from different windows (walk.py:156-158 vs :175-182). They differ by that window pair's scale ratio.
  - Fix: persist c2w_i = G_w* · local_w*(i), where w* is the window in which i is most central.
- Z_i, conf_i, K_i: from output/da3_run/results_output/frame_<f>.npz, on the DA3 grid. Valid pixel = conf > 0, finite, Z > 0.
- tol_rel: MEASURED on the session, not chosen.
  - Every keyframe sits in two consecutive I3 windows (overlap 0.5).
  - Pool |z_a − z_b| / ((z_a + z_b)/2) over the pixels valid in both copies, sampled on the grid below.
  - tol_rel = the q-quantile of that pool, with q = covis.tol_quantile (0.95, a declared confidence like heldout_confidence).
  - It is computed in measure_walk while the windows exist and persisted, so a resume never needs the windows.

1.2 Co-visibility by depth consistency (VGG-T3)
- Samples S_i: a regular pixel grid with stride s_i = max(1, floor(sqrt(valid_px_i / samples_per_frame))) over the valid pixels, unprojected with c2w_i (precision/chunk_check.py:60 unproject).
  - samples_per_frame = 1500, a bound: binomial σ of C at 0.3 ≈ 1.2 %.
- Lookup map of j: Z_j block-median-pooled to cells of ceil(s_j/2) px. A cell is valid only if it holds a valid pixel.
- c(i→j) = |{x ∈ S_i : x projects inside j with z > 0 onto a valid cell, and |z − Z_j(cell)| ≤ tol_rel · Z_j(cell)}| / |S_i|.
  - Occluded points, contradicting points and points outside j's frustum all count as not co-visible.
  - Projection uses precision/flyers.py:162 project.
- C(i,j) = min(c(i→j), c(j→i)): j must share at least τ of EACH frame's content.
- τ = covis.tau = 0.3, VGG-T3's published value. H is calibrated AT this τ; changing τ means recalibrating H.

1.3 Co-visibility length ℓ(i) (forward reach)
- Scan j = i+1, i+2, … up to min(n−1, i+cap−1), counting passes (C ≥ τ) and failures.
- STOP at the first j where failures outnumber passes (the majority rule).
- ℓ(i) = (last passing j before the stop) − i, or 0 if no j passed.
- Why majority:
  - An isolated bad frame (blur, a person crossing, DA3 artefact) does not cut the reach.
  - A far revisit after a long non-co-visible gap does not extend it. Revisits belong to the loop closures, not to the chunk size.
- Censoring:
  - If the scan stops at i+cap−1 without failing, then ℓ(i) = cap (a lower bound).
  - If it reaches the END of the walk without failing (the censored tail [i_c, n), where i_c is the first frame from which every later frame is censored), the tail is defined to cost exactly ONE length: δ = 1/(n − i_c) per frame.
  - Without this, the walk's end would read as fast motion.
- Break: ℓ(i) = 0 means the next keyframe shares < τ of the content. Then δ(i) = 1 and the step is listed in the report's \"breaks\".

1.4 Motion budget (VGGT-Motion)
- δ(i) = 1 / max(ℓ(i), 1).
- D(a,b) = Σ_{i=a}^{b−1} δ(i), via prefix sums. It counts co-visibility lengths walked from a to b, including the step out of b−1 (conservative by one step).
- D does not depend on keyframe density, because ℓ is counted in the same keyframes. This is why the calibration transfers across keyframe quanta.
- Effects:
  - A static stretch has ℓ → cap and δ ≈ 0, so it costs nothing.
  - A turn makes C collapse by rotation, so it costs budget. No separate flow or turn detector is needed: depth-consistent reprojection sees rotation and translation alike.
  - Near-field sideways motion costs most. Far background costs least.

1.5 Single pass
- If D(0,n) ≤ H AND n ≤ cap, the plan is [(0,n)] and the existing single-pass branch runs.
- cap = the existing capacity formula (map_worker.py:2310-2316, card TOTAL memory, so it is deterministic per card).

1.6 Blocks and chunks (50 % overlap semantics kept)
- Cut [0,n) into blocks B_0..B_{m−1} with boundaries 0 = b_0 < b_1 < … < b_m = n.
- chunk k = B_k ∪ B_{k+1} = [b_k, b_{k+2}), for k = 0..m−2.
- The overlap of chunks k and k+1 is exactly block B_{k+1}: half of each chunk measured in co-visibility length, and exactly 60/30-style when the blocks are equal.
- Fork rules hold by construction:
  - starts and ends strictly increasing;
  - chunk k ends where chunk k+2 starts, so no frame is in three chunks;
  - owner = frame_owner steps by +1;
  - first range starts at 0, last range ends at n.
- Block feasibility:
  - L_min ≤ b−a ≤ L_max, with L_min = min_chunk_frames // 2 and L_max = cap // 2. So every chunk is ≤ cap and ≥ min_chunk_frames.
  - D(a,b) ≤ H/2, or b−a = L_min. A stretch so fast that even the minimum chunk exceeds the budget is allowed and flagged \"over budget at minimum\".
  - min_chunk_frames = 24: the existing clamp, now declared. It is ≥ loop_chunk_size 20 and ≥ the intra-chunk floor of 8. It gives a seam of ≥ 12 frames, above the fork's 8-frame fallback floor (sim3utils.py:958).

1.7 Where the cuts go (exact, deterministic, no weights)
Choose boundaries by lexicographic optimisation:
  (a) minimum number of blocks m. Every seam costs, so use the fewest chunks the budget allows.
  (b) among those, minimise the WORST seam's median depth z̄(B_{k+1}) over the interior blocks B_1..B_{m−2}.
      - z̄ = median over the block's frames of each frame's median valid DA3 depth.
      - A seam's rotation error is multiplied by the distance of what the chunks see (the user's observation on zaragoza), so seams should sit on near structure.
  (c) then minimise the WORST rotation across a cut: θ(b) = angle(R_{b−1}, R_b) from c2w.
      - A chunk end must not land mid-turn (VGGT-Motion \"encapsulate turns\").
  (d) then the leftmost boundary vector.
- Implementation:
  - one DP over prefixes for m (O(n·L_max));
  - then a bottleneck feasibility DP with a binary search over the sorted distinct z̄ values, then the same over θ;
  - then a leftmost-tie DP.
  - Total is about 1–2 M transitions for n ≈ 2000.

1.8 Determinism
- All inputs are persisted files.
- Sample grids are fixed; there is no RNG.
- Pairs are computed independently and assembled by index, so the result is the same for any number of workers.
- Sums are float64 in index order; DP ties go leftmost.
- intake/covis.json caches ℓ, δ, z̄, θ and tol_rel, stamped with the walk.json sha256, n and the covis config block. A resume replans from it bit-identically.
- Not deterministic ACROSS a re-run of I3: DA3 on the GPU is not bit-stable (see risks).

1.9 The one free parameter: H = covis.lengths_per_chunk (co-visibility lengths per chunk)
Calibration needs data that does not exist on disk today:
- pccr/2026-08-24, pccr/2026-08-31 and zaragoza/2026-06-03 hold only frames/ (plus output/ entries); none of the three has frames/selected_frames.json.
- observatorio/2026-08-11 has a selected_frames.json in an older schema (no \"keyframes\" key).

For each calibration scene (pccr on the scan where 5 m was validated and 15 m drifted, zaragoza, observatorio; test2 and bufferStop optional), the user launches:
- the intake keyframe selection (CPU), then
- I3 on the GPU: `python -m intake.walk --session …`. The window depth files must be regenerated; pccr took ≈ 28 min for 1998 kf per the 2026-10-05 log.
- measure_walk, extended as in section 2.

Then, CPU only, under the pod caps:
```
python -m reconstruction.chunk_covis --session <pccr> --layout walk:5 --layout walk:15
python -m reconstruction.chunk_covis --session <zaragoza> --total
python -m reconstruction.chunk_covis --session <observatorio> --total
```
- `--layout walk:X` rebuilds the uniform 50 % layout at X m of CHAINAGE (walk.json), not of frames, and prints D per chunk.
- lo = max(max_k D(pccr 5 m chunk k), D_total(zaragoza), D_total(observatorio)[, other single-pass verdicts]).
- hi = min_k D(pccr 15 m chunk k). Every 15 m chunk counts as failed, because the 90.5 cm seam cannot be assigned to one chunk.
- If lo < hi: H = sqrt(lo·hi) (equal ratio margin to the nearest success and the nearest failure), or H = lo (least extrapolation). This is the user's choice. Write lo, hi and every D into the config comment.
- If lo ≥ hi: co-visibility does NOT separate the validated verdicts. The design is refuted by its own calibration and does not ship. This is the first thing the calibration tests.
- Also check n ≤ cap on zaragoza at its grid (1080p scales the per-frame footprint by about 5.4×). A single pass that the card cannot hold is decided by capacity, not by H.

======================================================================
2. DATA FLOW
======================================================================
Pre-Omega inputs:
- frames/selected_frames.json (parallax.py:762-782): n, frame ids, order.
- output/da3_windows/window_NNNN.npz (extract_da3_depth.py:247-252): read only inside measure_walk (walk.py:361), on a fresh I3.
- output/da3_run/results_output/frame_<f>.npz (walk.py:172-195): depth, conf, K per keyframe.
- intake/walk.json (walk.py:379-401): chainage, used only for metres in the report.
- intake/coverage_warnings.json (parallax.py:799-808): cross-check only. The report lists any cut that falls in a static, pure_rotation or tracking_lost run. It is not an input.

Changes in intake/walk.py:
- chain_windows (:119): new kwarg return_placements=True, which also returns G per window. Callers: measure_walk, revisit_reference.
- write_anchors (:172): also return the map frame → (window, index) already computed in `best`.
- measure_walk (:361):
  - writes intake/walk_poses.npz {frames, c2w (from the anchor's window), window} (about 128 KB for 2000 kf);
  - adds walk.json \"depth_tol\": {quantile, tol_rel, n_pixels}.
- WALK_VERSION 1 → 2, so walk_is_current (:299) rejects an old walk.json that has no walk_poses.npz.

New module server/reconstruction/chunk_covis.py (numpy only):
- CovisConfig (frozen dataclass) and load_covis_config(simple_cfg): strict.
  - A missing key raises, naming the key.
  - When chunking == covis, lengths_per_chunk null or ≤ 0 raises \"not calibrated\".
- load_inputs(session_dir) → frames, c2w, anchor paths, tol_rel, chainage.
- pair_covis(Xi, Zj_cells, Kj, w2cj, tol_rel) → c(i→j).
- reach_lengths(inputs, cfg, cap, workers) → ℓ, censored tail i_c, breaks, pairs_evaluated.
  - Parallel over contiguous i-blocks (ProcessPool, 2–3 workers, OMP=2), with a sliding window of pooled j-maps.
- step_costs(ℓ, i_c) → δ. seam_depths(…) → z̄ per keyframe. cut_angles(c2w) → θ.
- plan_blocks(δ, z̄, θ, n, cap, H, min_chunk) → blocks plus the binding constraint per block.
- plan_covis(session_dir, cfg, cap) → plan doc. It reuses intake/covis.json when its stamp matches.
- main(): the CLI dry run.
```
python -m reconstruction.chunk_covis --session S [--lengths H] [--cap auto|N] [--layout walk:X]... [--total] [--workers 3] [--out f.json]
```
  - It prints n, cap, D_total, ℓ quantiles, the chunk table (range, frames, metres, D, seam frames, seam z̄ m, end θ°, why) and single pass yes/no.
  - It never writes output/.
  - --cap auto uses omega_capacity(), moved from map_worker :2310-2316 into reconstruction/chunk_plan.py so the worker and the CLI share it.

Reports:
- intake/covis.json (plan-independent, survives _invalidate_on_new_chunk_plan): version, stamp, tau, tol_rel, samples_per_frame, ℓ/δ/z̄/θ per keyframe, i_c, breaks, D_total, pairs_evaluated, seconds.
- output/chunk_plan.json, version 2:
  - phase \"covis-planned\"
  - n_keyframes
  - chunk_ranges (explicit)
  - seam_overlaps [e_k − s_{k+1}]
  - chunk_size = max length and overlap = min seam (units.py:38 still requires both)
  - walk_m
  - \"method\": \"covis\"
  - \"covis\": {H, tau, tol_rel, D_total, blocks, chunks: [{range, frames, chainage_m, D, why: budget | capacity | minimum | walk_end | over_budget_at_minimum}], seams: [{frames, z_med_m}], cuts: [{frame, chainage_m, theta_deg, coverage_flag}]}
- A single pass writes no chunk_plan.json (as today). Its decision and reason (\"D_total X ≤ H\") go to intake/covis.json and the log.

Cost: pccr-like scenes take seconds (the scan stops after ≈ 2ℓ). The worst case, an open scene of 2000 kf scanning to cap, is about 3.5 M directional pairs × ~173 µs, about 10 min on one core and about 3–4 min on 3 workers, all CPU and inside the pod caps.

======================================================================
3. FORK CHANGES (vendor/VGGT-Long), explicit Model.chunk_ranges with a per-seam overlap
======================================================================
F1. loop_utils/metric_lock.py, next to frame_owner (:1192). Pure functions, shared with the server:
- uniform_chunk_ranges(n, size, ov): the vendor formula, moved out of process_long_sequence.
- validate_chunk_ranges(ranges, n, min_seam) → list of int tuples. It RAISES on:
  - (1) a malformed or empty list (and converts numpy ints to int);
  - (2) first start ≠ 0 or last end ≠ N;
  - (3) starts or ends not strictly increasing;
  - (4) a seam with e_k − s_{k+1} < min_seam;
  - (5) e_{k−1} > s_{k+1} (a frame in three chunks);
  - (6) owner = frame_owner(…) not ≥ 0 everywhere, a chunk owning no frame, or an owner step outside {0, +1}.
  - (7) a chunk shorter than max(8, loop_chunk_size) only logs a WARNING.
- seam_slices(ci, k) → (slice(s_{k+1}−s_k, e_k−s_k), slice(0, e_k−s_{k+1})).

F2. vggt_long.py:120-121
- Read chunk_size and overlap with .get.
- self.chunk_ranges_cfg = Model.get('chunk_ranges').
- self.min_seam = Model['min_seam_frames'] (mandatory when chunk_ranges is present).

F3. vggt_long.py:3218-3231
- With chunk_ranges: chunk_indices = validate_chunk_ranges(cfg, N, min_seam) and num_chunks = len(...).
- If N ≠ ranges[-1][1]: fail loudly (post-stride N vs the server's n_keyframes).
- Otherwise uniform_chunk_ranges.

F4. vggt_long.py:3330-3346 (exact_seam_align and the vendor fallback): wp, conf and mask use seam_slices instead of [-self.overlap:] / [:self.overlap].

F5. vggt_long.py:1891-1897 (_stac_seam_chain): same change as F4.

F6. vggt_long.py:2619-2632 (_stac_ensemble_uncertainty)
- Shift the real ranges: [(a+off, min(b+off, N)) for a, b in chunk_indices if a+off+2 < N].
- Deduplicate the ranges that clip to N.

F7. vggt_long.py:3305-3306: log each chunk's length and each seam's overlap.

F8. Resume stamps:
- process_single_chunk (:300-345, save at ~388): store predictions['_stac_range'] = [s, e]. On load, a stamp that is present and differs raises _StacPlanMismatch.
- metric_lock.json (:414-425), chunk_health.json (:691-704) and scale_graph.json (:2319-2337): write \"chunk_indices\". A report whose chunk_indices differ is ignored and recomputed.

No change is needed anywhere else (map section D), given rules 1–6. For uniform ranges, seam_slices equals the old slicing, so the walk branch stays bit-identical.

======================================================================
4. SERVER CHANGES
======================================================================
Config, server/config.yaml reconstruction.simple:
```
chunking: walk                 # walk | covis — stays walk until calibration + user's verdict
covis:
  tau: 0.3                     # VGG-T3 published; H is calibrated at this tau
  tol_quantile: 0.95           # declared confidence over the session's own two-window depth disagreement
  samples_per_frame: 1500      # BOUND: sampling error of C ≈ 1.2 % at 0.3
  min_chunk_frames: 24         # the existing clamp, now declared (≥ loop_chunk_size 20, seam ≥ 12 ≥ 8)
  lengths_per_chunk: null      # H — CALIBRATED (§1.9), evidence in this comment; null + covis fails the load
```
- reconstruction.vggtomega.min_seam_frames: 8 (the fork's fallback floor), written to Model.min_seam_frames.
- A typed loader plus a config-load test cover all of these keys.

map_worker.py:
- :2039 import omega_capacity and plan_covis. :2310-2316 compute the capacity through omega_capacity().
- :2332-2333 when chunking == covis:
  - a missing _walk_doc raises (covis needs I3; no silent fallback);
  - plan_covis runs as a capped subprocess (`python -m reconstruction.chunk_covis --out`), like the other heavy jobs;
  - single pass goes to branch B (:2377);
  - otherwise a new branch \"covis-planned\", next to A (:2334-2377): plan_anchor_indices(ranges=…), _ensure_anchors, _apply_chunked_metric(cfg_v, ranges), the SALAD min_gap as in branch A, the revisit reference, and _persist_chunk_plan(ranges, n, \"covis-planned\", walk0, extra=report).
- :2064-2084 _persist_chunk_plan(ranges, n_kf, phase, walk, extra=None):
  - chunk_size = max length, overlap = min seam, seam_overlaps, explicit chunk_ranges.
  - The walk and pinned callers (:2377, :2437, :2688) pass uniform_chunk_ranges(…), so their plans are unchanged.
- :2086-2088 _apply_chunked_metric(cfg_v, ranges): write Model.chunk_size and overlap as above, plus Model.chunk_ranges and Model.min_seam_frames.
  - The single-pass branches (:2377-2379, :2459-2460, :2471-2472) pop chunk_ranges.
- :2377-2383 (single pass), an existing gap: before unlinking chunk_plan.json, call _invalidate_on_new_chunk_plan with {chunk_ranges: [[0,n]], n_keyframes: n}.
  - Today a single pass over the chunk files of an old plan hits _StacPlanMismatch and the run fails, instead of wiping them.
- :2747-2749 invalidation log: keep \"size/overlap\" for uniform plans (the test pins \"296/148\") and add \"k chunks, lengths a–b, seams c–d\". The comparison (:2738-2741) is already range-based.
- :1539-1616 _emit_omega_depth(save, out, ranges, sel, pipe): start = ranges[k][0] and owner = frame_owner. The ranges come from maplong_run/chunk_sim3.json \"chunk_indices\", which records what actually ran, with chunk_plan.json as fallback. The per-record `chunk` field that precision and floor read is then correct.
- :3139-3141, :3210, :3295, :3335-3337 _generate_origins:
  - abs_idx = frame_local + ranges[K][0];
  - meta frame_global_start/end come from the ranges, on both paths;
  - chunk_step is written only for uniform plans (legacy readers).

Other consumers:
- reconstruction/chunk_plan.py:
  - rename chunk_ranges to uniform_chunk_ranges and keep the old name as an alias;
  - plan_anchor_indices(n, size, ov, per_chunk, *, ranges=None);
  - add omega_capacity();
  - re-export validate_chunk_ranges.
- segmentation/tsdf_export.py:349-418: (chunk, local) via ranges + frame_owner, read from chunk_sim3.json, then chunk_plan.json, then the chunk_*_meta frame_global_start. Legacy chunk_step only when none exists. This covers mv_consistency and native_depth through it.
- reconstruction/trace_normals.py:121-135: the same ranges source, instead of the YAML size and overlap.
- Unchanged (already range-based): correction/units.py (both keys still written), floor.py, consistency.py, kfgraph.py, photobundle.py, revisit.py, certify/scale_stage.py, loops/structural.py, instance_loops.py, quality/ab_elastic.py, every precision/* consumer (via `chunk`).

======================================================================
5. TESTS (synthetic, CPU, no GPU)
======================================================================
New server/tests/test_chunk_covis.py. Scenes use analytic ray–plane or ray–box depth on a small grid (about 60×100), written as anchors plus walk_poses.npz in tmp:
- Near sideways walk (wall at 1.4 m, 0.087 m/kf, 300 kf):
  - ℓ is within ±1 kf of the analytic overlap length;
  - the plan gives several short chunks, all D ≤ H;
  - every layout passes validate_chunk_ranges.
- Far background (15 m, same walk): single pass, D_total ≤ H.
- Mixed (first half at 1.4 m, second at 15 m): short chunks, then one long chunk covering the far half; chunk lengths are non-decreasing.
- Turn (90° in place over 10 kf inside a box room):
  - no cut boundary inside the turn frames;
  - the whole turn lies inside at least one chunk.
- Static stretch (50 kf stationary): δ ≈ 0 there, no extra chunk.
- Break (a jump of the camera): δ = 1 at the break, listed in \"breaks\".
- Majority rule: one corrupted depth frame does not shorten ℓ of its neighbours by more than 1; a far revisit after a long gap does not extend ℓ.
- Censored tail: D(tail) = 1; a fully co-visible scene gives D_total = 1.
- Capacity: a far scene with n > cap gives chunks ≤ cap, and the cut is at the lowest-z̄ / lowest-θ position.
- Minimum: H tiny gives blocks at L_min, flagged over_budget_at_minimum.
- Determinism: workers=1 and workers=3, and two runs, give byte-identical plan JSON; a resume from covis.json gives the identical plan.
- Config: a missing key fails naming it; covis with H null fails.
- tol_rel: windows with known depth noise recover the injected quantile.

New server/tests/test_chunk_ranges_validator.py (imports the fork's metric_lock):
- uniform_chunk_ranges equals the vendor formula on the 7 existing cases;
- each rejection rule 1–6 fails as specified;
- numpy ints are normalised;
- seam_slices equals [-ov:] / [:ov] on uniform layouts and is correct on uneven ones.

Updates:
- test_vendor_config_wiring: plan[\"chunk_ranges\"] == cfg[\"Model\"][\"chunk_ranges\"] for the covis and walk branches; the single pass has no chunk_ranges.
- test_omega_emit: the new ranges signature, plus an uneven-layout case.
- tsdf_export / trace_normals: a position → (chunk, local) case on uneven ranges.
- test_chunk_plan_invalidation: same n with different explicit ranges wipes; the same ranges with a different \"covis\" report do not wipe; single pass after a chunked plan wipes.
- test_intake_walk: chain_windows placements; walk_poses uses the anchor's window; WALK_VERSION 2 rejects v1.
- test_chunked_metric :987 and :1071: fix the stale `/ 0.086` source strings (they already fail today); the walk branch's plan ranges are unchanged (regression).
- synth_metric.write_aligned_chunks(ranges=…): optional uneven layout for the witness and depth tests.

======================================================================
6. RISKS / UNVERIFIABLE UNTIL A REAL RUN
======================================================================
- The core hypothesis is that \"co-visibility lengths per chunk\" predict Omega's drift inside a chunk. It rests on 3–5 visual verdicts and one parameter. The mixed-scene plan (short then long chunks) and seams between unequal neighbours are unverified until a real run and the user's eye.
- The calibration may be infeasible (lo ≥ hi). That refutes the design before any code ships, and it is the first output.
- I3 chain error lowers C at long baselines, worst in the near field. This is conservative (shorter chunks), but a bad DA3 seam (pccr has shown 55.8 cm) can split a scene Omega would hold. On the validated scenes the calibration includes it; elsewhere it is unknown.
- Re-running I3 is not bit-stable (DA3 on the GPU), so D can move a few % and flip a plan near H. Persisted inputs make a resume exact; the midpoint H gives margin.
- Capacity can override H. At zaragoza's 1080p grid, cap is about 164 frames, so the \"single pass\" verdict holds only if n ≤ cap.
- Keyframe pruning (VGGT-Motion) is NOT proposed. It changes the keyframe set that every downstream stage indexes. Static frames cost no budget, only capacity.
- The seam stages were tuned on uniform 50 % layouts, and σ_seam is constant per seam whatever its frame count. Their behaviour on uneven seams is unmeasured.
- The tie-break order (seam depth before turn angle) is reasoned, not measured.

NEEDS THE USER'S OK BEFORE ANY CODE
- chunking default stays `walk`.
- H rule: midpoint or lo.
- The tie-break order.
- tau 0.3, tol_quantile 0.95, samples_per_frame 1500, min_chunk_frames 24, min_seam_frames 8.
- Persisting walk_poses.npz and bumping WALK_VERSION (every session is wiped already).
- The single-pass invalidation fix.
- Who launches the intake and I3 runs for the calibration.

Files referenced: /workspace/stac-build/server/workers/map_worker.py, /workspace/stac-build/server/reconstruction/chunk_plan.py, /workspace/stac-build/server/intake/walk.py, /workspace/stac-build/server/correction/revisit.py, /workspace/stac-build/server/config.yaml, /workspace/stac-build/vendor/VGGT-Long/vggt_long.py, /workspace/stac-build/vendor/VGGT-Long/loop_utils/metric_lock.py. New: /workspace/stac-build/server/reconstruction/chunk_covis.py, /workspace/stac-build/server/tests/test_chunk_covis.py, /workspace/stac-build/server/tests/test_chunk_ranges_validator.py.",
    "maps": {
      "fork": "CHUNK-LAYOUT MAP: VGGT-Long fork (vendor/VGGT-Long), read-only. No files were changed and nothing was run.

Bottom line: only five places in the fork assume one fixed chunk size and overlap. Everything else already reads each chunk's real (start, end) from self.chunk_indices. On top of the obvious range checks, the variable layout needs one extra rule: a frame may belong to at most two chunks. Several stages only ever compare chunk k with chunk k-1. The bigger risk is on the server side, where five consumers rebuild the uniform layout from chunk_size/overlap on their own.

================================================================
A. WHERE chunk_indices IS BUILT, AND HOW TO PASS Model.chunk_ranges
================================================================
vggt_long.py:120-121  __init__ reads `self.config['Model']['chunk_size']` and `['overlap']` as mandatory keys, so a missing key raises KeyError.
  - Change: read both with .get, and add `self.chunk_ranges_cfg = self.config['Model'].get('chunk_ranges')`.

vggt_long.py:3218-3231  process_long_sequence builds the layout:
  - It refuses overlap >= chunk_size (3219), uses one chunk when N <= chunk_size, otherwise step = chunk_size - overlap, num_chunks = ceil((N-overlap)/step), end = min(start+chunk_size, N).
  - Change: when chunk_ranges is given, set `self.chunk_indices = validate_chunk_ranges(self.chunk_ranges_cfg, N, min_overlap)` and `num_chunks = len(self.chunk_indices)`; otherwise keep the vendor formula.
  - The check must run here, not in __init__. N is only known after run() (vggt_long.py:3910-3930) applies selected_frames and frame_stride, so the ranges are in post-stride keyframe-index space. If N differs from the server plan's n_keyframes, fail loudly.

Where the validator should live: as a pure function in loop_utils/metric_lock.py, next to frame_owner (metric_lock.py:1192). The server already imports frame_owner from there (reconstruction/certify/scale_stage.py:40-47), so both sides would share one validator. Suggested companions there:
  - `uniform_chunk_ranges(n, size, ov)`: the vendor formula, moved out of process_long_sequence.
  - `seam_frames(ci, k)`: returns range(ci[k+1][0], ci[k][1]).

Validation rules, with the reason for each:
  1. A non-empty list of [start, end) pairs, converted to Python ints and stored as tuples. Several resume checks compare `[list(ci) for ci in chunk_indices]` against JSON, so numpy ints or floats would make them report a mismatch.
  2. ranges[0][0] == 0 and ranges[-1][1] == N.
     Why: save_camera_poses (3846-3885) leaves all_poses[idx] = None for a frame no chunk covers, then crashes on pose.flatten().
  3. Starts and ends both strictly increasing.
     Why: the seam logic assumes chunk k+1 starts after chunk k; the vendor fallback find_chunk_index (sim3utils.py:438) bisects on the starts.
  4. Contiguous: every seam overlaps by at least min_overlap frames, i.e. e_k - s_{k+1} >= min_overlap.
     - Hard floors in the code: 1 frame for the exact seam (robust_rigid on the pixel correspondences) and the metric-lock seam ratio.
     - 3 frames, or scale_drift_gate (metric_lock.py:381-389) holds out nothing from that seam.
     - 8 frames, or the vendor fallback weighted_align_point_maps (sim3utils.py:958, align_min_inlier_frames) only logs a warning.
     - Make min_overlap a declared config key (e.g. Model.min_seam_frames) rather than a number written into the code.
  5. No frame in three chunks: e_{k-1} <= s_{k+1}. This is mandatory, because these sites only handle k-1 with k:
     - metric_lock.py:762-784 elastic_corrections: the right-seam loop overwrites the corr[g-start] slot the left seam just wrote.
     - vggt_long.py:1592-1640 blend_copies: blends pairwise in sequence, so a frame in three chunks gets an asymmetric result.
     - vggt_long.py:1788-1810 prepare_backfill: the (k, local) key gets overwritten.
     - Seams k-1/k+1 are never measured at 975-1035 (elastic fit), 2570-2600 (uncertainty, which also hard-codes witnesses: 2) and 436-497 (metric-lock seams).
  6. With owner = frame_owner(ranges, N): every frame has owner >= 0, every chunk owns at least one frame, and the owner only ever steps by 0 or +1.
     Why: the pose graph at 2939-2942 adds the seam residual using seam_res[owner[g]], which is only right if the owner moves from k to k+1.
  7. Recommended, not required: each chunk length >= max(8, loop_chunk_size).
     - Intra-chunk correction skips chunks with S < 8 (1247).
     - get_frame_range (sim3utils.py:454) shrinks a loop-bridge window to the whole chunk when the chunk is shorter than the window.

================================================================
B. FORK SITES THAT ASSUME ONE CHUNK SIZE / ONE OVERLAP — MUST CHANGE
================================================================
B1. vggt_long.py:3330-3346  Main alignment loop (exact_seam_align and the vendor fallback).
  - Current code: `point_map1 = wp1[-self.overlap:]`, `point_map2 = wp2[:self.overlap]`, the same for conf1/conf2, and the same for mask1/mask2 (3343-3344).
  - Assumes: every seam shares exactly `overlap` frames.
  - Change: n = chunk_indices[k][1] - chunk_indices[k+1][0]; use [-n:] and [:n] on world_points, world_points_conf and mask (equivalently [s_{k+1}-s_k : e_k-s_k] for chunk k).

B2. vggt_long.py:1891-1897  _stac_seam_chain (the provisional chain used by the spatial gate).
  - Same [-self.overlap:] / [:self.overlap] slicing; same change as B1.

B3. vggt_long.py:2619-2632  _stac_ensemble_uncertainty (off unless ensemble_offset_frames > 0).
  - Assumes uniform chunks: rebuilds a shifted layout from step = chunk_size - overlap and end = min(start + chunk_size, N).
  - Change: shift the real ranges instead: `ranges = [(a+off, min(b+off, N)) for a, b in chunk_indices if a+off+2 < N]`, dropping duplicates where several clip to N.
  - _stac_ensemble_apply (2650+) is already owner-based and needs no change.

B4. vggt_long.py:3219-3231  The layout build itself (see A).

B5. vggt_long.py:3305-3306  Log line \"N chunks of size {chunk_size} with {overlap} overlap\".
  - Cosmetic: print each chunk's length and each seam's overlap instead.

================================================================
C. RESUME CHECKS AND CHUNK-PLAN STAMPS
================================================================
Already layout-safe (each stores the actual chunk_indices list and compares it):
  - elastic_seams.json: 1036, 1068
  - intra_chunk.json: 1235, 1319
  - depth_graph.json: 1401, 1499
  - uncertainty.json: 2560, 2603
  - pose_graph.json: 2880, 2893, 2917, 3087
  - chunk_sim3.json: 3562-3566. This file records the ranges actually run, so it is the best single source of truth for server consumers.

Weak (worth fixing alongside the change):
  - vggt_long.py:300-345 process_single_chunk only checks the frame count on disk (`_n_on_disk != expected_frames`, then _StacPlanMismatch). Two plans with the same per-index lengths pass this check; that is already true today (60/30 and 60/20 both give 60-frame chunks).
    Change: stamp `predictions['_stac_range'] = [start, end]` before np.save (~388, non-loop chunks) and on load raise _StacPlanMismatch when it is present and differs. The ensemble path already stamps _stac_range (2640).
  - vggt_long.py:414-425 metric_lock.json: the `already` set is keyed by chunk index with no plan stamp; the same applies to chunk_health.json at 691-704.
    Change: add \"chunk_indices\" to both reports and refuse or ignore them when it differs. Today only the server's _invalidate_on_new_chunk_plan (map_worker.py:2718-2760, which compares chunk_ranges) protects this.
  - vggt_long.py:2319-2337 scale_graph.json: write chunk_indices next to n_chunks.

================================================================
D. FORK SITES ALREADY SAFE FOR VARIABLE CHUNKS (no change if rules 1-6 hold)
================================================================
Metric lock and scale:
  - 436-497 per-chunk anchors use u = loc/(S_k-1) with S_k = end-start (448); the seam ratio uses the intersection range(max(nxt[0], start), end) (493).
  - 584-603 drift frames use np.linspace over S_k (593).
  - The rest of the scale machinery is generic in n_chunks and per-chunk positions: solve_scale_graph (metric_lock.py:133), solve_scale_drift (293), scale_drift_gate (357), seam_residuals, scale_break_diagnosis, flag_sick_chunks / flag_suspect_chunks / chunk_trust.

Elastic seams:
  - 975-1035 per-frame fits on prev_tail = range(max(nxt0, start), end).
  - elastic_corrections (metric_lock.py:734-785) computes the shared range and L per seam. Its docstring's \"overlap start = chunk centre\" only holds at 50 %, but alpha is linear over the real overlap, so the correction stays continuous.
  - smooth_seam_fits (613) clips its window on short seams.

Intra-chunk and depth graph:
  - 1245-1310 run per chunk with its own S; blend_chunk_fields (metric_lock.py:1362) uses triangular weights over each chunk's own S.
  - Depth graph 1387-1520 works per owner frame.

Writers:
  - blend_copies 1592-1640 uses range(max(sb, sa), min(ea, eb)).
  - Depth cap and far contradictions 1700-1760 are owner-based.
  - prepare_backfill 1760-1810 uses range(start, min(e_j, end)).
  - _stac_owned_confs 852-889, _stac_write_origins (start = chunk_indices[K][0]) and _stac_write_chunk_outputs (S = end-start).

Loops and bridges (loop_chunk_size):
  - 3244-3285: half = loop_chunk_size/2; the window comes from get_frame_range inside the owner chunk.
  - bridge_layout (loop_bridges.py:149-166), _stac_lock_bridges, _stac_measure_loops, _stac_scale_close and _stac_verify_loops.
  - LoopModel.py:218 uses loop_chunk_size only for the SALAD non-local band.
  - The vendor loop path (3418-3480) uses relative begins.
  - loop_chunk_size never touches the chunk layout.

Uncertainty and pose graph:
  - _stac_uncertainty 2570-2600, _stac_holdout_pairs, _stac_odometry_sigma and loop_judge.split_loop_edges are owner-based.
  - Pose graph 2930-2944 needs rule 6; its apply step at 3095-3135 runs per frame.

Apply and outputs:
  - The apply loop 3597-3660 is indexed by chunk number; the single-chunk path at 3576 is unchanged when chunk_ranges = [[0, N]].
  - save_camera_poses 3788-3885 uses owner and all_camera_poses[k][0] = chunk_indices[k] (343, 364); it needs rule 2.

================================================================
E. SERVER CONSUMERS THAT RE-DERIVE THE UNIFORM LAYOUT (outside the fork, but they will break)
================================================================
Passing the ranges in:
  - map_worker.py:2086-2088 _apply_chunked_metric writes only chunk_size and overlap.
    Change: add `cfg_v[\"Model\"][\"chunk_ranges\"] = ranges`, built from the same list that _persist_chunk_plan stores. It is called at 2347, 2420 and 2669.
    The single-pass branches (2377-2379, 2459-2460, 2471-2472) should `pop(\"chunk_ranges\", None)`.
    Add a wiring test to test_vendor_config_wiring checking that plan[\"chunk_ranges\"] == cfg[\"Model\"][\"chunk_ranges\"].
  - map_worker.py:2064-2084 _persist_chunk_plan builds its ranges from chunk_ranges(n, size, ov).
    Change: take the explicit ranges. Keep chunk_size/overlap keys in the plan (max length / min seam, plus a seam_overlaps list), because correction/units.py:38 requires int(plan[\"chunk_size\"]) and int(plan[\"overlap\"]); alternatively relax that check.
    The invalidation message at map_worker.py:2747 prints size/overlap.
  - reconstruction/chunk_plan.py:90-106 plan_anchor_indices calls chunk_ranges(n, size, ov) to place the DA3 anchors (used at map_worker 2345, 2418, 2651).
    Change: take the ranges list, otherwise some chunks may get no anchors.

Rebuilding the layout after Omega:
  - map_worker.py:1539-1584 and 1616 _emit_omega_depth rebuilds the uniform layout and uses `start = k*step`.
    Change: read maplong_run/chunk_sim3.json \"chunk_indices\" (or chunk_plan.json), use start = ranges[k][0] and owner = frame_owner.
  - map_worker.py:3139-3141, 3210, 3295, 3335-3337 origins and chunk meta use chunk_step = chunk_size - overlap, `abs_idx = frame_local + K*chunk_step`, and write frame_global_start/end as K*chunk_step.
    Change: use ranges[K][0]. The inline path at 3210 writes only \"chunk_step\"; it should write frame_global_start/end per chunk.
  - segmentation/tsdf_export.py:349-418 maps position to (chunk, local) with K = min(pos // chunk_step, n-1), local = pos - K*chunk_step, reading chunk_step from the meta or the YAML.
    Change: use ranges plus frame_owner.
  - reconstruction/trace_normals.py:121-135 builds the layout from vggt_omega_config.yaml chunk_size/overlap; change it to read the ranges.

Already range-based, no change needed:
  - correction/floor.py:617-627
  - correction/consistency.py:246
  - correction/kfgraph.py:431
  - reconstruction/certify/scale_stage.py:36-52
  - reconstruction/loops/structural.py:305-347
  - reconstruction/loops/instance_loops.py:642-649
  - reconstruction/quality/ab_elastic.py (reads chunk_sim3.json)

Out of scope:
  - frames/storage.py:660-670: the legacy streaming backend's chunk meta.
  - stray_da3_streaming.py: has its own chunk_indices.

Tests that pin uniform slicing:
  - server/tests/test_chunked_metric.py:44-60 (chunk_ranges must equal the vendor slicing)
  - test_chunk_plan_invalidation.py:14-16
  - synth_metric.py:328-331 and 891-894 (chunk_step in the meta)
  - test_vendor_config_wiring.py:84-86",
      "server": "SERVER-SIDE CHUNK PLANNING: CONTROL FLOW, CONSUMERS AND TESTS (read-only; nothing run, nothing modified)

Short answer: chunk_plan.json already stores explicit `chunk_ranges`, and most of correction/ and certify/ uses only those ranges. Five places still rebuild the layout as `k * (chunk_size - overlap)`: the vendor slicing, `_emit_omega_depth`, `_generate_origins`, the tsdf_export depth loader and trace_normals. These have to move to the ranges first.

== 1. HOW THE LAYOUT IS DECIDED TODAY (server/workers/map_worker.py, `_run_vggtomega` at :1898) ==

In order:

a. :1906-1913 `_n_selected` is the length of selected_frames.json. :1918-1920 stops vLLM (exclusive GPU).

b. :1931-1946 `scale_anchor_frames` is 0 in config, so no subset is picked and every keyframe anchors.
   :1953-1973 is a pre-plan of per-chunk anchors, `plan_anchor_indices(n, chunk_frames, chunk_frames//2)`. It is dead today because `chunk_frames` is 0.

c. :1979-1994 measures the I3 walk BEFORE Omega, only when precision is enabled.
   - If `intake.walk.walk_is_current` (walk.py:299) holds, `load_walk` is used.
   - Otherwise `run_da3_windows` and then `measure_walk` (walk.py:361) run.
   - The result `_walk_doc` comes from <session>/intake/walk.json. Its fields: `walk_length_m`, per-keyframe `chainage` [{frame, chainage_m}], `windows`, `seams`.
   - plan_chunks never uses the per-keyframe chainage; it only uses walk/n as an average m/kf.

d. :1995-2023 extracts DA3 per-frame anchors for frames still missing.

e. :2039-2041 imports `walk_length_m`, `plan_anchor_indices`, `plan_chunks`, `chunk_ranges`. Then `_build_vggtomega_config` (:1510-1536):
   - Base config is vendor/VGGT-Long/configs/stac_vggtomega.yaml: chunk_size 120, overlap 60, loop_chunk_size 20 (:12-14).
   - Overridden by reconstruction.vggtomega.chunk_size/chunk_overlap at :1522-1523 and loop_enable at :1524.
   - Returns through `_apply_stac_model_keys` (:1439).

f. Helpers defined inside the function:
   - `_persist_chunk_plan(_chunk,_ov,_n_kf,_phase,_walk)` at :2064-2084. Plan schema: {version 1, phase, n_keyframes, chunk_size, overlap, chunk_ranges = chunk_ranges(n,size,ov), walk_m}. It calls `_invalidate_on_new_chunk_plan` and then writes output/chunk_plan.json.
   - `_apply_chunked_metric(cfg_v,_chunk,_ov)` at :2086-2204 sets:
     - Model.chunk_size and Model.overlap (:2087-2088)
     - loop_enable True, using_sim3 False
     - the metric_lock block (anchor_dir, sigma_seam, sigma_anchor, …)
     - adjustment flags
     - loops / scale / graph / authority / certify plus Loop.SALAD (fork_loop_salad)
     - anchor_extract, absolute_rows, VIO
   - It does NOT write `loop_chunk_size`. That value stays at 20 from the YAML; the vendor reads it at vggt_long.py:3244. The only server write of loop_chunk_size is `_build_da3_config` at :2978 (DA3 path).
   - `_ensure_anchors` at :2206. `_omega_pass` at :2213 writes output/vggt_omega_config.yaml and runs run_mapanything.sh → vggt_long.py, then `_postprocess_reconstruction`.

g. Capacity (:2292-2322):
   - `_chunk_cfg = chunk_frames` (0).
   - `_free = _gpu_total_gb()`: the card's TOTAL memory, not free memory.
   - If explicit, `_cap = chunk_frames` and it only warns when the card is short.
   - Otherwise `_per_frame = 0.086 * (gw*gh)/(464*832)` (grid from `_omega_grid_wh`) and `_cap = max(24, int((_free-4.0)/_per_frame))` (:2316).
   - Finally `_chunk_cfg = _cap`.

h. Inputs: `_max_walk0` = max_walk_single_pass_m (15.0, config.yaml:1730), `_cw0` = chunk_walk_m (15.0, config.yaml:1743), `_walk0` = I3 walk, `_pin0` = chunk_frames_over_walk (0, config.yaml:1773).
   :2332-2333 `_chunk_it = walk known AND scale_align AND NOT (n <= cap AND (max_walk<=0 OR walk<=max_walk))`.

i. Branches:
   - **A, walk-planned** (:2334-2377):
     - If pinned, size = max(24, min(pin, cap)). Otherwise `plan_chunks(n, walk0, cw0, max_size=max(cap,24))`. Overlap is size//2.
     - Calls `plan_anchor_indices`, `_ensure_anchors` and `_apply_chunked_metric`.
     - SALAD `min_gap` = ceil(visit_drift.min_walk_m / (walk0/n)), using the average m/kf.
     - Writes the revisit reference, then `_persist_chunk_plan(...,\"walk-planned\", walk0)`.
   - **B, single pass** (:2377-2401): n <= cap. chunk_size = max(n,2), overlap 0, loop off, chunk_plan.json unlinked (:2383), intra_chunk flag set.
   - **C, pinned without a walk** (:2402-2437): fixed size from chunk_frames_over_walk, half overlap. SALAD band = ceil(min_walk_m*_fx/_cw). `_persist_chunk_plan(\"pinned-chunked\")`.
   - **D, strided walk probe** (:2438-2470): n > cap and no walk. stride = ceil(n/cap), one chunk over the probe, chunk_plan.json unlinked (:2463).
   - **E, scale_align off** (:2470-2475): cap with cap//2 overlap, no plan persisted.

j. :2476 runs `_apply_conf_filter`, :2479 the first `_omega_pass`. Then `_metricize_and_orient` (:2491-2569):
   - `_emit_omega_depth(chunk_size, overlap)` (:2496-2500)
   - scale_align, plus the verifier in chunked-metric mode (:2521-2535)
   - orient
   - `walk_length_m(camera_poses.txt)` (:2546), which is evidence only
   - deletion of `_tmp_results_aligned` unless certify.keep_aligned_chunks

k. :2575-2584 stamps the Omega walk into chunk_plan.json as `walk_m`.

l. Phase-2 re-run (:2604-2697):
   - Guarded at :2622-2623: `_simple_on and not _chunked_already and _scale_align_on and _walk_doc is None and (_probe_sel or walk_m > max_walk)`. It only fires when there is no I3 walk.
   - `plan_chunks(n, Omega walk_m, chunk_walk, max_size=max(cap,24))` (:2627).
   - Deletes chunk PLYs/origins/meta and maplong_run (:2657-2667), rebuilds the config, `_apply_chunked_metric`, SALAD band, `_persist_chunk_plan(\"chunked-metric\")` (:2688), runs the second pass, then `_metricize_and_orient`.

m. `_invalidate_on_new_chunk_plan` (:2718-2768):
   - The comparison is ALREADY range-based: `old.chunk_ranges == plan.chunk_ranges and n_keyframes` equal (:2738-2741).
   - Only the log message (:2747-2749) formats `chunk_size/overlap`.
   - Kept on a new plan: `_PLAN_INDEPENDENT_OUTPUT` (:2712) and `_PLAN_INDEPENDENT_MAPLONG` (:2714).

n. reconstruction/chunk_plan.py:
   - `walk_length_m` :22
   - `plan_chunks` :37-70: size = round(chunk_walk_m / (walk/n)), clamped to [min 24, max 150 or the caller's max_size], overlap size//2. A single uniform size.
   - `chunk_ranges` :73-87: step = size-overlap, last chunk clipped. It is equivalent to the vendor's `num_chunks = (N-ov+step-1)//step` at vggt_long.py:3218-3231.
   - `plan_anchor_indices` :90-106: picks from `chunk_ranges(n,size,ov)` and would need a ranges-taking variant.

Vendor assumptions (out of scope, but they block variable ranges):
- vggt_long.py:3218-3231 builds `chunk_indices` from chunk_size/overlap.
- Seam alignment slices `[-self.overlap:]` / `[:self.overlap]` at :1894-1897 and :3336-3344.
- Ensemble ranges at :2625-2629.
- The resume guard compares per-chunk frame counts at :321-340.

== 2. CONSUMERS (field read → assumes uniform?) ==

UNIFORM size/overlap (would break with variable ranges):
1. map_worker.py:1539 `_emit_omega_depth(chunk_size, overlap)`:
   - :1550 step, :1580-1584 rebuilds `_chunks` uniformly, :1616 `start = k*step`.
   - It writes the per-keyframe record field `chunk` (the owner by nearest centre).
   - Precision, floor and others depend on this field, so it has to take explicit ranges.
2. map_worker.py:3087 `_generate_origins`:
   - :3139-3141 `chunk_step = Model.chunk_size - overlap`.
   - The fallback path computes `abs_idx = frame_local + K*chunk_step` (:3295).
   - chunk_NNN_meta.json is written with `chunk_step` (:3210 inline path) and `chunk_step`, `frame_global_start = K*chunk_step`, `frame_global_end` (:3335-3337).
   - The inline-origins path takes `frame_global` from the vendor, so it is correct; only the meta is uniform.
3. segmentation/tsdf_export.py:305 `_resolve_mapanything_depth`:
   - `chunk_step` comes from chunk_*_meta.json (:352-358), or from the vggt_omega_config.yaml size-overlap (:368).
   - `K = min(pos // chunk_step, n_chunks-1)`, `local = pos - K*chunk_step` (:417-418).
   - Used by tsdf_export:1186 and :2064, reconstruction/mv_consistency.py:125-128 and reconstruction/native_depth.py:272-273.
4. reconstruction/trace_normals.py:101 `_load_frame_depth_index`: reads Model.chunk_size/overlap from output/vggt_omega_config.yaml (:120-126) and computes `start = ci*step` (:129). Used by trace_normals:240 and surface_fit/consolidate.py:523 (cloud consolidation normals).
5. reconstruction/quality/ab_elastic.py:57-58 (A/B harness) uses Model.chunk_size/overlap, but takes chunk_indices from the caller.
6. Legacy, not on the Omega path: frames/storage.py:149-150 and :658-670 (meta from config chunk_size); main.py:902-903 CHUNK_SIZE/CHUNK_OVERLAP, which is only logged at :1131.

RANGE-BASED (already works with variable sizes, assuming only adjacent chunks overlap):

| Consumer | What it reads / does | Uniform? |
|---|---|---|
| correction/units.py:26-44 `load_chunk_plan` | Validates chunk_ranges as a list of [a,b]. ALSO requires `int(chunk_size)`, `int(overlap)`, `int(n_keyframes)` (:38). A variable plan must keep these keys or this check must change. | No |
| units.py:47-52 `chunks_of_keyframe` | Range membership. | No |
| correction/kfgraph.py:349,358,430-432 | `chunks_of_keyframe` + plot axvspan over chunk_ranges. | No |
| correction/consistency.py:195,203,245-247 | Same as kfgraph. | No |
| correction/photobundle.py:370,375 | `chunks_of_keyframe`. | No |
| correction/revisit.py:541,570-571 | `chunks_of_keyframe`. | No |
| correction/floor.py:603-635 `_blend_across_overlaps` | Reads chunk_plan.json directly (not via units). Overlap = [ranges[a+1][0], ranges[a][1]), consecutive chunks only. Owner comes from the records' `chunk` (floor.py:572-584). Needs the record chunk ids to index chunk_ranges. | No (adjacent pairs only) |
| reconstruction/certify/scale_stage.py:36-51 `chunk_of_keyframes` | chunk_ranges, or [(0,n)] without a plan, then vendor `frame_owner` (metric_lock.py:1192, nearest centre). | No |
| scale_stage.py:479-551 solve | Owner-based. Seams are `range(n_chunks-1)`, i.e. consecutive. The report echoes chunk_ranges. | No |
| scale_stage.py:634-665 `scale_transforms` | Owner-based pivots. | No |
| scale_stage callers | certify/run.py:406, correction/visit_drift_run.py:484, quality/known_answer.py:65. | No |
| reconstruction/loops/instance_loops.py:642-651 → loops/structural.py:303-353 | Membership of the median keyframe. | No |

Side note on structural.py: with NO plan, `write_absolute_rows` iterates `range(len(None or []))`, so rows without a chunk are dropped.

PRECISION reads the per-keyframe record field `chunk` (written by `_emit_omega_depth`), NOT chunk_plan.json:
- chunk_check.py:245-251. Seams at :439-452 are adjacent chunks in keyframe order, pooled by chainage.
- depth_on_f5.py:417-434 and :521-539 (per-chunk confidence floor and chunk PLYs).
- corrected_cloud.py:334-394.
- epoch0_cloud.py:137-183.
- mono_ab.py:144-160.

These are layout-agnostic, but only as correct as `_emit_omega_depth`'s ownership. gauge.py:691-695 reads maplong_run/metric_lock.json \"seams\" (vendor output).

DELETE / CLEANUP only:
- pipeline_manager.py:511,524 (Replace clears chunk_plan.json and chunk_*_meta.json).
- main.py:377 and cloudcompy_worker.py:266 delete chunk_* files.
- main.py:1958-1967 counts cached chunk_*.npy for resume.

chunk_*_meta.json readers that ignore the layout: segmentation/pipeline.py:1014 and :2814 (resolution fields only); bim/occlusion_raycaster.py:69-87 (\"cameras\", which map_worker never writes).

== 3. TESTS PINNING THE CURRENT PLANNING ==

**server/tests/test_chunked_metric.py**
- :33 `walk_length_m`.
- :44 `chunk_ranges` == the vendor formula for 7 (n,cs,ov) cases.
- :57 `plan_anchor_indices` gives at least 2 picks per chunk.
- :217 `frame_owner` over chunk_ranges: every frame owned once, owner[mid] == k.
- :987 config chunk_frames == 0, AND the source contains `\"_cap = max(24, int((_free - 4.0) / 0.086))\"` after stop_semantic_service. STALE: the source now reads `/ _per_frame` (:2316), so the string occurs 0 times and this test fails as written (not run).
- :1000 `chunk_ranges(n, max(n,2), 0) == [(0,n)]`.
- :1008 `chunk_ranges(1200,500,250)`: overlap is half, no chunk exceeds the capacity, last end is 1200.
- :1017 every capacity-layout chunk gets anchors.
- :1027 in the source, `_cf_anchor = ...` comes before `_run_da3_anchor(...sorted(set(_missing))`.
- :1040 config: max_walk_single_pass_m == 15.0, chunk_walk_m > 0, chunk_frames_over_walk >= 0, scale_anchor_frames == 0, chunk_frames == 0.
- :1055 `plan_chunks(216,43.7,12.0) == (59,29)`; `(600,80,12)` gives ov = size//2 and about 12 m; clamps to 150 (default max) and 24.
- :1071 source order \"DA3 metric anchor on ALL\" < cap string < \"[chunk-plan] measured walk:\" < \"re-run CHUNKED at\", and exactly 2 `if not _omega_pass(`. STALE too: it uses the same missing cap string.
- :1084 the exact source text of the re-run guard (:2622-2623), `_max_walk > 0`, `_phase2, _ov2 = 0, 0`, `if _phase2 >= _n_selected and not _probe_sel:`.
- :1100 `chunk_ranges(216,60,30)`: first (0,60), 7 chunks, anchors in each.

**server/tests/test_intake_walk.py**
- :78 `plan_windows` (DA3 windows, not chunks): uniform window width, full coverage, at least the planned overlap.
- :142 `plan_chunks(1329,104.8,12.0,max_size=870)` is deterministic, ov == size//2, about 12 m within 0.1, and the last `chunk_ranges` end is 1329.

**server/tests/test_chunk_plan_invalidation.py**
- :13 the `_plan` fixture builds {version, phase, n_keyframes, chunk_size, overlap, chunk_ranges via chunk_ranges()}.
- :39 the same plan wipes nothing.
- :46 296/148 → 293/146 wipes the listed products and keeps the listed set. The log must contain \"NEW PLAN\", \"296/148\" and \"293/146\"; that text comes from chunk_size/overlap.
- :64 chunks with no recorded plan are wiped; a fresh run is untouched.
- :75 the vendor source has `class _StacPlanMismatch` and `except _StacPlanMismatch:\
 raise`.

**server/tests/test_omega_emit.py**
- :20 single chunk (CS=S, OV=0): record keys, chunk 0, frame_global.
- :51 N=12, CS=4, OV=2, chunks built uniformly in the test: the record `chunk` is the nearest-centre owner. Pins the signature `_emit_omega_depth(save, out, chunk_size, overlap, sel, pipe)`.

**Other tests**
- test_floor_chunk_blend.py:17-47: the plan holds ONLY chunk_ranges [[0,40],[20,60]]. 20 keyframes are blended; outside the overlap each chunk keeps its motion; no plan means no change.
- test_correction_units.py:28-55: no plan → None/[]; a plan from chunk_ranges(100,40,20) round-trips; `chunks_of_keyframe(25)` == [0,1]; corrupt plan → RuntimeError; :57 regex H1 forbids fixed divisors.
- test_correction_depth_after_gauge.py:
  - :39 PLAN [[0,20],[10,30]] with size 20, overlap 10, n_keyframes.
  - :112 and :275 `chunk_of_keyframes`.
  - :215-238 pccr 83/41 six-range plan.
  - :242-262 `scale_transforms` over ranges [(0,20),(10,30),(20,40)].
  - :388 PCCR_PLAN.
- test_graph_f2.py:173-178 regulated rows and `write_absolute_rows` with explicit chunk_ranges.
- test_pipeline_auto_chain.py:172-173: the source slice from `def _apply_chunked_metric` to `def _ensure_anchors` must contain \"fork_loop_salad\".

**Synthetic generators**
- synth_metric.py:328-339 has its own uniform `chunk_ranges`, used at :385.
- synth_metric.py:853-895 `write_aligned_chunks` writes a uniform chunk_plan.json and chunk_000_meta.json `chunk_step`, which the tsdf_export loader reads in witness/depth tests.
- synth_correction.py:229 and :381 pass through an optional chunk_plan.
- reconstruction/quality/adversarial.py:70 calls write_aligned_chunks(24,12).

== WHAT A VARIABLE-RANGE PLAN TOUCHES FIRST ==
- The vendor must accept explicit chunk_indices, with the seam overlap per pair instead of `self.overlap`.
- `_emit_omega_depth` (it writes the `chunk` field every precision and floor stage trusts).
- `_generate_origins` meta (`chunk_step`, `frame_global_start`).
- tsdf_export `_resolve_mapanything_depth`.
- trace_normals `_load_frame_depth_index`.
- `plan_anchor_indices` needs a ranges input.
- units.py:38, which requires chunk_size and overlap keys.
- The two stale source-text tests (test_chunked_metric.py:987 and :1071).

The per-keyframe chainage needed to cut ranges at fixed metres already exists in intake/walk.json (walk.py:398).",
      "signals": "PRE-OMEGA SIGNALS FOR A VARIABLE CHUNK PARTITION (VGG-T3 co-visibility + VGGT-Motion motion). Read-only investigation, no files modified.

DATA CAVEAT: every session on disk is currently wiped to frames + video (`server/projects/{pccr,zaragoza}/scans/*/src_default/output` is empty, and no `walk.json` exists). So the numbers below come from logs/pipeline logs (pccr 2026-08-24: 1998 kf, 153 windows of 26, process_res 840, walk 102.3 m, seam disagreement median 0.9 cm / max 55.8 cm) and from capped synthetic benchmarks.

1. METRIC c2w + METRIC DEPTH + K FOR EVERY KEYFRAME, ON ONE FRAME, BEFORE OMEGA
- Order: `map_worker.py:1972-1994` runs I3 (`run_da3_windows`) and `measure_walk` BEFORE the chunk decision (`map_worker.py:2318-2378`) and before Omega. It is gated on `_scale_align_on` + `precision.enabled` (`map_worker.py:1979-1982`).
- Window files: `output/da3_windows/window_NNNN.npz`, written at `extract_da3_depth.py:247-252`. They hold:
  - `frames`
  - `depth` (S,H,W) metric, multi-view aligned to the NESTED metric branch
  - `depth_mono` (per-frame monocular metric, `:230-238`)
  - `conf` = clip(conf−1, 0) (`:227`)
  - `extrinsics` w2c (S,3|4,4), translations metric
  - `intrinsics` (S,3,3) on the DA3 grid
  - `scale_factor`, `is_metric`
- Getting the poses:
  - `walk.load_window` (`walk.py:100-106`, converts to c2w via `_to_c2w` `walk.py:85-97`) feeds `walk.chain_windows` (`walk.py:119-159`), which returns `{frame: c2w}` for ALL keyframes in window 0's frame: metric, not gravity-aligned.
  - Placement is RIGID (chordal-mean R + mean t, `walk.py:140-145`). The inter-window scale ratio is only reported (`span_ratio`, `walk.py:149-155`), never applied.
  - A frame keeps the pose of the FIRST window that holds it (`walk.py:156-158`).
- Depth + K per keyframe: `output/da3_run/results_output/frame_<num>.npz` (depth, conf, grid intrinsics), written by `write_anchors` (`walk.py:172-195`). Each frame's depth comes from the window where it sits MOST CENTRALLY. So depth window ≠ pose window for most frames, and they differ by that window pair's scale ratio. This is small on a healthy seam but wrong across a broken seam (55.8 cm max seen on pccr).
- What survives window deletion (deleted after F2 at `precision/runner.py:230-237` and at chain end `:240-244`; `walk.py:342-358`):
  - anchors frame_*.npz (depth/conf/K)
  - `windows.json`
  - `intake/walk.json`, which holds only chainage per frame, window ranges + scale_factor, and seams (`walk.py:379-401`). NO poses.
  - `output/salad_revisit_reference.json`: centres + forward (z axis) per keyframe, dist_bar_m, cos_bar, hfov (`walk.py:419-458`). NO roll, so no full R.
- Gap: on a fresh run the windows exist before Omega, so full c2w comes from `chain_windows`. On a resume (`walk_is_current` → `map_worker.py:1984-1987`, `walk.py:299-339`) the windows may be gone, and full c2w is NOT recoverable without re-running I3. The log shows 153 windows ≈ 28 min GPU (23:36 → 00:04 on 2026-10-05). Persisting the chained c2w in `measure_walk` (1998×4×4 float64 ≈ 128 KB) would close this. Not done; your call.
- DA3 grid: the whole frame is resized (`upper_bound_resize`, `depth_anything_3/api.py:146`, then a divisible-by-14 RESIZE at `input_processor.py:239`, `:365`), so there is no crop and the grid↔native mapping is exact. `process_res` \"native\" = long side rounded up to a multiple of 14 (`walk.py:214-228`). pccr 464×832 → 840 → grid ≈ 462×840. zaragoza 1920×1080 → 1932 → ≈ 1932×1092.
- Session K (native) is in `intake/focal_probe.json`: K, fx..cy, `grid_wh`, `native_wh` (`focal.py:133-142`). Co-visibility can stay entirely on the DA3 grid with each anchor's own K, so no conversion is needed.

2. VGG-T3 CO-VISIBILITY BY DEPTH CONSISTENCY: RECIPE AND COST
- Recipe:
  - For i: unproject a stride-s grid of the anchor depth, keeping conf>0 (or the 10 % min-max floor), with chained c2w_i. `precision/chunk_check.py:60-68 unproject(depth, K, c2w, stride)` already does exactly this.
  - Project into j with w2c_j and K_j (`precision/flyers.py:162 project` → u, v, z).
  - Keep points that land in the frame with z>0.
  - Look up z_j in j's depth subsampled at 1/q; count a point consistent when |z−z_j| ≤ tol·z_j. A point with z > z_j(1+tol) is occluded (excluded); z < z_j(1−tol) contradicts.
  - covis(i,j) = consistent / samples_i, symmetrised with min or mean. Grouping follows VGG-T3: join a group when covis > 0.3 with any member.
  - Existing tolerance in the repo: max(0.10 m, 0.10·z) (`config.yaml:3176-3191`, `correction.revisit`).
- Cost for 1998 kf with j in [i+1, i+400]: 719,000 pairs. Synthetic numpy benchmark, 1 core, capped (8 cores, nice, ulimit):
  - i-samples 1456 (stride 16), j-map 1/8 (105×57): 173 µs/pair → about 124 s total; j-maps 48 MB in RAM.
  - i-samples 5985 (stride 8), j-map 1/4: 717 µs/pair → about 516 s; 193 MB.
- Loading the 1998 compressed anchors (synthetic files): pccr grid ≈ 17 ms each → about 34 s; zaragoza 1080p ≈ 83 ms each → about 166 s.
- Verdict: about 2–3 min on one core at modest sampling; it parallelises trivially over i (2–4 processes) and stays within the pod caps. A cheap pair prefilter from centres + forward (camera distance and the half-FOV angle test of `vendor/VGGT-Long/LoopModels/calibration.py:24-31`) would remove most far, non-facing pairs first.

3. MOTION SIGNALS (VGGT-Motion) ALREADY PERSISTED
- `frames/selected_frames.json` (`parallax.py:762-782`):
  - `keyframes[]` {frame, file, anchor (= previous keyframe), parallax_px, closed_by ∈ quantum | track_loss | coverage_break} (`parallax.py:480-482`, `529-531`)
  - `windows[]` {anchor, anchor_is_keyframe, closed_by, n_frames = raw frames between keyframes (a static-redundancy measure), keyframe, max_parallax_px} (`parallax.py:521-527`)
- `intake/coverage_warnings.json` (`parallax.py:799-808`):
  - `frames[]`: every usable frame's FrameMeasure {frame, anchor, n_tracks, disp_px (median track displacement), parallax_px (0.9-quantile of residual after the pure-rotation fit), fb_px, lost, lost_reason} (`parallax.py:111-125`) plus flags static / pure_rotation / tracking_lost (`parallax.py:581-588`, assembled at `706-711`).
  - Each frame keeps its last measurement, made against its window's anchor (the previous keyframe).
  - `warnings[]`: runs of ≥5 frames of static / pure_rotation / tracking_lost / exposure (`parallax.py:590-617`).
- `frames/quality_features.json`: per frame fft, laplacian, luma, clip_frac, inter_frame_diff (legacy motion), sharp_rank (`quality.py:108`, `218-223`, document `330-350`).
- NOT persisted: the fitted camera rotation per frame. `fit_rotation` returns the rotation vector (`parallax.py:330`), but it is kept only as a warm start (`self.x`, `parallax.py:383`, `434`). Its |w| at each keyframe would be the LK turn angle from the previous keyframe, a pre-DA3 turn signal; it is a one-field addition to FrameMeasure. Not done.
- Turns from DA3:
  - With windows on disk: angle between consecutive chained c2w, arccos((tr(Rkᵀ·Rk+1)−1)/2).
  - Without windows: angle between consecutive `forward` vectors in `salad_revisit_reference.json` (yaw+pitch, no roll).
  - Divide by Δchainage (`walk.json` chainage, `walk.py:395`) for deg/m. m/kf from the same chainage gives the static-density signal.
- Chain quality per seam: `walk.json` `seams[]` {n_shared, centre_disagreement_median_m / max_m, span_ratio} (`walk.py:152-155`). A bad seam is itself a candidate cut.

4. HELPERS TO REUSE
- Poses / walk: `intake/walk.py:85 _to_c2w`, `:100 load_window`, `:111 _chordal_mean`, `:119 chain_windows`, `:162 walk_of`, `:172 write_anchors`, `:419 revisit_reference`.
- Projection: `precision/chunk_check.py:60 unproject`, `precision/flyers.py:162 project`, `precision/flyers.py:172 view_votes` (per-point agree/contra by relative depth over neighbouring keyframes, the same test as above), `correction/revisit.py:94 _project`.
- `correction/revisit.py:103 covisibility`: a full z-buffer, depth-consistent, occlusion-aware co-visibility matrix (cell 8 px, tol max(0.10 m, 0.10·z), sample_per_kf 1500). It takes a post-Omega CorrectionSession cloud; the algorithm ports directly to DA3 anchors + chained poses.
- Pair prefilter: `vendor/VGGT-Long/LoopModels/calibration.py:7-31` (centres + forward, dist_bar + cos_bar).
- K / grids: `precision/camera.py:165 grid_full_frame_resize`, `:181 grid_to_native`, `:190 native_to_grid`, `:199 K_grid_to_native`, `:211 K_native_to_grid`, `:223 grid_valid_mask`, `:524 rescale_K`; `intake/focal.py:53 native_intrinsics`.
- Integration point for a variable partition:
  - The vendor slices FIXED-step chunks (`vendor/VGGT-Long/vggt_long.py:3221-3231`).
  - `reconstruction/chunk_plan.py:73 chunk_ranges`, `:90 plan_anchor_indices`, and `map_worker.py:2064-2084 _persist_chunk_plan` / `:2086 _apply_chunked_metric` all assume size + overlap. `chunk_plan.json` already stores an explicit `chunk_ranges` list.
  - So variable ranges need the vendor to accept explicit `chunk_indices`.

Benchmark scripts (scratch only): `/tmp/claude-0/-workspace-stac-build/448cd311-7766-4e9d-b6c6-746bda2501e6/scratchpad/covis_bench.py` and `/tmp/claude-0/-workspace-stac-build/448cd311-7766-4e9d-b6c6-746bda2501e6/scratchpad/load_bench.py`"
    }
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Understand"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Design"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "map:fork",
      "phaseIndex": 1,
      "phaseTitle": "Understand",
      "agentId": "a35485201cf691d40",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1791264252419,
      "queuedAt": 1791264252405,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "sed -n 613,662p loop_utils/metric_lock.py",
      "promptPreview": "POD RULES (hard): read-only investigation — do NOT run the pipeline, GPU jobs, or heavy scripts. If you run any python, cap it: (ulimit -v 30000000; OMP_NUM_THREADS=2 timeout 300 taskset -c 0-7 nice -n 15 /workspace/miniforge3/envs/da3/bin/python ...). Do NOT modify any file. Repo: /workspace/stac-build (server/, vendor/VGGT-Long/). Report file:line references.

TASK: map how the Omega fork (vendo…",
      "promptFramed": true,
      "lastProgressAt": 1791264588620,
      "tokens": 237343,
      "toolCalls": 45,
      "durationMs": 336197,
      "resultPreview": "CHUNK-LAYOUT MAP: VGGT-Long fork (vendor/VGGT-Long), read-only. No files were changed and nothing was run.

Bottom line: only five places in the fork assume one fixed chunk size and overlap. Everything else already reads each chunk's real (start, end) from self.chunk_indices. On top of the obvious range checks, the variable layout needs one extra rule: a frame may belong to at most two chunks. Sev…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "map:server",
      "phaseIndex": 1,
      "phaseTitle": "Understand",
      "agentId": "afb65e84616b05635",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1791264252420,
      "queuedAt": 1791264252405,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /workspace/stac-build/server/tests && sed -n 39,41p test…",
      "promptPreview": "POD RULES (hard): read-only investigation — do NOT run the pipeline, GPU jobs, or heavy scripts. If you run any python, cap it: (ulimit -v 30000000; OMP_NUM_THREADS=2 timeout 300 taskset -c 0-7 nice -n 15 /workspace/miniforge3/envs/da3/bin/python ...). Do NOT modify any file. Repo: /workspace/stac-build (server/, vendor/VGGT-Long/). Report file:line references.

TASK: map the SERVER side of chunk …",
      "promptFramed": true,
      "lastProgressAt": 1791264598061,
      "tokens": 210191,
      "toolCalls": 69,
      "durationMs": 343804,
      "resultPreview": "SERVER-SIDE CHUNK PLANNING: CONTROL FLOW, CONSUMERS AND TESTS (read-only; nothing run, nothing modified)

Short answer: chunk_plan.json already stores explicit `chunk_ranges`, and most of correction/ and certify/ uses only those ranges. Five places still rebuild the layout as `k * (chunk_size - overlap)`: the vendor slicing, `_emit_omega_depth`, `_generate_origins`, the tsdf_export depth loader an…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "map:signals",
      "phaseIndex": 1,
      "phaseTitle": "Understand",
      "agentId": "a1d19a0f8251440f2",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1791264252421,
      "queuedAt": 1791264252405,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /workspace/stac-build/vendor/depth-anything-3/src/depth_…",
      "promptPreview": "POD RULES (hard): read-only investigation — do NOT run the pipeline, GPU jobs, or heavy scripts. If you run any python, cap it: (ulimit -v 30000000; OMP_NUM_THREADS=2 timeout 300 taskset -c 0-7 nice -n 15 /workspace/miniforge3/envs/da3/bin/python ...). Do NOT modify any file. Repo: /workspace/stac-build (server/, vendor/VGGT-Long/). Report file:line references.

TASK: inventory the signals availab…",
      "promptFramed": true,
      "lastProgressAt": 1791264588018,
      "tokens": 171232,
      "toolCalls": 46,
      "durationMs": 333748,
      "resultPreview": "PRE-OMEGA SIGNALS FOR A VARIABLE CHUNK PARTITION (VGG-T3 co-visibility + VGGT-Motion motion). Read-only investigation, no files modified.

DATA CAVEAT: every session on disk is currently wiped to frames + video (`server/projects/{pccr,zaragoza}/scans/*/src_default/output` is empty, and no `walk.json` exists). So the numbers below come from logs/pipeline logs (pccr 2026-08-24: 1998 kf, 153 windows …"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "design:synthesis",
      "phaseIndex": 2,
      "phaseTitle": "Design",
      "agentId": "a56a0cb489143fbc1",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1791264601102,
      "queuedAt": 1791264601097,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /workspace/stac-build/server/projects && for p in */scan…",
      "promptPreview": "You design VARIABLE-SIZE CHUNKING for the STAC reconstruction pipeline (VGGT-Omega chunked by the VGGT-Long fork). Do NOT modify files; do NOT run anything heavy. Produce an implementable design the user can approve.

CONTEXT (user decisions, measured facts):
- No fixed chunk size fits all scenes. Measured: pccr (indoor, targets ~1.4 m away, camera walks sideways facing the targets): 5 m chunks va…",
      "promptFramed": true,
      "lastProgressAt": 1791265216557,
      "tokens": 142622,
      "toolCalls": 6,
      "durationMs": 615455,
      "resultPreview": "VARIABLE-SIZE CHUNKING: CO-VISIBILITY-PLANNED OMEGA CHUNKS
(Design only. No files changed, nothing run. Every default listed under \"NEEDS THE USER'S OK\" waits for his OK.)

Core idea in one line: measure, before Omega, how many keyframes each keyframe stays co-visible for (the VGG-T3 depth-consistency test). Add up the inverse of that length along the walk as a \"motion budget\" (VGGT-Motion). Cut t…"
    }
  ],
  "totalTokens": 761388,
  "totalToolCalls": 166
}