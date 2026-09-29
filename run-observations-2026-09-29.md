---
name: run-observations-2026-09-29
description: "Open observations from the first end-to-end F0-F7 run (pccr 2026-08-31, 2026-09-29) to review with the user once it finishes — exclusion prompts, chunking as a measurement, stale config keys"
metadata:
  node_type: memory
  type: project
  originSessionId: 753c7b1a-0d98-4b5c-9b35-35dddd408b32
  modified: 2026-09-29T02:27:14.183Z
---

Observations the user asked to park ("anotalo … para revisar luego cuando
termine") during the first full "Reconstruir" with F0-F7 inside the
reconstruction stage (pccr scan 2026-08-31, 289 kf / 2100 witness, walk 17.5 m):

1. **Exclusion prompts would erase a parked train.** `intake.content.prompts.
   dynamic` = [person, worker, moving vehicle, animal, train]. In the depot scene
   (a stopped train IS the geometry) the VLM tags the keyframe `dynamic`, SAM3
   masks the train, F4/F6/F7 EXCLUDE it. Proposal (config only): keep exclusion
   for person/worker/animal + occluders; drop `moving vehicle` and `train`; the
   VLM `dynamic` tag stays as evidence in content_tags.json. USER: "no me lo vas
   a excluir eh!" — decide before running that scene.
2. **15 m / 12 m (`max_walk_single_pass_m`, `chunk_walk_m`) are declared, not
   measured.** Evidence: same keyframe count (213-255) broke on pccr 19 m and
   worked on observatorio 11.2 m / test2 12.9 m → neither frames nor metres is
   the mechanism; VGGT-Long's FAQ: drift grows with hops at small baseline.
   Proposal "coherence probe" (F3-bis): Omega on nested windows L = 32/64/128/256
   kf of the same stretch, aligned to the I3 DA3 walk (local error only, seam
   median 1.3 cm); chunk = largest L whose drift stays within the walk's own
   noise (heldout_change bootstrap). Removes both constants.
3. **`reconstruction.simple.chunk_frames_over_walk: 152`** claims to pin the
   chunk size in keyframes and ignore `chunk_walk_m`, yet the walk-planned run
   used 198 kf (12 m). One of the two descriptions is stale — verify plan_chunks.
4. **`mask_sky: true` ran on an indoor server room** (388/388 frames masked);
   measured cost on pccr 1.7 % of pixels (worst frame 12 %). Declared default;
   the user may want it off indoors.
5. VLM tags on this scene: dynamic 0, occluder 8, reflective 72, low_info 0; the
   intake's SAM3 pass wrote 0 exclusion masks. SAM3 3.1: 13 fps per session
   (first session 200 s = model load).

6. **single_witness dropped 48.9 %** at the chunk merge (41.5 M of 84.9 M pts;
   2 chunks of 198 kf = 12 m, 289 kf / 17.5 m = 0.06 m/kf). Reference points:
   60/30 on pccr (0.087 m/kf) dropped 11.6 %, ONE chunk 60.1 %. Denser keyframes
   + longer chunks → the frames agree less on where surfaces are — the
   VGGT-Long FAQ mechanism (drift per hop at small baseline). Evidence for the
   coherence probe (item 2) and a question for the user: is a 12 px quantum too
   dense for Omega? (parallax_quantum_px is the knob that sets hops per metre.)
7. **The in-stage chunk merge (inline cloudcompy worker) also ran the deferred
   mask→cloud projection** onto the WORKING cloud (167 instances on 21 M points)
   — wasted: F7 replaces that cloud and the cloud stage projects again. Fix:
   the merge config should skip the projection (a `postprocessing` flag read by
   workers/cloudcompy_worker.py). Harmless for correctness (the later
   projection sees a newer cloud and re-runs).
8. Merge numbers: voxel 5 mm 43.3 M → 21.3 M, SOR 2.0 % (mean d 6.8 mm σ 2.4),
   final 20.85 M pts / 616 MB; scale_align s = 1.0196 after the metric lock.

9. **F0 vs the intake's focal probe disagree by 7 %**: camera.json from Omega's
   per-frame K median fx 366.3 / fy 370.3; intake/focal_probe.json (DA3 nested,
   16 frames) fx 394.8 / fy 408.6. Same sensor, two instruments. F5's rungs R1
   (fx, fy, cx, cy) decide by held-out reprojection — check refine.json's
   `camera.after` against both when the run ends.
10. **F2 gauge on this run**: held-out log RMS da3_windows 0.0416 vs da3_mono
    0.0808 → da3_windows applied ("better beyond the noise"); 0 relative rows
    (no instances before F7 — by design now).

11. **F5: the camera stays as Omega estimated it** (fx 366, fy 370, no lens
    distortion). F5 tried to also adjust the camera and judged each try on the
    20 % of tracks it never used (held-out reprojection error):
    R0 camera fixed 0.605 px (best) · R1 +focal/centre 0.645 (worse) ·
    R2 +distortion 0.774 (worse) · R3 focal per time block — cost 2.44 px vs
    0.76. Every camera adjustment made the error worse → only camera positions
    and 3-D points are refined. Also settles item 9: Omega's fx 366 beat the DA3
    probe's 395. Deterministic: same numbers bit for bit on two runs.
12. **F3 resolution probe**: Omega at 512 disagrees least (median 0.38 %),
    768 1.25 %, 1024 2.28 % — more pixels is worse (checkpoint trained at 512).
    Production already at 512: nothing to change.
13. **Pipeline stopped at F5 on the first run**: the old epoch publish built a
    Potree octree inside the refine step's env (mapanything), which lacks
    `laspy`. Fixed by design (F2/F5 publish poses only, commit 2904c34); chain
    resumed by hand from F5.

14. **F6/F7 ROOT CAUSES (audit 2026-09-29, three lenses, file:line):**
    a. **Omega's chunked poses over-measure the walk by 80 %**: epoch 0 walk
       31.75 m vs DA3 windows 17.47 m; chunk 1's baselines 2.5× too long.
       WHY THE CHUNKS WERE TOO BIG: `chunk_walk_m: 12` was calibrated on the
       INFLATED Omega walk (pccr 44 m → 12 m inflated = 59 kf = the accepted
       60/30 layout; plan_chunks docstring says so). F2 switched the planner's
       input to the REAL DA3 walk (17.5 m) → 12 real metres = 198 kf, 3.4× the
       layout that worked, same constant. `chunk_frames_over_walk: 152` is
       ignored on the walk-planned branch (map_worker.py:2196-2240 vs :2388).
       Immediate option: chunk_walk_m ≈ 5 m of REAL walk (pccr's accepted
       0.087 m/kf × 60); the coherence probe (item 2) is the measured answer. The
       gauge (F2) does not touch it (depth scale only: 31.15 m); **F5's BA fixes
       it → 18.17 m (+4 % vs DA3)**. F4/F5 work. Omega's depth maps stay metric
       per chunk (lock), so depth and poses were inconsistent in epochs 0/1 —
       why single_witness dropped 49 % and epochs 0 and 1 look layered.
    b. **F7 does not fuse** (fuse.py:169-192): every entering pixel of every
       keyframe becomes a point; dedup merges only within a 4 mm cube while the
       consistency tolerance that admitted the points is τ_rel·d = 1.9 % (5.8 cm
       at 3 m, 14× the voxel). Measured on epoch 3: 89 % of points share a 2 cm
       voxel with ≥ 2 keyframes; floor cells hold 12 keyframes, 53 mm vertical
       span (p90 160 mm), 3.5 points per 4 mm column → the layers.
    c. **Tier 1 = Omega's prior confirmed by Omega's ADJACENT priors**
       (depth_sweep.py:449, :897-904): 190/289 frames have both top-2 witnesses
       within ±3 keyframes → correlated, not independent; 94.7 % of the cloud;
       no Omega-confidence gate (every depth>0 pixel is a prior, :611); 22 % of
       points entered with exactly 2 such witnesses.
    d. **s_k per keyframe is an unsmoothed median** (depth_sweep.py:669-688),
       ignores the gauge; neighbour jumps median 1.1 %, max 27 % (38 pairs > 5 %)
       while the gauge's model says 0.25 % → each keyframe at its own depth
       scale → layers, staircase floor. Relative to F5's poses s_k is the right
       correction (0.71 at the walk's end where Omega's chain was 2.5× long),
       but it must be SMOOTH along the walk.
    e. **No contradiction test**: consistency() counts agreements only
       (depth_sweep.py:402); a view seeing free space through a point never
       counts against it → floaters and no way to pick one layer of a pair.
    f. **τ_px = 518 px** (depth_sweep.py:556: RMS of per-track held-out RMS, 2.3 %
       degenerate tracks up to 90 850 px; median 0.60 px) → the round-trip pixel
       test never rejects. Same bug in refine.py:491 heldout_bound → witness
       bound 106 px, 1556/1794 "localized".
    g. **Dedup order ignores tier** (fuse.py:71-74, :192): 87 901 tier-0 measured
       depths lost their voxel to a tier-1 prior with more self-agreeing views.
    h. **β capped at 0.10** (depth_sweep.py:811) while the landmark-calibrated
       prior error is 42 % q0.95 → 36 % of pixels hit the search boundary, tier 0
       collapses to 2.1 %; the confidence calibration then reads exactly
       1/(1−0.1)−1 = 0.111 in every cell — truncated by the cap, circular.
    i. **ZNCC null floor 0.787 is genuine** (images swapped, poses kept,
       frustum-disjoint pool :793-840): the real chance level of a 7×7 ZNCC on a
       464×832 @ 1.2 Mbit/s video. Lowering it would admit false matches. Tier 0
       is starved by the capture, not by the null.
    j. τ_rel 1.9 % is the median disagreement of Omega×s_k_i vs Omega×s_k_j —
       s_k noise judging s_k noise (F3 measured Omega's own mismatch at 0.38 %).
    k. Omega confidence, ZNCC, content_flags, residual_rel decide nothing in the
       fused output; origins v2 records `source` but the octree does not carry it
       (viewer hides tier 1 only via confidence == 0).
15. **CORRECTIONS APPLIED 2026-09-29** (see commits): consuming fusion with merge
    radius τ_rel·d and tier-first order (b, g); contradiction count and
    free-space rejection (e); robust τ_px and witness bound (f); s_k smoothed
    along the walk (d); Omega confidence floor on the prior (c, partial);
    beta_max raised to the measured quantile (h). Not changed: tier-1 adjacency
    (c) beyond the confidence floor and the contradiction test; τ_rel (j); the
    capture resolution (i).
16. **Epoch selection broken across a fused epoch**: run_select (correction/
    run.py:258-265) requires corrections/epoch_<e>.npz for every edge; F7's
    epoch is a new cloud, no npz can exist. Also select_epoch restores only from
    `_epoch_<target>/` although stored dirs are deltas (3→2 would strip poses;
    3→1 would leave no cloud). Fix in progress: `kind: new_cloud` epochs =
    identity edge + mandatory instance-store refit; path-aware swap. USER
    DECISION PENDING: production deletes `_epoch_*` after F7 (his 2026-09-28
    order) → nothing to compare in the viewer; keep `_epoch_0/` (≈ 2 GB, or its
    potree ≈ 1.4 GB) or not.
17. **USER VERDICTS 2026-09-29 in the viewer**: epoch 3 (F7) "muy mal —
    muchísimos duplicados, el piso es un desastre, duplicados y voladores"; the
    segmentation OBBs "un lindo desastre" (computed over the layered cloud);
    epoch 1 (gauge-warped Omega) "es un desastre" too. Epoch 0 shown next.
18. **F0 vs intake focal probe**: Omega K fx 366.3 / fy 370.3 vs DA3 probe fx
    394.8 / fy 408.6 (7 %). F5 judged by held-out: R0 (camera fixed) 0.605 px <
    R1 0.645 < R2 0.774; R3 cost 2.44 → Omega's camera stays. Deterministic bit
    for bit across two runs.
19. **F3 resolution probe**: 512 mismatch 0.38 % / 768 1.25 % / 1024 2.28 % →
    more pixels is worse (checkpoint trained at 512). Nothing to change.
20. **First run stopped at F5**: the old epoch publish built a Potree octree in
    the refine env (mapanything) which lacks `laspy`. Fixed by design: F2/F5
    publish poses only (2904c34); chain resumed by hand from F5.
21. **Rolling shutter** detected, declared, not corrected: Spearman 0.20, p≈0,
    828 k observations.
22. **Capture**: source video 464×832 @ 60 fps, 1.16 Mbit/s. "Native
    resolution" adds almost nothing over Omega's 688×384; precision scans need a
    higher-resolution, higher-bitrate capture.

23. **USER DECISIONS 2026-09-29 on the open items**: 2 → B (launch with
    chunk_walk_m 5 m; implement the Omega coherence probe during the run, for
    the next one) · 14c → A (keep tier-1 rule: 3 views + no contradiction +
    confidence floor; measure adjacency in this run before hardening) · 14j → A
    (τ_rel measured per run) · 14k → A (confidence slider hides tier 1) · 22 → A
    (noted) · 4 mask_sky stays · 16 keep `_epoch_0/`, selectable from the UI ·
    1 exclusion never targets trains/vehicles.

**Why:** each is a decision the user must take, not a bug to fix silently.
**How to apply:** raise them together when the run ends; implement only what he
approves. See [[user-wants-results-fast]].
