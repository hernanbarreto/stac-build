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

24. **NIGHT RUN 2026-09-29 — the door's layers, measured**: #125 (404 k pts) spans
    p05 −13 … p95 +11 cm along its normal, fed by the SAME 44 keyframes that see it
    — Omega's per-frame depth noise (±4 % at 2.5 m), not a revisit; each prior
    entered because its adjacent keyframes (sharing the error) agreed within
    τ_rel while the layers sit 10 cm apart, and the consuming fusion merges only
    within 2.5 cm. → F6 now (fd56ef0): contradiction over EVERY view whose frustum
    sees the point (best depth of each: swept where signalled, prior elsewhere);
    tier 1 needs an agreeing view ≥ correction.visit_drift.min_walk_m (1 m) of
    walk away (14c: measured, then hardened — the user's "medir primero" rule was
    honoured by this measurement); F7 runs the CloudCompy recipe (voxel + SOR,
    postprocessing params) inside, reasons voxel/sor. COST: the all-view
    contradiction made F6's consistency pass ~15× slower (289 keyframes × up to
    ~100 covisible views: ~90 min vs 6). A cost bound (max contradiction views
    by covisibility rank) is the next lever if it stays this slow.
25. **OBBs "not on the objects" — cause found**: opening the session in the
    viewer (main.py session-load fallback) ran a RANSAC floor leveling on the
    fused cloud — a baked-orientation session — found 2.4° / −28 cm and SAVED
    floor_transform.npz (10:06); the OBBs were then computed in that frame.
    Fixed (guard on .orientation_applied, identity, nothing saved); the wrong
    npz set aside as floor_transform.npz.wrong-ransac-20260929. (The raw
    OBB-centre-vs-centroid offsets I measured — median 28 cm, 5.5 m on an 11 m
    wall — are mostly legitimate: an OBB centre is the extent's midpoint, not
    the centroid.)
26. **Segmentation "muy incompleta" — FIRST DIAGNOSIS WAS WRONG (corrected
    2026-09-29 14:xx)**: prompt_search_frames does NOT limit detection — SAM3's
    text prompt is applied to ALL frames (find_text_batch, "to be applied to
    *all* frames"), and every add_prompt RESETS the session
    (sam3_multiplex_tracking.add_prompt → reset_state), so prompting every 2
    keyframes (USER request) would keep only the last prompt: a no-op. NOT
    configured; told the user. Real candidates (SAM3.1 builder,
    model_builder.py ~719-750): new_det_thresh 0.7 for a NEW object (detection
    0.5), masklet confirmation = 3 consecutive detections + hotstart 15 frames —
    on parallax-spaced keyframes an object seen in < 3 consecutive keyframes is
    never confirmed; and the VLM consolidation 30 concepts → 21 groups ('white
    folding table', 'doorbell panel', 'conduit' → 0 masklets). NEXT: measure
    (SAM3 detector scores on the keyframes where the desks are visible) — GPU,
    after the night chain.
27. **COLMAP tier 2 implemented** (c7920ab): depth_colmap writes each keyframe's
    COLMAP depth with confirmations/contradictions by the sweep's own rule; F7
    admits it where the sweep measured nothing (source 20, tier 0 > 2 > 1,
    viewer confidence = views / n_views); switch depth.colmap.as_tier.
28. **Epoch-0 cloud after F7** (9019988): precision/epoch0_cloud.py — chunks (or
    the Omega records via the exact prescale→epoch-0 similarity, 0.9707) through
    the cleaner recipe + consolidation + octree into _epoch_0/, registered in the
    manifest; map_worker runs it before deleting the chunks.

29. **USER VERDICT 2026-09-29 16:30 — "EPOCH 0 CORRECTED" is the best cloud so far**:
    Omega's own per-keyframe depth (the complete epoch-0 cloud) re-projected with the
    core's corrections only — depth × s_k (gauge+BA scale, pooled ±1 m of walk), F5's
    camera (fx 391.9 vs Omega 364.3) and F5's poses (the Omega pose graph had left the
    loop 77 cm open) — same confidence gate, same cleaning (voxel+SOR), no F6/F7
    filtering: 25.5 M pts. USER: "prácticamente corrigió la duplicidad del desk, no lo
    hizo perfecto por poco, y prácticamente quedó corregido el piso, por poco, aún tiene
    duplicidad pero poca". → the duplicates were POSE/SCALE errors (chunk scale ×1.34,
    open loop, focal −7 %), which F2+F5 fix; the rest is Omega's per-frame depth noise.
    Built by scratchpad/epoch0_corrected.py into output/_epoch0c/ (not a registered
    epoch). Candidate light filters on it: the historical witness filter (single_witness,
    fed the corrected poses/depths) and the own-error contradiction majority vote.
    Open question to the user: make "Omega + F0/F2/F4/F5 + cleaning" the product and
    keep F6/F7 for high-resolution captures.

30. **Floor on the epoch-0-corrected cloud (USER: "quedó inclinado, aplicar
    obligatoriamente la corrección de suelo que tenemos")**, both halves of what the
    pipeline does, measured on _epoch0c/full (21.8 M pts):
    (a) correction/floor.solve_floor (model plane, the run_floor solver): 271 of 289
    keyframes gave a floor anchor; local scatter 5.2 mm, drift +7.3 mm/m, repeatability
    47.7 mm, tilt bar 1.57°; per-keyframe translations ≤ 9.2 cm, no rotation. It KEPT a
    level change: keyframes 63–288 (8.4 m of walk) sit +136 mm above the reference floor
    of keyframes 0–62 — the drift explains 54 mm, the remaining 83 mm clears the 48 mm
    repeatability, so by design (step demotion preserves real level changes) it was NOT
    corrected. If pccr's floor is one level, that 13.6 cm is residual error the model
    refuses to touch — the user's eye decides.
    (b) the global re-level (the dominant low-band plane, fit_plane_ransac as
    level_floor_core, baked instead of floor_transform.npz because no floor instance is
    projected on this cloud yet): tilt 2.64° → 0.10°, floor centre y 0.428 → 0.001 m.
    Result _epoch0c/floor/ (cleaned_cloud.ply, floor_solution.npz, level_transform.npz,
    potree) — in the viewer slot 2026-09-29 ~17:00. NOTE the per-keyframe solver's
    reference is the FIRST keyframes' floor, so it removes only the variation along the
    walk; the global level is what puts the scene on +Y — the pipeline runs both, in
    this order, after every epoch.

31. **EPOCHS 5 AND 6 ON pccr (2026-09-29 17:35, by the pipeline's own path — scratchpad/
    publish_0c.py, run at the user's word)**: epoch 5 = the epoch-0-corrected cloud
    (+witness +MLS, 21.8 M pts) published as a new-cloud epoch; SAM3 masks projected on
    it (46 instances; the floor = 4 masklets, 5.05 M pts); epoch 6 = correction-module
    floor alignment, model LEVEL (applied, 20 min transactional); display re-level from
    the segmented floor (0.02° → 0, y = 0). MEASURED after: floor per chunk in the
    display frame c0 +0.1 / c1 −0.4 / c2 0.0 / **c3 −18.4** / c4 +0.2 / c5 +0.1 cm (c3's
    low band in the CLOUD is a tail of under-floor flyers: from the depth maps its floor
    is at −0.3 cm); floor-layer thickness per 1 m cell median **16.9 cm** (epoch 5
    without the per-keyframe step: 11.5; epoch 0: 20.3) — the per-keyframe floor
    alignment THICKENS the layer even in `level` mode (its per-keyframe translations
    follow anchors with ~5 cm scatter and separate revisit copies); the chunk check on
    epoch 6 still finds a +15 cm bump at the END of chunk 0 (kf ~55–62): keyframes the
    solver demoted inherit the correction of their neighbours. Both epochs selectable;
    epoch 5's stored display transform is 2.95° / −0.35 m (it shows leveled too).
32. **The chunk/keyframe floor check (e88bf01)** — what it says of pccr epoch 5: chunk 0
    INTRA (floor trend −2.4 … +14.8 cm inside the chunk, seams consistent), 1 undecided
    (+2.9 cm), 2–5 ok; the ceiling is NOT a witness in this scene (58 cm of jumps at
    seams with a quiet floor: ducts/fixtures in the top band), DA3's per-chunk scale
    wanders ±12 % (confirmation only). So on pccr the check can SEE the chunk-0 drift
    but cannot say pose vs depth from floor/ceiling; the seams' horizontal surfaces
    (desks) or the walls' horizontal edges would be the next witness.

33. **visit_drift.measure on epoch 6 (2026-09-29 18:00, nothing applied)** — the
    instrument for the user's "the same disagreement the floor had, in scale/rotation/
    translation": 59 masklets, 21 with 2+ visits → 5 determined closures (bar 10 cm =
    2× repeatability 4.99 cm), all between the START (kf 0–19, chunk 0) and the END
    (kf 254–288, chunk 5) of the walk: chair#100 8.2 cm (4.9 m away), desk#166 21.9 cm
    (4.5 m), desk#167 19.2 cm (2.9 m), ~90 % radial, one common direction (+x, −z);
    TWO FALSE IDENTITIES survived: monitor#120 (2.8 m, two monitors — priced in
    scale_rows by its 286 cm tangential residual but still in instance_loops with σ 5
    cm) and light_fixture#85 (5.1 m, two fixtures). From the 3 genuine ones (OBB centres
    + closures): translation-only (5.7, 2.0, −15.2) cm residuals 8.5/5.7/3.3; rigid
    (3.1°) no better; Sim(3) s 0.969 / 3.1° residuals 5.7/6.0/2.7 — 7 DOF from 3 points,
    indicative only: the loop closes to ~15 cm at the objects, not proportional to
    distance (not a clean scale), scale vs rotation NOT decidable with 3 objects. The
    chunk-pair Sim(3) check needs ≥ 4–5 objects with two visits → the segmentation's
    completeness is what makes the geometry verifiable. The existing correction for this
    measurement is the drift-rate model (visit_drift_run, the 09-18 algorithm); NOT run
    (user's call; the two false closures would have to be excluded first).

34. **SILHOUETTE LEAK EXPERIMENT (2026-09-29 night, USER: "dale, probemos")** — pccr,
    metal_support_column #101 (6 kf, object 1.3–1.75 m, far background 0.4 m behind)
    and red_fire_extinguisher #1 (6 kf, 3–3.9 m, background 1.2 m behind). Per keyframe
    on the object's mask: leak share = the object's OWN edge pixels facing the far
    background whose depth leaks > 20 % of the gap; leak p95 = the tail; interior =
    median |d/d_omega − 1| inside the eroded mask after a per-object rescale; relief =
    p90 − p10 of the depth inside (a flattened prediction shows small).
        column        leak  p95   interior  relief    extinguisher  leak  p95  interior relief
        Omega          89 %  29 cm    —      35 cm                   76 %  45 cm   —      24 cm
        DA3 full       87 %  27 cm  0.8 %    36 cm                   48 %  30 cm  0.7 %   15 cm
        DA3 crop       53 %  17 cm  6.6 %    14 cm                    0 %   0 cm  1.8 %   10 cm
        DA3 MASKED     35 %  12 cm  4.3 %    16 cm                    0 %   0 cm  1.3 %    8 cm
    → DA3 on the full frame is NOT sharper than Omega at the silhouette (the leak is
    context mixing, both models do it). DA3 on the MASKED, ISOLATED object removes it:
    leak 89 → 35 % and 76 → 0 %, tails 29 → 12 cm and 45 → 0 cm, interior within 1–4 %
    of Omega after the rescale, relief consistent with the objects' real depth (a
    column ~16 cm, an extinguisher ~8 cm; Omega's 35/24 cm are inflated by the leak).
    Caveats: no scale of its own (Omega's per-object median gives it), monocular
    interior shape (the silhouette consensus from other views is its check), only for
    segmented objects (75 % of the cloud), one inference per (object, keyframe) mask
    (2 383 on pccr ≈ 40 min GPU; restrict to masks with a far background). PLYs of the
    two objects × 4 variants in output/_exp_depth_edges/ (scratchpad/exp_depth_edges.py).

**Why:** each is a decision the user must take, not a bug to fix silently.
**How to apply:** raise them together when the run ends; implement only what he
approves. See [[user-wants-results-fast]].
