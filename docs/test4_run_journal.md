# test4 run journal — improvement rounds from the run-6 baseline

Purpose: never regress blindly again. Every reconstruction run of test4 gets one
entry here: config delta, log fingerprint numbers, and the USER'S VISUAL verdict
(the numbers alone do NOT capture serpenteado/duplicates — proven 2026-07-11).

Scene facts (do not re-derive): test4 is a LINEAR outdoor walk A→B, single pass,
81.1 m, 1690 frames. **0 loop closures is CORRECT for this scene.** Chunked-metric:
210 kf (motion quantum 58), 9 chunks × 42, overlap 21.

Baseline: **run 6** = log `logs/server_20260711_080047.log` (08:29), produced by
vendor VGGT-Long `ea178bc` + stac `26ffde4`. Restored exactly in stac `d2e280e`.
Post-run-6 additives (sick-aware ownership `c476283`, finereg health-index+rollback
`6f808eb`) reverted — user validated they did not improve test4 visually.
DISCARDED FOREVER: ICP dense_fusion (tested, fails, texture mis-mapping).

## Reference fingerprint — run 6 (what an identical re-run must show)

| Metric | Value |
|---|---|
| Phase-1 scale / phase-2 scale | s≈40.444 / s=0.9698 |
| Anchor held-out error | 0.78% → 0.38% |
| Chunk scale spread | 7.214–11.778 (×1.63) |
| Health | 9/9 healthy, 0 sick |
| Exact-seam median residuals (8 seams, cm) | 6.6, 10.3, 8.2, 10.9, 5.3, 4.8, 4.3, 8.4 |
| Elastic before → after (median cm) | 4.6–9.6 → 2.75–5.65 |
| Elastic per-frame fits (seam 7 = worst) | \|t\| median 63 cm, max 159 cm — the smoking gun |
| Depth-graph | REFUSED (⛔ model ladder) — expected |
| Intra-chunk | 7 APPLY / 2 IDENTITY (chunks 1, 5); held-out improves ~11→8 … 5→3 cm; max corrections 8.8–58.1 cm (chunk 7 the largest) — identical across runs |
| finereg | 6 non-adjacent pairs, 147 poses, 6 chunks corrected, plane 116.7→116.6 mm (RANSAC → the ONLY nondeterministic stage; 4–6 pairs, 117–120 mm across runs) |
| Coverage trim | 13 tail kf dropped, walk kept 60.5/81.1 m |
| Cloud | ~39.0M merged → ~24–25M cleaned |

Known visual defects AT run 6 (the targets): serpenteado on straight edges,
duplicated objects in overlap zones, end-of-walk section mispositioned, holes
vs da3_streaming density.

## Plan (agreed 2026-07-11)

- **A1** control re-run on d2e280e — must match fingerprint above.
- **A2** `elastic_seam: false` — serpenteado in overlap bands gone? → elastic guilty.
- **A3** `scale_drift: false` — chunk-long smooth bowing gone? → drift guilty.
- **B** fix the culprit (elastic: along-trajectory smoothing + cap + held-out gate;
  drift: harden gate).
- **C** global pose refinement without loops: DA3 unary priors in Sim3LoopOptimizer.
- **D** density (no ICP): ownership backfill (neighbour writes only where owner wrote
  nothing), quantum 58→~40, conf_percentile 20→10, revisit tail trim.
- **E** long-term: windowed RGB-D BA (docs/pose_refinement.md spec).

User launches every run from the UI; assistant edits config/code between runs and
verifies each new log against this file.

## Runs

| # | Date | Commit | Config delta | Fingerprint vs run6 | Visual verdict (user) |
|---|---|---|---|---|---|
| run6 | 2026-07-11 08:29 | 26ffde4 (vendor ea178bc) | — | reference | serpenteado + duplicates + end mispositioned + holes; BEST so far |
| A1 | 2026-07-11 18:21 | d2e280e | none (control) | ✅ MATCH: all deterministic stages bit-identical (scales 40.4436/0.9698, anchor 0.78→0.38%, seams 6.6…8.4 cm, elastic, intra 7A/2I, depth-graph ⛔, trim 13 tail kf). finereg (nondet): 5 pairs (0-2,0-3,1-3,2-4,5-7 — lost 6-8), 126 poses, 5 chunks, plane 117.25→117.95 mm (slightly WORSENED, no rollback in run6 finereg). Cloud 38.9M→23.78M | ✅ user: looks like run 6 |
| A2 | 2026-07-11 18:57 | 0cc1858 | `elastic_seam: false` (flag now configurable; depth-cap far-drop self-off — not a density gain) | elastic OFF confirmed (0 [elastic] lines). Upstream identical (seams 6.6…8.4). Phase-2 scale 0.9714 (vs .9698 — points moved less). Intra-chunk verdicts shifted: 4 APPLY (1,4,7,8) vs 7 — endpoints clamp to a different seam consensus. finereg 7 pairs/147 poses/6 chunks, 117.7→118.0 mm. Cloud 39.9M (+1M = far-drop off) → 25.6M | ❌ user: NO visible difference — serpenteado AND duplicates unchanged. **Elastic acquitted** |
| A3 | 2026-07-11 19:29 | 12eee62 | `scale_drift: false`, elastic back to true | drift OFF confirmed (0 DRIFT lines, no "drift APPLIED"). Scale spread widened ×1.63→×1.72; no held-out anchor improvement (0.78→0.38 line gone). Seams shifted (6.9…9.8 cm), elastic similar, intra 6 APPLY, finereg 7 pairs/168 poses/7 chunks 119.9→119.4 mm. Cloud 39.2M→22.9M | ❌ user: MANY more duplicates + serpenteado persists. **Drift acquitted for serpenteado AND proven to REDUCE duplicates — keep ON** |
| A4 | 2026-07-11 20:02 | caf35c2 | `intra_chunk: false` (drift+elastic back to run-6 true), "run 3 behaviour" | intra OFF confirmed (0 intra lines). Drift applied, anchor 0.78→0.38%, spread ×1.63, seams + elastic identical to run6, finereg 5 pairs/147 poses/6 chunks 119.4→118.1 mm. Cloud 39.1M→25.5M | ❌ user: serpenteado FIRM; duplicates ≈ run6 or a touch more. **Intra-chunk acquitted (and mildly protective)** |
| E1 | 2026-07-11 21:07 | 9587e3c (vendor 01f6fa9) | BATCH: B elastic tamed + C finereg anneal + D backfill & conf 10% + E-lite hybrid DA3 write | B ✅ (fits capped, max moves 12.9–30 cm vs 142.5, seam residuals held 3.5–5.6). D ✅ (3.73M px backfilled; cloud 45.5M→33.2M, +38%). E-lite ⚠ BUG: re-run path kept 25-anchor extraction → only 5–8/42 frames per chunk re-shaped (36 without DA3 map). C ❌ DEFECTIVE: solver proposed 700–1500 mm (point-to-plane NULL-SPACE, in-plane translation invisible to the cost); clamped 250 mm steps applied to ALL 9 chunks/210 poses for a 0.4 mm gain (119.63→119.22 mm) | ❌ user: MASSIVE drift, one of the WORST results — explained by C applying null-space motion + partial hybrid mixing shapes within chunks |
| E2 | 2026-07-11 21:43 | e2effcf | fixes: finereg null-space projection + ≥3% gain else FULL ROLLBACK; hybrid DA3 for ALL 210 kf on the re-run path | Hybrid ✅ FULL: 42/42 frames re-shaped in all 9 chunks (~85M px, shape corrections 2.6–7.9% median). Scale drift now SKIPPED by its own gate (hybrid shapes reduce intra-chunk drift; held-out 0.68→0.54%). Seams mostly better (2.7–11.7 cm), elastic after 1.8–5.1 cm (best yet). finereg ✅ SAFE: projection removed 20–327 mm null-space per unit; round 1 worsened worst sep (112.89→119.40) → FULL ROLLBACK, 0 chunks touched. Baseline worst sep improved to 112.89 mm (was 117–120). ⚠ depth-graph disagreement UP: 2.17% before (was 0.72%) — per-frame DA3 shapes are less inter-frame consistent than omega's; still refused/untouched. Cloud 46.3M→35.0M (densest yet) | ⚖ user (refined): first impression "disaster", but on inspection **BETTER than run6 in serpenteado (reduced!) and density; the dominant remaining problem is POSE/DEPTH DRIFT (layering)**. Diagnosis: the DA3 shape STRAIGHTENS (bet validated) but per-frame substitutions are mutually inconsistent — per-frame near-band ratio jitter + per-frame monocular affine distortion (the 0.72→2.17% depth-graph jump is the signature) → same surface at slightly different depths per frame = layered drift. finereg was a no-op (rolled back) — not the cause |

### PHASE A CONCLUSION (2026-07-11)

All three correction stages acquitted by visual A/B: elastic (A2), scale drift
(A3 — protective vs duplicates, pinned ON), intra-chunk (A4 — mildly protective).
**Serpenteado is INTRINSIC to omega's raw feed-forward output** (per-frame
pose/depth noise, no BA inside a chunk) — user's read, consistent with
docs/pose_refinement.md ("VGGT must be finished with BA"). DA3 does not show it
(but has onion/duplicate problems of its own). Correction stages restored to
run-6 config (all ON). Attack paths that remain: Phase C (DA3 unary priors →
global consistency/duplicates), Phase E (windowed RGB-D BA or DA3-depth-through-
omega-poses → the only fixes that can straighten the waviness itself).

### BATCH B+C+D+E-lite implemented (2026-07-11, commit 9587e3c, vendor 01f6fa9)

User decision: GPU runs are expensive → implement everything remaining at once,
one validation run. All config-gated (each can be turned off individually):

- **B** elastic taming: fits smoothed along trajectory (win 5) + |t| cap 30 cm
  (`elastic_smooth_win`, `elastic_max_t_m`). Log: `[elastic] fits tamed: ...`.
- **C** finereg anneal: ≤3 bounded rounds, over-cap steps CLAMPED to 250 mm/round,
  total budget 750 mm/unit, FULL ROLLBACK if worst separation doesn't improve
  (`anneal_rounds`, `max_total_correction_m`). Log: `anneal round N/3 — worst ...`.
- **D** density: ownership backfill — non-owner writes exactly the owner-dropped
  pixels (`ownership_backfill`); `conf_percentile` 20→10. Log: `backfill armed`,
  `+ backfills N owner-dropped px`.
- **E-lite** hybrid DA3 write (THE serpenteado bet): every keyframe adopts DA3's
  depth SHAPE at omega's scale/pose right after metric-lock (`hybrid_da3`,
  `hybrid_da3_far_m: 15`). DA3 runs on ALL 210 kf (~8× anchor extraction time,
  one-off cache) → **metric-lock fingerprint WILL shift (denser anchors) — do
  not read that as a regression**. Log: `[hybrid-da3] chunk k: N/42 frames
  re-shaped ...`.

Expected in the E1 log vs run6: hybrid-da3 lines (new), tamed elastic (smaller
max translations), anneal rounds in finereg (worst separation should finally
drop well below ~117 mm or roll back), backfill px counts, bigger cloud.
Suite: 106 passed.
