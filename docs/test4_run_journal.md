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
| Intra-chunk | SKIP all chunks — expected |
| finereg | 6 non-adjacent pairs, 147 poses, 6 chunks corrected, plane 116.7→116.6 mm (RANSAC → mild run-to-run variation is normal) |
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
| A1 | pending | d2e280e | none (control) | | |
