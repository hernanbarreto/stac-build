# Phases B & C — A/B results & verdicts (2026-08-11)

Judged strictly against the pre-registered bars in
[phase_bc_ab_criteria.md](phase_bc_ab_criteria.md). Harness:
`server/tools/mesh_ab_metrics.py`; TSDF variants share the production config
(geometry only, no texture) under `output/tsdf/ab_*` per session; metrics +
crop PLY pairs persisted under `output/tsdf/ab_metrics/`.

## Phase B — multi-view consistency filter

Mask generation cost: 210 keyframes in 26 s (test4) / 66 in 4 s (test2), GPU.
Global discard: test4 18.6% (per-frame median 17.1%), test2 25.7% — both well
under the 40% pose/scale alarm.

### test2 (66 kf, single-pass) — common patches [0, 1, 3, 8, 11]

| variant | RMS med mm | per-patch | bimodal | TSDF time |
|---|---|---|---|---|
| baseline | 9.00 | 8.7, 9.5, 6.1, 9.0, 10.5 | 2/5 | 409 s |
| **mv_mask** | **8.70** | 8.8, 6.9, 3.9, 8.7, 10.1 | **1/5** | 374 s |
| mv_median | 9.22 | 7.5, 10.5, 5.8, 9.2, 10.0 | 2/5 | 592 s |

### test4 (210 kf, chunked-metric, G1 baseline) — common patches [0, 2, 5, 6, 7, 10, 11]

| variant | RMS med mm | per-patch | bimodal | TSDF time |
|---|---|---|---|---|
| baseline | 5.89 | 11.4, 3.9, 5.9, 8.9, 8.6, 1.9, 4.6 | 2/7 | 1157 s |
| **mv_mask** | **5.87** | 11.0, 2.6, 5.9, 8.8, 8.4, 1.9, 4.7 | 2/7 | 976 s |
| mv_median | 6.23 | 9.7, 2.6, 6.2, 9.6, 8.8, 1.9, 5.1 | 1/7 | 1221 s |

### Verdict

- **mv_mask → default ON** (`tsdf.mv_consistency: true`). Passed every
  pre-registered bar on BOTH sessions: RMS not worse (−3.3% / −0.3%),
  double-surface fraction down or equal, discard under the alarm, and the
  added cost is NEGATIVE (the TSDF integrates fewer pixels: −8.5% / −15.6%
  wall time including mask generation).
- **mv_median → rejected** (stays behind `mv_replace_median`): RMS worse than
  the mask variant on both sessions (9.22 vs 8.70; 6.23 vs 5.87). Its lone
  bright spot (test4 bimodal 1/7) does not clear the RMS bar. Consistent with
  the E2 journal lesson: substituting depth values mixes per-frame estimates
  that are mutually inconsistent — masking keeps only what agrees, replacing
  re-introduces the mixture.
- Visual crop pairs: `output/tsdf/ab_metrics/crop_p*_{baseline,mv_mask}.ply`
  per session, same world-space zones, for the user's visual confirmation.

## Phase C — native-resolution depth + voxel matrix (test2, clean GPU-exclusive runs)

Full 3×3 matrix, Phase-B mask ON in every cell, production TSDF config,
geometry only. Times are CLEAN (one job per GPU — the earlier contaminated
sweep measured 3.1× for the 8 mm step; clean it is 1.8×). DA3 hi-res detail
source: 66 keyframes at process_res 1008, one-off cached per session.

### Quality (COMMON patches, n=10) and cost

| cell | RMS med mm | bimodal | TSDF time | peak GPU |
|---|---|---|---|---|
| **off @ 12 mm** (production) | **8.85** | **3/10** | 374 s | — |
| guided @ 12 | 8.86 | 7/10 | 494 s | 17.7 GB |
| da3 @ 12 | 8.84 | 7/10 | 445 s | 16.9 GB |
| off @ 8 | 9.01 | 4/10 | 679 s (1.8×) | 17.5 GB |
| guided @ 8 | 9.05 | 7/10 | 829 s | 17.5 GB |
| da3 @ 8 | 9.12 | 5/10 | 809 s | 18.8 GB |
| off @ 6 | 9.34* | 4/10* | 1098 s (2.9×) | 18.7 GB |
| guided @ 6 | 9.65* | 7/10* | 1682 s | 18.9 GB |
| da3 @ 6 | 9.67* | 7/10* | 1655 s | 18.9 GB |

(* summary over measured patches; common-patch table for the 12/8 rows in
`ab_metrics_c/mesh_ab.json`, crops included.)

### Verdicts (pre-registered bars)

1. **native_depth_method → stays OFF.** RMS ties at the production voxel
   (8.85 / 8.86 / 8.84 — within 0.2%), but BOTH refinement methods more than
   double the double-surface incidence (3/10 → 7/10). The added high-frequency
   content carries inter-frame inconsistencies that the coarser depth
   naturally averaged out — the same failure mode that A/B'd the legacy
   `upsample_depth` OFF, now measured precisely. The bar said "double-surface
   must not increase": both fail decisively. Implemented, flag-gated,
   documented.
2. **Voxel → stays 12 mm.** 8 mm passes the cost bar (1.8× ≤ 2×) but WORSENS
   RMS (8.85 → 9.01; the bar required a ≥5% gain); 6 mm fails cost (2.9×) and
   quality both. Finer voxels only sharpen the noise our depth already has —
   the resolution ceiling is the depth consistency, not the grid. That is
   PRECISELY what Phase D attacks (photometric optimization).
3. **test7 confirmation — not required** (pre-registered: only if test2 showed
   a win). **test4 excluded by data**: its source video is 360×640, below the
   omega grid — no native detail exists to recover.
