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

## Phase C — native-resolution depth (results appended when the sweep lands)
