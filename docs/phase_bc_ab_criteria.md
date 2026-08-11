# Phases B & C — A/B acceptance criteria (defined BEFORE the verdicts)

Precision task. Same pre-registration discipline as Phase A
(scale_ab_criteria.md): this file is written before the deciding runs finish;
verdicts are judged against it without goal-moving.

Metrics harness: `server/tools/mesh_ab_metrics.py` — planar-patch RMS (mm) and
the double-surface (onion) statistic measured on the SAME world-space patches
across variants (seeds picked once on the baseline mesh); comparisons use
COMMON measured patches only. Crop PLY pairs of the same zones are exported for
the visual verdict (the journal rule: numbers alone do not capture everything).

## Phase B — multi-view consistency filter (tsdf.mv_consistency)

Variants: `baseline` (filter off), `mv_mask` (inconsistent pixels masked),
`mv_median` (surviving pixels replaced by the median of consistent estimates).
Sessions: test2 (single-pass) and test4 (chunked-metric, production baseline
G1). All cells share production TSDF config, geometry only.

**The mask variant becomes default ON iff, on BOTH sessions (common patches):**

- planar-patch RMS median is **not worse** (≤ +2% relative), AND
- the bimodal (double-surface) patch fraction **does not increase**, AND
- global discard stays **under the 40% pose/scale alarm**, AND
- added wall-time is **≤ 15%** of the TSDF stage.

The median variant replaces the mask variant only if it beats it on RMS with
the same double-surface bar. If either session regresses, default stays OFF
and the filter remains available behind the flag.

## Phase C — native-resolution depth (tsdf.native_depth_method) + voxel matrix

Grid on test2 (the session with true 1080p video): method
{off, guided_filter, da3_detail_transfer} × voxel {12, 8, 6 mm}, Phase-B mask
ON in every cell (C consumes post-consistency depth by design). test4 is
EXCLUDED by data: its source video is 360×640 — BELOW the omega grid — so
"native resolution" does not exist for it (recorded as a data limitation, not
skipped silently). test7 (1080×1920) is the confirmation session if test2
shows a win.

**A refinement method earns default ON iff, at the production voxel (12 mm):**

- planar-patch RMS median must **NOT worsen** (> +2% relative ⇒ reject — new
  noise on smooth surfaces = the upsampling is mis-calibrated, the exact
  failure that killed the legacy `upsample_depth`), AND
- the double-surface fraction does not increase, AND
- visual crops of edges do not show smearing (user verdict at phase close).

`da3_detail_transfer` is additionally judged against `guided_filter` (the
mandatory deterministic baseline): it must beat it on RMS or edge crops to
justify its DA3 hi-res extraction cost (time/VRAM measured and reported).

**Voxel default** moves from 12 mm only to the ELBOW of the quality/cost curve:
the finest voxel whose RMS gain vs the next coarser step is ≥ 5% relative
while time and VRAM stay within 2× of the 12 mm cell. Otherwise 12 mm stays.

## Recorded interim results (test2, written 2026-08-11 before test4 finished)

Common patches [0, 1, 3, 8, 11]:

| variant | RMS med (mm) | per-patch | bimodal |
|---|---|---|---|
| baseline | 9.00 | 8.7, 9.5, 6.1, 9.0, 10.5 | 2/5 |
| mv_mask | **8.70** | 8.8, 6.9, **3.9**, 8.7, 10.1 | **1/5** |
| mv_median | 9.22 | 7.5, 10.5, 5.8, 9.2, 10.0 | 2/5 |

mv_mask ≤ baseline on every common patch; global discard test2 25.7% /
test4 18.6% (both under the 40% alarm); TSDF wall time 374 s (mask) vs 409 s
(baseline) — the filter pays for itself in integration time (fewer pixels).
mv_median shows no consistent benefit. Pending: test4 cells.
