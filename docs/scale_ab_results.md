# scale_align v2 — A/B results & verdict (2026-08-11)

Judged strictly against the pre-registered criteria in
[scale_ab_criteria.md](scale_ab_criteria.md). Harness: `server/tools/scale_ab.py`
after `server/tools/extract_da3_anchors.py --count 32 --pairs 3` on each session
(the original DA3 anchors had been freed from disk). Sessions: test2 (short
single-pass, s≈11.3), test4 (81 m chunked-metric, residual s≈0.97), test7
(longest, chunked-metric, residual s≈0.99).

Note on test4/test7: these outputs are chunked-metric, so the global scale_align
stage measured here is the RESIDUAL/verification pass (per-chunk metric-lock
already happened inside the omega pass). test2 is the clean single-pass case
where s carries the full metric burden.

## test2

anchors available: 36 — eval pairs: 7

| mode | K | s | used | held-out err % | jack max dev % | jack MAD % | pair reproj % |
|---|---|---|---|---|---|---|---|
| global_median | 12 | 11.2643 | global_median | 9.06 | 1.39 | 1.39 | 2.88 |
| affine_robust | 12 | 11.0832 | scale_only_robust | 8.65 | 1.36 | 0.62 | 2.89 |
| depth_dependent | 12 | 11.0832 | scale_only_robust | 8.65 | 1.36 | 0.62 | 2.89 |
| global_median | 24 | 11.3015 | global_median | 5.91 | 0.17 | 0.17 | 2.88 |
| affine_robust | 24 | 11.0185 | scale_only_robust | 8.05 | 0.59 | 0.26 | 2.89 |
| global_median | 32 | 11.3707 | global_median | 5.67 | 0.44 | 0.44 | 2.88 |
| affine_robust | 32 | 11.1333 | scale_only_robust | 6.20 | 0.45 | 0.18 | 2.89 |

## test4

anchors available: 38 — eval pairs: 5

| mode | K | s | used | held-out err % | jack max dev % | jack MAD % | pair reproj % |
|---|---|---|---|---|---|---|---|
| global_median | 12 | 0.9559 | global_median | 6.45 | 0.32 | 0.32 | 2.87 |
| affine_robust | 12 | 0.8144 | **affine_robust** | 5.33 | **4.98** | 0.90 | 2.73 |
| global_median | 24 | 0.9785 | global_median | 4.49 | 1.03 | 1.03 | 2.86 |
| affine_robust | 24 | 0.9822 | scale_only_robust | 4.50 | 0.93 | 0.36 | 2.86 |
| global_median | 32 | 0.9785 | global_median | 5.74 | 1.03 | 1.03 | 2.86 |
| affine_robust | 32 | 0.9862 | scale_only_robust | 4.84 | 0.65 | 0.22 | 2.87 |

## test7

anchors available: 38 — eval pairs: 5

| mode | K | s | used | held-out err % | jack max dev % | jack MAD % | pair reproj % |
|---|---|---|---|---|---|---|---|
| global_median | 12 | 0.9916 | global_median | 8.01 | 0.12 | 0.12 | 2.94 |
| affine_robust | 12 | 0.9665 | scale_only_robust | 9.07 | 2.58 | 0.55 | 2.95 |
| global_median | 24 | 0.9724 | global_median | 7.01 | 0.70 | 0.70 | 2.94 |
| affine_robust | 24 | 0.9862 | scale_only_robust | 7.02 | 1.20 | 0.34 | 2.94 |
| global_median | 32 | 1.0029 | global_median | 9.96 | 1.01 | 1.01 | 2.94 |
| affine_robust | 32 | 1.0139 | scale_only_robust | 10.19 | 1.34 | 0.49 | 2.95 |

(`depth_dependent` rows omitted where identical to `affine_robust`: on every
cell its CV/BIC gate degraded through the same ladder to the same model.)

## Verdicts (against the pre-registered criteria)

### 1. Estimator mode → `global_median` STAYS the default

- The evidence gate itself rejected the structured models in **8/9 cells**: the
  offset / depth-dependent gain did not beat scale-only under leave-frames-out
  CV + BIC. Per the pre-registered clause, that IS the finding: **our DA3↔Ω
  residuals carry no exploitable offset or depth-linear structure** at these
  anchor counts.
- The single cell that kept the affine model (test4, K=12) failed the
  stability bar: jackknife max deviation 0.32% → 4.98% (the offset makes s
  hostage to which anchors are present — exactly the instability the criteria
  guard against) and pulled s −15% off the production value.
- `scale_only_robust` (the degraded form: Huber gain fit instead of the ratio
  median) was never better than `global_median` beyond noise on held-out error
  at matching K, and pair reprojection was flat (2.87–2.95%) across all modes —
  the candidates differ by <2% in s, inside that metric's sensitivity floor.

### 2. Anchor count → 12 STAYS the default

Pre-registered bar: jackknife improves ≥30% relative AND held-out not worse, on
≥2/3 sessions. K=24 passed only on test2 (jack 1.39→0.17%); on test4 and test7
the jackknife WORSENED (0.32→1.03%, 0.12→0.70%). → 1/3, not adopted.

**Honest observation for the record** (no goal-moving — the default stands):
held-out anchor depth error improved with K=24 on **3/3** sessions
(9.06→5.91, 6.45→4.49, 8.01→7.01%). If a future phase needs a tighter scale
prior, re-testing K with the Phase E external scorecard as judge is the
natural follow-up.

### 3. Depth-coverage anchor top-up → implemented, default OFF (neutral)

The top-up triggers on all three scenes (1–2 uncovered depth bins), but its
measured effect is neutral: pair reprojection ±0.002 pts, jackknife mixed
(test2 1.39→0.94%, test4 flat, test7 0.12→0.66%), s moves ≤0.5%. Per the
evidence rule (worse-or-neutral ⇒ OFF), `scale_anchor_depth_coverage: false`.

### 4. VIO source → implemented, awaiting real data (external dependency)

No VIO recording exists for the current sessions — genuine EXTERNAL data
dependency (needs a capture app exporting the trajectory, docs/VIO_FORMAT.md).
All VIO paths are unit-tested synthetically (segment robustness under drift,
fail-hard gates, priority + agreement reporting); the first real VIO session
will report the VIO↔DA3 agreement automatically in `scale_diagnostics.json`.

## Side finding

The per-session `scale_diagnostics.json` (now emitted on every run) contains
the residual-vs-depth profile that motivated the structured models; on these
sessions the profile is flat within noise after the near-band gate — consistent
with the near-band + confidence gates (validated earlier on test3) having
already removed the far-field DA3 bias the affine/depth models were designed to
absorb.
