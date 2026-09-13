# Correction by multi-view consistency — status 2026-09-09 (to continue next session)

Session pccr `2026-08-31` (216 keyframes, 7 chunks of 59 kf / overlap 29,
26 M points). Cloud untouched (epoch 0). Everything below is READ-ONLY
diagnostics unless stated.

## What was established today (in order)

1. **Object-marked correction and the drift-rate line** (server/correction/,
   committed earlier) fix one duplicate but do not solve the walk: the error
   is continuous, not a single closure.
2. **Geometric revisit detector** (`correction/revisit.py`, uncommitted):
   finds WHERE/WHEN the walk sees the same place again (31/08: kf 1–14 ↔
   207–215, the desk). Correct for where/when, wrong for how much: every
   closure it measured was an ICP on partial/planar geometry (block ICP,
   joint ICP, co-visible-only ICP) and came out short or garbage. Applied
   once (0.35°/20 cm, epoch 1) → still duplicated → user Undo. DO NOT reuse
   its closure magnitudes.
3. **Chunk pose graph** (`correction/posegraph.py`) and **keyframe pose
   graph from pairwise ICP** (`correction/kfgraph.py`): solver math verified
   exact on known errors, but fed by per-pair ICP transforms they are
   garbage in (kf-pairs sharing 1 m of floor gave 10°/70 cm). Conclusion:
   NEVER estimate a 6-DOF transform per keyframe pair from a small shared
   patch.
4. **Block consistency matrix** (`correction/consistency.py`): the first
   version (median NN of a keyframe's points vs the pooled others in 1 m
   blocks) hides everything (consecutive keyframes dominate the pool). The
   pairwise version showed "deviations" of 80–135 cm at kf 20–26 that turned
   out to be ARTEFACTS: two keyframes writing different surfaces (floor vs
   cabinet) inside the same 1 m block. A block is not a surface. The user
   caught this; see `output/corrections/inspect/block_323_*.png`.
5. **Photometric reprojection check** — THE VALID MEASUREMENT (user's idea:
   stand at every keyframe, reproject what the other keyframes wrote with
   their own colours into its photo, compare with the photo). Implemented in
   `server/logs/_reproj_check_0831.py` (script) and
   `correction/photobundle.py::render_pair`. Real photo vs synthetic photo
   (z-buffered splat of the other keyframe's coloured points), dense optical
   flow (OpenCV DIS) on the covered pixels, median in px and in cm
   (px · z / fx). Validated where we know the truth:

   | case | kf | points of | flow | error |
   |---|---|---|---|---|
   | consecutive | 12 | 10 | 2.4 px | 1.5 cm |
   | consecutive | 10 | 12 | 3.6 px | 2.6 cm |
   | chunk seam 0/1 | 58 | 60–61 | 4.3 px | 1.8 cm |
   | chunk seam 5/6 | 180 | 182–183 | 3.3 px | 1.5 cm |
   | revisit | 9 | 213–215 | 47 px | 33 cm |
   | revisit | 214 | 8–10 | 34 px | 27 cm |

   Findings: chunk seams are FINE (same as consecutive keyframes); the
   error accumulates along the walk and only shows at the revisit. A pixel
   is one surface, so no block/surface mixing. Images:
   `output/corrections/inspect/reproj_*.png` (panels: real | synthetic |
   50/50 overlay | flow magnitude in red).

6. **Photometric bundle adjustment** (`correction/photobundle.py::run_photobundle`,
   `solve_poses`): one SE(3) per keyframe (kf 0 fixed), residual =
   reprojection error of X_j·p in camera k moved by X_k minus the flow
   target, weight = image gradient, LM with closed-form Jacobians, outer
   render→flow→solve rounds. Solver verified exact on synthetic known pose
   errors (`logs/` inline checks; convergence is quadratic once the LM
   damping is allowed to decay to `kfgraph.damping_min`).
   First full run on 31/08 (7481 candidate pairs, 3470 with coverage,
   1.36 M correspondences, 2 rounds, 78 min) — `output/corrections/photobundle.json`,
   `photobundle_poses.npz`, `photobundle_corrs_round0.npz` is NOT written by
   that run (cache added afterwards, see pending):

   | pairs by gap | before | after |
   |---|---|---|
   | 1–3 kf | 3.8 px | 4.0 px |
   | 4–10 kf | 5.8 px | 5.2 px |
   | 11–30 kf | 8.7 px | 7.1 px |
   | revisit (>100 kf, 17 pairs) | 32 px | 24 px |

   kf 214 vs kf 0–3: 36–46 px → 22–30 px (21–26 cm → 14 cm). Improvement
   but NOT closed. Two signs of noise driving the solve: neighbours got
   slightly worse (3.8 → 4.0 px) and the solved poses carry 5° rotations at
   kf 50 and a 39 cm / 7° step at kf 193 — not drift, the fit absorbing bad
   flow (repetitive tile joints ~ the 30–45 px offsets, hole borders,
   reflections) under a pure squared loss.

## Pending (the user stopped here: "dejalo comentado en un md")

Next thing to try, NOT yet written (the edit was rejected mid-way, nothing
of it is in the tree):

1. **Robust loss in `solve_poses`**: Huber IRLS on the 2-D reprojection
   residual, threshold `photo.huber_px` ≈ 4 px (the neighbour floor). Weight
   factor `min(1, δ/|r|)`; cost `r²` inside δ, `2δ|r| − δ²` outside. Config
   key must be added to `PhotoConfig` (config.py), `config.yaml`
   `correction.photo.huber_px`, and `tests/synth_correction.py`.
2. **Cache the round-0 correspondences** (`photobundle_corrs_round0.npz`,
   identity poses) so re-solves skip the 20-minute render; run with
   `outer_iters: 1` for trials (≈25 min: cached solve + final measurement).
3. **Weight balance near/far**: 3300 neighbour pairs vs 17 revisit pairs;
   consider per-pair normalisation so a revisit pair is not drowned.
4. If the flow ambiguity on tiles persists: coarse-to-fine flow (downscaled
   images first) or gradient-image flow; the depth channel (Δz vs k's own
   depth map) for the along-ray component.
5. Only after the far pairs reach the neighbour floor: apply through the
   existing transactional pipeline (`stage_transaction` / `swap_transaction`
   with `R_kf, t_kf` from `photobundle_poses.npz`), Approve/Undo in the UI.

## Verify the solver before any run (both checks passed today)

- `solve_poses` with synthetic exact correspondences: cost → 0, pose error 0
  in 5 iterations (see the inline script in this session's log).
- Inspect a pair before/after: `python server/logs/_photo_after_0831.py`
  (renders `inspect/photo_{before,after}_kf{9,214,12}.png`).

## Files (all uncommitted, on top of 87dcedd)

- `server/correction/revisit.py`, `posegraph.py`, `kfgraph.py`,
  `consistency.py`, `photobundle.py`; `run.py::run_revisit`;
  `api.py` `/api/correction/revisit`; UI button "🔁 Detect & close revisits"
  and the verdict-dialog rows (`ui/src/App.tsx`,
  `ui/src/components/CorrectionVerdictDialog.tsx`).
- `server/config.yaml` sections `correction.revisit`, `consistency`,
  `kfgraph`, `photo`, `posegraph`; matching dataclasses in `config.py`;
  `tests/synth_correction.py` mirrors them; `tests/test_correction_posegraph.py`.
- Scripts: `server/logs/_revisit_run_0831.py`, `_consistency_0831.py`,
  `_kfgraph_0831.py`, `_photo_0831.py`, `_photo_after_0831.py`,
  `_reproj_check_0831.py`, `_show_block_0831.py`, `_closure_tol_0831.py`.
- Outputs: `server/projects/pccr/scans/2026-08-31/src_default/output/corrections/`
  (`revisits.json`, `consistency.json`, `consistency_matrix.npy`,
  `kfgraph*.{json,npz,png}`, `photobundle.json`, `photobundle_poses.npz`,
  `inspect/*.png`). Ledger has one revisit run (257af8a7) applied and undone.
- The paused fusion module (`server/fusion/`, `scan_fusion:` /
  `metric_validation:` config) is untouched since the previous session.

## Working rules the user set today (keep)

- Design and validate with him BEFORE running; no trial-and-error runs on
  his data; read-only diagnostics only when he asked for them.
- Answer short. When asked "cómo viene", report progress and ETA only.
- Scale is coherent inside one scan: the variables are rotation +
  translation per keyframe. Scale only between scans (fusion).
- "Segment everything" (SAM3 + DINOv3) is the other valid way to define a
  region as one surface; not used yet for correction.
- Never claim a matrix/metric is valid without showing a known-good and a
  known-bad case on the real data.
