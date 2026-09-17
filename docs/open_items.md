# Open items — fix at the end of the run

Running list. USER 2026-09-15: "vamos anotando para corregir al final".
Each item says what was OBSERVED, where, and why it matters. Delete an item
only when it is fixed and verified, never because it looks small.

Readings that turned out WRONG are kept and marked, so nobody spends the
afternoon rediscovering them.

---

## 0. THE DESIGN — the cloud is AUDITED against the mask, and nothing is cut

USER 2026-09-15, arrived at over the course of the pccr run: *"no hay nada que
cortar, es la auditoría de tu propia nube contra el ground truth de la
máscara, es eso"*.

**The measurement.** Project an instance's points into each keyframe where it
has a mask and accumulate a DENSITY map at mask resolution — a heat map, not a
contour. Contours are useless here: *"si usas contorno, podrías tener apenas
voladores que te dirían acá hay duplicado"*. A handful of flyers stretch an
outline but never form a mode; a duplicate does.

**What the map means** — it is WHERE the concentrations fall relative to the
mask, never how many there are:

| what is seen | verdict |
|---|---|
| every concentration on the mask (1 or n islands) | correct, possibly incomplete — confidence filtering leaves holes, holes are not errors |
| one on the mask, another off it or displaced | geometry in the wrong place |
| all of them off the mask | the instance is displaced in this frame |
| mask with no concentration | missing coverage, not a position error |

Generalised to n parts: each concentration is judged on its own, and several
can be wrong at once, each with its own displacement.

**The distinction that falls out for free**, and the useful one: a
concentration off-mask in one visit's frames but ON-mask in the other's is a
DRIFT DUPLICATE — correct it. One that is off-mask in EVERY visit is not a
duplicate, it is attached junk — declare it, do not correct it. Neither case
cuts anything.

**Where it wires** — `segmentation/mask_filter.py`, which already holds the
masks, the poses, the trace grid, the frame-space mapping and the visit
grouping, and today uses them to trim. It stops being a filter and becomes the
audit.

**What it replaces**
- the point removal in `segmentation/pipeline.py` (the `keep_geo` trim before
  `_clean_segment_subcloud`) — gone entirely;
- the split decision in `loops/instance_loops.py`: the
  `identity_reject_factor × δ(L)` rule and the `instance-split` action. No
  instance is ever split again;
- `copy_evidence`'s role as split arbiter (`_reproject_verdict`);
- the greedy loop's measure in `certify/iterate.py`: instead of counting points
  inside masks, it measures whether an off-mask concentration moved onto it.

**What survives** — measuring the closure between two concentrations with ICP
(`measure_copy`, `instance_edges`). What changes is which pairs it is asked
about.

**Numbers this removes** — `loops.spatial.identity_reject_factor`, the drift
budget on this path (`drift_floor_m`, `drift_rate_m_per_m`),
`mask_filter.min_inside_frac / max_drop_frac / min_votes`, and the three recall
thresholds of `loops.reprojection`.

**Where a number could sneak back in** — telling a "concentration" from dust
needs a notion of scale. It must come from the mask itself (the object's own
mask extent in that frame defines what concentrated means for that object),
never from a picked constant. If that turns out not to be enough, say so
BEFORE inventing one.

---

## 1. The mask→cloud matcher runs TWICE per pipeline run

**Observed** — pccr run of 2026-09-14 23:59, `logs/server_20260914_235915.log`:
the full matching stage runs to completion (line 1517 → 1764: 52 instances,
85.3 % coverage), saves `segmentation_result.json`, rebuilds the Potree octree,
logs `[sam3] Segmentation complete: 52 instances` — and then a SECOND full pass
starts at line 1773 over the same 28,412,874 points, repeating the geometric
mask filter and the eight space-dedupe merges with identical numbers.

**Why it matters** — each pass maps 28 M points, projects every instance into
its mask keyframes and re-runs the dedupe. It is several minutes of identical
work, and it grows with the cloud. Nothing is wrong with the RESULT (the second
pass is deterministic and lands on the same 185 → 177), so this is cost, not
correctness.

**Where to look** — two call sites of `_match_masks_to_cloud`,
`segmentation/pipeline.py:3033` and `:3165`. The second is the cached-loading
path (`apply_segmentation_to_cloud`), which should have found the freshly
written result and returned it instead of re-matching. Find out what makes the
cache look stale — the Potree rebuild and the `frames_valid/` cleanup both
happen between the write and the second call.

---

## 2. The split gate fragmented the floor and the ceiling

**Observed** — same run, during CERTIFY: seventeen `instance-split` actions and
still going when the design above was agreed. The floor 44 lost three pieces
(597,339 / 10,900 / 8,075 points), the ceiling 151 lost five (380,562 /
168,836 / 57,677 / 31,776 / 15,508 / 1,247), each split repainting the masks
and rebuilding `scene_r.db` over 28 M points. The rule that fired every time:

    copies N m apart > 0.90 m: not drift — two objects fused

0.90 m is `identity_reject_factor: 3.0` × δ(L) = `max(drift_floor_m 0.30,
drift_rate_m_per_m 0.013 × walk)`. On this scene the 0.30 FLOOR won, so the
threshold that cut the floor and the ceiling is an invented floor times an
invented factor. The `3.0` was introduced by commit 9365535 (2026-09-13) with
no derivation; the `0.013` claims "measured Omega drift 1.3 cm/m" in its YAML
comment but pccr measured 3 cm/m (commit a32bf58) — a measurement from another
scene, more than 2× off here.

**The fix is item 0**: nothing is ever split. The audit says which
concentration is out of place, and that feeds the CORRECTION.

**One piece of the machinery worked and should be kept** — bounded objects were
protected: `instance 113 (black_office_chair) reprojection → same_object: one
copy lands on the other's mask under a single rigid shift` … `the frames
overrule the split`.

**SUPERSEDED READINGS, do not re-derive**
- *"the arbitration cannot judge extended surfaces, so it abstains"* — wrong.
  The recall pair is 0.99 / 0.00: it measures fine and says copy B is not on
  this instance's mask anywhere.
- *"self-recall ~0 is evidence, so those points should LEAVE the instance the
  way the mask filter removes points"* — the first half stands, the second is
  wrong. Nothing leaves. Off-mask concentrations are declared, not deleted.
- *coordinate frames are NOT the problem*: `_reproject_verdict` passes
  `session.xyz`, the raw frame, same as the poses — checked.

---

## 3. `drift_min_gain: 0.2` still decides by an invented number

The last threshold in the correction path that REJECTS something with no
measurement behind it. Its replacement is item 0's audit, the same yardstick as
everything else. Not urgent: the drift model only runs when the greedy loop
accepts nothing.

---

## 4. The "no decision literals" test only covers ONE package

USER 2026-09-15: *"sin números hardcodeados, cosa que debemos corregir en todos
los casos"*.

`tests/test_correction_config.py::test_no_decision_literals_outside_config`
scans `server/correction/` and nothing else (`PKG` at line 16). Every literal
in `segmentation/`, `reconstruction/loops/`, `reconstruction/certify/` and
`reconstruction/quality/` is unchecked — which is precisely where the numbers
that decided the splits and the trims live. Extend the scan, package by
package, and make each offender either a config key with its provenance in the
comment or a measurement.

---

## 5. The arbitrary-numbers ledger in CLAUDE.md has one wrong entry

It files the drift budget (`spatial.drift_floor_m` / `drift_rate_m_per_m`)
under "INVENTED BUT ONLY WARNS". In the pose graph it does only warn. In the
split path it **decides**, and it cut the floor and the ceiling. Move it, and
re-check the rest of that section the same way: for each number, find every
call site before classifying it.

---

## 6. `coverage_sample.cover_keyframes` can never run where it is called

OBSERVED on the pccr run of 2026-09-16 23:04: the VLM stage logs

```
[cover] no camera evidence — coverage cannot be measured
coverage unavailable — falling back to 8 evenly spaced keyframe(s)
```

**The message is wrong and the dependency is wrong.** `cover_keyframes` asks
`surface_fit.hole_audit._evidence`, whose `__init__` returns on its second line
when `output/seg_masks.npz` is missing — the SAM3 MASKS. The coverage sampler
runs in the VLM stage, which is BEFORE SAM3, so that file cannot exist yet and
the fallback fires on every run. The cameras were there the whole time
(`camera_poses.txt`, `intrinsic.txt` written by the reconstruction).

Coverage is a purely geometric question — which cameras see the whole cloud —
and needs the cloud, the poses and K, never a mask. Fix: call
`segmentation.session_io._load_camera_source(session_dir, output_dir)` directly
(what `_Evidence` uses internally for the cameras) and leave `_evidence` to the
post-SAM3 hole audit. And say which file is missing, not "no camera evidence".

## 7. The σ floor's third tier measures something looser than the other two

`certify/repeatability.py` (2026-09-16) reads, in order: `uncertainty.json`
(two copies of the SAME frame), `elastic_seams.json` (per-shared-frame residual)
and `intra_chunk.json` (held-out agreement between DIFFERENT frames of one
chunk). The third exists so a SINGLE-CHUNK session can still measure itself —
but it is not the same question, and on pccr the numbers differ by 6x:

```
uncertainty.json   4.77 cm      elastic_seams  ~3.5 cm      intra_chunk  ~29 cm
```

A single-chunk session would therefore get a σ floor six times looser than a
chunked one — not because it is less precise, but because the only measurement
available is laxer. The acta declares the source, so nothing is hidden, but σ
floors are NOT comparable across sessions of different shape. Fix: find a
single-chunk measurement of the same tightness (frames vs their own copies, not
frames vs each other) or declare the tier explicitly as a different quantity.

## 8. USER VERDICT 2026-09-16 — the closed cloud is much better

After the run above (loop closure applied for the first time: 74.6 -> 1.0 cm
over a 44 m walk, scale verification passing at 0.9844): *"la nube es mucho
mejor que antes, aunque sí le falta para ser perfecta, pero está mucho mejor"*.

What made it possible, in order: the bridge verdict stopped rejecting by an
invented 10 cm and started measuring σ by split-half held-out (0.170 m, judged
against the session's OWN 5.1 cm seam agreement); and the shared per-frame
field stopped being overwritten by the elastic stage, so the closure survived
on the camera side. Both are in the 2026-09-16 commits; the second is the bug
that made the first look destructive.

## 9. A copy pair with a 5.5x scale ratio wrote a scale row

pccr 2026-09-17, certification: `exposed_metal_wiring#185 kf 202<->1` measured
`s_ab 0.1805` — one copy 5.5x the size of the other — and the pair WROTE a
scale row. Two runs of wiring of different lengths matched as one object; the
ICP residual (2.4 cm) does not betray it because the fit absorbs the scale.

The aggregate then came out at `|log r| 0.283` and only `scale.max_correction_log:
0.2` stopped it: **a single poisoned row nearly applied a 33 % scale correction
to a session that had just verified at 1.6 %**. The limit declared and applied
identity, which is the right failure — but it is a blunt backstop, not a fix.

A copy pair whose measured scale ratio is that far from 1 is not evidence of
scale, it is evidence of a MIS-MATCH, and it should be declared as such where it
is measured (loops_posthoc.copy_scale_rows) instead of travelling to the solver.

## 10. The reprojection verdict passed two false duplicates the greedy caught

Same run: `desk#137` (kf 197<->1, 48 cm apart, s_ab 1.12) and
`exposed_metal_wiring#185` both came out `ambiguous` from the reprojection
verdict — kept with σ×3 — and BOTH were then rejected by the greedy loop for
the only reason that matters: applying them pushes points OFF their masks
(7,413,187 and 7,232,317 against a 7,079,933 baseline).

So the σ inflation and the greedy did their job, but the verdict that exists to
answer "are these two copies the same object?" answered "maybe" for two pairs
that are demonstrably not. Every one of the 21 pairs except one came out
`ambiguous` (`min_self_recall` / `min_cross_recall: 0.4`, `min_agreeing_frac:
0.6` — all three in the CLAUDE.md ledger as invented and deciding). A verdict
that says "maybe" to everything carries no information: it should be measured
against what the session itself achieves, the way the loop bridges now are.

## 11. The geometric revisit detector found 37 co-visible pairs and ZERO regions

pccr 2026-09-16 (`output/corrections/revisits.json`): `n_pairs: 37`,
`regions: []`. On 2026-09-14 04:00 — the run that produced the epoch chain the
user judged good — the same detector reported `4 revisit region(s) → 2 loop
edge(s) (2 full, 2 duplicated)`, and those geometric edges are what the pose
graph closed into epoch 1. Today's 23 loop edges are ALL instance loops; the
geometric source contributed nothing.

`revisit.py` is unchanged since that run (`git log` on the file stops at
458fa28, 2026-09-14 04:52), so the collapse comes from the data, not the code.
A region survives three gates in `_regions_from_hits`: a `region_cell_m` voxel
block with ≥ `evidence.min_object_points_solve` (300) hits, ≥2 visits separated
by `revisit.min_gap_kf` (30), and ≥300 hits in the (early, late) cross subset.
Which of the three empties is NOT measured — `detect_revisits` writes only the
surviving regions, never the histogram of what the blocks held. It should
report the counts per gate, the way every other stage declares what it dropped.

Until then it is unknown whether the new reconstruction genuinely revisits less
(it walks the same 19.3 m) or whether one of the three gates now fires on a
scene it did not fire on two days ago.

## 12. The single-witness drop runs before the masks exist — declared limit

`witness.drop_statuses: [single_witness]` (USER 2026-09-17) removes the points
inside `gpu_cloud_clean`, which is the earliest place where "no quiero que esos
voladores se usen para computar nada, ni para comparar ni para siquiera
calcular el OBB" is actually true: the VLM, SAM3, the mask↔cloud matching, the
OBBs, the reprojection verdicts and the certification all read the cloud that
step writes.

The price is stated, not hidden: at that moment there is no segmentation, so
`mask_votes` and `mask_conflicts` are zero for every point (verified on pccr
2026-09-16 — both columns are all-zero in the delivered cloud) and the status
is the PROVISIONAL, multi-view-only one. A point that fewer than two
neighbouring keyframes agreed with, but whose own SAM3 mask would later have
confirmed it, is removed and never gets that second chance. The criterion is
purely geometric by construction, which is what was asked for.

If a scene ever loses real surface to this, the fix is not a threshold: it is
to run the witness a second time after segmentation (the module already
supports it, `reconstruction.witness.run`) and drop then, at the cost of the
OBBs being computed on the unfiltered cloud — which is exactly what the user
did not want.

## 13. The known-answer envelope stopped being monotone with loop density

`test_certify_f3.py::test_known_answer_and_envelope_monotone_with_loop_density`
passes at 2a63e53 and fails at 4eac771. Reverting the drift-consensus change
did NOT fix it, so the cause is one of the two corrections to how a closure is
applied: the SE(3) screw distribution (`correction/distribute.py`) or the
projection about the object's centroid (`correction/solve.py`).

What still passes is the part that matters most: the injected error lands, the
instrument recovers it, the error after is smaller than before and the
recovered fraction is in (0, 1]. What fails is
`env["monotone_with_loop_density"]` — the envelope of correctable error is
supposed to GROW as more loops are available, and with the new maths the
ordering across densities broke.

Both changes are load-bearing and measured: the projection turned a 60 cm
correction that asked for 14 m into one that asks for 17 cm, and it is what
finally produced an epoch on pccr after three days at epoch 0. So the envelope
regression is not a reason to revert them — it is a reason to find out which
density inverted and why. The per-density dict is truncated in pytest's
assertion repr; run the test with `-vv` (or print `env["per_density"]`) to get
the numbers.

Hypothesis to test first: the envelope is probed by injecting increasing error
until the §9 gates fail, and the screw distribution changes the SHAPE of what a
given closure applies at intermediate keyframes. A denser loop set may now hit
a gate earlier for a reason that has nothing to do with how much error is
correctable.

## 14. A LOCAL correction is spread over the WHOLE walk

USER 2026-09-17, comparing epochs 1 and 2 in the viewer: in epoch 2 "el piso
prácticamente logró fusionarse", and at the same time "hay zonas cercanas a la
de conflicto que han empeorado un poco, se nota en las costuras de algunas
paredes".

Both are true, and together they explain the objective. Epoch 2 was applied
after `revisit_region_4` — the first geometric region the greedy ever accepted
— and the objective called that iteration a 49% regression while closure
(0.36 → 0.26 m), seams (0.0254 → 0.0229) and duplicates (16 → 13) all
improved. The objective was not wrong: it was seeing the collateral damage the
user can see on the wall seams. Each of us was looking at one half.

The cause is a mismatch of scale. The evidence is LOCAL — a 3 m voxel block
that measures 22 cm — but `distribute` applies it through the drift-rate model,
which assumes the error accumulates smoothly with the distance walked and
therefore moves EVERY keyframe in proportion to its chainage. A localised
error spread that way fixes its own corner and displaces geometry that was
already right.

The drift-rate model is correct for what it was written for (USER 2026-09-09:
the accumulated error of the walk, pinned by a duplicate). It is the wrong
carrier for a region that says "this block, and only this block, is 22 cm out".

Measured on pccr after the run: mask_conflict 6.93% → 7.18% and verified
91.06% → 90.59%, i.e. the point-status criterion did not improve even though
the floor visibly fused — consistent with gaining in the corner and losing
around it.

## 15. The epoch transaction re-dirties the vote field

`witness.drop_statuses` removes the single-witness mass inside
`gpu_cloud_clean`, at merge time. But the epoch transaction recomputes the
witnesses on the warped cloud and nothing applies the drop there, so every
epoch brings new points with fewer than two agreeing views: pccr epoch 2 has
383,001 of them (1.73%) where the delivered cloud had none. The user's
acceptance criterion is that Votes and Point status are both green, so the
drop has to be part of the transaction too — or the field has to be declared
as provisional inside an epoch.
