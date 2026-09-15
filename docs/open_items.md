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
