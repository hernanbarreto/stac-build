# scale_align v2 — A/B acceptance criteria (defined BEFORE running)

Precision task, Phase A.6. This file is committed **before** the A/B runs; the
verdict is judged against it with no post-hoc goal-moving (same discipline as
the bundle_adjust A/B).

## Harness

`server/tools/scale_ab.py` over **3 sessions** (test4 = long outdoor walk /
chunked-metric; test7 = longest available; test2 = short single-pass) after
`server/tools/extract_da3_anchors.py --count 32 --pairs 3` repopulates the DA3
anchors freed after the original runs.

Internal-consistency metrics (no external reference — that is Phase E):

1. **held-out anchor depth error (%)** — k-fold by frame; primary
   generalization metric.
2. **jackknife stability** — leave-one-anchor-out spread of s (max relative
   deviation, MAD); primary stability metric.
3. **eval-pair depth reprojection error (%)** — DA3 metric depth of a keyframe
   projected into its consecutive neighbour at the candidate s; discriminates
   the absolute s.
4. **plane-patch RMS (mm)** — reported for the record; between pure-scalar
   candidates it only reflects the s ratio (stated in the harness output).

## Criteria

### Estimator mode (`scale_mode` default)

`affine_robust` (or `depth_dependent`) replaces `global_median` as the config
default **only if, on at least 2 of the 3 sessions**:

- held-out anchor depth error improves by **≥ 5% relative** vs `global_median`,
- jackknife max relative deviation is **not worse by > 20% relative**,
- eval-pair reprojection error is **not worse by > 2% relative**,

**and** on no session is any of the three metrics worse by **> 10% relative**
(no catastrophic regression allowed). `depth_dependent` is additionally judged
against the `affine_robust` result with the same bar (the ladder must be earned
step by step).

If a structured mode auto-degrades to scale-only on ≥ 2 sessions (CV/BIC gate),
that is itself the verdict: the structure is not supported by our scenes, the
default stays `global_median`, and the finding is documented.

### Anchor count (`scale_anchor_frames` default)

24 (or 32) replaces 12 **only if, on at least 2 of the 3 sessions**:

- jackknife max relative deviation improves by **≥ 30% relative**, and
- held-out anchor depth error does not worsen by **> 2% relative**.

Cost context: DA3 anchor extraction is seconds per frame (one-off per run) —
negligible vs the pipeline, so the bar is about statistical benefit, not cost.

### VIO source

Cannot be A/B'd on the existing sessions: **no VIO recording exists for them
(genuine EXTERNAL data dependency — needs a capture app that exports the
trajectory, see docs/VIO_FORMAT.md).** All VIO code paths are covered by
synthetic unit tests (drift injection, segment robustness, fail-hard gates);
the first real VIO session will report the VIO↔DA3 agreement automatically.

## Outcome recording

The A/B tables (scale_ab.md/json per session) + the verdict against these
criteria go into the Phase A close-out summary; the chosen defaults land in
config.yaml with the justification, and losers stay implemented but OFF —
evidence over preference, no exceptions.
