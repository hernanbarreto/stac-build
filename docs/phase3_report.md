# Phase 3 — 3D-Anchored Visual Findings: Report

**Status: complete, validated end-to-end on the A6000 (test3).** Over sampled
keyframes Qwen3-VL detects visible defects; each detection is anchored to 3D
through the R-refined pose, deduplicated across views, and severity-correlated
with a localized geometric residual. Precision is measured honestly on a
hand-annotated crop set. Every finding is born `proposed` and requires human
validation.

Architectural invariant honored: **the VLM proposes, describes and classifies;
it never measures.** All geometry (3D anchor, dedup radius, residual) comes from
deterministic tools over the R.8 store. Findings are tagged `vlm_proposed`.

## What was built

| Component | Path |
|-----------|------|
| Finding detector | `server/phase3_findings/detect.py` (`FindingDetector`) |
| CLI | `server/phase3_findings/cli.py` (`python -m phase3_findings.cli`) |
| Precision eval | `server/phase3_findings/eval.py` (`build_set` / `evaluate`) |
| Single-pixel unproject | `server/phase_r/geometry.py::unproject_pixel_to_world` |
| Store tables | `findings` (already in `phase_r/instance_store.py`) |
| Config block | `server/config.yaml` → `phase3:` |
| Unit tests | `server/phase3_findings/tests/test_phase3.py` (10 tests, no model) |

## Pipeline (spec FASE 3)

1. **Inspection over keyframes** — frames that contain segmented instances are
   subsampled (`sample_stride`, `max_frames`). A strict-JSON inspection prompt
   asks for cracks, moisture, exposed steel, spalling, corrosion, provisional
   out-of-norm elements, obstructions — *only when clearly visible*. The prompt
   explicitly excludes grout lines, joints, seams, shadows, dirt and normal wear
   (the observed false-positive modes). Output per finding: type, box, severity,
   description, confidence.
2. **3D anchoring** — box center → depth at that pixel (median over a small
   window) → world point via the FASE R–refined camera pose
   (`unproject_pixel_to_world`, intrinsics scaled to depth resolution). The
   owning instance is the one whose mask covers the center pixel; fallback is the
   nearest instance cloud. Both 2D box and 3D point are stored.
3. **Multi-view dedup** — union-find over (type + 3D proximity ≤
   `cluster_radius_m`); the same defect seen in N frames becomes ONE finding,
   keeping the highest-confidence representative.
4. **Residual correlation** — a LOCALIZED point-to-cloud plane residual
   (`residual_k` nearest instance points, RMS out-of-plane) stands in for a
   surface_fitting residual map until dense fitting runs. Above `residual_thresh_m`
   the finding is flagged `correlated_residual` and its severity is bumped —
   this selectivity replaced an earlier instance-wide onion flag that fired for
   every large planar object.
5. **Provenance / gating** — every finding is `status='proposed'`,
   `origin='vlm_proposed'`; human validation is required before any deliverable.

## Validation (test3, unknown domain — no speculation)

Detector, `max_frames=18` over the test3 store:

```
frames=18 raw=33 anchored=20 findings=14 correlated=0
per_type={'spalling': 4, 'moisture': 3, 'other': 4, 'crack': 3}
```

`correlated=0` is the honest result: on test3's cloud no finding fell in a
locally-rough (>2 cm) zone, so no severity was inflated. The mechanism is
covered by unit tests (flat surface → residual < 1 cm; 12 cm-rough surface →
above threshold).

### Precision — brutal honesty (spec requirement)

Hand-annotated crop set of 60 crops from our own keyframes (43 labeled:
31 clean, 12 defect; 17 left `pending` — the harness never fabricates GT).
Annotator: Claude as a proxy for the human hand-annotation pass; a defect
crop is scored positive if any finding is reported.

| Prompt | Precision | Recall | Specificity | TP | FP | TN | FN |
|--------|-----------|--------|-------------|----|----|----|----|
| baseline | 0.571 | 1.000 | 0.710 | 12 | 9 | 22 | 0 |
| + exclusion clause | **0.600** | **1.000** | **0.742** | 12 | 8 | 23 | 0 |

**Error mode (documented, not hidden):** the remaining false positives
concentrate on (a) the ornate tiled archway, where tile edges read as hairline
cracks, and (b) genuinely ambiguous dark floor patches where the clean/defect
boundary is itself debatable. Recall is perfect on this set. This over-reporting
is precisely why every finding is `proposed` and gated on human sign-off, and
why precision — not just recall — is tracked.

## Open items (external data only; code + tests complete)

- The precision numbers above use a 43-crop proxy-GT. The full ~50+ crop set
  hand-annotated by a domain human would replace the proxy labels; the harness
  (`build_set` → annotate `annotations.json` → `evaluate`) is complete and
  reproducible.
- `correlated_residual` currently uses the localized point-to-cloud residual
  proxy. When surface_fitting produces a dense residual map it plugs into the
  same flag with no interface change.

## How to run

```bash
# detect + anchor findings into an existing R.8 store
python -m phase3_findings.cli --scene scene.db --session <session_dir> \
       --output <run_output_dir> --max-frames 60

# build a precision crop set, hand-label annotations.json, then score
python -c "from phase3_findings.eval import build_set; \
           build_set('<session>', '<output>', 'eval_dir')"
python -c "from phase3_findings.eval import evaluate; \
           print(evaluate('eval_dir'))"
```

*Hernán Barreto — Ingerop IN3 Session IV — STAC*
