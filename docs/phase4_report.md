# Phase 4 — Capture QC: Report

**Status: complete, validated end-to-end on the A6000 (test3).** Two consumers:
an ingestion pre-filter (before DA3/VGGT) and a post-reconstruction coverage
report that emits a recapture checklist. Cheap classical metrics do the bulk of
the work; the VLM is invoked only where it adds value.

Architectural invariant honored: **the VLM judges image usefulness and phrases
recapture instructions; it never measures geometry.** Blur/exposure metrics,
voxel density, zone grouping and frame visibility are deterministic.

## What was built

| Component | Path |
|-----------|------|
| Classical metrics | `server/phase4_qc/metrics.py` (Laplacian variance + exposure) |
| Ingestion pre-filter | `server/phase4_qc/prefilter.py` (`CaptureQC`) |
| Coverage report | `server/phase4_qc/coverage.py` (`CoverageReport`) |
| CLI (two subcommands) | `server/phase4_qc/cli.py` (`prefilter`, `coverage`) |
| Config block | `server/config.yaml` → `phase4:` |
| Unit tests | `server/phase4_qc/tests/test_phase4.py` (10 tests, no model) |

## 4A — Ingestion pre-filter (spec: cost-aware, never silently drops)

Every sampled frame gets cheap metrics: variance of the Laplacian (blur) and
exposure (mean luma + clipped-pixel fractions). Classical triage:

- `lap_var ≤ blur_drop_below` → **drop** (logged, no VLM)
- `lap_var ≥ blur_keep_above` and exposure OK → **keep** (no VLM)
- otherwise (borderline blur or suspicious exposure) → **escalate to VLM**

The VLM (only on the ambiguous band) judges blur, **occlusion by people/
equipment**, exposure and general usefulness, and makes the keep/drop call.
Frames are **never deleted**: `write()` emits `qc_manifest.json` and a
`qc_drop_log.txt` recording exactly which frames were rejected and why. When
`escalate_to_vlm=false`, ambiguous frames are kept and flagged — never silently
dropped.

### Validation (test3, 26 frames, stride 12)

```
frames=26 kept=14 dropped=12 vlm_escalated=20
```

- Classical alone decided the clear cases (sharp keeps; frame 300 dropped on blur).
- The VLM caught motion blur the Laplacian rated borderline — e.g. frame 36
  (`lap_var=119`, just under the keep threshold) → VLM `severe blur, dark` →
  drop; frames 84/96 (`lap_var≈85`) → `slight blur, usable` → keep.
- Only the ambiguous band hit the VLM (20 of 26); 6 were settled classically.

## 4B — Coverage report → recapture checklist (spec)

The pooled instance cloud is voxelized (`voxel_size_m`). Occupied voxels are
grouped into 6-connected components; a component whose **total** point count is
in the bottom `low_density_pct` percentile (and above the `min_zone_points`
noise floor, and below `max_zone_fraction` of all points — so the main body is
never flagged) is an **under-sampled zone**. For each zone the frames that view
its centroid are recovered by projecting through the R-refined poses
(`frames_viewing`), and the VLM phrases a plain-language recapture instruction
on the best frame. `write()` emits `coverage_report.json` and
`recapture_checklist.txt`.

### Validation (test3)

test3's reconstruction is a single dense connected body (floor + walls + door
touch), so **0 isolated under-sampled zones** — the honest result; the earlier
degenerate behavior (flagging the whole 1.6 M-point cloud as one "zone") is
fixed by the `max_zone_fraction` guard and locked by a unit test. The positive
path (an isolated sparse cluster → one zone at its centroid, with a written
recapture instruction) is covered by `test_low_density_zone_isolates_sparse_region`
and was exercised live (e.g. *"Re-capture the top of the stairs and the archway
from a lower angle, closer to the steps, to fill in the sparse 3D points."*).

## Open items (external data only; code + tests complete)

- A multi-region scan with a genuinely disconnected under-scanned corner is
  needed to exercise 4B's positive path on real data end-to-end; the geometry is
  unit-tested and the code path ran live. No code pending.

## How to run

```bash
# 4A ingestion pre-filter
python -m phase4_qc.cli prefilter --session <session_dir> --out qc_dir [--no-vlm]

# 4B coverage + recapture checklist
python -m phase4_qc.cli coverage --scene scene.db --session <session_dir> \
       --output <run_output_dir> --out cov_dir
```

*Hernán Barreto — Ingerop IN3 Session IV — STAC*
