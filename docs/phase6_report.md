# Phase 6 — Report Generation: Report

**Status: complete, validated end-to-end on the A6000 (test3).** A per-scene
supervision-report draft generator that assembles every upstream phase into a
traceable bilingual (ES/FR) Markdown document with image assets. The generator
measures nothing itself — it reads `tool_measured` values and labels VLM text as
proposed.

## What was built

| Component | Path |
|-----------|------|
| Report builder | `server/phase6_report/report.py` (`ReportBuilder`) |
| Bilingual strings | `server/phase6_report/i18n.py` (ES / FR) |
| CLI | `server/phase6_report/cli.py` (`--lang es|fr|both`) |
| Config block | `server/config.yaml` → `phase6:` (added below) |
| Unit tests | `server/phase6_report/tests/test_phase6.py` (4 tests, no model) |

## Sections (spec FASE 6)

1. **Inventory (Phase 2)** — id, class, material, state, OBB dimensions; class/
   material/state carry a provenance marker (⚠️ proposal, or ✅ validated) and a
   label-conflict flag; dimensions are `tool_measured` with a trace.
2. **Findings (Phase 3)** — sorted high-severity first: type, severity, 3D
   coordinate, owning instance, confidence, description, a cropped figure from
   the source frame, and the provenance marker (proposal unless human-validated).
   `correlated_residual` findings note the geometric correlation.
3. **Measurements (Phase 5 tools)** — per-instance volume, plus plumb (walls/
   columns/beams) and level (floors/slabs) for structural classes; each value
   carries its `tool(args)` + timestamp trace and is `tool_measured`.
4. **Coverage (Phase 4)** — reads a `coverage_report.json`; lists under-sampled
   zones as a recapture checklist, or states none were detected.
5. **Reconstruction quality (Phase R)** — per-instance onion metric (bimodal +
   separation) and vote entropy, dynamic-pixel fraction, and any logged
   scale/marker conflicts — each with its trace.

## The two invariants (enforced + tested)

- **Traceability** — every number is followed by `tool(args), timestamp`. The
  timestamp is *injected* (CLI passes UTC now, tests pass a fixed value) so
  reports are reproducible and deterministic.
- **Provenance** — VLM observations (class, material, state, finding text) are
  marked *proposal pending validation* unless their store status is
  `human_validated`; geometry is marked *tool_measured*. Verified by
  `test_traceability_and_provenance_es`.

## Validation (test3)

Generated `report_es.md` and `report_fr.md` (14 finding figures each) from the
test3 store. Excerpt (ES):

```
| 3 | tiled floor | concrete | stained | 3.45×0.77×1.86 m (get_object_size(id=3), …) | ⚠️ propuesta … |
### Tipo: spalling — Severidad: alta
- Ubicación 3D (m): 0.12, -0.25, 1.23 · Instancia: 3 · Confianza: 0.95
- ⚠️ propuesta (pendiente de validación)
```

FR renders the same structure with translated labels (`Rapport de supervision`,
`Constats`, `Mesures`, …). All measurement rows are `tool_measured` with traces;
all inventory/finding VLM text is flagged as proposal (nothing was human-
validated in this store).

Honest note: on this single-window store no gravity was estimated, so `get_plumb`
/`get_level` report large tilts against the assumed +Y-up axis — correctly tagged
`tool_measured` with the "supply gravity" caveat from the Phase 5 tool. With a
gravity-aligned store these become georeferenced.

## Open items (external data only; code + tests complete)

- Human validation of findings/classifications flips their markers from proposal
  to validated — a data/workflow step; the marker logic and store field
  (`status='human_validated'`) are implemented and tested.

## How to run

```bash
python -m phase6_report.cli --scene scene.db --session <session_dir> \
       --out report_dir --lang both --coverage cov_dir/coverage_report.json
```

*Hernán Barreto — Ingerop IN3 Session IV — STAC*
