"""Correction module (USER 2026-09-08 redesign; automatic since 2026-09-18).

Converts the parallel-copies symptom of long-walk drift into loop closures
measured on the objects the scan saw twice (``visit_drift`` +
``visit_drift_run``, run by the certification stage of the pipeline), solves
the depth/scale they observe, re-levels the floor (``run.run_floor``) and
applies transactionally under a geometry epoch + append-only ledger.

The MANUAL flow — the UI "Corrections" button: mark objects → evidence →
observability → solve, the revisit closure, and its ``/api/correction/*``
router — was REMOVED on 2026-09-24 by the user's order. What the user still
decides is which epoch is on screen (``/api/certify/select``).

Doctrine (CLAUDE.md + USER 2026-09-06):
  * The cloud is the truth; only this module modifies it, through
    apply → validate → select the epoch to show.
  * An applied correction IS the cloud; the next one runs on top of it, and
    every epoch stays selectable (USER 2026-09-16).
  * The person proposes, the geometry measures (everything tool_measured,
    applications tagged human_directed).
  * The rest of the scene is the exam; a failed gate applies NOTHING.
  * Fail-fast; no silent fallbacks; nothing hardcoded (config.yaml
    ``correction:`` holds every parameter).
"""

from correction.config import CorrectionConfig, CorrectionConfigError, load_correction_config  # noqa: F401
