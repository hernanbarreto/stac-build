"""Multi-scan fusion + per-measurement metric validation (USER 2026-09-08).

Replaces ``fuse_scans.py`` and ``cloudcompy_register.py``. The metric chain
doctrine:

  * The cloud is the truth; the scale has an OWNER. Every scan declares its
    scale source: ``user_measurement | multiscan_consensus | vio |
    da3_anchors`` (unvalidated). ``bim_registration`` is reserved for the
    future and is a requirement of nothing.
  * NEVER symmetric scale. The reference scan is untouched by the fusion
    (identity, scale 1); every adjustment lands on the scan being fused and
    is recorded. ``reference_scale`` is gone (always 1).
  * A tolerance is not a fix: an out-of-threshold scale difference is
    reported with its diagnosis and rejected (or applied only with a
    recorded override).
  * The person proposes, the geometry measures: the user picks the
    known-dimension element and types its true value; the system measures
    between FITTED PRIMITIVES (surface_fit), never between two clicks.
  * The rest of the scene is the exam (held-out floor / unpaired overlap /
    common unpaired instances). A failed gate applies NOTHING.
  * Fail-fast, transactional, per-point provenance intact in the fused
    product (frame_global/pixel/confidence + scan → a concrete keyframe of
    a concrete scan).

Shares ``server/correction/`` (geometry epoch, ledger, per-keyframe
distribute, transactional apply) — never a local copy.
"""

from fusion.config import (FusionConfig, FusionConfigError,  # noqa: F401
                           load_fusion_config)
