"""User-directed correction module (USER 2026-09-08 redesign).

Converts the parallel-copies symptom of long-walk drift into a human-directed
loop closure: the user marks duplicated objects, the geometry solves,
distributes per keyframe, validates against the rest of the scene and applies
transactionally under a geometry epoch + append-only ledger.

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
