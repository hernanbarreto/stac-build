"""§6.3 per-point ``status`` from configurable rules (``witness.rules``).

    unobserved      no witness could be computed for the point's frame
    verified        mv_votes ≥ verified_min_mv_votes and
                    mask_conflicts ≤ verified_max_mask_conflicts
    single_witness  fewer corroborating views than the rule asks for (only
                    its own frame — or too few neighbours — vouch for it)
    mask_conflict   mask_conflicts > mask_votes and ≥ conflict_min: the
                    point lands in OTHER instances' masks more often than in
                    its own (a segmentation leak or a misplaced point)
    dynamic         the point belongs to an instance classified dynamic
                    (§4.4) — never a geometric witness

Precedence: dynamic > mask_conflict > verified > single_witness; unobserved
only when no multi-view witness exists AND no mask says otherwise.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

STATUS_CODES: Dict[str, int] = {"unobserved": 0, "verified": 1, "single_witness": 2,
                                "mask_conflict": 3, "dynamic": 4}
STATUS_NAMES: Dict[int, str] = {v: k for k, v in STATUS_CODES.items()}


def assign_status(mv_votes: np.ndarray, observed: np.ndarray, mask_votes: np.ndarray,
                  mask_conflicts: np.ndarray, dynamic: np.ndarray, rules) -> np.ndarray:
    """uint8 status codes (STATUS_CODES) for every point."""
    mv = np.asarray(mv_votes, np.int32)
    obs = np.asarray(observed, bool)
    mvt = np.asarray(mask_votes, np.int32)
    mct = np.asarray(mask_conflicts, np.int32)
    dyn = np.asarray(dynamic, bool)
    st = np.full(len(mv), STATUS_CODES["unobserved"], np.uint8)
    single = obs & (mv < int(rules.verified_min_mv_votes))
    verified = obs & (mv >= int(rules.verified_min_mv_votes)) & \
        (mct <= int(rules.verified_max_mask_conflicts))
    # verified needs the multi-view witness; a mask alone never verifies
    st[single] = STATUS_CODES["single_witness"]
    st[verified] = STATUS_CODES["verified"]
    conflict = (mct > mvt) & (mct >= int(rules.conflict_min))
    st[conflict] = STATUS_CODES["mask_conflict"]
    st[dyn] = STATUS_CODES["dynamic"]
    return st


def status_counts(status: np.ndarray) -> Dict[str, int]:
    st = np.asarray(status, np.uint8)
    return {name: int((st == code).sum()) for name, code in STATUS_CODES.items()}


def status_mask(status: np.ndarray, names) -> np.ndarray:
    codes = [STATUS_CODES[str(n)] for n in names]
    return np.isin(np.asarray(status, np.uint8), codes)
