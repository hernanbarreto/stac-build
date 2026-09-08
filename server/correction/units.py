"""Correction units: the KEYFRAME is atomic; visits are keyframe runs.

What it decides:
  * ``visits_from_keyframes``: groups the keyframes that observe a marked
    object's points into VISITS — maximal runs of consecutive keyframes,
    bridging gaps of at most ``evidence.visit_gap_kf`` (evidence keyframes are
    sparse; a small hole is still the same physical visit).
  * ``load_chunk_plan``: the ONLY way chunks enter this module — the plan the
    reconstruction persisted (``output/chunk_plan.json``, written by
    workers/map_worker.py). No plan (phase-1 single pass) means no chunks and
    the module works purely per keyframe. Deriving chunk membership from a
    fixed divisor is forbidden (H1) and structurally impossible here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CHUNK_PLAN_FILE = "chunk_plan.json"


def load_chunk_plan(output_dir) -> Optional[dict]:
    """The persisted reconstruction chunk plan, or None (single-pass session).
    A malformed plan is an error, never silently ignored."""
    p = Path(output_dir) / CHUNK_PLAN_FILE
    if not p.exists():
        return None
    try:
        plan = json.loads(p.read_text())
        ranges = plan["chunk_ranges"]
        if not isinstance(ranges, list) or not all(
                isinstance(r, list) and len(r) == 2 for r in ranges):
            raise ValueError("chunk_ranges must be a list of [start, end)")
        int(plan["chunk_size"]); int(plan["overlap"]); int(plan["n_keyframes"])
        return plan
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"corrupt {CHUNK_PLAN_FILE} at {p}: {e} — re-run the "
            f"reconstruction (map_worker persists the plan) or delete the "
            f"file if this session was a single pass") from e


def chunks_of_keyframe(plan: Optional[dict], kf: int) -> List[int]:
    """Chunk ids whose [start, end) range holds the keyframe (overlapping
    chunks share keyframes by design). Empty when there is no plan."""
    if plan is None:
        return []
    return [i for i, (a, b) in enumerate(plan["chunk_ranges"]) if a <= kf < b]


def visits_from_keyframes(kfs: np.ndarray, gap_kf: int) -> List[List[int]]:
    """Group sorted unique keyframe indices into visits: consecutive runs with
    gaps ≤ gap_kf bridged. Returns a list of sorted keyframe lists, earliest
    first."""
    uniq = sorted(set(int(k) for k in np.asarray(kfs).ravel() if k >= 0))
    visits: List[List[int]] = []
    for k in uniq:
        if visits and k - visits[-1][-1] <= gap_kf + 1:
            visits[-1].append(k)
        else:
            visits.append([k])
    return visits
