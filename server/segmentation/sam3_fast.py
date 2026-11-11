"""SAM 3.1's per-frame planning metadata without the deep copy of its score history (USER 2026-11-11:
"le tenés que encontrar la vuelta, un pccr3108 debe dar 30' como mucho").

MEASURED on pccr 2026-08-31 (cProfile, 'wall' over 120 keyframes, 22.5 s): the vendor's planning phase
builds the next frame's metadata with ``copy.deepcopy`` of the previous one
(``Sam3VideoBase._create_planning_metadata``), and that copy holds the score of every object on EVERY
frame propagated so far (``obj_id_to_sam2_score_frame_wise``: frame → {object id: 0-d tensor}). Copied
again on every frame, it grows with the frames already done — 60,806 tensor deep copies, 4.9 s of the
22.5 (22 %) on 120 frames, more on the 250 of a batch.

Every writer of that history writes the CURRENT frame's entry only (sam3_multiplex_base.py 626 / 1220 /
1232, sam3_multiplex_tracking.py 2429 / 2869, sam3_video_base.py 505 / 1004 / 1015 — ``[frame_idx]``
.update / item assignment) and no score tensor is ever modified in place. :func:`shallow_planning_metadata`
therefore copies the history's per-frame dicts (each frame its own dict, as the deep copy gave) and shares
their score tensors; the other fields are deep-copied exactly as the vendor does. The metadata every
later step reads holds the same values: the masks are the vendor's, bit for bit
(tests/test_sam3_fast.py on the card).

``models.segmentation.planning_copy``: ``shallow`` (this) | ``deep`` (the vendor's own copy)."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

PLANNING_COPIES = ("shallow", "deep")
_VENDOR = {}


def _history_copy(hist):
    """The per-frame score history with every frame's dict copied and its tensors shared."""
    if isinstance(hist, defaultdict):
        out = defaultdict(hist.default_factory)
    else:
        out = type(hist)()
    for frame, scores in hist.items():
        out[frame] = dict(scores) if isinstance(scores, dict) else deepcopy(scores)
    return out


def shallow_planning_metadata(self, tracker_metadata_prev):
    """``Sam3VideoBase._create_planning_metadata`` with the score history copied by
    :func:`_history_copy`; every other field deep-copied as the vendor does."""
    score_key = "obj_id_to_tracker_score_frame_wise"
    if score_key not in tracker_metadata_prev:
        score_key = "obj_id_to_sam2_score_frame_wise"
    return {
        "obj_ids_per_gpu": deepcopy(tracker_metadata_prev["obj_ids_per_gpu"]),
        "obj_ids_all_gpu": None,
        "num_obj_per_gpu": deepcopy(tracker_metadata_prev["num_obj_per_gpu"]),
        "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
        score_key: _history_copy(tracker_metadata_prev[score_key]),
        "obj_id_to_last_occluded": {},
        "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
    }


def install_planning_copy(mode: str) -> str:
    """Make ``mode`` the planning copy of every SAM 3 video model in this process (the method is the
    BASE class's: the multiplex subclass extends it through ``super()``). Returns the mode installed."""
    if mode not in PLANNING_COPIES:
        raise ValueError(f"models.segmentation.planning_copy must be one of {PLANNING_COPIES} (got {mode!r})")
    if mode == "deep" and "vendor" not in _VENDOR:
        return mode                                  # never patched in this process: the vendor's own
    from sam3.model.sam3_video_base import Sam3VideoBase
    if "vendor" not in _VENDOR:
        _VENDOR["vendor"] = Sam3VideoBase.__dict__["_create_planning_metadata"]
    Sam3VideoBase._create_planning_metadata = (shallow_planning_metadata if mode == "shallow"
                                               else _VENDOR["vendor"])
    return mode
