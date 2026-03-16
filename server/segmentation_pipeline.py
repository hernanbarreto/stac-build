# Backward-compat shim — module moved to segmentation/pipeline.py
from segmentation.pipeline import *  # noqa: F401,F403

# Private functions also used by main.py (import * skips _ prefixed names)
from segmentation.pipeline import (
    _save_masks,
    _parse_raw_masks,
    _match_and_save_result,
    _load_ply_origins,
    _prepare_valid_frames,
    _run_sam3_batched,
    _prepare_batch_dir,
    _parse_raw_masks,
    _match_ids_iou,
    _compute_obb,
    _find_nearest_keyframe,
)
