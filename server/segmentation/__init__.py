# STAC-BUILD: Segmentation Package
# Re-exports for backward compatibility — callers can still do:
#   from segmentation_pipeline import run_segmentation, apply_segmentation_to_cloud
#   from segmentation_manager import SegmentationManager
#   from sam3_wrapper import get_sam3_wrapper
#   from scene_analyzer import analyze_scene, classify_occluders

from segmentation.pipeline import (
    run_segmentation,
    apply_segmentation_to_cloud,
    _save_masks,
    _parse_raw_masks,
    _match_and_save_result,
    _load_ply_origins,
)
from segmentation.sam3_wrapper import get_sam3_wrapper, SAM3Wrapper
from segmentation.scene_analyzer import analyze_scene, classify_occluders
from segmentation.manager import SegmentationManager
