# STAC-BUILD: Segmentation Package
# Re-exports for backward compatibility — callers can still do:
#   from segmentation_pipeline import run_segmentation, apply_segmentation_to_cloud
#   from segmentation_manager import SegmentationManager
#   from sam3_wrapper import get_sam3_wrapper
# (the InternVL3 scene analyzer is no longer re-exported: docs/plan_determinismo.md point 156,
#  2026-10-08 — the pipeline has no VLM fallback; its module is not imported by any stage)

# The re-exports are LAZY (PEP 562, 2026-11-01): importing any submodule — segmentation.session_io,
# segmentation.fuse_parent, read by most stage processes — used to import the whole pipeline, the SAM3
# wrapper and the manager first: 9 s of CPU per stage process on this pod (python -X importtime).
_EXPORTS = {
    "run_segmentation": "segmentation.pipeline", "apply_segmentation_to_cloud": "segmentation.pipeline",
    "_save_masks": "segmentation.pipeline", "_parse_raw_masks": "segmentation.pipeline",
    "_match_and_save_result": "segmentation.pipeline", "_load_ply_origins": "segmentation.pipeline",
    "get_sam3_wrapper": "segmentation.sam3_wrapper", "SAM3Wrapper": "segmentation.sam3_wrapper",
    "SegmentationManager": "segmentation.manager",
}


def __getattr__(name):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value
    return value
