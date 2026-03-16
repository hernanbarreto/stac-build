# STAC-BUILD: BIM Package
# Re-exports for backward compatibility
from bim.comparison import run_comparison, extract_all_ifc_triangles
from bim.registration import register, transform_points
from bim.coverage_store import CoverageStore, ScanCoverageResult, SampleStatus, ElementState, OccluderType
from bim.occlusion_raycaster import (
    load_camera_positions, classify_bim_surface, build_segment_labels,
    find_best_cameras, cylindrical_query,
)
