# ADR-001: Pivot to 2D-Primary Analysis Architecture

**Date:** 2026-03-19  
**Status:** Accepted  
**Author:** Hernán Barreto  
**Participants:** Hernán Barreto + AI pair programming

---

## Context

STAC Build was originally designed around a 3D-centric pipeline: video → MapAnything 3D reconstruction → point cloud → BIM comparison (Cloud-to-Mesh). While functional, extensive work on the segmentation pipeline (confidence filtering, RANSAC face detection, voxel mesh visualization, depth peeling, non-destructive point assignment) revealed a fundamental limitation: **the 3D reconstruction pipeline introduces cumulative errors that limit the precision of deviation detection.**

## Decision

**Adopt a dual-engine architecture with 2D analysis as the primary comparison method and 3D reconstruction as a secondary engine for navigation and visualization.**

## Rationale

### 1. Resolution Loss in 3D Reconstruction

Original camera frames capture ~1MP+ per frame (e.g., 720×1280). The reconstruction pipeline (depth estimation → backprojection → chunk merging → SIM3 alignment → SOR → Potree octree) loses significant information at every step. Deviation detection at mm-level requires maximum available resolution — which is in the 2D image.

### 2. SOTA AI Models Operate in 2D

All state-of-the-art vision models are optimized for image-space inference:

| Model | Task | License |
|-------|------|---------|
| PE Spatial (Meta) | Dense spatial features, detection, tracking | Apache 2.0 |
| PLM-8B (Meta) | Scene understanding, material ID (replaces InternVL3) | Apache 2.0 |
| DepthLM (Meta, ICLR 2026 Oral) | VLM metric depth (matches pure vision models) | CC-BY-NC |
| SAM3 | Instance segmentation | Apache 2.0 |

No 3D-native models exist at equivalent quality.

### 3. Perfect Camera Poses Enable BIM→2D Projection

MapAnything provides cam2world poses and recovered intrinsics. Combined with the scan-to-BIM registration (`T_scan→BIM` from gizmo+ICP), BIM elements can be projected onto each camera frame: `pixel = K × [R|t]_BIM × P_BIM`. This enables direct pixel-level comparison.

### 4. 3D Error Chain vs Direct 2D Comparison

3D path: `depth → backprojection → merging → alignment → SOR → C2M deviation`  
2D path: `frame + pose → BIM projection → pixel comparison`  
The 2D path has fewer error sources and preserves original data fidelity.

### 5. PE Spatial Superior to DINOv2/DINOv3

PE Spatial (Meta Perception Encoder, Apache 2.0) outperforms DINOv2 on all dense spatial tasks (ADE20k, DAVIS, LVIS, COCO). DINOv3 was analyzed and rejected — PE Spatial is both superior and commercially licensable.

## Architecture

- **Engine 1 (Primary): 2D Analysis** — PE Spatial backbone + BIM reprojection + pixel-level deviation detection + material identification
- **Engine 2 (Secondary): 3D Reconstruction** — MapAnything → dense point cloud for navigation, visualization, Potree streaming, spatial context

## Camera Pose Requirements

The BIM→2D reprojection requires accurate camera poses in BIM coordinates. Two sub-problems:

1. **Initial Localization**: Current approach uses gizmo+ICP alignment (`T_scan→BIM`). Future: ARKit/ARCore absolute positioning via Unity capture app.
2. **Successive Pose Accuracy**: For 20mm tolerance at 3m distance, need <0.5° rotation and <5cm translation error. MapAnything feed-forward poses satisfy this within chunks. Pose uncertainty is compensated by expanding the BIM reprojection search window.

## Consequences

- 3D reconstruction remains for navigation and visualization — it is NOT being removed
- BIM comparison gains pixel-level precision (from mm-level C2M)
- All future analysis features (material ID, safety, defect detection) benefit from 2D
- Coverage engine works in both 2D (per-frame visibility) and 3D (spatial)
- Unity capture app becomes more important: ARKit/ARCore provides absolute positioning
