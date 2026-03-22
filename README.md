<p align="center">
  <img src="docs/assets/stac_banner.png" alt="STAC Build" width="600"/>
</p>

<p align="center">
  <strong>Spatio-Temporal Awareness Core — Construction Dimensional Control System</strong>
</p>

<p align="center">
  <em>AI-powered As-Built vs As-Planned comparison using 2D image analysis with BIM reprojection and dense 3D reconstruction</em>
</p>

---

## What is STAC?

**STAC Build** is a construction dimensional control system that compares **As-Built reality** (captured via smartphone video) against **As-Planned design** (BIM/IFC models) to detect geometric deviations and track construction progress.

The system uses a **dual-engine architecture**: a **2D Analysis Engine** that projects BIM elements onto high-resolution camera frames for pixel-level comparison, and a **3D Reconstruction Engine** that generates dense point clouds for navigation, visualization, and spatial context.

### Why 2D-Primary?

The core insight: **analysis is more precise in the original 2D image space** than in reconstructed 3D point clouds.

| Dimension | 2D Image Space | 3D Reconstructed Space |
|-----------|---------------|----------------------|
| **Resolution** | Full sensor resolution (~1MP+ per frame) | Lossy: depth→backprojection→merge→filter |
| **AI Model Precision** | SOTA models (SAM, VLM, PE Spatial) optimized for 2D | No equivalent 3D-native models at this quality |
| **Error Chain** | Direct — pixel vs projected BIM | Cumulative — depth→alignment→SOR→octree |
| **Deviation Detection** | Pixel-level BIM overlay comparison | Point-to-mesh distance (noisy) |
| **Material Identification** | Native — VLM on original pixels | Indirect — must trace back to source frame |

The 2D engine uses **perfect camera poses from MapAnything** to project BIM geometry onto each frame, enabling direct pixel-level comparison without the noise introduced by 3D reconstruction.

### Core Workflow

```
📱 Capture Video (MP4)
    │
    ▼
🧠 Dual-Engine Processing
    │
    ├─ ENGINE 1: 2D Analysis (Primary)
    │   ├─ Frame selection (H/F ratio + blur filter)
    │   ├─ Camera pose estimation (MapAnything cam2world)
    │   ├─ BIM → 2D reprojection (K × [R|t] × P_BIM)
    │   ├─ Per-frame deviation analysis (pixel-level)
    │   ├─ Material identification (PE Spatial + VLM)
    │   └─ Multi-frame coverage accumulation
    │
    ├─ ENGINE 2: 3D Reconstruction (Secondary)
    │   ├─ Dense point cloud (MapAnything)
    │   ├─ SIM3 chunk alignment + auto-leveling
    │   ├─ Instance segmentation (SAM3 + VLM)
    │   └─ Potree octree for streaming visualization
    │
    ▼
📐 BIM Comparison & Registration
    ├─ Scan-to-BIM alignment (gizmo + ICP)
    ├─ Pose transformation: T_scan→BIM
    ├─ Cloud-to-Mesh deviation (C2M)
    └─ Coverage analysis per BIM element
    │
    ▼
📊 Visualization & Reports
    ├─ Sábana: color-coded deviation map
    ├─ Per-frame BIM overlay (2D analysis results)
    ├─ Potree: level-of-detail point cloud streaming
    ├─ BIM overlay: Three.js + IFC rendering
    └─ Per-element metrics: coverage %, deviation stats
```

---

## Technology Stack

### 2D Analysis Engine

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Spatial Backbone** | [PE Spatial](https://github.com/facebookresearch/perception_models) (Meta) | Dense spatial features — segmentation, detection, depth, tracking. ViT-G/14, ~1.8B params. Apache 2.0 |
| **Scene Analysis VLM** | [PLM-8B](https://github.com/facebookresearch/perception_models) (Meta) | Perception Language Model — material identification, scene understanding, occlusion classification. Replaces InternVL3 |
| **Metric Depth** | [DepthLM](https://github.com/facebookresearch/DepthLM_Official) (Meta) | VLM-based per-pixel metric depth estimation (Pixtral 12B). ICLR 2026 Oral (top 1.2%) |
| **Scene Analysis (current)** | [InternVL3](https://github.com/OpenGVLab/InternVL) | Current VLM for object identification and labeling. Being migrated to PLM-8B |

### 3D Reconstruction Engine

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Reconstruction** | [MapAnything](https://github.com/facebookresearch/MapAnything2) (Meta FAIR) | Dense 3D reconstruction + camera poses from monocular video. Apache 2.0 |
| **Segmentation** | [SAM3](https://github.com/facebookresearch/sam2) (Segment Anything 3) | Instance segmentation of construction elements |
| **Post-processing** | [CloudCompPy](https://www.cloudcompare.org/) | SOR filtering, octree generation |
| **Point Cloud Viz** | [Potree](https://potree.github.io/) + [Three.js](https://threejs.org/) | Level-of-detail point cloud rendering |

### Platform

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **BIM Parsing** | [IfcOpenShell](https://ifcopenshell.org/) | IFC geometry extraction |
| **Backend** | Python 3.11, Flask, WebSocket | Pipeline orchestration, API |
| **Frontend** | React + TypeScript + Vite | IDE-style viewer interface |
| **Infrastructure** | Docker, CUDA 12.1 | GPU-accelerated containerized deployment |

---

## Features

### 2D Analysis Engine
- **BIM Reprojection**: Project BIM elements onto each camera frame using MapAnything poses + intrinsics
- **Pixel-Level Deviation Detection**: Compare projected BIM edges and surfaces against actual image content
- **Material Identification**: PE Spatial + VLM analyze original image pixels for material classification
- **Multi-Frame Coverage**: Accumulate analysis results across frames for complete element coverage
- **Pose-Aware Tolerance**: Expand matching regions based on estimated pose uncertainty

### Dense 3D Reconstruction
- **MapAnything** processes video frames to produce dense depth maps and camera poses
- Chunk-based processing with SIM3 overlap alignment for long sequences
- Smart frame selection using ORB-SLAM H/F ratio keyframe detection
- Blur filtering via Laplacian variance analysis
- RANSAC auto-leveling (floor detection and gravity alignment)

### Semantic Segmentation
- **SAM3** segments individual construction elements in 3D
- **InternVL3** VLM automatically scans frames and generates an object inventory (being replaced by **PLM-8B**)
- DBSCAN spatial clustering separates co-labeled instances
- RANSAC face detection per segment with non-destructive point assignment
- Voxel mesh visualization with normal-snapped quads
- Interactive retroactive prompting: point-and-click to re-segment

### BIM Integration
- Full IFC parsing: geometry extraction for all physical elements
- Scan-to-BIM registration via gizmo alignment + ICP refinement
- Cloud-to-Mesh (C2M) deviation calculation per element
- Coverage analysis: percentage of BIM surface observed by scan
- Quality classification: Good / Regular / Bad per element

### Sábana Visualization
- Color-coded deviation map: Green (within tolerance) → Yellow → Red (out of tolerance)
- Rendered as a point cloud overlaid on BIM for direct visual inspection
- Semi-transparent BIM and scan cloud to highlight deviations
- Per-element statistics: mean, max, P95 deviation, pass rate

### Potree Streaming
- Custom Potree integration for level-of-detail rendering of massive point clouds
- Hierarchical octree with progressive loading
- Point budget management for smooth navigation

### Team & Session Management
- JWT authentication with role-based access (admin/viewer)
- Multi-user team workspaces
- Session persistence with full pipeline state
- Real-time WebSocket progress streaming
- Activity logging per user

---

## Architecture

```
stac-builder/
├─ server/                    # Python backend
│   ├─ main.py                # Flask app, WebSocket, API routes
│   ├─ reconstruction_native_wrapper.py  # Reconstruction pipeline orchestration
│   ├─ frame_quality.py       # Blur detection (Laplacian)
│   ├─ frame_selector.py      # Visual novelty filter (H/F ratio)
│   ├─ alignment_manager.py   # SIM3 alignment + auto-leveling
│   ├─ scene_analyzer.py      # InternVL3 scene inventory
│   ├─ segmentation/          # SAM3 + DBSCAN + RANSAC face detection
│   │   └─ pipeline.py        # Full segmentation pipeline
│   ├─ bim_comparison.py      # C2M deviation, sábana, coverage
│   ├─ bim_registration.py    # Scan-to-BIM alignment (ICP)
│   ├─ cloudcompy_postprocess.py # SOR filter, Potree conversion
│   └─ config.yaml            # All pipeline configuration
│
├─ ui/                        # React + TypeScript frontend
│   └─ src/
│       ├─ App.tsx             # Main application shell
│       ├─ components/
│       │   ├─ IFCLoader.ts    # BIM model rendering
│       │   ├─ PotreeLoader.ts # Point cloud streaming
│       │   └─ InteractiveSegmentation.tsx
│       └─ pages/
│           └─ LoginPage.tsx   # Authentication
│
├─ vendor/                    # AI model integrations
│   ├─ MapAnything2/           # 3D reconstruction + camera poses
│   ├─ perception_models/     # PE Spatial backbone + PLM-8B VLM (Apache 2.0)
│   ├─ DepthLM_Official/      # VLM metric depth estimation (Pixtral 12B)
│   ├─ sam3/                   # SAM3 video segmentation
│   └─ PotreeConverter/        # Octree generation
│
├─ docs/                      # Documentation
│   ├─ ROADMAP.md
│   ├─ FUTURE_VISION.md
│   ├─ ARCHITECTURE.md
│   └─ SCANNING_GUIDE.md
│
└─ static/                    # Legacy viewer + camera capture
```

---

## Camera Pose & Localization

The 2D analysis engine depends on accurate camera poses for BIM reprojection. STAC uses a layered approach:

| Source | Type | Accuracy | When |
|--------|------|----------|------|
| **MapAnything** | SfM feed-forward | High relative, ~cm inter-frame | Always (core pipeline) |
| **Gizmo + ICP** | Manual alignment → refinement | Depends on user + ICP | Current registration method |
| **ARKit/ARCore** | VIO + IMU + LiDAR | ~cm short-range, drift over distance | Future (Unity capture app) |

**Initial Localization**: The scan-to-BIM registration (gizmo + ICP) provides the transformation `T_scan→BIM` that maps MapAnything poses to BIM coordinates. This is the same registration used today for 3D comparison, now repurposed for 2D reprojection.

**Error Handling**: Pose uncertainty at distance `d` is compensated by expanding the BIM reprojection search window proportionally. PE Spatial features enable robust local matching even with ±20px uncertainty.

---

## Pipeline Configuration

All pipeline parameters are centralized in `server/config.yaml`:

```yaml
# Key configuration sections
server:
  chunk_size: 30              # Frames per reconstruction chunk
  chunk_overlap: 10           # Overlap frames for alignment

frame_selection:
  enabled: true               # Visual novelty filter
  hf_ratio_threshold: 0.45    # H/F ratio for parallax detection

models:
  depth:
    name: "facebook/map-anything-apache"
    device: "cuda"
  segmentation:
    sam3_checkpoint: "sam3_hiera_large.pt"

alignment:
  method: "scale+se3"         # SIM3 alignment
  auto_leveling:
    enabled: true             # RANSAC floor detection

bim:
  deviation:
    tolerance_mm: 20.0        # Deviation threshold
    coverage_proximity_m: 0.15
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3060 6GB (CPU fallback for large models) | RTX 4090 (24GB VRAM) |
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 32 GB | 64 GB |
| **Storage** | SSD 500GB | NVMe 1TB+ |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hernanbarreto/stac-build.git
cd stac-build

# Build and run with Docker
docker compose up --build

# Or run locally
pip install -r requirements.txt
cd ui && npm install && npm run build && cd ..
python server/main.py
```

Access the application at `http://localhost:5000`

---

## Supported Formats

| Type | Formats |
|------|---------|
| **Video input** | MP4, AVI, MOV |
| **BIM models** | IFC 2x3, IFC 4, IFC 4.3 |
| **Point clouds** | PLY (native), Potree octree |
| **Export** | PLY, JSON metrics, Potree |

---

## Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for the full development roadmap and [FUTURE_VISION.md](docs/FUTURE_VISION.md) for the strategic platform vision.

---

<p align="center">
  <sub>Designed and developed by <strong>Hernán Barreto</strong></sub>
</p>
