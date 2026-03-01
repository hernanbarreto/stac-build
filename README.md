<p align="center">
  <img src="docs/assets/stac_banner.png" alt="STAC Build" width="600"/>
</p>

<p align="center">
  <strong>Spatio-Temporal Awareness Core — Construction Dimensional Control System</strong>
</p>

<p align="center">
  <em>AI-powered As-Built vs As-Planned comparison using dense 3D reconstruction and BIM</em>
</p>

---

## What is STAC?

**STAC Build** is a construction dimensional control system that compares **As-Built reality** (captured via smartphone video) against **As-Planned design** (BIM/IFC models) to detect geometric deviations and track construction progress.

The system captures video from a phone camera, reconstructs a dense 3D point cloud using AI-based depth estimation, segments construction elements, matches them to BIM components, and visualizes deviations as a color-coded "sábana" (deviation map) directly overlaid on the point cloud.

### Core Workflow

```
📱 Capture Video (MP4)
    │
    ▼
🧠 Dense 3D Reconstruction (DA3 — Depth Anything 3)
    ├─ Monocular depth estimation
    ├─ Camera pose tracking (SLAM)
    └─ Aligned point cloud generation
    │
    ▼
🔍 Semantic Segmentation (SAM3 + VLM)
    ├─ SAM3: Instance segmentation of construction elements
    └─ InternVL3: Automatic scene inventory and element labeling
    │
    ▼
📐 BIM Comparison
    ├─ IFC parsing (IfcOpenShell)
    ├─ Scan-to-BIM registration (ICP)
    ├─ Cloud-to-Mesh deviation (C2M)
    └─ Coverage analysis per BIM element
    │
    ▼
📊 Visualization & Reports
    ├─ Sábana: color-coded deviation point cloud
    ├─ Potree: level-of-detail streaming for massive clouds
    ├─ BIM overlay: Three.js + IFC rendering
    └─ Per-element metrics: coverage %, deviation stats
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Depth & SLAM** | [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) | Dense 3D reconstruction from monocular video |
| **Segmentation** | [SAM3](https://github.com/facebookresearch/sam2) (Segment Anything 3) | Instance segmentation of construction elements |
| **Scene Analysis** | [InternVL3](https://github.com/OpenGVLab/InternVL) (Vision-Language Model) | Automatic object identification and labeling |
| **BIM Parsing** | [IfcOpenShell](https://ifcopenshell.org/) | IFC geometry extraction |
| **Point Cloud Viz** | [Potree](https://potree.github.io/) + [Three.js](https://threejs.org/) | Level-of-detail point cloud rendering |
| **Post-processing** | [CloudCompPy](https://www.cloudcompare.org/) | SOR filtering, octree generation |
| **Backend** | Python 3.11, Flask, WebSocket | Pipeline orchestration, API |
| **Frontend** | React + TypeScript + Vite | IDE-style viewer interface |
| **Infrastructure** | Docker, CUDA 12.1 | GPU-accelerated containerized deployment |

---

## Features

### Dense 3D Reconstruction
- **Depth Anything 3** processes video frames to produce dense depth maps and camera poses
- Chunk-based processing with SIM3 overlap alignment for long sequences
- Smart frame selection using ORB-SLAM H/F ratio keyframe detection
- Blur filtering via Laplacian variance analysis
- RANSAC auto-leveling (floor detection and gravity alignment)

### Semantic Segmentation
- **SAM3** segments individual construction elements in 3D
- **InternVL3** VLM automatically scans frames and generates an object inventory
- DBSCAN spatial clustering separates co-labeled instances
- Oriented Bounding Boxes (OBB) for each segmented element
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
│   ├─ da3_native_wrapper.py  # DA3 pipeline orchestration
│   ├─ frame_quality.py       # Blur detection (Laplacian)
│   ├─ frame_selector.py      # Visual novelty filter (H/F ratio)
│   ├─ alignment_manager.py   # SIM3 alignment + auto-leveling
│   ├─ scene_analyzer.py      # InternVL3 scene inventory
│   ├─ segmentation_pipeline.py # SAM3 + DBSCAN + OBB
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
│       │   └─ ...
│       └─ pages/
│           └─ LoginPage.tsx   # Authentication
│
├─ vendor/                    # AI model integrations
│   ├─ depth_anything_3/      # DA3 streaming inference
│   └─ sam3/                  # SAM3 video segmentation
│
├─ static/                    # Legacy viewer + camera capture
│   ├─ viewer.html
│   └─ camera.html
│
└─ docs/                      # Documentation
    ├─ SCANNING_GUIDE.md
    ├─ BIM_INTEGRATION_PLAN.md
    └─ COVERAGE_STRATEGY.md
```

---

## Pipeline Configuration

All pipeline parameters are centralized in `server/config.yaml`:

```yaml
# Key configuration sections
server:
  chunk_size: 30              # Frames per DA3 chunk
  chunk_overlap: 10           # Overlap frames for alignment

frame_selection:
  enabled: true               # Visual novelty filter
  hf_ratio_threshold: 0.45    # H/F ratio for parallax detection

models:
  depth:
    name: "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
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
| **GPU** | RTX 3060 (12GB VRAM) | RTX 4090 (24GB VRAM) |
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

- [ ] **Long-range scanning**: Zoom-aware capture with adaptive intrinsics
- [ ] **Occlusion-aware coverage**: Ray-casting + SAM3 + VLM for intelligent occlusion detection
- [ ] **Multi-source scanning**: Multiple operators scanning concurrently, cloud merging
- [ ] **Multi-level support**: Cross-floor scanning with BIM storey alignment
- [ ] **Unity + ARKit/ARCore**: Mobile capture app with native intrinsics

See [COVERAGE_STRATEGY.md](docs/COVERAGE_STRATEGY.md) for the detailed design of upcoming features.

---

## Origin

STAC Build was presented at **Ingerop Paris** on February 3, 2026, during the IN3 Modernization Program for Mexico City Metro Line 1. It was selected by the Science & Technology department and will be accelerated through the **Impulse Partners** program in partnership with Ingerop France.

---

<p align="center">
  <sub>Designed and developed by <strong>Hernán Barreto</strong> — Ingerop IN3</sub>
</p>
