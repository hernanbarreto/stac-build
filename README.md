<p align="center">
  <img src="docs/assets/stac_banner.png" alt="STAC Build" width="360"/>
</p>

<p align="center">
  <strong>Spatio-Temporal Awareness Core — Construction Dimensional Control System</strong>
</p>

<p align="center">
  <em>AI-powered As-Built vs As-Planned comparison via dense 3D reconstruction and BIM deviation analysis</em>
</p>

---

## What is STAC?

**STAC Build** is a construction dimensional control system that compares **As-Built reality** (captured via smartphone video) against **As-Planned design** (BIM/IFC models) to detect geometric deviations and track construction progress.

The system reconstructs a dense, metric **point cloud + textured mesh** from a phone
video (optionally with iPad/iPhone LiDAR), segments construction elements, and
compares the result against the BIM/IFC model to measure deviations and coverage.

### Core Workflow (the real 6-stage pipeline)

Orchestrated by `pipeline_manager.py`; each stage runs as an isolated subprocess.

```
📱 Capture: smartphone video (MP4)  +  optional 📷 Stray Scanner (LiDAR + ARKit)
    │
    ▼
🔨 1. 3D Reconstruction   (backend: da3 │ mapanything │ hybrid │ hybrid_cond │ lidar)
    ├─ Frame selection (H/F-ratio keyframes │ stride │ all) + optional blur filter
    ├─ DA3 metric depth + K  →  MapAnything (per-chunk, inside the VGGT-Long framework)
    │     • mapanything   : MapAnything estimates the poses (DA3 poses NOT used)
    │     • hybrid_cond   : ARKit-conditioned DA3 → MapAnything FULL prior (depth+K+poses)
    ├─ SALAD/DINOv2 loop closure + Sim3 global alignment
    └─ RANSAC auto-leveling (floor → gravity)
    │
    ▼
🧹 2. Cloud Cleaning      → CloudCompPy SOR + voxel merge → cleaned_cloud.ply
    ▼
🧊 3. TSDF Mesh           → textured surface mesh (Open3D VoxelBlockGrid, GPU, tiling + photo bake)
    ▼
🔍 4. Scene Analysis      → VLM (InternVL3): object inventory that prompts segmentation
    ▼
🏷️ 5. Segmentation        → SAM3: 2D instance masks matched to the cloud at display time
    ▼
✨ 6. Instance Cleaning   → merge/dedup per-object instances across frames
    │
    ▼
📐 BIM Comparison & Registration
    ├─ Scan-to-BIM alignment (gizmo + ICP)
    ├─ Cloud-to-Mesh deviation (C2M)
    └─ Coverage analysis per BIM element
    │
    ▼
📊 Visualization & Reports
    ├─ Sábana: color-coded deviation map
    ├─ Potree: level-of-detail point cloud streaming
    ├─ BIM overlay: Three.js + IFC rendering
    └─ Per-element metrics: coverage %, deviation stats
```

---

## Technology Stack

### Reconstruction, Analysis & Segmentation

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **DA3** | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Metric monocular depth (+ optional poses). Runs standalone (`da3` backend) **or** as the per-frame **metric depth prior fed into MapAnything**. In `hybrid_cond` it is ARKit-pose-conditioned + LiDAR-calibrated |
| **MapAnything** (default) | [MapAnything](https://github.com/facebookresearch/map-anything) (Meta) inside the [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) framework | Feed-forward metric 3D, run **per chunk** with chunk overlap + **SALAD/DINOv2 loop closure + Sim3** global alignment. Fed DA3 depth+K as prior (poses too in `hybrid_cond`); generates the global point cloud |
| **Loop closure** | [DINOv2](https://github.com/facebookresearch/dinov2) / SALAD | Place-recognition retrieval for loop detection; Sim3 optimization closes drift over long sequences |
| **Stray Scanner** | [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662) (iOS) | iPhone/iPad Pro LiDAR + ARKit capture for `hybrid` / `hybrid_cond` / `lidar` modes |
| **Scene Analysis (VLM)** | [InternVL3](https://github.com/OpenGVLab/InternVL) | Scans frames to build an object inventory that prompts segmentation |
| **Segmentation** | [SAM3](https://github.com/facebookresearch/sam2) (Segment Anything 3) | Open-vocabulary instance segmentation of construction elements |
| **Cloud merge** | [CloudCompPy](https://www.cloudcompare.org/) | SOR outlier removal + voxel downsample, chunk/LiDAR-complement merge → `cleaned_cloud.ply` |
| **TSDF mesh** | [Open3D](https://www.open3d.org/) VoxelBlockGrid (CUDA) | Textured surface mesh from the cleaned cloud (GPU integrate, spatial tiling, photo bake) |
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

### Dense 3D Reconstruction
- **Multi-backend architecture** (`reconstruction.backend` in `config.yaml`, dispatched in `map_worker.py`):
  - **`mapanything`** (default, video-only): DA3 produces per-frame **metric depth + intrinsics** → fed as a **prior** into **MapAnything** (run per-chunk inside the VGGT-Long framework). MapAnything **estimates its own poses** (DA3 poses are intentionally NOT used) and closes loops via SALAD/DINOv2 + Sim3 → global point cloud.
  - **`hybrid_cond`** (Stray / LiDAR, full prior): Stray ARKit + LiDAR → DA3 is **pose-conditioned** (ARKit poses via cam_enc) with **LiDAR-calibrated** metric depth → MapAnything receives the **FULL prior (depth + K + poses injected)** → loop closure.
  - **`da3`**: DA3 streaming standalone (neural depth + SLAM, no LiDAR, no MapAnything).
  - **`hybrid`**: DA3 calibrated with Stray LiDAR depth (estimate-then-inject poses).
  - **`lidar`**: pure LiDAR backprojection (Stray only, no neural inference).
  - **`gaus_slam*`** / **`nerfstudio`**: experimental Gaussian-surfel SLAM and NeuS-SDF variants.
- **DA3-as-prior** is opt-in (`mapanything.use_da3_priors`), forced ON for `hybrid_cond`. Poses are injected only when `da3_prior_use_poses` (auto-true in `hybrid_cond`); otherwise only depth + K.
- **Stray Scanner integration**: auto-detection of iOS data (`odometry.csv`, `depth/`, intrinsics).
- Chunk-based MapAnything inference (`chunk_size` / `overlap`) with Sim3 overlap alignment for long sequences.
- Frame selection (H/F-ratio keyframes, uniform stride, or all-frames) + optional Laplacian blur filter.
- Confidence filtering at the authors' defaults (DA3 `conf_thresh_percentile ≈ 40`, MapAnything/VGGT `conf_threshold_coef 0.75`).
- RANSAC auto-leveling (floor detection and gravity alignment).
- **TSDF meshing** (final stage): the cleaned cloud is meshed into a textured surface — Open3D VoxelBlockGrid on GPU, spatial tiling for large scenes, multi-view photo bake.

### Semantic Segmentation
- **Scene Analysis (VLM)**: InternVL3 scans frames and generates an object inventory that prompts segmentation
- **SAM3**: open-vocabulary instance segmentation — per-frame **2D instance masks**, stored cloud-agnostically and **matched to the point cloud at display time** (no destructive 3D assignment)
- **Instance cleaning** stage: merges/dedups per-object instances across frames
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
│   ├─ main.py                # FastAPI app, WebSocket, API routes
│   ├─ pipeline_manager.py    # 6-stage pipeline orchestrator
│   ├─ workers/               # Subprocess workers (GPU-isolated)
│   │   ├─ base.py            # WorkerPipe IPC protocol
│   │   ├─ map_worker.py      # 1. Reconstruction dispatcher (da3/mapanything/hybrid/hybrid_cond/lidar)
│   │   ├─ cloudcompy_worker.py # 2. Cloud cleaning (SOR + voxel) + LiDAR complement merge
│   │   ├─ tsdf_worker.py     # 3. TSDF textured mesh (Open3D VoxelBlockGrid, GPU)
│   │   ├─ vlm_worker.py      # 4. InternVL3 scene analysis
│   │   ├─ sam3_worker.py     # 5. SAM3 segmentation
│   │   └─ instance_cleaner_worker.py # 6. Instance cleaning (merge/dedup)
│   ├─ stray_da3_streaming.py # Hybrid + LiDAR-only DA3 subclasses
│   ├─ run_da3_hybrid_main.py # Entry point for hybrid/LiDAR subprocess
│   ├─ run_da3_hybrid.sh      # Shell launcher (conda env activation)
│   ├─ ingestors/
│   │   ├─ stray_detector.py  # Auto-detect Stray Scanner data in sessions
│   │   └─ stray_scanner.py   # Load ARKit poses, LiDAR depth, intrinsics
│   ├─ frame_quality.py       # Blur detection (Laplacian)
│   ├─ frame_selector.py      # Visual novelty filter (H/F ratio)
│   ├─ alignment_manager.py   # SIM3 alignment + auto-leveling
│   ├─ scene_analyzer.py      # InternVL3 scene inventory
│   ├─ segmentation/pipeline.py # 2D instance masks → display-time cloud matching
│   ├─ bim_comparison.py      # C2M deviation, sábana, coverage
│   ├─ bim_registration.py    # Scan-to-BIM alignment (ICP)
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
│   ├─ depth-anything-3/      # DA3 — metric depth (prior into MapAnything, or standalone)
│   ├─ VGGT-Long/             # framework: per-chunk MapAnything + SALAD/DINOv2 loop closure + Sim3
│   ├─ MapAnything2/          # MapAnything model (per-chunk feed-forward metric 3D)
│   ├─ dinov3/                # DINO backbone (loop-closure place retrieval)
│   ├─ sam3/                  # SAM3 open-vocabulary segmentation
│   ├─ CloudComPy310/         # CloudCompare Python — cloud SOR + voxel merge
│   ├─ nvdiffrast/            # GPU texture bake for the TSDF mesh
│   ├─ mvs-texturing/         # UV-atlas texture baking (texrecon, CPU fallback)
│   └─ PotreeConverter/       # Octree generation
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

Accurate camera poses drive the whole pipeline — they place every depth observation into the global cloud, refine it via loop closure, and let the TSDF mesh be photo-textured from the source frames. STAC uses a layered approach:

| Source | Type | Accuracy | When |
|--------|------|----------|------|
| **MapAnything + loop closure** | Feed-forward poses, SALAD/Sim3-refined | High relative, ~cm inter-frame | **Default** (`mapanything` / `hybrid_cond`) — these become `camera_poses.txt` used downstream |
| **ARKit (Stray Scanner)** | VIO + IMU + LiDAR | ~cm absolute, metric-scale | `hybrid_cond` — injected into MapAnything as a **pose prior**, then loop-closed |
| **DA3 SLAM** | Neural SLAM (SALAD), loop-closed | High relative, ~cm inter-frame | `da3` standalone backend |
| **Gizmo + ICP** | Manual alignment → refinement | Depends on user + ICP | Scan→BIM registration |

**Pose Sources** (per the actual code):
- **`mapanything` (default):** **MapAnything estimates the poses.** DA3 contributes only the **metric depth + intrinsics** prior — its poses are intentionally NOT used (`da3_prior_use_poses: false`). SALAD/DINOv2 loop closure + Sim3 then refine them globally.
- **`hybrid_cond`:** Stray **ARKit poses condition DA3** (cam_enc) and are **injected into MapAnything as the full prior** (depth + K + poses). MapAnything still loop-closes them, so the priors initialize but do not freeze the solution.
- **`da3` standalone:** DA3's own neural SLAM (SALAD features) handles all pose estimation.

**Global consistency**: over long sequences, per-chunk poses drift; SALAD/DINOv2 place-recognition detects loops and Sim3 optimization closes them, keeping the global cloud metrically consistent. The refined poses are written to `camera_poses.txt` and reused by every downstream stage (TSDF, texture bake, BIM overlay).

---

## Pipeline Configuration

All pipeline parameters are centralized in `server/config.yaml`:

```yaml
# Key configuration sections
reconstruction:
  backend: mapanything         # da3 | mapanything | hybrid | hybrid_cond | lidar | gaus_slam*
  da3:                         # Depth Anything 3 (depth prior, or standalone backend)
    chunk_size: 120
    overlap: 60
    conf_threshold_coef: 0.75  # author default: drop conf < 0.75×mean
  mapanything:                 # MapAnything model, run inside the VGGT-Long framework
    use_da3_priors: true       # feed DA3 metric depth + intrinsics as prior
    da3_prior_use_poses: false # also inject DA3 poses (auto-true in hybrid_cond)
    conf_threshold_coef: 0.75  # VGGT-Long point-cloud confidence filter
    map_conf_percentile: 40    # MapAnything inference confidence floor
  lidar:                       # Stray Scanner (hybrid / hybrid_cond / lidar)
    trust_range: 5.0           # LiDAR depth trust range (meters, iPad ≈ 5m)
    confidence_threshold: 1    # ARKit confidence: 0=low, 1=medium, 2=high
    fallback_to_da3: true      # use DA3 if no Stray data found

tsdf:                          # final stage: textured mesh from cleaned_cloud.ply
  voxel_length: 0.01           # 1 cm voxels
  tsdf_tiling: auto            # spatial tiling so large scenes fit the GPU grid
  texture_mode: vertex_gpu     # multi-view photo bake (nvdiffrast)

frame_selection:
  enabled: true               # Visual novelty filter
  hf_ratio_threshold: 0.45    # H/F ratio for parallax detection

alignment:
  method: "scale+se3"         # SIM3 alignment
  auto_leveling:
    enabled: true             # RANSAC floor detection

bim:
  deviation:
    tolerance_mm: 20.0        # Deviation threshold
    coverage_proximity_m: 0.15
```

### Reconstruction Backend Selection

The API endpoint `GET /api/sessions/{id}/available_backends` auto-detects which backends are viable for each session based on the presence of Stray Scanner data (`depth/`, `odometry.csv`, `camera_matrix.csv`).

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
| **LiDAR input** | Stray Scanner (depth PNGs + odometry CSV + intrinsics CSV) |
| **BIM models** | IFC 2x3, IFC 4, IFC 4.3 |
| **Point clouds** | PLY (native), Potree octree |
| **Meshes** | GLB (TSDF textured surface mesh) |
| **Export** | PLY, GLB, JSON metrics, Potree |

---

## Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for the full development roadmap and [FUTURE_VISION.md](docs/FUTURE_VISION.md) for the strategic platform vision.

---

<p align="center">
  <sub>Designed and developed by <strong>Hernán Barreto</strong></sub>
</p>
