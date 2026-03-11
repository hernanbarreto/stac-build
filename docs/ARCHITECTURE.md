# STAC-Builder — Architecture Reference

> **Living document** — Must be updated as new modules, integrations, or architectural changes are introduced per the [ROADMAP.md](./ROADMAP.md).
>
> Last updated: 2026-03-08

---

## System Overview

STAC-Builder is an AI-powered construction and asset lifecycle platform that transforms phone-based video into accurate 3D reconstructions, performs segmentation, BIM comparison, and coverage analysis. It comprises a **Python/FastAPI backend**, a **React/Three.js frontend**, and **vendored AI models** (DA3, SAM3, InternVL3).

---

## Architecture Diagram

```mermaid
graph TB
    subgraph UI["React Frontend"]
        App["App.tsx — Main App"]
        VP["Viewport.tsx — 3D Renderer"]
        PL["PotreeLoader.ts — LOD Octree"]
        IS["InteractiveSegmentation.tsx"]
        BN["BIMNavigator.tsx"]
        DO["DeviationOverlay.tsx"]
        BA["BIMAnalysisPanel.tsx"]
        TP["TeamPanel.tsx + WebRTC"]
        AP["AdminPage.tsx"]
    end

    subgraph Server["FastAPI Backend"]
        Main["main.py — API + WebSocket Hub"]
        PM["pipeline_manager.py — 5-Stage Orchestrator"]

        subgraph Workers["Subprocess Workers"]
            W1["da3_worker.py — 3D Reconstruction"]
            W2["cloudcompy_worker.py — Cloud Cleaning"]
            W3["vlm_worker.py — Scene Analysis"]
            W4["sam3_worker.py — Segmentation"]
            W5["instance_cleaner_worker.py — DBSCAN"]
        end

        subgraph CoreModules["Core Processing"]
            DA3W["da3_native_wrapper.py — RealtimeDA3"]
            AM["alignment_manager.py"]
            SP["segmentation_pipeline.py"]
            BC["bim_comparison.py"]
            BR["bim_registration.py"]
            ZD["zoom_detector.py"]
            ICP["icp_aligner.py"]
            CS["coverage_store.py"]
            OR["occlusion_raycaster.py"]
            FQ["frame_quality.py + frame_selector.py"]
        end

        subgraph Support["Supporting"]
            CFG["config.py + config.yaml"]
            PP["project_paths.py"]
            VP2["vendor_paths.py"]
            FS["frame_storage.py"]
            DB["db.py / db_project.py / db_team.py"]
            Auth["auth.py / routes_auth.py / routes_team.py"]
        end
    end

    subgraph Vendor["Vendored AI Models"]
        DA3["depth_anything_3 — DA3_Streaming"]
        SAM3["sam3 — SAM3 Video Predictor"]
        CC["cloudcompy — CloudCompPy"]
        PC["PotreeConverter 2.1"]
    end

    UI --> Server
    PM --> Workers
    W1 --> DA3W --> DA3
    W4 --> SP --> SAM3
    W3 --> SA["scene_analyzer.py — InternVL3"]
    W2 --> CC
```

---

## Server Layer

### Configuration & Infrastructure

| File | Purpose |
|------|---------|
| `config.py` | YAML config loader, `DictConfig` wrapper, `get_param()` helper |
| `config.yaml` | All server/DA3/SAM3/VLM/segmentation/cleaning parameters |
| `vendor_paths.py` | Resolves DA3/SAM3/CloudCompPy paths, adds to `sys.path` |
| `project_paths.py` | `ProjectPaths` + `SourceContext` — hierarchical path resolution: `projects/` → scan days → sources → frames/output |
| `db.py` | SQLite setup + user/auth tables |
| `auth.py` | JWT token generation + verification |

### Main Server — `main.py`

The monolithic FastAPI server. Key sections:

- **PLY utilities**: `cloud_to_binary()`, `load_ply_to_numpy()`, `numpy_to_ply_bytes()`
- **Cloud processing**: `_run_cloudcompy_postprocess()`, `_align_cloud_to_floor()` (RANSAC), `_send_cleaned_cloud()`, `_send_sabana_cloud()`
- **Connection managers**: `CameraManager`, `ViewerManager` (WebSocket connection pools)
- **Online processing**: `chunk_processing_worker()` (real-time DA3 chunk loop), `_resolve_segmentation_prompt()` (VLM auto-prompt)
- **REST endpoints**: Session CRUD, BIM upload/compare, segmentation management, pipeline control, alignment save, mode switching
- **WebSockets**: Viewer (pipeline trigger, cloud loading, progress callbacks, multi-scan), Camera (MASt3R/DA3 flows), SLAM (real-time frame processing), Team (presence, chat, WebRTC signaling), Logs (real-time server log forwarding)

### Pipeline Manager — `pipeline_manager.py`

Orchestrates the 5-stage reconstruction pipeline sequentially as subprocesses with Pipe IPC:

1. **DA3** → 3D Reconstruction
2. **CloudCompPy** → Cloud Cleaning (SOR, voxel, normals)
3. **VLM** → Scene Analysis (InternVL3)
4. **SAM3** → Video Segmentation
5. **Instance Cleaner** → DBSCAN per instance

Key classes: `StageId`, `PipelineStage`, `JobStatus`, `StageState`, `PipelineJob`, `PipelineManager`

### Workers — `server/workers/`

All workers share a pattern: `run(conn, session_dir, config)` → `run_worker_safe(_work_fn, conn, ...)` → `WorkerPipe` IPC.

| Worker | What It Does |
|--------|-------------|
| `base.py` | `WorkerPipe` (IPC: progress/log/done/error/cancel), `run_worker_safe()` |
| `da3_worker.py` | Dispatches to `_run_da3_single` or `_run_da3_multi_segment` based on zoom analysis. Multi-segment: per-zoom blur filter → novelty selection → DA3 reconstruction → ICP inter-segment alignment |
| `cloudcompy_worker.py` | Runs `run_cloudcompy.sh`. Creates `cleaned_cloud.ply`, links to `merged/`, computes `floor_transform.npz` |
| `vlm_worker.py` | InternVL3 scene analysis → `vlm_analysis.json` (prompt + frame_map) |
| `sam3_worker.py` | SAM3 segmentation using VLM prompt → `segmentation.json` + `seg_masks.npz` |
| `instance_cleaner_worker.py` | DBSCAN cluster isolation per segmented instance |

### Core Processing Modules

| Module | Purpose |
|--------|---------|
| `da3_native_wrapper.py` | `RealtimeDA3` wraps `DA3_Streaming`. Per-chunk async processing with callbacks, zoom intrinsics correction, gravity transform management, metadata saving. **Fast path**: skips `super().__init__()` when `preloaded_model` is available |
| `da3_config_builder.py` | Builds DA3 config dict from `config.yaml` (model checkpoint, weights, chunk_size, overlap, conf_threshold, etc.) |
| `alignment_manager.py` | `AlignmentManager`: RANSAC floor detection, SIM3 gravity correction, chunk-to-chunk alignment, point cloud generation with depth/conf thresholds, segmentation ID continuity mapping |
| `icp_aligner.py` | Scaled ICP registration: auto voxel sizing, FPFH features, RANSAC global registration, point-to-plane refinement, scale estimation from correspondences |
| `segmentation_pipeline.py` | Batched SAM3 processing per category, IoU ID matching across batches, NPZ mask storage with upsert logic, display-time point-to-mask matching via PLY origin fields, OBB computation |
| `sam3_wrapper.py` | `SAM3Wrapper`: model load/unload, batch/interactive/text-prompt modes, VRAM OOM recovery, mask caching, interactive session with propagation streaming |
| `scene_analyzer.py` | InternVL3 scene analysis: auto model size selection (2B/8B), dynamic resolution preprocessing, multi-frame analysis with keyframe loading, category merging with synonym detection, frame_map building |
| `bim_comparison.py` | IFC mesh extraction (Revit UniqueId suffix matching), C2M distance computation (KDTree + exact triangle projection), deviation reports, sábana generation |
| `bim_registration.py` | Hierarchical BIM–scan registration: floor plane RANSAC alignment (height+tilt 3DOF) → object XZ+yaw sweep (3DOF) → full 6DOF transform |
| `coverage_store.py` | Per-element cumulative coverage: mesh surface sampling, SampleStatus (COVERED/OCCLUDED/NOT_BUILT/NOT_VISIBLE), persistent NPZ storage, timeline tracking |
| `occlusion_raycaster.py` | Ray-based BIM surface classification: camera pose loading (from DA3 extrinsics), cylindrical ray queries, best-camera selection, per-sample status classification |
| `zoom_detector.py` | 3 zoom detection methods (ffprobe focal length, ORB feature density, DCT frequency), segment identification with merging |
| `frame_quality.py` | Laplacian blur detection with adaptive threshold, inter-frame diff |
| `frame_selector.py` | ORB-SLAM H/F ratio keyframe selection: symmetric transfer error, Sampson distance, chi-squared scoring |
| `frame_storage.py` | `FrameStorage`: session management, frame-to-disk, chunk organization with overlap, PLY save/load with origin traceability (frame_global, pixel_row, pixel_col) |

---

## Frontend Layer

### `App.tsx` — Main Application

- **Session sidebar**: project listing with stats (frame count, cloud size, BIM presence, sábana)
- **Toolbar**: measurement tools, alignment, section box, pipeline controls
- **Pipeline dialog**: stage ordering, per-stage enable/disable, progress tracking
- **Panel system**: BIM Navigator, Deviation Overlay, BIM Analysis, Team, Segmentation
- **Console panel**: real-time server log streaming via WebSocket

### `Viewport.tsx` — 3D Renderer

Three.js-based point cloud renderer:
- Custom GLSL shaders (vertex: depth-based point sizing, fragment: circular points with EDL-like shading + section box clipping)
- Potree LOD octree loading via `PotreeLoader`
- OBB visualization from segmentation results
- Measurement tools (distance, angle)
- Floor alignment gizmo (TransformControls)
- BIM model display (IFC loader integration)
- Sábana (deviation) cloud rendering with transparency
- Section box for clipping

### `PotreeLoader.ts` — LOD Octree Loader

Custom Potree 2.0 octree loader:
- Parses `metadata.json`, `hierarchy.bin`, `octree.bin`
- Priority-queue visibility traversal (projected pixel size threshold)
- Point budget management (loaded/unloaded nodes)
- Chunked hierarchy for deep octrees (proxy nodes → sub-chunk HTTP Range requests)
- Floor transform matrix application

### Other Components

| Component | Purpose |
|-----------|---------|
| `InteractiveSegmentation.tsx` | SAM3 interactive segmentation: click/text prompts, mask painting, SSE propagation streaming, instance management |
| `BIMNavigator.tsx` | IFC tree explorer: Project→Site→Building→Storey→Elements with search, visibility, opacity |
| `DeviationOverlay.tsx` | BIM comparison UI: match selection, tolerance config, results histogram |
| `BIMAnalysisPanel.tsx` | Post-comparison report: per-element quality/advance/coverage, global progress |
| `IFCLoader.ts` | IFC→Three.js mesh loading with tree hierarchy parsing |
| `TeamPanel.tsx` | Team chat, online presence, task assignment |
| `WebRTCCall.tsx` | Video/audio calling between team members |
| `AdminPage.tsx` | User/team CRUD, session-to-team assignment |
| `LoginPage.tsx` | JWT login form |

---

## Vendor Layer

| Vendor | Purpose |
|--------|---------|
| **depth_anything_3** | `DA3_Streaming`: chunk-based monocular depth → 3D reconstruction with SIM3 alignment and loop closure |
| **sam3** | SAM3 Video Predictor: text/point-prompt segmentation with video propagation |
| **cloudcompy** | CloudCompPy: SOR filtering, voxel downsampling, duplicate removal, normal estimation |
| **PotreeConverter** | Converts PLY point clouds to Potree 2.0 octree format |

---

## Pipeline Sequence

```mermaid
sequenceDiagram
    participant UI as React UI
    participant API as FastAPI
    participant PM as PipelineManager
    participant DA3W as DA3 Worker
    participant CCW as CloudCompPy Worker
    participant VLMW as VLM Worker
    participant SAMW as SAM3 Worker
    participant ICW as Instance Cleaner

    UI->>API: Start Pipeline (viewer WebSocket)
    API->>PM: start_pipeline(session_id, stages)

    PM->>DA3W: Stage 1 — 3D Reconstruction
    Note over DA3W: Zoom detection → segment splitting<br/>Per-segment: blur filter → novelty selection<br/>→ DA3 reconstruction → ICP alignment
    DA3W-->>PM: done (chunk PLYs)

    PM->>CCW: Stage 2 — Cloud Cleaning
    Note over CCW: CloudCompPy: SOR + voxel + normals<br/>→ cleaned_cloud.ply + floor_transform.npz
    CCW-->>PM: done

    PM->>VLMW: Stage 3 — Scene Analysis
    Note over VLMW: InternVL3 multi-frame analysis<br/>→ vlm_analysis.json
    VLMW-->>PM: done

    PM->>SAMW: Stage 4 — Segmentation
    Note over SAMW: SAM3 batched per-category<br/>→ segmentation.json + seg_masks.npz
    SAMW-->>PM: done

    PM->>ICW: Stage 5 — Instance Cleaning
    Note over ICW: DBSCAN per instance<br/>→ segmentation_result.json with OBBs
    ICW-->>PM: done

    PM-->>API: Pipeline complete
    API-->>UI: WebSocket: pipeline_done + load results
```

---

## Data Layout — Project Structure

```
projects/<project-slug>/
├── project.json                    # Project metadata
├── ifcs/                          # Uploaded IFC files
├── scans/<date>/<source>/
│   ├── frames/                    # Extracted video frames
│   │   ├── 00000.jpg ... NNNNN.jpg
│   │   ├── frame_quality.json     # Blur analysis
│   │   ├── selected_frames.json   # Novelty-filtered keyframes
│   │   └── zoom_analysis.json     # Per-frame zoom levels
│   └── output/
│       ├── chunk_*.ply            # DA3 raw reconstruction chunks
│       ├── chunk_*_meta.json      # Per-chunk metadata (cameras, intrinsics)
│       ├── chunk_*_origins.npz    # Per-point origin traceability
│       ├── cleaned_cloud.ply      # CloudCompPy cleaned cloud
│       ├── floor_transform.npz    # RANSAC floor alignment (s, R, t)
│       ├── vlm_analysis.json      # InternVL3 scene inventory
│       ├── segmentation.json      # SAM3 raw seg results
│       ├── segmentation_result.json # DBSCAN-cleaned with OBBs
│       ├── seg_masks.npz          # Compressed per-frame masks
│       └── potree/                # PotreeConverter output (LOD octree)
├── merged/
│   ├── merged_cloud.ply           # → symlink to cleaned_cloud.ply
│   └── floor_transform.npz       # → symlink
├── segmentation/                  # Project-level segmentation
├── coverage/                      # Cumulative coverage tracking
│   ├── timeline.json
│   └── element_*.npz
└── bim_comparison/
    ├── sabana.ply                 # Deviation-colored point cloud
    └── sabana_meta.json           # Per-element quality report
```

---

## Key Design Decisions

1. **Model caching**: DA3 model is loaded once and reused across zoom segments via `preloaded_model` in `RealtimeDA3.__init__` (fast path skips `super().__init__()`)
2. **Zoom segmentation**: Videos with zoom changes are split into segments, each processed independently, then aligned with ICP
3. **Origin traceability**: Every point in the PLY carries `(frame_global, pixel_row, pixel_col)` — this is how segmentation masks (2D) map to 3D points at display-time
4. **Gravity alignment**: Computed once from first chunk, shared across all segments. Floor transform persisted to NPZ for consistent loading
5. **Pipeline is subprocess-based**: Each stage runs in its own `multiprocessing.Process` with `Pipe` IPC — the server process never loads GPU models
6. **Potree LOD**: PotreeConverter generates octree; frontend `PotreeLoader` does priority-queue visibility traversal with configurable point budget
7. **All parameters in `config.yaml`**: No hardcoded values — all thresholds, model paths, and tuning parameters are configurable
