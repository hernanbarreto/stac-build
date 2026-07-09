<p align="center">
  <img src="docs/assets/stac_banner.png" alt="STAC Build" width="360"/>
</p>

<p align="center">
  <strong>Spatio-Temporal Awareness Core — Construction Dimensional Control System</strong>
</p>

<p align="center">
  <em>AI-powered As-Built vs As-Planned comparison via dense 3D reconstruction, semantic anchoring and BIM deviation analysis</em>
</p>

---

## What is STAC?

**STAC Build** is a construction dimensional control system that compares **As-Built reality** (captured via smartphone video) against **As-Planned design** (BIM/IFC models) to detect geometric deviations and track construction progress.

The system reconstructs a dense, metric **point cloud + textured mesh** from a phone
video (optionally with iPad/iPhone LiDAR), understands and segments the scene with a
local VLM + SAM3, **anchors the reconstruction semantically** (Phase R: instance-aware
pose/depth refinement), and compares the result against the BIM/IFC model to measure
deviations and coverage. A persistent **Qwen3-VL semantic layer** then makes the scene
queryable (spatial Q&A with deterministic measurement tools) and reportable.

### Core Workflow (the SIMPLE one-pass pipeline)

Orchestrated by `pipeline_manager.py` (`DEFAULT_STAGE_ORDER`); each stage runs as an
isolated subprocess. The pipeline is **automatic and resume-aware** (per-stage artifact
probes: finished stages are skipped) and **fail-fast** (a stage that cannot do its job
raises — no silent fallbacks). One concept: **sparse frames → ONE reconstruction pass →
no window seams → no "onion"** (validated 2026-07-09 against the VGGT-Omega web demo).
Heavy stages get the GPU to themselves (the vLLM semantic service is stopped during
reconstruction and SAM3, and auto-restarts on the next VLM use).

```
📱 Capture: smartphone video (MP4)  +  optional 📷 Stray Scanner (LiDAR + ARKit)
    │
    ▼
🔨 1. 3D Reconstruction   (backend: vggtomega ← default │ mapanything │ da3 │ hybrid │ hybrid_cond │ lidar)
    ├─ Laplacian blur filter → temporal sampling at ~4 fps (reconstruction.simple.target_fps)
    ├─ VGGT-Ω (CVPR 2026) in ONE single pass when the set fits (≤600 frames AND free
    │     VRAM, ~86 MB/frame): no chunking, no Sim3 gluing, no loop closure needed.
    │     Longer videos fall back to windowed mode (500/250, 50% overlap) with a
    │     real free-VRAM check that shrinks the chunk instead of OOM-ing.
    ├─ aggressive point-confidence filter (conf >= mean×0.6 ≈ the web demo at 20%)
    ├─ metric scale: 12 evenly-spread DA3 anchor frames, ISOLATED per-frame inference
    │     (no streaming) → s = median(DA3/Ω depth) applied as a global similarity
    │     (fails hard if the scale cannot be recovered)
    └─ upright orientation BAKED into cloud + poses (reconstruction/orient.py):
          gravity = mean camera-down axis, floor at y=0 — deterministic, no floor RANSAC
    │
    ▼
🔍 2. Scene Understanding  → Qwen3-VL (semantic service): understands the scene from
    │                        ~8 keyframes and emits ONE RICH noun phrase per object
    │                        type ("concrete support column"), consolidated across
    │                        frames. No per-keyframe detection, no boxes.
    ▼
🏷️ 3. Segmentation         → SAM3: each phrase runs as its OWN concept session over ALL
    │                        sampled frames — SAM3 finds, labels and tracks; its
    │                        tracking IS the instance identity. Post-pass: same-SPACE
    │                        dedupe (5 cm voxel occupancy) merges duplicates of one
    │                        physical object, tiny slivers are dropped, and the
    │                        canonical instance store (scene_r.db) is rebuilt from the
    │                        clean instances (display frame) for phases 2-6.
    ▼
🧹 4. Cloud Cleaning       → CloudCompPy SOR + voxel merge → cleaned_cloud.ply
    │                        + mask→cloud instance mapping
    ▼
🧊 5. TSDF Mesh            → textured surface mesh (Open3D VoxelBlockGrid, GPU, 3D cube
    │                        tiling welded into one mesh; depth_source: mapanything —
    │                        the SAME Ω keyframe depth as the cloud; texrecon UV-atlas
    │                        photo texture)
    │
    ▼
📐 BIM Comparison & Registration
    ├─ Scan-to-BIM alignment (gizmo + ICP)
    ├─ Cloud-to-Mesh deviation (C2M)
    └─ Coverage analysis per BIM element
    │
    ▼
📊 Visualization, Q&A & Reports
    ├─ Sábana: color-coded deviation map
    ├─ Potree: level-of-detail point cloud streaming
    ├─ BIM overlay: Three.js + IFC rendering
    ├─ AI Assistant: spatial Q&A with animated measurement replay
    └─ Bilingual (ES/FR) supervision report drafts with per-number tool traces
```

Notes on the stage machinery (`pipeline_manager.py`):
- Registered stages: RECONSTRUCTION, VLM, SAM3, PHASE_R, CLOUDCOMPY, TSDF, plus an
  INSTANCE_CLEANER stage (per-instance DBSCAN + smoothing) that is registered but not
  part of the default order.
- `pipeline.auto_segment: false` disables the semantic chain (VLM/SAM3/Phase R) for a
  pure-geometry run; GauS-SLAM experimental backends skip the cloud-cleaning stage.

### Pose accuracy — what actually runs today

The current worksite default is the **VGGT-Ω backbone**: SOTA feed-forward camera poses
(CVPR 2026, +77% on Sintel, robust to dynamic scenes — moving people/machinery), made
metric by aligning to DA3 depth. On top of it:

1. **Loop closure** — SALAD/DINOv2 place recognition + Sim3 optimization inside
   VGGT-Long closes drift over long sequences.
2. **`scale_align`** (`reconstruction/scale_align.py`) — one global similarity from the
   median DA3/Ω depth ratio makes poses + cloud metric. Fail-fast: no scale → abort.
3. **`fine_register`** — plane-constrained inter-chunk registration with per-chunk
   pieces and per-frame interpolation (absorbs intra-chunk drift on long hostile
   captures).
4. **Phase R** (`server/phase_r/`) — instance-aware inter-window Sim(3) pose-graph
   refinement + depth regularization, with an A/B fail-safe and writeback into cloud
   and TSDF (see the Semantic Intelligence Layer below).

**About the two-pass bundle adjustment.** The repo ships a full dense two-pass
COLMAP/Ceres BA over learned VGGSfM tracks
(`reconstruction/{vggt_tracks,colmap_ba,run_colmap_ba,densify_fillers,reproject_chunks}.py`:
pass 1 refines keyframes with pose priors, pass 2 localises inter-keyframe fillers
against the fixed map, then densifies them back into the cloud). It is currently
**disabled by default** (`bundle_adjust.enabled: false`): on the metric VGGT-Ω
reconstruction the A/B against the no-BA baseline came out *worse*, so Ω poses +
`scale_align` are kept as-is. Do not re-enable for `vggtomega` without re-validating.
The legacy inter-keyframe ICP (`dense_fusion`) is likewise off.

**Auto-leveling** (RANSAC floor detection → gravity alignment) lives in
`alignment_manager.py` and runs as part of alignment, not inside the reconstruction
worker.

---

## Technology Stack

### Reconstruction, Understanding & Segmentation

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **VGGT-Ω** (default backbone) | [VGGT-Ω](https://vggt-omega.github.io/) (CVPR 2026) | SOTA feed-forward camera/pose backbone, dynamic-scene robust. Runs per chunk inside VGGT-Long; up-to-scale → made metric via `scale_align` against DA3 depth. Gated weights (`vendor/vggt-omega-weights/`) |
| **DA3** | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Metric monocular depth (+ optional poses). The **metric anchor** for `vggtomega`, the **depth+K prior** for `mapanything`, and a standalone SLAM backend (`da3`). In `hybrid_cond` it is ARKit-pose-conditioned + LiDAR-calibrated |
| **MapAnything** (option) | [MapAnything](https://github.com/facebookresearch/map-anything) (Meta) inside [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) | Feed-forward metric 3D backbone alternative; fed DA3 depth+K as prior (poses too in `hybrid_cond`). Its keyframe depth is also the TSDF `depth_source` |
| **Loop closure** | [DINOv2](https://github.com/facebookresearch/dinov2) / SALAD | Place-recognition retrieval + Sim3 optimization closes drift over long sequences |
| **Keyframe selection** | DINOv2-cosine (default) / parallax | `frames_selector: dino` — appearance-redundancy cut (0.99 cosine), the literature-recommended family for transformer multi-view stages. `parallax` (triangulation-angle, aborts on pure rotation), `stride`, `none` also available |
| **Bundle adjustment** (off by default) | [VGGSfM](https://github.com/facebookresearch/vggsfm) tracks + [pycolmap](https://github.com/colmap/colmap)/Ceres | Dense two-pass pose-prior BA + filler densification — shipped but disabled for `vggtomega` (degraded the metric result in A/B) |
| **Stray Scanner** | [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662) (iOS) | iPhone/iPad Pro LiDAR + ARKit capture for `hybrid` / `hybrid_cond` / `lidar` modes |
| **Semantic service** | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) on [vLLM](https://github.com/vllm-project/vllm) 0.19 | Persistent local VLM (`127.0.0.1:8799`, OpenAI-compatible) shared by every consumer: scene understanding, auto-prompting, classification, findings, QC, Q&A orchestration, reports. Optional 32B-FP8 candidate |
| **Scene understanding / auto-prompter** | Qwen3-VL (semantic service) | Open-vocabulary grounded detection over keyframes → SAM3 box prompts. (InternVL3 remains only as a disabled legacy fallback, `scene_analysis.enabled: false`) |
| **Segmentation** | [SAM3](https://github.com/facebookresearch/sam2) — SAM 3.0 default, SAM 3.1 optional (`models.segmentation.version`) | Tracked per-frame **2D instance masks**, stored cloud-agnostically; mask→cloud mapping is deferred to the cleaning stage |
| **Phase R geometry** | [R3D](https://github.com/facebookresearch/r3d) (vendored, adapted) | Depth-lift, plurality vote, KNN filtering, gravity-aligned OBBs — reused for semantic anchoring and the spatial-Q&A scene build |
| **Cloud merge** | [CloudComPy](https://www.cloudcompare.org/) | SOR outlier removal + voxel downsample, chunk/LiDAR-complement merge → `cleaned_cloud.ply` |
| **TSDF mesh** | [Open3D](https://www.open3d.org/) VoxelBlockGrid (CUDA) | Textured surface mesh (GPU integrate, 3D cube tiling welded into one mesh, long-edge cull 0.10 m, hole-fill) |
| **Texture** | [MVS-Texturing](https://github.com/nmoehrle/mvs-texturing) (texrecon, default) / [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | UV-atlas photographic texture bake (default `texture_mode: texrecon`); GPU vertex bake available |
| **Per-object meshes** | MeshFlow (vendored; replaced ShapeR) | Generative per-object mesh reconstruction |
| **Point Cloud Viz** | [Potree](https://potree.github.io/) + [Three.js](https://threejs.org/) | Level-of-detail point cloud rendering |

### Platform

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **BIM Parsing** | [IfcOpenShell](https://ifcopenshell.org/) | IFC geometry extraction |
| **Backend** | Python, **FastAPI + uvicorn** (HTTPS, self-signed certs), WebSockets | Pipeline orchestration + API on port **8765** (`scripts/start.sh`, conda env `da3`) |
| **Frontend** | React 18 + TypeScript + Vite (+ **Electron** desktop) | IDE-style viewer; industrial design system (tokens, mono numerics). On headless pods the Electron app is viewable over **noVNC** (`:6080/vnc.html`, `scripts/launch_electron.sh`) |
| **Auth** | JWT (HS256, python-jose) + bcrypt | Role-based access (admin/viewer), team workspaces, activity logging |
| **Infrastructure** | Conda envs per component, CUDA (sm_86), Docker available | GPU-accelerated deployment; dev/reference GPU: RTX A6000 48 GB |

---

## Features

### Dense 3D Reconstruction
- **Multi-backend architecture** (`reconstruction.backend` in `config.yaml`, dispatched in `map_worker.py`):
  - **`vggtomega`** (default, video-only): **VGGT-Ω** per-chunk poses + cloud inside VGGT-Long (loop closure on), made **metric** by `scale_align` against DA3 depth. No ICP dense-fusion, no BA — its poses are the reference. Robust to dynamic worksite scenes.
  - **`mapanything`** (video-only): DA3 metric depth + intrinsics fed as a **prior** into **MapAnything** (per-chunk, VGGT-Long framework). MapAnything estimates its own poses (`da3_prior_use_poses: false`) and closes loops via SALAD/DINOv2 + Sim3.
  - **`hybrid_cond`** (Stray / LiDAR, full prior): Stray ARKit + LiDAR → DA3 is **pose-conditioned** (ARKit poses via cam_enc) with **LiDAR-calibrated** metric depth → MapAnything receives the FULL prior (depth + K + poses) → loop closure.
  - **`da3`**: DA3 streaming standalone (neural depth + SLAM, no LiDAR, no MapAnything).
  - **`hybrid`**: DA3 calibrated with Stray LiDAR depth (estimate-then-inject poses).
  - **`lidar`**: pure LiDAR backprojection (Stray only, no neural inference).
  - **`gaus_slam*`** / **`nerfstudio`**: experimental Gaussian-surfel SLAM and NeuS-SDF variants.
- **Chunked inference** (`chunk_size: 120` / `chunk_overlap: 60`) with Sim3 overlap alignment for long sequences; `fine_register` (plane-constrained, per-chunk pieces) absorbs intra-chunk drift.
- **Keyframe selection**: `dino` (DINO-cosine 0.99) by default; `parallax` geometric selection available — DA3 depth+pose on all blur-valid frames, keyframes by median triangulation angle, **aborts on pure rotation / no baseline**. Optional Laplacian blur filter (`blur_filter: true`).
- Confidence filtering at the authors' defaults (DA3 prior percentile 40, VGGT-Long `conf_threshold_coef: 0.75`).
- RANSAC auto-leveling (floor detection → gravity alignment, `alignment_manager.py`).
- **Fail-fast pipeline**: every stage aborts on failure (no silent fallbacks) — a finished run means every stage actually worked. Resume-aware: finished stages are skipped via artifact/freshness probes.
- **TSDF meshing** (final stage): Open3D VoxelBlockGrid on GPU, 1.2 cm voxels, 3D cube tiling (10 m) welded into one mesh, long-edge cull (0.10 m) + hole-fill, Taubin smoothing, `depth_source: mapanything` (the same Ω keyframe depth as the cloud → cloud/mesh consistency), **texrecon UV-atlas photo texture**.

### Scene Understanding & Segmentation
- **Auto-prompter (Phase 1)**: Qwen3-VL open-vocabulary grounded detection over keyframes → SAM3 box prompts, with geometric (pose-based IoU) temporal association. The construction vocabulary (`autoprompt/vocabulary.yaml`) is a **canonicalization overlay, never a detection filter** — the system segments *everything* it sees. No canned-category fallback: scene understanding drives segmentation or the stage fails loudly. Adaptive keyframe sampling keeps VLM calls bounded.
- **SAM3**: tracked per-frame 2D instance masks, stored cloud-agnostically; the mask→cloud mapping is **deferred** to the cleaning stage so it lands on the Phase-R-corrected cloud.
- **Instance cleaning** (optional stage): per-instance DBSCAN + CloudCompPy smoothing.
- **Segmentation Manager** in the UI for human review/correction and interactive retroactive prompting; a batch CLI (`segmentation/autoprompt/cli.py`) runs video-in → tracked-masks-out without the UI.

### BIM Integration
- Full IFC parsing: geometry extraction for all physical elements
- Scan-to-BIM registration via gizmo alignment + ICP refinement
- Cloud-to-Mesh (C2M) deviation calculation per element (`tolerance_mm: 50`, warning/error/critical bands at 10/20/30 mm)
- Coverage analysis: percentage of BIM surface observed by scan (`coverage_proximity_m: 0.15`)
- Quality classification: Good / Regular / Bad per element
- `GET /api/sessions/{id}/available_backends` auto-detects which backends are viable per session (presence of Stray data: `depth/`, `odometry.csv`, `camera_matrix.csv`)

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
- Real-time WebSocket progress streaming (`/ws/logs`, `/ws/viewer`, `/ws/camera`, `/ws/team`)
- Activity logging per user

---

## Semantic Intelligence Layer (Phases 0–7)

A persistent **Qwen3-VL** vision-language layer sits on top of the reconstruction
pipeline, turning the point cloud into a **queryable, supervised, reportable**
scene. It is governed by one inviolable rule:

> **The VLM proposes, describes, detects, classifies and orchestrates.
> It NEVER measures.** Every metric comes from deterministic tools over geometry
> / `surface_fitting`. Every output is tagged `vlm_proposed`, `tool_measured`, or
> `human_validated`.

| Phase | Path | Purpose | Status |
|-------|------|---------|--------|
| **0 · Semantic Service** | `server/semantic/` | Persistent vLLM 0.19 serving **Qwen3-VL-8B-Instruct** on `127.0.0.1:8799` (OpenAI-compatible, tool-calling, images), own `semantic` env; all consumers go through `semantic_client` (`LLMBackend`: `qwen_local` default, `qwen_local_large` = Qwen3-VL-32B-FP8 candidate). JSONL call logging, healthcheck, smoke tests | ✅ |
| **1 · Auto-prompter** | `server/segmentation/autoprompt/` | Understanding-driven, **open-vocabulary** detection → SAM3 box prompts. Segments *everything*; the construction vocabulary is a canonicalization overlay, never a filter. Batch CLI + pre-populated sessions for human review | ✅ |
| **R · Semantic anchoring** | `server/phase_r/` (+ `workers/phase_r_worker.py`, **in the default pipeline**) | Canonical R.8 instance store (SQLite), gravity-aligned OBBs, vote-entropy + onion (double-surface) metrics, inter-window Sim(3) pose graph, class-conditioned depth regularization, A/B fail-safe, and **writeback** of refined poses/depth into the cloud and TSDF fusion | ✅ |
| **5 · Spatial Q&A** | `server/phase5_qa/` | Deterministic `SpatialTools` (distance, clearance, plumb, level, volume, fits_through, flatness bridge, height profile, measure_between, findings, alignment health…) + user-defined **evaluation volumes**, driven by a tool-calling orchestrator. CLI + `POST /api/spatial_qa` + `/api/scene/*` | ✅ |
| **2 · Classification** | `server/phase2_classify/` | Per-instance class / material / state, label-conflict flags, structural→`surface_fitting` routing; enriches the R.8 store (never creates parallel objects) | ✅ |
| **3 · Findings** | `server/phase3_findings/` | Cracks / moisture / spalling / corrosion detection, **3D-anchored** via the R-refined pose, multi-view deduped, residual-correlated; honest precision eval; everything born `proposed` until human-validated | ✅ |
| **4 · Capture QC** | `server/phase4_qc/` | Ingestion pre-filter (cheap Laplacian blur + exposure; VLM only on the ambiguous band) + post-reconstruction coverage → recapture checklist. Never deletes frames silently | ✅ |
| **6 · Report** | `server/phase6_report/` | Bilingual **ES/FR** supervision-report draft; every number carries a `tool(args)+timestamp` trace; VLM text flagged *pending validation* | ✅ |
| **7 · Validation** | `server/phase7_validation/` | 28-question spatial-Q&A suite (incl. insufficient-data traps), GPU coexistence probe (vLLM + full reconstruction on one A6000 48 GB), reproducible Pitch-2 demo | ✅ |

Each phase ships a CLI, unit tests, and phase reports under `docs/`
(`phase{0,1,3,4,6,7}_report.md`). The only open items are **external data / human**
dependencies (hand-segmented GT, a multi-window corridor scan, larger annotated sets).

**Run the semantic service** (own env, isolated from the reconstruction envs):

```bash
bash scripts/build_semantic_env.sh     # one-time, on a fresh box
bash setup_weights.sh semantic         # download Qwen3-VL-8B-Instruct
bash scripts/serve_semantic.sh         # → http://127.0.0.1:8799/v1
# optional 32B candidate: bash setup_weights.sh semantic-large
```

### Immersive AI Assistant (in the viewer)

The 3D viewer includes an **AI Assistant** panel (`AssistantPanel.tsx`):
ask a question in natural language and the answer is measured by the Phase-5
tools and **replayed as animated three.js geometry** — you literally see *how*
each distance, volume, plumb angle or clearance was measured. Drop **evaluation
volumes** into the scene to assess spaces (occupancy, free m³, "does this fit?").
The chat is backed by `POST /api/spatial_qa`; geometry by `POST /api/scene/*`.

### Reproducible end-to-end demo

```bash
scripts/demo_pitch2.sh <session_dir> scene.db out_dir
# segment → Phase R store → classify → findings → coverage
# → 8 spatial questions answered WITH traces → bilingual report draft
```

---

## Architecture

```
stac-build/
├─ server/                    # Python backend (FastAPI + uvicorn, port 8765, HTTPS)
│   ├─ main.py                # FastAPI app, WebSockets, API routes (+ auth/team/spatial-QA routers)
│   ├─ pipeline_manager.py    # Stage orchestrator — DEFAULT_STAGE_ORDER:
│   │                          #   RECONSTRUCTION → VLM → SAM3 → PHASE_R → CLOUDCOMPY → TSDF
│   ├─ workers/               # Subprocess workers (GPU-isolated, WorkerPipe IPC)
│   │   ├─ base.py            # WorkerPipe IPC protocol
│   │   ├─ map_worker.py      # 1. Reconstruction dispatcher (vggtomega/mapanything/da3/hybrid/...)
│   │   │                      #    + frame selection + scale_align + fine_register (+ optional BA)
│   │   ├─ vlm_worker.py      # 2. Scene understanding (Qwen3-VL auto-prompter; InternVL3 = disabled fallback)
│   │   ├─ sam3_worker.py     # 3. SAM3 tracked 2D instance masks
│   │   ├─ phase_r_worker.py  # 4. Phase R semantic anchoring (pose/depth refinement + writeback)
│   │   ├─ cloudcompy_worker.py # 5. Cloud cleaning (SOR + voxel) + deferred mask→cloud mapping
│   │   ├─ tsdf_worker.py     # 6. TSDF textured mesh (Open3D VoxelBlockGrid, GPU)
│   │   └─ instance_cleaner_worker.py # optional: per-instance DBSCAN + smoothing
│   ├─ semantic/              # Phase 0: vLLM launcher, LLMBackend client, healthcheck, call log
│   ├─ segmentation/          # SAM3 wrapper, 2D-mask store, autoprompt/ (Phase 1)
│   ├─ phase_r/               # Phase R: instance store, vote, onion, residuals, regularization,
│   │                          #   metric hierarchy, fail-safe, writeback
│   ├─ phase2_classify/ … phase7_validation/   # Phases 2–7 (CLI + tests each)
│   ├─ frames/selector.py     # Keyframe selection: DINOv2-cosine (default) + parallax (geometric)
│   ├─ frame_quality.py       # Blur detection (Laplacian)
│   ├─ reconstruction/        # scale_align, fine-register/geometry, texture bake,
│   │   │                      #   surface_fit/, per-object machinery
│   │   ├─ scale_align.py     #   metric scale for the VGGT-Ω path (align to DA3)
│   │   ├─ vggt_tracks.py     #   VGGSfM correspondences   ┐
│   │   ├─ colmap_ba.py       #   two-pass COLMAP/Ceres BA │ shipped, disabled by default
│   │   ├─ run_colmap_ba.py   #   BA runner                │ (bundle_adjust.enabled: false)
│   │   ├─ densify_fillers.py #   filler densification     │
│   │   └─ reproject_chunks.py#   chunk re-projection      ┘
│   ├─ ingestors/             # Stray Scanner auto-detection + loaders (ARKit poses, LiDAR depth)
│   ├─ alignment_manager.py   # SIM3 alignment + RANSAC auto-leveling
│   ├─ bim_comparison.py      # C2M deviation, sábana, coverage
│   ├─ bim_registration.py    # Scan-to-BIM alignment (ICP)
│   ├─ auth/                  # JWT auth, roles, team workspaces, activity log
│   └─ config.yaml            # All pipeline configuration (single source of truth)
│
├─ ui/                        # React 18 + TypeScript + Vite (+ Electron desktop)
│   └─ src/
│       ├─ App.tsx             # Main application shell
│       ├─ components/
│       │   ├─ AssistantPanel.tsx  # AI Assistant (spatial Q&A + measurement replay)
│       │   ├─ IFCLoader.ts    # BIM model rendering
│       │   ├─ PotreeLoader.ts # Point cloud streaming
│       │   └─ InteractiveSegmentation.tsx
│       └─ pages/LoginPage.tsx
│
├─ vendor/                    # AI model integrations (git-ignored; see VENDORS.lock.md)
│   ├─ vggt-omega/            # VGGT-Ω backbone (default backend; gated weights →
│   │   │                      #   vendor/vggt-omega-weights/)
│   ├─ VGGT-Long/             # framework: per-chunk backbone + SALAD/DINOv2 loop closure + Sim3
│   │                          #   + vendored VGGSfM tracker + VGGTOmega adapter (STAC fork, submodule)
│   ├─ depth-anything-3/      # DA3 — metric depth anchor/prior (STAC fork, private submodule)
│   ├─ MapAnything2/          # MapAnything model (alternative backbone; TSDF depth source)
│   ├─ r3d/                   # R3D machinery (Phase R + spatial-Q&A scene build)
│   ├─ sam3/ + sam31/         # SAM3 (3.0 default) / SAM 3.1 (optional)
│   ├─ CloudComPy310/         # CloudCompare Python — cloud SOR + voxel merge
│   ├─ mvs-texturing/         # texrecon UV-atlas photo texture (default texture_mode)
│   ├─ nvdiffrast/            # GPU vertex texture bake (alternative)
│   ├─ meshflow/              # per-object generative meshes (replaced ShapeR)
│   └─ PotreeConverter/       # Octree generation
│
├─ docs/                      # ARCHITECTURE, ROADMAP, FUTURE_VISION, SCANNING_GUIDE,
│                              #   pose_refinement.md, r3d_catalog.md, phaseN_report.md, migration/
├─ scripts/                   # start.sh (backend), serve_semantic.sh, launch_electron.sh,
│                              #   setup_vendors.sh, setup_pod_envs.sh, demo_pitch2.sh, …
└─ static/                    # Legacy viewer + camera capture
```

---

## Camera Pose & Localization

Accurate camera poses drive the whole pipeline — they place every depth observation into the global cloud, refine it via loop closure, and let the TSDF mesh be photo-textured from the source frames.

| Source | Type | Accuracy | When |
|--------|------|----------|------|
| **VGGT-Ω + loop closure + scale_align** | Feed-forward poses, SALAD/Sim3-refined, DA3-scaled | SOTA relative (CVPR'26), metric via DA3 anchor | **Default** (`vggtomega`) — these become `camera_poses.txt` used downstream |
| **MapAnything + loop closure** | Feed-forward poses (DA3 depth+K prior), SALAD/Sim3-refined | High relative, ~cm inter-frame | `mapanything` / `hybrid_cond` backends |
| **ARKit (Stray Scanner)** | VIO + IMU + LiDAR | ~cm absolute, metric-scale | `hybrid_cond` — injected into MapAnything as a **pose prior**, then loop-closed |
| **DA3 SLAM** | Neural SLAM (SALAD), loop-closed | High relative, ~cm inter-frame | `da3` standalone backend |
| **Phase R Sim(3) pose graph** | Instance-anchored inter-window refinement | Seam/drift correction, A/B fail-safe | Anchored pipeline (after SAM3, before fusion) |
| **Gizmo + ICP** | Manual alignment → refinement | Depends on user + ICP | Scan→BIM registration |

**Global consistency**: over long sequences, per-chunk poses drift. Three active layers
fix it: (1) SALAD/DINOv2 place-recognition + Sim3 loop closure inside VGGT-Long,
(2) `fine_register` plane-constrained inter-chunk registration (per-chunk pieces +
per-frame interpolation), and (3) **Phase R**'s instance-anchored inter-window Sim(3)
pose graph, whose refined poses/depth are **written back** so the cleaned cloud and the
TSDF integrate at the same corrected poses. The metric hierarchy is inviolable:
ChArUco/Umeyama + survey network always outrank semantic anchors; conflicts are logged,
never silently resolved.

---

## Pipeline Configuration

All pipeline parameters are centralized in `server/config.yaml`. Current shipped values:

```yaml
reconstruction:
  backend: "vggtomega"         # vggtomega (default) | mapanything | da3 | hybrid | hybrid_cond | lidar
  frames_selector: "dino"      # dino (DINO-cosine, default) | parallax (geometric) | stride | none
  blur_filter: true            # Laplacian blur cull before selection
  vggtomega:                   # default backbone (gated weights → vendor/vggt-omega-weights/)
    chunk_size: 120
    chunk_overlap: 60
    loop_closure: true
    resolution: 512
    scale_align: true          # metric: global similarity vs DA3 depth (fails hard if unrecoverable)
  mapanything:                 # alternative backbone (VGGT-Long framework)
    use_da3_priors: true       # feed DA3 metric depth + intrinsics as prior
    da3_prior_use_poses: false # also inject DA3 poses (auto-true in hybrid_cond)
    conf_threshold_coef: 0.75
  fine_register:
    enabled: true              # plane-constrained inter-chunk registration (per-chunk pieces)
  bundle_adjust:
    enabled: false             # OFF — degraded the metric vggtomega result in A/B; machinery kept
  dense_fusion:
    enabled: false             # OFF — legacy inter-keyframe ICP, superseded

frame_selection:
  dino_threshold: 0.99         # cosine keyframe threshold (frames_selector: dino)
  theta_parallax: 1.5          # deg (frames_selector: parallax)
  min_global_baseline_m: 0.3   # parallax mode: pure rotation / no baseline → ABORT

tsdf:                          # final stage: textured mesh
  voxel_length: 0.012          # 1.2 cm voxels
  depth_source: mapanything    # Ω keyframe depth ONLY — same source as the cloud
  tsdf_tile_length_m: 10.0     # 3D cube tiling welded into one mesh
  tsdf_max_edge_m: 0.10        # cull triangles bridging gaps/discontinuities
  fill_holes: true
  texture_mode: "texrecon"     # UV-atlas photographic texture (MVS-Texturing)

alignment:
  method: "scale+se3"          # SIM3 alignment
  auto_leveling:
    enabled: true              # RANSAC floor detection → gravity

bim:
  deviation:
    tolerance_mm: 50           # C2M threshold (warning/error/critical: 10/20/30 mm)
    coverage_proximity_m: 0.15

semantic:                      # Phase 0 service
  service: { host: 127.0.0.1, port: 8799 }
  backends:
    qwen_local: { model_id: "Qwen/Qwen3-VL-8B-Instruct" }        # default
    qwen_local_large: { model_id: "Qwen/Qwen3-VL-32B-Instruct-FP8" }  # opt-in candidate
```

---

## Environments & Services

Each heavy component lives in its own conda env (export YAMLs in `docs/migration/`):

| Env | Used by |
|-----|---------|
| `da3` | Backend server (`scripts/start.sh`) + DA3 |
| `semantic` | vLLM / Qwen3-VL service (`scripts/serve_semantic.sh`) |
| `sam3` | SAM3 segmentation worker |
| `mapanything` | MapAnything / VGGT-Long / VGGT-Ω backbone (+ BA tooling) |
| `CloudComPy310` | CloudCompPy cloud cleaning |
| `meshflow` | Per-object generative meshes |
| `nodejs` | Vite dev server + Electron |

`init_pod.sh` boots everything on a fresh pod as tmux sessions: `backend` (API on
8765), `vite` (UI dev server), `electron` (desktop app over noVNC on `:6080/vnc.html`,
skip with `START_ELECTRON=0`), `semantic` (vLLM + healthcheck window, skip with
`START_SEMANTIC=0` when the GPU is fully needed by a heavy reconstruction), `claude`.

**Hardware**: developed and validated on a single **RTX A6000 48 GB** (sm_86), with the
semantic service and a full reconstruction coexisting on the same GPU (Phase 7
coexistence probe; `gpu_memory_utilization` budgeted in `config.yaml`). Everything runs
locally — no paid external APIs.

---

## Quick Start

### 1. Clone WITH submodules (required — a plain `git clone` will NOT work)

Several vendored dependencies are pinned as **git submodules**, including two **STAC forks
that carry local patches the pipeline depends on**:

| Submodule | Remote | Notes |
|-----------|--------|-------|
| `vendor/VGGT-Long` | `hernanbarreto/VGGT-Long` (STAC fork) | loop-closure + sky-removal + DA3-prior patches + VGGT-Ω adapter |
| `vendor/depth-anything-3` | `hernanbarreto/Depth-Anything-3` (STAC fork, **private**) | cam-encoder pose conditioning + sky drop |

```bash
# Clone the repo AND all submodules in one step (recommended)
git clone --recursive https://github.com/hernanbarreto/stac-build.git
cd stac-build

# If you already cloned without --recursive:
git submodule update --init --recursive
```

> ⚠️ `vendor/depth-anything-3` points to a **PRIVATE** STAC fork. You need GitHub access to
> `hernanbarreto/Depth-Anything-3` (a PAT / SSH key configured) or the submodule fetch fails.
> The patches there are **required** — without them DA3 behaves differently than here.

### 2. Provision the git-ignored vendors (NOT in the repo)

Heavy/third-party vendors are **git-ignored** and are **not** fetched by clone or by Docker
(`Dockerfile` does `COPY vendor/ ./vendor/`, i.e. it copies whatever is already on disk).
The git-based ones are pinned and restored automatically:

```bash
bash scripts/setup_vendors.sh          # clone every pinned git vendor + init submodules
bash scripts/setup_vendors.sh --list   # show the full manifest without touching anything
```

This restores the pinned clones — `r3d`, `sam31`, `nvdiffrast`, `meshflow`,
`mvs-texturing`, `oneTBB-src`, `vggt-omega`, `ShapeR` — at their locked commits.
The **non-git** vendors (weights / build trees / prebuilt binaries) still need manual
provisioning: `sam3` (default segmentation baseline), `cloudcompy` / `CloudComPy310`,
`MapAnything2`, `PotreeConverter`, `oneTBB` (built from `oneTBB-src`), and
`vggt-omega-weights` (**gated** — request access at `huggingface.co/facebook/VGGT-Omega`,
place `vggt_omega_1b_512.pt` in `vendor/vggt-omega-weights/`). The authoritative
inventory — every vendor, its source, pin, and provisioning method — is
[`vendor/VENDORS.lock.md`](vendor/VENDORS.lock.md).

### 3. Set up environments & download model weights

```bash
bash scripts/setup_pod_envs.sh         # restore the conda envs (da3, sam3, mapanything, semantic, …)

./setup_weights.sh all                 # da3 (DINO-SALAD) + sam3 + vlm + semantic (Qwen3-VL-8B)
# or individually: ./setup_weights.sh  da3 | sam3 | vlm | semantic | semantic-large
# (DA3 / InternVL3 model weights auto-download via HF Hub on first run)
```

### 4. Run

```bash
bash init_pod.sh          # pod boot: backend + UI + Electron/noVNC + semantic, in tmux

# or start pieces manually:
bash scripts/start.sh              # backend (env da3) → https://localhost:8765
bash scripts/serve_semantic.sh     # semantic service → http://127.0.0.1:8799/v1
cd ui && npm run dev:web           # UI dev server (Vite)

# Docker (copies the local vendor/ tree into the image):
docker compose up --build
```

Access the application at `https://localhost:8765` (self-signed certificate).

---

## Supported Formats

| Type | Formats |
|------|---------|
| **Video input** | MP4, AVI, MOV |
| **LiDAR input** | Stray Scanner (depth PNGs + odometry CSV + intrinsics CSV) |
| **BIM models** | IFC 2x3, IFC 4, IFC 4.3 |
| **Point clouds** | PLY (native), Potree octree |
| **Meshes** | GLB (TSDF textured surface mesh; per-object meshes via MeshFlow) |
| **Export** | PLY, GLB, JSON metrics, Potree, Markdown reports (ES/FR) |

---

## Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for the full development roadmap and [FUTURE_VISION.md](docs/FUTURE_VISION.md) for the strategic platform vision.

---

<p align="center">
  <sub>Designed and developed by <strong>Hernán Barreto</strong></sub>
</p>
