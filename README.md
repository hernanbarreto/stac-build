<p align="center">
  <img src="docs/assets/stac_banner.png" alt="STAC Build" width="360"/>
</p>

<p align="center">
  <strong>Spatio-Temporal Awareness Core — Construction Dimensional Control System</strong>
</p>

<p align="center">
  <em>AI-powered As-Built vs As-Planned comparison via dense metric 3D reconstruction, photometric surface refinement and BIM deviation analysis</em>
</p>

---

## What is STAC?

**STAC Build** is a construction dimensional control system that compares **As-Built
reality** (captured with a smartphone video) against **As-Planned design** (BIM/IFC
models) to detect geometric deviations and track construction progress.

From a phone video (optionally iPad/iPhone LiDAR) the system produces a dense,
**metric** point cloud and a **photometrically refined textured mesh**, registers them
against the BIM/IFC model, and measures deviations and coverage. A persistent local
**Qwen3-VL semantic layer** can then segment the scene, classify instances, detect
findings, answer spatial questions with deterministic measurement tools, and draft
bilingual supervision reports — all local, no paid external APIs.

---

## The default pipeline (precision mode, `vggtomega_pgsr`)

Orchestrated by `pipeline_manager.py`. Each stage runs as an isolated subprocess
(spawned process group, Pipe IPC); the pipeline is **resume-aware** (per-stage artifact
probes skip finished stages) and **fail-fast** (a stage that cannot do its job raises —
no silent fallbacks). GPU-heavy stages take the GPU exclusively: the vLLM semantic
service is stopped before reconstruction, PGSR and SAM3, and restarts on the next VLM
use.

The stage order is:

```
RECONSTRUCTION → [VLM → SAM3]* → CLOUDCOMPY → PGSR → TSDF
```

\* The semantic chain (VLM auto-prompter → SAM3) is registered in the pipeline but
currently **disabled by default** (`pipeline.auto_segment: false`) while the SAM3
session handling is reworked — segmentation runs **on demand** from the UI or CLI
instead. `pipeline.auto_tsdf: true` guarantees every run ends with the textured mesh,
never Potree-only.

### Stage 1 — Metric reconstruction (VGGT-Ω, the SIMPLE one-pass concept)

One concept, validated against the VGGT-Ω web demo: **sparse motion keyframes → ONE
reconstruction pass → no window seams → no "onion"**. Chunking exists only as a
measured escape hatch (phase 2 below), never as the default.

```
📱 video frames (all of them, e.g. 1097)
 │
 ├─ Frame quality analysis (frames/quality.py)
 │    FFT sharpness + Laplacian variance (adaptive p15 blur gate)
 │    + inter_frame_diff = mean gray difference @320px (pixel-motion proxy)
 │    → frames/frame_quality.json
 │
 ├─ Motion keyframe selection (parallax-uniform)
 │    one keyframe per keyframe_motion_quantum (80) of ACCUMULATED pixel motion,
 │    sharpest frame per window; all-blurry windows keep their least-blurry frame.
 │    Standing still adds ~no keyframes; fast walking adds more (no 9 m jumps).
 │    → frames/selected_frames.json  (single source of truth downstream)
 │
 ├─ DA3 metric anchors: 12 evenly-spread keyframes, ISOLATED per-frame inference
 │    (depth-anything/DA3NESTED-GIANT-LARGE-1.1, no streaming — scale is a depth
 │    RATIO, poses don't participate) → output/da3_run/results_output/
 │
 ├─ VGGT-Ω single pass (vendor VGGT-Long framework, env `mapanything`)
 │    · sky masked per frame (skyseg.onnx) before confidence voting
 │    · single-pass gate: n ≤ min(600, free-VRAM budget) → one chunk, loop
 │      closure OFF (nothing to close)
 │    · confidence filter: drop the bottom 10% of valid points (percentile mode)
 │    → chunk PLYs + camera_poses.txt + per-frame Ω depth (omega_run/)
 │
 ├─ scale_align (reconstruction/scale_align.py) — FAIL-HARD metric scale
 │    s = median over keyframes of median(DA3/Ω depth) on the near-25% band ∩
 │    top-10%-confidence pixels; global_median mode (structured models lost their
 │    own CV gate in the A/B campaign). A VIO trajectory (docs/VIO_FORMAT.md),
 │    when present in the session, SETS the scale and DA3 becomes the cross-check.
 │    → output/scale_diagnostics.json (per-anchor ratios, MAD, scale_confidence)
 │
 ├─ Upright orientation (reconstruction/orient.py, gated)
 │    gravity = consensus camera-down over all poses; refuses below 0.7 alignment
 │    (the CloudCompy floor leveler is the fallback); floor at y=0
 │
 ├─ pose_refine (reconstruction/pose_refine.py) — SELF-GATED, enabled
 │    joint point-to-plane multi-view optimization over per-frame corrections
 │    (pair window 15, odometry smoothness, identity leash); applies only if the
 │    fresh held-out measurement actually improves, else identity
 │
 └─ Walk probe: trajectory length measured in METERS (post-scale).
      walk ≤ 15 m → the single pass IS the result.
      walk > 15 m → phase 2: chunked-metric re-run (below).
```

**Phase 2 — chunked-metric** (only when the measured walk exceeds Omega's 15 m
comfort range; Omega drifts ~1.3 cm/m past it):

- **Coverage trim**: static or optically-zoomed head/tail keyframe runs are dropped
  (median-step decade criterion + robust-z on per-frame focal) — mid-walk stretches are
  never cut.
- **Denser re-selection**: the quantum is re-derived so each ~12 m chunk holds ~45
  keyframes.
- **Chunks sized by walked meters** (12 m each, clamped [24, 150] keyframes, 50%
  overlap), each metric-locked by **3 DA3 anchors spread inside every chunk**.
- Vendor machinery (STAC fork of VGGT-Long): **SE(3) seam gluing from exact pixel
  correspondences** (scale is not a degree of freedom), a **scale graph** over seams +
  anchors with self-gated per-chunk scale drift, **frame ownership** (one writer per
  frame, non-owner backfills exactly the dropped pixels), **elastic per-frame seam
  consensus** (shared pixels share one 3D position, poses move with points),
  **intra-chunk consensus fields** (held-out gated, worst case identity), a **depth
  graph** (frames agree on shared-surface depth), and a session-derived **write-depth
  cap** that drops far points contradicted by near observations.
- **SALAD/DINOv2 loop closure runs here** (and only here — the single pass has nothing
  to close).

### Stage 2 — Cloud post-processing (CloudCompPy, env `CloudComPy310`)

Merge the chunk PLYs, inject per-point **traceability scalar fields** (`frame_global`,
`pixel_row`, `pixel_col`, `confidence` — they survive every later filter), voxel
subsample (5 mm) + SOR outlier removal → `cleaned_cloud.ply`. Then a **scene
consolidate** pass (normal-aware robust MLS, adaptive radius 2–6 cm, opposite faces
never merge) tightens the surface in place — the raw measurement is preserved as
`cleaned_cloud_raw.ply` and `surface_fit` residuals always use it. Finally the
**Potree octree** is built (confidence and origin fields carried into LAS).

### Stage 3 — PGSR photometric refinement (the precision stage, env `pgsr`)

The step that gives precision mode its name (~2 h/scene, exclusive GPU):

- **Scene export**: native-resolution keyframe images + the pipeline's poses/intrinsics
  in COLMAP layout, seeded with `cleaned_cloud.ply` (≤1.5 M points).
- **Training**: PGSR (Planar-based Gaussian Splatting, vendor `zju3dv/PGSR` @
  `de24f1a3`) at the vendor's published max-quality regime — 30 000 iterations, NCC
  scale 0.5, outdoor densify/cull thresholds, exposure compensation, native
  resolution. Geometric single-view (from iter 7000) and multi-view NCC losses make
  the Gaussians converge to actual surfaces. `torch.set_num_threads(8)` is
  load-bearing (vendor parity; without it the multi-view stage is ~10× slower on
  many-core boxes). Photometric *pose* refinement is OFF (it lost its A/B: RMS +11%).
- **Render/export**: every camera re-rendered → per-frame **photometrically verified
  depth** (`output/pgsr_render/frame_*.npz`) + `report.json` (PSNR, VRAM, timing).
- **Consistent cloud**: the viewer cloud is REBUILT by unprojecting the PGSR depths at
  the final poses through three gates — depth-edge cull, a 2-view consistency vote
  (2% relative tolerance), and a 20 m cap — then 6 mm voxel + SOR →
  `pgsr_cloud.ply`, and the **Potree octree is rebuilt from it**, so the cloud you
  see and the mesh agree. `cleaned_cloud.ply` is left untouched (it remains the
  source for segmentation and BIM).

### Stage 4 — TSDF textured mesh (Open3D CUDA)

`depth_source: auto` resolves to the **PGSR renders** whenever the precision stage ran
(explicit `pgsr_render` is fatal if renders are missing — never a silent fallback). In
precision mode the cleaned-cloud pixel mask is deliberately **dropped**
(`pgsr_mask_to_cleaned_cloud: false`): the rendered depth is photometrically
optimized, and masking would re-import the cloud's holes — this is what gives the mesh
**full coverage**. The multi-view geometric consistency filter (`mv_consistency`,
4 neighbours / 2 agreeing views within 2%) screens the depth on **both** paths: the Ω
chunk depth in fast mode, and the PGSR renders in precision mode (the same vote that
gates the consistent cloud — rendered depth is optimized, not multi-view verified).

Integration: VoxelBlockGrid on GPU, **12 mm voxels** (8/6 mm lost their A/B: worse RMS
at 1.8–2.9× cost), SDF truncation 6 cm, fixed 10 m 3D cube tiling welded on a shared
global grid, depth clipped to the reliable 15 m band. Post-mesh chain: long-edge cull
(0.10 m bridge/spike triangles) → speck cleanup (aborts if it would drop >5% of
triangles) → hole fill (≤0.25 m — doors and windows stay open) → Taubin smoothing (×5,
shrinkage-free) → quadric decimation (≈4 M triangles, auto-scaled to scene size). An
**ICP-snap gate** aligns mesh + cameras to the cloud before texturing, but only when
fitness/RMSE/motion bounds all pass — otherwise the mesh stays in the pose frame.
Texture: **texrecon** UV-atlas photographic bake (MVS-Texturing, up to 400–600 views);
the GLB ships meshopt + WebP compressed (~6×), with the uncompressed original kept
alongside.

**Artifacts of a finished run** (under the session's `output/`): `cleaned_cloud.ply` +
`cleaned_cloud_raw.ply`, `pgsr_cloud.ply` (+ rebuilt `potree/`), `pgsr_render/` +
`pgsr_model/`, `tsdf/scene/scene.glb` (+ `.orig`), `camera_poses.txt`,
`scale_diagnostics.json`, `pose_refine_report.json`, `scene_consolidate_report.json`.

### Keyframe density (current experiment)

`keyframe_motion_quantum` is **80** since 2026-08-15 (user decision after a visual A/B
on a real scene: markedly more complete, less ghosting — denser keyframes give PGSR
~3× more training views). The measured trade-offs on the same scene: scale anchor MAD
3.5%→11.1% (confidence 0.82→0.50), the walk probe over-measures (drift zigzag) so
chunked mode fires more readily, and runtime roughly doubles. **Status: under
evaluation across more scenes** — 250 had won the earlier pose-proxy A/B and remains
the documented fallback.

---

## Pose & scale accuracy — measured state (A/B campaign, 2026-08)

Every default below was decided by a pre-registered A/B on real scenes (full tables in
`docs/scale_ab_results.md`, `docs/phase_bc_ab_results.md`):

- **Scale**: one global similarity from 12 DA3 anchors (near-band ∩ top-confidence).
  Held-out anchor depth error 4.5–9% median across test sessions; anchor ratio MAD
  3.5–6.7% (at quantum 250); every run persists `scale_diagnostics.json` (per-anchor
  ratios, jackknife, `scale_confidence` 0–1). Structured estimators (affine /
  depth-dependent) were rejected by their own cross-validation gate on all sessions.
  A **VIO trajectory** takes over the scale when present (`docs/VIO_FORMAT.md`).
- **Surface (fast mode)**: multi-view consistency filter ON — planar-patch RMS
  9.00→8.70 mm, fewer double surfaces, faster TSDF.
- **Precision mode** (`vggtomega_pgsr`, the default): **+86% mesh coverage** and a
  better error tail (p90 10.83→10.01 mm) at ~2 h extra per scene; planar RMS ties the
  fast mode.
- **Active refinement**: the point-to-plane `pose_refine` stage is ON and self-gated
  (measured −52% ghost layering on jerky close-range captures; applies nothing unless
  its own held-out metric improves).
- **Retired machinery** (shipped, off, do not re-enable without re-validating): the
  two-pass COLMAP/Ceres bundle adjustment over VGGSfM tracks (degraded the metric
  result), `fine_register` plane-constrained chunk registration (no usable signal),
  `dense_fusion` inter-keyframe ICP (superseded), native-resolution depth refinement
  (doubled double-surface incidence), PGSR photometric pose refinement (RMS +11%).
- **Not yet measured**: parity vs RealityScan (external scorecard pending).

---

## Technology stack

### Models & reconstruction

| Component | Technology | Role |
|-----------|-----------|------|
| **VGGT-Ω** (backbone) | [VGGT-Ω](https://vggt-omega.github.io/) 1B @ 512 px (CVPR 2026) inside the STAC fork of [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) | Feed-forward camera poses + dense points, dynamic-scene robust. Gated weights (`vendor/vggt-omega-weights/vggt_omega_1b_512.pt`) |
| **DA3** | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) `DA3NESTED-GIANT-LARGE-1.1` (STAC private fork) | The **metric anchor**: isolated per-frame depth on 12 keyframes → global scale; also a standalone SLAM backend |
| **PGSR** | [PGSR](https://github.com/zju3dv/PGSR) @ `de24f1a3` (planar Gaussian splatting; Inria non-commercial license) | The precision stage: photometric surface optimization → verified depth renders that feed the TSDF and rebuild the viewer cloud |
| **Loop closure** | [DINOv2](https://github.com/facebookresearch/dinov2) + SALAD place recognition | Chunked-metric mode only (the single pass has no loops to close) |
| **Sky removal** | `skyseg.onnx` | Per-frame sky mask before confidence voting |
| **TSDF + mesh** | [Open3D](https://www.open3d.org/) VoxelBlockGrid (CUDA) | 12 mm textured mesh, tiled + welded, hole-filled, Taubin-smoothed |
| **Texture** | [MVS-Texturing](https://github.com/nmoehrle/mvs-texturing) (texrecon) | UV-atlas photographic texture (default); nvdiffrast GPU vertex bake available |
| **Cloud post** | [CloudComPy](https://www.cloudcompare.org/) | Merge + scalar-field injection, voxel + SOR, MLS scene consolidate |
| **Point cloud viz** | [Potree](https://potree.github.io/) + [Three.js](https://threejs.org/) | Level-of-detail streaming of the consistent cloud |
| **Semantic service** | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) on [vLLM](https://github.com/vllm-project/vllm) 0.19 | Persistent local VLM (`127.0.0.1:8799`, OpenAI-compatible, tool calling); optional 32B-FP8 candidate backend |
| **Segmentation** | [SAM 3.1](https://github.com/facebookresearch/sam3) Object Multiplex (**default**; SAM 3.0 rollback available) | Tracked per-frame 2D instance masks — 6.1× faster than 3.0 at equivalent quality (measured); tracking IS the instance identity |
| **Per-object meshes** | MeshFlow (vendored, gated ckpt) | Generative per-object visual meshes — explicitly **non-metric**, never for architectural classes |
| **Surface fitting** | `reconstruction/surface_fit/` (plane→cylinder→sphere→swept→b-spline ladder) | Deterministic primitive fitting + residuals — the measurement engine behind findings and spatial Q&A |

### Reconstruction backends (`reconstruction.backend`)

| Backend | Status | What it is |
|---------|--------|------------|
| **`vggtomega_pgsr`** | **DEFAULT** | Full VGGT-Ω pipeline + the PGSR precision stage (+86% mesh coverage, ~2 h extra) |
| `vggtomega` | wired | Fast mode: same reconstruction, no PGSR stage |
| `da3` | wired | DA3 streaming SLAM standalone (neural depth, SALAD loops) |
| `mapanything` | wired (legacy fallback) | DA3 depth+K prior into MapAnything per chunk (VGGT-Long framework) |
| `hybrid` / `hybrid_cond` | wired | Stray Scanner captures: LiDAR-calibrated DA3, optionally ARKit-pose-conditioned |
| `lidar` | wired | Pure LiDAR backprojection (Stray only, no neural inference) |
| `gaus_slam*`, `nerfstudio` | removed | No dispatch branch remains; config vestiges only |

### Platform

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **BIM parsing** | [IfcOpenShell](https://ifcopenshell.org/) | IFC geometry extraction |
| **Backend** | Python, FastAPI + uvicorn, WebSockets | API on port **8765**; HTTPS (self-signed) when started via `scripts/start.sh` |
| **Frontend** | React 18 + TypeScript + Vite (+ Electron desktop) | IDE-style viewer; on headless pods the Electron app is viewable over noVNC (`:6080/vnc.html`) |
| **Auth** | JWT (HS256) + bcrypt | Role-based access (**admin / manager / viewer**), team workspaces, activity logging |
| **Infrastructure** | Conda envs per component, CUDA sm_86 | Dev/reference GPU: RTX A6000 48 GB; everything local, no paid APIs |

---

## Semantic Intelligence Layer (Phases 0–7)

A persistent **Qwen3-VL** vision-language layer sits on top of the geometry pipeline,
turning the point cloud into a queryable, supervised, reportable scene. It is governed
by one inviolable rule:

> **The VLM proposes, describes, detects, classifies and orchestrates.
> It NEVER measures.** Every metric comes from deterministic tools over geometry /
> `surface_fitting`. Every output is tagged `vlm_proposed`, `tool_measured`, or
> `human_validated`.

**Current integration**: the in-pipeline chain (VLM → SAM3) is temporarily disabled by
default (`pipeline.auto_segment: false`) while SAM3 session handling is reworked —
a default run is pure geometry, and segmentation + the phases below run **on demand**
(UI, CLI, or API). Consequences of a segmentation-less run: PGSR trains without
dynamic-object masks (static-scene assumption, logged explicitly) and no per-object
mesh crops are produced.

| Phase | Path | Purpose | Wiring |
|-------|------|---------|--------|
| **0 · Semantic service** | `server/semantic/` | vLLM 0.19 serving **Qwen3-VL-8B-Instruct** on `127.0.0.1:8799` (OpenAI-compatible, hermes tool parser, 8 images/prompt, deterministic T=0), own `semantic` env; every consumer goes through `semantic_client` (`qwen_local` default, `qwen_local_large` = Qwen3-VL-32B-FP8 candidate). JSONL call log, healthcheck, GPU handover: stopped for reconstruction/PGSR/SAM3, restarted by the next consumer | service |
| **1 · Auto-prompter** | `server/segmentation/autoprompt/` | Scene understanding over ~8 keyframes → ONE RICH noun phrase per object type (**concept phrases, no boxes**); each phrase runs as its own SAM3 concept session over the sampled frames — SAM3 finds, labels and tracks, its tracking IS the identity. The construction vocabulary is a canonicalization overlay, never a detection filter: the system segments *everything*. Batch CLI available | CLI + pipeline (gated) |
| **Instance data layer** | `server/phase_r/` | What remains of the former "Phase R": the canonical instance store (`scene_r.db`, R3D-derived schema — single source of objects for phases 2/3/5/6), gravity-aligned OBB geometry, deterministic plane-fit depth regularization. The inter-window Sim(3) anchoring machinery was **removed 2026-07-09** (the one-pass pipeline has no window seams, and it never survived its own A/B gate) | library |
| **2 · Classification** | `server/phase2_classify/` | Per-instance class / material / state via best-keyframe crops; label-conflict flags; architectural→`surface_fitting` routing. Enriches the store, never creates parallel objects | CLI |
| **3 · Findings** | `server/phase3_findings/` | Cracks / moisture / spalling / corrosion detection, 3D-anchored via depth + pose, multi-view deduped, residual-correlated severity; everything born `proposed` until human-validated | CLI |
| **4 · Capture QC** | `server/phase4_qc/` | Ingestion pre-filter (cheap blur/exposure, VLM only on the ambiguous band — frames are never silently deleted) + post-reconstruction coverage → recapture checklist | CLI |
| **5 · Spatial Q&A** | `server/phase5_qa/` | ~20 deterministic `SpatialTools` (distance, clearance, plumb, level, span, volume, fits_through, flatness, height profile, alignment health…) + user-defined evaluation volumes, driven by a bounded tool-calling loop | CLI + **HTTP**: `POST /api/spatial_qa`, `POST /api/scene/objects`, `/api/scene/volumes/*`, `GET /api/semantic/status?warmup=` |
| **6 · Report** | `server/phase6_report/` | Bilingual **ES/FR** supervision-report draft; every number carries a `tool(args)+timestamp` trace; VLM prose flagged *pending validation* | CLI |
| **7 · Validation** | `server/phase7_validation/` | 28-question spatial-Q&A suite (incl. insufficient-data traps) + the GPU coexistence probe (vLLM + full reconstruction on one A6000 48 GB) | CLI |

**Run the semantic service** (own env, isolated from the reconstruction envs):

```bash
bash scripts/build_semantic_env.sh     # one-time, on a fresh box
bash setup_weights.sh semantic         # download Qwen3-VL-8B-Instruct
bash scripts/serve_semantic.sh         # → http://127.0.0.1:8799/v1
# optional 32B candidate: bash setup_weights.sh semantic-large
```

### Immersive AI Assistant (in the viewer)

The 3D viewer includes an **AI Assistant** panel (`AssistantPanel.tsx`): ask a question
in natural language and the answer is measured by the Phase-5 tools and **replayed as
animated three.js geometry** — you literally see *how* each distance, volume, plumb
angle or clearance was measured. Drop **evaluation volumes** into the scene to assess
spaces (occupancy, free m³, "does this fit?"). Chat backed by `POST /api/spatial_qa`;
geometry by `POST /api/scene/*`.

### Reproducible end-to-end demo

```bash
scripts/demo_pitch2.sh <session_dir> scene.db out_dir
# segment → instance store → classify → findings → coverage
# → 8 spatial questions answered WITH traces → bilingual report draft
```

---

## BIM integration & visualization

- Full IFC parsing (IFC 2x3 / 4 / 4.3): geometry extraction for all physical elements.
- Scan-to-BIM registration: gizmo alignment + ICP refinement (50 iterations).
- **Cloud-to-Mesh (C2M) deviation** per element: `tolerance_mm: 50`, heatmap bands
  warning/error/critical at 10/20/30 mm.
- **Coverage analysis**: % of BIM surface observed (`coverage_proximity_m: 0.15`);
  per-element quality classification Good ≥80% / Regular ≥50% / Bad.
- **Sábana**: color-coded deviation map rendered as a point overlay on the
  semi-transparent BIM (3 mm points, 1 mm subsample), with per-element mean / max /
  P95 / pass-rate statistics.
- **Potree streaming**: LOD octree of the consistent cloud with progressive loading
  and point-budget management.

## Team & session management

- JWT authentication, roles **admin / manager / viewer**, multi-user team workspaces,
  per-user activity logging.
- Session persistence with full pipeline state; resume-aware re-runs.
- Real-time WebSockets: `/ws/logs`, `/ws/viewer`, `/ws/camera`, `/ws/team`, `/ws/scan`.

---

## Architecture

```
stac-build/
├─ server/                    # Python backend (FastAPI + uvicorn, port 8765)
│   ├─ main.py                # FastAPI app, WebSockets, API routes (+ auth/team/spatial-QA routers)
│   ├─ pipeline_manager.py    # Stage orchestrator — default order:
│   │                          #   RECONSTRUCTION → [VLM → SAM3]* → CLOUDCOMPY → PGSR → TSDF
│   │                          #   (* gated by pipeline.auto_segment, currently false)
│   ├─ workers/               # Subprocess workers (GPU-isolated, WorkerPipe IPC)
│   │   ├─ map_worker.py      #   reconstruction dispatcher: frame quality → motion keyframes
│   │   │                      #   → DA3 anchors → Ω pass → scale_align → orient → pose_refine
│   │   │                      #   → walk probe → (chunked-metric phase 2 when needed)
│   │   ├─ cloudcompy_worker.py # cloud merge/clean/consolidate + Potree
│   │   ├─ pgsr_worker.py     #   PGSR precision stage + consistent-cloud rebuild
│   │   ├─ tsdf_worker.py     #   TSDF textured mesh (Open3D VoxelBlockGrid, GPU)
│   │   ├─ vlm_worker.py / sam3_worker.py   # semantic chain (gated)
│   │   └─ instance_cleaner_worker.py       # on-demand API action (not a pipeline stage)
│   ├─ reconstruction/        # scale_align, orient, pose_refine, chunk_plan, pgsr_{export,train,cloud},
│   │   │                      #   dynamic_masks, mv_consistency, texture_bake, surface_fit/
│   │   └─ (colmap_ba, vggt_tracks, fine_register, …)   # retired machinery, shipped but off
│   ├─ frames/                # quality analysis + selectors (motion / fps / dino / parallax)
│   ├─ semantic/              # Phase 0: vLLM launcher, LLMBackend client, healthcheck, call log
│   ├─ segmentation/          # SAM3 wrapper (3.1 multiplex), mask store, autoprompt/ (Phase 1)
│   ├─ phase_r/               # instance store (scene_r.db) + OBB geometry (data layer)
│   ├─ phase2_classify/ … phase7_validation/   # Phases 2–7 (CLI each; Phase 5 also HTTP)
│   ├─ bim/                   # C2M deviation, sábana, coverage, registration (root shims kept)
│   ├─ ingestors/             # Stray Scanner auto-detection + loaders (ARKit poses, LiDAR depth)
│   ├─ auth/                  # JWT auth, roles, team workspaces, activity log
│   └─ config.yaml            # All pipeline configuration (single source of truth)
│
├─ ui/                        # React 18 + TypeScript + Vite (+ Electron desktop)
│   └─ src/components/        # AssistantPanel, Viewport (Potree + GLB w/ meshopt),
│                              #   InteractiveSegmentation, BIM panels, TeamPanel, …
│
├─ vendor/                    # AI model integrations (git-ignored; authoritative
│   │                          #   inventory + pins: vendor/VENDORS.lock.md)
│   ├─ VGGT-Long/             # STAC fork (submodule): Ω adapter + chunked-metric machinery
│   ├─ depth-anything-3/      # STAC fork (PRIVATE submodule): DA3 + patches
│   ├─ vggt-omega/ + vggt-omega-weights/   # Ω code + gated 1B/512 checkpoint
│   ├─ pgsr/                  # PGSR @ de24f1a3 (patched: no pytorch3d dep)
│   ├─ sam3/ + sam31/         # SAM 3.0 / SAM 3.1 (3.1 = default)
│   ├─ CloudComPy310/, MapAnything2/, r3d/, mvs-texturing/, nvdiffrast/,
│   └─ meshflow/, PotreeConverter/, oneTBB/
│
├─ docs/                      # ARCHITECTURE, SCANNING_GUIDE, VIO_FORMAT, A/B results,
│                              #   phaseN reports, migration/ (env exports)
├─ scripts/                   # start.sh, serve_semantic.sh, launch_electron.sh,
│                              #   setup_vendors.sh, setup_pod_envs.sh, demo_pitch2.sh, …
└─ static/                    # Legacy viewer + camera capture
```

---

## Configuration (current shipped defaults)

All pipeline parameters live in `server/config.yaml` (loaded once at server start — a
config change requires a backend restart). The load-bearing defaults:

```yaml
pipeline:
  auto_tsdf: true              # every run ALWAYS ends with the textured mesh
  auto_segment: false          # semantic chain on demand (temporarily out of the default run)

reconstruction:
  backend: "vggtomega_pgsr"    # DEFAULT: precision mode (+86% mesh coverage, ~2 h/scene extra)
                               # "vggtomega" = fast mode without the PGSR stage
  vggtomega:
    scale_align: true          # metric scale vs DA3 anchors — fails hard if unrecoverable
    scale_mode: global_median  # structured models lost their own CV gate in the A/B
    scale_vio: true            # a VIO trajectory, when present, SETS the scale
    loop_closure: true         # effective only in chunked-metric mode
  simple:                      # the one-pass pipeline (enabled)
    frame_selection: motion    # parallax-uniform keyframes (fps/dino/parallax also exist)
    keyframe_motion_quantum: 80.0   # 80 since 2026-08-15 (visual A/B) — UNDER EVALUATION;
                               # 250 is the previous pose-proxy-validated value
    max_walk_single_pass_m: 15.0    # beyond → phase 2 chunked-metric re-run
    chunk_walk_m: 12.0         # phase-2 chunks sized by walked meters, 50% overlap
    conf_percentile: 10.0      # drop the bottom 10% of valid points
    scale_anchor_frames: 12    # DA3 metric anchors (12 vs 24/32 A/B'd: 12 stays)
  pose_refine:
    enabled: true              # point-to-plane joint refinement, SELF-GATED
  pgsr:                        # precision stage (vendor PGSR @ de24f1a3, env `pgsr`)
    iterations: 30000          # ~2 h/scene, PSNR ~24.8, ~11 GB VRAM (measured)
    ncc_scale: 0.5             # vendor max-quality regime + exposure compensation
    pose_refine: false         # photometric pose refinement LOST its A/B (RMS +11%)
    consistent_cloud: true     # rebuild cloud + Potree from the PGSR depths
  bundle_adjust: { enabled: false }   # retired: degraded the metric result
  fine_register: { enabled: false }   # retired: no usable signal

tsdf:                          # final stage: textured mesh
  depth_source: auto           # PGSR renders when the precision stage ran; else Ω depth
  pgsr_mask_to_cleaned_cloud: false   # precision mode = full coverage (verified depth)
  voxel_length: 0.012          # 12 mm voxels (8/6 mm A/B'd: worse at 1.8–2.9× cost)
  mv_consistency: true         # multi-view depth filter (fast-mode depth path)
  native_depth_method: "off"   # A/B'd: doubled double-surface incidence
  tsdf_max_edge_m: 0.10        # cull bridge/spike triangles
  texture_mode: "texrecon"     # UV-atlas photographic texture

models:
  segmentation:
    version: "sam3.1"          # Object Multiplex — 6.1× faster than 3.0, measured

semantic:                      # Phase 0 service
  service: { host: 127.0.0.1, port: 8799 }
  backends:
    qwen_local: { model_id: "Qwen/Qwen3-VL-8B-Instruct" }             # default
    qwen_local_large: { model_id: "Qwen/Qwen3-VL-32B-Instruct-FP8" }  # opt-in candidate

bim:
  deviation:
    tolerance_mm: 50           # C2M threshold (warning/error/critical: 10/20/30 mm)
    coverage_proximity_m: 0.15
```

---

## Environments & services

Each heavy component lives in its own conda env (export YAMLs in `docs/migration/`):

| Env | Used by |
|-----|---------|
| `da3` | Backend server (`scripts/start.sh`) + DA3 + SAM3 (in-process) |
| `semantic` | vLLM / Qwen3-VL service (`scripts/serve_semantic.sh`) |
| `mapanything` | VGGT-Ω / VGGT-Long / MapAnything reconstruction pass |
| `pgsr` | PGSR precision trainer (`server/run_pgsr.sh`) |
| `CloudComPy310` | CloudCompPy cloud cleaning |
| `meshflow` | Per-object generative meshes |
| `nodejs` | Vite dev server + Electron + GLB compression tools |

`init_pod.sh` boots everything on a fresh pod as tmux sessions: `backend` (API on
8765), `vite` (UI dev server), `electron` (desktop app over noVNC on `:6080/vnc.html`,
skip with `START_ELECTRON=0`), `semantic` (vLLM + healthcheck window, skip with
`START_SEMANTIC=0` when the GPU is fully needed by a heavy reconstruction), `claude`.

**Hardware**: developed and validated on a single **RTX A6000 48 GB** (sm_86), with the
semantic service (GPU util 0.50) and a full reconstruction coexisting on the same GPU
(Phase 7 coexistence probe). Everything runs locally — no paid external APIs.

---

## Quick start

### 1. Clone WITH submodules (required — a plain `git clone` will NOT work)

Two vendored dependencies are pinned as **git submodules**, both STAC forks carrying
local patches the pipeline depends on:

| Submodule | Remote | Notes |
|-----------|--------|-------|
| `vendor/VGGT-Long` | `hernanbarreto/VGGT-Long` (STAC fork) | Ω adapter + chunked-metric machinery (seam gluing, metric lock, elastic consensus, depth graph) + sky removal |
| `vendor/depth-anything-3` | `hernanbarreto/Depth-Anything-3` (STAC fork, **private**) | cam-encoder pose conditioning + sky drop |

```bash
git clone --recursive https://github.com/hernanbarreto/stac-build.git
cd stac-build
# if already cloned without --recursive:
git submodule update --init --recursive
```

> ⚠️ `vendor/depth-anything-3` points to a **PRIVATE** STAC fork. You need GitHub
> access to `hernanbarreto/Depth-Anything-3` (PAT / SSH key) or the submodule fetch
> fails. The patches there are **required**.

### 2. Provision the git-ignored vendors

Heavy/third-party vendors are git-ignored and not fetched by clone or Docker. The
git-based ones are pinned and restored automatically:

```bash
bash scripts/setup_vendors.sh          # clone every pinned git vendor + init submodules
bash scripts/setup_vendors.sh --list   # show the full manifest without touching anything
```

This restores the pinned clones — `pgsr`, `r3d`, `sam31`, `nvdiffrast`, `meshflow`,
`mvs-texturing`, `oneTBB-src`, `vggt-omega` — at their locked commits. The **non-git**
vendors (weights / build trees / prebuilt binaries) still need manual provisioning:
`sam3`, `cloudcompy` / `CloudComPy310`, `MapAnything2`, `PotreeConverter`, `oneTBB`,
and `vggt-omega-weights` (**gated** — request access at
`huggingface.co/facebook/VGGT-Omega`, place `vggt_omega_1b_512.pt` in
`vendor/vggt-omega-weights/`). The authoritative inventory — every vendor, its source,
pin, and provisioning method — is [`vendor/VENDORS.lock.md`](vendor/VENDORS.lock.md).

### 3. Set up environments & download model weights

```bash
bash scripts/setup_pod_envs.sh         # restore the conda envs
./setup_weights.sh all                 # da3 (DINO-SALAD) + sam3 + vlm + semantic (Qwen3-VL-8B)
# or individually: ./setup_weights.sh  da3 | sam3 | vlm | semantic | semantic-large
# (DA3 model weights auto-download via HF Hub on first run)
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

## Supported formats

| Type | Formats |
|------|---------|
| **Video input** | MP4, AVI, MOV |
| **LiDAR input** | Stray Scanner (depth PNGs + odometry CSV + intrinsics CSV) |
| **VIO input** | `vio_trajectory.csv/json` (`docs/VIO_FORMAT.md`) — sets the metric scale when present |
| **BIM models** | IFC 2x3, IFC 4, IFC 4.3 |
| **Point clouds** | PLY (native), Potree octree |
| **Meshes** | GLB (TSDF textured surface mesh, meshopt+WebP compressed) |
| **Export** | PLY, GLB, JSON metrics, Potree, Markdown reports (ES/FR) |

---

## Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for the development roadmap and
[FUTURE_VISION.md](docs/FUTURE_VISION.md) for the strategic platform vision.

---

<p align="center">
  <sub>Designed and developed by <strong>Hernán Barreto</strong></sub>
</p>
