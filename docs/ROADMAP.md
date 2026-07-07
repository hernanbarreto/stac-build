# STAC Build — Development Roadmap

> Dependency-ordered, no deadlines — just logical sequence.  
> Check items off as they are completed.

---

## ✅ Semantic Intelligence Layer — Qwen3-VL (Delivered)

A persistent Qwen3-VL layer over the reconstruction pipeline, under one rule:
**the VLM proposes/describes/classifies/orchestrates; it never measures** — every
metric is a deterministic tool over geometry, tagged `vlm_proposed` /
`tool_measured` / `human_validated`. Implemented, tested, and integrated
end-to-end (see `docs/phase0_report.md` … `docs/phase7_report.md`).

- [x] **Phase 0** — persistent vLLM/Qwen3-VL semantic service (`semantic` env, `127.0.0.1:8799`)
- [x] **Phase 1** — understanding-driven, open-vocabulary auto-prompter → SAM3
- [x] **Phase R** — instance store, gravity, vote/onion metrics, inter-window Sim(3) pose graph, depth regularization, A/B fail-safe, **writeback into the TSDF fusion**
- [x] **Phase 5** — spatial Q&A tools + user-defined evaluation volumes + orchestrator (`/api/spatial_qa`, `/api/scene/*`)
- [x] **Phase 2** — per-instance class / material / state, conflict flags, structural routing
- [x] **Phase 3** — 3D-anchored visual findings (cracks/moisture/…), multi-view dedup, honest precision eval
- [x] **Phase 4** — capture QC pre-filter + coverage recapture checklist (never deletes frames)
- [x] **Phase 6** — bilingual ES/FR supervision report, fully traceable
- [x] **Phase 7** — 28-question validation suite, GPU coexistence, Pitch-2 demo
- [x] **Immersive AI Assistant UI** — chat that replays each answer as animated 3D measurements; evaluation-volume placement

**Open (external data / human only — code + tests complete):**
- [ ] `qwen_local_large` run (needs the larger quantized weights + a service restart)
- [ ] Phase 1 per-class recall/precision (needs hand-segmented ground truth)
- [ ] Phase R full 160 m corridor A/B + real multi-window pose writeback (needs a multi-window scan)
- [ ] Phase 3 full ~50-crop precision + Phase 4 false-discard rate (need human labels)

---

## ✅ Tier 0: Core Platform (Done)

Everything here is implemented and working.

- [x] MapAnything dense 3D reconstruction (chunked, SIM3 overlap)
- [x] **DA3 (Depth Anything 3) Streaming** — primary reconstruction backend with SLAM + loop closure
- [x] **Hybrid reconstruction (LiDAR + DA3)** — fuses Stray Scanner LiDAR depth with DA3 neural depth
- [x] **LiDAR-only reconstruction** — DA3 SLAM backbone with raw LiDAR depth injection (no neural inference)
- [x] **Stray Scanner integration** — auto-detection of iOS LiDAR data (ARKit poses, depth, intrinsics)
- [x] **LiDAR complement generation** — backprojects raw LiDAR using post-loop-closure poses, merged by CloudCompPy
- [x] **Available backends API** — `GET /api/sessions/{id}/available_backends` with auto-detection
- [x] Multi-backend reconstruction dispatcher (`map_worker.py`)
- [x] Frame quality filtering (Laplacian blur detection)
- [x] Visual novelty frame selector (H/F ratio, ORB-SLAM inspired)
- [x] Alignment manager (SIM3 + RANSAC auto-leveling)
- [x] SAM3 instance segmentation (video mode + DBSCAN clustering)
- [x] InternVL3 scene analysis (VLM object inventory)
- [x] RANSAC face detection per segment + non-destructive point assignment
- [x] Voxel mesh visualization with normal-snapped quads
- [x] Segment visibility toggles (classification attribute in Potree)
- [x] BIM comparison (IFC parsing, C2M deviation, sábana)
- [x] Coverage analysis (mesh sampling + KDTree proximity)
- [x] Potree octree streaming (custom LOD loader)
- [x] CloudCompPy post-processing (SOR filter + LiDAR complement merge)
- [x] Three.js BIM overlay (IFC → mesh rendering)
- [x] React/TypeScript frontend (IDE-style viewer)
- [x] JWT authentication + role-based access
- [x] Team/session management
- [x] WebSocket real-time pipeline progress
- [x] Configurable pipeline stages (drag-and-drop ordering)

### 🔧 Tier 0.5: Hybrid Reconstruction Industrialization (In Progress)

- [ ] **UI backend selector** — dropdown in Pipeline Dialog to choose reconstruction backend (calls `available_backends` endpoint)
- [ ] **WebSocket backend override** — pass selected backend from UI to server, override `config.yaml` at runtime
- [ ] Parameterize DA3 resolution in `StrayLiDAROnly` (currently hardcoded 378×504)
- [ ] Replace fragile relative path in `_generate_lidar_complement` with direct session path

---

## ✅ Tier 1: Coverage Engine (Done)

**Why first:** Foundation for accurate progress tracking. Without cumulative coverage and occlusion handling, all downstream features report incorrect data.

**Depends on:** Tier 0

- [x] **1.1 Coverage Store** — Per-element cumulative coverage, NPZ persistence, merge logic
- [x] **1.2 Occlusion Ray-Caster** — Camera→BIM raycasting, classification (COVERED | OCCLUDED | NOT_BUILT | NOT_VISIBLE)
- [x] **1.3 SAM3 Occluder Identification** — Map occluding points to segment labels
- [x] **1.4 VLM Occlusion Classifier** — 3-tier heuristic→cache→VLM, permanent vs temporary classification
- [x] **1.5 Element State Machine** — NOT_STARTED → IN_PROGRESS → COMPLETED → VERIFIED
- [x] **1.6 Pipeline Integration** — Config, UI badges, cumulative coverage in BIMAnalysisPanel

---

## 🔬 Tier 2: 2D Analysis Engine

**Why now:** This is the architectural pivot. Moving the primary analysis from 3D point clouds to 2D image space dramatically improves precision, leverages SOTA vision models, and eliminates the error chain of 3D reconstruction.

**Depends on:** Tier 0 (MapAnything poses) + Tier 1 (coverage tracking)

### Rationale

The 2D image space preserves full sensor resolution (~1MP+ per frame) while 3D reconstruction loses information at every step (depth→backprojection→merging→SOR→octree). All state-of-the-art vision models (PE Spatial, SAM, VLM) operate natively in 2D. Camera poses from MapAnything enable direct BIM-to-2D projection for pixel-level comparison.

> **Full pipeline specification:** [ADR-002: 2D Comparison Pipeline](./002-2d-comparison-pipeline.md)

### Pipeline Phases → Implementation Items

| Phase | Description | Items |
|-------|-------------|-------|
| Phase 1: Alignment | MapAnything poses + gizmo registration | ✅ Existing |
| Phase 2: BIM Render | Render IFC from camera poses → RGB + depth + element ID + edges | 2.1 |
| Phase 3A: Geometric Deviation | PE Spatial feature comparison (real vs BIM render) | 2.2, 2.3 |
| Phase 3B: Metric Depth | DepthLM depth vs BIM z-buffer → Δdepth in meters | 2.8 |
| Phase 3C: Material & State | PLM-8B classification per visible element | 2.4 |
| Phase 4: Aggregation | Multi-frame weighted results per element | 2.5 |
| Phase 5: Visualization | Sábana 2.0, per-frame overlay, element gallery | 2.5 |

### AI Model Stack

| Model | Role | Parameters | Conda Env |
|-------|------|------------|----------|
| **PE-Spatial-G14-448** | Dense spatial backbone — segmentation, detection, depth features | ~1.8B (ViT-G/14) | `pe_spatial` |
| **PLM-8B** | VLM for scene analysis, material ID, occlusion classification | 8B | `pe_spatial` |
| **DepthLM Pixtral 12B** | Metric depth estimation per-pixel | 12B | `depthlm` |

> **Note:** InternVL3 remains active in the current pipeline (`scene_analyzer.py`) until PLM-8B is fully integrated and validated. Migration happens at step 2.7.

### Implementation

- [ ] **2.1 BIM Reprojection Engine**
  - [ ] Load BIM mesh geometry (vertices, edges, faces per element)
  - [ ] For each frame: project BIM elements using `K × [R|t]_BIM × P_BIM`
  - [ ] Pose transformation: compose `T_scan→BIM` (from gizmo+ICP) with MapAnything cam2world
  - [ ] Occlusion-aware rendering: only project visible BIM faces (z-buffer)
  - [ ] Output: per-frame BIM overlay masks with element IDs

- [ ] **2.2 PE Spatial Integration**
  - [ ] Integrate PE-Spatial-G14-448 as dense spatial feature backbone
  - [ ] Extract per-pixel spatial features from each frame
  - [ ] Feature matching between BIM-projected regions and actual image
  - [ ] Pose uncertainty compensation: expand matching region proportional to estimated error
  - [ ] Env: `pe_spatial` conda env, GPU preferred (~3.6GB bf16), CPU fallback

- [ ] **2.3 2D Deviation Detection**
  - [ ] Compare projected BIM edges vs detected edges in image (PE Spatial features)
  - [ ] Compute per-pixel deviation between expected BIM surface and observed surface
  - [ ] Classify per-element: GOOD / REGULAR / BAD based on pixel-level analysis
  - [ ] Aggregate multi-frame results per element (weighted by viewing angle and distance)

- [ ] **2.4 Material Identification (PLM-8B)**
  - [ ] PLM-8B (Perception Language Model, 8B params) analyzes image regions corresponding to BIM elements
  - [ ] Material classification: concrete, steel, masonry, glass, etc.
  - [ ] Cross-reference with BIM material specifications
  - [ ] Flag material mismatches
  - [ ] Env: `pe_spatial`, CPU-only on dev (16GB bf16), GPU on cloud

- [ ] **2.5 Multi-Frame Coverage in 2D**
  - [ ] Track which BIM elements are visible in each frame (from reprojection)
  - [ ] Accumulate coverage across frames (multi-view coverage)
  - [ ] Compute viewing quality per element (angle, distance, occlusion)
  - [ ] Update coverage store with 2D analysis results

- [ ] **2.6 Camera Pose Localization**
  - [ ] Initial: use existing gizmo+ICP `T_scan→BIM` alignment
  - [ ] Future: ARKit/ARCore absolute positioning (Unity capture app)
  - [ ] Pose error estimation per frame (for matching window expansion)
  - [ ] Dual-source fusion (ARKit + MapAnything) when both available

- [ ] **2.7 Migrate InternVL3 → PLM-8B**
  - [ ] Validate PLM-8B scene analysis quality against InternVL3 baseline
  - [ ] Once validated, replace InternVL3 in `scene_analyzer.py` and `vlm_worker.py`
  - [ ] Migrate occlusion classification from InternVL3 to PLM-8B
  - [ ] Remove InternVL3 vendor deps and conda env

- [ ] **2.8 DepthLM Metric Depth**
  - [ ] Integrate DepthLM (Pixtral 12B) for per-pixel metric depth estimation
  - [ ] Per-frame depth maps complement MapAnything geometry
  - [ ] Use metric depth for BIM deviation verification (independent of 3D reconstruction)
  - [ ] Env: `depthlm` conda env, CPU-only on dev (24GB bf16), GPU on cloud

---

## 🔭 Tier 3: Long-Range & Multi-Source Scanning

**Why third:** Extends the reach and scale of scanning. Tiers 1-2 give us accurate analysis; Tier 3 gives us more area to cover.

**Depends on:** Tier 0 (can be started in parallel with Tier 2)

- [x] **3.1 Zoom Detection** → REPLACED by zoom-lock capture
  - [x] Zoom is per-session metadata in `scan_meta.json`
  - [x] Manual zoom_level config for existing captures

- [x] **3.2 MapAnything Zoom Intrinsics Correction**
  - [x] Session-level: `f_corrected = f_mapanything × zoom_level`
  - [x] Applied uniformly in MapAnything ChunkWrapper

- [ ] **3.3 Multi-Source Foundations**
  - [ ] Project data model: Project → N Scans
  - [ ] Scan metadata: operator, zone, timestamp, source_type, zoom_level
  - [ ] Independent processing per scan (existing pipeline)
  - [ ] BIM alignment as common reference frame
  - [ ] Cloud merge with deduplication
  - [ ] Coverage store accumulates across all scans

---

## 📄 Tier 4: Document Intelligence

**Why fourth:** Independent from scanning improvements — can be developed in parallel. Provides the contractual foundation that Tiers 5+ need.

**Depends on:** Tier 0 only (WebSocket, auth, UI infrastructure)

- [ ] **4.1 Document Storage & Indexing** — File upload (PDF, DOCX, XLSX), hierarchical folders, full-text indexing
- [ ] **4.2 RAG Engine** — Document parsing, chunking, embeddings, vector DB, LLM chat with source citations
- [ ] **4.3 Contract Chat Interface** — Chat UI, API endpoint, conversation history
- [ ] **4.4 Contradiction Detection** — Cross-document analysis, amendment vs base spec, inter-discipline conflicts
- [ ] **4.5 Requirements Extraction** — LLM-assisted extraction from tender docs
- [ ] **4.6 Requirements Matrix** — Status tracking, BIM element linking, compliance dashboard

---

## 📊 Tier 5: BIM 5D — Schedule & Certification

**Why fifth:** Needs Tier 1 (accurate coverage/progress) + Tier 4 (contract requirements) to produce meaningful outputs.

**Depends on:** Tier 1 (coverage engine) + Tier 4 (requirements)

- [ ] **5.1 Schedule Data Model** — WBS import, activity→BIM linking, planned dates
- [ ] **5.2 Gantt Visualization** — Planned vs actual, critical path, scan-driven progress
- [ ] **5.3 Scan-Fed Progress** — `progress = f(coverage × quality)`, delay detection
- [ ] **5.4 S-Curve** — Planned vs actual, comparison chart
- [ ] **5.5 Certification Engine** — Certificate with scan evidence, threshold enforcement, PDF generation, digital signatures

---

## 📐 Tier 6: Engineering Document Control

**Depends on:** Tier 4 (document storage, requirements matrix)

- [ ] **6.1 Drawing Management** — Upload, version control, status workflow, observations
- [ ] **6.2 Requirement-Drawing Linking** — Gap detection, AI-assisted compliance
- [ ] **6.3 BIM Consistency** — Drawing↔BIM discrepancy detection
- [ ] **6.4 Pending Items Tracker** — Observations register, status workflow, dashboard

---

## 🛡️ Tier 7: Quality, Safety & Environment

**Depends on:** Tier 1 (deviations) + Tier 4 (requirements) + Tier 6 (doc control)

- [ ] **7.1 Quality Management** — Inspection checklists, auto-NCR from deviations, quality KPIs
- [ ] **7.2 Worker Safety & PPE Detection** — SAM3 body segmentation (anonymized), VLM PPE verification, hazard detection, GDPR compliant
- [ ] **7.3 Environmental Management** — Requirements tracking, compliance reporting
- [ ] **7.4 RAMS** — Requirements matrix, verification linked to commissioning (metro, rail, energy)

---

## 💬 Tier 8: Communication & Meeting Governance

**Depends on:** Tier 0 (WebSocket infrastructure)

- [ ] **8.1 Project Chat** — Real-time messaging, contextual threads, AI assistant (RAG)
- [ ] **8.2 Meeting Management** — Audio recording + Whisper transcription, LLM auto-minutes, digital signatures
- [ ] **8.3 Agreement Tracking** — Status tracking, AI fulfillment checks, legal compliance

---

## 🔗 Tier 9: Blockchain Audit Trail

**Depends on:** Tier 5 (certification) + Tier 6 (documents)

- [ ] **9.1 Event Hashing** — SHA-256 of every critical event
- [ ] **9.2 Immutable Ledger** — Merkle tree + TSA, chain integrity, digital signatures
- [ ] **9.3 Verification API** — Public verification endpoint for auditors
- [ ] **9.4 Regulatory Standards Engine** — Standards library, auto-linking, compliance dashboard

---

## 🔧 Tier 10: STAC Maintain — Asset Lifecycle

**Depends on:** Tier 5 (certification) + Tier 7 (QSE) + Tier 9 (blockchain)

- [ ] **10.1 Digital Twin Handover** — Operations package, asset register, baseline snapshot
- [ ] **10.2 Preventive Maintenance** — Auto scheduling, BIM-aware, compliance tracking
- [ ] **10.3 Corrective Maintenance & Work Orders** — Full context, parts linking, repair history
- [ ] **10.4 Deterioration Monitoring** — Periodic re-scans, baseline comparison, trend analysis
- [ ] **10.5 Technician Traceability** — Profiles, tools, materials, blockchain recording
- [ ] **10.6 Warranty Management** — Dates, alerts, claim support

---

## 📱 Tier 11: STAC Build Capture (Unity)

**Depends on:** Tier 3 (multi-source stabilized) + Tier 2 (2D engine needs pose data)

- [ ] **11.1 Unity Camera App** — AR Foundation, scanning guidance, controlled capture
- [ ] **11.2 Metadata Injection** — Per-frame intrinsics, IMU data, GPS/BLE, session metadata
- [ ] **11.3 ARKit/ARCore Pose Stream** — Dual pose source: ARKit VIO + MapAnything NN
- [ ] **11.4 Source Verification** — Cryptographic signing, chain of custody

---

## Dependency Graph

```
Tier 0 (DONE)
  │
  ├──→ Tier 1: Coverage Engine (DONE)
  │       │
  │       ├──→ Tier 2: 2D Analysis Engine ← CURRENT PRIORITY
  │       │       │
  │       │       └──→ Tier 11: Unity Capture (dual pose source)
  │       │
  │       ├──→ Tier 5: BIM 5D (Gantt, S-Curve, Certification)
  │       │       │
  │       │       ├──→ Tier 9: Blockchain Audit Trail
  │       │       │
  │       │       └──→ Tier 10: STAC Maintain (lifecycle)
  │       │
  │       └──→ Tier 7: QSE + RAMS + Worker Safety
  │               │
  │               └──→ Tier 10: STAC Maintain
  │
  ├──→ Tier 3: Long-Range + Multi-Source (parallel with Tier 2)
  │       │
  │       └──→ Tier 11: STAC Build Capture (Unity)
  │
  ├──→ Tier 4: Document Intelligence (parallel with Tier 2)
  │       │
  │       ├──→ Tier 5: BIM 5D
  │       ├──→ Tier 6: Engineering Doc Control
  │       │       │
  │       │       └──→ Tier 7: QSE + RAMS
  │       │
  │       └──→ Tier 9: Blockchain
  │
  └──→ Tier 8: Communication (parallel, richer with 4-7)
```

---

## Quick Reference: What Unlocks What

| When this is done... | ...these become possible |
|---------------------|------------------------|
| **Tier 1** (Coverage) | Accurate progress %, element states, scan-fed Gantt |
| **Tier 2** (2D Analysis) | **Pixel-level deviation detection, material ID, precise BIM comparison** |
| **Tier 3** (Long-range) | Large project scanning, multi-operator workflows |
| **Tier 4** (Documents) | Contract chat, requirements tracking, contradiction detection |
| **Tier 2 + 4** | 2D-verified certification backed by both pixel evidence AND contract requirements |
| **Tier 5** (BIM 5D) | S-curve, payment certificates, delay analysis |
| **Tier 6** (Eng. Docs) | Drawing-to-BIM linking, version control, compliance |
| **Tier 7** (QSE) | Auto-NCRs from deviations, **worker safety detection (PPE, hazards)** |
| **Tier 8** (Comms) | Meeting minutes, agreement tracking, project chat |
| **Tier 9** (Blockchain) | Immutable audit trail, legal-grade evidence |
| **Tier 10** (Maintain) | **Lifecycle maintenance, deterioration monitoring, recurring revenue** |
| **Tier 11** (Capture) | **Dual pose sources (ARKit+NN), standardized metadata, guided scanning** |
