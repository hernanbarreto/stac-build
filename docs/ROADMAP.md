# STAC Build — Development Roadmap

> Dependency-ordered, no deadlines — just logical sequence.  
> Check items off as they are completed.

---

## ✅ Tier 0: Core (Done)

Everything here is implemented and working.

- [x] DA3 dense 3D reconstruction (chunked, SIM3 overlap)
- [x] Frame quality filtering (Laplacian blur detection)
- [x] Visual novelty frame selector (H/F ratio, ORB-SLAM inspired)
- [x] Alignment manager (SIM3 + RANSAC auto-leveling)
- [x] SAM3 instance segmentation (video mode + DBSCAN clustering)
- [x] InternVL3 scene analysis (VLM object inventory)
- [x] BIM comparison (IFC parsing, C2M deviation, sábana)
- [x] Coverage analysis (mesh sampling + KDTree proximity)
- [x] Potree octree streaming (custom LOD loader)
- [x] CloudCompPy post-processing (SOR filter)
- [x] Three.js BIM overlay (IFC → mesh rendering)
- [x] React/TypeScript frontend (IDE-style viewer)
- [x] JWT authentication + role-based access
- [x] Team/session management
- [x] WebSocket real-time pipeline progress
- [x] Configurable pipeline stages (drag-and-drop ordering)

---

## 🔨 Tier 1: Coverage Engine

**Why first:** This is the foundation for accurate progress tracking. Without cumulative coverage and occlusion handling, all downstream features (certification, S-curve, progress) would report incorrect data.

**Depends on:** Tier 0 (done)

- [ ] **1.1 Coverage Store** (`coverage_store.py`)
  - [ ] Per-element cumulative coverage data model
  - [ ] Surface sample persistence (coverage_history.npz)
  - [ ] Merge logic: new scan + historical → union (coverage only increases)
  - [ ] Coverage timeline tracking (scan-by-scan snapshots)

- [ ] **1.2 Occlusion Ray-Caster** (`occlusion_raycaster.py`)
  - [ ] Extract camera positions from DA3 extrinsics
  - [ ] Cylindrical ray query (scan_tree along camera→BIM ray)
  - [ ] Classify BIM surface samples: COVERED | OCCLUDED | NOT_BUILT | NOT_VISIBLE
  - [ ] Integrate with existing `compute_coverage_pct()` flow

- [ ] **1.3 SAM3 Occluder Identification**
  - [ ] Map occluding points to SAM3 segment labels
  - [ ] Report which segment blocks which BIM element

- [ ] **1.4 VLM Occlusion Classifier**
  - [ ] Extend `scene_analyzer.py` with occlusion classification prompt
  - [ ] Classify occluders: permanent (furniture, MEP) vs temporary (debris, scaffold)
  - [ ] Store classification per occlusion event

- [ ] **1.5 Element State Machine**
  - [ ] State model: NOT_STARTED → IN_PROGRESS → COMPLETED → VERIFIED
  - [ ] Automatic transitions based on coverage + quality thresholds
  - [ ] OCCLUDED_PERMANENT: freeze coverage at last known value
  - [ ] OCCLUDED_TEMPORARY: flag for re-scan

- [ ] **1.6 Pipeline Integration**
  - [ ] Update `run_comparison()` in `bim_comparison.py` to use coverage engine
  - [ ] Update `save_sabana()` to include occlusion data in metadata
  - [ ] Update UI to show element state + occlusion indicators

---

## 🔭 Tier 2: Long-Range & Multi-Source Scanning

**Why second:** Extends the reach and scale of scanning. Tier 1 gives us accurate coverage tracking; Tier 2 gives us more area to cover.

**Depends on:** Tier 0 (can be started in parallel with Tier 1)

- [ ] **2.1 Zoom Detection** (`zoom_detector.py`)
  - [ ] EXIF focal length extraction
  - [ ] Feature density analysis (ORB count per frame)
  - [ ] FOV estimation from feature distribution
  - [ ] Output: zoom_level per frame

- [ ] **2.2 Sequence Splitter**
  - [ ] Segment video frames by zoom band (1x, 3x, 5x+)
  - [ ] Handle zoom transitions (ramp detection)
  - [ ] Generate sub-session manifests per segment

- [ ] **2.3 Adaptive Frame Quality**
  - [ ] Scale blur threshold with zoom: `thresh *= (1 + zoom/5)`
  - [ ] Stricter motion blur rejection for high-zoom frames
  - [ ] Minimum sharpness requirements per zoom band

- [ ] **2.4 DA3 Intrinsics Injection**
  - [ ] Compute zoom-adjusted intrinsics: `f_zoom = f_base × zoom_factor`
  - [ ] Pass intrinsics matrix to `model.inference(images, intrinsics=K)`
  - [ ] Update `da3_native_wrapper.py` to support per-segment intrinsics

- [ ] **2.5 Segment Registration**
  - [ ] ICP alignment of detail segment clouds to context cloud
  - [ ] Feature matching for coarse alignment
  - [ ] Merge registered segments into unified cloud
  - [ ] Handle scale differences between zoom levels

- [ ] **2.6 Multi-Source Foundations**
  - [ ] Project data model: Project → N Scans
  - [ ] Scan metadata: operator, zone, timestamp, source_type
  - [ ] Independent processing per scan (existing pipeline)
  - [ ] BIM alignment as common reference frame
  - [ ] Cloud merge with deduplication
  - [ ] Coverage store accumulates across all scans

---

## 📄 Tier 3: Document Intelligence

**Why third:** Independent from scanning improvements — can be developed in parallel. Provides the contractual foundation that Tiers 4+ need.

**Depends on:** Tier 0 only (WebSocket, auth, UI infrastructure)

- [ ] **3.1 Document Storage & Indexing**
  - [ ] File upload API (PDF, DOCX, XLSX)
  - [ ] Hierarchical folder structure (contract → discipline → section)
  - [ ] Document metadata extraction (title, date, pages)
  - [ ] Full-text indexing

- [ ] **3.2 RAG Engine**
  - [ ] Document parsing (PyMuPDF / Unstructured.io)
  - [ ] Text chunking with overlap
  - [ ] Embedding model integration (sentence-transformers or similar)
  - [ ] Vector database (ChromaDB or Qdrant)
  - [ ] LLM integration for chat responses
  - [ ] Source citation in answers (page, section, document)

- [ ] **3.3 Contract Chat Interface**
  - [ ] Chat UI component in frontend
  - [ ] API endpoint for RAG queries
  - [ ] Conversation history per user
  - [ ] Context-aware: can reference BIM elements, drawings, requirements

- [ ] **3.4 Contradiction Detection**
  - [ ] Cross-document analysis via LLM
  - [ ] Amendment vs base specification comparison
  - [ ] Inter-discipline conflict detection
  - [ ] Flagged contradictions dashboard

- [ ] **3.5 Requirements Extraction**
  - [ ] LLM-assisted extraction of requirements from tender docs
  - [ ] Structured output: requirement ID, text, discipline, type
  - [ ] Human review/approval workflow

- [ ] **3.6 Requirements Matrix**
  - [ ] Requirements database per discipline
  - [ ] Status tracking: NOT_STARTED | IN_PROGRESS | COMPLIANT | NON_COMPLIANT | WAIVED
  - [ ] Link requirements to BIM elements
  - [ ] Compliance dashboard (% per discipline, per section)
  - [ ] Export to spreadsheet

---

## 📊 Tier 4: BIM 5D — Schedule & Certification

**Why fourth:** Needs Tier 1 (accurate coverage/progress) + Tier 3 (contract requirements) to produce meaningful outputs.

**Depends on:** Tier 1 (coverage engine) + Tier 3 (requirements)

- [ ] **4.1 Schedule Data Model**
  - [ ] WBS (Work Breakdown Structure) import/creation
  - [ ] Activity → BIM element linking
  - [ ] Planned dates per activity
  - [ ] Predecessor/successor relationships

- [ ] **4.2 Gantt Visualization**
  - [ ] Gantt chart component (DHTMLX or custom)
  - [ ] Planned vs actual bars
  - [ ] Critical path highlighting
  - [ ] Scan-driven progress auto-update

- [ ] **4.3 Scan-Fed Progress**
  - [ ] Activity progress = f(coverage_cumulative × quality)
  - [ ] Auto-update when new scan is processed
  - [ ] Delay detection: actual behind planned
  - [ ] Impact analysis on downstream activities

- [ ] **4.4 S-Curve**
  - [ ] Planned S-curve from schedule
  - [ ] Actual S-curve from scan history
  - [ ] Comparison chart (D3.js / Chart.js)
  - [ ] Early/late analysis

- [ ] **4.5 Certification Engine**
  - [ ] Certificate template with scan evidence
  - [ ] Threshold: no certificate below X% coverage + Y% quality
  - [ ] Certificate includes: scan date, coverage, deviation report, screenshots
  - [ ] PDF generation
  - [ ] Digital signature workflow
  - [ ] Payment milestone linking

---

## 📐 Tier 5: Engineering Document Control

**Why fifth:** Needs Tier 3 (document infrastructure + requirements) to link drawings to requirements.

**Depends on:** Tier 3 (document storage, requirements matrix)

- [ ] **5.1 Drawing Management**
  - [ ] Upload engineering drawings (PDF, DWG conversion)
  - [ ] Version control with full history
  - [ ] Status workflow: DRAFT → REVIEW → APPROVED → SUPERSEDED
  - [ ] Observations and comments per drawing

- [ ] **5.2 Requirement-Drawing Linking**
  - [ ] Associate drawings to contract requirements
  - [ ] Gap detection: requirements without drawings
  - [ ] AI-assisted compliance check

- [ ] **5.3 BIM Consistency**
  - [ ] Detect drawing ↔ BIM discrepancies
  - [ ] Flag BIM updates needed when drawings change
  - [ ] Drawing-to-BIM element association

- [ ] **5.4 Pending Items Tracker**
  - [ ] Observations register per drawing/element
  - [ ] Status: OPEN → ADDRESSED → VERIFIED → CLOSED
  - [ ] Dashboard: pending items by discipline, age, priority

---

## 🛡️ Tier 6: Quality, Safety & Environment

**Why sixth:** Needs Tier 1 (deviations trigger NCRs) + Tier 3 (quality requirements from contract).

**Depends on:** Tier 1 (deviations) + Tier 3 (requirements) + Tier 5 (doc control)

- [ ] **6.1 Quality Management**
  - [ ] Inspection checklists per activity/discipline
  - [ ] Non-conformance reports (NCR) auto-generated from deviations
  - [ ] NCR workflow: OPEN → REVIEW → CORRECTIVE → VERIFIED → CLOSED
  - [ ] Quality KPIs (first-pass yield, NCR rate, rework %)

- [ ] **6.2 Worker Safety & PPE Detection**
  - [ ] SAM3 body segmentation (detect human silhouettes, NOT identities)
  - [ ] Face anonymization pipeline (auto-blur before storage)
  - [ ] VLM PPE verification prompts (hard hat, vest, harness, gloves, glasses)
  - [ ] Work-at-height detection (DA3 elevation + VLM harness check)
  - [ ] Restricted zone violation (3D worker position vs BIM hazard zones)
  - [ ] Improper tool usage detection (VLM context analysis)
  - [ ] Safety NCR auto-generation from detected violations
  - [ ] Safety KPIs dashboard (PPE compliance rate, violations trend)
  - [ ] GDPR compliance: no biometric data, no individual tracking
  - [ ] Permit management (work permits, hot work, confined space)

- [ ] **6.3 Environmental Management**
  - [ ] Environmental requirements tracking
  - [ ] Compliance reporting
  - [ ] Waste/emissions monitoring

- [ ] **6.4 RAMS (where applicable)**
  - [ ] RAMS requirements matrix
  - [ ] Verification linked to test/commissioning results
  - [ ] Applicable for metro, rail, energy projects

---

## 💬 Tier 7: Communication & Meeting Governance

**Why seventh:** Adds collaboration layer. Can be started earlier if needed but full value comes with Tiers 3-6 in place.

**Depends on:** Tier 0 (WebSocket infrastructure exists)

- [ ] **7.1 Project Chat**
  - [ ] Real-time messaging (WebSocket-based)
  - [ ] Channels per team/discipline
  - [ ] Contextual threads (about a BIM element, drawing, NCR)
  - [ ] AI assistant (RAG over all project data)

- [ ] **7.2 Meeting Management**
  - [ ] Meeting scheduling / calendar
  - [ ] Audio recording + Whisper transcription
  - [ ] LLM auto-generated minutes (key points, decisions, action items)
  - [ ] Digital signature of minutes (on-site, tablet/phone)

- [ ] **7.3 Agreement Tracking**
  - [ ] All meeting agreements become tracked items
  - [ ] Status: AGREED → IN_PROGRESS → COMPLETED → VERIFIED
  - [ ] AI checks if agreements are being fulfilled
  - [ ] Legal compliance verification (notice periods, quorum)

---

## 🔗 Tier 8: Blockchain Audit Trail

**Why last:** The seal of trust. All other tiers produce the events that get recorded. Blockchain is the immutability layer on top.

**Depends on:** Tier 4 (certification) + Tier 5 (documents) — needs events to record

- [ ] **8.1 Event Hashing**
  - [ ] SHA-256 hash of every critical event (scan, certificate, amendment, NCR, meeting)
  - [ ] Structured event format: { type, hash, timestamp, signer, references }

- [ ] **8.2 Immutable Ledger**
  - [ ] MVP: Merkle tree with TSA (Trusted Timestamp Authority)
  - [ ] Each block references previous → chain integrity
  - [ ] Digital signature of responsible party
  - [ ] Future: upgrade to Hyperledger or public chain

- [ ] **8.3 Verification API**
  - [ ] "Was this certificate issued on this date?" → verify hash
  - [ ] "Has this scan data been modified since?" → compare hashes
  - [ ] "Did this person approve this drawing?" → verify signature
  - [ ] Public verification endpoint for auditors

- [ ] **8.4 Regulatory Standards Engine**
  - [ ] Standards library (upload + index norms)
  - [ ] Auto-linking: standard clauses → requirements → BIM elements
  - [ ] Compliance dashboard per standard

---

## 🔧 Tier 9: STAC Maintain — Asset Lifecycle

**Why ninth:** The crown jewel. After everything is built, verified, documented, and sealed with blockchain — STAC transitions to maintenance mode for the life of the asset.

**Depends on:** Tier 4 (certification) + Tier 6 (QSE) + Tier 8 (blockchain)

- [ ] **9.1 Digital Twin Handover**
  - [ ] Auto-generate operations package at project completion
  - [ ] Asset register from SAM3 segmentation + BIM elements
  - [ ] Baseline condition snapshot (day-zero 3D reference)
  - [ ] Documentation package: specs, drawings, certificates, warranties
  - [ ] Project mode → Maintenance mode transition in UI

- [ ] **9.2 Preventive Maintenance**
  - [ ] Maintenance plan auto-generation from installed equipment
  - [ ] BIM-aware scheduling (MEP systems, intervals, calibrations)
  - [ ] Calendar + alerts for upcoming tasks
  - [ ] Compliance tracking: overdue, completed, skipped

- [ ] **9.3 Corrective Maintenance & Work Orders**
  - [ ] Work order creation, assignment, tracking, closure
  - [ ] Full context: drawings, specs, construction method, materials
  - [ ] Parts/materials linking (installed → replacement needed)
  - [ ] Repair history per element

- [ ] **9.4 Deterioration Monitoring (Re-Scanning)**
  - [ ] Periodic maintenance scans (same DA3 pipeline)
  - [ ] Compare current state vs as-built baseline
  - [ ] Crack/deformation progression tracking
  - [ ] Structural health trend analysis
  - [ ] Early warning alerts for degradation thresholds

- [ ] **9.5 Technician Traceability**
  - [ ] Technician profiles: ID, company, licenses, certifications
  - [ ] Tool/equipment tracking with calibration dates
  - [ ] Materials used: batch numbers, specs, warranties
  - [ ] All interventions recorded in blockchain

- [ ] **9.6 Warranty Management**
  - [ ] Warranty dates and conditions per element
  - [ ] Expiration alerts
  - [ ] Claim support: full construction + maintenance history as evidence

---

## Dependency Graph

```
Tier 0 (DONE)
  │
  ├──→ Tier 1: Coverage Engine
  │       │
  │       ├──→ Tier 4: BIM 5D (Gantt, S-Curve, Certification)
  │       │       │
  │       │       ├──→ Tier 8: Blockchain Audit Trail
  │       │       │
  │       │       └──→ Tier 9: STAC Maintain (lifecycle + maintenance)
  │       │
  │       └──→ Tier 6: QSE + RAMS + Worker Safety
  │               │
  │               └──→ Tier 9: STAC Maintain
  │
  ├──→ Tier 2: Long-Range + Multi-Source  (parallel with Tier 1)
  │
  ├──→ Tier 3: Document Intelligence      (parallel with Tier 1)
  │       │
  │       ├──→ Tier 4: BIM 5D
  │       ├──→ Tier 5: Engineering Doc Control
  │       │       │
  │       │       └──→ Tier 6: QSE + RAMS
  │       │
  │       └──→ Tier 8: Blockchain
  │
  └──→ Tier 7: Communication              (parallel, but richer with 3-6)
```

---

## Quick Reference: What Unlocks What

| When this is done... | ...these become possible |
|---------------------|------------------------|
| **Tier 1** (Coverage) | Accurate progress %, element states, scan-fed Gantt |
| **Tier 2** (Long-range) | Large project scanning, multi-operator workflows |
| **Tier 3** (Documents) | Contract chat, requirements tracking, contradiction detection |
| **Tier 1 + 3** | Certification backed by both scan evidence AND contract requirements |
| **Tier 4** (BIM 5D) | S-curve, payment certificates, delay analysis |
| **Tier 5** (Eng. Docs) | Drawing-to-BIM linking, version control, compliance |
| **Tier 6** (QSE) | Auto-NCRs from deviations, **worker safety detection (PPE, hazards)** |
| **Tier 7** (Comms) | Meeting minutes, agreement tracking, project chat |
| **Tier 8** (Blockchain) | Immutable audit trail, legal-grade evidence, public transparency |
| **Tier 9** (Maintain) | **Lifecycle maintenance, deterioration monitoring, recurring revenue** |
