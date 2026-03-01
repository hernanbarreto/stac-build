# STAC Build — Future Vision
## From 3D Scanning to Integrated Construction Management Platform

> **Author:** Hernán Barreto — Ingerop IN3  
> **Date:** 2026-03-01  
> **Status:** Strategic roadmap — living document

---

## The Vision

STAC evolves from a dimensional control tool into a **complete, AI-powered construction project management platform** that unifies:

- Physical verification (3D scanning, BIM comparison)
- Contract and tender documentation
- Project scheduling and cost control (BIM 5D)
- Quality, safety, and environmental management
- Engineering document control
- Team communication and meeting governance

The core principle: **every claim, certificate, and payment is backed by verifiable, AI-audited physical evidence.** This creates an unprecedented level of transparency for public works, where every dollar spent can be traced to demonstrated construction progress and quality compliance.

---

## Module Map

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAC BUILD PLATFORM                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  CORE (Current)                                                        │
│  ├─ 3D Reconstruction (DA3)                                            │
│  ├─ Semantic Segmentation (SAM3)                                       │
│  ├─ Scene Analysis (InternVL3 VLM)                                     │
│  ├─ BIM Comparison (IFC → sábana)                                      │
│  └─ Potree Visualization                                               │
│                                                                        │
│  PLANNED (Near-term)                                                   │
│  ├─ Zoom/Distance Scanning                                             │
│  ├─ Occlusion-Aware Coverage Engine                                    │
│  ├─ Multi-Source Scanning                                               │
│  └─ Multi-Level Support                                                │
│                                                                        │
│  FUTURE MODULES                                                        │
│  ├─ 📄 Contract & Tender Documentation                                 │
│  ├─ 📊 BIM 5D: Schedule + Cost (Gantt, S-Curve, Certification)         │
│  ├─ 📐 Engineering Document Control                                    │
│  ├─ ✅ Quality, Safety & Environment (QSE + RAMS)                      │
│  ├─ 💬 Communication & Meeting Governance                              │
│  └─ 📏 Regulatory & Standards Engine                                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Contract & Tender Documentation Module

### Purpose

Centralize all contractual documentation and make it **queryable via AI chat**, turning thousands of pages of tender specifications into an instant-access knowledge base.

### Capabilities

- **Document ingestion**: Upload tender packages (PDFs, DOCX, spreadsheets) organized by discipline and section
- **Structured storage**: Tree-based document hierarchy
  ```
  Contract/
  ├─ Tender Specifications/
  │   ├─ Technical/
  │   │   ├─ Civil/
  │   │   ├─ Structural/
  │   │   ├─ MEP/
  │   │   ├─ Electrical/
  │   │   └─ Systems/
  │   └─ General/
  ├─ Contractual/
  │   ├─ Terms & Conditions/
  │   ├─ Payment Schedule/
  │   └─ Penalties & Bonuses/
  ├─ Amendments/
  │   ├─ Amendment 001 (2026-03-15)/
  │   ├─ Amendment 002 (2026-06-01)/
  │   └─ ...
  └─ Annexes/
  ```

- **AI chat interface (RAG)**: Ask questions in natural language, get answers with source citations
  - *"What is the specified concrete strength for foundation elements?"*
  - *"What are the penalty clauses for delays in Milestone 3?"*
  - *"What tolerance does the spec require for facade alignment?"*

- **Contradiction detection**: AI scans across all documents and amendments to flag:
  - Conflicting requirements between disciplines
  - Amendments that contradict base specifications
  - Ambiguous or missing requirements

- **Amendment tracking**: Full version history of contractual changes, with diff view showing what changed

### Requirements Matrix

- **Auto-generated requirements matrix** extracted from tender documents per discipline
- Each requirement tracked with status: `NOT_STARTED | IN_PROGRESS | COMPLIANT | NON_COMPLIANT | WAIVED`
- Linked to BIM elements where applicable (e.g., "concrete cover 40mm" → linked to all IfcColumn elements)
- Dashboard view: compliance percentage per discipline, per section

### Technology

- **RAG (Retrieval-Augmented Generation)**: Document embeddings + vector database (e.g., ChromaDB, Qdrant) + LLM for chat
- **Document parsing**: Apache Tika, PyMuPDF, or Unstructured.io for PDF/DOCX extraction
- **Contradiction engine**: Cross-reference analysis via LLM with structured prompts

---

## 2. BIM 5D: Schedule + Cost Control

### Purpose

Integrate time (4D) and cost (5D) into the BIM model, with construction progress driven by **actual scan data** rather than manual reports.

### Capabilities

#### Gantt / Schedule Management

- Import/create project schedule with WBS (Work Breakdown Structure)
- Each activity linked to BIM elements
- Progress automatically updated from scan comparison:
  ```
  Activity: "Level 3 — Structural Walls"
  ├─ Planned: 2026-04-01 → 2026-04-30
  ├─ BIM Elements: [Wall-301, Wall-302, ..., Wall-312]
  ├─ Coverage: 78% (from latest scan)
  ├─ Quality: 94% within tolerance
  └─ Computed Progress: 73% (coverage × quality)
  ```
- Critical path visualization
- Delay detection and impact analysis

#### S-Curve & Certification

- **S-Curve generation**: Planned vs actual progress over time
- **Certification engine**:
  - Certificates of work tied to verified physical progress
  - No certificate issued without scan verification above threshold
  - Each certificate includes:
    - Scan evidence (date, coverage, deviation report)
    - BIM element completion status
    - Quality compliance percentage
    - Photos/screenshots from scan
  - Digital signature workflow
- **Payment milestones** linked to verified progress
- **Transparency**: Every payment backed by verifiable scan data — public audit trail

#### Value Proposition

> For public works: **no more paying for work that hasn't been done or was done incorrectly.** Every certificate is backed by AI-verified 3D evidence.

### Technology

- **Gantt**: Custom implementation or integration with open-source (e.g., DHTMLX Gantt, Frappe Gantt)
- **S-Curve**: D3.js or Chart.js visualization
- **Certification**: PDF generation with digital signature (e.g., PyHanko)

---

## 3. Engineering Document Control

### Purpose

Full lifecycle management of engineering drawings and documents, with verification against contract requirements and BIM model consistency.

### Capabilities

- **Drawing management**:
  - Upload engineering drawings (PDF, DWG via conversion)
  - Version control with full history
  - Status workflow: `DRAFT → REVIEW → APPROVED → SUPERSEDED`
  - Observations and comments per drawing
  - Pending items tracker

- **Requirement verification**:
  - Each drawing linked to contract requirements it fulfills
  - AI-assisted check: "Does this drawing comply with specification X?"
  - Gap detection: requirements without associated drawings

- **BIM consistency check**:
  - Detect when engineering drawings deviate from the BIM model
  - Flag discrepancies for resolution
  - Track BIM updates needed when drawings change
  - Version timeline: drawing v3 → BIM update required → BIM v4

- **Drawing-to-BIM linking**:
  - Associate drawing sheets with BIM elements/zones
  - Click on a BIM element → see associated drawings
  - Click on a drawing → see which BIM elements it covers

### Technology

- **Drawing parsing**: PDF.js for viewing, potential OCR for title block extraction
- **Version control**: Git-like versioning for documents
- **BIM linking**: Metadata associations stored in project database

---

## 4. Quality, Safety & Environment (QSE + RAMS)

### Purpose

Integrated management of quality control, occupational safety, environmental compliance, and RAMS (Reliability, Availability, Maintainability, Safety) where applicable.

### Quality Management

- **Inspection checklists**: Per activity, per discipline
- **Non-conformance reports (NCR)**: Triggered automatically by scan deviations
  - NCR auto-generated when deviation exceeds tolerance
  - Linked to scan data, BIM element, and responsible party
  - Workflow: `OPEN → UNDER_REVIEW → CORRECTIVE_ACTION → VERIFIED → CLOSED`
- **Quality KPIs**: First-pass yield, NCR rate, rework percentage

### Safety Management

- **Safety observation reports**: Log incidents, near-misses, observations
- **Hazard identification**: AI analysis of scan images for safety violations
  - Missing guardrails, unsecured scaffolding, blocked exits
  - VLM (InternVL3) prompted for safety-specific analysis
- **Permit management**: Work permits, hot work permits, confined space

### Environmental Management

- **Environmental monitoring**: Track waste, emissions, noise levels
- **Compliance tracking**: Environmental requirements from tender specs
- **Impact reports**: Per activity environmental impact assessment

### RAMS (where applicable)

- Reliability, Availability, Maintainability, Safety analysis
- Applicable for infrastructure projects (metro, rail, energy)
- RAMS requirements tracked alongside construction progress
- Verification matrix linked to test/commissioning results

---

## 5. Communication & Meeting Governance

### Purpose

All project communication flows through STAC, creating an auditable record of decisions, agreements, and action items.

### Chat

- **Project chat**: Real-time messaging per team/discipline
- **Contextual threads**: Start a conversation about a specific BIM element, drawing, or NCR
- **AI assistant**: Ask questions about the project (RAG over all project data)

### Meeting Management

- **Meeting scheduling**: Calendar integration
- **Auto-generated minutes**: AI processes meeting audio/notes and produces:
  - Key discussion points
  - Decisions taken
  - Action items with responsible parties and deadlines
  - Agreements reached
- **Digital signature**: Attendees sign minutes on-site (tablet/phone)
- **Agreement tracking**: All agreements become tracked items
  - Status: `AGREED → IN_PROGRESS → COMPLETED → VERIFIED`
  - AI checks if agreements are being fulfilled
- **Legal compliance**: Verify that meeting processes comply with contractual requirements
  - Required notice periods for meetings
  - Quorum requirements
  - Authority to make decisions

### Technology

- **Audio transcription**: Whisper (OpenAI) or equivalent
- **Minutes generation**: LLM summarization with structured output
- **Digital signatures**: Cryptographic signing with PKI or simple biometric
- **Chat**: WebSocket-based real-time messaging (already have WebSocket infrastructure)

---

## 6. Regulatory & Standards Engine

### Purpose

Centralized repository of applicable standards, codes, and norms, linked to project requirements and verification activities.

### Capabilities

- **Standards library**: Upload and index applicable norms
  - ACI 117, ACI 301 (concrete tolerances)
  - ISO 1803 (construction tolerances)
  - DIN 18202 (dimensional tolerances)
  - Local building codes
  - Project-specific standards

- **Auto-linking**: AI associates standard clauses with:
  - Tender requirements
  - BIM elements
  - Quality checklists
  - Tolerance configurations

- **Compliance dashboard**: Per standard, what percentage of requirements are verified

---

## Integration Architecture

All future modules connect through the existing STAC core:

```
                    ┌─────────────────┐
                    │   STAC CORE     │
                    │ (3D + BIM + AI) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │Contract │        │ BIM 5D  │        │  QSE    │
    │  Docs   │◄──────►│ Gantt   │◄──────►│ RAMS    │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │  Eng.   │        │  Cert.  │        │ Meeting │
    │  Docs   │◄──────►│ S-Curve │◄──────►│  Gov.   │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │   REGULATORY    │
                    │    ENGINE       │
                    └─────────────────┘
```

### Data Flow Example: Wall Construction

```
1. Contract says: "Wall W-301 must be 200mm thick, plumb within 3mm/m"
2. Engineering drawing D-STR-042 details the wall → linked to BIM element
3. Gantt says: wall scheduled for April 15-20
4. Scan on April 22: wall built, 198mm thick, plumb 2.1mm/m
5. System:
   ├─ Coverage: 92% → progress updated in Gantt
   ├─ Quality: GOOD (within tolerance) → NCR not triggered
   ├─ Requirement: COMPLIANT → requirements matrix updated
   ├─ Certificate: eligible for payment
   └─ S-Curve: actual progress recorded
```

---

## Implementation Priority

| Phase | Module | Dependency | Estimated Effort |
|-------|--------|------------|-----------------|
| **Current** | 3D Core + BIM Comparison | — | Done |
| **P1** | Coverage Engine (zoom, occlusion) | Core | 3-4 weeks |
| **P2** | Multi-source scanning | Core | 2-3 weeks |
| **P3** | Contract docs + RAG chat | — | 4-6 weeks |
| **P4** | Requirements matrix | P3 | 2-3 weeks |
| **P5** | BIM 5D: Gantt + progress linking | Core + P1 | 4-6 weeks |
| **P6** | S-Curve + Certification | P5 | 3-4 weeks |
| **P7** | Engineering document control | P3 | 3-4 weeks |
| **P8** | QSE management | Core | 3-4 weeks |
| **P9** | Meeting governance + chat | — | 4-5 weeks |
| **P10** | Regulatory engine | P3 | 2-3 weeks |
| **P11** | RAMS (if applicable) | P8 | 2-3 weeks |

---

## AI Models Used Across Platform

| Model | Current Use | Future Use |
|-------|------------|------------|
| **DA3** | 3D reconstruction | Same |
| **SAM3** | Instance segmentation | Safety hazard detection |
| **InternVL3** | Scene inventory | Occlusion classification, safety analysis |
| **LLM (TBD)** | — | Contract chat (RAG), contradiction detection, meeting minutes, requirement extraction |
| **Whisper** | — | Meeting audio transcription |
| **Embedding model** | — | Document vectorization for RAG |

---

## Competitive Positioning

No existing platform combines ALL of these:

| Feature | OpenSpace | Avvir | Doxel | Procore | STAC |
|---------|-----------|-------|-------|---------|------|
| 3D from phone video | ❌ | ❌ | ❌ | ❌ | ✅ |
| AI segmentation | ❌ | Partial | Partial | ❌ | ✅ |
| BIM deviation map | ❌ | ✅ | ✅ | ❌ | ✅ |
| Occlusion-aware coverage | ❌ | ❌ | ❌ | ❌ | 🔜 |
| Contract RAG chat | ❌ | ❌ | ❌ | ❌ | 🔜 |
| Scan-verified certificates | ❌ | ❌ | ❌ | ❌ | 🔜 |
| BIM 5D with scan data | ❌ | ❌ | ❌ | ❌ | 🔜 |
| Meeting auto-minutes | ❌ | ❌ | ❌ | ❌ | 🔜 |
| Requirements traceability | ❌ | ❌ | ❌ | Partial | 🔜 |
| QSE/RAMS management | ❌ | ❌ | ❌ | Partial | 🔜 |

> **STAC's unique differentiator**: Physical reality verification (3D scanning) is the foundation for everything else. Certificates, payments, and compliance are backed by AI-verified evidence — not self-reported progress.
