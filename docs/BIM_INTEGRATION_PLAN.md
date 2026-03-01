# STAC Build — BIM Integration & Multi-Cloud Comparison Plan

## 1. Vision

STAC Build will evolve from a single-cloud analysis tool to a **full 4D construction monitoring platform** that:

1. **Manages multiple point clouds per session** (different capture dates)
2. **Imports and displays BIM models** from major providers
3. **Compares clouds against each other** to track construction progress over time
4. **Compares clouds against BIM** to detect deviations, completeness, and clashes

---

## 2. Architecture Overview

### 2.1 Current Session Structure

```
Session (single capture)
└── output/
    ├── cleaned_cloud.ply
    ├── potree/ (LOD octree)
    ├── floor_transform.npz
    ├── segmentation.json
    └── segmentation_result.json
```

### 2.2 Proposed Session Structure

A session becomes a **project/site** containing multiple captures and an optional BIM reference:

```
Project (site)
├── bim/
│   ├── model.ifc              ← Original BIM file
│   ├── model.glb              ← Converted 3D mesh for viewer
│   ├── elements.json          ← Extracted BIM elements (walls, slabs, etc.)
│   └── bim_transform.npz      ← Alignment transform (BIM ↔ cloud coordinates)
├── captures/
│   ├── 2026-01-15/
│   │   ├── cleaned_cloud.ply
│   │   ├── potree/
│   │   ├── floor_transform.npz
│   │   ├── segmentation_result.json
│   │   └── metadata.json       ← Capture date, device, operator, notes
│   ├── 2026-01-31/
│   │   └── ...
│   └── 2026-02-15/
│       └── ...
├── comparisons/
│   ├── c2c_20260115_vs_20260131.npz   ← Cloud-to-Cloud distances
│   ├── c2b_20260131_vs_bim.npz        ← Cloud-to-BIM distances
│   └── progress_report.json           ← Computed progress metrics
└── project.json                ← Project metadata, capture list, BIM info
```

---

## 3. BIM Import

### 3.1 Supported Formats (by priority)

| Format | Extension | Provider | Strategy |
|--------|-----------|----------|----------|
| **IFC** | `.ifc` | Open standard (buildingSMART) | **Native support** via IfcOpenShell |
| **glTF/GLB** | `.gltf`, `.glb` | Universal 3D | Direct load in Three.js |
| **OBJ** | `.obj` | Universal | Direct load in Three.js |
| **FBX** | `.fbx` | Autodesk | Three.js FBXLoader |
| **Revit** | `.rvt` | Autodesk | Export to IFC/glTF (user responsibility) |
| **DWG/DXF** | `.dwg`, `.dxf` | AutoCAD | Limited — 2D only or via ODA converter |
| **STEP/IGES** | `.step`, `.iges` | Mechanical CAD | Via PythonOCC or FreeCAD |

### 3.2 IFC Processing Pipeline

```
                    ┌──────────────┐
   Upload .ifc  →   │ IfcOpenShell  │
                    │ (Python)      │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        elements.json   model.glb    spatial_tree.json
        (BIM elements   (3D mesh     (spatial hierarchy:
         with IFC IDs,   for viewer   Site > Building >
         types, props)   via Three.js) Storey > Space)
```

**Key processing steps:**

1. **Parse IFC** — Extract geometry, element types (IfcWall, IfcSlab, IfcColumn, etc.), properties, and spatial hierarchy
2. **Convert to GLB** — Triangulate BIM geometry → glTF/GLB for real-time viewer rendering
3. **Extract element metadata** — Store element IDs, types, materials, dimensions in `elements.json`
4. **Build spatial tree** — Organize elements by building storey and space for navigation
5. **Generate element point sampling** — Sample points on BIM surfaces for cloud-to-BIM distance computation

### 3.3 BIM Alignment

The BIM model and point clouds are typically in different coordinate systems:

- **BIM**: Architect's coordinate system (project north, local origin)
- **Cloud**: Scanner/SLAM coordinate system (instrument origin)

**Alignment options:**

1. **Manual gizmo alignment** — Use the existing alignment gizmo to manually position the BIM relative to the cloud (or vice versa)
2. **Georeferenced alignment** — If both BIM and cloud have georeferenced coordinates (survey points), compute the rigid transform automatically
3. **ICP (Iterative Closest Point)** — Given a rough manual alignment, refine automatically using ICP between BIM surfaces and cloud points
4. **Control points** — User picks corresponding points on BIM and cloud; system computes optimal rigid transform (SVD-based, minimum 3 pairs)

**Recommended approach**: Start with manual gizmo alignment (already built). Add control-point alignment as next step. ICP as refinement.

---

## 4. Multi-Cloud Sessions

### 4.1 Capture Management

Each capture within a project represents a **point cloud from a specific date**:

- **Metadata**: Date, time, capture device, operator, notes, weather conditions
- **Independent processing**: Each capture runs through its own pipeline (cleaning, Potree octree, segmentation)
- **Shared alignment**: All captures align to the same project coordinate system (via floor_transform or BIM reference)

### 4.2 Capture Timeline UI

```
┌─────────────────────────────────────────────────┐
│  Project: Hospital El Pilar                      │
│  BIM: hospital_pilar_v3.ifc  ✅ Loaded           │
│                                                   │
│  ─────────────────────────────────────────────── │
│  📅 Timeline                                      │
│  ● 2026-01-15  ▶ 13.2M pts  ✅ Segmented         │
│  ● 2026-01-31  ▶  3.4M pts  ✅ Segmented         │
│  ● 2026-02-15  ▶  8.7M pts  ⏳ Processing        │
│  ─────────────────────────────────────────────── │
│  📊 Comparisons                                   │
│  ● Jan 15 → Jan 31   Δ 42% new construction      │
│  ● Jan 31 → BIM      87% complete, 3 deviations  │
└─────────────────────────────────────────────────┘
```

### 4.3 Viewer: Multi-Cloud Toggle

The 3D viewer should allow:

- **Layer toggling** — Show/hide individual captures and BIM independently
- **Temporal slider** — Scrub through capture dates to see construction evolution
- **Split view** — Side-by-side comparison of two captures or capture-vs-BIM
- **Overlay mode** — Superimpose two datasets with transparency

---

## 5. Cloud-to-Cloud Comparison (C2C)

### 5.1 Purpose

Track **construction progress** between captures:

- What was **built** between Date A and Date B?
- What was **demolished** or removed?
- How much of the total project is **complete**?

### 5.2 Algorithm

1. **Load** cloud A and cloud B (both aligned to project coordinates)
2. **Build KD-tree** on cloud B
3. **For each point in cloud A**, find nearest neighbor in cloud B → distance
4. **Classify** each point:
   - Distance < threshold → **unchanged** (green)
   - Point in A, no match in B → **removed** (red)
   - Point in B, no match in A → **added** (blue)
5. **Compute metrics**: % area changed, volume of new construction, etc.

### 5.3 Visualization

- **Color-mapped point cloud** — Distance values mapped to a color gradient (green → yellow → red)
- **Change map** — Separate layers for added/removed/unchanged points
- **Statistical summary** — Histogram of distances, total area of change

### 5.4 Libraries

- **Open3D** — KD-tree, ICP, point cloud registration (Python)
- **scipy.spatial.KDTree** — Fast nearest-neighbor queries
- **CloudCompare CLI** — M3C2 algorithm for signed cloud-to-cloud distances (most accurate for construction)

---

## 6. Cloud-to-BIM Comparison (C2B)

### 6.1 Purpose

Compare **as-built** (point cloud) against **as-designed** (BIM model):

- **Geometric deviation** — How far is the actual construction from the design?
- **Completeness** — Which BIM elements are already built?
- **Clashes** — Elements that interfere or conflict with the design

### 6.2 Algorithm

1. **Sample BIM surfaces** — Generate a dense point sampling on each BIM element surface
2. **Build KD-tree** on BIM point samples
3. **For each cloud point**, find nearest BIM surface point → distance and element ID
4. **Per-element analysis**:
   - If enough cloud points match a BIM element → **element is built**
   - If most matched points are within tolerance → **element is correct**
   - If matched points deviate significantly → **element has deviations**
   - If no cloud points match → **element is not yet built**
5. **Generate report**: Per-element status (built/not-built/deviated), per-storey progress, overall completion %

### 6.3 Deviation Classification

| Status | Condition | Color |
|--------|-----------|-------|
| ✅ Built & correct | >80% coverage, mean deviation < 2cm | Green |
| ⚠️ Built with deviation | >80% coverage, mean deviation 2-5cm | Yellow |
| ❌ Major deviation | >80% coverage, mean deviation > 5cm | Red |
| 🔲 Not yet built | <20% coverage | Gray |
| 🔶 Partially built | 20-80% coverage | Orange |

### 6.4 Tolerances

Configurable per project and per element type:

```yaml
tolerances:
  default: 0.02          # 2cm general tolerance
  structural:
    columns: 0.01        # 1cm for structural columns
    slabs: 0.015         # 1.5cm for slabs
    beams: 0.01          # 1cm for beams
  architectural:
    walls: 0.025         # 2.5cm for walls
    partitions: 0.03     # 3cm for partitions
    facades: 0.02        # 2cm for facades
  mep:
    ducts: 0.05          # 5cm for MEP ducts
    pipes: 0.03          # 3cm for pipes
```

---

## 7. Reports & Export

### 7.1 Progress Reports

Auto-generated reports containing:

- **Overall completion %** (by volume, area, or element count)
- **Per-storey breakdown** — Which floors are complete
- **Per-element status table** — Every BIM element with built/not-built status
- **Deviation heat map** — Top-down view with color-coded deviations
- **Timeline chart** — Progress curve over captures

### 7.2 Export Formats

- **PDF** — Formatted progress report with images and charts
- **Excel/CSV** — Raw data tables for client analysis
- **BCF** (BIM Collaboration Format) — Industry-standard issue tracking format that can be imported back into Revit/Navisworks
- **IFC with markup** — Annotated IFC file with deviation data

---

## 8. Implementation Phases

### Phase 1: Multi-Cloud Foundation (2 weeks)

**Goal**: Restructure sessions to support multiple captures per project.

- [ ] Redesign session/project data model (`project.json`)
- [ ] Backend: multi-capture storage structure under `captures/`
- [ ] Backend: API endpoints for managing captures within a project
- [ ] Frontend: capture timeline UI with add/remove/toggle captures
- [ ] Frontend: layer toggle (show/hide individual clouds)
- [ ] Migration: convert existing single-cloud sessions to new structure

### Phase 2: BIM Import & Display (2 weeks)

**Goal**: Import IFC/glTF BIM models and display them alongside point clouds.

- [ ] Backend: BIM upload endpoint (accept IFC, glTF, OBJ, FBX)
- [ ] Backend: IFC → GLB conversion pipeline (IfcOpenShell + pythonocc-core)
- [ ] Backend: Extract BIM element metadata (`elements.json`)
- [ ] Backend: Build spatial hierarchy tree (`spatial_tree.json`)
- [ ] Frontend: Three.js GLBLoader for BIM visualization
- [ ] Frontend: BIM element tree panel (browse by storey/type)
- [ ] Frontend: Click BIM element → show properties panel
- [ ] Frontend: Transparency/wireframe/solid display modes for BIM

### Phase 3: BIM-Cloud Alignment (1 week)

**Goal**: Align BIM model and point clouds to the same coordinate system.

- [ ] Extend alignment gizmo to work on BIM model (not just cloud)
- [ ] Implement control-point alignment (pick 3+ pairs on BIM and cloud)
- [ ] Save BIM alignment transform (`bim_transform.npz`)
- [ ] Auto-apply alignment on session load

### Phase 4: Cloud-to-Cloud Comparison (2 weeks)

**Goal**: Compute and visualize differences between captures.

- [ ] Backend: C2C distance computation (KD-tree based)
- [ ] Backend: Change classification (added/removed/unchanged)
- [ ] Backend: CloudCompare M3C2 integration (optional, better accuracy)
- [ ] Backend: Cache comparison results
- [ ] Frontend: Color-mapped comparison visualization
- [ ] Frontend: Change histogram and statistics panel
- [ ] Frontend: Temporal slider for multi-capture animation

### Phase 5: Cloud-to-BIM Comparison (2 weeks)

**Goal**: Compare as-built cloud against BIM design.

- [ ] Backend: BIM surface point sampling
- [ ] Backend: C2B distance computation per element
- [ ] Backend: Element completeness analysis
- [ ] Backend: Deviation classification per tolerance thresholds
- [ ] Frontend: Color-coded BIM elements (built/not-built/deviated)
- [ ] Frontend: Deviation heat map overlay
- [ ] Frontend: Per-element deviation details panel

### Phase 6: Reports & BCF Export (1 week)

**Goal**: Generate professional progress reports and industry-standard exports.

- [ ] Backend: PDF report generation (progress charts, maps, tables)
- [ ] Backend: Excel/CSV export
- [ ] Backend: BCF export for Revit/Navisworks integration
- [ ] Frontend: Report configuration dialog (select metrics, date range)
- [ ] Frontend: Report preview and download

---

## 9. Technology Stack

### Backend (Python)

| Component | Library | Purpose |
|-----------|---------|---------|
| IFC parsing | `ifcopenshell` | Parse IFC files, extract geometry and properties |
| IFC → mesh | `pythonocc-core` or `trimesh` | Triangulate BIM geometry |
| GLB export | `trimesh`, `pygltflib` | Convert meshes to glTF/GLB |
| KD-tree | `scipy.spatial.KDTree` | Nearest-neighbor for C2C/C2B |
| Point cloud ops | `open3d` | ICP, registration, normals |
| C2C (advanced) | `CloudCompare` CLI | M3C2 signed distances |
| PDF reports | `reportlab` or `weasyprint` | Generate PDF progress reports |
| BCF export | `bcf-client` or custom | BIM Collaboration Format files |

### Frontend (TypeScript/Three.js)

| Component | Library | Purpose |
|-----------|---------|---------|
| BIM viewer | `three` GLTFLoader | Display GLB BIM models |
| BIM interaction | Custom | Element selection, property display |
| Color mapping | Custom shader | Deviation/distance visualization |
| Timeline | Custom or `vis-timeline` | Capture timeline UI |
| Charts | `chart.js` or `recharts` | Progress charts in reports |

---

## 10. Key Considerations

### 10.1 Performance

- BIM models can be **very large** (hundreds of MB for complex buildings). Need LOD (Level of Detail) and streaming.
- C2C/C2B computations on 10M+ point clouds need **GPU acceleration** or efficient spatial indexing.
- Multiple clouds loaded simultaneously → memory management critical.

### 10.2 Coordinate Systems

- All data (BIM, all clouds) must be in a **single project coordinate system**.
- The alignment gizmo is the foundation — every capture and the BIM must be aligned.
- Consider **georeferencing** via survey control points for large sites.

### 10.3 Data Integrity

- BIM models are **read-only** within STAC — no editing, only comparison.
- Original files (IFC, PLY) must be preserved; processed versions (GLB, etc.) can be regenerated.
- Comparison results are cached but can be recomputed if inputs change.

### 10.4 Industry Standards

- IFC 4.0 / IFC 4.3 compatibility
- BCF 2.1 for issue management
- COBie for facility management integration (future)
- E57 point cloud format support (surveying industry standard)

---

## 11. User Workflow

```
1. Create Project (site name, location)
2. Upload BIM (IFC/glTF)
3. Upload first cloud capture (or run SLAM pipeline)
4. Align cloud to BIM (gizmo or control points)
5. Run segmentation on cloud
6. Compare cloud vs BIM → see what's built
7. ...time passes, construction progresses...
8. Upload new capture
9. Auto-align to project coordinates
10. Compare new capture vs previous → see progress
11. Compare new capture vs BIM → updated completion %
12. Generate progress report → send to client
13. Repeat 7-12 throughout construction
```

---

*Document created: February 24, 2026*
*STAC Build — Construction Monitoring Platform*
