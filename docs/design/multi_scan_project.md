# Multi-scan projects — layers, object-based registration, change comparison

Status: DESIGN (2026-09-06) — nothing implemented. Requires the user's approval.

## 0. What the user asked (his words, distilled)

- A **project = one place**; it holds **N scans** from different days.
  `pccr_v1` (bigger, OLDER) and `pccr` (partial, NEWER, reconstructing now) are
  two scans of the SAME place → they must be **merged into one project named
  `pccr`**: pccr_v1 becomes scan day **X**, current pccr becomes scan day **Y**.
  The `_v1` suffix only existed because the name was taken.
- Opening a project with several scans: choose **which scans are shown** —
  all at once (overlaid) or one — with toggles, like segments today.
- After segmenting the same elements in each scan (door1, box1 in both), the
  user must be able to **fit one cloud onto the other** using those shared
  objects, **compose** them, and see **similarities and differences**.
- The **reference scan is chosen by the user** per composition.
- Scans may be **complementary** (different parts of the place, no points in
  common): they still compose (each in its place) but cannot be registered
  against each other; ideally they share **≥2 objects at the border**.

## 1. What already exists (reuse, do not reinvent)

| piece | where | reuse |
|---|---|---|
| Project layout `projects/<slug>/project.json` + `scans/<date>/src_<source>/{frames,output}` | `project_paths.py` (`ProjectPaths`, `for_source`, `list_scan_days`) | the container is already there; `project.json.scan_days` is empty today |
| Session resolution = **latest scan day only** | `project_paths.resolve_session` | must become explicit per scan |
| One Potree per session `/potree/{session}/` → `ctx.merged_potree` (latest) | `main.py:1096` | needs `/potree/{project}/{scan_key}/` |
| Pipeline already accepts `scan_key = "date/source"` and multi-scan sequential runs | `main.py run_pipeline`, `pipeline_manager.start_pipeline(scan_key=…)` | keep |
| Per-scan artifacts: cloud, Potree, `segmentation_result.json`, `classification.npy`, `floor_transform.npz`, approved corrections | per `output/` | untouched — everything stays per scan |
| Floor alignment (height + tilt) and object alignment (XZ + yaw sweep, overlap score) | `bim/registration.py` `align_floors`, `align_objects`, `classify_matches` | adapt from cloud↔BIM to cloud↔cloud |
| Trimmed ICP / Kabsch on object pairs, held-out validation, one-level undo + Approve, pending banner | `segmentation/correction_analysis.py` | same machinery for scan↔scan registration |
| Segment membership + OBBs | `classification.npy` + result JSON (v4 design) | object correspondences by label |
| Viewer: single octree group `potree-octree`, `potree_ready` message `{url, points, floorTransform}`, per-segment visibility texture | `ui/src/components/Viewport.tsx` | becomes one octree group **per scan** |

## 2. Data model

`project.json` (project = place):
```json
{
  "name": "pccr", "slug": "pccr",
  "scans": [
    {"key": "2026-09-03/default", "label": "scan X (completo)", "role": "reference"},
    {"key": "2026-07-11/default", "label": "scan Y (parcial)"}
  ],
  "composition": {
    "reference": "2026-09-03/default",
    "transforms": {
      "2026-07-11/default": {"matrix": [16 numbers], "method": "objects",
                              "pairs": ["door1", "box1"], "rms_cm": 2.1,
                              "approved_at": "…"}
    }
  }
}
```
- Every scan keeps its own frame (its cloud, Potree, segmentation, floor
  transform, approved chunk corrections) — nothing is rewritten by composition.
- The **composition transform** of a non-reference scan is a rigid 4×4 in the
  reference scan's frame, stored in `project.json`, applied by the VIEWER at
  display time (like `floorTransform` today) and by any tool that computes
  across scans. The reference scan's transform is identity.
- Complementary scans with no registration: transform = identity (or a manual
  gizmo placement), `method: "manual"` / `"none"`.
- Changing the reference re-expresses all stored transforms (T_new = T_ref_old⁻¹·T) — no data loss.

## 3. Migration: pccr_v1 + pccr → `pccr`

Precondition: **no pipeline running** (the pccr reconstruction must finish
first; the "one and only one" guard already refuses concurrent runs).

Explicit, idempotent script `scripts/merge_projects.py --into pccr --from pccr_v1`:
1. `projects/pccr_v1/scans/2026-09-03/*` → `projects/pccr/scans/2026-09-03/*`
   (move, same filesystem → instant). Existing `projects/pccr/scans/2026-07-11`
   stays as scan Y.
2. Rewrite `projects/pccr/project.json` with both scans; reference = the
   user's choice (default: the older/complete one, X).
3. Anything keyed by session id inside a scan's `output/` (chat notes,
   dossiers, `scene_r.db`, `user_prefs`) is per scan and moves with it; a
   one-line check lists any file that still contains the literal `pccr_v1`.
4. `projects/pccr_v1/` is left as an empty shell until the user deletes it
   (nothing destructive without his word).

Verification: both scans open in the project; each shows its own cloud,
segments, corrections; `GET /sessions/pccr/scans` lists two.

## 4. Backend

- `resolve_session(session_id, scan_key=None)`: explicit scan; default stays
  "latest" ONLY for legacy callers, and every new endpoint passes the key.
- `/potree/{project}/{scan_key}/{file}` alongside the old route.
- `GET /sessions/{project}/scans` already exists → add `composition` block.
- `POST /api/project/{project}/composition/reference` — set reference.
- `POST /api/project/{project}/composition/register` — body `{source_scan,
  target_scan, pairs:[{label_src, label_dst}]}` → returns the rigid + report
  (pending); `approve` / `undo` reuse the correction state pattern (one
  pending composition at a time, banner).
- `POST /api/project/{project}/composition/manual` — gizmo placement of a
  whole scan (same as the chunk gizmo, scan-wide).
- `GET /api/project/{project}/compare?a=&b=` — differences (§6).

## 5. Viewer — scans as layers

- Panel "Scans": one row per scan (label, date, points, reference badge),
  checkbox visible/hidden, radio "reference". All checked by default when the
  project has ≥2 scans? → **user decides** (proposal: reference on, others
  on, remembered in user prefs).
- One `potree-octree-<scan>` group per visible scan, each with its own
  `floorTransform × compositionTransform`. Segments, OBBs, meshes and
  volumes of a scan live under its group → toggling a scan hides everything
  of that scan at once.
- Colour mode "by scan" (tint per scan) in addition to RGB / segments.
- Selecting a non-reference scan → gizmo (move/rotate, no scale, centered on
  the scan) → Save = manual composition (pending → Approve/Undo).

## 6. Registration by shared objects (scan ↔ scan)

Evidence = segments the user marks in BOTH scans (same label or explicit
pairing in a small dialog: `door1 ↔ door1`, `box1 ↔ box1`).

**USER RULES (2026-09-06), binding:**
- **Only INVARIANT objects are evidence**: elements that do not change
  position or size between days (doors, walls, fixed racks, columns) — never
  chairs, loose boxes, anything movable. The pairing dialog asks "invariant?"
  per pair; non-invariant pairs are excluded from registration and only
  used later, in the comparison, as change candidates.
- **Scale is estimated and split symmetrically.** The two scans may differ
  slightly in scale. The factor `s` between them is measured on the invariant
  objects (distances between their dominant planes); neither scan is forced
  onto the other: one is scaled by `√s`, the other by `1/√s`, so both end up
  equally complete, at the same position and size ("punto medio de toda la
  escena"). The composition transform is therefore a SIMILARITY (scale +
  rigid); the reference scan carries its half of the scale too.
- **Judge dominant planes, never the bbox.** A single floater inflates a
  bbox. Every invariant object is described by its dominant RANSAC planes
  (door leaf, rack faces, wall): position = plane locations, size = distance
  between opposite parallel planes (a rack's real width, a door's height),
  orientation = normals. Registration, scale estimation and comparison are
  plane-vs-plane, not box-vs-box. Reuse `surface_fit`'s RANSAC plane ladder.

1. Per pair: points of each segment from its scan's `classification.npy`
   (v4 access layer), expressed in each scan's floor-aligned frame; fit the
   dominant planes of each copy (RANSAC ladder, floaters ignored by
   construction).
2. Scale first: `s` = ratio of plane-to-plane distances (and inter-object
   plane distances) between the two copies, robust median over all invariant
   pairs; apply `√s` / `1/√s` symmetrically. Then initial guess: floors
   already at Y=0 in both → only XZ + yaw unknown → yaw sweep aligning the
   dominant plane normals and positions (robust, no local minima), then
   trimmed ICP on the union of paired segments (`correction_analysis._icp`)
   → similarity 4×4 per scan.
3. Guards (the ones §3 of MEJORAS_OBLIGATORIAS demands for corrections):
   ≥2 pairs required (one pair leaves yaw ambiguous — refuse, tell the user
   to segment a second shared object at the border); plausibility bounds;
   **held-out test**: apply to the whole source scan and measure the residual
   of surfaces NOT used as evidence (floor band, walls, any unpaired segment
   present in both) — if it worsens, REJECT with the reason on screen, do not
   apply.
4. Result = pending composition transform (§2), Approve / Undo. Nothing in
   either scan's files changes — only `project.json`.
5. Complementary scans (no pairs): registration refused by design; the user
   composes manually (gizmo) or leaves identity.

## 6b. Engine for §6–§7: CloudComPy (USER 2026-09-06)

The geometry runs on CloudCompare's own tools (env `CloudComPy310`, already
used by the on-load path), as subprocess jobs exchanging PLY/npy files:

- **Registration**: `cc.ICP(data, model, finalOverlapRatio, adjustScale=True)`
  → transform + scale factor + RMS, fed ONLY with the invariant objects'
  points of both scans (derived from `classification.npy`, exported as
  temporary PLYs). The symmetric scale split (√s / 1/√s) is applied on the
  result by us.
- **Dominant planes**: `cc.RansacSD` (RANSAC Shape Detection) per object
  copy → size/position fingerprint and held-out checks, replacing the
  home-grown RANSAC.
- **Comparison**: `computeCloud2CloudDistances` (C2C scalar field →
  common / only-A / only-B by threshold) and **M3C2** (signed distances
  along normals with statistical significance — the standard tool for
  multi-temporal change detection) for the per-region layer; per-object
  deltas from the `RansacSD` planes of each copy.
- **Ours**: which objects are invariant, the symmetric scale, the guards
  (≥2 objects, plausibility, held-out on unused surfaces), the pending →
  Approve/Undo flow, and camera COVERAGE so "absent" is never reported
  where a scan simply did not look (M3C2 cannot know that).

## 7. Comparison — similarities and differences

Inputs: reference scan R, composed scan S (transform applied), both floor-
aligned. Two levels, both tool_measured:

**Per object (segments)**: for every label present in either scan:
- in both → judged on DOMINANT PLANES (never the bbox — floaters): offset
  and normal delta between corresponding planes (`moved / same`), delta of
  plane-to-plane distances (`resized`), per-point nearest-distance p50/p95
  between the two copies (`deformed`);
- only in R → `missing in S` (removed, or S does not cover that area —
  decided by coverage, next point); only in S → `new`.

**Per region (cloud)**: voxelize both (same grid, e.g. 5 cm) in the reference
frame → cells `common / only-R / only-S / uncovered` (uncovered = outside the
other scan's camera coverage, so "absent" is not reported as a change where
the scan simply did not look — the coverage field from the camera poses gives
this). Output: a colour layer in the viewer (green common, blue only-R, red
only-S, grey uncovered) + a table.

Report saved as `projects/<slug>/compare/<R>__<S>.json` + shown in a panel;
the chat can read it (`get_session_info` extended with scans).

## 8. Order of work (each phase verifiable, each gated by the user)

1. Merge script + data model (`project.json` scans/composition) + explicit
   scan resolution + per-scan Potree route. Verify: pccr opens with two scans.
2. Viewer layers (per-scan octree groups, panel, colour by scan, reference
   radio). Verify: both clouds visible/hidden independently, correct frames.
3. Manual composition gizmo + pending/approve/undo. Verify: place scan Y by
   hand, approve, persists across reload.
4. Object-based registration with guards. Verify on pccr: door1+box1 paired
   → residual < 3 cm on held-out surfaces; with one pair → refused with
   message.
5. Comparison (per object + per region) + panel. Verify: table matches what
   the eye sees on door1/box1/racks.

Dependencies: §1 of MEJORAS_OBLIGATORIAS (membership v4) should land before
phase 4 (it needs cheap access to segment points in both scans).
