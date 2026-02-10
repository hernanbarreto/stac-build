"""
Segmentation Pipeline — Cloud-agnostic 2D instance mask storage + display-time matching.

Flow:
  1. Run SAM3 video propagation in BATCHES (overlapping windows for ID continuity)
  2. Match object IDs across batches using IoU in overlap regions
  3. Save masks as compressed NPZ (seg_masks.npz) + metadata (segmentation.json)
  4. At display time: apply_segmentation_to_cloud() matches masks against any PLY
     using per-point origin metadata (frame_global, pixel_row, pixel_col)

Usage from main.py:
    from segmentation_pipeline import run_segmentation, apply_segmentation_to_cloud
    run_segmentation(frames_dir, output_dir, prompt="chair")  # saves masks
    result = apply_segmentation_to_cloud(output_dir, ply_path)  # matches at display time
"""

import os
import json
import shutil
import numpy as np
import cv2
import gc
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("SegPipeline")


def run_segmentation(frames_dir: str, output_dir: str, prompt: str) -> dict:
    """
    Full segmentation pipeline: batched SAM3 → IoU ID matching → mask-to-point mapping.
    
    Uses the same blur-filtered frame set as DA3 to ensure frame_global indices match.
    Valid frames are copied to frames_valid/ with sequential numbering, then cleaned up.
    """
    from config import cfg
    
    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    seg_cfg = cfg["models"]["segmentation"]
    batch_size = seg_cfg.get("batch_size", 50)
    batch_overlap = seg_cfg.get("batch_overlap", 10)
    iou_threshold = seg_cfg.get("iou_match_threshold", 0.3)
    
    print(f"[SegPipeline] Starting segmentation for prompt: '{prompt}'")
    print(f"[SegPipeline] Frames: {frames_dir}  |  Batch: {batch_size} frames, {batch_overlap} overlap")
    
    # ── Step 1: Prepare valid frames (matching DA3's blur filter) ──
    seg_frames_dir, frame_files, frames_valid_dir = _prepare_valid_frames(frames_dir)
    
    total_frames = len(frame_files)
    print(f"[SegPipeline] Using {total_frames} valid frames for segmentation")
    
    if total_frames == 0:
        return {"error": "No frames found", "instances": []}
    
    try:
        # ── Step 2: Run SAM3 in batches with IoU matching ──
        all_masks = _run_sam3_batched(
            seg_frames_dir, frame_files, prompt,
            batch_size, batch_overlap, iou_threshold
        )
        
        if not all_masks:
            print("[SegPipeline] ⚠️ SAM3 produced no masks")
            return {"error": "No masks generated", "instances": []}
        
        # ── Step 3: Save masks and metadata ──
        seg_meta = _save_masks(output_dir, all_masks, prompt, cfg)
        
        return seg_meta
    
    finally:
        # ── Cleanup: vaciar frames_valid/ completamente ──
        if frames_valid_dir and frames_valid_dir.exists():
            shutil.rmtree(str(frames_valid_dir), ignore_errors=True)
            print(f"[SegPipeline] 🧹 frames_valid/ vaciado")


def _prepare_valid_frames(frames_dir: Path):
    """
    Copy valid (non-blurry) frames to frames_valid/ with sequential numbering
    matching DA3's frame_global indexing.
    
    Returns:
        (seg_frames_dir, frame_files, frames_valid_dir)
        - seg_frames_dir: directory to use for SAM3 (frames_valid/ or frames/)
        - frame_files: sorted list of filenames in seg_frames_dir
        - frames_valid_dir: path to cleanup (None if no filtering applied)
    """
    import shutil
    
    # Check if frame_quality.json exists (same filter DA3 uses)
    fq_path = frames_dir / "frame_quality.json"
    if not fq_path.exists():
        # No blur filtering — use all frames directly
        print(f"[SegPipeline] No frame_quality.json found — using all frames")
        frame_files = sorted([
            f for f in os.listdir(frames_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ], key=lambda f: int(os.path.splitext(f)[0]))
        return frames_dir, frame_files, None
    
    # Load valid frame list
    with open(fq_path) as f:
        fq_data = json.load(f)
    
    valid_filenames = sorted(
        [f["file"] for f in fq_data["frames"] if f["valid"]],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    
    total = fq_data["total_frames"]
    rejected = fq_data["rejected_frames"]
    print(f"[SegPipeline] Frame quality filter: {len(valid_filenames)}/{total} valid ({rejected} blurry removed)")
    
    if not valid_filenames:
        return frames_dir, [], None
    
    # Crear frames_valid/ limpio (borrar cualquier resto de runs anteriores)
    frames_valid_dir = frames_dir.parent / "frames_valid"
    if frames_valid_dir.exists():
        shutil.rmtree(str(frames_valid_dir))
    frames_valid_dir.mkdir()
    
    # Copy valid frames with sequential numbering (matching DA3's frame_global)
    index_mapping = {}
    seq_frame_files = []
    
    for seq_idx, orig_filename in enumerate(valid_filenames):
        src = frames_dir / orig_filename
        ext = src.suffix
        new_name = f"{seq_idx:06d}{ext}"
        dst = frames_valid_dir / new_name
        
        if src.exists():
            shutil.copy2(str(src), str(dst))
            seq_frame_files.append(new_name)
    
    print(f"[SegPipeline] Copied {len(seq_frame_files)} valid frames to {frames_valid_dir}")
    
    return frames_valid_dir, seq_frame_files, frames_valid_dir


# ═══════════════════════════════════════════════════════════════════
#  BATCHED SAM3 PROCESSING
# ═══════════════════════════════════════════════════════════════════

def _run_sam3_batched(frames_dir: Path, frame_files: List[str], prompt: str,
                      batch_size: int, batch_overlap: int,
                      iou_threshold: float) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Process frames in overlapping batches, matching IDs across batches via IoU.
    
    Returns:
        Dict[orig_frame_idx → {global_obj_id: binary_mask}]
    """
    from sam3_wrapper import get_sam3_wrapper
    
    total_frames = len(frame_files)
    batch_step = batch_size - batch_overlap
    
    # Calculate batch boundaries
    batches = []
    start = 0
    while start < total_frames:
        end = min(start + batch_size, total_frames)
        batches.append((start, end))
        if end >= total_frames:
            break
        start += batch_step
    
    print(f"[SegPipeline] Processing {total_frames} frames in {len(batches)} batches")
    
    sam3 = get_sam3_wrapper()
    
    # Global state
    all_masks = {}  # orig_frame_idx → {global_obj_id: mask}
    next_global_id = 1
    prev_batch_masks = None  # Masks from previous batch (for overlap IoU matching)
    prev_overlap_start = None
    
    for batch_idx, (b_start, b_end) in enumerate(batches):
        batch_frame_files = frame_files[b_start:b_end]
        batch_len = len(batch_frame_files)
        
        print(f"\n[SegPipeline] ── Batch {batch_idx}/{len(batches)-1}: "
              f"frames {b_start}–{b_end-1} ({batch_len} frames) ──")
        
        # Create temp directory with sequential symlinks for this batch
        batch_dir, index_mapping = _prepare_batch_dir(frames_dir, batch_frame_files, b_start)
        
        try:
            # Run SAM3 on this batch
            raw_results = sam3.process_batch(
                str(batch_dir), prompt, index_mapping
            )
            
            if not raw_results:
                print(f"[SegPipeline] Batch {batch_idx}: no masks produced")
                continue
            
            # Parse raw SAM3 output into structured format
            batch_masks = _parse_raw_masks(raw_results)
            
            if not batch_masks:
                print(f"[SegPipeline] Batch {batch_idx}: no valid masks after parsing")
                continue
            
            # Match IDs with previous batch via IoU in overlap region
            if batch_idx == 0:
                # First batch: assign initial global IDs
                id_remap = {}
                batch_obj_ids = set()
                for frame_masks in batch_masks.values():
                    batch_obj_ids.update(frame_masks.keys())
                for local_id in sorted(batch_obj_ids):
                    id_remap[local_id] = next_global_id
                    next_global_id += 1
            else:
                # Subsequent batches: IoU matching in overlap zone
                overlap_start_frame = b_start  # Original frame idx where overlap begins
                overlap_end_frame = prev_overlap_start + batch_step + batch_overlap - 1 if prev_overlap_start is not None else b_start + batch_overlap - 1
                
                id_remap, next_global_id = _match_ids_iou(
                    prev_batch_masks, batch_masks,
                    overlap_start=overlap_start_frame,
                    overlap_end=min(overlap_end_frame, b_end - 1),
                    iou_threshold=iou_threshold,
                    next_global_id=next_global_id
                )
            
            # Apply ID remap and merge into global masks
            # For overlap frames, prefer the newer batch (fresher propagation)
            remapped_batch = {}
            for orig_idx, frame_masks in batch_masks.items():
                remapped = {}
                for local_id, mask in frame_masks.items():
                    global_id = id_remap.get(local_id, local_id)
                    remapped[global_id] = mask
                all_masks[orig_idx] = remapped
                remapped_batch[orig_idx] = remapped
            
            # Store REMAPPED batch masks for next overlap matching
            # (so IoU matching compares against global IDs, not local)
            prev_batch_masks = remapped_batch
            prev_overlap_start = b_start
            
            unique_objects = set()
            for fm in batch_masks.values():
                unique_objects.update(fm.keys())
            print(f"[SegPipeline] Batch {batch_idx}: {len(batch_masks)} frames, "
                  f"{len(unique_objects)} objects → remapped to {len(set(id_remap.values()))} global IDs")
            
        finally:
            # Clean up temp directory
            shutil.rmtree(batch_dir, ignore_errors=True)
    
    # Unload SAM3 to free VRAM
    sam3.unload_model()
    gc.collect()
    
    total_objects = set()
    for fm in all_masks.values():
        total_objects.update(fm.keys())
    print(f"\n[SegPipeline] SAM3 complete: {len(all_masks)} frames, "
          f"{len(total_objects)} unique objects across all batches")
    
    return all_masks


def _prepare_batch_dir(frames_dir: Path, batch_files: List[str], 
                       start_idx: int) -> Tuple[Path, Dict[int, int]]:
    """
    Create a temp directory with sequentially numbered symlinks for a batch.
    
    Returns:
        (batch_dir, index_mapping) where index_mapping = {local_idx: original_frame_idx}
    """
    batch_dir = Path(tempfile.mkdtemp(prefix="sam3_batch_"))
    index_mapping = {}
    
    for local_idx, filename in enumerate(batch_files):
        src = frames_dir / filename
        ext = src.suffix
        dst = batch_dir / f"{local_idx:06d}{ext}"
        dst.symlink_to(src)
        
        # Extract original frame index from filename
        orig_idx = int(os.path.splitext(filename)[0])
        index_mapping[local_idx] = orig_idx
    
    return batch_dir, index_mapping


def _parse_raw_masks(raw_results: Dict[int, dict]) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Convert raw SAM3 output to structured format.
    
    Input:  {orig_frame_idx: {"out_binary_masks": ndarray, "out_obj_ids": ndarray}}
    Output: {orig_frame_idx: {obj_id: binary_mask_2d}}
    """
    structured = {}
    
    for frame_idx, outputs in raw_results.items():
        if "out_binary_masks" not in outputs:
            continue
        
        masks = outputs["out_binary_masks"]
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        
        obj_ids = outputs.get("out_obj_ids", None)
        if obj_ids is not None and hasattr(obj_ids, 'cpu'):
            obj_ids = obj_ids.cpu().numpy()
        
        frame_masks = {}
        if masks.ndim == 3:
            for i in range(masks.shape[0]):
                oid = int(obj_ids[i]) if obj_ids is not None and i < len(obj_ids) else i
                if masks[i].any():
                    frame_masks[oid] = masks[i]
        elif masks.ndim == 2:
            if masks.any():
                oid = int(obj_ids[0]) if obj_ids is not None and len(obj_ids) > 0 else 0
                frame_masks[oid] = masks
        
        if frame_masks:
            structured[frame_idx] = frame_masks
    
    return structured


def _match_ids_iou(prev_masks: Dict[int, Dict[int, np.ndarray]],
                   curr_masks: Dict[int, Dict[int, np.ndarray]],
                   overlap_start: int, overlap_end: int,
                   iou_threshold: float,
                   next_global_id: int) -> Tuple[Dict[int, int], int]:
    """
    Match object IDs between batches using IoU in the overlap region.
    
    Returns:
        (id_remap, next_global_id) where id_remap = {curr_local_id → global_id}
    """
    # Collect overlap frame indices present in both batches
    overlap_frames = []
    for fidx in range(overlap_start, overlap_end + 1):
        if fidx in prev_masks and fidx in curr_masks:
            overlap_frames.append(fidx)
    
    if not overlap_frames:
        # No overlap — assign fresh IDs to all objects
        curr_obj_ids = set()
        for fm in curr_masks.values():
            curr_obj_ids.update(fm.keys())
        id_remap = {}
        for cid in sorted(curr_obj_ids):
            id_remap[cid] = next_global_id
            next_global_id += 1
        print(f"[SegPipeline] IoU: No overlap frames → {len(id_remap)} new IDs")
        return id_remap, next_global_id
    
    # Collect all object IDs from each batch in the overlap region
    prev_obj_ids = set()
    curr_obj_ids = set()
    for fidx in overlap_frames:
        prev_obj_ids.update(prev_masks[fidx].keys())
        curr_obj_ids.update(curr_masks[fidx].keys())
    
    prev_obj_ids = sorted(prev_obj_ids)
    curr_obj_ids = sorted(curr_obj_ids)
    
    # Compute aggregated IoU matrix: prev_obj × curr_obj
    iou_matrix = np.zeros((len(prev_obj_ids), len(curr_obj_ids)))
    
    for fidx in overlap_frames:
        prev_fm = prev_masks.get(fidx, {})
        curr_fm = curr_masks.get(fidx, {})
        
        for pi, pid in enumerate(prev_obj_ids):
            if pid not in prev_fm:
                continue
            pmask = prev_fm[pid].astype(bool)
            
            for ci, cid in enumerate(curr_obj_ids):
                if cid not in curr_fm:
                    continue
                cmask = curr_fm[cid].astype(bool)
                
                # Handle resolution mismatch
                if pmask.shape != cmask.shape:
                    cmask = cv2.resize(cmask.astype(np.uint8), 
                                      (pmask.shape[1], pmask.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                
                intersection = np.logical_and(pmask, cmask).sum()
                union = np.logical_or(pmask, cmask).sum()
                if union > 0:
                    iou_matrix[pi, ci] += intersection / union
    
    # Average across frames
    iou_matrix /= max(len(overlap_frames), 1)
    
    # Greedy matching (Hungarian would be ideal but greedy is simpler and sufficient)
    id_remap = {}
    used_prev = set()
    
    # Build previous batch's local-id → global-id mapping
    # Previous batch masks already have global IDs from earlier remapping
    prev_local_to_global = {pid: pid for pid in prev_obj_ids}  # Already global IDs
    
    # Sort by IoU descending for greedy matching
    pairs = []
    for pi, pid in enumerate(prev_obj_ids):
        for ci, cid in enumerate(curr_obj_ids):
            if iou_matrix[pi, ci] >= iou_threshold:
                pairs.append((iou_matrix[pi, ci], pid, cid))
    pairs.sort(reverse=True)
    
    used_curr = set()
    for iou_val, pid, cid in pairs:
        if pid in used_prev or cid in used_curr:
            continue
        # Match: current object cid maps to previous global ID pid
        global_id = prev_local_to_global.get(pid, pid)
        id_remap[cid] = global_id
        used_prev.add(pid)
        used_curr.add(cid)
        print(f"[SegPipeline] IoU: obj {cid} → global {global_id} (IoU={iou_val:.3f})")
    
    # Unmatched current objects get new global IDs
    for cid in curr_obj_ids:
        if cid not in id_remap:
            id_remap[cid] = next_global_id
            print(f"[SegPipeline] IoU: obj {cid} → NEW global {next_global_id}")
            next_global_id += 1
    
    return id_remap, next_global_id


# ═══════════════════════════════════════════════════════════════════
#  MASK STORAGE (cloud-agnostic)
# ═══════════════════════════════════════════════════════════════════

def _save_masks(output_dir: Path, all_masks: Dict[int, Dict[int, np.ndarray]],
                prompt: str, cfg: dict):
    """
    Save SAM3 masks as compressed NPZ + metadata JSON.
    
    seg_masks.npz: per-frame, per-object boolean masks
       keys: "f{frame}_o{obj_id}" → bool array (H, W)
       plus: "obj_ids" → int array, "frames" → int array
    
    segmentation.json: metadata only (prompt, objects, colors, resolution)
    """
    colors = cfg["visualization"]["segment_colors"]
    
    # Load resolution from chunk metadata
    meta_files = sorted(output_dir.glob("chunk_*_meta.json"))
    scaled_res = [504, 280]
    original_res = [1280, 720]
    if meta_files:
        with open(meta_files[0]) as f:
            meta = json.load(f)
            scaled_res = meta.get("scaled_resolution", scaled_res)
            original_res = meta.get("original_resolution", original_res)
    
    # Collect all object IDs and frames
    obj_ids = set()
    frame_indices = sorted(all_masks.keys())
    for fm in all_masks.values():
        obj_ids.update(fm.keys())
    obj_ids = sorted(obj_ids)
    
    # Build NPZ data dict
    npz_data = {
        "obj_ids": np.array(obj_ids, dtype=np.int32),
        "frames": np.array(frame_indices, dtype=np.int32),
        "scaled_res": np.array(scaled_res, dtype=np.int32),
    }
    
    mask_count = 0
    for frame_idx, frame_masks in all_masks.items():
        for obj_id, mask in frame_masks.items():
            key = f"f{frame_idx}_o{obj_id}"
            # Ensure mask is at scaled resolution
            if mask.shape[0] != scaled_res[0] or mask.shape[1] != scaled_res[1]:
                mask = cv2.resize(mask.astype(np.uint8),
                                 (scaled_res[1], scaled_res[0]),
                                 interpolation=cv2.INTER_NEAREST)
            npz_data[key] = mask.astype(np.uint8)  # uint8 compresses well
            mask_count += 1
    
    # Save compressed NPZ
    masks_path = output_dir / "seg_masks.npz"
    np.savez_compressed(masks_path, **npz_data)
    masks_mb = masks_path.stat().st_size / (1024 * 1024)
    print(f"[SegPipeline] ✅ Saved masks: {masks_path.name} "
          f"({mask_count} masks, {len(obj_ids)} objects, {len(frame_indices)} frames, "
          f"{masks_mb:.1f} MB)")
    
    # Save metadata JSON (lightweight)
    instances = []
    for i, obj_id in enumerate(obj_ids):
        instances.append({
            "id": int(obj_id),
            "label": prompt,
            "instance_id": i + 1,
            "color": colors[i % len(colors)],
        })
    
    segmentation = {
        "version": "3.0",
        "prompt": prompt,
        "resolution": {"scaled": scaled_res, "original": original_res},
        "instances": instances,
        "mask_file": "seg_masks.npz",
    }
    
    seg_path = output_dir / "segmentation.json"
    with open(seg_path, 'w') as f:
        json.dump(segmentation, f, indent=2)
    print(f"[SegPipeline] ✅ Saved metadata: {seg_path.name} ({len(instances)} instances)")
    
    return segmentation


# ═══════════════════════════════════════════════════════════════════
#  DISPLAY-TIME MATCHING (cloud-agnostic)
# ═══════════════════════════════════════════════════════════════════

def _load_ply_origins(ply_path: Path):
    """Load point origins (frame_global, pixel_row, pixel_col) and xyz from a binary PLY.
    Returns (xyz, frame_global, pixel_row, pixel_col) or None if no origins.
    """
    try:
        with open(ply_path, 'rb') as f:
            n_pts = 0
            has_origins = False
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n_pts = int(line.split()[-1])
                if 'frame_global' in line:
                    has_origins = True
                if line == 'end_header':
                    break
            
            if not has_origins or n_pts == 0:
                return None
            
            dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ('frame_global', '<i4'),
                ('pixel_row', '<i2'), ('pixel_col', '<i2')
            ])
            data = np.frombuffer(f.read(), dtype=dtype)
            xyz = np.column_stack([data['x'], data['y'], data['z']])
            return xyz, data['frame_global'], data['pixel_row'], data['pixel_col']
    except Exception as e:
        print(f"[SegPipeline] Error loading PLY origins from {ply_path}: {e}")
        return None


def _compute_obb(points_xyz: np.ndarray) -> dict:
    """Compute minimum Oriented Bounding Box for floor-aligned coordinates.
    
    Uses convex hull + rotating calipers on the XZ plane (floor) to find the
    minimum-area footprint, then extends vertically (Y axis).
    Coordinates must be floor-aligned (Y = up).
    """
    if len(points_xyz) < 4:
        center = points_xyz.mean(axis=0)
        return {
            "center": center.tolist(),
            "half_extents": [0.01, 0.01, 0.01],
            "rotation": [[1,0,0],[0,1,0],[0,0,1]]
        }
    
    # Y extent (vertical)
    y_min = points_xyz[:, 1].min()
    y_max = points_xyz[:, 1].max()
    half_y = (y_max - y_min) / 2.0
    cy = (y_min + y_max) / 2.0
    
    # Project to XZ plane for 2D minimum bounding rectangle
    pts_xz = points_xyz[:, [0, 2]]  # (N, 2): [x, z]
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_xz)
        hull_pts = pts_xz[hull.vertices]
    except Exception:
        # Fallback: use all points
        hull_pts = pts_xz
    
    # Rotating calipers: try each hull edge as candidate orientation
    n_hull = len(hull_pts)
    best_area = float('inf')
    best_angle = 0.0
    best_min = np.zeros(2)
    best_max = np.zeros(2)
    
    for i in range(n_hull):
        edge = hull_pts[(i + 1) % n_hull] - hull_pts[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue
        angle = np.arctan2(edge[1], edge[0])
        
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rot2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated = hull_pts @ rot2d.T
        rmin = rotated.min(axis=0)
        rmax = rotated.max(axis=0)
        area = (rmax[0] - rmin[0]) * (rmax[1] - rmin[1])
        
        if area < best_area:
            best_area = area
            best_angle = angle
            best_min = rmin
            best_max = rmax
    
    # Half extents in the rotated frame: [along-edge, Y, perpendicular]
    half_x = (best_max[0] - best_min[0]) / 2.0
    half_z = (best_max[1] - best_min[1]) / 2.0
    
    # Center in rotated 2D space
    cx_rot = (best_max[0] + best_min[0]) / 2.0
    cz_rot = (best_max[1] + best_min[1]) / 2.0
    
    # Transform center back to world XZ
    cos_back = np.cos(best_angle)
    sin_back = np.sin(best_angle)
    rot_back = np.array([[cos_back, -sin_back], [sin_back, cos_back]])
    center_xz = rot_back @ np.array([cx_rot, cz_rot])
    
    center = [float(center_xz[0]), float(cy), float(center_xz[1])]
    half_extents = [float(half_x), float(half_y), float(half_z)]
    
    # Rotation matrix: Y-axis rotation by best_angle
    # Maps box local X → world direction along edge, local Z → perpendicular
    rotation = [
        [float(cos_back), 0.0, float(-sin_back)],
        [0.0, 1.0, 0.0],
        [float(sin_back), 0.0, float(cos_back)]
    ]
    
    return {
        "center": center,
        "half_extents": half_extents,
        "rotation": rotation
    }


def _find_nearest_keyframe(frame_idx: int, keyframes: list) -> Optional[int]:
    """Find the nearest keyframe to a given frame index."""
    if not keyframes:
        return None
    if frame_idx in keyframes:
        return frame_idx
    
    best = keyframes[0]
    best_dist = abs(frame_idx - best)
    for kf in keyframes:
        d = abs(frame_idx - kf)
        if d < best_dist:
            best = kf
            best_dist = d
    return best


def apply_segmentation_to_cloud(output_dir, ply_path=None) -> dict:
    """
    Match saved SAM3 masks against a loaded PLY cloud at display time.
    
    For each point in the PLY, uses its (frame_global, pixel_row, pixel_col)
    to check if it falls inside any object's mask. Points from filtered clouds
    that were removed simply won't be in the PLY → not highlighted.
    
    Args:
        output_dir: session output directory containing seg_masks.npz + segmentation.json
        ply_path: path to the PLY to match against. If None, auto-detects
                  cleaned_cloud.ply or chunk_000.ply
    
    Returns:
        v2.0-compatible dict with type="segmentation", instances with globalIndices + obb
    """
    from config import cfg
    
    output_dir = Path(output_dir)
    
    # Load metadata
    seg_path = output_dir / "segmentation.json"
    if not seg_path.exists():
        return {"error": "No segmentation.json", "instances": []}
    
    with open(seg_path) as f:
        metadata = json.load(f)
    
    # Load masks
    masks_path = output_dir / metadata.get("mask_file", "seg_masks.npz")
    if not masks_path.exists():
        # Legacy format (v2.0) — return as-is for backward compatibility
        metadata["type"] = "segmentation"
        return metadata
    
    masks_data = np.load(masks_path)
    obj_ids = masks_data["obj_ids"].tolist()
    keyframes = masks_data["frames"].tolist()
    scaled_res = masks_data["scaled_res"].tolist()
    
    # Auto-detect PLY if not provided
    if ply_path is None:
        cleaned = output_dir / "cleaned_cloud.ply"
        raw = output_dir / "chunk_000.ply"
        ply_path = cleaned if cleaned.exists() else raw
    ply_path = Path(ply_path)
    
    # Load cloud origins
    origins = _load_ply_origins(ply_path)
    if origins is None:
        print(f"[SegPipeline] ⚠️ PLY {ply_path.name} has no origin fields")
        return {"error": "PLY has no origins", "instances": []}
    
    xyz, frame_global, pixel_row, pixel_col = origins
    n_pts = len(frame_global)
    cloud_label = ply_path.stem
    print(f"[SegPipeline] Matching masks against {cloud_label} ({n_pts:,} points)...")
    
    # Apply floor alignment to xyz (same transform the viewer applies)
    xyz_display = xyz  # default: use raw xyz
    try:
        from alignment_manager import get_alignment_manager
        am = get_alignment_manager()
        s, R, t = am.compute_leveling_from_points(xyz)
        if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
            xyz_display = s * (xyz @ R.T) + t
            print(f"[SegPipeline]   Floor alignment applied to OBB coordinates")
    except Exception as e:
        print(f"[SegPipeline]   ⚠️ Floor alignment unavailable for OBB: {e}")
    
    # Group cloud points by frame for efficient lookup
    frame_groups = {}  # frame_idx → array of point indices
    frame_arr = frame_global.astype(np.int32)
    for pt_idx in range(n_pts):
        f = int(frame_arr[pt_idx])
        if f not in frame_groups:
            frame_groups[f] = []
        frame_groups[f].append(pt_idx)
    # Convert to numpy arrays for vectorized operations
    for f in frame_groups:
        frame_groups[f] = np.array(frame_groups[f], dtype=np.int64)
    
    print(f"[SegPipeline]   {len(frame_groups)} unique frames in cloud")
    
    # Match each object's masks against cloud points
    colors = cfg["visualization"]["segment_colors"]
    instances = []
    total_segmented = 0
    
    for i, obj_id in enumerate(obj_ids):
        matched_indices = []
        
        for cloud_frame, pt_indices in frame_groups.items():
            # Find nearest SAM3 keyframe
            nearest_kf = _find_nearest_keyframe(cloud_frame, keyframes)
            if nearest_kf is None:
                continue
            
            # Load mask for this frame+object
            mask_key = f"f{nearest_kf}_o{obj_id}"
            if mask_key not in masks_data:
                continue
            
            mask = masks_data[mask_key].astype(bool)
            
            # Look up each point's pixel in the mask
            rows = np.clip(pixel_row[pt_indices], 0, mask.shape[0] - 1)
            cols = np.clip(pixel_col[pt_indices], 0, mask.shape[1] - 1)
            in_mask = mask[rows, cols]
            
            matched = pt_indices[in_mask]
            if len(matched) > 0:
                matched_indices.append(matched)
        
        if not matched_indices:
            continue
        
        all_matched = np.concatenate(matched_indices)
        all_matched = np.unique(all_matched)  # deduplicate
        total_segmented += len(all_matched)
        
        # Build instance data
        instance = {
            "id": int(obj_id),
            "label": metadata.get("prompt", "object"),
            "instance_id": i + 1,
            "color": colors[i % len(colors)],
            "total_points": int(len(all_matched)),
            "globalIndices": all_matched.tolist(),
        }
        
        # Compute OBB using floor-aligned coordinates
        if len(all_matched) >= 4:
            instance["obb"] = _compute_obb(xyz_display[all_matched])
        
        instances.append(instance)
        print(f"[SegPipeline]   Object {obj_id} ('{metadata.get('prompt', '')}' #{i+1}): "
              f"{len(all_matched):,} points matched")
    
    coverage = round(total_segmented / max(1, n_pts), 4)
    
    result = {
        "type": "segmentation",
        "version": "3.0",
        "prompt": metadata.get("prompt", ""),
        "cloud_source": cloud_label,
        "total_points": n_pts,
        "segmented_points": total_segmented,
        "coverage": coverage,
        "instances": instances,
        "resolution": metadata.get("resolution", {}),
    }
    
    print(f"[SegPipeline] ✅ {len(instances)} instances matched against {cloud_label}, "
          f"{total_segmented:,}/{n_pts:,} points ({coverage*100:.1f}% coverage)")
    
    return result

