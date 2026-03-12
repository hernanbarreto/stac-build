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
import torch
import numpy as np
import cv2
import gc
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("SegPipeline")


def run_segmentation(frames_dir: str, output_dir: str, prompt: str,
                     frame_map: dict = None, on_progress=None) -> dict:
    """
    Full segmentation pipeline: batched SAM3 → IoU ID matching → mask-to-point mapping.
    
    Supports multiple categories separated by ';' (e.g., "sofa;cushion;table").
    Uses the same blur-filtered frame set as reconstruction to ensure frame_global indices match.
    Valid frames are copied to frames_valid/ with sequential numbering, then cleaned up.
    
    Args:
        frame_map: Optional dict mapping category label → list of frame filenames.
                   If provided, SAM3 only processes frames where each category was detected.
    """
    from config import cfg
    
    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    seg_cfg = cfg["models"]["segmentation"]
    batch_size = seg_cfg.get("batch_size", 50)
    batch_overlap = seg_cfg.get("batch_overlap", 10)
    iou_threshold = seg_cfg.get("iou_match_threshold", 0.3)
    
    # Split prompt by ';' for multi-category support
    categories = [c.strip() for c in prompt.split(";") if c.strip()]
    if not categories:
        return {"error": "Empty prompt", "instances": []}
    
    print(f"[SegPipeline] Starting segmentation for {len(categories)} categories: {categories}")
    print(f"[SegPipeline] Frames: {frames_dir}  |  Batch: {batch_size} frames, {batch_overlap} overlap")
    
    # ── Step 1: Prepare valid frames (matching reconstruction's blur + novelty filter) ──
    frame_sel_cfg = cfg.get("frame_selection", {})
    frame_stride = cfg.get("server", {}).get("frame_stride", 1)
    seg_frames_dir, frame_files, frames_valid_dir = _prepare_valid_frames(
        frames_dir, frame_stride, frame_sel_cfg
    )
    
    total_frames = len(frame_files)
    print(f"[SegPipeline] Using {total_frames} valid frames for segmentation")
    
    if total_frames == 0:
        return {"error": "No frames found", "instances": []}
    
    try:
        # ── Step 2: Run SAM3 in batches with IoU matching (per category) ──
        all_masks, obj_labels = _run_sam3_batched(
            seg_frames_dir, frame_files, categories,
            batch_size, batch_overlap, iou_threshold,
            output_dir=output_dir, cfg=cfg,
            frame_map=frame_map,
            on_progress=on_progress,
        )
        
        if not all_masks:
            print("[SegPipeline] ⚠️ SAM3 produced no masks")
            return {"error": "No masks generated", "instances": []}
        
        # ── Step 3: Save masks and metadata ──
        seg_meta = _save_masks(output_dir, all_masks, categories, obj_labels, cfg)
        
        # ── Step 4: Match masks to cloud and cache final result ──
        result = _match_and_save_result(output_dir)
        if result.get("instances"):
            return result
        return seg_meta
    
    finally:
        # ── Cleanup: vaciar frames_valid/ completamente ──
        if frames_valid_dir and frames_valid_dir.exists():
            shutil.rmtree(str(frames_valid_dir), ignore_errors=True)
            print(f"[SegPipeline] 🧹 frames_valid/ vaciado")


def _prepare_valid_frames(frames_dir: Path, frame_stride: int = 1,
                          frame_sel_cfg: dict = None):
    """
    Copy valid frames to frames_valid/ with sequential numbering
    matching reconstruction's frame_global indexing.
    
    Frame selection priority:
      1. selected_frames.json (visual novelty H/F filter) — if available
      2. frame_stride (fixed decimation) — fallback
    
    Args:
        frame_stride: Take 1 every N valid frames (fallback, from config.yaml)
        frame_sel_cfg: frame_selection config section (for novelty filter)
    
    Returns:
        (seg_frames_dir, frame_files, frames_valid_dir)
    """
    import shutil
    
    # ── Try visual novelty filter first (selected_frames.json) ──
    sel_path = frames_dir / "selected_frames.json"
    use_novelty = False
    
    if frame_sel_cfg and frame_sel_cfg.get("enabled", False) and sel_path.exists():
        try:
            from frame_selector import load_selected_frames
            selected_files = load_selected_frames(str(frames_dir))
            if selected_files:
                valid_filenames = selected_files
                use_novelty = True
                print(f"[SegPipeline] 🎯 Using visual novelty keyframes: {len(valid_filenames)} frames")
        except ImportError:
            print("[SegPipeline] ⚠️ frame_selector not available, falling back to stride")
    
    if not use_novelty:
        # ── Fallback: blur filter + stride ──
        fq_path = frames_dir / "frame_quality.json"
        if not fq_path.exists():
            print(f"[SegPipeline] No frame_quality.json found — using all frames")
            frame_files = sorted([
                f for f in os.listdir(frames_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ], key=lambda f: int(os.path.splitext(f)[0]))
            if frame_stride > 1:
                original = len(frame_files)
                frame_files = frame_files[::frame_stride]
                print(f"[SegPipeline] 📐 Frame stride {frame_stride}: {original} → {len(frame_files)} frames")
            return frames_dir, frame_files, None
        
        with open(fq_path) as f:
            fq_data = json.load(f)
        
        valid_filenames = sorted(
            [f["file"] for f in fq_data["frames"] if f["valid"]],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        
        if frame_stride > 1:
            original = len(valid_filenames)
            valid_filenames = valid_filenames[::frame_stride]
            print(f"[SegPipeline] 📐 Frame stride {frame_stride}: {original} → {len(valid_filenames)} valid frames")
    
    if not use_novelty:
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
    
    # Copy valid frames with sequential numbering (matching reconstruction's frame_global)
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

def _run_sam3_batched(frames_dir: Path, frame_files: List[str], categories: List[str],
                      batch_size: int, batch_overlap: int,
                      iou_threshold: float,
                      output_dir: Path = None, cfg: dict = None,
                      frame_map: dict = None,
                      on_progress=None) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, str]]:
    """
    Process frames in overlapping batches, one category at a time.
    Each category gets its own SAM3 pass; obj_ids are remapped to avoid collisions.
    Saves incrementally after each category if output_dir and cfg are provided.
    
    If frame_map is provided, each category only processes the frames listed
    for that category (from VLM analysis), creating a temp directory with
    consecutive numbering for SAM3 propagation.
    
    Returns:
        (all_masks, obj_labels)
        - all_masks: Dict[orig_frame_idx, {global_obj_id: binary_mask}]
        - obj_labels: Dict[global_obj_id, category_label]
    """
    from sam3_wrapper import get_sam3_wrapper
    
    total_frames = len(frame_files)
    batch_step = batch_size - batch_overlap
    
    # Build a lookup: frame filename → sequential index in frame_files
    frame_name_to_idx = {}
    for idx, fname in enumerate(frame_files):
        # Map both the sequential name (000000.jpg) and try to find original name
        frame_name_to_idx[fname] = idx
        # Also map without leading zeros for fuzzy matching
        base = os.path.splitext(fname)[0].lstrip('0') or '0'
        frame_name_to_idx[base] = idx
    
    print(f"[SegPipeline] Processing {total_frames} frames x {len(categories)} categories")
    if frame_map:
        print(f"[SegPipeline] VLM frame_map available for {len(frame_map)} categories")
    
    def _log_vram(label):
        """Diagnostic: log GPU memory state at a given point."""
        if not torch.cuda.is_available():
            return
        try:
            alloc = torch.cuda.memory_allocated() / (1024**3)
            resrv = torch.cuda.memory_reserved() / (1024**3)
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            print(f"[VRAM] {label}: alloc={alloc:.2f}GB  reserved={resrv:.2f}GB  "
                  f"driver_free={free_gb:.2f}GB  total={total_gb:.2f}GB")
        except Exception as e:
            print(f"[VRAM] {label}: error reading - {e}")
    
    sam3 = get_sam3_wrapper()
    
    # Master state across all categories
    all_masks = {}  # orig_frame_idx -> {global_obj_id: mask}
    obj_labels = {}  # global_obj_id -> category_label
    global_id_offset = 0  # Offset to remap IDs between categories
    
    for cat_idx, category in enumerate(categories):
        print(f"\n[SegPipeline] === Category {cat_idx+1}/{len(categories)}: '{category}' ===")
        _log_vram(f"cat {cat_idx+1} START")
        if on_progress:
            cat_pct = (cat_idx / max(len(categories), 1)) * 100
            on_progress(cat_pct, f"Processing category {cat_idx+1}/{len(categories)}: {category}")
        
        # ── Per-category frame selection from VLM frame_map ──
        cat_frame_files = frame_files  # Default: all frames
        cat_frame_indices = list(range(total_frames))  # Maps local index → original index in frame_files
        cat_label = category.split(";")[0].strip().lower() if ";" in category else category.strip().lower()
        
        if frame_map:
            # Find matching frame_map entry for this category
            matched_frames = None
            for map_label, map_frames in frame_map.items():
                if (map_label.lower() == cat_label or 
                    cat_label in map_label.lower() or 
                    map_label.lower() in cat_label):
                    matched_frames = map_frames
                    break
            
            if matched_frames and len(matched_frames) > 0:
                # Filter frame_files to only include those detected by VLM
                # matched_frames contains original filenames (e.g., "00012.jpg")
                # frame_files contains sequential filenames (e.g., "000000.jpg")
                # We need to find which sequential frames correspond to the VLM frames
                cat_frame_files = []
                cat_frame_indices = []
                
                for seq_idx, seq_fname in enumerate(frame_files):
                    # Check if this sequential frame corresponds to any VLM-detected frame
                    # The valid frames were renumbered sequentially, so we check by index
                    for vlm_fname in matched_frames:
                        vlm_base = os.path.splitext(vlm_fname)[0]
                        seq_base = os.path.splitext(seq_fname)[0]
                        # Direct match (same filename)
                        if vlm_fname == seq_fname or vlm_base == seq_base:
                            cat_frame_files.append(seq_fname)
                            cat_frame_indices.append(seq_idx)
                            break
                        # Index-based match: VLM frame "00012" matches sequential frame at position 12
                        try:
                            vlm_idx = int(vlm_base)
                            seq_idx_val = int(seq_base)
                            if vlm_idx == seq_idx_val:
                                cat_frame_files.append(seq_fname)
                                cat_frame_indices.append(seq_idx)
                                break
                        except ValueError:
                            pass
                
                if cat_frame_files:
                    pct_saved = (1 - len(cat_frame_files) / total_frames) * 100
                    print(f"[SegPipeline]   VLM frame_map: {len(cat_frame_files)}/{total_frames} frames "
                          f"({pct_saved:.0f}% saved)")
                else:
                    # VLM didn't find this category in any frame — skip entirely
                    print(f"[SegPipeline]   VLM frame_map: no matching frames found — skipping category")
                    continue
            else:
                # No VLM data at all for this category — skip
                print(f"[SegPipeline]   No VLM frame_map for '{cat_label}' — skipping category")
                continue
        
        # Compute batches for this category's frame subset
        cat_total = len(cat_frame_files)
        if cat_total == 0:
            print(f"[SegPipeline]   Skipping '{category}' — no frames in range")
            continue
            
        cat_batches = []
        s = 0
        while s < cat_total:
            e = min(s + batch_size, cat_total)
            cat_batches.append((s, e))
            if e >= cat_total:
                break
            s += batch_step
        
        def _process_category(category, batches, frames_dir, frame_files, sam3,
                              batch_size, batch_overlap, iou_threshold):
            """Process all batches for a single category. Raises on OOM."""
            batch_step = batch_size - batch_overlap
            cat_masks = {}
            next_global_id = 1
            prev_batch_masks = None
            prev_overlap_start = None
            
            for batch_idx, (b_start, b_end) in enumerate(batches):
                batch_frame_files = frame_files[b_start:b_end]
                batch_len = len(batch_frame_files)
                
                print(f"\n[SegPipeline] ── Batch {batch_idx}/{len(batches)-1}: "
                      f"frames {b_start}–{b_end-1} ({batch_len} frames) ──")
                if on_progress:
                    batch_pct = ((cat_idx * len(batches) + batch_idx + 1) / max(len(categories) * len(batches), 1)) * 100
                    on_progress(batch_pct, f"Batch {batch_idx+1}/{len(batches)} for '{category}'")
                _log_vram(f"  batch {batch_idx} BEFORE")
                
                batch_dir, index_mapping = _prepare_batch_dir(frames_dir, batch_frame_files, b_start)
                
                try:
                    raw_results = sam3.process_batch(
                        str(batch_dir), category, index_mapping
                    )
                    
                    if not raw_results:
                        print(f"[SegPipeline] Batch {batch_idx}: no masks produced")
                        continue
                    
                    batch_masks = _parse_raw_masks(raw_results)
                    
                    if not batch_masks:
                        print(f"[SegPipeline] Batch {batch_idx}: no valid masks after parsing")
                        continue
                    
                    if batch_idx == 0:
                        id_remap = {}
                        batch_obj_ids = set()
                        for frame_masks in batch_masks.values():
                            batch_obj_ids.update(frame_masks.keys())
                        for local_id in sorted(batch_obj_ids):
                            id_remap[local_id] = next_global_id
                            next_global_id += 1
                    else:
                        overlap_start_frame = b_start
                        overlap_end_frame = prev_overlap_start + batch_step + batch_overlap - 1 if prev_overlap_start is not None else b_start + batch_overlap - 1
                        
                        id_remap, next_global_id = _match_ids_iou(
                            prev_batch_masks, batch_masks,
                            overlap_start=overlap_start_frame,
                            overlap_end=min(overlap_end_frame, b_end - 1),
                            iou_threshold=iou_threshold,
                            next_global_id=next_global_id
                        )
                    
                    remapped_batch = {}
                    for orig_idx, frame_masks in batch_masks.items():
                        remapped = {}
                        for local_id, mask in frame_masks.items():
                            global_id = id_remap.get(local_id, local_id)
                            remapped[global_id] = mask
                        cat_masks[orig_idx] = {**cat_masks.get(orig_idx, {}), **remapped}
                        remapped_batch[orig_idx] = remapped
                    
                    prev_batch_masks = remapped_batch
                    prev_overlap_start = b_start
                    
                    unique_objects = set()
                    for fm in batch_masks.values():
                        unique_objects.update(fm.keys())
                    print(f"[SegPipeline] Batch {batch_idx}: {len(batch_masks)} frames, "
                          f"{len(unique_objects)} objects → remapped to {len(set(id_remap.values()))} global IDs")
                    _log_vram(f"  batch {batch_idx} AFTER")
                    
                finally:
                    shutil.rmtree(batch_dir, ignore_errors=True)
            
            return cat_masks
        
        def _recover_sam3(sam3):
            """Full SAM3 unload/reload cycle to recover from CUDA OOM."""
            print("[SegPipeline] 🔄 OOM detected — unloading SAM3 for recovery...")
            sam3.unload_model()
            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            _log_vram("  after OOM cleanup")
            print("[SegPipeline] 🔄 Reloading SAM3...")
            sam3.load_model()
            _log_vram("  after SAM3 reload")
        
        # ── Try category, recover from OOM once, skip if OOM again ──
        cat_masks = {}
        try:
            cat_masks = _process_category(
                category, cat_batches, frames_dir, cat_frame_files, sam3,
                batch_size, batch_overlap, iou_threshold
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                _recover_sam3(sam3)
                print(f"[SegPipeline] 🔄 Retrying category '{category}'...")
                try:
                    cat_masks = _process_category(
                        category, cat_batches, frames_dir, cat_frame_files, sam3,
                        batch_size, batch_overlap, iou_threshold
                    )
                except Exception as e2:
                    if "out of memory" in str(e2).lower():
                        print(f"[SegPipeline] ⛔ Category '{category}' failed twice with OOM — skipping")
                        _recover_sam3(sam3)
                    else:
                        print(f"[SegPipeline] ⚠️ Category '{category}' retry failed: {e2}")
            else:
                print(f"[SegPipeline] ⚠️ Category '{category}' failed: {e}")
        
        # Collect unique obj_ids for this category
        cat_obj_ids = set()
        for fm in cat_masks.values():
            cat_obj_ids.update(fm.keys())
        
        print(f"[SegPipeline] Category '{category}': {len(cat_obj_ids)} objects across {len(cat_masks)} frames")
        
        # Remap this category's IDs to global space (offset by previous categories)
        cat_id_remap = {}
        for local_id in sorted(cat_obj_ids):
            global_id = local_id + global_id_offset
            cat_id_remap[local_id] = global_id
            obj_labels[global_id] = category
        
        # Merge into all_masks with remapped global IDs
        # Map local category frame indices back to global frame_files indices
        for local_frame_idx, frame_masks in cat_masks.items():
            # local_frame_idx is an index into cat_frame_files
            # We need to map it back to the global frame_files index
            if local_frame_idx < len(cat_frame_indices):
                global_frame_idx = cat_frame_indices[local_frame_idx]
            else:
                global_frame_idx = local_frame_idx  # Fallback
            
            if global_frame_idx not in all_masks:
                all_masks[global_frame_idx] = {}
            for local_id, mask in frame_masks.items():
                all_masks[global_frame_idx][cat_id_remap[local_id]] = mask
        
        # Advance offset for next category
        if cat_obj_ids:
            global_id_offset = max(cat_id_remap.values())
        
        # Incremental save: persist results after each category
        if output_dir and cfg and all_masks:
            try:
                categories_so_far = categories[:cat_idx + 1]
                _save_masks(output_dir, all_masks, categories_so_far, obj_labels, cfg)
                print(f"[SegPipeline] 💾 Incremental save: {cat_idx+1}/{len(categories)} categories saved")
                # Only match against cloud if this category found new objects
                if cat_obj_ids:
                    new_global_ids = set(cat_id_remap.values())
                    if on_progress:
                        save_pct = ((cat_idx + 1) / max(len(categories), 1)) * 100
                        on_progress(save_pct, f"Matching masks to cloud ({cat_idx+1}/{len(categories)} categories)...")
                    _match_and_save_result(output_dir, new_obj_ids=new_global_ids)
                else:
                    print(f"[SegPipeline] ⏭️ No new objects — skipping cloud matching")
            except Exception as e:
                print(f"[SegPipeline] ⚠️ Incremental save failed: {e}")
        
        # VRAM cleanup between categories
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        _log_vram(f"cat {cat_idx+1} END (after cleanup)")
    
    # Unload SAM3 to free VRAM
    sam3.unload_model()
    gc.collect()
    
    total_objects = set()
    for fm in all_masks.values():
        total_objects.update(fm.keys())
    print(f"\n[SegPipeline] SAM3 complete: {len(all_masks)} frames, "
          f"{len(total_objects)} unique objects across {len(categories)} categories")
    
    return all_masks, obj_labels


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
    no_key_count = 0
    empty_mask_count = 0
    total_frames = len(raw_results)
    
    for frame_idx, outputs in raw_results.items():
        if "out_binary_masks" not in outputs:
            no_key_count += 1
            continue
        
        masks = outputs["out_binary_masks"]
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        
        # Squeeze singleton dimensions: (N,1,H,W) → (N,H,W) or (1,H,W) → (H,W)
        while masks.ndim > 3:
            masks = masks.squeeze(1)
        # Handle (1,H,W) → could be single object
        if masks.ndim == 1:
            continue
        
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
        else:
            empty_mask_count += 1
    
    # Debug logging
    if no_key_count > 0:
        print(f"[SegPipeline] _parse_raw_masks: {no_key_count}/{total_frames} frames had no 'out_binary_masks' key")
    if empty_mask_count > 0:
        print(f"[SegPipeline] _parse_raw_masks: {empty_mask_count}/{total_frames} frames had all-zero masks (SAM3 found nothing)")
    if total_frames > 0 and len(structured) == 0:
        # Log the first frame's mask shape for debugging
        first_key = next(iter(raw_results))
        first_out = raw_results[first_key]
        if "out_binary_masks" in first_out:
            m = first_out["out_binary_masks"]
            shape = m.shape if hasattr(m, 'shape') else 'N/A'
            dtype = m.dtype if hasattr(m, 'dtype') else 'N/A'
            print(f"[SegPipeline] _parse_raw_masks: 0 valid masks! First frame mask shape={shape}, dtype={dtype}")
    
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
    if prev_masks and curr_masks:
        for fidx in range(overlap_start, overlap_end + 1):
            if fidx in prev_masks and fidx in curr_masks:
                overlap_frames.append(fidx)
    
    if not overlap_frames:
        # No overlap — assign fresh IDs to all objects
        curr_obj_ids = set()
        for fm in (curr_masks or {}).values():
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
                categories: List[str], obj_labels: Dict[int, str], cfg: dict):
    """
    Save SAM3 masks as compressed NPZ + metadata JSON.
    Upsert logic: if an obj_id already exists in the NPZ (same object from a
    previous incremental save), keep its ID and overwrite its masks.
    If it's genuinely new, assign a new ID.
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
    
    # ── Load existing data ──
    masks_path = output_dir / "seg_masks.npz"
    seg_path = output_dir / "segmentation.json"
    
    existing_npz = {}
    existing_instances = []
    existing_prompts = []
    max_existing_id = -1
    existing_frames = set()
    existing_obj_ids = set()
    
    if masks_path.exists():
        try:
            old_data = np.load(masks_path)
            for key in old_data.files:
                if key.startswith("f") and "_o" in key:
                    existing_npz[key] = old_data[key]
            if "obj_ids" in old_data:
                existing_obj_ids = set(old_data["obj_ids"].tolist())
            if "frames" in old_data:
                existing_frames = set(old_data["frames"].tolist())
        except Exception as e:
            print(f"[SegPipeline] ⚠️ Could not load existing NPZ: {e}")
    
    if seg_path.exists():
        try:
            with open(seg_path) as f:
                old_meta = json.load(f)
            existing_instances = old_meta.get("instances", [])
            existing_prompts = old_meta.get("prompts", [])
            if not existing_prompts and old_meta.get("prompt"):
                existing_prompts = [old_meta["prompt"]]
            for inst in existing_instances:
                max_existing_id = max(max_existing_id, inst.get("id", -1))
            for oid in existing_obj_ids:
                max_existing_id = max(max_existing_id, oid)
        except Exception as e:
            print(f"[SegPipeline] ⚠️ Could not load existing metadata: {e}")
            existing_instances = []
            existing_prompts = []
    
    # ── Upsert: keep existing IDs, only remap genuinely new ones ──
    new_obj_ids_raw = set()
    for fm in all_masks.values():
        new_obj_ids_raw.update(fm.keys())
    new_obj_ids_raw = sorted(new_obj_ids_raw)
    
    id_remap = {}
    next_id = max_existing_id + 1
    reused = 0
    for raw_id in new_obj_ids_raw:
        if raw_id in existing_obj_ids:
            # Same object already saved — keep its ID, overwrite masks
            id_remap[raw_id] = raw_id
            reused += 1
        else:
            # Genuinely new object — assign new ID
            id_remap[raw_id] = next_id
            next_id += 1
    
    if reused > 0:
        print(f"[SegPipeline] Upsert: {reused} existing objects updated, "
              f"{len(new_obj_ids_raw) - reused} new objects added")
    
    # ── Build merged NPZ data ──
    # Start with existing masks
    npz_data = dict(existing_npz)
    
    # Add new masks with remapped IDs
    new_frame_indices = sorted(all_masks.keys())
    mask_count = len(existing_npz)
    
    for frame_idx, frame_masks in all_masks.items():
        for raw_obj_id, mask in frame_masks.items():
            remapped_id = id_remap[raw_obj_id]
            key = f"f{frame_idx}_o{remapped_id}"
            # Ensure mask is at scaled resolution
            if mask.shape[0] != scaled_res[0] or mask.shape[1] != scaled_res[1]:
                mask = cv2.resize(mask.astype(np.uint8),
                                 (scaled_res[1], scaled_res[0]),
                                 interpolation=cv2.INTER_NEAREST)
            npz_data[key] = mask.astype(np.uint8)
            mask_count += 1
    
    # Merge frame lists and obj_id lists
    all_frames = sorted(existing_frames | set(new_frame_indices))
    all_obj_ids = sorted(existing_obj_ids | set(id_remap.values()))
    
    npz_data["obj_ids"] = np.array(all_obj_ids, dtype=np.int32)
    npz_data["frames"] = np.array(all_frames, dtype=np.int32)
    npz_data["scaled_res"] = np.array(scaled_res, dtype=np.int32)
    
    # Save compressed NPZ
    np.savez_compressed(masks_path, **npz_data)
    masks_mb = masks_path.stat().st_size / (1024 * 1024)
    new_count = mask_count - len(existing_npz)
    print(f"[SegPipeline] ✅ Saved masks: {masks_path.name} "
          f"({new_count} new masks, {len(all_obj_ids)} total objects, "
          f"{len(all_frames)} frames, {masks_mb:.1f} MB)")
    
    # ── Build metadata JSON (upsert: update existing, add new) ──
    existing_by_id = {inst["id"]: inst for inst in existing_instances}
    
    max_existing_iid = 0
    for inst in existing_instances:
        max_existing_iid = max(max_existing_iid, inst.get("instance_id", 0))
    
    color_offset = len(existing_instances)
    new_count = 0
    for raw_id in sorted(new_obj_ids_raw):
        remapped_id = id_remap[raw_id]
        label = obj_labels.get(raw_id, categories[0] if categories else "object")
        
        if remapped_id in existing_by_id:
            # Update existing entry (label may have changed)
            existing_by_id[remapped_id]["label"] = label
        else:
            # New entry
            max_existing_iid += 1
            existing_by_id[remapped_id] = {
                "id": int(remapped_id),
                "label": label,
                "instance_id": max_existing_iid,
                "color": colors[(color_offset + new_count) % len(colors)],
            }
            new_count += 1
    
    all_instances = list(existing_by_id.values())
    
    # Track all prompts used
    all_prompts = list(existing_prompts)
    for cat in categories:
        if cat not in all_prompts:
            all_prompts.append(cat)
    
    segmentation = {
        "version": "3.0",
        "prompt": ";".join(categories),  # last prompt string used
        "prompts": all_prompts,  # all prompts ever used
        "resolution": {"scaled": scaled_res, "original": original_res},
        "instances": all_instances,
        "mask_file": "seg_masks.npz",
    }
    
    with open(seg_path, 'w') as f:
        json.dump(segmentation, f, indent=2)
    print(f"[SegPipeline] ✅ Saved metadata: {seg_path.name} "
          f"({new_count} new + {len(existing_instances)} existing = "
          f"{len(all_instances)} total instances)")
    
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


def _filter_instance_outliers(xyz: np.ndarray, indices: np.ndarray,
                               obj_id: int, min_samples: int = 10,
                               sor_k: int = 20, sor_std: float = 1.5) -> np.ndarray:
    """
    Remove outlier points from a segmented instance using DBSCAN + SOR.
    
    1. DBSCAN clusters the instance's 3D points (auto-calibrated eps)
    2. Keep only the largest cluster (removes satellite clusters)
    3. SOR refines within the cluster (removes borderline strays)
    
    Args:
        xyz: Full point cloud (N, 3) in display coordinates
        indices: Indices of points belonging to this instance
        obj_id: Object ID (for logging)
        min_samples: DBSCAN min_samples parameter
        sor_k: Number of neighbors for SOR
        sor_std: Standard deviation multiplier for SOR threshold
    
    Returns:
        Filtered indices array
    """
    points = xyz[indices]
    
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        
        # Auto-calibrate eps from k-NN distances (adapts to object density)
        k = min(min_samples, len(points) - 1)
        if k < 2:
            return indices
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, _ = nbrs.kneighbors(points)
        knn_dists = distances[:, -1]  # distance to k-th neighbor
        eps = np.percentile(knn_dists, 90)  # 90th percentile → robust eps
        
        # Step 1: DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            return indices  # all noise → keep originals
        
        largest_cluster = unique_labels[np.argmax(counts)]
        cluster_mask = labels == largest_cluster
        
        # Step 2: SOR on largest cluster
        cluster_points = points[cluster_mask]
        cluster_indices = indices[cluster_mask]
        
        if len(cluster_points) >= sor_k:
            nbrs2 = NearestNeighbors(n_neighbors=sor_k).fit(cluster_points)
            dists2, _ = nbrs2.kneighbors(cluster_points)
            mean_dists = np.mean(dists2[:, 1:], axis=1)
            threshold = np.mean(mean_dists) + sor_std * np.std(mean_dists)
            sor_mask = mean_dists < threshold
            cluster_indices = cluster_indices[sor_mask]
        
        removed = len(indices) - len(cluster_indices)
        if removed > 0:
            n_clusters = len(unique_labels)
            noise_count = np.sum(labels == -1)
            print(f"[SegPipeline]   DBSCAN obj {obj_id}: eps={eps:.4f}, "
                  f"{n_clusters} clusters, {noise_count} noise pts → "
                  f"removed {removed} outliers ({100*removed/len(indices):.1f}%)")
        
        return cluster_indices
        
    except ImportError:
        # Fallback: centroid + stddev filter
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        threshold = np.mean(dists) + 2 * np.std(dists)
        inlier_mask = dists < threshold
        return indices[inlier_mask]


def _match_masks_to_cloud(output_dir, ply_path=None, skip_filter_ids=None, only_obj_ids=None) -> dict:
    """
    Core processing: match SAM3 masks against PLY cloud with erosion,
    deconfliction, DBSCAN filtering, and OBB computation.
    
    This is CPU-intensive. Called once after segmentation to produce
    segmentation_result.json. Use apply_segmentation_to_cloud() for cached loading.
    
    Args:
        skip_filter_ids: Optional set of instance IDs whose DBSCAN/SOR filtering
                         was already done in a previous incremental run. Their
                         cached globalIndices will be reused as-is.
        only_obj_ids: Optional set of obj_ids to process. When set, only these
                      obj_ids will be matched against the cloud (for incremental
                      per-category matching). Other obj_ids are skipped entirely.
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
    
    # Apply SAME floor alignment the viewer uses (from saved transform)
    xyz_display = xyz  # default: use raw xyz
    transform_path = output_dir / "floor_transform.npz"
    if transform_path.exists():
        try:
            data = np.load(transform_path)
            s = float(data["s"])
            R = data["R"]
            t = data["t"]
            if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
                xyz_display = s * (xyz @ R.T) + t
                print(f"[SegPipeline]   Floor alignment loaded from {transform_path.name}")
        except Exception as e:
            print(f"[SegPipeline]   ⚠️ Could not load floor_transform.npz: {e}")
    else:
        # Fallback: compute alignment (for legacy sessions without saved transform)
        try:
            from alignment_manager import get_alignment_manager
            am = get_alignment_manager()
            s, R, t = am.compute_leveling_from_points(xyz)
            if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
                xyz_display = s * (xyz @ R.T) + t
                print(f"[SegPipeline]   Floor alignment computed (no saved transform)")
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
    # Uses erosion for tighter boundaries + deconfliction (each point → one obj_id)
    colors = cfg["visualization"]["segment_colors"]
    
    # Build lookup from obj_id → instance metadata (label, color)
    instance_meta = {}
    for inst in metadata.get("instances", []):
        instance_meta[inst["id"]] = inst
    
    # ── Filter out orphaned obj_ids (exist in NPZ but deleted from segmentation.json) ──
    # MUST happen before Phase 1: orphaned obj_ids in deconfliction would "steal" points
    # from valid objects, then get discarded in Phase 2, leaving those points unassigned.
    valid_obj_ids = [oid for oid in obj_ids if oid in instance_meta]
    if len(valid_obj_ids) < len(obj_ids):
        removed = len(obj_ids) - len(valid_obj_ids)
        print(f"[SegPipeline]   Skipping {removed} orphaned obj_ids (deleted from segmentation.json)")
        obj_ids = valid_obj_ids
    
    # Erosion kernel (configurable)
    erosion_iterations = 2
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # ── Phase 1: Match all objects, track per-point best assignment ──
    # For deconfliction: each point goes to the object with the smallest mask area
    point_obj_id = np.full(n_pts, -1, dtype=np.int32)     # winning obj_id per point
    point_mask_area = np.full(n_pts, np.inf, dtype=np.float64)  # smaller wins
    
    obj_mask_areas = {}  # obj_id → average mask area (for priority)
    
    for i, obj_id in enumerate(obj_ids):
        # Skip obj_ids not in the incremental set
        if only_obj_ids is not None and obj_id not in only_obj_ids:
            continue
        # Compute average mask area for this object (across all frames)
        frame_areas = []
        
        for cloud_frame, pt_indices in frame_groups.items():
            # Find nearest SAM3 keyframe
            nearest_kf = _find_nearest_keyframe(cloud_frame, keyframes)
            if nearest_kf is None:
                continue
            
            # Load mask for this frame+object
            mask_key = f"f{nearest_kf}_o{obj_id}"
            if mask_key not in masks_data:
                continue
            
            mask = masks_data[mask_key].astype(np.uint8)
            
            # ── Erosion: shrink mask edges for tighter boundaries ──
            # Adaptive: skip/reduce erosion for small masks to avoid eliminating them
            raw_area = float(np.sum(mask > 0))
            if raw_area < 2000:
                # Small object — no erosion (would shrink too much)
                pass
            elif raw_area < 10000:
                # Medium object — mild erosion (1 iteration)
                mask = cv2.erode(mask, erosion_kernel, iterations=1)
            else:
                # Large object — full erosion
                mask = cv2.erode(mask, erosion_kernel, iterations=erosion_iterations)
            mask = mask.astype(bool)
            
            mask_area = float(np.sum(mask))
            if mask_area == 0:
                continue
            frame_areas.append(mask_area)
            
            # Look up each point's pixel in the eroded mask
            rows = np.clip(pixel_row[pt_indices], 0, mask.shape[0] - 1)
            cols = np.clip(pixel_col[pt_indices], 0, mask.shape[1] - 1)
            in_mask = mask[rows, cols]
            
            matched = pt_indices[in_mask]
            
            # ── Deconfliction: assign point to smallest-mask object ──
            # Vectorized: only update points where this mask_area is smaller
            wins = mask_area < point_mask_area[matched]
            winning_pts = matched[wins]
            point_obj_id[winning_pts] = obj_id
            point_mask_area[winning_pts] = mask_area
        
        avg_area = np.mean(frame_areas) if frame_areas else 0
        obj_mask_areas[obj_id] = avg_area
    
    # ── Phase 2: Build instances by merging obj_ids with the same instance_id ──
    # Each logical object may have multiple obj_ids (one per batch), merge them.
    # Skip orphaned obj_ids that have no metadata in segmentation.json (deleted).
    from collections import defaultdict
    instance_groups = defaultdict(list)  # instance_id → [obj_id, ...]
    for obj_id in obj_ids:
        meta = instance_meta.get(obj_id)
        if meta is None:
            # Orphaned obj_id: exists in NPZ but was deleted from segmentation.json
            continue
        iid = meta.get("instance_id", obj_id)
        instance_groups[iid].append(obj_id)
    
    instances = []
    total_segmented = 0
    
    for iid, group_obj_ids in instance_groups.items():
        # Merge all points assigned to any obj_id in this instance group
        all_matched = np.where(np.isin(point_obj_id, group_obj_ids))[0].astype(np.int64)
        
        if len(all_matched) == 0:
            continue
        
        # ── Per-instance DBSCAN outlier removal ──
        # Skip if already filtered in a previous incremental run
        pre_filter_count = len(all_matched)
        if skip_filter_ids and iid in skip_filter_ids:
            # Reuse cached filtered indices (already passed DBSCAN+SOR)
            pass  # all_matched stays as-is from mask matching
        elif pre_filter_count >= 20:
            all_matched = _filter_instance_outliers(
                xyz_display, all_matched, iid
            )
        
        total_segmented += len(all_matched)
        
        # Look up label/color from the first obj_id's metadata
        meta_inst = instance_meta.get(group_obj_ids[0], {})
        label = meta_inst.get("label", "object")
        color = meta_inst.get("color", colors[len(instances) % len(colors)])
        
        # Build instance data
        instance = {
            "id": int(iid),
            "label": label,
            "instance_id": int(iid),
            "color": color,
            "total_points": int(len(all_matched)),
            "globalIndices": all_matched.tolist(),
        }
        
        # Compute OBB using floor-aligned coordinates
        if len(all_matched) >= 4:
            instance["obb"] = _compute_obb(xyz_display[all_matched])
        
        instances.append(instance)
        removed = pre_filter_count - len(all_matched)
        filter_info = f" (filtered {removed} outliers)" if removed > 0 else ""
        print(f"[SegPipeline]   Object '{label}' #{iid}: "
              f"{len(all_matched):,} points{filter_info}")
    
    # ── Phase 3: Cross-category Re-ID — merge instances with high 3D overlap ──
    # If VLM produced synonyms (e.g., "chair" + "wooden chair"), SAM3 may have
    # segmented the same physical object twice. Detect and merge by 3D point overlap.
    if len(instances) > 1:
        merge_threshold = 0.5  # If >50% of smaller set overlaps → merge
        merged_away = set()  # indices of instances absorbed by others
        
        for i in range(len(instances)):
            if i in merged_away:
                continue
            set_i = set(instances[i]["globalIndices"])
            
            for j in range(i + 1, len(instances)):
                if j in merged_away:
                    continue
                set_j = set(instances[j]["globalIndices"])
                
                intersection = len(set_i & set_j)
                if intersection == 0:
                    continue
                
                smaller_size = min(len(set_i), len(set_j))
                overlap_ratio = intersection / smaller_size
                
                if overlap_ratio >= merge_threshold:
                    # Merge: smaller into larger
                    if len(set_i) >= len(set_j):
                        # i absorbs j
                        set_i |= set_j
                        instances[i]["globalIndices"] = sorted(set_i)
                        instances[i]["total_points"] = len(set_i)
                        merged_away.add(j)
                        print(f"[SegPipeline]   🔗 Merged '{instances[j]['label']}' #{instances[j]['id']} "
                              f"into '{instances[i]['label']}' #{instances[i]['id']} "
                              f"(overlap={overlap_ratio:.0%})")
                    else:
                        # j absorbs i
                        set_j |= set_i
                        instances[j]["globalIndices"] = sorted(set_j)
                        instances[j]["total_points"] = len(set_j)
                        merged_away.add(i)
                        print(f"[SegPipeline]   🔗 Merged '{instances[i]['label']}' #{instances[i]['id']} "
                              f"into '{instances[j]['label']}' #{instances[j]['id']} "
                              f"(overlap={overlap_ratio:.0%})")
                        break  # i is merged away, stop inner loop
        
        if merged_away:
            pre_merge = len(instances)
            instances = [inst for idx, inst in enumerate(instances) if idx not in merged_away]
            # Recompute total_segmented and OBBs for merged instances
            total_segmented = 0
            for inst in instances:
                matched = np.array(inst["globalIndices"], dtype=np.int64)
                inst["total_points"] = len(matched)
                total_segmented += len(matched)
                if len(matched) >= 4:
                    inst["obb"] = _compute_obb(xyz_display[matched])
            print(f"[SegPipeline]   Cross-category merge: {pre_merge} → {len(instances)} instances")
    
    coverage = round(total_segmented / max(1, n_pts), 4)
    
    result = {
        "type": "segmentation",
        "version": "3.0",
        "prompt": metadata.get("prompt", ""),
        "prompts": metadata.get("prompts", [metadata.get("prompt", "")]),
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


def _match_and_save_result(output_dir, ply_path=None, new_obj_ids=None):
    """
    Run mask→cloud matching and save to segmentation_result.json.
    
    When new_obj_ids is provided (incremental mode), only matches those
    obj_ids against the cloud and merges them with the existing result.
    This avoids re-processing all previous instances after each category.
    """
    output_dir = Path(output_dir)
    result_path = output_dir / "segmentation_result.json"
    
    # Load previous result for incremental merge
    prev_instances = []
    prev_result = {}
    if result_path.exists():
        try:
            with open(result_path) as f:
                prev_result = json.load(f)
            prev_instances = prev_result.get("instances", [])
        except Exception:
            pass
    
    try:
        # Incremental: only match new category's objects
        result = _match_masks_to_cloud(
            output_dir, ply_path,
            only_obj_ids=new_obj_ids
        )
        
        if "error" in result or not result.get("instances"):
            # No new matches — keep previous result as-is
            if prev_instances:
                return prev_result
            return result
        
        new_instances = result["instances"]
        new_ids = {inst["id"] for inst in new_instances}
        
        # Merge: keep old instances (not replaced by new), add new
        merged = [inst for inst in prev_instances if inst["id"] not in new_ids]
        merged.extend(new_instances)
        
        # Recompute coverage from merged set
        total_pts = result.get("total_points", prev_result.get("total_points", 0))
        total_segmented = sum(inst.get("total_points", 0) for inst in merged)
        coverage = round(total_segmented / max(1, total_pts), 4)
        
        merged_result = {
            "type": "segmentation",
            "version": "3.0",
            "prompt": result.get("prompt", prev_result.get("prompt", "")),
            "prompts": result.get("prompts", prev_result.get("prompts", [])),
            "cloud_source": result.get("cloud_source", ""),
            "total_points": total_pts,
            "segmented_points": total_segmented,
            "coverage": coverage,
            "instances": merged,
            "resolution": result.get("resolution", {}),
        }
        
        with open(result_path, "w") as f:
            json.dump(merged_result, f)
        print(f"[SegPipeline] 💾 Saved segmentation_result.json "
              f"({len(merged)} instances, {coverage*100:.1f}% coverage)")
        return merged_result
    except Exception as e:
        print(f"[SegPipeline] ⚠️ Match-and-save failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "instances": []}


# Per-session lock to prevent parallel matching runs on the same output dir
import threading
_matching_locks: dict = {}  # output_dir_str -> threading.Lock
_matching_locks_guard = threading.Lock()

def _get_matching_lock(output_dir: Path) -> threading.Lock:
    """Get or create a lock for a specific session's output directory."""
    key = str(output_dir)
    with _matching_locks_guard:
        if key not in _matching_locks:
            _matching_locks[key] = threading.Lock()
        return _matching_locks[key]


def apply_segmentation_to_cloud(output_dir, ply_path=None) -> dict:
    """
    Load pre-computed segmentation result (instant) or fall back to
    full processing for backward compatibility with old sessions.
    
    Uses a per-session lock to prevent parallel matching runs when
    multiple callers (viewer WebSocket, /segments/ endpoint, etc.)
    request segmentation for the same session concurrently.
    """
    output_dir = Path(output_dir)
    
    # ── Fast path (no lock needed): cached result from segmentation time ──
    result_path = output_dir / "segmentation_result.json"
    transform_path = output_dir / "floor_transform.npz"
    # Invalidate cache if floor_transform.npz is newer (pipeline re-ran)
    # Note: gizmo alignment save touches segmentation_result.json to prevent
    # unnecessary DBSCAN recomputation in that case.
    if result_path.exists() and transform_path.exists():
        if transform_path.stat().st_mtime > result_path.stat().st_mtime:
            print(f"[SegPipeline] ⚠️ floor_transform.npz is newer than cache — invalidating")
            result_path.unlink()
    if result_path.exists():
        try:
            with open(result_path) as f:
                result = json.load(f)
            n_inst = len(result.get("instances", []))
            coverage = result.get("coverage", 0)
            print(f"[SegPipeline] ⚡ Loaded cached segmentation_result.json "
                  f"({n_inst} instances, {coverage*100:.1f}% coverage)")
            return result
        except Exception as e:
            print(f"[SegPipeline] ⚠️ Failed to load cached result: {e}")
    
    # ── Slow path: acquire per-session lock to prevent parallel matching ──
    lock = _get_matching_lock(output_dir)
    if not lock.acquire(blocking=False):
        # Another thread is already matching — wait for it to finish
        print(f"[SegPipeline] ⏳ Matching already in progress, waiting for result...")
        lock.acquire()  # Block until the other thread finishes
        lock.release()
        # The other thread should have cached the result — try loading it
        if result_path.exists():
            try:
                with open(result_path) as f:
                    result = json.load(f)
                print(f"[SegPipeline] ⚡ Loaded result from concurrent match")
                return result
            except Exception:
                pass
        return {"instances": [], "error": "Concurrent match produced no result"}
    
    try:
        print(f"[SegPipeline] No cached result, running full mask matching (will cache for next time)...")
        result = _match_masks_to_cloud(output_dir, ply_path)
        
        # Cache the result so next load is instant
        if "error" not in result and result.get("instances"):
            try:
                result_path = output_dir / "segmentation_result.json"
                with open(result_path, "w") as f:
                    json.dump(result, f)
                print(f"[SegPipeline] 💾 Cached result for instant future loads")
            except Exception as e:
                print(f"[SegPipeline] ⚠️ Failed to cache result: {e}")
        
        return result
    finally:
        lock.release()

