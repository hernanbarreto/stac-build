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
import re
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
                     frame_map: dict = None, on_progress=None,
                     boxes_map: dict = None) -> dict:
    """
    Full segmentation pipeline: batched SAM3 → IoU ID matching → mask-to-point mapping.

    Supports multiple categories separated by ';' (e.g., "sofa;cushion;table").
    Uses the same blur-filtered frame set as reconstruction to ensure frame_global indices match.
    Valid frames are copied to frames_valid/ with sequential numbering, then cleaned up.

    Args:
        frame_map: Optional dict mapping category label → list of frame filenames.
                   If provided, SAM3 only processes frames where each category was detected.
        boxes_map: Optional per-instance BOX SEEDS from the Phase 1 auto-prompter:
                   {label: {filename: [{"instance_id", "box_xywh"}, ...]}} with
                   normalized xywh. Boxes are fed to SAM3's detector pathway
                   together with the text prompt at the seeded frames, so
                   multiple same-label instances are seeded individually.
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
            boxes_map=boxes_map,
        )
        
        if not all_masks:
            print("[SegPipeline] ⚠️ SAM3 produced no masks")
            return {"error": "No masks generated", "instances": []}
        
        # ── Step 3: Save masks and metadata ──
        seg_meta = _save_masks(output_dir, all_masks, categories, obj_labels, cfg)
        
        # ── Step 4: Match masks to cloud and cache final result (ONCE) ──
        # In the anchored pipeline order (recon → vlm → sam3 → phase_r →
        # cloudcompy → tsdf) the cleaned cloud does not exist yet — the
        # mapping is DEFERRED to the cloudcompy stage, which calls
        # map_segmentation_to_cloud() after the (corrected) merge.
        if (output_dir / "cleaned_cloud.ply").exists():
            result = _match_and_save_result(output_dir)
        else:
            print("[SegPipeline] ⏭ No cleaned_cloud.ply yet — mask→cloud "
                  "mapping deferred to the cloudcompy stage")
            result = {"deferred_cloud_mapping": True, "instances": []}

        # (Step 5 removed: the ShapeR PKL export is gone — MeshFlow mesh
        # generation runs on demand via /api/segmentation/shape/export.)

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
    
    # selected_frames.json is the single source of truth (always written by map_worker
    # step 2 for every frames_selector mode: dino / stride / none). Consume it if present.
    if sel_path.exists():
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
            shutil.copyfile(str(src), str(dst))
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
                      on_progress=None,
                      boxes_map: dict = None) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, str]]:
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
    from segmentation.sam3_wrapper import get_sam3_wrapper
    
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
    # Start clean: a previous run that crashed mid-way leaves its reused session
    # and symlink dirs behind (both are released on the normal path below).
    sam3.release_batch_session()
    _clear_batch_dirs()

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
        
        # Per-instance box seeds for this category (Phase 1 auto-prompter):
        # {filename: [{"instance_id","box_xywh"}, ...]}, same label keys as
        # frame_map. Filled below into cat_boxes {cat-local position: [xywh]}.
        matched_boxes = None
        cat_boxes = {}
        if boxes_map:
            for map_label, per_file in boxes_map.items():
                if (map_label.lower() == cat_label or
                        cat_label in map_label.lower() or
                        map_label.lower() in cat_label):
                    matched_boxes = per_file
                    break

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
                        matched = vlm_fname == seq_fname or vlm_base == seq_base
                        if not matched:
                            # Index-based match: VLM "00012" ↔ sequential pos 12
                            try:
                                matched = int(vlm_base) == int(seq_base)
                            except ValueError:
                                matched = False
                        if matched:
                            cat_frame_files.append(seq_fname)
                            cat_frame_indices.append(seq_idx)
                            if matched_boxes and vlm_fname in matched_boxes:
                                cat_boxes[len(cat_frame_files) - 1] = [
                                    b["box_xywh"] for b in matched_boxes[vlm_fname]
                                    if b.get("box_xywh")]
                            break
                
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
                              batch_size, batch_overlap, iou_threshold,
                              boxes_by_pos=None):
            """Process all batches for a single category. Raises on OOM.
            boxes_by_pos: {category-local frame position: [xywh boxes]} from the
            Phase 1 auto-prompter — seeded into SAM3 alongside the text prompt."""
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

                # per-instance box seeds for this batch (local index space)
                batch_boxes = None
                if boxes_by_pos:
                    batch_boxes = {pos - b_start: bx for pos, bx in boxes_by_pos.items()
                                   if b_start <= pos < b_end and bx}
                    if batch_boxes:
                        print(f"[SegPipeline]   box seeds on {len(batch_boxes)} frame(s)")

                try:
                    raw_results = sam3.process_batch(
                        str(batch_dir), category, index_mapping,
                        boxes_by_local=batch_boxes or None,
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
                    # batch_dir is memoized and reused by the next concept — see
                    # _prepare_batch_dir. Freed by _clear_batch_dirs() at the end.
                    pass
            
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
                batch_size, batch_overlap, iou_threshold, boxes_by_pos=cat_boxes
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                _recover_sam3(sam3)
                print(f"[SegPipeline] 🔄 Retrying category '{category}'...")
                try:
                    cat_masks = _process_category(
                        category, cat_batches, frames_dir, cat_frame_files, sam3,
                        batch_size, batch_overlap, iou_threshold, boxes_by_pos=cat_boxes
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
        
        # Incremental save: persist MASKLETS after each category (cheap,
        # crash-safe). The mask→cloud matching + per-instance cleaning
        # (DBSCAN/SOR) runs ONCE at the end (Step 4) — running it per category
        # interleaved N full matching passes with the segmentation for no
        # benefit, and in the anchored pipeline order the cleaned cloud does
        # not even exist yet at this point.
        if output_dir and cfg and all_masks:
            try:
                categories_so_far = categories[:cat_idx + 1]
                _save_masks(output_dir, all_masks, categories_so_far, obj_labels, cfg)
                print(f"[SegPipeline] 💾 Incremental save: {cat_idx+1}/{len(categories)} categories saved")
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

    # The batch session and the symlink dirs were kept alive across concepts.
    try:
        sam3.release_batch_session()
    except Exception as e:  # noqa: BLE001
        print(f"[SegPipeline]   ⚠️ Could not release SAM3 session: {e}")
    _clear_batch_dirs()
    
    return all_masks, obj_labels


# Symlink dirs are keyed by their exact frame list and reused: every concept sees
# the SAME frames, and rebuilding the dir per concept also forced SAM3 to open a
# new session (new resource_path) and re-decode all of them. Cleared by
# _clear_batch_dirs() at the end of a segmentation run.
_BATCH_DIRS: Dict[tuple, Tuple[Path, Dict[int, int]]] = {}


def _prepare_batch_dir(frames_dir: Path, batch_files: List[str], 
                       start_idx: int) -> Tuple[Path, Dict[int, int]]:
    """
    Create (or reuse) a temp directory with sequentially numbered symlinks for a batch.
    
    Returns:
        (batch_dir, index_mapping) where index_mapping = {local_idx: original_frame_idx}
    """
    key = (str(frames_dir), tuple(batch_files))
    cached = _BATCH_DIRS.get(key)
    if cached is not None and cached[0].exists():
        return cached

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
    
    _BATCH_DIRS[key] = (batch_dir, index_mapping)
    return batch_dir, index_mapping


def _clear_batch_dirs():
    """Drop every memoized symlink dir (end of a segmentation run)."""
    for batch_dir, _ in _BATCH_DIRS.values():
        shutil.rmtree(batch_dir, ignore_errors=True)
    _BATCH_DIRS.clear()


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

    INVIOLABLE RULE (Phase R.1): cross-window re-identification happens
    EXCLUSIVELY here, by TRACKING CONTINUITY through the shared overlap frames.
    Matching instances by APPEARANCE between windows not connected by tracking
    is PROHIBITED: in tunnels/stations identical columns repeat every N metres
    and an appearance match creates false loop closures that destroy the
    reconstruction (perceptual aliasing). Objects with no overlap-frame IoU
    link get a FRESH global id — never a similarity-based merge.

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
    
    # Load resolution from chunk metadata (DA3/MapAnything backends)
    meta_files = sorted(output_dir.glob("chunk_*_meta.json"))
    scaled_res = None   # Will be detected from actual mask shape if not available
    original_res = None  # Will be detected from actual frames if not available
    if meta_files:
        with open(meta_files[0]) as f:
            meta = json.load(f)
            scaled_res = meta.get("scaled_resolution")
            original_res = meta.get("original_resolution")
    
    # Detect original resolution from frames on disk if not in metadata
    if original_res is None:
        frames_dir = output_dir.parent / "frames"
        if not frames_dir.exists():
            frames_dir = output_dir / "frames"
        if frames_dir.exists():
            sample_frames = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
            if sample_frames:
                sample_img = cv2.imread(str(sample_frames[0]))
                if sample_img is not None:
                    original_res = [sample_img.shape[0], sample_img.shape[1]]  # [H, W]
        if original_res is None:
            original_res = [720, 1280]  # Fallback only
            print(f"[SegPipeline] ⚠️ Could not detect original resolution, using fallback {original_res}")
    
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
    
    # If scaled_res not set (no chunk metadata), detect from first mask shape
    if scaled_res is None:
        for frame_masks in all_masks.values():
            for mask in frame_masks.values():
                scaled_res = [mask.shape[0], mask.shape[1]]
                break
            break
        if scaled_res is None:
            scaled_res = [original_res[0], original_res[1]]  # Use original as fallback
        print(f"[SegPipeline] Auto-detected mask resolution: {scaled_res[0]}x{scaled_res[1]}")
    
    for frame_idx, frame_masks in all_masks.items():
        for raw_obj_id, mask in frame_masks.items():
            remapped_id = id_remap[raw_obj_id]
            key = f"f{frame_idx}_o{remapped_id}"
            # Resize to scaled resolution only if needed (preserves native SAM3 output when no chunk metadata)
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
    
    # Save compressed NPZ — ATOMIC (tmp ending in .npz + replace): a crash
    # mid-write must never truncate the session's masks (2026-08-29)
    from segmentation.erase import _atomic_savez
    _atomic_savez(masks_path, npz_data)
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
        # Rich SAM3 concept phrases ("concrete support column") become compact
        # id-like labels here — the ONE place labels are persisted — so folder
        # names / JSON keys downstream never carry spaces.
        label = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")[:48] or "object"
        
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
    Dynamically reads the PLY header so it works regardless of extra fields.
    """
    _ply_type = {
        'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
        'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
    }
    try:
        with open(ply_path, 'rb') as f:
            n_pts = 0
            props = []
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n_pts = int(line.split()[-1])
                elif line.startswith('property') and n_pts > 0:
                    parts = line.split()
                    if len(parts) >= 3:
                        np_type = _ply_type.get(parts[1])
                        if np_type:
                            props.append((parts[2], np_type))
                elif line == 'end_header':
                    break

            prop_names = {p[0] for p in props}
            if 'frame_global' not in prop_names or n_pts == 0:
                return None

            dtype = np.dtype(props)
            data = np.frombuffer(f.read(), dtype=dtype)
            xyz = np.column_stack([data['x'], data['y'], data['z']])
            return xyz, data['frame_global'], data['pixel_row'], data['pixel_col']
    except Exception as e:
        print(f"[SegPipeline] Error loading PLY origins from {ply_path}: {e}")
        return None

def _write_corrected_ply(src_path: Path, dst_path: Path, xyz_corrected: np.ndarray):
    """Write a corrected PLY by replacing xyz in the original binary PLY.
    Preserves all other fields (colors, normals, confidence, origins).
    """
    _ply_type = {
        'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
        'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
    }
    with open(src_path, 'rb') as f:
        header_lines = []
        n_pts = 0
        props = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line.startswith('element vertex'):
                n_pts = int(line.split()[-1])
            elif line.startswith('property') and n_pts > 0:
                parts = line.split()
                if len(parts) >= 3:
                    np_type = _ply_type.get(parts[1])
                    if np_type:
                        props.append((parts[2], np_type))
            elif line == 'end_header':
                break
        
        dtype = np.dtype(props)
        data = np.frombuffer(f.read(), dtype=dtype).copy()
    
    # Replace xyz
    data['x'] = xyz_corrected[:, 0].astype(data['x'].dtype)
    data['y'] = xyz_corrected[:, 1].astype(data['y'].dtype)
    data['z'] = xyz_corrected[:, 2].astype(data['z'].dtype)
    
    # Write corrected PLY
    header = '\n'.join(header_lines) + '\n'
    with open(dst_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
    
    print(f"[SegPipeline] Wrote corrected cloud: {dst_path.name} ({n_pts:,} pts)")


def _compute_obb(points_xyz: np.ndarray, face_normals=None) -> dict:
    """Compute minimum Oriented Bounding Box for floor-aligned coordinates.
    
    If face_normals is provided (list of (normal, n_points) tuples from RANSAC),
    uses the dominant face normal to orient the OBB on the XZ plane.
    Otherwise falls back to convex hull + rotating calipers.
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
    
    # Project to XZ plane for 2D bounding rectangle
    pts_xz = points_xyz[:, [0, 2]]  # (N, 2): [x, z]
    
    best_angle = 0.0
    
    if face_normals and len(face_normals) > 0:
        # Use dominant face normal to orient the OBB
        # Find face with most points
        dominant_normal = max(face_normals, key=lambda fn: fn[1])[0]
        
        # Project face normal to XZ plane (ignore Y component)
        nxz = np.array([dominant_normal[0], dominant_normal[2]])
        nxz_len = np.linalg.norm(nxz)
        if nxz_len > 0.1:  # face has meaningful XZ component
            nxz = nxz / nxz_len
            # Angle of the face normal in XZ (the OBB aligns PERPENDICULAR to the face)
            best_angle = np.arctan2(nxz[1], nxz[0])
    else:
        # Fallback: convex hull + rotating calipers
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts_xz)
            hull_pts = pts_xz[hull.vertices]
        except Exception:
            hull_pts = pts_xz
        
        n_hull = len(hull_pts)
        best_area = float('inf')
        
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
    
    # Compute extents using best_angle
    cos_a = np.cos(-best_angle)
    sin_a = np.sin(-best_angle)
    rot2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = pts_xz @ rot2d.T
    rmin = rotated.min(axis=0)
    rmax = rotated.max(axis=0)
    
    half_x = (rmax[0] - rmin[0]) / 2.0
    half_z = (rmax[1] - rmin[1]) / 2.0
    
    # Center in rotated 2D space → world XZ
    cx_rot = (rmax[0] + rmin[0]) / 2.0
    cz_rot = (rmax[1] + rmin[1]) / 2.0
    
    cos_back = np.cos(best_angle)
    sin_back = np.sin(best_angle)
    rot_back = np.array([[cos_back, -sin_back], [sin_back, cos_back]])
    center_xz = rot_back @ np.array([cx_rot, cz_rot])
    
    center = [float(center_xz[0]), float(cy), float(center_xz[1])]
    half_extents = [float(half_x), float(half_y), float(half_z)]
    
    # Rotation matrix: Y-axis rotation by best_angle
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


def _find_nearest_keyframe(frame_idx: int, keyframes: list, max_dist: int = 5) -> Optional[int]:
    """Find the nearest keyframe to a given frame index, within max_dist."""
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
    
    if best_dist <= max_dist:
        return best
    return None


def _clean_segment_subcloud(xyz: np.ndarray, indices: np.ndarray,
                             obj_id: int) -> tuple:
    """
    Clean a segmented sub-cloud using voxel grid + local PCA + depth peeling.
    
    Pipeline:
      1. DBSCAN → keep largest cluster (remove satellite misattributions)
      2. Subdivide into voxel grid
      3. Per-voxel local PCA → classify as planar or complex
      4. Planar voxels: depth peel along local normal (remove onion layers)
      5. Non-planar voxels: SOR fallback (statistical outlier removal)
    
    All parameters are read from config.yaml (models.segmentation.segment_cleaning).
    
    Args:
        xyz: Full point cloud (N, 3) in display coordinates
        indices: Indices of points belonging to this instance
        obj_id: Object ID (for logging)
    
    Returns:
        Tuple of (filtered_indices, voxel_data) where voxel_data is a list of
        [cx, cy, cz, nx, ny, nz] for each planar voxel
    """
    from config import get_param
    
    cfg_prefix = 'models.segmentation.segment_cleaning'
    enabled = get_param(f'{cfg_prefix}.enabled', True)
    if not enabled:
        return indices, [], [], [], np.array([], dtype=np.int32)
    
    voxel_size = get_param(f'{cfg_prefix}.voxel_size', 0.05)
    min_pts_voxel = get_param(f'{cfg_prefix}.min_points_per_voxel', 10)
    planarity_thresh = get_param(f'{cfg_prefix}.planarity_threshold', 0.3)
    layer_tol = get_param(f'{cfg_prefix}.layer_tolerance', 0.005)
    hist_bins = get_param(f'{cfg_prefix}.histogram_bins', 50)
    dbscan_enabled = get_param(f'{cfg_prefix}.dbscan_enabled', True)
    dbscan_min_samples = get_param(f'{cfg_prefix}.dbscan_min_samples', 10)
    sor_k = get_param(f'{cfg_prefix}.sor_k', 20)
    sor_std = get_param(f'{cfg_prefix}.sor_std', 1.5)
    mad_mult = get_param(f'{cfg_prefix}.mad_multiplier', 3.0)
    
    points = xyz[indices]
    n_original = len(indices)
    _zr = lambda p: f"Z:[{p[:,2].min():.3f},{p[:,2].max():.3f}]" if len(p)>0 else "Z:empty"
    print(f"[SegPipeline]     step0 RAW: {len(indices):,} pts {_zr(points)}")
    
    # ── Step 1: DBSCAN → remove noise/tiny clusters (optional) ──
    dbscan_removed = 0
    if dbscan_enabled:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors

            k = min(dbscan_min_samples, len(points) - 1)
            if k < 2:
                return indices, [], [], [], np.array([], dtype=np.int32)

            # For very large sub-clouds, DBSCAN over every point is O(n^2) in time
            # and memory: a 13M-point object (e.g. a whole train) pins a core for
            # tens of minutes at >15GB RAM and often OOMs. Downsample to one
            # representative per voxel, cluster the representatives, then propagate
            # each voxel's verdict (cluster vs noise) back to all its points. Same
            # satellite-removal behaviour, but scales to any object size.
            DBSCAN_MAX_PTS = 1_500_000
            ds = len(points) > DBSCAN_MAX_PTS
            if ds:
                ds_vox = max(voxel_size, 0.05)
                vkeys = np.floor(points / ds_vox).astype(np.int64)
                _, inv = np.unique(vkeys, axis=0, return_inverse=True)
                n_vox = int(inv.max()) + 1
                cloud = np.zeros((n_vox, 3), dtype=np.float64)
                np.add.at(cloud, inv, points)
                cloud /= np.bincount(inv, minlength=n_vox)[:, None]
            else:
                cloud = points

            kk = min(dbscan_min_samples, len(cloud) - 1)
            if kk < 2:
                print(f"[SegPipeline]     step1 DBSCAN: skipped (only {len(cloud):,} clusterable)")
            else:
                nbrs = NearestNeighbors(n_neighbors=kk).fit(cloud)
                distances, _ = nbrs.kneighbors(cloud)
                eps = np.percentile(distances[:, -1], 90)

                labels = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit(cloud).labels_
                if not np.any(labels >= 0):
                    return indices, [], [], [], np.array([], dtype=np.int32)

                # Keep ALL clusters, only remove noise (label == -1). When
                # downsampled, map the per-voxel labels back to every point.
                point_labels = labels[inv] if ds else labels
                cluster_mask = point_labels >= 0

                dbscan_removed = int(np.sum(~cluster_mask))
                indices = indices[cluster_mask]
                points = points[cluster_mask]
                tag = f"voxel-ds {len(cloud):,} reps @ {ds_vox:.2f}m → " if ds else ""
                print(f"[SegPipeline]     step1 DBSCAN: {tag}{len(indices):,} pts "
                      f"(-{dbscan_removed}) {_zr(points)}")
        except ImportError:
            pass
    
    if len(points) < 20:
        return indices, [], [], [], np.array([], dtype=np.int32)
    
    # ── Check if RANSAC face detection is enabled ──
    ransac_enabled = get_param(f'{cfg_prefix}.ransac_enabled', True)
    if not ransac_enabled:
        total_removed = n_original - len(indices)
        print(f"[SegPipeline]   Clean obj {obj_id}: "
              f"{n_original:,} → {len(indices):,} pts "
              f"(DBSCAN -{dbscan_removed}, RANSAC disabled)")
        return indices, [], [], [], np.array([], dtype=np.int32)
    
    # ── Load RANSAC parameters ──
    ransac_tol = get_param(f'{cfg_prefix}.ransac_tolerance', 0.01)
    min_face_pts = get_param(f'{cfg_prefix}.min_face_points', 100)
    max_faces = get_param(f'{cfg_prefix}.max_faces', 8)
    face_thick = get_param(f'{cfg_prefix}.face_thickness', 0.01)
    
    # ── Step 2: RANSAC iterative plane detection ──
    remaining_mask = np.ones(len(points), dtype=bool)
    faces = []
    
    for face_i in range(max_faces):
        remaining_idx = np.where(remaining_mask)[0]
        if len(remaining_idx) < min_face_pts:
            break
        
        rem_pts = points[remaining_idx]
        
        best_inlier_count = 0
        best_normal = None
        best_d = 0.0
        n_iters = min(500, max(50, len(rem_pts) // 10))
        
        for _ in range(n_iters):
            sample_idx = np.random.choice(len(rem_pts), 3, replace=False)
            p0, p1, p2 = rem_pts[sample_idx]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue
            normal = normal / norm_len
            d = -np.dot(normal, p0)
            dists = np.abs(rem_pts @ normal + d)
            n_inliers = int(np.sum(dists < ransac_tol))
            if n_inliers > best_inlier_count:
                best_inlier_count = n_inliers
                best_normal = normal
                best_d = d
        
        if best_inlier_count < min_face_pts:
            break
        
        # Refine plane using all inliers via PCA
        rem_dists = np.abs(rem_pts @ best_normal + best_d)
        inlier_local = rem_dists < ransac_tol
        inlier_pts = rem_pts[inlier_local]
        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        cov = (centered.T @ centered) / len(inlier_pts)
        _, eigvecs = np.linalg.eigh(cov)
        refined_normal = eigvecs[:, 0]
        refined_d = -np.dot(refined_normal, centroid)
        
        rem_dists_refined = np.abs(rem_pts @ refined_normal + refined_d)
        inlier_local_refined = rem_dists_refined < ransac_tol
        inlier_global_idx = remaining_idx[inlier_local_refined]
        faces.append((refined_normal, refined_d, inlier_global_idx))
        remaining_mask[inlier_global_idx] = False
        
        print(f"[SegPipeline]     face {face_i}: {len(inlier_global_idx)} pts, "
              f"normal=[{refined_normal[0]:.2f},{refined_normal[1]:.2f},{refined_normal[2]:.2f}]")
    
    # ── Step 2b: Merge parallel faces (collapse onion layers) ──
    # If two faces have near-parallel normals (|dot| > 0.95), merge into one
    if len(faces) > 1:
        merged = []
        used = set()
        for i in range(len(faces)):
            if i in used:
                continue
            n_i, d_i, idx_i = faces[i]
            group_faces = [(n_i, d_i, idx_i)]
            group_idx = [idx_i]
            for j in range(i + 1, len(faces)):
                if j in used:
                    continue
                n_j = faces[j][0]
                dot = abs(np.dot(n_i, n_j))
                if dot > 0.95:  # near-parallel → same surface
                    group_faces.append(faces[j])
                    group_idx.append(faces[j][2])
                    used.add(j)
            
            # Merge all grouped indices
            combined_idx = np.concatenate(group_idx) if len(group_idx) > 1 else idx_i
            
            # Use the DOMINANT face's plane (most inlier points)
            # This ensures all parallel planes converge to the actual
            # front surface, not the average of front+back.
            dominant = max(group_faces, key=lambda f: len(f[2]))
            merged_normal = dominant[0].copy()
            
            # Recompute d using dominant face's inlier centroid
            dom_centroid = np.mean(points[dominant[2]], axis=0)
            merged_d = -np.dot(merged_normal, dom_centroid)

            
            merged.append((merged_normal, merged_d, combined_idx))
        
        print(f"[SegPipeline]     merged: {len(faces)} → {len(merged)} faces")
        faces = merged
    
    # ── Step 3: Assign ALL points to nearest face (non-destructive) ──
    # No points are removed — every post-DBSCAN point is kept.
    # Each point is assigned to the face whose plane is closest.
    result_indices = indices  # keep ALL points
    
    local_face_id = np.full(len(points), -1, dtype=np.int32)
    
    if faces:
        # Compute distance of each point to each face plane
        n_faces = len(faces)
        all_dists = np.full((len(points), n_faces), np.inf)
        for fi, (face_normal, face_d, face_idx) in enumerate(faces):
            all_dists[:, fi] = np.abs(points @ face_normal + face_d)
        
        # Assign each point to the nearest face
        nearest_face = np.argmin(all_dists, axis=1)
        nearest_dist = all_dists[np.arange(len(points)), nearest_face]
        
        if n_faces == 1:
            # Single face: assign ALL points (object is one surface)
            local_face_id[:] = 0
        else:
            # Multi-face: generous threshold to catch onion layers
            max_assign_dist = 0.10  # 10cm
            assign_mask = nearest_dist <= max_assign_dist
            local_face_id[assign_mask] = nearest_face[assign_mask]
    
    result_face_id = local_face_id
    
    total_removed = n_original - len(result_indices)
    n_assigned = int(np.sum(local_face_id >= 0))
    n_residual = int(np.sum(local_face_id < 0))
    
    print(f"[SegPipeline]   Clean obj {obj_id}: "
          f"{n_original:,} → {len(result_indices):,} pts "
          f"(DBSCAN -{dbscan_removed}, "
          f"{len(faces)} faces, {n_assigned} assigned, {n_residual} residual)")
    


    # ── Step 6: Generate voxel mesh data from detected faces ──
    # Use larger voxels for visualization (5cm) — independent of cleaning voxel_size
    voxel_data = []
    mesh_vs = 0.05  # 5cm visualization voxels
    if len(result_indices) >= 5 and faces:
        final_pts = xyz[result_indices]
        
        # Voxelize at 5cm for the mesh
        fv_keys = np.floor(final_pts / mesh_vs).astype(np.int64)
        fv_ids = fv_keys[:, 0] * 1_000_003 + fv_keys[:, 1] * 1_000_033 + fv_keys[:, 2]
        
        # Pass 1: compute face or PCA for each voxel
        voxel_info = {}  # grid_key → {centroid, face_fi, normal, pca_normal}
        for fv_id in np.unique(fv_ids):
            fv_mask = fv_ids == fv_id
            fv_pts = final_pts[fv_mask]
            if len(fv_pts) < 3:
                continue
            centroid = np.mean(fv_pts, axis=0)
            gk = tuple(np.floor(centroid / mesh_vs).astype(int))
            
            # Majority face
            vox_face_ids = result_face_id[fv_mask]
            face_ids_in_vox = vox_face_ids[vox_face_ids >= 0]
            if len(face_ids_in_vox) > 0:
                majority_fi = int(np.bincount(face_ids_in_vox).argmax())
                voxel_info[gk] = {'centroid': centroid, 'face_fi': majority_fi, 'normal': faces[majority_fi][0]}
            else:
                # PCA for residual
                centered = fv_pts - centroid
                cov = (centered.T @ centered) / len(fv_pts)
                eigenvalues = np.linalg.eigvalsh(cov)
                if max(eigenvalues[0], 1e-12) / max(eigenvalues[2], 1e-12) < planarity_thresh:
                    _, eigvecs = np.linalg.eigh(cov)
                    pca_normal = eigvecs[:, 0]
                    voxel_info[gk] = {'centroid': centroid, 'face_fi': -1, 'normal': pca_normal}
        
        # Pass 2: flood-fill residual voxels to neighbor faces
        # Check 26 neighbors; if a neighbor has a face and PCA normal is compatible, adopt it
        neighbor_offsets_26 = [(dx, dy, dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz) != (0,0,0)]
        changed = True
        while changed:
            changed = False
            for gk, info in list(voxel_info.items()):
                if info['face_fi'] >= 0:
                    continue  # already assigned
                for ox, oy, oz in neighbor_offsets_26:
                    nk = (gk[0]+ox, gk[1]+oy, gk[2]+oz)
                    nb = voxel_info.get(nk)
                    if nb and nb['face_fi'] >= 0:
                        # Check normal compatibility
                        dot = abs(np.dot(info['normal'], nb['normal']))
                        if dot > 0.8:
                            info['face_fi'] = nb['face_fi']
                            info['normal'] = faces[nb['face_fi']][0]
                            changed = True
                            break
        
        # Build final voxel_data with snapping
        for gk, info in voxel_info.items():
            centroid = info['centroid']
            normal = info['normal']
            fi = info['face_fi']
            
            if fi >= 0:
                # Snap to face plane
                face_d = faces[fi][1]
                dist_to_plane = np.dot(centroid, normal) + face_d
                centroid = centroid - dist_to_plane * normal
            
            voxel_data.append([
                float(centroid[0]), float(centroid[1]), float(centroid[2]),
                float(normal[0]), float(normal[1]), float(normal[2])
            ])
    # Build face_normals summary for OBB: [(normal, n_points), ...]
    face_normals_summary = [(fn, len(fi)) for fn, fd, fi in faces]
    
    # Face planes for point projection: [(normal, d), ...]
    face_planes = [(fn, fd) for fn, fd, fi in faces]
    
    return result_indices, voxel_data, face_normals_summary, face_planes, local_face_id


def _attach_unsegmented(instances, xyz_display: np.ndarray,
                        attach_dist_m: float = 0.03):
    """Attach unsegmented cloud points to the NEAREST instance whose existing
    points lie within ``attach_dist_m`` (see the call site for the doctrine).
    Returns (n_attached, grown_instance_positions). Conflicts resolve by
    smallest distance across instances."""
    from scipy.spatial import cKDTree

    N = len(xyz_display)
    owner = np.full(N, -1, dtype=np.int64)
    for k, inst in enumerate(instances):
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        owner[gi[(gi >= 0) & (gi < N)]] = k
    un_idx = np.nonzero(owner < 0)[0]
    if not len(un_idx):
        return 0, []
    un_pts = xyz_display[un_idx]
    best_d = np.full(len(un_idx), np.inf)
    best_k = np.full(len(un_idx), -1, dtype=np.int64)
    for k, inst in enumerate(instances):
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < N)]
        if len(gi) < 50:
            continue
        sub = gi[::max(1, len(gi) // 400_000)]
        d, _ = cKDTree(xyz_display[sub]).query(
            un_pts, k=1, distance_upper_bound=float(attach_dist_m))
        better = d < best_d
        best_d[better] = d[better]
        best_k[better] = k
    hit = best_k >= 0
    grown = []
    for k in range(len(instances)):
        add = un_idx[hit & (best_k == k)]
        if not len(add):
            continue
        gi = np.asarray(instances[k].get("globalIndices") or [], dtype=np.int64)
        merged = np.union1d(gi, add)
        instances[k]["globalIndices"] = merged.tolist()
        instances[k]["total_points"] = int(len(merged))
        grown.append(k)
        print(f"[SegPipeline]     📎 '{instances[k].get('label')}' "
              f"#{instances[k].get('id')}: +{len(add):,} pts")
    return int(hit.sum()), grown


def _enforce_exclusive_ownership(instances: list, n_pts: int) -> int:
    """INVARIANT (USER 2026-08-31): each cloud point is owned by exactly one
    instance (or unsegmented). Walks instances smallest-first so specific
    objects keep contested points and big surfaces lose them; also drops
    internal duplicates. Mutates ``instances`` in place; returns how many
    double-ownerships were resolved."""
    owner = np.full(int(n_pts), -1, dtype=np.int64)
    order = sorted(range(len(instances)),
                   key=lambda k: len(instances[k].get("globalIndices") or []))
    resolved = 0
    for k in order:
        inst = instances[k]
        raw = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = np.unique(raw[(raw >= 0) & (raw < n_pts)])
        free = owner[gi] < 0
        n_lost = int((~free).sum())
        if n_lost:
            resolved += n_lost
            print(f"[SegPipeline]   ⚠ exclusivity: '{inst.get('label')}' "
                  f"#{inst.get('instance_id', inst.get('id'))} released "
                  f"{n_lost:,} point(s) already owned by another instance")
        kept = gi[free]
        owner[kept] = k
        if len(kept) != len(raw):
            inst["globalIndices"] = kept.tolist()
            inst["total_points"] = int(len(kept))
    return resolved


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
        print(f"[SegPipeline] ⚠️ PLY {ply_path.name} has no origin fields. Falling back to 2D-only instances.")
        
        # Build lookup from obj_id → instance metadata (label, color)
        instance_meta = {}
        for inst in metadata.get("instances", []):
            instance_meta[inst["id"]] = inst
            
        # Group obj_ids by instance_id
        from collections import defaultdict
        instance_groups = defaultdict(list)
        for obj_id in obj_ids:
            meta = instance_meta.get(obj_id)
            if meta:
                iid = meta.get("instance_id", obj_id)
                instance_groups[iid].append(obj_id)
                
        colors = cfg["visualization"]["segment_colors"]
        dummy_instances = []
        for iid, group_obj_ids in instance_groups.items():
            meta_inst = instance_meta.get(group_obj_ids[0], {})
            label = meta_inst.get("label", "object")
            color = meta_inst.get("color", colors[len(dummy_instances) % len(colors)])
            dummy_instances.append({
                "id": int(iid),
                "label": label,
                "instance_id": int(iid),
                "color": color,
                "total_points": 0,
                "globalIndices": [],
            })
            
        return {"warning": "PLY has no origins (2D only)", "instances": dummy_instances}
    
    xyz, frame_global, pixel_row, pixel_col = origins
    n_pts = len(frame_global)
    cloud_label = ply_path.stem
    print(f"[SegPipeline] Matching masks against {cloud_label} ({n_pts:,} points)...")
    
    # Apply SAME floor alignment the viewer uses (from saved transform)
    xyz_display = xyz  # default: use raw xyz
    s, R, t = 1.0, np.eye(3), np.zeros(3)  # identity transform defaults
    transform_path = output_dir / "floor_transform.npz"
    # PRECEDENCE (fixed 2026-08-28): a saved floor_transform.npz ALWAYS wins —
    # the viewer applies it to the cloud unconditionally (potree_ready), and
    # level_floor / the alignment gizmo compose their deltas into it EVEN on
    # sessions with baked orientation (fine floor snap). The old order
    # (.orientation_applied → identity, npz ignored) computed freshly matched
    # OBBs in the RAW frame while the viewer showed the leveled cloud: every
    # box displaced by exactly the leveling delta (test3, 2026-08-28).
    # `.orientation_applied` now only suppresses the legacy auto-compute
    # fallback, which WOULD double-rotate a baked cloud.
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
            s, R, t = 1.0, np.eye(3), np.zeros(3)
    elif (output_dir / ".orientation_applied").exists():
        # reconstruction/orient.py baked +Y up and the floor at y=0 into the cloud
        # itself, measured from the camera-pose gravity over every frame. The raw
        # cloud IS the display frame — any further leveling would rotate it a second
        # time and _compute_obb's Y-up assumption would then hold in no frame at all.
        print("[SegPipeline]   Orientation baked from camera poses — display frame is identity")
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
    # USER ORDER 2026-08-29: erosion OFF by default — it was the #1 point
    # eater on test3 (131k mask-covered points excluded, 35.7% of the
    # unsegmented). Re-enable via segmentation.mask_erosion_iterations if
    # boundary bleed (masks claiming the neighbour's points) returns.
    try:
        erosion_iterations = int((cfg.get("segmentation", {}) or {})
                                 .get("mask_erosion_iterations", 0))
    except Exception:
        erosion_iterations = 0
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # ── Phase 1: Match all objects, track per-point best assignment ──
    # For deconfliction: each point goes to the object with the smallest mask area
    point_obj_id = np.full(n_pts, -1, dtype=np.int32)     # winning obj_id per point
    point_mask_area = np.full(n_pts, np.inf, dtype=np.float64)  # smaller wins
    
    obj_mask_areas = {}  # obj_id → average mask area (for priority)
    
    # Precompute original image resolution for pixel coord scaling
    # Sources (priority): 1) chunk metadata, 2) actual frame dimensions, 3) pixel coord max
    orig_h, orig_w = None, None
    
    # Try 1: chunk metadata (DA3, MapAnything backends)
    meta_files = sorted(output_dir.glob("chunk_*_meta.json"))
    if meta_files:
        try:
            with open(meta_files[0]) as f:
                chunk_meta = json.load(f)
            orig_res = chunk_meta.get("original_resolution")
            if orig_res:
                orig_h, orig_w = float(orig_res[0]), float(orig_res[1])  # [H, W]
        except Exception:
            pass
    
    # Try 2: read actual frame image from disk
    if orig_h is None:
        frames_dir = output_dir.parent / "frames"
        if not frames_dir.exists():
            frames_dir = output_dir / "frames"
        if frames_dir.exists():
            sample_frames = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
            if sample_frames:
                try:
                    sample_img = cv2.imread(str(sample_frames[0]))
                    if sample_img is not None:
                        orig_h, orig_w = float(sample_img.shape[0]), float(sample_img.shape[1])
                except Exception:
                    pass
    
    # Fallback 3: pixel coord max (last resort, works when coords span full image)
    if orig_h is None:
        orig_h = float(pixel_row.max() + 1)
        orig_w = float(pixel_col.max() + 1)
        print(f"[SegPipeline]   ⚠️ Original resolution estimated from pixel coords (fallback)")

    # ── Consistency guard ────────────────────────────────────────────────
    # pixel_row/pixel_col are the GROUND TRUTH of the projection space: the cloud
    # is projected at the reconstruction backend's resolution (e.g. DA3 ~688x384,
    # multiples of 16), but the RGB frames / SAM3 masks may have been saved at a
    # DIFFERENT resolution (e.g. 360x640). If we scale mask lookups using the
    # on-disk frame resolution while the points live in the projection resolution,
    # the per-axis factor is wrong and points map off-target — worse the farther
    # from the image center (objects near the border drift). If the detected orig
    # is smaller than the actual pixel-coord extent, it cannot be the projection
    # resolution, so anchor to the pixel coords instead.
    px_h = float(pixel_row.max() + 1)
    px_w = float(pixel_col.max() + 1)
    # The frame/metadata resolution can disagree with the projection in EITHER
    # direction: frames saved SMALLER than the projection (px > orig) OR LARGER
    # (px < orig, e.g. 1920x1080 originals while VGGT/DA3 ran at 688x384). In the
    # second case the old "px > orig" check never fired, so the mask scale stayed
    # ~1.0 (should be ~2.8) and every point mapped into a corner → 0% coverage.
    # For a full-scene cloud the projected points span the whole image, so
    # (px_h, px_w) IS the projection resolution. Anchor to it whenever it
    # disagrees with orig but shares the mask's aspect ratio (the aspect check
    # rules out transposed / partial-frame false positives).
    proj_ar = px_w / max(px_h, 1.0)
    mask_ar = float(scaled_res[1]) / max(float(scaled_res[0]), 1.0)
    disagrees = (abs(px_h - orig_h) > 2.0) or (abs(px_w - orig_w) > 2.0)
    if disagrees and abs(proj_ar - mask_ar) < 0.10:
        print(f"[SegPipeline]   ⚠️ Detected orig {orig_w:.0f}x{orig_h:.0f} ≠ projection "
              f"{px_w:.0f}x{px_h:.0f} (cloud traced at the backend resolution; frames/masks "
              f"saved at another) — using projection resolution for mask scaling")
        orig_h, orig_w = px_h, px_w
    elif px_h > orig_h or px_w > orig_w:
        print(f"[SegPipeline]   ⚠️ Detected orig {orig_w:.0f}x{orig_h:.0f} is smaller than the "
              f"pixel-coord extent {px_w:.0f}x{px_h:.0f} (frames saved at a different resolution "
              f"than the projection); using projection resolution from pixel coords")
        orig_h = max(orig_h, px_h)
        orig_w = max(orig_w, px_w)

    mask_h_ref, mask_w_ref = scaled_res[0], scaled_res[1]
    print(f"[SegPipeline]   Original resolution: {orig_w:.0f}x{orig_h:.0f}, mask: {mask_h_ref}x{mask_w_ref}")
    print(f"[SegPipeline]   Scale factors: row={mask_h_ref/orig_h:.4f}, col={mask_w_ref/orig_w:.4f}")
    
    for i, obj_id in enumerate(obj_ids):
        # Skip obj_ids not in the incremental set
        if only_obj_ids is not None and obj_id not in only_obj_ids:
            continue
        # Compute average mask area for this object (across all frames)
        frame_areas = []
        
        for cloud_frame, pt_indices in frame_groups.items():
            # EXACT MATCH ONLY: if SAM3 didn't generate a mask for this exact frame, skip these points.
            # No fuzzy "nearest frame" matching, because that maps background points from unsegmented frames 
            # to masks from completely different timestamps.
            mask_key = f"f{cloud_frame}_o{obj_id}"
            if mask_key not in masks_data:
                continue
            
            mask = masks_data[mask_key].astype(np.uint8)
            
            # ── Erosion: shrink mask edges for tighter boundaries ──
            # Adaptive: skip/reduce erosion for small masks to avoid eliminating them
            raw_area = float(np.sum(mask > 0))
            if erosion_iterations <= 0 or raw_area < 2000:
                # erosion disabled (user 2026-08-29) or small object
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
            # pixel_row/pixel_col are in ORIGINAL resolution,
            # masks are at scaled_res — must rescale before lookup
            mask_h, mask_w = mask.shape[:2]
            orig_rows = pixel_row[pt_indices].astype(np.float32)
            orig_cols = pixel_col[pt_indices].astype(np.float32)
            scaled_rows = (orig_rows * (mask_h / orig_h)).astype(np.int32)
            scaled_cols = (orig_cols * (mask_w / orig_w)).astype(np.int32)
            rows = np.clip(scaled_rows, 0, mask_h - 1)
            cols = np.clip(scaled_cols, 0, mask_w - 1)
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
        voxel_mesh_data = []
        face_normals_data = []
        face_planes_data = []
        face_id_data = np.array([], dtype=np.int32)
        if skip_filter_ids and iid in skip_filter_ids:
            # Reuse cached filtered indices (already cleaned)
            pass  # all_matched stays as-is from mask matching
        elif pre_filter_count >= 20:
            all_matched, voxel_mesh_data, face_normals_data, face_planes_data, face_id_data = _clean_segment_subcloud(
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
        
        # Add voxel mesh data if available
        if voxel_mesh_data:
            instance["voxel_mesh"] = {
                "voxel_size": 0.05,  # 5cm mesh voxels (independent of cleaning voxel_size)
                "count": len(voxel_mesh_data),
                "data": voxel_mesh_data,  # [[cx,cy,cz,nx,ny,nz], ...]
            }
        
        # Compute OBB from snapped voxel centroids (corrected geometry)
        if voxel_mesh_data and len(voxel_mesh_data) >= 4:
            voxel_centers = np.array([[v[0], v[1], v[2]] for v in voxel_mesh_data])
            instance["obb"] = _compute_obb(voxel_centers, face_normals=face_normals_data)
        elif len(all_matched) >= 4:
            instance["obb"] = _compute_obb(xyz_display[all_matched], face_normals=face_normals_data)
        
        # Store face projection data for point correction
        if face_planes_data and len(face_id_data) > 0:
            instance["_face_planes"] = face_planes_data
            instance["_face_id"] = face_id_data
        
        instances.append(instance)
        removed = pre_filter_count - len(all_matched)
        filter_info = f" (filtered {removed} outliers)" if removed > 0 else ""
        print(f"[SegPipeline]   Object '{label}' #{iid}: "
              f"{len(all_matched):,} points{filter_info}")
    
    # ── Instance post-processing config ────────────────────────────────
    try:
        from config import cfg as _seg_global_cfg
        _dd = (_seg_global_cfg.get("segmentation", {}) or {})
    except Exception:
        _dd = {}
    _merge_on = bool(_dd.get("merge_duplicates", False))

    # ── Phase 3: Cross-category Re-ID — merge instances with high 3D overlap ──
    # If VLM produced synonyms (e.g., "chair" + "wooden chair"), SAM3 may have
    # segmented the same physical object twice. Detect and merge by 3D point overlap.
    # OFF by default: the test is `intersection / smaller`, i.e. CONTAINMENT, so a
    # large instance absorbs anything lying inside it — distinct objects, not synonyms.
    if _merge_on and len(instances) > 1:
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

    # ── Same-SPACE dedupe + small-instance filter ──────────────────────
    # The index-overlap merge above only fires when two instances share the SAME
    # cloud points — but duplicates of one physical object (two SAM3 concepts, or
    # a track split) usually land on DISJOINT points (each mask claims different
    # frames), so they never intersect by index. Occupancy says the truth: the
    # same object occupies the same SPACE. Voxelize each instance (5 cm) and merge
    # when most of the smaller one sits inside the bigger one. Then drop crumbs
    # (tiny instances below min_instance_points — mask slivers, not objects).
    _vox = float(_dd.get("dedupe_voxel_m", 0.05))
    _dup_thr = float(_dd.get("dedupe_overlap", 0.5))
    _min_pts = int(_dd.get("min_instance_points", 300))
    if not _merge_on:
        print("[SegPipeline]   Instance merging DISABLED "
              "(segmentation.merge_duplicates: false) — instances kept distinct")

    if _merge_on and len(instances) > 1 and _vox > 0:
        vox_sets = []
        for inst in instances:
            idxs = np.asarray(inst["globalIndices"], dtype=np.int64)
            if len(idxs) == 0:
                vox_sets.append(set())
                continue
            v = np.floor(xyz_display[idxs] / _vox).astype(np.int64)
            vox_sets.append(set(map(tuple, v)))
        absorbed = set()
        order = sorted(range(len(instances)), key=lambda k: -len(vox_sets[k]))
        for a_pos, i in enumerate(order):
            if i in absorbed or not vox_sets[i]:
                continue
            for j in order[a_pos + 1:]:
                if j in absorbed or not vox_sets[j]:
                    continue
                inter = len(vox_sets[i] & vox_sets[j])
                if inter and inter / len(vox_sets[j]) >= _dup_thr:
                    # j (smaller) is the same physical object as i → absorb
                    merged_idx = sorted(set(instances[i]["globalIndices"])
                                        | set(instances[j]["globalIndices"]))
                    instances[i]["globalIndices"] = merged_idx
                    instances[i]["total_points"] = len(merged_idx)
                    vox_sets[i] |= vox_sets[j]
                    absorbed.add(j)
                    print(f"[SegPipeline]   🔗 Space-dedupe: '{instances[j]['label']}' "
                          f"#{instances[j]['id']} is the same object as "
                          f"'{instances[i]['label']}' #{instances[i]['id']} "
                          f"({inter / len(vox_sets[j]):.0%} of its space) — merged")
        if absorbed:
            pre = len(instances)
            instances = [inst for k, inst in enumerate(instances) if k not in absorbed]
            for inst in instances:
                m = np.array(inst["globalIndices"], dtype=np.int64)
                if len(m) >= 4:
                    inst["obb"] = _compute_obb(xyz_display[m])
            print(f"[SegPipeline]   Space-dedupe: {pre} → {len(instances)} instances")

    if _min_pts > 0:
        tiny = [inst for inst in instances if inst["total_points"] < _min_pts]
        if tiny:
            _tiny_desc = ", ".join("{}#{}({})".format(t["label"], t["id"], t["total_points"])
                                   for t in tiny[:10])
            print(f"[SegPipeline]   Dropped {len(tiny)} tiny instance(s) "
                  f"(<{_min_pts} pts): {_tiny_desc}{'...' if len(tiny) > 10 else ''}")
            instances = [inst for inst in instances if inst["total_points"] >= _min_pts]

    # ── Geometric completion — "pegar los puntos al lugar correcto" (USER
    # 2026-08-29): SAM3 leaves ~30% of the cloud unsegmented even where the
    # surface clearly belongs to a segmented object (masklets die frames away
    # from the prompt; masks under-cover inside their own frames). A point
    # within attach_dist_m of an instance's EXISTING points is part of that
    # surface → attached to the NEAREST instance. Purely geometric and
    # conservative: points nobody reaches stay unsegmented (never invent).
    _att_on = bool(_dd.get("attach_unsegmented", True))
    _att_d = float(_dd.get("attach_dist_m", 0.03))
    if _att_on and instances:
        try:
            n_att, grown = _attach_unsegmented(instances, xyz_display,
                                               attach_dist_m=_att_d)
            for k in grown:   # OBBs must include the attached points
                m = np.asarray(instances[k]["globalIndices"], dtype=np.int64)
                if len(m) >= 4:
                    instances[k]["obb"] = _compute_obb(xyz_display[m])
            if n_att:
                print(f"[SegPipeline]   📎 attach: {n_att:,} unsegmented points "
                      f"glued to their surfaces (≤{_att_d*100:.0f} cm)")
        except Exception as e:
            print(f"[SegPipeline] attach step failed (non-fatal): {e}")

    # ── EXCLUSIVITY INVARIANT (USER 2026-08-31): every point belongs to ONE
    # instance or none — never two. Masks can claim the same points for
    # different segments; merges/attach/incremental unions could double-own.
    # Conflicts resolve to the SMALLEST instance (a point on a small object
    # belongs to the object, not the big surface behind it) and are reported.
    try:
        _n_dup = _enforce_exclusive_ownership(instances, n_pts)
        if _n_dup:
            print(f"[SegPipeline]   ⚠ exclusivity enforced: {_n_dup:,} "
                  f"double-owned point(s) resolved")
    except Exception as e:
        print(f"[SegPipeline] exclusivity enforcement failed (non-fatal): {e}")

    # ── Canonical instance store (scene_r.db) — THE single source of objects
    # for spatial Q&A (phase5), classification (phase2), findings (phase3) and
    # reports (phase6). Rebuilt from scratch on every segmentation, straight
    # from the CLEAN instances: points/OBBs in the DISPLAY frame (the exact
    # geometry the viewer renders and the user measures against).
    try:
        _write_instance_store(output_dir, instances, xyz_display)
    except Exception as e:
        print(f"[SegPipeline] instance store build failed (non-fatal): {e}")

    total_segmented = sum(inst["total_points"] for inst in instances)
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
    
    # ── Phase 4: Project assigned points to face planes → corrected cloud ──
    # Project in display space (where face planes live), then convert back
    # to raw space for PLY output. Potree re-applies floorTransform on load.
    n_projected = 0
    # In incremental mode, start from corrected_cloud to preserve previous projections
    corrected_path = output_dir / "corrected_cloud.ply"
    if only_obj_ids is not None and corrected_path.exists():
        prev_origins = _load_ply_origins(corrected_path)
        if prev_origins is not None and len(prev_origins[0]) == len(xyz):
            xyz_corrected = prev_origins[0].copy()
        else:
            xyz_corrected = xyz.copy()
    else:
        xyz_corrected = xyz.copy()  # RAW space
    # Load existing classification to preserve previous objects in incremental mode
    class_path = output_dir / "classification.npy"
    if class_path.exists() and only_obj_ids is not None:
        classification = np.load(class_path)
        if len(classification) != len(xyz):
            classification = np.zeros(len(xyz), dtype=np.uint8)
    else:
        classification = np.zeros(len(xyz), dtype=np.uint8)
    
    for inst in instances:
        face_planes = inst.pop("_face_planes", None)
        face_id = inst.pop("_face_id", None)
        
        # SIEMPRE asignar la clasificación del objeto (aunque no tenga caras planas)
        global_indices = np.array(inst["globalIndices"], dtype=np.int64)
        seg_id = int(inst.get("id", 0))
        classification[global_indices] = min(seg_id, 255)

        # Si no hay caras planas, saltamos la corrección geométrica (proyección)
        if face_planes is None or face_id is None or len(face_id) == 0:
            continue
        
        pts_display = xyz_display[global_indices].copy()
        
        for fi, (fn, fd) in enumerate(face_planes):
            mask = face_id == fi
            n_mask = int(np.sum(mask))
            if n_mask == 0:
                continue
            pts_fi = pts_display[mask]
            dists = pts_fi @ fn + fd
            projected = pts_fi - np.outer(dists, fn)
            pts_display[mask] = projected
            n_projected += int(n_mask)
        
        # Convert projected display coords back to raw: raw = (display - t) @ R / s
        if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
            pts_raw = (pts_display - t) @ R / s
        else:
            pts_raw = pts_display
        xyz_corrected[global_indices] = pts_raw
    
    # Save classification sidecar — and remember whether it actually CHANGED
    # (USER 2026-09-06 "ajustá todo lo que sobra": a re-match after a chunk
    # correction produces the IDENTICAL classification, and the 5-minute
    # Potree rebuild that always ran here was pure waste).
    class_path = output_dir / "classification.npy"
    classification_changed = True
    if class_path.exists():
        try:
            old_cls = np.load(class_path, mmap_mode="r")
            classification_changed = not (
                len(old_cls) == len(classification)
                and np.array_equal(old_cls, classification))
        except Exception:  # noqa: BLE001
            pass
    np.save(class_path, classification)

    if n_projected > 0:
        corrected_path = output_dir / "corrected_cloud.ply"
        _write_corrected_ply(ply_path, corrected_path, xyz_corrected)
        print(f"[SegPipeline] ✏️ Projected {n_projected:,} points → {corrected_path.name} (raw space)")
        ply_override = corrected_path
    else:
        corrected_path = output_dir / "corrected_cloud.ply"
        ply_override = corrected_path if corrected_path.exists() else ply_path

    # Rebuild Potree ONLY when something the octree carries actually changed:
    # projected geometry, a different classification, or no octree at all.
    potree_missing = not (output_dir / "potree" / "metadata.json").exists()
    if n_projected > 0 or classification_changed or potree_missing:
        try:
            from potree_converter import convert_ply_to_potree
            session_dir = output_dir.parent
            success = convert_ply_to_potree(session_dir, force=True, ply_override=ply_override)
            if success:
                print(f"[SegPipeline] 🌲 Potree octree rebuilt from {ply_override.name}")
                result["reload_potree"] = True
            else:
                print(f"[SegPipeline] ⚠️ Potree rebuild failed")
        except Exception as e:
            print(f"[SegPipeline] ⚠️ Potree rebuild error: {e}")
    else:
        print("[SegPipeline] Potree untouched — classification identical, "
              "no projections (nothing to rebuild)")
    
    return result


def map_segmentation_to_cloud(output_dir) -> dict:
    """Deferred mask→cloud mapping (anchored pipeline order): run the FULL
    matching + per-instance cleaning ONCE against the (corrected, merged)
    cleaned cloud, then refresh the Phase R store's canonical OBBs so the
    assistant's boxes coincide exactly with the viewer's. Called by the
    cloudcompy stage when segmentation.json exists."""
    output_dir = Path(output_dir)
    if not (output_dir / "segmentation.json").exists():
        return {"error": "no segmentation.json", "instances": []}
    if not (output_dir / "cleaned_cloud.ply").exists():
        return {"error": "no cleaned_cloud.ply", "instances": []}
    result = _match_and_save_result(output_dir)
    # scene_r.db is (re)built inside the mask→cloud matching itself — points,
    # labels and OBBs all in the display frame, single source for phases 2-6.
    return result


def _write_instance_store(output_dir: Path, instances: list,
                          xyz_display: np.ndarray) -> None:
    """Write scene_r.db from clean instances + the display-frame cloud.
    Shared by the segmentation matcher (in-memory instances) and
    ``rebuild_instance_store`` (instances reloaded from disk)."""
    from phase_r.instance_store import InstanceStore
    _sp = output_dir / "scene_r.db"
    for _suffix in ("", "-wal", "-shm"):
        _f = Path(str(_sp) + _suffix)
        if _f.exists():
            _f.unlink()
    _st = InstanceStore(_sp)
    for inst in instances:
        _iid = int(inst["instance_id"])
        _m = np.asarray(inst["globalIndices"], dtype=np.int64)
        _st.upsert_instance(_iid, str(inst["label"]), source="sam3_concepts",
                            status="proposed", n_views=0,
                            label_origin="vlm_proposed")
        _st.set_points(_iid, xyz_display[_m])
        _obb = inst.get("obb") or {}
        if _obb.get("center") and _obb.get("half_extents"):
            _c = np.asarray(_obb["center"], float)
            _h = np.asarray(_obb["half_extents"], float)
            _Rd = np.asarray(_obb.get("rotation", np.eye(3)), float)
            _T = np.eye(4)
            _T[:3, :3] = _Rd
            _T[:3, 3] = _c
            _aabb = np.array([-_h[0], _h[0], -_h[1], _h[1], -_h[2], _h[2]])
            _st.set_obb(_iid, _T, _aabb, _c, n_points=int(inst["total_points"]),
                        obb_origin="tool_measured")
    _st.set_meta("built_from", "sam3_concepts_display_frame")
    _st.close()
    print(f"[SegPipeline] scene_r.db built: {len(instances)} instances "
          f"(display frame — Q&A/classify/findings read from here)")


def rebuild_instance_store(output_dir) -> bool:
    """Rebuild scene_r.db from the EXISTING segmentation_result.json, without
    re-running DBSCAN/matching. Sessions segmented before the store existed —
    or whose store a reconstruction re-run wiped while the result survived —
    have segmentation but no db, which left the spatial-Q&A chat tool-less
    (text answers, no 3D measurements). Same display-frame convention as
    ``_match_and_save_result``. Returns True when the store was written."""
    output_dir = Path(output_dir)
    result_path = output_dir / "segmentation_result.json"
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text())
        instances = [i for i in result.get("instances", [])
                     if i.get("globalIndices")]
        if not instances:
            return False
        ply_path = output_dir / (result.get("cloud_source") or "cleaned_cloud.ply")
        if not ply_path.exists():
            ply_path = output_dir / "cleaned_cloud.ply"
        if not ply_path.exists():
            return False
        import open3d as o3d
        xyz = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
        if not len(xyz):
            return False
        # Same display frame the matcher uses (viewer geometry): a saved
        # floor_transform.npz ALWAYS wins (the viewer applies it to the cloud
        # unconditionally — even on baked-orientation sessions, where
        # level_floor may have composed a fine-snap delta into it); baked
        # orientation without an npz → identity; else raw.
        xyz_display = xyz
        transform_path = output_dir / "floor_transform.npz"
        if transform_path.exists():
            try:
                data = np.load(transform_path)
                s, R, t = float(data["s"]), data["R"], data["t"]
                if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
                    xyz_display = s * (xyz @ R.T) + t
            except Exception as e:
                print(f"[SegPipeline] rebuild: floor_transform load failed: {e}")
        n_max = int(max(max(i["globalIndices"]) for i in instances))
        if n_max >= len(xyz):
            print(f"[SegPipeline] rebuild: indices exceed {ply_path.name} "
                  f"({n_max} >= {len(xyz)}) — stale result, not rebuilding")
            return False
        _write_instance_store(output_dir, instances, xyz_display)
        return True
    except Exception as e:
        print(f"[SegPipeline] instance store rebuild failed: {e}")
        return False


# (DINOv3 fase-4 refine DELETED by USER ORDER 2026-09-05)


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
        
        if "error" in result:
            # Fatal error, abort save
            if prev_instances:
                return prev_result
            return result
        
        if not result.get("instances"):
            # No new matches generated
            if prev_instances:
                return prev_result
            return result
        
        new_instances = result["instances"]
        new_ids = {inst["id"] for inst in new_instances}
        
        # Merge: keep old instances (not replaced by new), add new
        merged = [inst for inst in prev_instances if inst["id"] not in new_ids]
        merged.extend(new_instances)

        total_pts = result.get("total_points", prev_result.get("total_points", 0))
        # EXCLUSIVITY INVARIANT (USER 2026-08-31): a new category's masks can
        # claim points already owned by old instances (this merge was THE
        # double-ownership source). Smallest instance keeps contested points.
        try:
            _n_dup = _enforce_exclusive_ownership(merged, int(total_pts))
            if _n_dup:
                print(f"[SegPipeline]   ⚠ exclusivity (incremental merge): "
                      f"{_n_dup:,} double-owned point(s) resolved")
        except Exception as e:
            print(f"[SegPipeline] exclusivity enforcement failed (non-fatal): {e}")

        # Recompute coverage from merged set
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
    # USER DOCTRINE 2026-09-06 ("cada modificación recalcula; si es solo
    # carga, no se recalcula NADA"): loading NEVER invalidates and NEVER
    # recomputes. Every mutating action (segmenting, brush cleaning, gizmo
    # corrections, floor leveling) leaves ALL derived artifacts consistent
    # on disk before it finishes — so the load trusts the cached result
    # blindly. The staleness checks that used to live here (cloud/
    # transform/masks/code mtimes) WERE the bug: they made every session
    # load re-run matching+DBSCAN after any change.
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
                # Strip transient flags — only needed for first load after segmentation
                cache_result = {k: v for k, v in result.items()
                                if k not in ("reload_potree", "corrected_is_display_space")}
                with open(result_path, "w") as f:
                    json.dump(cache_result, f)
                print(f"[SegPipeline] 💾 Cached result for instant future loads")
            except Exception as e:
                print(f"[SegPipeline] ⚠️ Failed to cache result: {e}")
        
        return result
    finally:
        lock.release()

