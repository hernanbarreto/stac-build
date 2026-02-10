import sys
import os
import torch
import numpy as np
import asyncio
import glob
import json
import shutil
import argparse
from datetime import datetime
import gc
from pathlib import Path

# Import config to get DA3 path
try:
    from config import cfg
    _da3_path = cfg.get("da3", {}).get("install_path", "/home/hernan/Depth-Anything-3")
except ImportError:
    _da3_path = "/home/hernan/Depth-Anything-3"

# Add Depth-Anything-3 to path to allow imports
da3_root = os.path.abspath(_da3_path)
da3_streaming_path = os.path.join(da3_root, "da3_streaming")

# CRITICAL: da3_root must come BEFORE da3_streaming_path in sys.path
# Otherwise Python finds da3_streaming.py (file) before da3_streaming/ (package)
if da3_streaming_path not in sys.path:
    sys.path.insert(0, da3_streaming_path)
if da3_root not in sys.path:
    sys.path.insert(0, da3_root)  # MUST be last insert(0) so it ends up first

# DA3 imports are optional
DA3_NATIVE_AVAILABLE = False
DA3_Streaming = None
depth_to_point_cloud_vectorized = None
accumulate_sim3_transforms = None
save_confident_pointcloud_batch = None
process_loop_list = None

try:
    from da3_streaming.da3_streaming import DA3_Streaming as _DA3_Streaming
    from da3_streaming.da3_streaming import depth_to_point_cloud_vectorized as _depth_func
    DA3_Streaming = _DA3_Streaming
    depth_to_point_cloud_vectorized = _depth_func
    
    from loop_utils.sim3utils import (
        accumulate_sim3_transforms as _accum,
        save_confident_pointcloud_batch as _save,
        process_loop_list as _process
    )
    accumulate_sim3_transforms = _accum
    save_confident_pointcloud_batch = _save
    process_loop_list = _process
    
    DA3_NATIVE_AVAILABLE = True
    print("[DA3 Native] DA3_Streaming loaded successfully")
except ImportError as e:
    from config import cfg as _cfg
    if _cfg.get("slam_backend", "mast3r") == "da3":
        print(f"[DA3 Native] DA3_Streaming not available: {e}")
        print("[DA3 Native] RealtimeDA3 will not be functional")


# Stub class if DA3 not available
if DA3_Streaming is None:
    class DA3_Streaming:
        """Stub class when DA3 is not available."""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("DA3_Streaming is not available. Install pypose, numba and other DA3 dependencies.")


class RealtimeDA3(DA3_Streaming):
    """
    Wrapper around DA3_Streaming to support real-time feedback via callbacks.
    Inherits 100% of the logic, only injecting the callback hook.
    """
    def __init__(self, image_dir, save_dir, config, alignment_manager=None):
        # Initialize the parent class strictly
        super().__init__(image_dir, save_dir, config)

        # Ensure save_dir is set (parent class might not set it until run() is called)
        self.save_dir = save_dir

        # Initialize img_list immediately (copied from DA3_Streaming.run)
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        
        # Filter blurry frames via frame_quality.json (if available)
        try:
            from frame_quality import load_valid_frames
            valid_files = load_valid_frames(self.img_dir)
            if valid_files is not None:
                original_count = len(self.img_list)
                valid_set = set(valid_files)
                self.img_list = [p for p in self.img_list if os.path.basename(p) in valid_set]
                filtered = original_count - len(self.img_list)
                print(f"[DA3] 🔍 Frame quality filter: {original_count} → {len(self.img_list)} frames ({filtered} blurry removed)")
            else:
                print(f"[DA3] No frame_quality.json found — using all {len(self.img_list)} frames")
        except Exception as e:
            print(f"[DA3] ⚠️ Frame quality filter skipped: {e}")
        
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] All frames were filtered as blurry!")
        print(f"Found {len(self.img_list)} images")

        self.alignment_manager = alignment_manager
        self.gravity_transform = (1.0, np.eye(3), np.zeros(3))
        
    def _compute_initial_gravity_transform(self, chunk_data):
        # Deprecated: Logic moved to AlignmentManagerr
        pass
        
    def _save_metadata(self, chunk_id, result, alignment_transform, sample_indices=None):
        """Save complete metadata for FrameStorage and segmentation compatibility."""
        try:
            filename = f"chunk_{chunk_id:03d}_meta.json"
            filepath = os.path.join(self.save_dir, filename)

            s, R, t = alignment_transform

            # Format as 'cameras' dict for consistency with Online mode
            cameras = {}
            frame_count = result.extrinsics.shape[0] if hasattr(result.extrinsics, 'shape') else len(result.extrinsics)
            for i in range(frame_count):
                cameras[str(i)] = {
                    "extrinsics": result.extrinsics[i].tolist() if hasattr(result.extrinsics[i], 'tolist') else result.extrinsics[i],
                    "intrinsics": result.intrinsics[i].tolist() if hasattr(result.intrinsics[i], 'tolist') else result.intrinsics[i]
                }

            # Get scaled resolution from depth output (DA3 internal resolution)
            scaled_resolution = None
            if hasattr(result, 'depth') and result.depth is not None:
                scaled_resolution = list(result.depth[0].shape)  # [H, W]

            # Get original resolution from first image in img_list
            original_resolution = None
            if len(self.img_list) > 0:
                import cv2
                first_img = cv2.imread(self.img_list[0])
                if first_img is not None:
                    original_resolution = [first_img.shape[0], first_img.shape[1]]  # [H, W]

            # Get chunk step from config (for frame mapping)
            from config import cfg
            chunk_size = cfg["server"]["chunk_size"]
            chunk_overlap = cfg["server"]["chunk_overlap"]
            chunk_step = chunk_size - chunk_overlap

            data = {
                "chunk_id": chunk_id,
                "frame_count": frame_count,
                "cameras": cameras,
                "scaled_resolution": scaled_resolution,
                "original_resolution": original_resolution,
                "chunk_step": chunk_step,
                "frame_global_start": chunk_id * chunk_step,
                "frame_global_end": chunk_id * chunk_step + frame_count - 1,
                "alignment_transform": {
                    "scale": float(s),
                    "rotation": R.tolist() if hasattr(R, 'tolist') else R,
                    "translation": t.tolist() if hasattr(t, 'tolist') else t
                },
                "ply_pre_aligned": True  # PLY saved with alignment already applied
            }

            # Save sample_indices if provided (CRITICAL for segmentation index mapping)
            if sample_indices is not None:
                data["sample_indices"] = sample_indices.tolist() if isinstance(sample_indices, np.ndarray) else sample_indices

            with open(filepath, 'w') as f:
                json.dump(data, f)
            print(f"[DA3Wrapper] Saved Metadata: {filepath} (original_res={original_resolution}, scaled_res={scaled_resolution})")
        except Exception as e:
            print(f"[DA3Wrapper] Failed to save metadata: {e}")
            import traceback
            traceback.print_exc()

    async def process_long_sequence_async(self, callback=None):
        """
        Async version of process_long_sequence that allows yielding control to the event loop
        and sending updates via callback.
        """
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} \
                chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        pre_predictions = None
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")
            
            # RUN IN EXECUTOR to avoid blocking the loop during heavy GPU inference
            cur_predictions = await asyncio.to_thread(
                self.process_single_chunk, 
                self.chunk_indices[chunk_idx], 
                chunk_idx=chunk_idx
            )
            
            torch.cuda.empty_cache()

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(self.chunk_indices)-1})"
                )
                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                # Alignment Logic (Copied to ensure scope availability)
                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                s, R, t = await asyncio.to_thread(
                    self.align_2pcds,
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )
                self.sim3_list.append((s, R, t))

            pre_predictions = cur_predictions
            
            # 1. Calculate Current World Transform
            current_s, current_R, current_t = 1.0, np.eye(3), np.zeros(3)
            if chunk_idx > 0:
                 acc_transforms = accumulate_sim3_transforms(self.sim3_list)
                 current_s, current_R, current_t = acc_transforms[-1]
            
            # 2. Use AlignmentManagerr if available (User Requirement)
            # This delegates floor alignment (Auto-leveling) to the existing manager
            aligned_points = None
            final_s, final_R, final_t = current_s, current_R, current_t
            
            if self.alignment_manager:
                # Wrap DA3 predictions into a simple object compatible with AlignmentManagerr
                class ChunkWrapper:
                    def __init__(self, p, frame_count):
                        self.depths = np.squeeze(p.depth)
                        self.confs = p.conf
                        self.intrinsics = p.intrinsics
                        self.extrinsics = p.extrinsics
                        self.images = p.processed_images
                        self.frame_count = frame_count
                        
                        # Generate unaligned XYZ
                        # points shape: (N, H, W, 3)
                        pts = depth_to_point_cloud_vectorized(p.depth, p.intrinsics, p.extrinsics)
                        
                        # Process Images: (N, 3, H, W) -> (N, H, W, 3)
                        # Ensure we handle torch tensor or numpy array
                        imgs = self.images
                        if hasattr(imgs, 'permute'): # Torch
                             imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
                        elif imgs.ndim == 4 and imgs.shape[1] == 3: # Numpy NCHW
                             imgs = np.transpose(imgs, (0, 2, 3, 1))
                        
                        # Flatten
                        pts_flat = pts.reshape(-1, 3)
                        imgs_flat = imgs.reshape(-1, 3)
                        
                        # Normalize images from 0-255 to 0-1 if needed
                        # DA3 usually returns 0-1 float or 0-255 uint8?
                        # If uint8, convert to float for viewer compatibility (main.py expects float)
                        if imgs_flat.dtype == np.uint8:
                            imgs_flat = imgs_flat.astype(np.float32) / 255.0
                        
                        # Concatenate XYZ + RGB -> (Total, 6)
                        self.point_cloud = np.hstack((pts_flat, imgs_flat)).astype(np.float32)

                        # Add SIM3 info for manager to use
                        self.sim3_scale = current_s
                        self.sim3_rotation = current_R
                        self.sim3_translation = current_t
                        self.segmentation_masks = None # Only if we had SAM3 here
                
                # We need to compose the SIM3 transform BEFORE passing to Manager?
                # Actually Manager.add_chunk expects "ChunkResult" which has SIM3 applied?
                # No, AlignmentManagerr apply_gravity_correction expects logic.
                
                # In AlignmentManagerr.add_chunk(chunk_result):
                # It calls _generate_point_cloud using just the raw data + gravity.
                # BUT for chunks > 0, we need the accumulation.
                # AlignmentManagerr handles accumulation internally?
                # AlignmentManagerr.sim3_list.
                # BUT we are adding chunks here in RealtimeDA3.
                # We should use AlignmentManagerr ONLY for the auto-leveling of the FIRST chunk if we want to stay robust.
                # Or we feed it everything.
                
                # Let's simplify: RealtimeDA3 calculates the Accumulation (SIM3).
                # We want AlignmentManagerr to calculate Gravity (on chunk 0).
                # And apply Gravity * Accumulation on chunk N.
                
                # Step A: Feed Chunk 0 to Manager to get Gravity.
                if chunk_idx == 0:
                    actual_frame_count = cur_predictions.depth.shape[0]
                    wrapper = ChunkWrapper(cur_predictions, actual_frame_count)
                    # We need to hack the Manager to JUST compute gravity without storing full history if we don't want double storage?
                    # Actually Main.py resets manager. So we can use it fully.
                    aligned_chunk = self.alignment_manager.add_chunk(wrapper) # Will compute gravity on chunk 0
                    
                    # Read back the gravity transform
                    self.gravity_transform = self.alignment_manager.gravity_correction
                    print(f"[DA3Wrapper] Retrieved Gravity Transform from Manager: {self.gravity_transform}")
                
            # Now we have self.gravity_transform (either identity or from Manager)
            s_g, R_g, t_g = self.gravity_transform
            
            # Compose Final Transform: T_final = T_gravity * T_accum
            final_s = s_g * current_s
            final_R = R_g @ current_R
            final_t = s_g * (R_g @ current_t) + t_g
            
            # Apply Transform
            local_points = depth_to_point_cloud_vectorized(
                cur_predictions.depth, 
                cur_predictions.intrinsics, 
                cur_predictions.extrinsics
            )
            
            aligned_points = final_s * (local_points @ final_R.T) + final_t

            # 3. Save ALIGNED PLY with manual filtering to capture sample_indices
            ply_path = os.path.join(self.save_dir, f"chunk_{chunk_idx:03d}.ply")

            colors = cur_predictions.processed_images
            confs = cur_predictions.conf

            conf_threshold_coef = self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
            sample_ratio = self.config["Model"]["Pointcloud_Save"]["sample_ratio"]

            # Flatten arrays for filtering
            points_flat = aligned_points.reshape(-1, 3)
            colors_flat = colors.reshape(-1, 3) if colors.ndim == 4 else colors  # Handle both (N,H,W,3) and (N,3)
            confs_flat = confs.reshape(-1)

            # Apply confidence threshold filtering
            conf_threshold = max(0.0, np.mean(confs_flat) * conf_threshold_coef)
            valid_mask = confs_flat >= conf_threshold
            valid_indices = np.where(valid_mask)[0]

            points_filtered = points_flat[valid_indices]
            colors_filtered = colors_flat[valid_indices] if colors_flat.ndim > 1 else colors_flat
            confs_filtered = confs_flat[valid_indices]

            # Apply sampling and capture sample_indices
            sample_indices_for_meta = None
            if sample_ratio < 1.0 and len(points_filtered) > 0:
                n_samples = int(len(points_filtered) * sample_ratio)
                sample_indices_local = np.sort(np.random.choice(len(points_filtered), n_samples, replace=False))

                # Map back to original flat indices (before conf filtering)
                sample_indices_for_meta = valid_indices[sample_indices_local]

                points_final = points_filtered[sample_indices_local]
                colors_final = colors_filtered[sample_indices_local] if colors_filtered.ndim > 1 else colors_filtered
            else:
                points_final = points_filtered
                colors_final = colors_filtered

            # Save PLY manually (replicate save_confident_pointcloud_batch logic)
            # Use the DA3 function but with already-filtered/sampled data and sample_ratio=1.0
            # Format: points must be (N, 3) or (b, H, W, 3)
            save_confident_pointcloud_batch(
                points=points_final,  # Already (N, 3)
                colors=colors_final,  # Already (N, 3)
                confs=np.ones(len(points_final)),  # All points already passed threshold
                output_path=ply_path,
                conf_threshold=0.0,  # No additional filtering needed
                sample_ratio=1.0,  # No additional sampling needed
            )

            # --- SAVE POINT ORIGINS (For SAM3 Segmentation Mapping) ---
            # Each final point came from a specific (frame, pixel_row, pixel_col)
            # Flat index i in the (N*H*W) array maps to:
            #   frame_local = i // (H*W), row = (i % (H*W)) // W, col = i % W
            try:
                N_frames = cur_predictions.depth.shape[0]
                H_scaled = cur_predictions.depth.shape[-2] if cur_predictions.depth.ndim >= 3 else cur_predictions.depth.shape[0]
                W_scaled = cur_predictions.depth.shape[-1]
                HW = H_scaled * W_scaled
                
                # Get the final flat indices into the original (N*H*W) array
                if sample_ratio < 1.0 and sample_indices_for_meta is not None:
                    final_flat_indices = sample_indices_for_meta
                else:
                    final_flat_indices = valid_indices
                
                # Compute origin for each point
                frame_local = final_flat_indices // HW
                pixel_row = (final_flat_indices % HW) // W_scaled
                pixel_col = final_flat_indices % W_scaled
                
                # Convert local frame index to global frame index
                from config import cfg as _cfg
                chunk_size = _cfg["server"]["chunk_size"]
                chunk_overlap = _cfg["server"]["chunk_overlap"]
                chunk_step = chunk_size - chunk_overlap
                frame_global = frame_local + chunk_idx * chunk_step
                
                # Also map to original resolution pixels (for SAM3 mask lookup)
                # SAM3 runs on original images, DA3 runs on scaled resolution
                origin_path = os.path.join(self.save_dir, f"chunk_{chunk_idx:03d}_origins.npz")
                np.savez_compressed(origin_path,
                    frame_global=frame_global.astype(np.int32),
                    pixel_row=pixel_row.astype(np.int16),
                    pixel_col=pixel_col.astype(np.int16),
                    scaled_resolution=[H_scaled, W_scaled],
                )
                print(f"[DA3] Saved point origins: {origin_path} ({len(frame_global)} points)")
            except Exception as e:
                print(f"[DA3] ⚠️ Failed to save point origins: {e}")
                import traceback
                traceback.print_exc()

            # --- SAVE METADATA (For Retroactive Segmentation) ---
            self._save_metadata(chunk_idx, cur_predictions, (final_s, final_R, final_t), sample_indices_for_meta)
            
            # --- CALLBACK ---
            if callback:
                 # Pass the FINAL composed transform so the viewer also knows the correct pose if needed
                 await callback(chunk_idx, (final_s, final_R, final_t))


        # Loop Closure (Optional - commented out or kept minimal if user wants strictly streaming)
        # keeping original structure if needed later
        
        print("Apply alignment (Final Pass)")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        
        # --- Handle Single Chunk Case ---
        if len(self.chunk_indices) == 1:
            chunk_data_first = np.load(
                os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
            ).item()
            np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            
            points_first = depth_to_point_cloud_vectorized(
                chunk_data_first.depth,
                chunk_data_first.intrinsics,
                chunk_data_first.extrinsics,
            )
            ply_path_first = os.path.join(self.save_dir, "chunk_000.ply")
            
            conf_threshold = max(0.0, np.mean(chunk_data_first.conf) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"])
            save_confident_pointcloud_batch(
                points=points_first,
                colors=chunk_data_first.processed_images,
                confs=chunk_data_first.conf,
                output_path=ply_path_first,
                conf_threshold=conf_threshold,
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

        # --- Handle Multi Chunk Saving ---
        for chunk_idx in range(len(self.chunk_indices) - 1):
            s, R, t = self.sim3_list[chunk_idx]
            
            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True
            ).item()

            aligned_chunk_data = {}
            from loop_utils.alignment_torch import depth_to_point_cloud_optimized_torch, apply_sim3_direct_torch

            aligned_chunk_data["world_points"] = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                 aligned_chunk_data["world_points"], s, R, t
            )
            aligned_chunk_data["conf"] = chunk_data.conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, aligned_chunk_data)

            if chunk_idx == 0:
                 chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
                 ).item()
                 np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
                 
                 points_first = depth_to_point_cloud_vectorized(
                    chunk_data_first.depth, chunk_data_first.intrinsics, chunk_data_first.extrinsics
                 )
                 ply_path_first = os.path.join(self.save_dir, "chunk_000.ply")
                 save_confident_pointcloud_batch(
                    points=points_first,
                    colors=chunk_data_first.processed_images,
                    confs=chunk_data_first.conf,
                    output_path=ply_path_first,
                    conf_threshold=np.mean(chunk_data_first.conf) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                    sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"]
                 )

            points = aligned_chunk_data["world_points"].reshape(-1, 3)
            colors = (aligned_chunk_data["images"].reshape(-1, 3)).astype(np.uint8)
            confs = aligned_chunk_data["conf"].reshape(-1)
            ply_path = os.path.join(self.save_dir, f"chunk_{chunk_idx+1:03d}.ply")
            
            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"]
            )

        self.save_camera_poses()
        print("Done.")
