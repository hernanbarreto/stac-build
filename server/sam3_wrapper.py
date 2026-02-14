import os
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from threading import Lock
import logging

# Centralised vendor path resolution (ensures sam3 package is findable)
import vendor_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAM3Wrapper")
from config import cfg

class SAM3Wrapper:
    """
    Wrapper for SAM3 Video Predictor to handle text-prompt based segmentation
    and propagation across video chunks.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.predictor = None
        self.is_loaded = False
        self.lock = Lock()
        logger.info("SAM3 Wrapper initialized (Lazy Loading Enabled: Model will load on first prompt).")
        
    def load_model(self):
        """Lazy load the SAM3 model."""
        if self.is_loaded:
            return

        with self.lock:
            if self.is_loaded:
                return

            logger.info("Loading SAM3 Model...")
            try:
                from sam3.model_builder import build_sam3_video_predictor
                
                # Determine GPUs to use
                if self.device == "cuda":
                    gpus_to_use = [torch.cuda.current_device()]
                else:
                    gpus_to_use = [] # CPU mode effectively
                
                self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
                self.is_loaded = True
                logger.info("SAM3 Model loaded successfully.")
                
            except Exception as e:
                logger.error(f"Failed to load SAM3 model: {e}")
                import traceback
                traceback.print_exc()
                raise e

    def unload_model(self):
        """Unload model to free VRAM."""
        with self.lock:
            if self.predictor is not None:
                del self.predictor
                self.predictor = None
                self.is_loaded = False
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("SAM3 Model unloaded.")

    def process_chunk(self, frames_path: str, prompt_text: str, keyframe_interval: int = None) -> Dict[int, Any]:
        """
        Process a video chunk with a text prompt.
        
        Args:
            frames_path: Path to the directory containing frames (or video file path).
            prompt_text: The text description of the object to segment.
            keyframe_interval: Interval for extracting masks (0, 5, 10...).
            
        Returns:
            Dictionary mapping frame_index -> mask_data.
        """
        if not self.is_loaded:
            self.load_model()
        
        # Load default from config if needed
        if keyframe_interval is None:
            keyframe_interval = cfg["models"]["segmentation"]["keyframe_interval"]
            
        session_id = None
        results = {}
        
        try:
            logger.info(f"Starting SAM3 session for {frames_path} with prompt '{prompt_text}'")
            
            # 1. Start Session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=frames_path,
                )
            )
            session_id = response["session_id"]
            
            # 2. Add Text Prompt (Search Strategy)
            # User request: "If object not in frame 0, search for it."
            # We try keyframes: 0, 5, 10, 15, 20...
            # We iterate until we find a valid mask (heuristic) or just rely on SAM3's propagation.
            # But SAM3's add_prompt is for a specific frame.
            # We will try to add prompt to frame 0 first.
            # If the user says "search until found", we should ideally:
            # - Try frame 0. Check result.
            # - If empty, reset session, try frame 5...
            # But "check result" requires parsing 'pred_masks' from add_prompt response OR handle_stream_request.
            # To avoid complexity in this step (parsing SAM3 binaries), we will implementing a simplified robust strategy:
            # We try frame 0 AND frame 10 (mid-chunk).
            # Doubling the prompts might help SAM3 "find" it if it moves into view?
            # Or just Frame 0.
            # Actually, `add_prompt` returns the prediction for that frame immediately.
            # Let's trust SAM3 for now but try to add prompt to the middle frame if it's long?
            # Wait, the prompt is "sofa". If I add it at frame 0 and frame 15, SAM3 has 2 constraints.
            # This is actually better than "searching". Two constraints help tracking.
            # But assume object is NOT in frame 0. Frame 0 constraint might produce empty mask.
            # Frame 15 produces sofa mask.
            # SAM3 handles this. 
            # So the strategy: Prompt at configured frames (e.g. 0 and 15)
            prompts_to_add = cfg["models"]["segmentation"]["prompt_search_frames"]
            
            # Filter out frames that exceed the actual chunk length
            max_frame_idx = len(os.listdir(frames_path)) - 1
            valid_prompts = [p for p in prompts_to_add if p <= max_frame_idx]
            
            if not valid_prompts:
                 logger.warning(f"No valid prompt frames found (prompts: {prompts_to_add}, max_frame: {max_frame_idx}). Defaulting to 0.")
                 valid_prompts = [0]
            
            for f_idx in valid_prompts:
                try:
                    logger.info(f"Adding text prompt '{prompt_text}' to frame {f_idx}...")
                    _ = self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=f_idx,
                            text=prompt_text,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not add prompt to frame {f_idx}: {e}")
            
            # 3. Propagate
            logger.info("Propagating segmentation...")
            # We only need to store results for keyframes
            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = response["frame_index"]
                
                # Check if this is a keyframe we care about
                if frame_idx % keyframe_interval == 0:
                    results[frame_idx] = response["outputs"]
                    
                    # --- Debug Visualization ---
                    try:
                        import cv2
                        chunk_dir = Path(frames_path)
                        chunk_name = chunk_dir.name
                        session_dir = chunk_dir.parent.parent
                        debug_dir = session_dir / "output" / "debug_masks" / chunk_name
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        
                        outputs = response["outputs"]
                        
                        # Load original image
                        frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png'))])
                        if frame_idx < len(frame_files):
                            img_path = os.path.join(frames_path, frame_files[frame_idx])
                            img = cv2.imread(img_path)
                            
                            # Key is 'out_binary_masks'
                            # 'out_binary_masks': [N, H, W] or [1, H, W]
                            if 'out_binary_masks' in outputs:
                                mask = outputs['out_binary_masks']
                                if hasattr(mask, 'cpu'): mask = mask.cpu().numpy()
                                
                                # DEBUG LOGGING
                                if mask.size > 0:
                                    logger.info(f"Frame {frame_idx} Mask Stats: Shape={mask.shape}, Min={mask.min():.3f}, Max={mask.max():.3f}")
                                else:
                                    logger.warning(f"Frame {frame_idx}: Mask is empty/zero-size.")
                                
                                
                                # If shape [N, H, W], flatten/max
                                if mask.ndim == 3:
                                    # Take union - CHECK FOR EMPTY
                                    if mask.size > 0:
                                        mask = np.max(mask, axis=0) 
                                    else:
                                        # Handle empty mask case
                                        logger.warning(f"Frame {frame_idx}: Mask is empty (size 0).")
                                        continue # Skip visualization for this frame
                                    
                                if mask.shape[:2] != img.shape[:2]:
                                    mask = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                                
                                # Apply color overlay (Green)
                                color_mask = np.zeros_like(img)
                                color_mask[:, :, 1] = 255 # Green
                                
                                mask_bool = mask > 0.0
                                n_pixels = np.sum(mask_bool)
                                logger.info(f"Frame {frame_idx} Mask Pixels: {n_pixels}")
                                
                                if n_pixels > 0:
                                    img[mask_bool] = cv2.addWeighted(img[mask_bool], 0.5, color_mask[mask_bool], 0.5, 0)
                                else:
                                    logger.warning(f"Frame {frame_idx} has EMPTY mask!")
                                
                            save_path = debug_dir / f"frame_{frame_idx:03d}.jpg"
                            cv2.imwrite(str(save_path), img)
                    except Exception as e:
                        logger.warning(f"Failed to save debug mask for frame {frame_idx}: {e}")
                        import traceback
                        traceback.print_exc()
            
        except Exception as e:
            logger.error(f"Error during SAM3 processing: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 4. Reset/Close Session
            if session_id is not None:
                try:
                    self.predictor.handle_request(
                        request=dict(
                            type="reset_session",
                            session_id=session_id,
                        )
                    )
                except Exception as e:
                     logger.error(f"Error resetting session: {e}")
                     
        return results

    def process_full_sequence(self, frames_dir: str, prompt_text: str, 
                               keyframe_interval: int = None,
                               save_masks: bool = True) -> Dict[int, Any]:
        """
        Procesar la secuencia COMPLETA de frames (no por chunk).
        Esto evita segmentar el mismo frame múltiples veces debido al overlap.
        
        Args:
            frames_dir: Path al directorio de frames (frames/00000.jpg, 00001.jpg...)
            prompt_text: Texto del prompt para segmentación
            keyframe_interval: Intervalo para extraer máscaras
            save_masks: Si True, guarda máscaras en masks/frame_XXX.npz
            
        Returns:
            Dictionary mapping frame_index -> mask_data
        """
        if not self.is_loaded:
            self.load_model()
        
        if keyframe_interval is None:
            keyframe_interval = cfg["models"]["segmentation"]["keyframe_interval"]
        
        frames_path = Path(frames_dir)
        masks_dir = frames_path.parent / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        session_id = None
        results = {}
        
        try:
            logger.info(f"[SAM3-Full] Processing FULL sequence in {frames_dir} with prompt '{prompt_text}'")
            
            # Contar frames
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
            total_frames = len(frame_files)
            logger.info(f"[SAM3-Full] Found {total_frames} frames")
            
            # 1. Start Session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=frames_dir,
                )
            )
            session_id = response["session_id"]
            
            # 2. Add Prompts (a frames clave distribuidos)
            prompt_frames = cfg["models"]["segmentation"]["prompt_search_frames"]
            # Distribuir prompts a lo largo de la secuencia
            if total_frames > 100:
                # Para secuencias largas, añadir prompts adicionales
                step = total_frames // 5
                prompt_frames = list(set(prompt_frames + [0, step, step*2, step*3, step*4]))
            
            valid_prompts = [p for p in prompt_frames if p < total_frames]
            
            for f_idx in valid_prompts:
                try:
                    logger.info(f"[SAM3-Full] Adding prompt to frame {f_idx}")
                    self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=f_idx,
                            text=prompt_text,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not add prompt to frame {f_idx}: {e}")
            
            # 3. Propagate
            logger.info("[SAM3-Full] Propagating segmentation...")
            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = response["frame_index"]
                
                # Guardar en keyframes
                if frame_idx % keyframe_interval == 0:
                    outputs = response["outputs"]
                    results[frame_idx] = outputs
                    
                    # Guardar máscara a disco
                    if save_masks and 'out_binary_masks' in outputs:
                        mask = outputs['out_binary_masks']
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        
                        mask_path = masks_dir / f"frame_{frame_idx:05d}.npz"
                        np.savez_compressed(mask_path, mask=mask)
            
            # Guardar metadatos de segmentación
            meta_path = masks_dir / "segmentation_meta.json"
            import json
            with open(meta_path, 'w') as f:
                json.dump({
                    "prompt": prompt_text,
                    "keyframe_interval": keyframe_interval,
                    "total_frames": total_frames,
                    "masked_frames": list(results.keys())
                }, f, indent=2)
            
            logger.info(f"[SAM3-Full] Complete! {len(results)} frames with masks saved")
            
        except Exception as e:
            logger.error(f"Error during full sequence processing: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if session_id is not None:
                try:
                    self.predictor.handle_request(
                        request=dict(
                            type="reset_session",
                            session_id=session_id,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error resetting session: {e}")
        
        return results
    
    def process_batch(self, batch_dir: str, prompt_text: str,
                      index_mapping: Dict[int, int],
                      prompt_frames: List[int] = None) -> Dict[int, Any]:
        """
        Process a batch of frames in a temporary directory.
        
        Args:
            batch_dir: Path to directory with sequential batch frames (000000.jpg, ...)
            prompt_text: Text prompt for segmentation
            index_mapping: {batch_local_idx → original_frame_idx}
            prompt_frames: Local batch indices where to add prompts (None = auto)
        
        Returns:
            Dict[original_frame_idx → {"out_binary_masks": ndarray, "out_obj_ids": ndarray}]
        """
        if not self.is_loaded:
            self.load_model()
        
        batch_path = Path(batch_dir)
        batch_size = len(index_mapping)
        session_id = None
        results = {}
        
        try:
            logger.info(f"[SAM3-Batch] Processing {batch_size} frames from {batch_dir}")
            
            # 1. Start session
            response = self.predictor.handle_request(
                request=dict(type="start_session", resource_path=batch_dir)
            )
            session_id = response["session_id"]
            
            # 2. Add prompts at distributed frames
            if prompt_frames is None:
                # Auto-distribute: start, 1/3, 2/3, end
                if batch_size <= 10:
                    prompt_frames = [0]
                else:
                    step = batch_size // 4
                    prompt_frames = [0, step, step * 2, step * 3]
            
            for f_idx in prompt_frames:
                if f_idx >= batch_size:
                    continue
                try:
                    self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=f_idx,
                            text=prompt_text,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not add prompt to batch frame {f_idx}: {e}")
            
            # 3. Propagate (save ALL frames, keyframe_interval=1)
            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                local_idx = response["frame_index"]
                outputs = response["outputs"]
                
                # Map back to original frame index
                orig_idx = index_mapping.get(local_idx, local_idx)
                
                # Convert tensors to numpy
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if hasattr(mask, 'cpu'):
                        outputs["out_binary_masks"] = mask.cpu().numpy()
                if "out_obj_ids" in outputs:
                    oids = outputs["out_obj_ids"]
                    if hasattr(oids, 'cpu'):
                        outputs["out_obj_ids"] = oids.cpu().numpy()
                
                results[orig_idx] = outputs
            
            logger.info(f"[SAM3-Batch] Produced masks for {len(results)} frames")
            
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if session_id is not None:
                try:
                    self.predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id)
                    )
                except Exception as e:
                    logger.error(f"Error resetting session: {e}")
        
        return results
    
    def load_cached_masks(self, session_dir: Path, frame_indices: List[int] = None) -> Dict[int, Any]:
        """
        Cargar máscaras pre-calculadas desde masks/.
        
        Args:
            session_dir: Directorio base de la sesión
            frame_indices: Lista opcional de índices de frames a cargar
            
        Returns:
            Dictionary mapping frame_index -> mask_data
        """
        masks_dir = session_dir / "masks"
        if not masks_dir.exists():
            return {}
        
        results = {}
        mask_files = sorted(masks_dir.glob("frame_*.npz"))
        
        for mf in mask_files:
            try:
                # Extraer índice del nombre
                idx = int(mf.stem.split("_")[1])
                
                if frame_indices is not None and idx not in frame_indices:
                    continue
                
                data = np.load(mf)
                results[idx] = {"out_binary_masks": data["mask"]}
                
            except Exception as e:
                logger.warning(f"Failed to load mask {mf}: {e}")
        
        logger.info(f"[SAM3] Loaded {len(results)} cached masks from {masks_dir}")
        return results


# Singleton instance
_sam3_wrapper: Optional[SAM3Wrapper] = None

def get_sam3_wrapper() -> SAM3Wrapper:
    global _sam3_wrapper
    if _sam3_wrapper is None:
        _sam3_wrapper = SAM3Wrapper()
    return _sam3_wrapper
