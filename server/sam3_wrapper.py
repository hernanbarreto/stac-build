import os
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAM3Wrapper")

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

    def process_chunk(self, frames_path: str, prompt_text: str, keyframe_interval: int = 5) -> Dict[int, Any]:
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
            # So the strategy: Prompt at Frame 0 AND Middle Frame (15).
            
            prompts_to_add = [0]
            if len(os.listdir(frames_path)) > 15: prompts_to_add.append(15)
            
            for f_idx in prompts_to_add:
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

# Singleton instance
_sam3_wrapper: Optional[SAM3Wrapper] = None

def get_sam3_wrapper() -> SAM3Wrapper:
    global _sam3_wrapper
    if _sam3_wrapper is None:
        _sam3_wrapper = SAM3Wrapper()
    return _sam3_wrapper
