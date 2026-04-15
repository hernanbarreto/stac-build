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
        self._interactive_sessions: Dict[str, dict] = {}  # state_id → session info dict
        self._sam_version: str = "sam3"  # Updated on load
        logger.info("SAM3 Wrapper initialized (Lazy Loading Enabled: Model will load on first prompt).")

    @property
    def is_multiplex(self) -> bool:
        """Whether the loaded model is SAM 3.1 Object Multiplex."""
        return self._sam_version == "sam3.1"

    def _get_inference_state(self, session_id: str) -> Optional[dict]:
        """
        Version-aware accessor for SAM3 internal inference state.
        SAM 3:   predictor._ALL_INFERENCE_STATES[session_id]["state"]
        SAM 3.1: predictor.inference_states[session_id]  (MultiplexVideoPredictor)
        Returns None if state cannot be accessed.
        """
        try:
            if self.is_multiplex:
                # SAM 3.1 MultiplexVideoPredictor stores states differently
                states = getattr(self.predictor, 'inference_states', None)
                if states and session_id in states:
                    return states[session_id]
                # Fallback: try the model's session storage
                model = getattr(self.predictor, 'model', None)
                if model:
                    tracker = getattr(model, 'tracker', None)
                    if tracker:
                        model_inner = getattr(tracker, 'model', None)
                        if model_inner and hasattr(model_inner, '_ALL_INFERENCE_STATES'):
                            session = model_inner._ALL_INFERENCE_STATES.get(session_id, {})
                            return session.get("state", session)
                return None
            else:
                # SAM 3 legacy
                session = self.predictor._ALL_INFERENCE_STATES.get(session_id, {})
                return session.get("state", {})
        except Exception as e:
            logger.warning(f"Could not access inference state for {session_id}: {e}")
            return None

    def load_model(self):
        """Lazy load the SAM3/3.1 model based on config.yaml version."""
        if self.is_loaded:
            return

        with self.lock:
            if self.is_loaded:
                return

            seg_cfg = cfg.get("models", {}).get("segmentation", {})
            sam_version = seg_cfg.get("version", "sam3.1")
            self._sam_version = sam_version

            logger.info(f"Loading SAM {sam_version} Model...")
            try:
                # SAM3 requires CUDA GPU — fail clearly if not available
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "SAM3 requires a CUDA GPU but none is available. "
                        "In Docker, ensure: docker run --gpus all ... "
                        "and NVIDIA Container Toolkit is installed."
                    )

                if sam_version == "sam3.1":
                    from sam3.model_builder import build_sam3_predictor
                    multiplex_count = seg_cfg.get("multiplex_count", 16)
                    max_num_objects = seg_cfg.get("max_num_objects", 32)
                    use_compile = seg_cfg.get("compile", False)
                    use_fa3 = seg_cfg.get("use_fa3", False)
                    ckpt_path = seg_cfg.get("checkpoint_path", None)

                    self.predictor = build_sam3_predictor(
                        version="sam3.1",
                        checkpoint_path=ckpt_path,
                        multiplex_count=multiplex_count,
                        max_num_objects=max_num_objects,
                        compile=use_compile,
                        use_fa3=use_fa3,
                    )
                    logger.info(
                        f"SAM 3.1 Object Multiplex loaded "
                        f"(multiplex_count={multiplex_count}, "
                        f"max_objects={max_num_objects}, "
                        f"compile={use_compile}, fa3={use_fa3})"
                    )
                else:
                    # Legacy SAM 3
                    from sam3.model_builder import build_sam3_video_predictor
                    gpus_to_use = [torch.cuda.current_device()]
                    self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
                    logger.info("SAM 3 (legacy) Model loaded successfully.")

                self.is_loaded = True

            except Exception as e:
                logger.error(f"Failed to load SAM {sam_version} model: {e}")
                import traceback
                traceback.print_exc()
                raise e

    def unload_model(self):
        """Unload model to free VRAM."""
        with self.lock:
            if self.predictor is not None:
                try:
                    self.predictor.model.cpu()
                except:
                    pass
                del self.predictor
                self.predictor = None
                self.is_loaded = False
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"CUDA cleanup warning (non-fatal): {e}")
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
            _vram_before = 0
            if torch.cuda.is_available():
                _vram_before = torch.cuda.memory_allocated() / (1024**3)
            
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
                    prompt_response = self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=f_idx,
                            text=prompt_text,
                        )
                    )
                    # Log what SAM3 detected at the prompt frame
                    if prompt_response:
                        n_objs = 0
                        if "out_obj_ids" in prompt_response:
                            ids = prompt_response["out_obj_ids"]
                            n_objs = len(ids) if hasattr(ids, '__len__') else 0
                        has_mask = "out_binary_masks" in prompt_response
                        logger.info(f"[SAM3-Batch] Prompt '{prompt_text}' @ frame {f_idx}: {n_objs} objects, has_mask={has_mask}")
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
            
            _vram_after = 0
            if torch.cuda.is_available():
                _vram_after = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"[SAM3-Batch] Produced masks for {len(results)} frames (VRAM: {_vram_before:.2f}→{_vram_after:.2f} GB)")
            
        except Exception as e:
            is_oom = "out of memory" in str(e).lower()
            logger.error(f"Error during batch processing: {e}")
            import traceback
            traceback.print_exc()
            if is_oom:
                raise  # Let caller handle OOM recovery
        finally:
            if session_id is not None:
                try:
                    self.predictor.handle_request(
                        request=dict(type="close_session", session_id=session_id)
                    )
                except Exception as e:
                    logger.error(f"Error closing session: {e}")
            # Free GPU tensors from propagation after each batch
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        return results
    
    # ── Interactive Segmentation (point-based prompts) ──────────────────

    def init_interactive_session(self, frames_dir: str,
                                  keyframes: List[str] = None) -> str:
        """
        Initialize an interactive SAM3 session for manual point-based segmentation.

        Args:
            frames_dir: Path to the directory containing frames (e.g. .../frames/)
            keyframes: Optional list of keyframe filenames (e.g. ['00010.jpg', '00016.jpg', ...]).
                       When provided, only these frames are loaded into SAM3 (via a temp dir
                       with sequential numbering), matching the batch pipeline approach.

        Returns:
            state_id (same as SAM3 session_id) used to reference this session.
        """
        import tempfile
        import shutil

        if not self.is_loaded:
            self.load_model()

        frames_path = Path(frames_dir)
        load_dir = frames_dir  # Default: load all frames
        keyframe_mapping = None  # SAM3 frame index → original filename

        # If keyframes provided, create temp dir with only those files
        if keyframes and len(keyframes) > 0:
            keyframes_sorted = sorted(keyframes)
            temp_dir = tempfile.mkdtemp(prefix="sam3_kf_")
            keyframe_mapping = {}
            for i, kf_name in enumerate(keyframes_sorted):
                src = frames_path / kf_name
                if src.exists():
                    # Sequential naming: 000000.jpg, 000001.jpg, ...
                    ext = src.suffix
                    dst = Path(temp_dir) / f"{i:06d}{ext}"
                    os.symlink(str(src), str(dst))
                    keyframe_mapping[i] = kf_name
            load_dir = temp_dir
            logger.info(f"[SAM3-Interactive] Created keyframe temp dir with {len(keyframe_mapping)} frames")
        
        logger.info(f"[SAM3-Interactive] Starting session for {load_dir}")
        
        # Cast model to float32 BEFORE loading frames.
        # The backbone caches features during start_session; if the model is in
        # BFloat16 those cached tensors will be BFloat16, causing type mismatches
        # later when conv layers (float32 bias) consume them.
        try:
            self.predictor.model.float()
            logger.info("[SAM3-Interactive] Cast model to float32")
        except AttributeError:
            # SAM 3.1 MultiplexVideoPredictor may not expose .model directly
            logger.info("[SAM3-Interactive] Skipped float32 cast (SAM 3.1 multiplex)")

        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=load_dir,
            )
        )
        session_id = response["session_id"]

        # Seed empty cached_frame_outputs for all frames (SAM 3 only).
        # SAM 3.1 handles this internally via the multiplex controller.
        num_frames = 0
        if not self.is_multiplex:
            try:
                inference_state = self._get_inference_state(session_id)
                if inference_state:
                    num_frames = inference_state.get("num_frames", 0)
                    if num_frames > 0 and "cached_frame_outputs" in inference_state:
                        for fidx in range(num_frames):
                            if fidx not in inference_state["cached_frame_outputs"]:
                                inference_state["cached_frame_outputs"][fidx] = {}
                        logger.info(f"[SAM3-Interactive] Seeded cache for {num_frames} frames")
            except Exception as e:
                logger.warning(f"[SAM3-Interactive] Could not seed cache: {e}")
        else:
            logger.info("[SAM3-Interactive] SAM 3.1 multiplex — cache seeding not needed")

        # Track the session with metadata for cleanup and frame mapping
        self._interactive_sessions[session_id] = {
            "session_id": session_id,
            "keyframe_mapping": keyframe_mapping,  # SAM3 idx → original filename
            "keyframe_temp_dir": load_dir if keyframes else None,
        }
        logger.info(f"[SAM3-Interactive] Session started: {session_id} "
                     f"({num_frames} frames, keyframes={'yes' if keyframes else 'no'})")

        return session_id

    def get_interactive_session_info(self, state_id: str) -> Optional[dict]:
        """Get session info including keyframe mapping, if available."""
        return self._interactive_sessions.get(state_id)

    def clear_interactive_prompts(self, state_id: str) -> bool:
        """
        Clear all prompts/tracked objects from an interactive session WITHOUT
        destroying the session.  The frames stay loaded in memory, and the
        user can immediately add new prompts.

        Returns True on success.
        """
        if not self.is_loaded or self.predictor is None:
            return False

        try:
            # SAM 3.1: use reset_session + re-start (cleanest approach)
            if self.is_multiplex:
                self.predictor.handle_request(
                    request=dict(type="reset_session", session_id=state_id)
                )
                logger.info(f"[SAM3-Interactive] Cleared prompts via reset_session (SAM 3.1)")
                return True

            # SAM 3 legacy: manually clear internal state
            inference_state = self._get_inference_state(state_id)
            if inference_state is None:
                logger.warning(f"[SAM3-Interactive] clear_prompts: session {state_id} not found")
                return False

            num_frames = inference_state.get("num_frames", 0)

            # Clear tracked objects
            if "obj_ids" in inference_state:
                inference_state["obj_ids"] = []
            if "obj_id_to_idx" in inference_state:
                inference_state["obj_id_to_idx"] = {}
            # Clear output caches
            if "previous_stages_out" in inference_state:
                for i in range(len(inference_state["previous_stages_out"])):
                    inference_state["previous_stages_out"][i] = None
            # Clear tracking results but keep frame features
            if "output_dict" in inference_state:
                for key in list(inference_state["output_dict"].keys()):
                    inference_state["output_dict"][key] = {}
            # Re-seed cached_frame_outputs (empty dicts for each frame)
            if "cached_frame_outputs" in inference_state:
                for fidx in range(num_frames):
                    inference_state["cached_frame_outputs"][fidx] = {}

            logger.info(f"[SAM3-Interactive] Cleared prompts for session {state_id} ({num_frames} frames kept)")
            return True
        except Exception as e:
            logger.error(f"[SAM3-Interactive] Failed to clear prompts: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_interactive_prompt(
        self,
        state_id: str,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """
        Add a point-based prompt (positive or negative click) to a specific frame.

        Args:
            state_id: Session ID from init_interactive_session
            frame_idx: Frame index to add the prompt to
            obj_id: Object ID (allows multiple objects)
            points: [N, 2] array of (x, y) coordinates
            labels: [N] array of 1 (positive) or 0 (negative)

        Returns:
            Dict with 'success' and optionally 'mask' (binary mask as [H, W] numpy)
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("SAM3 model not loaded")

        try:
            logger.info(
                f"[SAM3-Interactive] Adding {len(points)} point(s) to frame {frame_idx}, "
                f"obj_id={obj_id}, labels={labels.tolist()}"
            )
            # SAM3 API uses 'point_labels', not 'labels'
            pts = points.tolist() if hasattr(points, 'tolist') else points
            pt_labels = labels.tolist() if hasattr(labels, 'tolist') else labels

            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=state_id,
                    frame_index=frame_idx,
                    obj_id=obj_id,
                    points=pts,
                    point_labels=pt_labels,
                )
            )

            # Mark the frame as having outputs so propagate_in_video sees it.
            # SAM 3 only — tracker add_new_points doesn't set previous_stages_out.
            # SAM 3.1 handles this internally via the multiplex controller.
            if not self.is_multiplex:
                try:
                    inference_state = self._get_inference_state(state_id)
                    if inference_state:
                        pso = inference_state.get("previous_stages_out")
                        if pso is not None and frame_idx < len(pso):
                            pso[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
                            logger.info(f"[SAM3-Interactive] Marked frame {frame_idx} as having outputs for propagation")
                except Exception as e:
                    logger.warning(f"[SAM3-Interactive] Could not mark frame for propagation: {e}")

            # Extract mask from response for preview
            result = {"success": True}
            if response and "outputs" in response:
                outputs = response["outputs"]
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    # Shape is typically [N_obj, H, W] — take first object union
                    if mask.ndim == 3:
                        if mask.shape[0] == 0:
                            mask = None
                        else:
                            mask = np.max(mask, axis=0)
                    if mask is not None:
                        result["mask"] = mask  # [H, W] binary

                    # Also cache the mask in the inference state for propagation (SAM 3 only)
                    if not self.is_multiplex:
                        try:
                            inference_state = self._get_inference_state(state_id)
                            if inference_state:
                                cached = inference_state.get("cached_frame_outputs", {})
                                cached[frame_idx] = {obj_id: mask}
                        except Exception:
                            pass

            return result
        except Exception as e:
            logger.error(f"[SAM3-Interactive] Failed to add prompt: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def add_text_prompt(
        self,
        state_id: str,
        frame_idx: int,
        text: str,
        obj_id: int = 1,
    ) -> dict:
        """
        Add a text-based prompt to a specific frame in an interactive session.

        Args:
            state_id: Session ID from init_interactive_session
            frame_idx: Frame index to add the prompt to
            text: Text description of the object to segment
            obj_id: Object ID (allows multiple objects)

        Returns:
            Dict with 'success' and optionally 'mask' (binary mask as [H, W] numpy)
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("SAM3 model not loaded")

        try:
            logger.info(
                f"[SAM3-Interactive] Text prompt on frame {frame_idx}: '{text}', obj_id={obj_id}"
            )
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=state_id,
                    frame_index=frame_idx,
                    obj_id=obj_id,
                    text=text,
                )
            )

            result = {"success": True}
            if response and "outputs" in response:
                outputs = response["outputs"]
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3:
                        mask = np.max(mask, axis=0)
                    result["mask"] = mask

            return result
        except Exception as e:
            logger.error(f"[SAM3-Interactive] Text prompt failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def propagate_interactive_session(self, state_id: str) -> Dict[int, Any]:
        """
        Propagate the interactive prompts across the whole video.
        Returns all results at once (non-streaming).
        """
        results = {}
        for frame_idx, num_frames, outputs in self.propagate_interactive_stream(state_id):
            results[frame_idx] = outputs
        return results

    def propagate_interactive_stream(self, state_id: str):
        """
        Generator that yields (frame_idx, num_frames, outputs) per frame.
        Enables SSE streaming of per-frame progress.
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("SAM3 model not loaded")

        # Get num_frames for progress calculation
        num_frames = 0
        inference_state = self._get_inference_state(state_id)
        if inference_state:
            num_frames = inference_state.get("num_frames", 0)
        # Fallback from session metadata
        if num_frames == 0:
            sess_meta = self._interactive_sessions.get(state_id, {})
            kf_map = sess_meta.get("keyframe_mapping", {})
            if kf_map:
                num_frames = len(kf_map)

        try:
            logger.info(f"[SAM3-Interactive] Propagating session {state_id} ({num_frames} frames)...")
            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=state_id,
                )
            ):
                frame_idx = response["frame_index"]
                outputs = response["outputs"]

                # Convert tensors to numpy
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if hasattr(mask, 'cpu'):
                        outputs["out_binary_masks"] = mask.cpu().numpy()
                if "out_obj_ids" in outputs:
                    oids = outputs["out_obj_ids"]
                    if hasattr(oids, 'cpu'):
                        outputs["out_obj_ids"] = oids.cpu().numpy()

                yield frame_idx, num_frames, outputs

            logger.info(f"[SAM3-Interactive] Propagation complete")
        except Exception as e:
            logger.error(f"[SAM3-Interactive] Propagation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Re-seed cached_frame_outputs so subsequent add_prompt calls work.
            # SAM 3.1 handles this internally — only needed for SAM 3 legacy.
            if not self.is_multiplex:
                try:
                    inference_state = self._get_inference_state(state_id)
                    if inference_state:
                        cached = inference_state.get("cached_frame_outputs", {})
                        n_frames = inference_state.get("num_frames", 0)
                        for fidx in range(n_frames):
                            if fidx not in cached:
                                cached[fidx] = {}
                        logger.info(f"[SAM3-Interactive] Re-seeded cache for {n_frames} frames after propagation")
                except Exception as e:
                    logger.warning(f"[SAM3-Interactive] Could not re-seed cache: {e}")

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
