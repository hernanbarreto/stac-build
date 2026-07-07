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

# Configure logging — avoid basicConfig() which adds duplicate handlers
# when Uvicorn already configures the root logger
logger = logging.getLogger("SAM3Wrapper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False
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
        logger.info("SAM3 Wrapper initialized (Lazy Loading Enabled: Model will load on first prompt).")
        
    def _stream(self, request: dict):
        """handle_stream_request under bf16 autocast.

        SAM 3.1's base predictor autocasts add_prompt internally but NOT
        propagate_in_video, and torch autocast is THREAD-LOCAL — the context
        entered at load time doesn't cover the executor thread running the
        propagation. Without this, bf16 memory features (created during
        add_prompt) hit fp32 conv biases and propagation dies with
        "Input type (c10::BFloat16) and bias type (float) should be the same".
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            yield from self.predictor.handle_stream_request(request=request)

    def _session_store(self) -> dict:
        """Predictor's session dict — SAM 3.0 exposes `_ALL_INFERENCE_STATES`,
        SAM 3.1's base predictor renamed it `_all_inference_states`. Both map
        session_id → {"state": inference_state, ...}."""
        if self.predictor is None:
            return {}
        store = getattr(self.predictor, "_ALL_INFERENCE_STATES", None)
        if store is None:
            store = getattr(self.predictor, "_all_inference_states", {})
        return store

    def load_model(self):
        """Lazy load the SAM3 model."""
        if self.is_loaded:
            return

        with self.lock:
            if self.is_loaded:
                return

            scfg = (cfg.get("models", {}) or {}).get("segmentation", {}) or {}
            version = str(scfg.get("version", "sam3"))
            logger.info(f"Loading SAM Model (version={version})...")
            try:
                if version == "sam3.1" and torch.cuda.is_available():
                    # SAM 3.1 Object Multiplex: joint multi-object tracking.
                    # Same handle_request / handle_stream_request API as 3.0,
                    # so everything below load_model() is version-agnostic.
                    # vendor_paths already put vendor/sam31 on sys.path.
                    from sam3.model_builder import build_sam3_multiplex_video_predictor
                    ckpt = scfg.get("checkpoint_path") or str(
                        vendor_paths.SAM31_WEIGHTS_DIR / "sam3.1_multiplex.pt")
                    if not Path(ckpt).exists():
                        raise FileNotFoundError(
                            f"SAM 3.1 checkpoint not found at {ckpt} — download "
                            "facebook/sam3.1 sam3.1_multiplex.pt to weights/sam31/")
                    self.predictor = build_sam3_multiplex_video_predictor(
                        checkpoint_path=ckpt,
                        max_num_objects=int(scfg.get("max_num_objects", 32)),
                        multiplex_count=int(scfg.get("multiplex_count", 16)),
                        use_fa3=bool(scfg.get("use_fa3", False)),
                        compile=bool(scfg.get("compile", False)),
                    )
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                    logger.info("SAM 3.1 Multiplex loaded on GPU (bfloat16 autocast active).")
                elif torch.cuda.is_available():
                    # SAM 3.0 GPU path: MultiGPU predictor with bfloat16 autocast
                    from sam3.model_builder import build_sam3_video_predictor
                    gpus_to_use = [torch.cuda.current_device()]
                    self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                    logger.info("SAM3 Model loaded on GPU (bfloat16 autocast active).")
                else:
                    if version == "sam3.1":
                        logger.warning("SAM 3.1 requires CUDA — falling back to SAM 3.0 CPU path")
                    # CPU path: use single-device Sam3VideoPredictor (no MultiGPU)
                    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
                    self.predictor = Sam3VideoPredictor()
                    logger.info("SAM3 Model loaded on CPU (float32, slower but functional).")

                self.is_loaded = True
                
            except Exception as e:
                logger.error(f"Failed to load SAM3 model: {e}")
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
            for response in self._stream(
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
            for response in self._stream(
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
                      prompt_frames: List[int] = None,
                      boxes_by_local: Dict[int, list] = None) -> Dict[int, Any]:
        """
        Process a batch of frames in a temporary directory.

        Args:
            batch_dir: Path to directory with sequential batch frames (000000.jpg, ...)
            prompt_text: Text prompt for segmentation
            index_mapping: {batch_local_idx → original_frame_idx}
            prompt_frames: Local batch indices where to add prompts (None = auto)
            boxes_by_local: Optional Phase 1 box seeds {local_idx: [[x,y,w,h], …]}
                (normalized 0..1). Seeded frames get text + bounding_boxes in the
                SAME add_prompt call (the patched predictor supports both), so
                multiple same-label instances are seeded individually instead of
                relying on the text prompt alone.

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
            
            # 2. Add prompts at distributed frames — prefer the frames that
            # carry Phase 1 box seeds (they pin individual instances)
            if prompt_frames is None:
                if boxes_by_local:
                    prompt_frames = sorted(boxes_by_local)[:6]
                elif batch_size <= 10:
                    prompt_frames = [0]
                else:
                    step = batch_size // 4
                    prompt_frames = [0, step, step * 2, step * 3]

            for f_idx in prompt_frames:
                if f_idx >= batch_size:
                    continue
                try:
                    request = dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=f_idx,
                        text=prompt_text,
                    )
                    seeds = (boxes_by_local or {}).get(f_idx)
                    if seeds:
                        request["bounding_boxes"] = [[float(c) for c in b] for b in seeds]
                        request["bounding_box_labels"] = [1] * len(seeds)
                    prompt_response = self.predictor.handle_request(request=request)
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
            for response in self._stream(
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
        
        with self.lock:
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=load_dir,
                )
            )
            session_id = response["session_id"]
        
        # SAM3's tracker pathway (point prompts) requires cached_frame_outputs
        # to exist for the target frame. The detector pathway (text prompts) 
        # doesn't need this. Since we allow point prompts as the first 
        # interaction, we must pre-seed empty caches.
        num_frames = 0
        try:
            session = self._session_store().get(session_id, {})
            inference_state = session.get("state", {})
            num_frames = inference_state.get("num_frames", 0)
            if "cached_frame_outputs" not in inference_state:
                inference_state["cached_frame_outputs"] = {}
            for fidx in range(num_frames):
                if fidx not in inference_state["cached_frame_outputs"]:
                    inference_state["cached_frame_outputs"][fidx] = {}
            logger.info(f"[SAM3-Interactive] Seeded tracker cache for {num_frames} frames")
        except Exception as e:
            logger.warning(f"[SAM3-Interactive] Could not seed cache: {e}")
        
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

    def get_mask_for_frame(self, state_id: str, frame_idx: int, obj_id: int = 1) -> Optional[np.ndarray]:
        """Get the current numpy mask for a specific frame and object from the cache."""
        if not self.is_loaded or self.predictor is None:
            return None
        with self.lock:
            session = self._session_store().get(state_id)
            if not session: 
                return None
            inference_state = session.get("state")
            if not inference_state:
                return None
            outputs = inference_state.get("cached_frame_outputs", {}).get(frame_idx)
            if not outputs: 
                return None
            mask_data = outputs.get(obj_id)
            if mask_data is None: 
                return None
            mask = mask_data.cpu().numpy() if hasattr(mask_data, 'cpu') else mask_data
            if mask.ndim == 3: 
                if mask.shape[0] == 0: return None
                mask = np.max(mask, axis=0)
            return mask

    def clear_interactive_prompts(self, state_id: str,
                                  obj_id: Optional[int] = None) -> bool:
        """
        Clear prompts WITHOUT destroying the session (frames stay loaded).

        obj_id given  → remove ONLY that object from tracking (SAM3's
                        remove_object) — the other queued objects survive.
        obj_id = None → reset the whole session (all tracked objects gone).

        Returns True on success.
        """
        if not self.is_loaded or self.predictor is None:
            return False

        try:
            with self.lock:
                if obj_id is not None:
                    self.predictor.handle_request(
                        request=dict(
                            type="remove_object",
                            session_id=state_id,
                            obj_id=int(obj_id),
                        )
                    )
                    logger.info(f"[SAM3-Interactive] Removed object {obj_id} "
                                f"from session {state_id}")
                    return True
                # Full reset: SAM3's official reset_session API
                self.predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=state_id,
                    )
                )

            logger.info(f"[SAM3-Interactive] Reset session {state_id} (prompts cleared)")
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

        # Dedup: if exact same prompt arrives again within 5s, return cached result
        import time as _time
        prompt_key = (state_id, frame_idx, obj_id, points.tobytes(), labels.tobytes())
        now = _time.time()
        if (hasattr(self, '_last_prompt_key') and self._last_prompt_key == prompt_key
                and (now - getattr(self, '_last_prompt_time', 0)) < 5.0):
            logger.info(f"[SAM3-Interactive] Dedup: skipping duplicate prompt for frame {frame_idx}")
            return getattr(self, '_last_prompt_result', {"success": True})
        self._last_prompt_key = prompt_key
        self._last_prompt_time = now

        try:
            logger.info(
                f"[SAM3-Interactive] Adding {len(points)} point(s) to frame {frame_idx}, "
                f"obj_id={obj_id}, labels={labels.tolist()}"
            )
            # SAM3 VideoPredictor.add_prompt expects List[List[float]] and List[int]
            # (see sam3_video_predictor.py line 137-138)
            pts = points.tolist() if hasattr(points, 'tolist') else list(points)
            pt_labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

            with self.lock:
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

            # Mark previous_stages_out so propagate_in_video knows this frame has prompts.
            # The detector pathway (text prompts) sets this automatically 
            # (sam3_video_inference.py line 399), but the tracker pathway (point prompts)
            # does NOT. We use the exact same dummy string SAM3 uses internally.
            try:
                session = self._session_store().get(state_id, {})
                inference_state = session.get("state", {})
                pso = inference_state.get("previous_stages_out")
                if pso is not None and frame_idx < len(pso):
                    pso[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
                    logger.info(f"[SAM3-Interactive] Marked frame {frame_idx} for propagation")
            except Exception as e:
                logger.warning(f"[SAM3-Interactive] Could not mark frame: {e}")

            # SAM 3.1 multiplex quirk: the request that REGISTERS the very
            # first object of a session initializes the tracker metadata and
            # returns EMPTY outputs (out_obj_ids=[]). Re-issuing the same
            # prompt once (idempotent — clear_old_points=True) returns the
            # real mask, so the first click previews like every other click.
            def _ids_of(resp):
                o = (resp or {}).get("outputs") or {}
                ids = o.get("out_obj_ids")
                if hasattr(ids, "cpu"):
                    ids = ids.cpu().numpy()
                return [] if ids is None else [int(i) for i in np.asarray(ids).ravel()]

            if obj_id not in _ids_of(response):
                logger.info(f"[SAM3-Interactive] obj {obj_id} missing from outputs "
                            "(first-object registration) — re-issuing prompt once")
                with self.lock:
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

            # Extract THIS object's mask for the preview. The multiplex
            # response contains every tracked object's mask — unioning them
            # (the old np.max) painted previous objects into the new one's
            # preview. Select by out_obj_ids; never union.
            result = {"success": True}
            if response and "outputs" in response:
                outputs = response["outputs"]
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3 and mask.shape[0] > 0:
                        ids = _ids_of(response)
                        if obj_id in ids:
                            mask = mask[ids.index(obj_id)]
                        elif len(ids) == 0 and mask.shape[0] == 1:
                            mask = mask[0]   # 3.0 path: single unlabeled mask
                        else:
                            mask = None
                    elif mask.ndim == 3:
                        mask = None
                    if mask is not None:
                        result["mask"] = mask  # [H, W] binary

            self._last_prompt_result = result
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
            with self.lock:
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

    def add_box_prompt(
        self,
        state_id: str,
        frame_idx: int,
        boxes_xywh: "list",
        obj_id: int = 1,
    ) -> dict:
        """
        Add BOX prompt(s) to a frame in an interactive session (Phase 1
        auto-prompter). Boxes drive SAM3's detector pathway.

        SAM3 box convention (vendor/sam31 .../sam3_video_inference.py:881-891):
            boxes are [xmin, ymin, width, height] in NORMALIZED 0..1 coords.
            A box that is a semantic prompt RESETS prior session state, so pass
            ALL boxes for this frame in one call; box and point prompts are
            mutually exclusive per call.

        Args:
            state_id: Session ID from init_interactive_session
            frame_idx: Frame index to add the prompt to
            boxes_xywh: list of [x, y, w, h], each normalized to 0..1
            obj_id: Object ID (single-object convenience; multi-box seeding uses
                    the batch/detector pathway to assign ids)

        Returns:
            Dict with 'success' and optionally 'mask' ([H, W] numpy).
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("SAM3 model not loaded")

        boxes = [[float(c) for c in b] for b in boxes_xywh]
        labels = [1] * len(boxes)  # 1 = foreground (0 short-circuits as no prompt)
        try:
            logger.info(
                f"[SAM3-Interactive] Box prompt on frame {frame_idx}: "
                f"{len(boxes)} box(es), obj_id={obj_id}"
            )
            with self.lock:
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=state_id,
                        frame_index=frame_idx,
                        obj_id=obj_id,
                        bounding_boxes=boxes,
                        bounding_box_labels=labels,
                    )
                )

            result = {"success": True}
            if response and "outputs" in response:
                outputs = response["outputs"]
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if hasattr(mask, "cpu"):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3:
                        mask = np.max(mask, axis=0)
                    result["mask"] = mask
                if "out_obj_ids" in outputs:
                    oids = outputs["out_obj_ids"]
                    result["obj_ids"] = oids.cpu().numpy().tolist() if hasattr(oids, "cpu") else list(oids)
            return result
        except Exception as e:
            logger.error(f"[SAM3-Interactive] Box prompt failed: {e}")
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

    def propagate_interactive_stream(self, state_id: str, selected_frames: Optional[List[int]] = None):
        """
        Generator that yields (frame_idx, num_frames, outputs) per frame.
        Enables SSE streaming of per-frame progress.
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("SAM3 model not loaded")

        # Get num_frames for progress calculation
        session = self._session_store().get(state_id, {})
        inference_state = session.get("state", {})
        num_frames = inference_state.get("num_frames", 0)
        # Fallback from session metadata
        if num_frames == 0:
            sess_meta = self._interactive_sessions.get(state_id, {})
            kf_map = sess_meta.get("keyframe_mapping", {})
            if kf_map:
                num_frames = len(kf_map)

        try:
            logger.info(f"[SAM3-Interactive] Propagating session {state_id} ({num_frames} frames), selected_frames={selected_frames is not None} ({len(selected_frames) if selected_frames else 0} entries)...")
            # NOTE: We intentionally do NOT hold self.lock here.
            # Propagation is a long-running generator that yields per-frame.
            # Holding a lock across yields would block ALL other SAM3 operations
            # for the entire duration (minutes). The server-side 409 guard already
            # prevents concurrent propagations on the same session.
            for response in self._stream(
                request=dict(
                    type="propagate_in_video",
                    session_id=state_id,
                    valid_frame_indices=selected_frames,
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
            pass  # No internal state manipulation needed (notebook pattern)

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
