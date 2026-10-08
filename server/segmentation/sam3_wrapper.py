import os
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
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

# ── detection / confirmation thresholds (models.segmentation.sam3_thresholds) ──
# USER 2026-09-29 ("debe segmentar todo"): SAM3 decides which detections become
# masklets by six numbers the vendor BUILDER hard-codes (no builder argument
# exposes them). They are declared per version in config.yaml — defaults EQUAL
# to the vendor's own values, so nothing changes until the user tunes them
# against output/segmentation_census.json — and written onto the built model
# object here, the one place every SAM3 model of this process is built. The
# model reads each of them from `self.<name>` at run time (verified in
# sam3_multiplex_base.py / sam3_video_base.py), so setting them after the build
# is equivalent to passing them to the constructor — except for the hotstart
# assertion the constructor runs, which `sam3_thresholds` re-checks.
SAM3_THRESHOLD_KEYS = ("score_threshold_detection", "new_det_thresh", "hotstart_delay",
                       "hotstart_unmatch_thresh", "hotstart_dup_thresh",
                       "masklet_confirmation_consecutive_det_thresh")
_SAM3_PROB_KEYS = ("score_threshold_detection", "new_det_thresh")


class SAM3ConfigError(RuntimeError):
    """A missing / invalid ``models.segmentation.sam3_thresholds`` key, or a
    vendor model that lacks one of the attributes it configures. It is NEVER a
    per-prompt failure: ``pipeline._run_sam3_batched`` re-raises it instead of
    skipping the category, so a bad key fails the stage naming it — instead of
    every prompt ending at zero masklets, which the census would read as
    "thresholds too strict"."""


class SAM3DeviceError(RuntimeError):
    """The configured SAM model cannot be built where this process runs (no CUDA):
    docs/plan_determinismo.md point 165 — the configured version is a REQUIREMENT, the
    CPU model of another version is never built in its place."""


class SAM3RunError(RuntimeError):
    """SAM3 failed inside a prompt (add_prompt, the propagation): the category's masks
    are NOT returned partially — point 92. Carries the phase and the prompt."""


class SAM3OutOfMemory(SAM3RunError):
    """CUDA out of memory inside a prompt (point 92: a failure — never a retry on a card
    whose free memory depends on what else was on it, never a skipped prompt)."""


# the vendor tracker's own message when, midway through a propagation, one of its multiplex
# states holds objects but no conditioning frame (sam3/model/video_tracking_multiplex_demo.py
# propagate_in_video). MEASURED 2026-10-08 on pccr 'pipe': deterministic — the same frame on
# every run, with or without the object cap, the association padding or torch's deterministic
# mode — so the frames before it are a complete, repeatable result and the rest of the batch
# is propagated again from there (segmentation.pipeline, the continuation batch)
VENDOR_STATE_WITHOUT_CONDITIONING = "No points are provided; please add points first"


class SAM3PropagationStopped(SAM3RunError):
    """The vendor tracker stopped midway through a propagation with
    :data:`VENDOR_STATE_WITHOUT_CONDITIONING`. ``partial`` holds the frames completed before
    it (``frames_done`` leading local frames, 0 … frames_done − 1, complete); the caller
    propagates the rest of the batch again from there. Any other error stays a failure."""

    def __init__(self, msg: str, partial: Dict[int, Any], frames_done: int):
        super().__init__(msg)
        self.partial = partial
        self.frames_done = int(frames_done)


SAM3_VERSIONS = ("sam3", "sam3.1")
# the seed of the deterministic torch state the model runs under — the one every GPU
# step of this repo uses (extract_da3_depth.DETERMINISTIC_SEED, 2026-10-07); SAM3's
# inference samples nothing, the seed only fixes what torch would otherwise draw
SAM3_DETERMINISTIC_SEED = 0


def segmentation_config() -> dict:
    """``models.segmentation`` of the config ``load_model`` builds from."""
    return (cfg.get("models", {}) or {}).get("segmentation", {}) or {}


def sam3_build_version(scfg: dict) -> str:
    """The model ``load_model`` BUILDS for ``scfg`` — the CONFIGURED version, and so the
    threshold block that applies. It no longer depends on CUDA (point 165): without a
    card ``load_model`` FAILS instead of building the 3.0 CPU model under a 3.1
    configuration. A version outside :data:`SAM3_VERSIONS` fails naming it."""
    version = str((scfg or {}).get("version", "sam3"))
    if version not in SAM3_VERSIONS:
        raise SAM3ConfigError(f"'models.segmentation.version' = {version!r} — expected one of "
                              f"{list(SAM3_VERSIONS)}")
    return version


def require_cuda(version: str) -> None:
    """The configured SAM runs on the GPU only; CUDA not visible = the stage fails (165)."""
    if not torch.cuda.is_available():
        raise SAM3DeviceError(
            f"SAM {version} is configured but CUDA is not visible to this process "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}) — the "
            f"configured version is a requirement; the CPU model is another model and is "
            f"never built in its place (docs/plan_determinismo.md point 165)")


def check_sam3_thresholds(scfg: Optional[dict] = None) -> Dict[str, Any]:
    """Validate the threshold block of the model that WILL be built (the SAM3
    worker calls this before any frame is touched; ``load_model`` before the
    vendor builder). Raises :class:`SAM3ConfigError` naming the key."""
    scfg = segmentation_config() if scfg is None else scfg
    return sam3_thresholds(scfg, sam3_build_version(scfg))


def sam3_thresholds(scfg: dict, version: str) -> Dict[str, Any]:
    """``models.segmentation.sam3_thresholds.<version>`` — strict: a missing
    key fails naming it, a value out of its range fails naming it
    (:class:`SAM3ConfigError`)."""
    where = f"models.segmentation.sam3_thresholds.{version}"
    block = (scfg or {}).get("sam3_thresholds")
    if not isinstance(block, dict) or not isinstance(block.get(version), dict):
        raise SAM3ConfigError(f"config.yaml is missing '{where}' — the SAM3 detection / "
                              f"confirmation thresholds of the model being built")
    sec = block[version]
    out: Dict[str, Any] = {}
    for k in SAM3_THRESHOLD_KEYS:
        if k not in sec:
            raise SAM3ConfigError(f"config.yaml is missing '{where}.{k}'")
        v = sec[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SAM3ConfigError(f"'{where}.{k}' must be a number, got {v!r}")
        if k in _SAM3_PROB_KEYS:
            if not 0.0 <= float(v) <= 1.0:
                raise SAM3ConfigError(f"'{where}.{k}' = {v} is a probability, outside [0, 1]")
            out[k] = float(v)
        else:
            if int(v) != v or int(v) < 0:
                raise SAM3ConfigError(f"'{where}.{k}' must be a non-negative integer, "
                                      f"got {v!r}")
            out[k] = int(v)
    if out["masklet_confirmation_consecutive_det_thresh"] < 1:
        raise SAM3ConfigError(f"'{where}.masklet_confirmation_consecutive_det_thresh' must "
                              f"be >= 1 (a masklet needs at least one detection to be "
                              f"confirmed)")
    if out["hotstart_delay"] > 0 and (out["hotstart_unmatch_thresh"] > out["hotstart_delay"]
                                      or out["hotstart_dup_thresh"] > out["hotstart_delay"]):
        raise SAM3ConfigError(f"'{where}': hotstart_unmatch_thresh and hotstart_dup_thresh "
                              f"must not exceed hotstart_delay (the vendor constructor "
                              f"asserts it)")
    return out


def _repo_relative(path) -> str:
    """A path relative to the repo root (a recorded path never carries the pod's prefix)."""
    try:
        return str(Path(path).resolve().relative_to(Path(vendor_paths._PROJECT_ROOT).resolve()))
    except (ValueError, OSError, AttributeError):
        return str(path)


def apply_sam3_thresholds(predictor, version: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``values`` (already validated by :func:`sam3_thresholds`) onto
    ``predictor.model`` and return ``{"version", "applied", "vendor_built"}``.
    A model that lacks one of the attributes (a vendor rename) FAILS the load
    (:class:`SAM3ConfigError`): the configured value would otherwise silently
    not apply — the bug class of 2026-09-23."""
    model = getattr(predictor, "model", None)
    if model is None:
        raise SAM3ConfigError(f"the SAM3 {version} predictor exposes no .model — the "
                              f"configured thresholds cannot reach it")
    missing = [k for k in values if not hasattr(model, k)]
    if missing:
        raise SAM3ConfigError(f"the SAM3 {version} model ({type(model).__name__}) has no "
                              f"attribute(s) {missing} — the vendor renamed them, and "
                              f"models.segmentation.sam3_thresholds.{version}."
                              f"{missing[0]} would silently not apply")
    built: Dict[str, Any] = {}
    for k, v in values.items():
        built[k] = getattr(model, k)
        setattr(model, k, v)
    logger.info("SAM3 %s thresholds (models.segmentation.sam3_thresholds.%s): %s",
                version, version,
                ", ".join(f"{k}={values[k]} (vendor built {built[k]})" for k in values))
    return {"version": version, "applied": values, "vendor_built": built}


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
        # ONE batch session kept open and reused across concepts (see _session_for).
        self._batch_session: Optional[Tuple[str, str]] = None  # (batch_dir, session_id)
        # what apply_sam3_thresholds wrote on the last model built (the census reads it)
        self.applied_thresholds: Optional[Dict[str, Any]] = None
        # the model built: version, checkpoint (path + sha256), device, dtype, the torch
        # numerics it runs under, the builder's arguments (points 91 / 165; the census and
        # segmentation.json record it)
        self.model_record: Optional[Dict[str, Any]] = None
        logger.info("SAM3 Wrapper initialized (Lazy Loading Enabled: Model will load on first prompt).")

    # ── Batch session reuse ──────────────────────────────────────────
    # SAM3's text pathway is one concept per pass: `add_prompt` starts with
    # `reset_state` ("since it's a semantic prompt, we start over"). But the
    # SESSION — the decoded frames — does not have to be rebuilt for each concept.
    # We used to open one session per concept, re-reading and re-decoding all N
    # frames from disk every time (~12 s × 46 concepts). The vendor's own
    # benchmark `forward()` does the opposite: init_state ONCE, then loop
    # add_prompt → propagate over the prompts. Same masks, a fraction of the I/O.

    def _session_for(self, batch_dir: str) -> str:
        """Session for this frame set, opened once and reused across concepts."""
        if self._batch_session and self._batch_session[0] == batch_dir:
            return self._batch_session[1]
        self.release_batch_session()          # only ever one open at a time
        response = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=batch_dir)
        )
        self._batch_session = (batch_dir, response["session_id"])
        logger.info(f"[SAM3-Batch] Opened session for {batch_dir} (reused across concepts)")
        return response["session_id"]

    def release_batch_session(self):
        """Close the reused batch session and free its frames. Call when the frame
        set changes, on error, and once segmentation is done."""
        if not self._batch_session:
            return
        _, session_id = self._batch_session
        self._batch_session = None
        try:
            self.predictor.handle_request(
                request=dict(type="close_session", session_id=session_id))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error closing batch session: {e}")
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        
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

            import repro
            scfg = segmentation_config()
            # the thresholds of the model about to be built are validated BEFORE
            # the vendor builder: a bad key fails in milliseconds, naming it,
            # instead of after a full model build per prompt (SAM3ConfigError is
            # re-raised by the per-category loop, never skipped)
            version = sam3_build_version(scfg)
            thresholds = sam3_thresholds(scfg, version)
            # the configured version is a requirement (point 165): no CUDA = no model
            require_cuda(version)
            # the cuBLAS workspace must be in the environment BEFORE this process's first
            # CUDA call (point 91): set here when CUDA is still untouched, refused when it
            # is already up without it — launch the process with repro.deterministic_env()
            repro.ensure_cublas_workspace()
            logger.info(f"Loading SAM Model (version={version})...")
            try:
                builder_args: Dict[str, Any] = {}
                ckpt_rec: Dict[str, Any]
                if version == "sam3.1":
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
                    # the cap: -1 = the vendor's no limit (point 90, DECIDIDO) — a cap that is
                    # reached drops objects by frame order, so there is none
                    builder_args = dict(
                        max_num_objects=int(scfg.get("max_num_objects", -1)),
                        multiplex_count=int(scfg.get("multiplex_count", 16)),
                        use_fa3=bool(scfg.get("use_fa3", False)),
                        compile=bool(scfg.get("compile", False)),
                    )
                    self.predictor = build_sam3_multiplex_video_predictor(
                        checkpoint_path=ckpt, **builder_args)
                    ckpt_rec = {"path": _repo_relative(ckpt), "sha256": repro.sha256_file(ckpt)}
                    self.applied_thresholds = apply_sam3_thresholds(
                        self.predictor, "sam3.1", thresholds)
                else:
                    # SAM 3.0 GPU path: MultiGPU predictor with bfloat16 autocast; the
                    # vendor builder resolves (auto-downloads) the checkpoint itself
                    from sam3.model_builder import build_sam3_video_predictor
                    gpus_to_use = [torch.cuda.current_device()]
                    builder_args = {"gpus_to_use": [int(g) for g in gpus_to_use]}
                    self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
                    ckpt_rec = {"path": None, "sha256": None,
                                "note": "resolved by the vendor builder (not a local file)"}
                    self.applied_thresholds = apply_sam3_thresholds(
                        self.predictor, "sam3", thresholds)
                # DETERMINISTIC NUMERICS after the build (point 91): the vendor turns TF32 on
                # at import and at the predictor's construction (sam3_multiplex_base.py:36,
                # sam3_multiplex_video_predictor.py:48) — off again here, deterministic
                # algorithms STRICT (an op without a deterministic kernel RAISES, never a
                # silent other result), cuDNN deterministic, seeds fixed. bf16 stays the
                # validated dtype (recorded, not changed).
                numerics = repro.enable_deterministic_torch(SAM3_DETERMINISTIC_SEED)
                torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                self.model_record = {
                    "version": version, "checkpoint": ckpt_rec, "device": "cuda",
                    "dtype": "bfloat16 (autocast)", "builder_args": builder_args,
                    "numerics": numerics,
                    "thresholds": dict(self.applied_thresholds or {}),
                }
                logger.info(f"SAM {version} loaded on GPU (bfloat16 autocast active; TF32 off, "
                            f"deterministic algorithms strict, seed {SAM3_DETERMINISTIC_SEED}; "
                            f"checkpoint sha256 {str(ckpt_rec.get('sha256'))[:12]}).")

                self.is_loaded = True

            except Exception as e:
                logger.error(f"Failed to load SAM3 model: {e}")
                import traceback
                traceback.print_exc()
                raise e

    def unload_model(self):
        """Unload model to free VRAM."""
        self.release_batch_session()   # its decoded frames belong to the predictor
        with self.lock:
            if self.predictor is not None:
                # BACKPORT of upstream facebookresearch/sam3 8f0b7f4 (2026-08-14,
                # "Restore autocast state on video predictor shutdown"): SAM 3.0's
                # tracker enters a LONG-LIVED bf16 autocast context at construction
                # and never exits it — after unload, unrelated torch code in the
                # same thread silently runs under bf16 (mixed-dtype native crash;
                # the backend segfaulted ~10 s after a refresh, 2026-08-29).
                try:
                    tracker = getattr(getattr(self.predictor, "model", None),
                                      "tracker", None)
                    ctx = getattr(tracker, "bf16_context", None)
                    if ctx is not None:
                        ctx.__exit__(None, None, None)
                        tracker.bf16_context = None
                        logger.info("SAM3 tracker bf16 autocast context exited "
                                    "(upstream 8f0b7f4 backport)")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"bf16 context cleanup warning (non-fatal): {e}")
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
        results = {}
        done_locals: List[int] = []
        phase = "start_session"

        try:
            logger.info(f"[SAM3-Batch] Processing {batch_size} frames from {batch_dir}")
            _vram_before = 0
            if torch.cuda.is_available():
                _vram_before = torch.cuda.memory_allocated() / (1024**3)

            # 1. Session for this frame set — opened once, reused for every concept
            session_id = self._session_for(batch_dir)

            # 2. ONE add_prompt. A text prompt is not tied to a frame ("text prompts
            # are NOT associated with a particular frame ... they apply to all frames",
            # sam3_video_inference.py:851) and every add_prompt calls reset_state, so
            # prompting at 4 spread frames only ever kept the LAST one — 3 wasted
            # forwards and 3 wasted resets per concept. Seed on the frame that carries
            # box seeds, if any; frame 0 otherwise.
            if prompt_frames:
                f_idx = next((f for f in prompt_frames if f < batch_size), 0)
            elif boxes_by_local:
                # the seeded frame with the most boxes — only one survives the reset
                f_idx = max((f for f in boxes_by_local if f < batch_size),
                            key=lambda f: len(boxes_by_local[f]), default=0)
                if len(boxes_by_local) > 1:
                    logger.info(f"[SAM3-Batch] {len(boxes_by_local)} seeded frames, "
                                f"using frame {f_idx}: add_prompt resets the session, "
                                f"so only one seed frame can survive")
            else:
                f_idx = 0

            # an add_prompt that fails is the prompt failing (point 92): it used to be a
            # warning, and the propagation then ran over a session with no prompt in it
            phase = "add_prompt"
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

            # 3. Propagate (save ALL frames, keyframe_interval=1)
            phase = "propagate_in_video"
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
                done_locals.append(int(local_idx))

            _vram_after = 0
            if torch.cuda.is_available():
                _vram_after = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"[SAM3-Batch] Produced masks for {len(results)} frames (VRAM: {_vram_before:.2f}→{_vram_after:.2f} GB)")
            
        except Exception as e:
            is_oom = "out of memory" in str(e).lower()
            logger.error(f"Error during batch processing ({phase}): {e}")
            import traceback
            traceback.print_exc()
            # the reused session may be in a bad state (and on OOM its frames are
            # the memory we need back) — drop it
            self.release_batch_session()
            # the vendor tracker's deterministic midway stop: the leading frames it completed
            # are a complete result and go back to the caller, who propagates the rest again
            # (only when they are the frames 0 … n−1 of the batch, propagated forward)
            if (phase == "propagate_in_video" and not is_oom
                    and VENDOR_STATE_WITHOUT_CONDITIONING in str(e)):
                n_done = len(done_locals)
                if n_done and sorted(done_locals) == list(range(n_done)):
                    raise SAM3PropagationStopped(
                        f"SAM3 propagation for prompt '{prompt_text}' stopped by the vendor tracker "
                        f"after {n_done} of {batch_size} frame(s) ({VENDOR_STATE_WITHOUT_CONDITIONING})",
                        partial=results, frames_done=n_done) from e
            # NEVER partial results (point 92): an error midway through the propagation
            # used to return the frames done so far as the category's masks, recorded as
            # 'ran'. The prompt fails, with its phase; an OOM is a failure like any other
            # (no retry on a card whose free memory depends on its co-tenants, no skip).
            cls = SAM3OutOfMemory if is_oom else SAM3RunError
            raise cls(f"SAM3 {phase} failed for prompt '{prompt_text}' "
                      f"({type(e).__name__}: {e})") from e
        finally:
            # NOTE: the session stays OPEN for the next concept — release_batch_session()
            # closes it when the frame set changes or segmentation ends.
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
                # Re-seed the tracker cache the point-prompt pathway needs
                # (same seeding as session init — a reset may wipe it, and
                # batched propagation / resume re-add point prompts right
                # after a reset).
                try:
                    session = self._session_store().get(state_id, {})
                    inference_state = session.get("state", {})
                    num_frames = inference_state.get("num_frames", 0)
                    if "cached_frame_outputs" not in inference_state:
                        inference_state["cached_frame_outputs"] = {}
                    for fidx in range(num_frames):
                        if fidx not in inference_state["cached_frame_outputs"]:
                            inference_state["cached_frame_outputs"][fidx] = {}
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[SAM3-Interactive] cache re-seed after "
                                   f"reset failed: {e}")

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
        _replay: bool = False,
    ) -> dict:
        """
        Add a point-based prompt (positive or negative click) to a specific frame.

        Every prompt is recorded in the session's prompt_log (unless it is a
        _replay of the log itself) so batched propagation can re-add SUBSETS
        of objects (USER 2026-09-06: 31 objects x 1329 frames OOM'd the
        whole 47 GB — objects must be processed in batches no matter how
        many the user marked).

        Args:
            state_id: Session ID from init_interactive_session
            frame_idx: Frame index to add the prompt to
            obj_id: Object ID (allows multiple objects)
            points: [N, 2] array of (x, y) coordinates
            labels: [N] array of 1 (positive) or 0 (negative)

        Returns:
            Dict with 'success' and optionally 'mask' (binary mask as [H, W] numpy)
        """
        if not _replay:
            _meta = self._interactive_sessions.get(state_id)
            if _meta is not None:
                _meta.setdefault("prompt_log", []).append({
                    "frame_idx": int(frame_idx), "obj_id": int(obj_id),
                    "points": np.asarray(points).tolist(),
                    "labels": np.asarray(labels).tolist()})
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

    def propagate_interactive_stream(self, state_id: str,
                                     selected_frames: Optional[List[int]] = None,
                                     output_prob_thresh: Optional[float] = None):
        """
        Generator that yields (frame_idx, num_frames, outputs) per frame.
        Enables SSE streaming of per-frame progress.

        USER ORDER 2026-08-29: propagation ALWAYS covers every keyframe.
        ``selected_frames`` is accepted for API compat but NOT forwarded —
        SAM 3.1's base predictor ignores ``valid_frame_indices`` anyway (only
        SAM 3.0 honored it), so forwarding it just hid the truth from the UI.
        ``output_prob_thresh`` overrides the tracker's output gate (vendor
        default 0.5): far from the prompted frame the object's score decays
        under it and SAM3 emits EMPTY masks (wall1 covered 4/12 frames) —
        0.0 emits the tracked mask wherever the tracker has one.
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
            if selected_frames:
                logger.info(f"[SAM3-Interactive] selected_frames ({len(selected_frames)}) "
                            "ignored — propagation always covers ALL frames (user order 2026-08-29)")
            logger.info(f"[SAM3-Interactive] Propagating session {state_id} ({num_frames} frames, "
                        f"output_prob_thresh={output_prob_thresh})...")
            # NOTE: We intentionally do NOT hold self.lock here.
            # Propagation is a long-running generator that yields per-frame.
            # Holding a lock across yields would block ALL other SAM3 operations
            # for the entire duration (minutes). The server-side 409 guard already
            # prevents concurrent propagations on the same session.
            _req = dict(type="propagate_in_video", session_id=state_id)
            if output_prob_thresh is not None:
                _req["output_prob_thresh"] = float(output_prob_thresh)
            for response in self._stream(request=_req):
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
            # RE-RAISE (2026-09-06): swallowing the error made a mid-way
            # OOM look like a normal end — partial masks were then saved
            # and flagged fully propagated. Callers decide what to keep.
            raise
        finally:
            pass  # No internal state manipulation needed (notebook pattern)

    def propagate_interactive_batched(self, state_id: str,
                                      output_prob_thresh: Optional[float] = None,
                                      obj_batch: int = 10):
        """Propagate in BATCHES of objects (USER 2026-09-06): the tracker's
        propagation state grows with objects x frames and 31 objects x 1329
        frames filled the whole 47 GB. No matter how many objects the user
        marked, at most ``obj_batch`` are propagated per pass: prompts are
        cleared, the batch's prompts re-added from the session prompt_log,
        the pass streamed, and per-frame outputs MERGED on CPU. Yields the
        same (frame_idx, num_frames, outputs) tuples as the plain stream —
        outputs carry the merged masks accumulated so far for that frame."""
        sess_meta = self._interactive_sessions.get(state_id, {})
        plog = sess_meta.get("prompt_log") or []
        obj_ids = sorted({int(p["obj_id"]) for p in plog})
        if not plog or len(obj_ids) <= obj_batch:
            yield from self.propagate_interactive_stream(
                state_id, output_prob_thresh=output_prob_thresh)
            return
        batches = [obj_ids[i:i + obj_batch]
                   for i in range(0, len(obj_ids), obj_batch)]
        logger.info(f"[SAM3-Interactive] BATCHED propagation: {len(obj_ids)} "
                    f"objects in {len(batches)} batch(es) of <= {obj_batch}")
        merged: Dict[int, dict] = {}
        if not hasattr(self, "batch_progress"):
            self.batch_progress = {}
        for bi, batch in enumerate(batches):
            logger.info(f"[SAM3-Interactive] ── object batch {bi + 1}/"
                        f"{len(batches)}: obj_ids {batch} ──")
            self.batch_progress[state_id] = {
                "batch": bi + 1, "n_batches": len(batches),
                "phase": "seeding", "frame": 0, "num_frames": 0,
                "objects": list(batch)}
            self.clear_interactive_prompts(state_id)
            # REAL GPU cleanup before every batch: a reset alone left 47 GB
            # resident after a failed pass and the next batch hung on its
            # first prompt (2026-09-06).
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"[SAM3-Interactive] GPU after cleanup: "
                            f"{torch.cuda.memory_allocated()/2**30:.1f} GB "
                            f"allocated")
            except Exception:  # noqa: BLE001
                pass
            bset = set(batch)
            n_fail = 0
            for pr in plog:
                if int(pr["obj_id"]) in bset:
                    res = self.add_interactive_prompt(
                        state_id, pr["frame_idx"], pr["obj_id"],
                        np.asarray(pr["points"], dtype=np.float32),
                        np.asarray(pr["labels"], dtype=np.int32),
                        _replay=True)
                    if isinstance(res, dict) and res.get("error"):
                        n_fail += 1
                        if n_fail >= 3:
                            raise RuntimeError(
                                f"batch {bi + 1}: prompts failing "
                                f"({res.get('error')}) — aborting instead "
                                f"of propagating garbage")
            _seen = 0
            for frame_idx, num_frames, outputs in \
                    self.propagate_interactive_stream(
                        state_id, output_prob_thresh=output_prob_thresh):
                _seen += 1
                self.batch_progress[state_id].update(
                    phase="propagating", frame=_seen, num_frames=num_frames)
                m = merged.get(frame_idx)
                bm = outputs.get("out_binary_masks")
                oi = outputs.get("out_obj_ids")
                if m is None:
                    merged[frame_idx] = {
                        "out_binary_masks": bm, "out_obj_ids": oi,
                        **{k: v for k, v in outputs.items()
                           if k not in ("out_binary_masks", "out_obj_ids")}}
                elif bm is not None and len(bm):
                    try:
                        m["out_binary_masks"] = (
                            np.concatenate([m["out_binary_masks"], bm])
                            if m.get("out_binary_masks") is not None
                            and len(m["out_binary_masks"]) else bm)
                        m["out_obj_ids"] = (
                            np.concatenate([np.asarray(m["out_obj_ids"]),
                                            np.asarray(oi)])
                            if m.get("out_obj_ids") is not None else oi)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"[SAM3-Interactive] merge failed on "
                                       f"frame {frame_idx}: {e}")
                yield frame_idx, num_frames, merged[frame_idx]
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        self.batch_progress.pop(state_id, None)
        logger.info(f"[SAM3-Interactive] BATCHED propagation complete "
                    f"({len(batches)} batches, {len(merged)} frames)")

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
