"""
Scene Analyzer — InternVL3-based automatic scene inventory.
=============================================================

Analyzes ALL keyframes from a scan to automatically detect every object/surface
category visible, eliminating the need for manual prompt input.

Integration:
    Pipeline calls `analyze_scene(frames_dir)` BEFORE segmentation.
    Returns (prompt, frame_map) where frame_map maps category → list of frame filenames.

Model: OpenGVLab/InternVL3-8B (MIT license)
    - BF16 on >=16GB VRAM
    - BF16 on <16GB VRAM (2B model)
    - FP32 on CPU (2B model, slow)
"""

import os
import sys
import json
import re
import time
import logging
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import Counter

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger("SceneAnalyzer")

# ── Model configuration ──────────────────────────────────────────────

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3-8B"
FALLBACK_MODEL_ID = "OpenGVLab/InternVL3-2B"

# ImageNet normalization (required by InternVL3)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Image preprocessing (from InternVL3 official) ────────────────────

def _build_transform(input_size: int = 448):
    """Build preprocessing transform for InternVL3."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from the available tile configurations."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """
    InternVL3 dynamic resolution preprocessing.
    Splits image into tiles of image_size × image_size.
    max_num=6 for dev (saves VRAM), 12 for production.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def _load_image(image_path: str, input_size: int = 448, max_num: int = 6):
    """Load and preprocess a single image for InternVL3."""
    image = Image.open(image_path).convert('RGB')
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size,
                                 use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


# ── GPU detection and model loading ──────────────────────────────────

def _select_model_and_dtype(model_id: str = None):
    """
    Auto-select model size and dtype based on available GPU memory.

    >=16GB VRAM → InternVL3-8B in BF16 (~16GB)
    <16GB VRAM → InternVL3-2B in BF16 (~4GB) — fits comfortably
    No GPU     → InternVL3-2B in FP32 on CPU (slow but works)

    Returns:
        (model_id, torch_dtype, device)
    """
    SMALL_MODEL = "OpenGVLab/InternVL3-2B"

    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU — using 2B model on CPU (slow)")
        return SMALL_MODEL, torch.float32, "cpu"

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU memory: {gpu_mem_gb:.1f} GB")

    if gpu_mem_gb >= 16:
        logger.info(f"Using {model_id} in BF16 (enough VRAM)")
        return model_id, torch.bfloat16, "cuda"
    else:
        logger.info(f"GPU < 16GB — using {SMALL_MODEL} in BF16 to fit in {gpu_mem_gb:.0f}GB")
        return SMALL_MODEL, torch.bfloat16, "cuda"


def _load_model(model_id: str = None, precision: dict = None):
    """
    Load InternVL3 model and tokenizer.
    Uses direct loading (no device_map, no quantization) for maximum compatibility.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModel

    model_id, dtype, device = _select_model_and_dtype(model_id)

    logger.info(f"Loading {model_id} (dtype={dtype}, device={device})...")
    t0 = time.time()

    # Check if flash_attn is available
    try:
        import flash_attn
        use_flash = True
        logger.info("Flash Attention available — using it")
    except ImportError:
        use_flash = False
        logger.info("Flash Attention not available — using standard attention")
    # WORKAROUND: InternVL3's modeling_intern_vit.py calls torch.linspace().item()
    # during __init__, which fails when PyTorch's device context manager redirects
    # tensor creation to the "meta" device. We monkey-patch torch.linspace to
    # always create on CPU during model loading.
    _orig_linspace = torch.linspace
    def _safe_linspace(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return _orig_linspace(*args, **kwargs)
    
    torch.linspace = _safe_linspace
    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            use_flash_attn=use_flash,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
    finally:
        torch.linspace = _orig_linspace  # Always restore

    if device == "cuda":
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False,
        local_files_only=True,
    )

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s")

    return model, tokenizer


# ── Keyframe loading ─────────────────────────────────────────────────

def _load_keyframes(frames_dir: Path) -> List[str]:
    """
    Load the exact keyframes used for 3D reconstruction.
    
    REQUIRES selected_frames.json (visual novelty H/F keyframe filter).
    This file is produced by the frame_selector during reconstruction processing and
    contains the "selected_files" list — the keyframes used for the point cloud.
    
    Raises:
        FileNotFoundError: if selected_frames.json is missing
        ValueError: if the file is empty or cannot be parsed
    
    Returns:
        List of absolute frame paths (sorted)
    """
    selected_json = frames_dir / "selected_frames.json"
    
    if not selected_json.exists():
        raise FileNotFoundError(
            f"selected_frames.json not found in {frames_dir}. "
            f"Reconstruction must run first to produce keyframes before VLM analysis."
        )
    
    try:
        with open(selected_json) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError, PermissionError) as e:
        raise ValueError(
            f"Cannot read selected_frames.json: {e}. "
            f"Check that the file is not locked by another process."
        )
    
    # FIX: The JSON uses "selected_files", NOT "selected_frames"
    selected_names = data.get("selected_files", [])
    
    if not selected_names:
        raise ValueError(
            f"selected_frames.json has no 'selected_files' entries. "
            f"File may be corrupted or from an incompatible version."
        )
    
    frame_paths = []
    for name in selected_names:
        fp = frames_dir / name
        if fp.exists():
            frame_paths.append(str(fp))
        else:
            logger.warning(f"Keyframe listed but missing on disk: {name}")
    
    if not frame_paths:
        raise ValueError(
            f"selected_frames.json lists {len(selected_names)} keyframes "
            f"but none exist on disk in {frames_dir}"
        )
    
    logger.info(f"Loaded {len(frame_paths)}/{len(selected_names)} keyframes from selected_frames.json")
    return sorted(frame_paths)


# ── Default prompts (used when config.yaml doesn't specify them) ─────

_DEFAULT_SCENE_PROMPT = """<image>
You are an expert visual analyst for 3D scanning. Identify every UNIQUE physical object
and surface visible in this image.

MANDATORY categories (ALWAYS include if visible):
- floor (with material), wall (with material), ceiling, door, window

For each element provide:
- "label": short name (1-3 words, lowercase)
- "sam3_hint": 2-3 word specific phrase for a segmentation model
  (include material or color, e.g. "ceramic floor", "white wall")

Rules:
1. Return a JSON array of {"label": "...", "sam3_hint": "..."} objects
2. NEVER duplicate — same object = one entry
3. Maximum 20 entries per image
4. Structural elements (floor, wall, ceiling) are MANDATORY

Output ONLY the JSON array, nothing else."""

_DEFAULT_MERGE_PROMPT = """Merge the following category lists from {n_frames} scan views into one unified inventory.

{categories_per_frame}

Rules:
1. Merge duplicates and synonyms (e.g., "wooden chair" + "chair" = "chair")
2. Max {max_categories} entries — prioritize items seen in most frames
3. MANDATORY: floor, wall, ceiling must be in the final list if detected in any frame
4. Keep the best sam3_hint for each merged category
5. Return JSON array of {{"label": "...", "sam3_hint": "..."}} objects

Output ONLY the JSON array."""


# ── Core analysis logic ──────────────────────────────────────────────

def _analyze_single_frame(model, tokenizer, image_path: str,
                           max_tiles: int = 6,
                           prompt: str = None) -> List[Dict[str, str]]:
    """
    Analyze a single frame and extract object categories with descriptions.

    Args:
        prompt: VLM prompt to use (from config). Falls back to default.

    Returns:
        List of dicts: [{"label": "...", "description": "..."}, ...]
    """
    prompt = prompt or _DEFAULT_SCENE_PROMPT
    pixel_values = _load_image(image_path, max_num=max_tiles)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(dtype).to(device)

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    try:
        response = model.chat(
            tokenizer, pixel_values,
            prompt,
            generation_config
        )

        # Parse JSON response
        categories = _parse_categories(response)
        labels = [c.get("label", "?") for c in categories]
        logger.info(f"  Frame {Path(image_path).name}: {len(categories)} items → {labels}")
        return categories

    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def _normalize_label(label: str) -> str:
    """Normalize a label for comparison: lowercase, strip, remove articles."""
    label = label.lower().strip()
    # Remove leading articles/adjectives that cause false splits
    for prefix in ["a ", "an ", "the ", "some "]:
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label


def _labels_are_synonyms(a: str, b: str) -> bool:
    """Check if two labels refer to the same object (substring match).
    
    Examples:
        'wooden chair' and 'chair' → True
        'office desk' and 'desk' → True
        'floor' and 'wooden floor' → True
        'chair' and 'table' → False
    """
    a, b = _normalize_label(a), _normalize_label(b)
    if a == b:
        return True
    # One contains the other as a complete word
    return a in b or b in a


def _merge_categories(model, tokenizer,
                      categories_per_frame: Dict[str, List[Dict[str, str]]],
                      max_categories: int = 25,
                      required_categories: list = None) -> List[Dict[str, str]]:
    """
    Merge categories from multiple frames, deduplicating by label.
    Uses substring matching to detect synonyms.
    Force-injects required categories if missing.
    """
    # Collect all categories
    all_items = []
    for cats in categories_per_frame.values():
        all_items.extend(cats)

    # Group by normalized label, keep the best sam3_hint
    label_map = {}  # canonical_label → {"label": ..., "sam3_hint": ...}
    label_counts = Counter()
    # Map from any detected label → canonical label
    synonym_map = {}

    for item in all_items:
        label = item.get("label", "").lower().strip()
        hint = item.get("sam3_hint", item.get("description", label))
        if not label:
            continue

        # Check if this label is a synonym of an existing canonical label
        canonical = None
        for existing in label_map:
            if _labels_are_synonyms(label, existing):
                canonical = existing
                break

        if canonical is None:
            # New canonical label — use the shorter, simpler form
            canonical = label
            label_map[canonical] = {"label": canonical, "sam3_hint": hint}
        else:
            # Synonym found — keep the shorter label as canonical
            if len(label) < len(canonical):
                # New label is shorter → adopt it as canonical
                old_data = label_map.pop(canonical)
                old_count = label_counts.pop(canonical, 0)
                label_map[label] = {
                    "label": label,
                    "sam3_hint": old_data["sam3_hint"] if len(old_data["sam3_hint"]) > len(hint) else hint
                }
                label_counts[label] = old_count
                canonical = label
            else:
                # Keep existing canonical, but maybe update hint if more specific
                if len(hint) > len(label_map[canonical]["sam3_hint"]):
                    label_map[canonical]["sam3_hint"] = hint

        synonym_map[label] = canonical
        label_counts[canonical] += 1

    # Sort by frequency (most common first), cap at max_categories
    sorted_labels = sorted(label_map.keys(),
                          key=lambda x: label_counts[x], reverse=True)[:max_categories]

    merged = [label_map[lbl] for lbl in sorted_labels]

    # Force-inject required categories if missing
    if required_categories:
        existing_labels = {_normalize_label(m["label"]) for m in merged}
        for req in required_categories:
            req_label = _normalize_label(req.get("label", ""))
            # Check if any existing label is a synonym
            already_present = any(_labels_are_synonyms(req_label, el) for el in existing_labels)
            if not already_present:
                merged.append({
                    "label": req.get("label", req_label),
                    "sam3_hint": req.get("sam3_hint", req.get("description", req_label))
                })
                logger.info(f"Force-injected required category: {req_label}")

    logger.info(f"Merged {len(all_items)} raw → {len(merged)} unique categories "
                f"(synonym groups: {len(synonym_map)} → {len(label_map)})")
    return merged


def _build_frame_map(categories_per_frame: Dict[str, List[Dict[str, str]]],
                     merged: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Build frame_map: for each merged category, list exactly which frame files
    contain it.
    
    Uses fuzzy matching: a frame's detected category matches a merged category
    if the labels are equal after normalization, or if one contains the other.
    
    Returns:
        Dict mapping category label → sorted list of frame filenames
    """
    merged_labels = {item["label"].lower().strip() for item in merged}
    
    # For each frame, see which merged labels its detections match
    frame_map = {item["label"]: [] for item in merged}
    
    for frame_path, frame_cats in categories_per_frame.items():
        frame_name = Path(frame_path).name
        detected_labels = {c.get("label", "").lower().strip() for c in frame_cats}
        
        for item in merged:
            merged_lbl = item["label"].lower().strip()
            # Check if any detected label matches this merged label
            matched = False
            for det_lbl in detected_labels:
                if det_lbl == merged_lbl:
                    matched = True
                    break
                # Fuzzy: one contains the other (e.g., "concrete wall" matches "wall")
                if det_lbl in merged_lbl or merged_lbl in det_lbl:
                    matched = True
                    break
            
            if matched:
                frame_map[item["label"]].append(frame_name)
    
    # Sort frame lists
    for label in frame_map:
        frame_map[label] = sorted(set(frame_map[label]))
    
    return frame_map


# ── Parse VLM responses ──────────────────────────────────────────────

def _parse_categories(response: str) -> List[Dict[str, str]]:
    """
    Parse VLM response into a list of category dicts.
    Handles:
      - JSON wrapped in markdown code fences (```json ... ```)
      - Nested JSON objects like {"objects": [...]}
      - JSON array of {"label": ..., "description": ...} objects
      - JSON array of plain strings (legacy)
      - Comma-separated fallback
    """
    response = response.strip()

    # ── Step 1: Strip markdown code fences ────────────────────
    # VLM often wraps response in ```json ... ```
    fence_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)
    fence_match = fence_pattern.search(response)
    if fence_match:
        response = fence_match.group(1).strip()

    # ── Step 2: Try to parse as JSON ─────────────────────────
    parsed = None

    # Try full response as JSON first
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        pass

    # If that failed, try to extract JSON array or object
    if parsed is None:
        # Find the first { or [ to start of JSON
        for i, ch in enumerate(response):
            if ch in ('{', '['):
                try:
                    parsed = json.loads(response[i:])
                    break
                except json.JSONDecodeError:
                    # Try finding matching bracket from the end
                    closer = '}' if ch == '{' else ']'
                    j = response.rfind(closer)
                    if j > i:
                        try:
                            parsed = json.loads(response[i:j + 1])
                            break
                        except json.JSONDecodeError:
                            continue

    # ── Step 3: Extract items from parsed JSON ───────────────
    if parsed is not None:
        items_list = None

        # If it's already a list, use it directly
        if isinstance(parsed, list):
            items_list = parsed
        elif isinstance(parsed, dict):
            # Look for common nested keys: "objects", "categories", "items", "elements"
            for key in ("objects", "categories", "items", "elements", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    items_list = parsed[key]
                    break
            # If no known key, try the first list value in the dict
            if items_list is None:
                for v in parsed.values():
                    if isinstance(v, list):
                        items_list = v
                        break

        if items_list:
            items = []
            for entry in items_list:
                if isinstance(entry, dict):
                    label = str(entry.get("label", "")).strip()
                    # Support both sam3_hint (new) and description (legacy) fields
                    sam3_hint = str(entry.get("sam3_hint", entry.get("description", label))).strip()
                    # Validate label — skip if it looks like JSON garbage
                    if label and _is_valid_label(label):
                        items.append({"label": label, "sam3_hint": sam3_hint})
                elif isinstance(entry, str) and entry.strip():
                    cleaned = entry.strip()
                    if _is_valid_label(cleaned):
                        items.append({"label": cleaned, "sam3_hint": cleaned})
            if items:
                return items

    # ── Step 4: Fallback — comma-separated strings ───────────
    if ',' in response:
        parts = response.split(',')
        items = []
        for p in parts:
            cleaned = p.strip().strip("\"'[]{}()")
            if cleaned and _is_valid_label(cleaned):
                items.append({"label": cleaned, "sam3_hint": cleaned})
        if items:
            return items

    # ── Step 5: Fallback — newline-separated ─────────────────
    lines = response.split('\n')
    items = []
    for line in lines:
        line = line.strip().lstrip('-•*0123456789.)')
        line = line.strip().strip("\"'")
        if line and _is_valid_label(line):
            items.append({"label": line, "sam3_hint": line})

    return items


def _is_valid_label(label: str) -> bool:
    """Check if a string looks like a valid category label (not JSON garbage)."""
    label = label.strip()
    if not label or len(label) > 60:
        return False
    # Reject if it contains any JSON-related characters
    for ch in label:
        if ch in ('"', '\\', '{', '}', '[', ']'):
            return False
    # Reject markdown fences, JSON literals, colon patterns
    label_lower = label.lower()
    for tok in ('```', 'true', 'false', 'null', '":'):
        if tok in label_lower:
            return False
    # Reject if mostly punctuation
    punct_count = sum(1 for c in label if c in ':,;()/')
    if punct_count > len(label) * 0.3:
        return False
    return True


# ── Occlusion Classification ────────────────────────────────────────

# Known labels that are clearly temporary or permanent (avoid VLM call)
_KNOWN_TEMPORARY = {
    "scaffold", "scaffolding", "debris", "rubble", "tools", "tool",
    "ladder", "bucket", "tarp", "tarpaulin", "wheelbarrow", "pallet",
    "cone", "barrier", "formwork", "shoring", "temporary wall",
    "construction fence", "crane", "hoist", "generator", "compressor",
    "cable reel", "cable drum", "safety net", "protective sheet",
    "plastic sheet", "cardboard", "packaging", "trash", "waste",
}

_KNOWN_PERMANENT = {
    "wall", "floor", "ceiling", "door", "window", "column", "beam",
    "staircase", "stairs", "railing", "handrail", "cabinet",
    "kitchen cabinet", "sink", "toilet", "bathtub", "shower",
    "radiator", "light fixture", "outlet", "switch", "pipe",
    "duct", "hvac", "air conditioning", "elevator", "lift",
    "fire extinguisher", "smoke detector", "sprinkler",
    "countertop", "built-in closet", "partition wall",
}

_OCCLUSION_PROMPT = """You are a construction site analyst. For each object label below, classify it as either "temporary" or "permanent":

- TEMPORARY: objects that will be removed after construction (scaffolding, debris, tools, formwork, protective covers, construction equipment)
- PERMANENT: objects that are permanently installed and part of the final building (walls, fixtures, MEP, furniture, installed equipment)

Objects to classify:
{labels}

Return a JSON object mapping each label to its classification.
Example: {{"scaffold": "temporary", "kitchen_cabinet": "permanent"}}
Output ONLY the JSON object."""


def classify_occluders(
    occluder_labels: list,
    session_dir: str = None,
) -> dict:
    """
    Classify occluder objects as permanent or temporary.
    
    Uses a heuristic lookup first (fast, no VLM needed),
    then falls back to InternVL3 for unknown labels.
    
    Args:
        occluder_labels: unique labels from SAM3 segmentation
        session_dir: optional, to check cached VLM analysis
    
    Returns:
        {label: "permanent"|"temporary"} for each input label
    """
    if not occluder_labels:
        return {}

    result = {}
    unknown_labels = []

    # Step 1: Heuristic classification (fast path)
    for label in occluder_labels:
        label_lower = label.lower().strip()
        if not label_lower:
            continue

        # Check known lists (substring match)
        is_temp = any(k in label_lower or label_lower in k for k in _KNOWN_TEMPORARY)
        is_perm = any(k in label_lower or label_lower in k for k in _KNOWN_PERMANENT)

        if is_temp and not is_perm:
            result[label] = "temporary"
        elif is_perm and not is_temp:
            result[label] = "permanent"
        else:
            unknown_labels.append(label)

    if not unknown_labels:
        logger.info(f"[OcclusionClassifier] All {len(result)} labels classified by heuristic")
        return result

    # Step 2: Check cached VLM analysis (avoid re-loading model)
    if session_dir:
        cached = _classify_from_cached_vlm(unknown_labels, session_dir)
        for label, classification in cached.items():
            result[label] = classification
            unknown_labels.remove(label)

    if not unknown_labels:
        logger.info(f"[OcclusionClassifier] {len(result)} labels classified (heuristic + cache)")
        return result

    # Step 3: VLM classification for remaining unknowns
    logger.info(f"[OcclusionClassifier] {len(unknown_labels)} labels need VLM: {unknown_labels}")
    try:
        vlm_result = _classify_via_vlm(unknown_labels)
        result.update(vlm_result)
    except Exception as e:
        logger.warning(f"[OcclusionClassifier] VLM classification failed: {e}")
        # Default unknown labels to "temporary" (safer — triggers re-scan)
        for label in unknown_labels:
            if label not in result:
                result[label] = "temporary"

    logger.info(f"[OcclusionClassifier] Final: {result}")
    return result


def _classify_from_cached_vlm(labels: list, session_dir: str) -> dict:
    """Try to classify labels using existing scene_analysis.json."""
    from pathlib import Path
    analysis_path = Path(session_dir) / "scene_analysis.json"
    if not analysis_path.exists():
        return {}

    try:
        with open(analysis_path) as f:
            analysis = json.load(f)
    except Exception:
        return {}

    # Scene analysis has category descriptions that may hint at permanence
    categories = {c.get("label", "").lower(): c for c in analysis.get("categories", [])}
    result = {}

    for label in labels:
        label_lower = label.lower()
        if label_lower in categories:
            cat = categories[label_lower]
            hint = cat.get("sam3_hint", "").lower()
            # If the hint mentions construction-related terms, likely temporary
            temp_keywords = ["scaffold", "debris", "tool", "temporary", "construction"]
            perm_keywords = ["installed", "fixture", "built", "cabinet", "pipe"]
            if any(k in hint for k in temp_keywords):
                result[label] = "temporary"
            elif any(k in hint for k in perm_keywords):
                result[label] = "permanent"

    return result


def _classify_via_vlm(labels: list) -> dict:
    """Use InternVL3 to classify occluder labels as permanent/temporary."""
    model, tokenizer = _load_model()
    generation_config = dict(max_new_tokens=512, do_sample=False)

    prompt = _OCCLUSION_PROMPT.format(labels=", ".join(labels))

    try:
        # Use a blank image (VLM needs an image input even for text-only)
        blank = Image.new("RGB", (448, 448), (128, 128, 128))
        transform = _build_transform(448)
        pixel_values = transform(blank).unsqueeze(0)
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(dtype).to(device)

        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

        # Parse JSON response
        try:
            parsed = json.loads(response.strip().strip("`").replace("```json", "").replace("```", ""))
            if isinstance(parsed, dict):
                result = {}
                for label in labels:
                    val = parsed.get(label, parsed.get(label.lower(), "temporary"))
                    result[label] = "temporary" if "temp" in str(val).lower() else "permanent"
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: default to temporary
        return {label: "temporary" for label in labels}

    finally:
        # Unload to free VRAM
        try:
            model.cpu()
        except:
            pass
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ── Public API ───────────────────────────────────────────────────────

def analyze_scene(frames_dir: str, config: dict = None, on_progress=None) -> tuple:
    """
    Analyze a scene and return auto-detected segmentation categories.

    This is the main entry point. It:
    1. Loads keyframes from selected_frames.json (REQUIRED — errors if missing)
    3. Loads InternVL3 with adaptive precision
    4. Analyzes each keyframe using the configurable prompt
    5. Merges and deduplicates categories (synonym-aware + required_categories)
    6. Builds frame_map: category → list of frame filenames where detected
    7. Unloads the model to free VRAM for SAM3
    8. Returns (prompt, frame_map) for SAM3

    Args:
        frames_dir: Path to directory containing extracted frames
        config: Optional scene_analysis config dict (from config.yaml)

    Returns:
        (prompt, frame_map) tuple where:
          - prompt: semicolon-separated description string for SAM3
          - frame_map: dict mapping category label → list of frame filenames
        Returns ("", {}) if analysis fails
    """
    config = config or {}
    enabled = config.get("enabled", True)
    if not enabled:
        logger.info("Scene analysis disabled in config")
        return "", {}

    frames_dir = Path(frames_dir).resolve()
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return "", {}

    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    max_tiles = config.get("max_tiles", 6)


    max_merged_categories = config.get("max_merged_categories", 25)
    vlm_prompt = config.get("prompt", None)  # None → use default
    required_categories = config.get("required_categories", [])

    # ── Step 1: Load keyframes (STRICT — errors if selected_frames.json missing) ──
    keyframes = _load_keyframes(frames_dir)



    print(f"\n{'═' * 60}")
    print(f"  🔍 SCENE ANALYZER — InternVL3 Configurable Inventory")
    print(f"{'═' * 60}")
    print(f"  Model:  {model_id}")
    print(f"  Frames: {len(keyframes)} keyframes (from selected_frames.json)")
    print(f"  Caps:   {max_merged_categories} merged categories max")
    print(f"  Prompt: {'config.yaml' if vlm_prompt else 'default'}")
    print(f"  Required categories: {[r.get('label') for r in required_categories]}")
    print(f"{'═' * 60}\n")

    t0 = time.time()

    # ── Step 2: Load model ──
    if on_progress:
        on_progress(10, "Loading InternVL3 model...")
    model, tokenizer = _load_model(model_id)

    # ── Step 3: Analyze EACH keyframe ──
    categories_per_frame = {}
    for i, frame_path in enumerate(keyframes):
        fname = Path(frame_path).name
        pct = 20 + int(70 * i / max(len(keyframes), 1))
        if on_progress:
            on_progress(pct, f"Analyzing frame {i+1}/{len(keyframes)}: {fname}")
        print(f"  [{i + 1}/{len(keyframes)}] Analyzing {fname}...", end="", flush=True)
        cats = _analyze_single_frame(
            model, tokenizer, frame_path, max_tiles,
            prompt=vlm_prompt
        )
        categories_per_frame[frame_path] = cats
        if cats:
            print(f" → {len(cats)} objects")
        else:
            print(f" → (no objects detected)")

    # ── Step 4: Merge categories (synonym-aware + required injection) ──
    if on_progress:
        on_progress(90, "Merging and deduplicating categories...")
    merged = _merge_categories(
        model, tokenizer, categories_per_frame,
        max_categories=max_merged_categories,
        required_categories=required_categories
    )

    # ── Step 5: Build frame_map (category → frame filenames) ──
    frame_map = _build_frame_map(categories_per_frame, merged)

    # ── Step 6: Unload model (free VRAM for SAM3) ──
    try:
        model.cpu()
    except:
        pass
    del model, tokenizer
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"VLM unloaded — {vram_free:.1f} GB VRAM free")

    elapsed = time.time() - t0

    # ── Step 7: Format output ──
    # SAM3 works best with SHORT specific hints (2-3 words):
    #   e.g. "checkered armchair", "ceramic floor", "white wall"
    # NOT just generic labels ("chair", "floor") or verbose descriptions
    prompt = ";".join(item.get("sam3_hint", item["label"]) for item in merged)

    print(f"\n{'─' * 60}")
    print(f"  ✅ Scene analysis complete in {elapsed:.1f}s")
    print(f"  📋 Detected {len(merged)} categories:")
    for i, item in enumerate(merged):
        n_frames = len(frame_map.get(item["label"], []))
        hint = item.get('sam3_hint', item['label'])
        print(f"     {i + 1:2d}. {item['label']:25s} → \"{hint}\"  ({n_frames} frames)")
    print(f"\n  🏷️  SAM3 prompt ({len(merged)} categories):")
    print(f"     \"{prompt}\"")
    print(f"{'─' * 60}\n")

    # Save analysis results for reproducibility
    results = {
        "model_id": model_id,
        "n_keyframes_analyzed": len(keyframes),


        "max_merged_categories": max_merged_categories,
        "prompt_source": "config.yaml" if vlm_prompt else "default",
        "elapsed_seconds": round(elapsed, 2),
        "categories": merged,
        "frame_map": frame_map,
        "prompt": prompt,
        "per_frame": {
            Path(k).name: v for k, v in categories_per_frame.items()
        }
    }

    results_path = frames_dir.parent / "scene_analysis.json"
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        logger.warning(f"Could not save results: {e}")

    return prompt, frame_map


# ── CLI interface ────────────────────────────────────────────────────

def main():
    """CLI: python scene_analyzer.py /path/to/frames [--model MODEL_ID]"""
    import argparse

    parser = argparse.ArgumentParser(description="Scene Analyzer — InternVL3")
    parser.add_argument("frames_dir", help="Path to frames directory")
    parser.add_argument("--model", default=None, help="Model ID override")
    parser.add_argument("--max-tiles", type=int, default=6, help="Max tiles per image")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config = {}
    if args.model:
        config["model_id"] = args.model
    config["max_tiles"] = args.max_tiles

    prompt, frame_map = analyze_scene(args.frames_dir, config)

    if prompt:
        print(f"\n{'=' * 60}")
        print(f"PROMPT: {prompt}")
        print(f"\nFRAME MAP:")
        for label, frames in frame_map.items():
            print(f"  {label}: {len(frames)} frames → {frames[:5]}{'...' if len(frames) > 5 else ''}")
        print(f"{'=' * 60}")
    else:
        print("\n❌ No categories detected")
        sys.exit(1)


if __name__ == "__main__":
    main()
