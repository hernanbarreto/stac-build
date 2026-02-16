"""
Scene Analyzer — InternVL3-based automatic scene inventory.
=============================================================

Analyzes representative frames from a scan to automatically detect
equipment/object categories, eliminating the need for manual prompt input.

Integration:
    Pipeline calls `analyze_scene(frames_dir)` BEFORE segmentation.
    Returns a semicolon-separated category string compatible with SAM3 prompts.

Model: OpenGVLab/InternVL3-8B (MIT license)
    - FP16 on >=16GB VRAM
    - INT8 on >=8GB VRAM
    - INT4 on <8GB VRAM
"""

import os
import sys
import json
import time
import logging
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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

    ≥16GB VRAM → InternVL3-8B in BF16 (~16GB)
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
        ).eval()
    finally:
        torch.linspace = _orig_linspace  # Always restore

    if device == "cuda":
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s")

    return model, tokenizer


# ── Frame sampling ───────────────────────────────────────────────────

def _sample_representative_frames(frames_dir: Path, n_frames: int = 5) -> List[str]:
    """
    Sample N representative frames evenly distributed across the scan.

    Strategy: Take frames at 0%, 25%, 50%, 75%, 100% of the sequence.
    This ensures coverage of start, middle, and end of the space being scanned.
    """
    exts = {".jpg", ".jpeg", ".png"}
    all_frames = sorted([
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in exts
    ])

    if len(all_frames) == 0:
        logger.warning(f"No frames found in {frames_dir}")
        return []

    if len(all_frames) <= n_frames:
        return [str(f) for f in all_frames]

    # Evenly spaced indices
    indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
    selected = [str(all_frames[i]) for i in indices]

    logger.info(f"Sampled {len(selected)} representative frames from {len(all_frames)} total")
    return selected


# ── Prompt engineering for construction + industrial scenes ───────────

SCENE_ANALYSIS_PROMPT = """<image>
You are an expert AEC (Architecture, Engineering, Construction) inspector analyzing a photo from a 3D LiDAR/photogrammetry scan of a building or construction site.

Your task: Identify ALL distinct segmentable elements visible in this image.
For each element, provide a SHORT DESCRIPTIVE PHRASE that would help a segmentation model isolate it precisely.

Cover ALL of these domains:

CONSTRUCTION & ARCHITECTURE:
- Surfaces: walls, floors, ceilings (describe material if visible: concrete wall, tiled floor, drop ceiling, drywall partition, exposed concrete slab, brick wall)
- Openings: doors, windows, skylights, hatches (describe type: glass door, fire door, sliding window, curtain wall)
- Structure: columns, beams, slabs, foundations, stairs, railings, ramps
- Finishes: baseboards, moldings, paint, tiles, carpeting

MEP (Mechanical, Electrical, Plumbing):
- Pipes: water pipes, drainage pipes, gas pipes, conduits (describe: exposed copper pipe, PVC drain pipe, insulated pipe)
- Ducts: air ducts, exhaust ducts, ductwork (describe: rectangular sheet metal duct, round flexible duct, insulated duct)
- Electrical: panels, cable trays, junction boxes, conduits, switches, outlets, light fixtures
- Plumbing: sinks, faucets, toilets, drains, water heaters
- HVAC: air handling units, diffusers, grilles, thermostats, split AC units, radiators

EQUIPMENT & OBJECTS:
- Industrial: pumps, motors, compressors, tanks, generators, transformers
- Safety: fire extinguishers, smoke detectors, sprinkler heads, exit signs, fire alarm panels
- Furniture: tables, chairs, desks, cabinets, shelving units
- Other: signage, access panels, hatches, ladders

Rules:
1. Return a JSON array of objects, each with "label" (short name) and "description" (2-5 word descriptive phrase for segmentation)
2. The "description" should be visual and specific enough for a segmentation AI to find the exact object
3. Include construction surfaces (walls, floor, ceiling) — they ARE important for BIM comparison
4. Maximum 25 entries
5. If multiple instances of same type exist with different materials, list them separately

Example output:
[
  {"label": "concrete wall", "description": "exposed gray concrete wall surface"},
  {"label": "floor", "description": "polished concrete floor"},
  {"label": "drop ceiling", "description": "suspended acoustic tile ceiling"},
  {"label": "pipe", "description": "exposed copper pipe running along ceiling"},
  {"label": "cable tray", "description": "metal cable tray with cables"},
  {"label": "electrical panel", "description": "gray metal electrical distribution panel"},
  {"label": "fire extinguisher", "description": "red fire extinguisher on wall mount"},
  {"label": "duct", "description": "rectangular sheet metal air duct"},
  {"label": "door", "description": "metal fire-rated door with push bar"},
  {"label": "column", "description": "reinforced concrete structural column"},
  {"label": "light fixture", "description": "fluorescent light fixture on ceiling"}
]

Output ONLY the JSON array, nothing else."""

MERGE_PROMPT = """Merge the following category lists from {n_frames} scan views into one unified inventory.

{categories_per_frame}

Rules:
1. Merge duplicates, keep the most descriptive version
2. Max 25 entries, prioritize most frequently seen
3. Return JSON array of {{"label": "...", "description": "..."}} objects

Output ONLY the JSON array."""



# ── Core analysis logic ──────────────────────────────────────────────

def _analyze_single_frame(model, tokenizer, image_path: str,
                           max_tiles: int = 6) -> List[Dict[str, str]]:
    """
    Analyze a single frame and extract object categories with descriptions.

    Returns:
        List of dicts: [{"label": "...", "description": "..."}, ...]
    """
    pixel_values = _load_image(image_path, max_num=max_tiles)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(dtype).to(device)

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    try:
        response = model.chat(
            tokenizer, pixel_values,
            SCENE_ANALYSIS_PROMPT,
            generation_config
        )

        # Parse JSON response
        categories = _parse_categories(response)
        labels = [c.get("label", "?") for c in categories]
        logger.info(f"  Frame {Path(image_path).name}: {len(categories)} items → {labels}")
        return categories

    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}")
        return []


def _merge_categories(model, tokenizer,
                      categories_per_frame: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    Merge categories from multiple frames, deduplicating by label.
    Keeps the most descriptive version for each label.
    """
    from collections import Counter

    # Collect all categories
    all_items = []
    for cats in categories_per_frame.values():
        all_items.extend(cats)

    # Group by normalized label, keep the longest description
    label_map = {}  # normalized_label → {"label": ..., "description": ...}
    label_counts = Counter()

    for item in all_items:
        label = item.get("label", "").lower().strip()
        desc = item.get("description", label)
        if not label:
            continue

        label_counts[label] += 1

        if label not in label_map:
            label_map[label] = {"label": label, "description": desc}
        else:
            # Keep the longer/more descriptive one
            if len(desc) > len(label_map[label]["description"]):
                label_map[label]["description"] = desc

    # Sort by frequency (most common first), cap at 25
    sorted_labels = sorted(label_map.keys(),
                          key=lambda x: label_counts[x], reverse=True)[:25]

    merged = [label_map[lbl] for lbl in sorted_labels]

    logger.info(f"Merged {len(all_items)} raw → {len(merged)} unique categories")
    return merged


def _parse_categories(response: str) -> List[Dict[str, str]]:
    """
    Parse VLM response into a list of category dicts.
    Handles:
      - JSON array of {"label": ..., "description": ...} objects
      - JSON array of plain strings (legacy)
      - Comma-separated fallback
    """
    response = response.strip()

    # Try JSON array first
    try:
        start = response.find('[')
        end = response.rfind(']')
        if start != -1 and end != -1:
            parsed = json.loads(response[start:end + 1])
            if isinstance(parsed, list):
                items = []
                for entry in parsed:
                    if isinstance(entry, dict):
                        label = str(entry.get("label", "")).strip()
                        desc = str(entry.get("description", label)).strip()
                        if label:
                            items.append({"label": label, "description": desc})
                    elif isinstance(entry, str) and entry.strip():
                        # Legacy plain string format
                        items.append({"label": entry.strip(), "description": entry.strip()})
                return items
    except json.JSONDecodeError:
        pass

    # Fallback: comma-separated strings
    if ',' in response:
        parts = response.split(',')
        return [{"label": p.strip().strip('"\'[]'), "description": p.strip().strip('"\'[]')}
                for p in parts if p.strip().strip('"\'[]')]

    # Fallback: newline-separated
    lines = response.split('\n')
    items = []
    for line in lines:
        line = line.strip().lstrip('-•*0123456789.)')
        line = line.strip().strip('"\'')
        if line and len(line) < 80:
            items.append({"label": line, "description": line})

    return items


# ── Public API ───────────────────────────────────────────────────────

def analyze_scene(frames_dir: str, config: dict = None) -> str:
    """
    Analyze a scene and return auto-detected segmentation categories.

    This is the main entry point. It:
    1. Samples 3-5 representative frames
    2. Loads InternVL3 with adaptive precision
    3. Analyzes each frame for object categories + descriptions
    4. Merges and deduplicates categories
    5. Unloads the model to free VRAM for DA3/SAM3
    6. Returns a semicolon-separated DESCRIPTION string for SAM3

    The descriptions are used as SAM3 text prompts (SAM3 understands
    descriptive phrases like "exposed concrete wall" or "rectangular metal duct").

    Args:
        frames_dir: Path to directory containing extracted frames
        config: Optional scene_analysis config dict

    Returns:
        Semicolon-separated description string for SAM3
        Returns empty string if analysis fails
    """
    config = config or {}
    enabled = config.get("enabled", True)
    if not enabled:
        logger.info("Scene analysis disabled in config")
        return ""

    frames_dir = Path(frames_dir).resolve()
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return ""

    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    n_frames = config.get("sample_frames", 5)
    max_tiles = config.get("max_tiles", 6)

    print(f"\n{'═' * 60}")
    print(f"  🔍 SCENE ANALYZER — InternVL3 Auto-Inventory")
    print(f"{'═' * 60}")
    print(f"  Model:  {model_id}")
    print(f"  Frames: {n_frames} representative samples")
    print(f"  Scope:  Construction + MEP + Equipment")
    print(f"{'═' * 60}\n")

    t0 = time.time()

    # ── Step 1: Sample frames ──
    selected_frames = _sample_representative_frames(frames_dir, n_frames)
    if not selected_frames:
        logger.error("No frames available for analysis")
        return ""

    # ── Step 2: Load model ──
    model, tokenizer = _load_model(model_id)

    # ── Step 3: Analyze each frame ──
    categories_per_frame = {}
    for i, frame_path in enumerate(selected_frames):
        print(f"  [{i + 1}/{len(selected_frames)}] Analyzing {Path(frame_path).name}...")
        cats = _analyze_single_frame(model, tokenizer, frame_path, max_tiles)
        categories_per_frame[frame_path] = cats

    # ── Step 4: Merge categories ──
    merged = _merge_categories(model, tokenizer, categories_per_frame)

    # ── Step 4b: Ensure mandatory structural categories are always present ──
    # The smaller VLM model may miss obvious surfaces. These are always relevant
    # for construction/BIM scenes and SAM3 will simply skip categories not visible.
    MANDATORY_STRUCTURAL = [
        {"label": "floor", "description": "floor surface"},
        {"label": "wall", "description": "wall surface"},
        {"label": "ceiling", "description": "ceiling surface"},
        {"label": "stairs", "description": "staircase or steps"},
        {"label": "door", "description": "door"},
    ]
    existing_labels = {item["label"].lower() for item in merged}
    # Also check descriptions for substring match (e.g., VLM detected "concrete wall" covers "wall")
    existing_text = " ".join(f"{item['label']} {item.get('description','')}" for item in merged).lower()
    for mandatory in MANDATORY_STRUCTURAL:
        # Skip if the mandatory keyword appears anywhere in existing labels/descriptions
        if any(mandatory["label"] in lbl for lbl in existing_labels) or mandatory["label"] in existing_text:
            continue
        merged.append(mandatory)
        logger.info(f"  Added mandatory structural: {mandatory['label']}")

    # ── Step 5: Unload model (free VRAM for SAM3) ──
    # Move to CPU first to release GPU tensors, then delete all references
    try:
        model.cpu()
    except:
        pass
    del model, tokenizer
    # Multiple gc.collect passes to break circular references
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"VLM unloaded — {vram_free:.1f} GB VRAM free")

    elapsed = time.time() - t0

    # ── Step 6: Format output ──
    # SAM3 gets the DESCRIPTIONS (richer text = better segmentation)
    prompt = ";".join(item["description"] for item in merged)

    print(f"\n{'─' * 60}")
    print(f"  ✅ Scene analysis complete in {elapsed:.1f}s")
    print(f"  📋 Detected {len(merged)} categories:")
    for i, item in enumerate(merged):
        print(f"     {i + 1:2d}. {item['label']:25s} → \"{item['description']}\"")
    print(f"\n  🏷️  SAM3 prompt ({len(merged)} categories):")
    print(f"     \"{prompt}\"")
    print(f"{'─' * 60}\n")

    # Save analysis results for reproducibility
    results = {
        "model_id": model_id,
        "n_frames_analyzed": len(selected_frames),
        "elapsed_seconds": round(elapsed, 2),
        "categories": merged,
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

    return prompt


# ── CLI interface ────────────────────────────────────────────────────

def main():
    """CLI: python scene_analyzer.py /path/to/frames [--model MODEL_ID] [--frames N]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="InternVL3 Scene Analyzer — Auto-detect segmentation categories"
    )
    parser.add_argument("frames_dir", help="Directory containing scan frames")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help=f"Model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--frames", type=int, default=5,
                        help="Number of representative frames to analyze (default: 5)")
    parser.add_argument("--max-tiles", type=int, default=6,
                        help="Max image tiles (6=dev, 12=production)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    config = {
        "enabled": True,
        "model_id": args.model,
        "sample_frames": args.frames,
        "max_tiles": args.max_tiles,
    }

    prompt = analyze_scene(args.frames_dir, config)

    if prompt:
        print(f"\n{'=' * 40}")
        print(f"RESULT: {prompt}")
        print(f"{'=' * 40}")
        # Also print in a format that can be piped
        sys.exit(0)
    else:
        print("ERROR: No categories detected", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
