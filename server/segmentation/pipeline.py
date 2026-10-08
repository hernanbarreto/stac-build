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
import time as _time
import cv2
import gc
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from atomic_io import atomic_write_json
from segmentation import mask_space

logger = logging.getLogger("SegPipeline")

# ═══════════════════════════════════════════════════════════════════
#  DETERMINISM (docs/plan_determinismo.md, wave 2 — 2026-10-08)
# ═══════════════════════════════════════════════════════════════════
#
# Every decision of this module that used to hang on a bare bar — one frame's IoU
# (point 93), a share of 0.5 (104), a count ratio and a ray majority (105), the
# footprint argmin of an OBB (109 / 143 / 145) — is now judged by THE USER'S RULE
# (point 1, `loop_utils/metric_lock.decide_change`): the judges' values exceed the
# bar significantly (the 95 % interval of their paired difference entirely above
# it), with at least `min_judge_closures(confidence)` judges (5 at 0.95), and by a
# margin of at least `correction_graph.graph.improvement_error_factor` × the
# MEASURED error. Otherwise the simplest outcome — separate, whole, every point —
# and the margin is recorded in the result (`decisions`). The user's own criteria
# (dedupe_overlap 0.5, min_points 1000, min_walk_m 1 m, mask_dedupe_iou 0.90, the
# four mask-filter rules) are UNCHANGED: only how close a case sits to them is
# measured and declared.

#: the staging directory the SAM3 stage builds its fresh store in (point 83); swapped
#: into output/ only when the whole run finished
SAM3_STAGING_DIRNAME = "_sam3_store.staging"
#: per-prompt seconds and the run's total — a separate record, never a compared artifact (point 125)
SEGMENTATION_TIMING_NAME = "segmentation_timing.json"
#: the identity of the octree beside output/potree (point 123: rebuilt only on a changed stamp)
POTREE_STAMP_NAME = "potree_stamp.json"
#: per-point view counts of the mask audit (point 119): judged views / views that left it outside
OUT_OF_PLACE_VIEWS_NAME = "out_of_place_views.npz"
#: the config sections the projection reads — the parts of the frozen run configuration it is
#: stamped with (point 123); a key missing from the configuration is a missing section, not a default
PROJECTION_CONFIG_SECTIONS = ("segmentation", "visualization", "correction_graph",
                              "models", "correction", "alignment")


def _fork_on_path() -> None:
    """vendor/VGGT-Long on sys.path (as precision/refine.py does) for `loop_utils`."""
    import sys as _sys
    from repro import FORK_ROOT
    p = str(FORK_ROOT)
    if p not in _sys.path:
        _sys.path.insert(0, p)


def _decision_params(raw_cfg: dict) -> Tuple[float, float, int]:
    """(error_factor, confidence, min_judges) of the user's rule, from the configuration
    the stage runs with: `correction_graph.graph.improvement_error_factor` (1.1, USER
    2026-10-07 "1.1 en todos los casos") and `heldout_confidence` (0.95); the judges
    minimum is `loop_judge.min_judge_closures(confidence)` (5 at 0.95)."""
    from reconstruction.loops.config import improvement_error_factor
    _fork_on_path()
    from loop_utils.loop_judge import min_judge_closures
    fac = float(improvement_error_factor(raw_cfg))
    try:
        conf = float(raw_cfg["correction_graph"]["graph"]["heldout_confidence"])
    except (KeyError, TypeError) as e:
        raise KeyError("config.yaml is missing correction_graph.graph.heldout_confidence") from e
    return fac, conf, int(min_judge_closures(conf))


def _robust_sigma(values) -> float:
    """The measured spread of the judges' values: 1.4826 × MAD (the normal-consistent
    scale, robust to one judge off) — the 'error' decide_change's margin is taken against
    when the quantity has no instrument error of its own."""
    v = np.asarray(values, np.float64).ravel()
    if v.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _decide_above_bar(values, bar: float, error: float, *, factor: float, confidence: float,
                      min_judges: int) -> dict:
    """THE USER'S RULE applied to 'do the judges' values exceed ``bar``':
    `decide_change(before=values, after=bar)` — paired difference value − bar,
    positive = above. `improves` is the verdict; the dict carries every margin."""
    _fork_on_path()
    from loop_utils.metric_lock import decide_change
    v = np.asarray(values, np.float64).ravel()
    out = decide_change(v, np.full(v.size, float(bar)), error=float(error),
                        error_factor=float(factor), confidence=float(confidence),
                        min_judges=int(min_judges))
    out["bar"] = float(bar)
    return out


def _stable_point_keys(P: np.ndarray, quantum_m: float = 1e-4) -> np.ndarray:
    """A deterministic key per point from its OWN coordinates (point 109: the sample of
    an object is chosen by a stable per-point key, never by position in a list that one
    extra point reorders): the coordinates quantised at ``quantum_m`` (0.1 mm — finer
    than any cloud of this pipeline resolves, so no two distinct points of an object
    collide except exact duplicates, which get the same key and the same fate), mixed
    by splitmix64. Sorting by the key is a fixed order of the point SET."""
    q = np.floor(np.asarray(P, np.float64) / float(quantum_m)).astype(np.int64)
    h = (q[:, 0].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
         ^ q[:, 1].astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
         ^ q[:, 2].astype(np.uint64) * np.uint64(0x94D049BB133111EB))
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


def _stable_sample(P: np.ndarray, n: int) -> np.ndarray:
    """The ``n`` points of ``P`` with the smallest stable keys (all of them when fewer)."""
    if len(P) <= int(n):
        return np.arange(len(P))
    keys = _stable_point_keys(P)
    order = np.argsort(keys, kind="stable")
    return np.sort(order[: int(n)])


def _pack_voxels(v: np.ndarray) -> np.ndarray:
    """World-anchored integer voxel coordinates (N, 3) → one int64 key each (21 bits per
    axis around the origin: ±1 048 576 cells, 52 km at 5 cm — a BOUND of the packing,
    checked, never a decision)."""
    v = np.asarray(v, np.int64)
    half = np.int64(1 << 20)
    if len(v) and (np.abs(v).max() >= half):
        raise ValueError(f"voxel coordinates reach {int(np.abs(v).max())} cells from the origin "
                         f"— beyond the {int(half)} the packing holds")
    return ((v[:, 0] + half) << np.int64(42)) | ((v[:, 1] + half) << np.int64(21)) | (v[:, 2] + half)


def _projection_config(output_dir: Path) -> Tuple[dict, str]:
    """(configuration, source) the projection runs with: the job's FROZEN
    `output/run_config.yaml` when the session holds one (point 69: every stage reads
    the frozen copy), else the backend's config.yaml (a session projected outside a
    pipeline job — the viewer's refresh); the source is recorded in the stamp."""
    try:
        from intake.run_config import has_run_config, load_run_config
        session_dir = Path(output_dir).parent
        if has_run_config(session_dir):
            doc, sha = load_run_config(session_dir)
            return doc, f"run_config.yaml {sha}"
    except ImportError:
        pass
    from config import cfg as _cfg
    return _cfg, "config.yaml"


def projection_stamp(output_dir, ply_path=None, cfg: Optional[dict] = None,
                     cfg_source: Optional[str] = None) -> dict:
    """The identity of a projection's product (point 123): every INPUT (the cloud, the
    raw mask store and its metadata, the keyframe list, the poses, the session camera,
    the floor transform and the record-grid declaration when they exist), the CODE of
    this module and of everything it calls to decide, and the configuration sections
    it reads. `segmentation_result.json` carries it; a result is reused only on an
    identical stamp, and the octree beside it is keyed the same way."""
    from repro import stamp as _stamp
    out = Path(output_dir)
    if cfg is None:
        cfg, cfg_source = _projection_config(out)
    ply = Path(ply_path) if ply_path is not None else out / "cleaned_cloud.ply"
    inputs = {"cloud": ply}
    for name in ("seg_masks.npz", "segmentation.json", "camera_frames.txt", "camera_poses.txt",
                 "camera.json", "floor_transform.npz", "corrected_cloud.json"):
        if (out / name).is_file():
            inputs[name] = out / name
    code = ["segmentation.pipeline", "segmentation.mask_filter", "segmentation.mask_space",
            "segmentation.fuse_parent", "segmentation.republish", "segmentation.session_io",
            "segmentation.shape_proposer", "segmentation.object_captioner",
            "reconstruction.surface_fit.hole_audit", "alignment_manager"]
    sections = {k: (cfg.get(k) if isinstance(cfg, dict) else None)
                for k in PROJECTION_CONFIG_SECTIONS}
    sections["config_source"] = str(cfg_source)
    return _stamp(inputs=inputs, code=code, config=sections)


def run_segmentation(frames_dir: str, output_dir: str, prompt: str,
                     frame_map: dict = None, on_progress=None,
                     boxes_map: dict = None, prompt_status: dict = None,
                     fallback_prompts: dict = None, defer_cloud_mapping: bool = False) -> dict:
    """
    Full segmentation pipeline: batched SAM3 → IoU ID matching → mask-to-point mapping.
    ``defer_cloud_mapping`` skips the in-run mask→cloud matching even when the cloud exists —
    the caller projects itself (the second VLM pass: the in-run full match took 1 h 52 min on
    pccr 2026-10-04 against the cloud stage's 23 min projection that followed it anyway).

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
        prompt_status: Optional dict the caller owns, filled IN PLACE with one
                   entry per category — ``ran`` / ``skipped`` / ``failed`` /
                   ``not_reached`` + reason (``_run_sam3_batched``). It survives
                   an exception, so the census can tell a prompt SAM3 never
                   completed from one it ran and that confirmed nothing.
    """
    from config import cfg

    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()

    seg_cfg = cfg["models"]["segmentation"]
    batch_size = seg_cfg.get("batch_size", 50)
    batch_overlap = seg_cfg.get("batch_overlap", 10)
    iou_threshold = seg_cfg.get("iou_match_threshold", 0.3)
    # SAM3 sometimes hands out several object ids for ONE observation; masks that
    # agree above this IoU inside a frame are collapsed (segmentation.mask_dedupe_iou)
    mask_dedupe_iou = float((cfg.get("segmentation", {}) or {}).get("mask_dedupe_iou", 0.90))

    # Split prompt by ';' for multi-category support
    categories = [c.strip() for c in prompt.split(";") if c.strip()]
    if not categories:
        return {"error": "Empty prompt", "instances": []}

    if prompt_status is not None:
        for c in categories:
            prompt_status[c] = {"status": "not_reached",
                                "reason": "the SAM3 run stopped before this prompt"}

    print(f"[SegPipeline] Starting segmentation for {len(categories)} categories: {categories}")
    print(f"[SegPipeline] Frames: {frames_dir}  |  Batch: {batch_size} frames, {batch_overlap} overlap")

    # ── Step 1: Prepare valid frames (matching reconstruction's blur + novelty filter) ──
    frame_sel_cfg = cfg.get("frame_selection", {})
    frame_stride = cfg.get("server", {}).get("frame_stride", 1)
    seg_frames_dir, frame_files, frames_valid_dir, keyframe_numbers = _prepare_valid_frames(
        frames_dir, frame_stride, frame_sel_cfg
    )

    total_frames = len(frame_files)
    print(f"[SegPipeline] Using {total_frames} valid frames for segmentation")

    if total_frames == 0:
        for c in (prompt_status or {}):
            prompt_status[c] = {"status": "skipped", "reason": "no keyframes to segment"}
        return {"error": "No frames found", "instances": []}

    # ── A FRESH STORE (docs/plan_determinismo.md point 83): SAM3 writes into a staging
    # directory — ids from the first masklet, no upsert into what a previous run left —
    # and the finished store replaces output/'s at the end, in one move. A run that fails
    # leaves the previous store exactly as it was (and its staging directory deleted).
    staging = output_dir / SAM3_STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(str(staging))
    staging.mkdir(parents=True)
    timing = {"prompts": {}, "seconds": None}
    t_run = _time.time()

    try:
        # ── Steps 2+3: SAM3 in batches with IoU matching, per category, each
        # category's masks SAVED as it finishes (frames_valid/ is numbered
        # 0,1,2… — SAM3 keys its masks by that KEYFRAME POSITION, not by the
        # video frame number). The per-category save IS the save: re-upserting
        # the whole run once more at the end shifted every id of the first
        # category by one (see _run_sam3_batched).
        all_masks, obj_labels, seg_meta = _run_sam3_batched(
            seg_frames_dir, frame_files, categories,
            batch_size, batch_overlap, iou_threshold, mask_dedupe_iou,
            output_dir=staging, cfg=cfg,
            frame_map=frame_map,
            on_progress=on_progress,
            boxes_map=boxes_map,
            prompt_status=prompt_status,
            fallback_prompts=fallback_prompts,
            session_output_dir=output_dir,
            timing=timing["prompts"],
        )

        if seg_meta is None:
            print("[SegPipeline] ⚠️ SAM3 produced no masks")
            shutil.rmtree(str(staging), ignore_errors=True)
            return {"error": "No masks generated", "instances": []}

        # the finished store: canonical member order, compact, sealed with the keyframe
        # list it was segmented on and the reconstruction it belongs to — then swapped in
        seg_meta = finalize_sam3_store(staging, output_dir, keyframe_numbers, categories, cfg,
                                       log=lambda m: print(f"[SegPipeline] {m}"))
        timing["seconds"] = round(_time.time() - t_run, 1)
        atomic_write_json(output_dir / SEGMENTATION_TIMING_NAME, timing, indent=1)

        # ── Step 4: Match masks to cloud and cache final result (ONCE) ──
        # In the anchored pipeline order (recon → vlm → sam3 → phase_r →
        # cloudcompy → tsdf) the cleaned cloud does not exist yet — the
        # mapping is DEFERRED to the cloudcompy stage, which calls
        # map_segmentation_to_cloud() after the (corrected) merge.
        if (output_dir / "cleaned_cloud.ply").exists() and not defer_cloud_mapping:
            result = _match_and_save_result(output_dir)
        else:
            print("[SegPipeline] ⏭ mask→cloud mapping deferred to the caller"
                  + ("" if defer_cloud_mapping else " (no cleaned_cloud.ply yet — the cloudcompy stage projects)"))
            result = {"deferred_cloud_mapping": True, "instances": []}

        # (Step 5 removed: the ShapeR PKL export is gone — MeshFlow mesh
        # generation runs on demand via /api/segmentation/shape/export.)

        if result.get("instances"):
            return result
        return seg_meta

    finally:
        # the memoized SAM3 symlink dirs (idempotent; _run_sam3_batched clears
        # them itself on the normal path, not when a category raises)
        _clear_batch_dirs()
        if staging.exists():                      # a run that did not finish: nothing of it stays
            shutil.rmtree(str(staging), ignore_errors=True)
        # ── Cleanup: vaciar frames_valid/ completamente ──
        if frames_valid_dir and frames_valid_dir.exists():
            shutil.rmtree(str(frames_valid_dir), ignore_errors=True)
            print(f"[SegPipeline] 🧹 frames_valid/ vaciado")


# ═══════════════════════════════════════════════════════════════════
#  THE SAM3 STORE: canonical, compact, sealed (points 83 / 101 / 114)
# ═══════════════════════════════════════════════════════════════════

#: the npz members that are not masks, in the order the canonical store writes them (last)
STORE_META_KEYS = ("obj_ids", "frames", "scaled_res", mask_space.NPZ_KEY, mask_space.KEYFRAMES_KEY,
                   "reconstruction_id")
#: zip entry time of every member of a canonical store (the zip format stores one; a clock
#: there made two identical stores differ in bytes)
STORE_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _mask_key_parts(key: str) -> Optional[Tuple[int, int]]:
    m = mask_space._MASK_KEY_RE.match(str(key))
    return (int(m.group(2)), int(m.group(1))) if m else None          # (oid, frame)


def write_canonical_store(path: Path, members: Dict[str, np.ndarray],
                          source=None) -> None:
    """Write a mask store with its members in CANONICAL order — every mask by
    (object, frame), then the metadata members in :data:`STORE_META_KEYS` order —
    compact (no dead bytes of a replaced entry) and with a fixed zip entry time,
    so one logical store is one byte sequence (point 101). ``members`` holds the
    arrays to write; ``source`` (an open NpzFile) supplies any mask named in
    ``members`` with the value None — read one at a time, never all in RAM.
    Atomic: a temporary file, then rename."""
    import zipfile
    path = Path(path)
    mask_keys = sorted((k for k in members if _mask_key_parts(k) is not None), key=_mask_key_parts)
    meta_keys = [k for k in STORE_META_KEYS if k in members]
    stray = sorted(set(members) - set(mask_keys) - set(meta_keys))
    if stray:
        raise ValueError(f"the mask store cannot hold members {stray[:5]} — not masks "
                         f"(f<frame>_o<oid>) nor known metadata")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for k in mask_keys + meta_keys:
                arr = members[k]
                if arr is None:
                    arr = source[k]
                import io as _io
                buf = _io.BytesIO()
                np.lib.format.write_array(buf, np.asanyarray(arr), allow_pickle=False)
                info = zipfile.ZipInfo(k + ".npy", date_time=STORE_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                zf.writestr(info, buf.getvalue())
        os.replace(tmp, path)
        if path.name == "seg_masks.npz":
            mask_space.invalidate(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_npz_canonical(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """An .npz with its members in NAME order, deflated, with the fixed zip entry
    time of :data:`STORE_ZIP_TIME` — np.savez_compressed stamps the wall clock
    into every entry, so one array set gave another byte sequence on every run.
    Atomic (temporary file, then rename)."""
    import io as _io
    import zipfile
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for k in sorted(arrays):
                buf = _io.BytesIO()
                np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
                info = zipfile.ZipInfo(k + ".npy", date_time=STORE_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                zf.writestr(info, buf.getvalue())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def compact_mask_store(path: Path) -> int:
    """Rewrite an existing store canonically (:func:`write_canonical_store`) from
    its own members, one at a time. Returns the number of masks."""
    path = Path(path)
    z = np.load(path)
    try:
        members: Dict[str, Optional[np.ndarray]] = {}
        for k in z.files:
            if _mask_key_parts(k) is not None:
                members[k] = None
            elif k in STORE_META_KEYS:
                members[k] = np.asarray(z[k])
            else:
                raise ValueError(f"{path} holds the member {k!r}, neither a mask nor known metadata")
        write_canonical_store(path, members, source=z)
        return sum(1 for k in members if _mask_key_parts(k) is not None)
    finally:
        z.close()


def sam3_store_seal(cfg: dict, categories: List[str], keyframes: List[int],
                    scaled_res, reconstruction_id: Optional[str], npz_path: Path) -> dict:
    """The seal of a store (point 83): the mask bytes (``npz_path``), the prompts,
    the keyframe list, the SAM3 model (version, checkpoint, thresholds as applied,
    the batch parameters) and the code that ran it — a `repro.stamp`."""
    from repro import stamp as _stamp
    scfg = dict((cfg.get("models") or {}).get("segmentation") or {})
    applied = None
    try:
        from segmentation.sam3_wrapper import get_sam3_wrapper
        applied = getattr(get_sam3_wrapper(), "applied_thresholds", None)
    except Exception:  # noqa: BLE001 — no wrapper (a synthetic run): recorded as None
        applied = None
    ckpt = scfg.get("checkpoint_path")
    ckpt_sha = None
    if ckpt and Path(str(ckpt)).is_file():
        from repro import sha256_file
        ckpt_sha = sha256_file(Path(str(ckpt)))
    seg = dict(cfg.get("segmentation") or {})
    return _stamp(inputs={"seg_masks.npz": Path(npz_path)},
                  code=["segmentation.pipeline", "segmentation.sam3_wrapper",
                        "segmentation.mask_space"],
                  config={"prompts": list(categories),
                          "keyframes": [int(k) for k in keyframes],
                          "sam3": {"version": scfg.get("version"), "checkpoint_path": ckpt,
                                   "checkpoint_sha256": ckpt_sha,
                                   "thresholds_configured": scfg.get("sam3_thresholds"),
                                   "thresholds_applied": applied,
                                   "batch_size": scfg.get("batch_size"),
                                   "batch_overlap": scfg.get("batch_overlap"),
                                   "iou_match_threshold": scfg.get("iou_match_threshold"),
                                   "multiplex_count": scfg.get("multiplex_count"),
                                   "max_num_objects": scfg.get("max_num_objects")},
                          "mask_dedupe_iou": seg.get("mask_dedupe_iou"),
                          "scaled_res": [int(x) for x in scaled_res],
                          "reconstruction_id": reconstruction_id})


def finalize_sam3_store(staging: Path, output_dir: Path, keyframes: List[int],
                        categories: List[str], cfg: dict, log=print) -> dict:
    """The staging store → the session's store: the npz rewritten canonically with
    the keyframe list and the reconstruction id as members (points 101 / 114 / 51),
    `segmentation.json` sealed (`stamp`, `reconstruction_id`, `keyframes_sha256`),
    both moved into ``output_dir`` replacing what was there, and the products of
    the previous store (the projection result, the fusion map, the broadcast)
    removed — they described another store. Returns the metadata document."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    from repro import sha256_json
    src_npz, src_json = staging / "seg_masks.npz", staging / "segmentation.json"
    if not (src_npz.exists() and src_json.exists()):
        raise RuntimeError(f"the SAM3 run saved no store in {staging}")
    rid = reconstruction_id_or_none(output_dir)
    z = np.load(src_npz)
    try:
        members: Dict[str, Optional[np.ndarray]] = {k: None for k in z.files
                                                    if _mask_key_parts(k) is not None}
        for k in STORE_META_KEYS:
            if k in z.files:
                members[k] = np.asarray(z[k])
        members[mask_space.KEYFRAMES_KEY] = np.asarray([int(k) for k in keyframes], np.int32)
        members["reconstruction_id"] = np.array(str(rid) if rid else "")
        scaled_res = [int(x) for x in np.asarray(members["scaled_res"]).ravel()]
        write_canonical_store(src_npz, members, source=z)
        n_masks = sum(1 for k in members if _mask_key_parts(k) is not None)
    finally:
        z.close()
    doc = json.loads(src_json.read_text())
    doc[RECONSTRUCTION_ID_KEY] = rid
    doc["keyframes_sha256"] = sha256_json([int(k) for k in keyframes])
    doc["n_keyframes"] = int(len(keyframes))
    doc["stamp"] = sam3_store_seal(cfg, categories, keyframes, scaled_res, rid, src_npz)
    atomic_write_json(src_json, doc, indent=1, sort_keys=True)
    # the swap: masks first, then the metadata that seals them (a reader that finds a
    # segmentation.json whose stamp names other mask bytes refuses it)
    for name in ("segmentation_result.json", "fusion_map.json", "seg_broadcast.json"):
        p = output_dir / name
        if p.exists():
            p.unlink()
    os.replace(src_npz, output_dir / "seg_masks.npz")
    os.replace(src_json, output_dir / "segmentation.json")
    mask_space.invalidate(output_dir)
    shutil.rmtree(str(staging), ignore_errors=True)
    log(f"store sealed: {n_masks} masks, {len(doc.get('instances') or [])} masklets, "
        f"{len(keyframes)} keyframes, reconstruction {str(rid)[:12] if rid else 'none'} — "
        f"replaced output/seg_masks.npz + segmentation.json")
    return doc


def check_store_seal(output_dir) -> List[str]:
    """What differs between `segmentation.json`'s seal and the store on disk — []
    for a sealed store that is intact; the reasons otherwise (no seal, other mask
    bytes). A store the interactive manager edited is unsealed by construction."""
    from repro import sha256_file
    out = Path(output_dir)
    p = out / "segmentation.json"
    if not p.exists():
        return ["no segmentation.json"]
    doc = json.loads(p.read_text())
    st = doc.get("stamp")
    if not isinstance(st, dict) or "sha256" not in st:
        return ["segmentation.json carries no seal (a store written before 2026-10-08, or "
                "edited by the interactive manager)"]
    want = (st.get("inputs") or {}).get("seg_masks.npz")
    npz = out / "seg_masks.npz"
    if not npz.exists():
        return ["seg_masks.npz is missing"]
    have = sha256_file(npz)
    if want != have:
        return [f"seg_masks.npz changed since it was sealed ({str(want)[:12]} -> {have[:12]})"]
    return []


def _prepare_valid_frames(frames_dir: Path, frame_stride: int = 1,
                          frame_sel_cfg: dict = None):
    """
    Copy valid frames to frames_valid/ with sequential numbering.

    The sequence number is the KEYFRAME POSITION (0, 1, 2 …) and it is the
    index SAM3's masks are keyed by, together with camera_poses.txt and the
    hole audit. It is NOT the cloud's ``frame_global``, which carries the REAL
    video frame number (1, 60, 97 … ) — this docstring used to claim the two
    matched, and the mask→cloud matcher believed it: on pccr 2026-09-14 only
    13 of 216 keyframes lined up, by numeric accident, and those 13 took the
    wrong keyframe's mask. ``_mask_frame_lookup`` translates between the two.
    
    Frame selection priority:
      1. selected_frames.json (visual novelty H/F filter) — if available
      2. frame_stride (fixed decimation) — fallback
    
    Args:
        frame_stride: Take 1 every N valid frames (fallback, from config.yaml)
        frame_sel_cfg: frame_selection config section (for novelty filter)
    
    Returns:
        (seg_frames_dir, frame_files, frames_valid_dir, keyframe_numbers) — the last one
        the VIDEO frame number of every position (the keyframe list the masks are
        segmented on, written into the store: docs/plan_determinismo.md point 114)
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
            return frames_dir, frame_files, None, [int(os.path.splitext(f)[0]) for f in frame_files]
        
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
        return frames_dir, [], None, []

    # Crear frames_valid/ limpio (borrar cualquier resto de runs anteriores)
    frames_valid_dir = frames_dir.parent / "frames_valid"
    if frames_valid_dir.exists():
        shutil.rmtree(str(frames_valid_dir))
    frames_valid_dir.mkdir()
    
    # Copy valid frames with sequential numbering (the KEYFRAME POSITION — see
    # the docstring: it is not the cloud's frame_global).
    #
    # The list is filtered to what is ON DISK *before* numbering: a missing
    # keyframe used to leave a hole in the sequence (seq_idx kept counting)
    # while the returned list closed it, so the mask key, the list index and
    # the keyframe position were three different numbers for every frame
    # after the hole — and the store declares a POSITION space.
    missing = [f for f in valid_filenames if not (frames_dir / f).exists()]
    if missing:
        print(f"[SegPipeline] ⚠️ {len(missing)} selected keyframe(s) are not on "
              f"disk (e.g. {missing[:3]}) — segmenting the {len(valid_filenames) - len(missing)} that are")
        valid_filenames = [f for f in valid_filenames if (frames_dir / f).exists()]
    index_mapping = {}
    seq_frame_files = []
    
    keyframe_numbers = []
    for seq_idx, orig_filename in enumerate(valid_filenames):
        src = frames_dir / orig_filename
        ext = src.suffix
        new_name = f"{seq_idx:06d}{ext}"
        dst = frames_valid_dir / new_name
        shutil.copyfile(str(src), str(dst))
        seq_frame_files.append(new_name)
        m = re.search(r"(\d+)", os.path.splitext(orig_filename)[0])
        if m is None:
            raise ValueError(f"keyframe file {orig_filename!r} carries no frame number — the "
                             f"store cannot declare the keyframe list it is segmented on")
        keyframe_numbers.append(int(m.group(1)))

    print(f"[SegPipeline] Copied {len(seq_frame_files)} valid frames to {frames_valid_dir}")

    return frames_valid_dir, seq_frame_files, frames_valid_dir, keyframe_numbers


# ═══════════════════════════════════════════════════════════════════
#  BATCHED SAM3 PROCESSING
# ═══════════════════════════════════════════════════════════════════

def _run_sam3_batched(frames_dir: Path, frame_files: List[str], categories: List[str],
                      batch_size: int, batch_overlap: int,
                      iou_threshold: float, mask_dedupe_iou: float,
                      output_dir: Path = None, cfg: dict = None,
                      frame_map: dict = None,
                      on_progress=None,
                      boxes_map: dict = None,
                      prompt_status: dict = None,
                      fallback_prompts: dict = None,
                      session_output_dir: Path = None,
                      timing: dict = None):
    """
    Process frames in overlapping batches, one category at a time.

    ``output_dir`` is where the store is WRITTEN (the staging directory of
    docs/plan_determinismo.md point 83); ``session_output_dir`` the session's
    output/ (the keyframe list and the frames' resolution are read there), the
    same directory when not given. ``timing`` (optional) receives the seconds per
    prompt — a separate record (point 125), never part of ``prompt_status``.
    Each category gets its own SAM3 pass; obj_ids are remapped to avoid collisions.

    With ``output_dir`` and ``cfg`` every category's masks are SAVED as the
    category finishes — only THAT category's masks, appended to the store
    (``_save_masks`` upsert). It used to re-upsert EVERY mask of the run after
    every category (and once more at the end): the first save gave the 1-based
    SAM3 ids store ids 0,1,2…, the second found 1,2… already "existing" and
    wrote raw id k over store id k — the first category left a duplicate object
    and chimeras (pccr 2026-09-29: SAM3 said 166 objects, the store held 167),
    and every save held two copies of all masks in RAM. A save that fails is
    RAISED: it is the only save, nothing re-writes those masks later.

    If frame_map is provided, each category only processes the frames listed
    for that category (from VLM analysis), creating a temp directory with
    consecutive numbering for SAM3 propagation.

    ``prompt_status`` (optional, filled in place): per category ``ran`` (with
    the objects / frames it produced and its seconds), ``skipped`` (no frames
    to run on) or ``failed`` (+ reason). A :class:`SAM3ConfigError` is never a
    per-category failure: it is re-raised and fails the run.

    Returns:
        (all_masks, obj_labels, seg_meta)
        - all_masks: Dict[orig_frame_idx, {global_obj_id: binary_mask}] — only
          when NOT saving (no output_dir / cfg); empty otherwise
        - obj_labels: Dict[global_obj_id, category_label]
        - seg_meta: what the last save wrote (segmentation.json), None when
          nothing was saved
    """
    from segmentation.sam3_wrapper import SAM3ConfigError, get_sam3_wrapper
    import time as _time

    persist = output_dir is not None and cfg is not None
    status = prompt_status if prompt_status is not None else {}
    seg_meta = None
    t_sam3 = _time.time()
    n_done = 0
    
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
                    status[category] = {"status": "skipped",
                                        "reason": "none of its VLM frame_map frames is "
                                                  "among the keyframes"}
                    continue
            else:
                # No VLM data at all for this category — skip
                print(f"[SegPipeline]   No VLM frame_map for '{cat_label}' — skipping category")
                status[category] = {"status": "skipped",
                                    "reason": "no VLM frame_map entry for this category"}
                continue
        
        # Compute batches for this category's frame subset
        cat_total = len(cat_frame_files)
        if cat_total == 0:
            print(f"[SegPipeline]   Skipping '{category}' — no frames in range")
            status[category] = {"status": "skipped", "reason": "no frames in range"}
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
                              mask_dedupe_iou, boxes_by_pos=None):
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

                    batch_masks, n_dup_ids = _dedupe_masks_per_frame(
                        batch_masks, mask_dedupe_iou, cfg=cfg, record=decisions)
                    if n_dup_ids:
                        print(f"[SegPipeline] Batch {batch_idx}: {n_dup_ids} object id(s) were "
                              f"the same observation (IoU ≥ {mask_dedupe_iou:g} in the majority "
                              f"of their shared frames, point 93) — collapsed")

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
                            next_global_id=next_global_id,
                            cfg=cfg, record=decisions,
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
        
        # ── Run the category. ANY SAM3 error FAILS THE RUN (docs/plan_determinismo.md
        # point 92, 2026-10-08): `sam3_wrapper.process_batch` raises SAM3RunError /
        # SAM3OutOfMemory and they propagate — no one-OOM retry, no "failed twice,
        # skipping", no prompt recorded as failed while the others go on. A run
        # delivers every prompt or nothing (the staging store is discarded by the
        # caller, the previous store stays as it was). A SAM3ConfigError
        # (models.segmentation.sam3_thresholds, a vendor rename) is the same: the
        # prompt's status names it before the error leaves.
        cat_masks = {}
        t_cat = _time.time()
        decisions: list = []                 # the identity decisions of this prompt (point 93)
        cat_status = {"status": "ran"}
        try:
            cat_masks = _process_category(
                category, cat_batches, frames_dir, cat_frame_files, sam3,
                batch_size, batch_overlap, iou_threshold, mask_dedupe_iou,
                boxes_by_pos=cat_boxes
            )
        except SAM3ConfigError as e:
            status[category] = {"status": "failed", "reason": f"SAM3ConfigError: {e}"}
            raise
        except Exception as e:
            status[category] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
            print(f"[SegPipeline] ⛔ Category '{category}' failed ({type(e).__name__}: {e}) — "
                  f"the run fails: no partial segmentation (point 92)")
            raise
        
        # THE FALLBACK (USER 2026-09-30): the bare category confirmed NOTHING → SAM3
        # lacks detail; try its origins (the VLM's descriptions, the merged names) in
        # order, keep the first that finds something — under THIS category's label.
        # An origin that errors fails the run like the category itself (point 92).
        if not any(len(fm) for fm in cat_masks.values()):
            _tried = []
            for _alt in (fallback_prompts or {}).get(category, []) or []:
                _tried.append(_alt)
                print(f"[SegPipeline] '{category}' confirmed nothing — retrying with its origin '{_alt}'")
                try:
                    _alt_masks = _process_category(
                        _alt, cat_batches, frames_dir, cat_frame_files, sam3,
                        batch_size, batch_overlap, iou_threshold, mask_dedupe_iou,
                        boxes_by_pos=cat_boxes)
                except Exception as _e:
                    status[category] = {"status": "failed",
                                        "reason": f"origin '{_alt}': {type(_e).__name__}: {_e}"}
                    raise
                if any(len(fm) for fm in _alt_masks.values()):
                    cat_masks = _alt_masks
                    cat_status["fallback_prompt"] = _alt
                    break
            if _tried:
                cat_status["fallbacks_tried"] = _tried

        # Collect unique obj_ids for this category
        cat_obj_ids = set()
        for fm in cat_masks.values():
            cat_obj_ids.update(fm.keys())
        
        print(f"[SegPipeline] Category '{category}': {len(cat_obj_ids)} objects across {len(cat_masks)} frames")
        if cat_status["status"] == "ran":
            cat_status.update(n_objects=len(cat_obj_ids), n_frames=len(cat_masks))
        # the clock goes to the timing record, never into the status the census copies
        # (point 125: segmentation_census.json differed on every run by these seconds)
        cat_seconds = round(_time.time() - t_cat, 1)
        if timing is not None:
            timing[category] = cat_seconds
        if decisions:
            cat_status["identity_decisions"] = decisions
        status[category] = cat_status
        
        # Remap this category's IDs to global space (offset by previous categories)
        cat_id_remap = {}
        for local_id in sorted(cat_obj_ids):
            global_id = local_id + global_id_offset
            cat_id_remap[local_id] = global_id
            obj_labels[global_id] = category
        
        # THIS category's masks with remapped global IDs, keyed by the global
        # frame_files index (local_frame_idx is an index into cat_frame_files)
        cat_global = {}
        for local_frame_idx, frame_masks in cat_masks.items():
            if local_frame_idx < len(cat_frame_indices):
                global_frame_idx = cat_frame_indices[local_frame_idx]
            else:
                global_frame_idx = local_frame_idx  # Fallback
            dst = cat_global.setdefault(global_frame_idx, {})
            for local_id, mask in frame_masks.items():
                dst[cat_id_remap[local_id]] = mask
        del cat_masks
        
        # Advance offset for next category
        if cat_obj_ids:
            global_id_offset = max(cat_id_remap.values())
        
        # Save THIS category's masklets as it finishes (crash-safe; see the
        # docstring for why only this category's). The mask→cloud matching +
        # per-instance cleaning runs ONCE at the end (Step 4) — in the intake
        # order the cleaned cloud does not even exist yet at this point.
        if persist:
            if cat_global:
                try:
                    # appended into the STAGING store (compacted canonically once, at
                    # the end of the run — finalize_sam3_store); the keyframe list and
                    # the frames' resolution come from the session's output/
                    seg_meta = _save_masks(output_dir, cat_global, categories[:cat_idx + 1],
                                           obj_labels, cfg,
                                           frame_space=mask_space.SPACE_KEYFRAME,
                                           session_output_dir=session_output_dir,
                                           compact=False)
                except Exception as e:
                    status[category] = {**cat_status, "status": "failed",
                                        "reason": f"its masks could not be saved: "
                                                  f"{type(e).__name__}: {e}"}
                    raise
                print(f"[SegPipeline] 💾 Saved category {cat_idx+1}/{len(categories)}")
        else:
            for global_frame_idx, frame_masks in cat_global.items():
                all_masks.setdefault(global_frame_idx, {}).update(frame_masks)
        del cat_global

        # measured, not assumed: SAM3 time per prompt on THIS run, and what the
        # remaining prompts will take at that rate
        n_done += 1
        per_prompt = (_time.time() - t_sam3) / n_done
        n_left = len(categories) - (cat_idx + 1)
        print(f"[SegPipeline] ⏱ prompt {cat_idx+1}/{len(categories)} took "
              f"{cat_seconds:.0f} s; {per_prompt:.0f} s/prompt so far → "
              f"~{per_prompt * n_left / 60:.0f} min for the {n_left} left")
        if on_progress and n_left:
            on_progress(((cat_idx + 1) / max(len(categories), 1)) * 100,
                        f"SAM3 {cat_idx+1}/{len(categories)} prompts, "
                        f"{per_prompt:.0f} s/prompt → ~{per_prompt * n_left / 60:.0f} min left")
        
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
    
    n_ran = sum(1 for c in categories if (status.get(c) or {}).get("status") == "ran")
    print(f"\n[SegPipeline] SAM3 complete: {len(obj_labels)} unique objects across "
          f"{len(categories)} categories ({n_ran} ran, {len(categories) - n_ran} skipped / "
          f"failed) in {(_time.time() - t_sam3) / 60:.1f} min")

    # The batch session and the symlink dirs were kept alive across concepts.
    try:
        sam3.release_batch_session()
    except Exception as e:  # noqa: BLE001
        print(f"[SegPipeline]   ⚠️ Could not release SAM3 session: {e}")
    _clear_batch_dirs()
    
    return all_masks, obj_labels, seg_meta


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



def _identity_judge(ious, bar: float, cfg: Optional[dict]) -> dict:
    """THE USER'S RULE for 'are these two SAM3 ids one object' (docs/plan_determinismo.md
    point 93, DECIDIDO): the IoUs of the frames where BOTH ids appear are the judges —
    the ids join only when the majority of those frames exceeds the bar (the median
    above it, its 95 % interval entirely above it) with at least 5 judges. The
    DECIDIDO names no error term, so the margin condition is void (error 0): the
    verdict is the significance and the judges. The dict carries every margin."""
    if cfg is None:
        from config import cfg as _cfg
        cfg = _cfg
    fac, conf, min_j = _decision_params(cfg)
    out = _decide_above_bar(ious, bar, 0.0, factor=fac, confidence=conf, min_judges=min_j)
    out["n_frames_above_bar"] = int(np.sum(np.asarray(ious, np.float64) >= bar))
    return out


def _match_ids_iou(prev_masks: Dict[int, Dict[int, np.ndarray]],
                   curr_masks: Dict[int, Dict[int, np.ndarray]],
                   overlap_start: int, overlap_end: int,
                   iou_threshold: float,
                   next_global_id: int, cfg: Optional[dict] = None,
                   record: Optional[list] = None) -> Tuple[Dict[int, int], int]:
    """
    Match object IDs between batches using IoU in the overlap region.

    INVIOLABLE RULE (Phase R.1): cross-window re-identification happens
    EXCLUSIVELY here, by TRACKING CONTINUITY through the shared overlap frames.
    Matching instances by APPEARANCE between windows not connected by tracking
    is PROHIBITED: in tunnels/stations identical columns repeat every N metres
    and an appearance match creates false loop closures that destroy the
    reconstruction (perceptual aliasing). Objects with no overlap-frame IoU
    link get a FRESH global id — never a similarity-based merge.

    A link is decided by :func:`_identity_judge` over the overlap frames where
    both ids appear (point 93): no longer an average IoU over the whole overlap
    compared with the bar in one shot. Pairs that pass are matched greedily by
    their median IoU; every judged pair lands in ``record`` with its margins.

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

    # the IoU of every (prev, curr) pair in every overlap frame where BOTH appear
    ious: Dict[Tuple[int, int], List[float]] = {}
    for fidx in overlap_frames:
        prev_fm = prev_masks.get(fidx, {})
        curr_fm = curr_masks.get(fidx, {})

        for pid in prev_obj_ids:
            if pid not in prev_fm:
                continue
            pmask = prev_fm[pid].astype(bool)

            for cid in curr_obj_ids:
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
                    ious.setdefault((pid, cid), []).append(float(intersection / union))

    # the judge, pair by pair (point 93); only the pairs that pass may link
    pairs = []
    for (pid, cid), vals in sorted(ious.items()):
        if max(vals) < iou_threshold:
            continue                                  # no frame reaches the bar: nothing to judge
        d = _identity_judge(vals, iou_threshold, cfg)
        if record is not None:
            record.append({"kind": "batch_link", "prev": int(pid), "curr": int(cid),
                           "n_judges": int(d["n_judges"]), "median_iou": float(np.median(vals)),
                           "linked": bool(d["improves"]), "ci_margin": d["ci_margin"],
                           "judges_margin": d["judges_margin"], "reason": d["reason"]})
        if d["improves"]:
            pairs.append((float(np.median(vals)), pid, cid))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    # Greedy matching (Hungarian would be ideal but greedy is simpler and sufficient)
    id_remap = {}
    used_prev = set()
    used_curr = set()
    for iou_val, pid, cid in pairs:
        if pid in used_prev or cid in used_curr:
            continue
        # Match: current object cid maps to previous global ID pid (already global)
        id_remap[cid] = pid
        used_prev.add(pid)
        used_curr.add(cid)
        print(f"[SegPipeline] IoU: obj {cid} → global {pid} (median IoU={iou_val:.3f})")

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
                categories: List[str], obj_labels: Dict[int, str], cfg: dict,
                *, frame_space: str, session_output_dir: Optional[Path] = None,
                compact: bool = True):
    """
    Save SAM3 masks as compressed NPZ + metadata JSON.
    Upsert logic: if an obj_id already exists in the NPZ (same object from a
    previous incremental save), keep its ID and overwrite its masks.
    If it's genuinely new, assign a new ID.

    ``output_dir`` is the directory of the STORE being written; ``session_output_dir``
    (default: the same) the session's output/, where the keyframe list and the
    frames live — the batch pipeline writes a fresh store in a staging directory
    (docs/plan_determinismo.md point 83) and the interactive manager upserts into
    the session's own (the one place the incremental update is allowed to stay).
    ``compact`` (default True) rewrites the store canonically after the save —
    members in (object, frame) order, no dead bytes, fixed zip times (point 101);
    the batch pipeline passes False and compacts once at the end of the run.

    ``frame_space`` (mandatory, keyword-only) DECLARES which index space
    ``all_masks`` is keyed by — ``mask_space.SPACE_KEYFRAME`` for the batch
    pipeline, which numbers frames_valid/ 0,1,2…, or ``SPACE_VIDEO`` for the
    interactive and Resume paths, which translate SAM3's sequential index
    back to the real video frame number through ``kf_mapping``. All three
    upsert into the SAME file, and until this argument existed the file ended
    up holding both conventions at once: on pccr oid 110 was keyed 0,1,2,3…
    and oid 213 keyed 1,60,97,…, so every reader took the wrong mask for one
    of them, or none. The incoming frames are now translated into whatever
    space the store already uses (a new store adopts the writer's), the
    choice is WRITTEN INTO the npz, and a frame that cannot be translated
    fails the save instead of corrupting it.
    """
    from segmentation import mask_space as _mspace

    colors = cfg["visualization"]["segment_colors"]
    output_dir = Path(output_dir)
    session_output_dir = Path(session_output_dir) if session_output_dir is not None else output_dir

    # ── ONE space per store (see the docstring) ──
    # the store's space is the store's; a translation (interactive masks in video
    # numbers into a positional store) reads the SESSION's keyframe list
    dst_space = _mspace.store_space(output_dir, frame_space)
    if dst_space != frame_space:
        conv = _mspace.convert_frames(session_output_dir, all_masks.keys(), frame_space, dst_space)
        all_masks = {conv[int(f)]: v for f, v in all_masks.items()}
        print(f"[SegPipeline] [MaskSpace] incoming masks are in {frame_space}, the store is "
              f"in {dst_space} — translated {len(all_masks)} frames")
    frame_space = dst_space

    # the mask resolution is what SAM3 produced; the frames' resolution is read from the
    # frames on disk (never from chunk_*_meta.json leftovers — point 108: a previous
    # run's Omega chunk metadata is not a declaration of this store's grid)
    scaled_res = None   # detected from the actual mask shape
    original_res = None  # detected from the actual frames

    # Detect original resolution from frames on disk
    if original_res is None:
        frames_dir = session_output_dir.parent / "frames"
        if not frames_dir.exists():
            frames_dir = session_output_dir / "frames"
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
    
    # the f*_o* entries the store already holds: KEPT AS STORED (never decompressed
    # or recompressed — see _atomic_append_npz); only this call's masks are written
    existing_mask_keys = set()
    store_readable = False
    existing_instances = []
    existing_prompts = []
    max_existing_id = -1
    existing_frames = set()
    existing_obj_ids = set()
    old_meta: dict = {}
    
    if masks_path.exists():
        try:
            with np.load(masks_path) as old_data:
                existing_mask_keys = {key for key in old_data.files
                                      if key.startswith("f") and "_o" in key}
                if "obj_ids" in old_data.files:
                    existing_obj_ids = set(old_data["obj_ids"].tolist())
                if "frames" in old_data.files:
                    existing_frames = set(old_data["frames"].tolist())
            store_readable = True
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
            # a mask folded into an object is GONE from the parent but its oid
            # is not free: the archive and the `absorbed` record still speak
            # about it, so a new masklet reusing it would collide with both
            max_existing_id = max(max_existing_id,
                                  int(old_meta.get("id_high_water") or -1))
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
    
    # ── The entries THIS call writes (the store's other masks stay as stored) ──
    npz_data = {}
    
    # Add new masks with remapped IDs
    new_frame_indices = sorted(all_masks.keys())
    mask_count = 0
    
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
    # the store describes itself: no later reader has to guess (or measure)
    # which of the two frame spaces these keys are in
    npz_data[_mspace.NPZ_KEY] = _mspace.declaration(frame_space)
    
    # Save compressed NPZ — ATOMIC (tmp ending in .npz + replace): a crash
    # mid-write must never truncate the session's masks (2026-08-29). Into a
    # readable store the new entries are APPENDED: re-reading and recompressing
    # every mask the store already holds made each save cost the whole store —
    # quadratic over a run that saves once per prompt (pccr: ~10 s per rewrite
    # of 2,211 masks, 814 MB decompressed in RAM).
    from segmentation.erase import _atomic_append_npz, _atomic_savez
    if store_readable:
        # the store's own metadata members (the keyframe list it was segmented on, its
        # reconstruction id) travel with it through an upsert
        with np.load(masks_path) as old_data:
            for k in STORE_META_KEYS:
                if k in old_data.files and k not in npz_data:
                    npz_data[k] = np.asarray(old_data[k])
        _atomic_append_npz(masks_path, npz_data, keep=existing_mask_keys)
    else:
        _atomic_savez(masks_path, npz_data)
    if compact:
        compact_mask_store(masks_path)
    _mspace.invalidate(output_dir)
    masks_mb = masks_path.stat().st_size / (1024 * 1024)
    new_count = mask_count
    print(f"[SegPipeline] ✅ Saved masks: {masks_path.name} "
          f"({new_count} new masks, {len(all_obj_ids)} total objects, "
          f"{len(all_frames)} frames, {masks_mb:.1f} MB)")
    
    # ── Build metadata JSON (upsert: update existing, add new) ──
    existing_by_id = {inst["id"]: inst for inst in existing_instances}
    
    max_existing_iid = 0
    for inst in existing_instances:
        max_existing_iid = max(max_existing_iid, inst.get("instance_id", 0))
        for _p in (inst.get("parts") or []):          # retired, not free
            max_existing_iid = max(max_existing_iid,
                                   int(_p.get("instance_id", 0)))
    try:
        max_existing_iid = max(max_existing_iid,
                               int((old_meta or {}).get("instance_id_high_water") or 0))
    except Exception:  # noqa: BLE001
        pass
    
    color_offset = len(existing_instances)
    new_count = 0
    for raw_id in sorted(new_obj_ids_raw):
        remapped_id = id_remap[raw_id]
        label = obj_labels.get(raw_id, categories[0] if categories else "object")
        # Rich SAM3 concept phrases ("concrete support column") become compact
        # id-like labels here — the ONE place labels are persisted — so folder
        # names / JSON keys downstream never carry spaces. The rule lives in
        # census.concept_label so the census attributes masklets by it too.
        from segmentation.census import concept_label
        label = concept_label(label)
        
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
    # an upsert keeps the store's identity members (the reconstruction id, the keyframe
    # list's digest) but NOT its seal: the seal names the mask bytes the SAM3 stage wrote,
    # and this store no longer holds exactly those (point 83 — the interactive manager's
    # edits are unsealed by construction)
    for k in ("reconstruction_id", "keyframes_sha256", "n_keyframes"):
        if isinstance(old_meta, dict) and k in old_meta:
            segmentation[k] = old_meta[k]
    # THE MODEL that drew these masks (points 91 / 165): version, checkpoint path + sha256,
    # device, dtype, numerics, builder args, thresholds — the wrapper's record of the model
    # it built (None when the wrapper built none, e.g. a test double: recorded as such)
    from segmentation.sam3_wrapper import get_sam3_wrapper
    segmentation["sam3_model"] = getattr(get_sam3_wrapper(), "model_record", None)

    atomic_write_json(seg_path, segmentation, indent=2)
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

def _voxel_components(pts: np.ndarray, voxel_m: float):
    """Labels of the 26-connected components of the occupied-voxel grid.

    The same test ``reconstruction.loops.instance_loops.disjoint_clusters``
    uses to tell two copies apart, in one O(N) pass: two points share a label
    exactly when a chain of occupied voxels joins them.
    """
    key = np.floor(np.asarray(pts, np.float64) / float(voxel_m)).astype(np.int64)
    vox, inv = np.unique(key, axis=0, return_inverse=True)
    n = len(vox)
    if n <= 1:
        return np.zeros(len(pts), np.int64)
    base = vox.min(axis=0)
    span = (vox.max(axis=0) - base + 3).astype(np.int64)
    if float(span[0]) * float(span[1]) * float(span[2]) > 9.0e18:
        return np.zeros(len(pts), np.int64)

    def _pack(v):
        d = v - base + 1
        return (d[:, 0] * span[1] + d[:, 1]) * span[2] + d[:, 2]

    packed = _pack(vox)
    order = np.argsort(packed, kind="stable")
    packed_sorted = packed[order]
    parent = np.arange(n)

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    for off in [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]:
        nb = _pack(vox + np.asarray(off, np.int64))
        pos = np.searchsorted(packed_sorted, nb)
        ok = pos < n
        pos = np.where(ok, pos, 0)
        hit = ok & (packed_sorted[pos] == nb)
        if not hit.any():
            continue
        for a, b in zip(np.flatnonzero(hit), order[pos[hit]]):
            ra, rb = _find(int(a)), _find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([_find(i) for i in range(n)], dtype=np.int64)
    return roots[inv]


def _share_verdict(n_part: int, n: int, bar: float, factor: float) -> Tuple[str, float, float]:
    """Does a share ``n_part / n`` sit above, below or AT a bar? (docs/plan_determinismo.md
    point 145, DECIDIDO): the share's own sampling error is the binomial
    ``sqrt(s (1 - s) / n)`` — measured from the counts, nothing chosen — and the share
    differs from the bar only when it does so by at least ``factor`` × that error.
    Returns ('above' | 'below' | 'tie', share, error)."""
    n = max(int(n), 1)
    s = float(n_part) / n
    err = float(np.sqrt(max(s * (1.0 - s), 0.0) / n))
    if s - bar >= factor * err:
        return "above", s, err
    if bar - s >= factor * err:
        return "below", s, err
    return "tie", s, err


def _obb_core_points(points_xyz: np.ndarray, cfg: dict, factor: Optional[float] = None):
    """The points that actually make up the element, flyers left out.

    An OBB from ``.min()``/``.max()`` is decided by its two most extreme
    points, so ONE stray point a metre off the object inflates the box by a
    metre — pccr 2026-09-14, the user reading the segment list: "muchas están
    infladas por voladores". That box is not cosmetic: the split gate measures
    the separation between copies from it, and the floor levelling picks its
    candidate by its height.

    USER's criterion: find where the concentration of points is. A flyer (or a
    small stray blob) is its own tiny connected component of the occupied-voxel
    grid (world-anchored), so every component holding less than
    ``min_component_frac`` of the instance is left out of the EXTENT. Nothing is
    deleted — the instance keeps all its points, only the box stops being
    defined by its strays.

    THE CUTS ARE JUDGED WITH THEIR SAMPLING ERROR (point 145, DECIDIDO): a
    component leaves only when its share sits below the bar by ≥ ``factor`` ×
    the share's binomial error, and stays otherwise (a 1.99 % blob and a 2.01 %
    one get the same fate: the simplest, in); the core replaces every point only
    when its kept share clears ``min_keep_frac`` by the same margin, else every
    point is used (the simplest). ``factor`` = correction_graph.graph.
    improvement_error_factor (read from the configuration when not given).

    Returns (core_indices, n_dropped, record). ``core_indices`` is None when
    every point is used.
    """
    n = len(points_xyz)
    rec = {"rule": "component share vs min_component_frac with its binomial error (point 145)",
           "n_points": int(n), "components": [], "kept_share": None, "kept_verdict": None,
           "used": "all"}
    if not cfg.get("enabled", True) or n < int(cfg.get("min_points", 200)):
        rec["used"] = "all (disabled or under min_points)"
        return None, 0, rec
    if factor is None:
        from config import cfg as _server_cfg
        factor = _decision_params(_server_cfg)[0]
    lab = _voxel_components(points_xyz, float(cfg.get("voxel_m", 0.05)))
    uniq, counts = np.unique(lab, return_counts=True)
    if len(uniq) == 1:
        rec["used"] = "all (one component)"
        return None, 0, rec
    bar = float(cfg.get("min_component_frac", 0.02))
    keep_lab = []
    for u, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda t: (-t[1], t[0])):
        verdict, share, err = _share_verdict(c, n, bar, factor)
        rec["components"].append({"points": int(c), "share": share, "error": err,
                                  "margin": share - bar, "verdict": verdict,
                                  "in_extent": verdict != "below"})
        if verdict != "below":
            keep_lab.append(u)
    if not keep_lab:
        rec["used"] = "all (no component clears the bar)"
        return None, 0, rec
    core = np.flatnonzero(np.isin(lab, np.asarray(keep_lab)))
    verdict, share, err = _share_verdict(len(core), n, float(cfg.get("min_keep_frac", 0.5)), factor)
    rec["kept_share"], rec["kept_error"], rec["kept_verdict"] = share, err, verdict
    if verdict != "above" or len(core) < 4:
        rec["used"] = "all (the kept share does not clear min_keep_frac by its error)"
        return None, 0, rec
    rec["used"] = "core"
    return core, int(n - len(core)), rec


def _fold_yaw(a: float) -> float:
    """A yaw into the canonical [0, π/2) range (point 109): a box is the same box
    at yaw θ, θ + 90° (extents swapped) and θ + 180°, so every candidate is
    expressed by the one representative and perpendicular faces give one byte
    sequence instead of a flip."""
    q = np.pi / 2.0
    return float(np.mod(float(a), q))


def _vertical_plane_yaws(points_xyz: np.ndarray, vcfg: dict) -> list:
    """[(yaw, support)] of each VERTICAL plane of an object, found by sequential
    RANSAC (seeded) on a STABLE sample of its points — the ``sample`` points with
    the smallest per-point key (:func:`_stable_sample`, point 109: one point more
    or fewer no longer redraws the whole sample). A plane counts when its normal
    is within ``vertical_tol_deg`` of horizontal; EVERY such plane is a candidate
    with its inlier count as support (point 143, DECIDIDO: no 10 % cut — a key
    ``min_plane_frac`` left in the configuration FAILS the load). The yaw is
    folded into [0, π/2)."""
    keys = ("ransac_iters", "dist_m", "vertical_tol_deg", "max_planes", "sample", "seed")
    missing = [k for k in keys if k not in vcfg]
    if missing:
        raise KeyError(f"config.yaml segmentation.obb_orientation is missing {missing}")
    if "min_plane_frac" in vcfg:
        raise KeyError("config.yaml segmentation.obb_orientation.min_plane_frac was removed "
                       "(docs/plan_determinismo.md point 143: every vertical RANSAC plane is a "
                       "candidate, its support decides among footprint ties) — delete the key")
    P = np.asarray(points_xyz, np.float64)
    if len(P) < 10:
        return []
    rng = np.random.default_rng(int(vcfg["seed"]))
    P = P[_stable_sample(P, int(vcfg["sample"]))]
    sin_tol = np.sin(np.radians(float(vcfg["vertical_tol_deg"])))
    dist = float(vcfg["dist_m"])
    yaws, rest = [], P
    for _ in range(int(vcfg["max_planes"])):
        if len(rest) < 3:
            break
        best_n, best_cnt, best_a = None, 0, None
        for _it in range(int(vcfg["ransac_iters"])):
            a, b, c = rest[rng.choice(len(rest), 3, replace=False)]
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n = n / ln
            if abs(n[1]) > sin_tol:          # not vertical: a horizontal face gives no yaw
                continue
            cnt = int((np.abs((rest - a) @ n) < dist).sum())
            if cnt > best_cnt:
                best_cnt, best_n, best_a = cnt, n, a
        if best_n is None or best_cnt < 3:
            break
        inl = np.abs((rest - best_a) @ best_n) < dist
        Q = rest[inl]
        c0 = Q.mean(0)
        n = np.linalg.svd(Q - c0, full_matrices=False)[2][2]      # least-squares normal
        yaws.append((_fold_yaw(np.arctan2(n[2], n[0])), int(best_cnt)))
        rest = rest[~inl]
    return yaws


def _compute_obb(points_xyz: np.ndarray, face_normals=None) -> dict:
    """Compute minimum Oriented Bounding Box for floor-aligned coordinates.

    If face_normals is provided (list of (normal, n_points) tuples from RANSAC),
    uses the dominant face normal to orient the OBB on the XZ plane.
    Otherwise falls back to convex hull + rotating calipers.
    Coordinates must be floor-aligned (Y = up).

    THE YAW (docs/plan_determinismo.md points 109 / 143, DECIDIDO): every
    candidate yaw is folded into [0, π/2); among the candidates whose footprint
    lies within ``factor`` × its MEASURED error of the minimum — the error
    propagated from the inlier band ``dist_m`` (the cloud's surface noise): a
    box of extents (ex, ez) measured to ±dist has δA = dist·(ex + ez) — the
    DOMINANT plane (most support) wins, the user's "coplanar con el plano
    dominante"; the record carries every candidate with its margin. THE EXTENT
    (point 145): the core points by :func:`_obb_core_points`, or every point.
    """
    points_xyz = np.asarray(points_xyz, np.float64)
    if len(points_xyz) < 4:
        center = points_xyz.mean(axis=0)
        return {
            "center": center.tolist(),
            "half_extents": [0.01, 0.01, 0.01],
            "rotation": [[1,0,0],[0,1,0],[0,0,1]]
        }

    # The EXTENT is taken from where the points concentrate, not from the two
    # most extreme ones — see _obb_core_points. Everything below reads
    # points_xyz, so the substitution is the whole change.
    from config import cfg as _server_cfg
    factor = _decision_params(_server_cfg)[0]
    _ocfg = ((_server_cfg.get("segmentation", {}) or {}).get("obb_core", {}) or {})
    _all = points_xyz
    _core, _n_dropped, _core_rec = _obb_core_points(points_xyz, _ocfg, factor)
    if _core is not None:
        points_xyz = points_xyz[_core]

    # Y extent (vertical)
    y_min = points_xyz[:, 1].min()
    y_max = points_xyz[:, 1].max()
    half_y = (y_max - y_min) / 2.0
    cy = (y_min + y_max) / 2.0

    # Project to XZ plane for 2D bounding rectangle
    pts_xz = points_xyz[:, [0, 2]]  # (N, 2): [x, z]

    best_angle = 0.0
    # THE YAW (USER 2026-09-30: "el cálculo de los OBB debe ser por RANSAC para saber cómo
    # orientarlo — muchas veces queda cruzado —, coplanar con el plano dominante, dejando el
    # menor vacío posible, siempre con la cara inferior paralela a y = 0"): the object's own
    # VERTICAL planes (sequential RANSAC; a horizontal face — a table top, a floor — gives no
    # direction in plan and used to leave the box on the world axes), and among them the one
    # whose box leaves the least empty footprint. No vertical plane → the minimum-area
    # rectangle (rotating calipers), as before.
    _vcfg = ((_server_cfg.get("segmentation", {}) or {}).get("obb_orientation", {}) or {})
    cands = list(_vertical_plane_yaws(points_xyz, _vcfg)) if _vcfg.get("enabled", False) else []
    for fn in (face_normals or []):
        n3 = np.asarray(fn[0], np.float64)
        nxz = np.array([n3[0], n3[2]])
        if np.linalg.norm(nxz) > 0.1:
            cands.append((_fold_yaw(np.arctan2(nxz[1], nxz[0])), int(fn[1]) if len(fn) > 1 else 0))

    def _extents(a):
        c, s_ = np.cos(-a), np.sin(-a)
        r = pts_xz @ np.array([[c, -s_], [s_, c]]).T
        return r.max(axis=0) - r.min(axis=0)

    yaw_rec = {"rule": "dominant plane among footprints within factor x error of the minimum "
                       "(points 109 / 143)", "candidates": [], "chosen": None}
    if cands:
        if "dist_m" not in _vcfg:
            raise KeyError("config.yaml segmentation.obb_orientation.dist_m is needed to judge "
                           "the footprint of the OBB candidates")
        dist = float(_vcfg["dist_m"])
        rows = []
        for yaw, support in cands:
            e = _extents(yaw)
            rows.append((float(yaw), int(support), float(e[0] * e[1]), float(dist * (e[0] + e[1]))))
        a_min = min(r[2] for r in rows)
        err_min = next(r[3] for r in rows if r[2] == a_min)
        tied = [r for r in rows if r[2] - a_min <= factor * err_min]
        # the dominant plane among the tied; a tie of support → the smaller footprint, then
        # the smaller yaw: a total order, so two runs pick one candidate
        chosen = max(tied, key=lambda r: (r[1], -r[2], -r[0]))
        best_angle = chosen[0]
        for r in sorted(rows, key=lambda r: (r[2], -r[1], r[0])):
            yaw_rec["candidates"].append({"yaw_deg": float(np.degrees(r[0])), "support": r[1],
                                          "footprint_m2": r[2], "footprint_error_m2": r[3],
                                          "margin_m2": r[2] - a_min,
                                          "tied_with_minimum": bool(r[2] - a_min <= factor * err_min)})
        yaw_rec["chosen"] = {"yaw_deg": float(np.degrees(best_angle)), "support": chosen[1],
                             "footprint_m2": chosen[2], "n_tied": len(tied)}
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
            angle = _fold_yaw(np.arctan2(edge[1], edge[0]))

            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            rot2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            rotated = hull_pts @ rot2d.T
            rmin = rotated.min(axis=0)
            rmax = rotated.max(axis=0)
            area = (rmax[0] - rmin[0]) * (rmax[1] - rmin[1])

            # a strictly smaller area, or the same area at a smaller yaw: a total order
            if area < best_area or (area == best_area and angle < best_angle):
                best_area = area
                best_angle = angle
        yaw_rec["chosen"] = {"yaw_deg": float(np.degrees(best_angle)), "support": None,
                             "footprint_m2": float(best_area) if np.isfinite(best_area) else None,
                             "n_tied": None, "source": "rotating calipers (no vertical plane)"}

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
        "rotation": rotation,
        # what the box was measured on: declared, so an inflated extent is
        # never mistaken for a measured one
        "n_points": int(len(_all)),
        "n_points_used": int(len(points_xyz)),
        "n_flyers_excluded": int(_n_dropped),
        # the decisions and their margins (points 109 / 143 / 145)
        "decisions": {"yaw": yaw_rec, "extent": _core_rec},
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
    _rng = np.random.default_rng(0)        # seeded: one cloud, one set of faces
    
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
            sample_idx = _rng.choice(len(rem_pts), 3, replace=False)
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


def _dedupe_masks_per_frame(batch_masks, iou_threshold: float, cfg: Optional[dict] = None,
                            record: Optional[list] = None):
    """Collapse object ids that SAM3 handed out for the SAME observation.

    The tracker occasionally returns one region under several ids: on pccr
    2026-09-14 frame 120 carried two 'white tiled floor' masks of 109,786
    pixels each, IoU 1.0, and frame 0 carried a second identical pair. Each
    copy then became its own 3-D instance and stole points from the other.

    Two ids are the same object when the MAJORITY of the frames where both
    appear agree above the bar (docs/plan_determinismo.md point 93, DECIDIDO:
    `_identity_judge` — the median IoU of those frames above `mask_dedupe_iou`,
    its 95 % interval entirely above it, at least 5 judge frames; otherwise they
    stay separate and the decision records its margin). One frame used to
    decide for the whole batch. The union is taken over the batch (so the
    tracks stay consistent frame to frame) and the lowest id survives. Masks
    that merely overlap — a monitor inside a desk, a sign against a wall —
    have a low IoU and are untouched; an IoU of 0.9 already demands the areas
    agree within 10 %, which is used to skip almost every pair without
    touching the pixels.

    Returns (deduped_masks, n_ids_collapsed); the input is not mutated.
    """
    if iou_threshold <= 0 or iou_threshold > 1:
        return batch_masks, 0
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # every frame where both ids appear is a judge: its IoU (0 when the areas are too far
    # apart to reach the bar — the pixels need not be touched to know that)
    ious: Dict[Tuple[int, int], List[float]] = {}
    for frame_idx in sorted(batch_masks):
        frame_masks = batch_masks[frame_idx]
        items = []
        for oid, m in frame_masks.items():
            mb = m.astype(bool, copy=False)
            items.append((int(mb.sum()), int(oid), mb))
        items.sort(key=lambda t: (t[0], t[1]))         # by area: IoU >= t needs areas within t
        oids = sorted(int(o) for o in frame_masks)
        for a in range(len(items)):
            area_a, oid_a, mask_a = items[a]
            if area_a == 0:
                continue
            for b in range(a + 1, len(items)):
                area_b, oid_b, mask_b = items[b]
                pair = (min(oid_a, oid_b), max(oid_a, oid_b))
                if area_a < iou_threshold * area_b:    # areas too far apart — and sorted, so
                    break                              # every later b is farther still
                if mask_a.shape != mask_b.shape:
                    continue
                inter = int(np.logical_and(mask_a, mask_b).sum())
                iou = inter / float(area_a + area_b - inter) if inter else 0.0
                ious.setdefault(pair, []).append(iou)
        # frames where both appear but one is far smaller / disjoint: IoU 0 — a judge too
        for i, oa in enumerate(oids):
            for ob in oids[i + 1:]:
                ious.setdefault((oa, ob), [])
    for pair in list(ious):
        n_both = sum(1 for fm in batch_masks.values() if pair[0] in fm and pair[1] in fm)
        vals = ious[pair]
        if n_both > len(vals):                          # frames judged without touching pixels
            vals = vals + [0.0] * (n_both - len(vals))
        ious[pair] = vals
    for (oid_a, oid_b), vals in sorted(ious.items()):
        if not vals or max(vals) < iou_threshold:
            continue                                   # no frame reaches the bar: nothing to judge
        d = _identity_judge(vals, iou_threshold, cfg)
        if record is not None:
            record.append({"kind": "same_observation", "ids": [int(oid_a), int(oid_b)],
                           "n_judges": int(d["n_judges"]), "median_iou": float(np.median(vals)),
                           "n_frames_above_bar": d["n_frames_above_bar"],
                           "collapsed": bool(d["improves"]), "ci_margin": d["ci_margin"],
                           "judges_margin": d["judges_margin"], "reason": d["reason"]})
        if d["improves"]:
            union(oid_a, oid_b)

    if not parent:
        return batch_masks, 0
    collapsed = {}
    for frame_idx, frame_masks in batch_masks.items():
        out = {}
        for oid, m in frame_masks.items():
            root = find(oid)
            prev = out.get(root)
            # the survivor keeps the LARGER mask: the copies agree by IoU, so
            # this only picks up the few pixels one of them saw and the other missed
            if prev is None or m.astype(bool).sum() > prev.astype(bool).sum():
                out[root] = m
        collapsed[frame_idx] = out
    n_collapsed = sum(1 for oid in parent if find(oid) != oid)
    return collapsed, n_collapsed


def _record_overlap(record: dict, absorbed: dict, keeper: dict,
                    overlap_ratio: float) -> None:
    """Write down an index-overlap merge, like every other merge does.

    This one printed its line and wrote nothing (pccr 2026-09-20). While the
    record was only a display filter that was merely untidy; now that the
    fusion REWRITES THE PARENT from it, a merge with no record is a mask that
    is never folded into its object and an entry left in the parent owning
    points that belong to the keeper — the parent would lie.
    """
    iid = absorbed.get("instance_id", absorbed.get("id"))
    into = keeper.get("instance_id", keeper.get("id"))
    if iid is None or into is None:
        return
    record[int(iid)] = {"into": int(into),
                        "into_label": keeper.get("label"),
                        "reason": "overlap_dedupe",
                        "label": absorbed.get("label"),
                        "overlap": round(float(overlap_ratio), 3)}


def _gap_rays(A: np.ndarray, B: np.ndarray, tree, cams: np.ndarray, gap_m: float,
              reach_m: float) -> List[int]:
    """The RAY JUDGES of 'is the gap between two components of one instance EMPTY SPACE the
    cameras see through' (docs/plan_determinismo.md point 105): one judge per INFORMATIVE
    ray (a camera × a sample of the gap with nothing standing before the sample), 1 when
    the ray meets measured geometry BEHIND the sample (the camera sees through the gap — air
    between three desks shows the floor behind), 0 when it meets nothing (a hole in one
    floor: nothing lies beyond it). The caller judges the list with the user's rule.

    Interior samples of the segment between the two closest points (every `gap_m`, at most
    20 — a BOUND of the cost, the samples themselves are a fixed linspace); for every camera
    the ray through a sample is walked beyond it every `gap_m` up to `reach_m`."""
    from scipy.spatial import cKDTree
    d, ia = cKDTree(A).query(B, k=1)
    jb = int(np.argmin(d))
    a, b = A[ia[jb]], B[jb]
    n = int(np.floor(np.linalg.norm(b - a) / gap_m))
    if n < 2:
        return []
    samples = a + (b - a) * (np.linspace(1, n - 1, min(n - 1, 20))[:, None] / n)   # BOUND (cost): ≤ 20
    steps = np.arange(1, max(int(reach_m / gap_m), 1) + 1) * gap_m
    rays: List[int] = []
    for C in cams:
        for smp in samples:
            ray = smp - C
            L = float(np.linalg.norm(ray))
            if L <= gap_m:
                continue
            ray /= L
            front = C + ray * np.arange(gap_m, L - gap_m, gap_m)[:, None]
            if len(front) and np.isfinite(tree.query(front, k=1, distance_upper_bound=gap_m / 2)[0]).any():
                continue                                     # something stands before the gap
            behind = smp + ray * steps[:, None]
            rays.append(int(np.isfinite(tree.query(behind, k=1, distance_upper_bound=gap_m / 2)[0]).any()))
    return rays


def _gap_is_free_space(A: np.ndarray, B: np.ndarray, tree, cams: np.ndarray, gap_m: float,
                       reach_m: float, *, factor: float, confidence: float,
                       min_judges: int) -> dict:
    """Free space between two components, by THE USER'S RULE over the ray judges of
    :func:`_gap_rays`: the share of rays that see through exceeds one half significantly
    (95 %), with ≥ 5 informative rays, by ≥ ``factor`` × its binomial error. Returns the
    decision dict (``improves`` = free space)."""
    rays = _gap_rays(A, B, tree, cams, gap_m, reach_m)
    p = float(np.mean(rays)) if rays else 0.0
    err = float(np.sqrt(max(p * (1.0 - p), 0.0) / max(len(rays), 1)))
    d = _decide_above_bar(rays, 0.5, err, factor=factor, confidence=confidence,
                          min_judges=min_judges)
    d.update({"n_rays": len(rays), "share_through": p})
    return d


def _camera_returned(fa: set, fb: set, cam_centre: Dict[int, np.ndarray], chain: Dict[int, float],
                     min_walk_m: float) -> bool:
    """Did the walk come back? True when a keyframe seeing A and one seeing B have cameras within
    `min_walk_m` of each other in space but more than `min_walk_m` apart along the walk
    (correction.visit_drift.min_walk_m — the USER's one definition of a visit)."""
    a = sorted(f for f in fa if f in cam_centre and f in chain)
    b = sorted(f for f in fb if f in cam_centre and f in chain)
    if not a or not b:
        return False
    CA = np.array([cam_centre[f] for f in a]); CB = np.array([cam_centre[f] for f in b])
    wa = np.array([chain[f] for f in a]); wb = np.array([chain[f] for f in b])
    near = np.linalg.norm(CA[:, None, :] - CB[None, :, :], axis=2) <= min_walk_m
    far_walk = np.abs(wa[:, None] - wb[None, :]) > min_walk_m
    return bool((near & far_walk).any())


def _covisible(kf_a: set, kf_b: set, covis_share: float, *, factor: float, confidence: float,
               min_judges: int) -> dict:
    """Are two components SEEN TOGETHER? (point 105, DECIDIDO) The judges are the keyframes of
    the component seen in FEWER keyframes; each says 1 when it also sees the other. The share
    exceeds `covis_share` (segmentation.dedupe_overlap) significantly, with ≥ 5 judges, by ≥
    ``factor`` × the share's binomial error — else they are NOT co-visible (the simplest)."""
    small, other = (kf_a, kf_b) if len(kf_a) <= len(kf_b) else (kf_b, kf_a)
    judges = [1 if f in other else 0 for f in sorted(small)]
    p = float(np.mean(judges)) if judges else 0.0
    err = float(np.sqrt(max(p * (1.0 - p), 0.0) / max(len(judges), 1)))
    d = _decide_above_bar(judges, covis_share, err, factor=factor, confidence=confidence,
                          min_judges=min_judges)
    d.update({"n_keyframes_small": len(small), "n_common": int(sum(judges)), "share": p})
    return d


def _split_covisible_components(instances: list, xyz_display: np.ndarray, frame_arr: np.ndarray,
                                gap_m: float, min_points: int, covis_share: float,
                                cam_centre: Optional[Dict[int, np.ndarray]] = None,
                                max_cams: int = 12, min_walk_m: Optional[float] = None,
                                id_base: Optional[int] = None,
                                decision: Optional[Tuple[float, float, int]] = None,
                                record: Optional[list] = None) -> int:
    """Split an instance that is several objects seen TOGETHER (pccr 2026-09-30, USER: "está mal
    que junte tres desk separados en uno solo ID").

    SAM3 can draw ONE mask over several neighbouring objects (desk #174: 2–3 separate blobs in 32
    of its 58 masks), and the space dedupe then absorbs each object's own instance into it. The
    instance's points are split into components separated by more than `gap_m` (the same gap
    under which `_merge_label_fragments` calls pieces contiguous), on the world-anchored grid
    (point 102). A component is a candidate object when it has `min_points`
    (correction.visit_drift.min_points, the USER's 1000: below it an object cannot be
    measured); smaller crumbs join the nearest candidate. Two candidates are DIFFERENT objects
    when (1) they are seen together — `_covisible`: the keyframes of the smaller one judge,
    by the user's rule against `covis_share` (segmentation.dedupe_overlap) — OR the walk never
    CAME BACK between them (`_camera_returned`, `min_walk_m` = correction.visit_drift.min_walk_m,
    the USER's 1 m: a drift duplicate needs a second pass near the first, while a row of desks
    is walked past once, one after the other) — AND (2) the gap between them is free space
    the cameras see through (`_gap_is_free_space`: the informative rays judge, by the same
    rule; EVERY camera that sees both components votes, in keyframe order — point 105: no
    stride subsample). A decision that does not pass leaves the instance whole (the simplest)
    and its margins are recorded. Without `cam_centre` nothing is split.

    Child ids (point 105): numbered from ``id_base`` + 1 — the raw store's highest id, which
    no merge of this projection moves — in the order (parent instance id, the child's stable
    spatial key: its centroid's cell on the world-anchored gap grid), never from the maximum
    id among whatever survived the merges. ``max_cams`` is accepted for the old callers and
    ignored. Returns the number of instances added. Mutates `instances`."""
    from scipy import ndimage
    if decision is None:
        from config import cfg as _server_cfg
        decision = _decision_params(_server_cfg)
    factor, confidence, min_judges = decision
    added = 0
    chain = {}
    if cam_centre and min_walk_m is not None:                # walked distance at every keyframe
        order = sorted(cam_centre)
        C = np.array([cam_centre[f] for f in order])
        run = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        chain = dict(zip(order, run.tolist()))
    tree_box = [None]                                         # the cloud's KD-tree, built on first need
    if id_base is None:
        id_base = max([int(i.get("id", 0)) for i in instances] + [0])
    splits = []                                               # (parent iid, inst, groups, P, comp, owner, gi)
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        if len(gi) < 2 * min_points:
            continue
        P = xyz_display[gi]
        # the gap grid ANCHORED AT THE WORLD ORIGIN (docs/plan_determinismo.md point 102,
        # DECIDIDO): cell = floor(x / gap) in float64 from (0, 0, 0) — anchored at the instance's
        # minimum, one extra extreme point moved every cell boundary and with it which fragments
        # touched; now an extra point changes only its own cell. The array index is shifted by the
        # lowest cell (labelling is translation-invariant: the components are the same).
        k = np.floor(P.astype(np.float64) / gap_m).astype(np.int64)
        k0 = k.min(0)
        ki = k - k0
        grid = np.zeros(ki.max(0) + 1, dtype=bool)
        grid[tuple(ki.T)] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
        if n < 2:
            continue
        comp = lab[tuple(ki.T)] - 1
        size = np.bincount(comp, minlength=n)
        cand = [c for c in np.argsort(-size, kind="stable") if size[c] >= min_points]
        if len(cand) < 2:
            continue
        ctr = {c: P[comp == c].mean(0) for c in cand}
        kfs = {c: set(np.unique(frame_arr[gi[comp == c]]).tolist()) for c in cand}
        if tree_box[0] is None and cam_centre:
            from scipy.spatial import cKDTree
            tree_box[0] = cKDTree(xyz_display)
        reach = float(np.linalg.norm(xyz_display.max(0) - xyz_display.min(0)))
        parent = {c: c for c in cand}

        def _root(x):
            while parent[x] != x:
                x = parent[x]
            return x
        iid = int(inst.get("instance_id", inst.get("id", 0)))
        for ia_, c1 in enumerate(cand):
            for c2 in cand[ia_ + 1:]:
                cv = _covisible(kfs[c1], kfs[c2], covis_share, factor=factor,
                                confidence=confidence, min_judges=min_judges)
                covis = bool(cv["improves"])
                # a drift duplicate needs the camera to COME BACK: seen from two passes whose cameras
                # stand within min_walk_m of each other with more than min_walk_m of walk between them.
                # A row of desks walked past once never has that — two objects, not two copies.
                dup_possible = (not covis) and bool(chain) and _camera_returned(kfs[c1], kfs[c2], cam_centre,
                                                                                 chain, min_walk_m)
                distinct = False
                fs_rec = None
                if not dup_possible and cam_centre:
                    # every camera that sees both components, in keyframe order (point 105)
                    common = kfs[c1] & kfs[c2]
                    fs = sorted(f for f in (common or (kfs[c1] | kfs[c2])) if f in cam_centre)
                    cams = np.array([cam_centre[f] for f in fs]) if fs else np.zeros((0, 3))
                    if len(cams):
                        fs_rec = _gap_is_free_space(P[comp == c1], P[comp == c2], tree_box[0], cams,
                                                    gap_m, reach, factor=factor,
                                                    confidence=confidence, min_judges=min_judges)
                        distinct = bool(fs_rec["improves"])
                if record is not None:
                    record.append({"kind": "covisible_split", "instance_id": iid,
                                   "components": [int(size[c1]), int(size[c2])],
                                   "covisible": {k_: cv[k_] for k_ in ("improves", "n_judges", "share",
                                                                      "ci_margin", "judges_margin",
                                                                      "error_margin", "reason")},
                                   "camera_returned": bool(dup_possible),
                                   "free_space": ({k_: fs_rec[k_] for k_ in ("improves", "n_rays",
                                                                             "share_through", "ci_margin",
                                                                             "judges_margin", "error_margin",
                                                                             "reason")}
                                                  if fs_rec is not None else None),
                                   "distinct": bool(distinct)})
                if not distinct:
                    parent[_root(c2)] = _root(c1)
        roots = {}
        for c in cand:
            roots.setdefault(_root(c), []).append(c)
        groups = sorted(roots.values(), key=lambda g: (-sum(size[x] for x in g), min(g)))
        if len(groups) < 2:
            continue
        owner = np.full(n, -1, dtype=np.int64)
        for gidx, g in enumerate(groups):
            owner[g] = gidx
        for c in range(n):                                   # crumbs -> the nearest candidate's group
            if owner[c] < 0:
                cc = P[comp == c].mean(0)
                owner[c] = owner[min(cand, key=lambda m: (float(np.linalg.norm(cc - ctr[m])), m))]
        splits.append((iid, inst, groups, P, comp, owner, gi))

    if not splits:
        return 0
    # the children's ids: by (parent instance id, the child's stable spatial key), from the raw
    # store's highest id — the same input gives the same ids whatever merged upstream
    children = []
    for iid, inst, groups, P, comp, owner, gi in splits:
        pt_group = owner[comp]
        for gidx in range(1, len(groups)):
            sel_pts = P[pt_group == gidx]
            key = tuple(int(v) for v in np.floor(sel_pts.mean(0) / gap_m).astype(np.int64))
            children.append((iid, key, inst, gidx))
    children.sort(key=lambda t: (t[0], t[1], t[3]))
    child_id = {(iid, gidx): int(id_base) + 1 + k for k, (iid, _key, _inst, gidx) in enumerate(children)}
    by_inst = {id(inst): (iid, groups, P, comp, owner, gi) for iid, inst, groups, P, comp, owner, gi in splits}
    new_list = []
    for inst in instances:
        hit = by_inst.get(id(inst))
        if hit is None:
            new_list.append(inst)
            continue
        iid, groups, P, comp, owner, gi = hit
        pt_group = owner[comp]
        for gidx in range(len(groups)):
            sel = sorted(gi[pt_group == gidx].tolist())
            if gidx == 0:
                inst["globalIndices"] = sel
                inst["total_points"] = len(sel)
                inst["split_into"] = len(groups)
                new_list.append(inst)
            else:
                nid = child_id[(iid, gidx)]
                new_list.append({**{k_: v for k_, v in inst.items()
                                    if k_ not in ("globalIndices", "obb", "split_into")},
                                 "id": nid, "instance_id": nid + 1, "globalIndices": sel,
                                 "total_points": len(sel),
                                 "split_from": int(iid)})
                added += 1
        print(f"[SegPipeline]   ✂ '{inst.get('label')}' #{iid}: "
              f"{len(groups)} objects seen together — split "
              f"({[int((pt_group == g_).sum()) for g_ in range(len(groups))]} pts)")
    instances[:] = new_list
    return added


def _merge_label_fragments(instances, xyz_display: np.ndarray, gap_m: float,
                           record=None, min_adjacent_pairs: Optional[int] = None,
                           decisions: Optional[list] = None):
    """Consolidate instances of the SAME label whose points are contiguous.

    SAM3 returns one mask per visually separable region, so one physical
    surface arrives as many instances carrying one label: pccr 2026-09-14 had
    48 'white_tiled_floor' instances for a single floor — one of 244k points
    and 47 fragments of 4–20k — and test3 had 9 'green_decorative_tile_border'.
    That is what the user sees as "el mismo objeto muchas veces", and it is
    also why no instance can carry two copies of itself, which leaves the
    duplicate detector with nothing to find.

    Geometry decides, as everywhere else: two instances with the SAME label
    whose occupied voxels touch within ``gap_m`` are one object — touching
    through at least ``min_adjacent_pairs`` distinct adjacent voxel pairs
    (docs/plan_determinismo.md point 104, DECIDIDO: 5, the judges' minimum of
    the user's rule; one flyer voxel used to weld two objects). Two rules keep
    this away from the containment merge that ``merge_duplicates`` disabled
    after it collapsed 147 instances to 3:
      * different labels are NEVER merged, so a wall cannot swallow the signs
        resting against it;
      * the test is CONTIGUITY, not containment, so nothing is absorbed for
        sitting inside somebody else's envelope.
    Two copies of one object left apart by drift are not contiguous either, so
    they survive as separate instances for the duplicate machinery.

    Mutates ``instances`` in place; returns the number of instances absorbed.
    ``record``, when given, receives ``absorbed_instance_id -> keeper`` so the
    session can say where every mask ended up instead of leaving it as a
    zero-point ghost in the list (USER 2026-09-17). ``decisions`` receives every
    touching pair with its count of adjacent voxel pairs and the verdict.
    """
    if gap_m <= 0 or len(instances) < 2:
        return 0
    if min_adjacent_pairs is None:
        from config import cfg as _server_cfg
        min_adjacent_pairs = _decision_params(_server_cfg)[2]
    from collections import defaultdict

    by_label = defaultdict(list)
    for k, inst in enumerate(instances):
        by_label[inst.get("label", "object")].append(k)

    parent = list(range(len(instances)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    offsets = np.array([[dx, dy, dz]
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                       dtype=np.int64)

    for label, members in by_label.items():
        if len(members) < 2:
            continue
        keys, owners = [], []
        for k in members:
            gi = np.asarray(instances[k].get("globalIndices") or [], dtype=np.int64)
            gi = gi[(gi >= 0) & (gi < len(xyz_display))]
            if not len(gi):
                continue
            v = np.unique(np.floor(xyz_display[gi] / gap_m).astype(np.int64), axis=0)
            keys.append(v)
            owners.append(np.full(len(v), k, dtype=np.int64))
        if len(keys) < 2:
            continue
        allv = np.concatenate(keys)
        allo = np.concatenate(owners)
        # pack (i,j,k) into one int64 so the neighbour lookup is a searchsorted
        base = allv.min(axis=0)
        span = (allv.max(axis=0) - base + 3).astype(np.int64)
        if np.prod(span.astype(float)) > 9.0e18:       # degenerate extent — skip this label
            continue

        def pack(v):
            d = v - base + 1
            return (d[:, 0] * span[1] + d[:, 1]) * span[2] + d[:, 2]

        packed = pack(allv)
        order = np.argsort(packed, kind="stable")
        packed_sorted, owner_sorted = packed[order], allo[order]
        pairs = []
        for off in offsets:
            if not off.any():
                continue
            nb = pack(allv + off)
            pos = np.searchsorted(packed_sorted, nb)
            ok = pos < len(packed_sorted)
            pos = np.where(ok, pos, 0)
            hit = ok & (packed_sorted[pos] == nb)
            if not hit.any():
                continue
            a, b = allo[hit], owner_sorted[pos[hit]]
            diff = a != b
            if diff.any():
                # the adjacent VOXEL pair (both cells) with its instance pair: distinct
                # pairs are what the contiguity is counted on
                va, vb = packed[hit][diff], nb[hit][diff]
                pairs.append(np.stack([np.minimum(a[diff], b[diff]), np.maximum(a[diff], b[diff]),
                                       np.minimum(va, vb), np.maximum(va, vb)], axis=1))
        if pairs:
            # millions of voxel adjacencies collapse to a handful of instance
            # pairs — unique them (by voxel pair) and COUNT them per instance pair
            allp = np.unique(np.concatenate(pairs), axis=0)
            inst_pairs, counts = np.unique(allp[:, :2], axis=0, return_counts=True)
            for (a, b), cnt in zip(inst_pairs, counts):
                contiguous = int(cnt) >= int(min_adjacent_pairs)
                if decisions is not None:
                    decisions.append({"kind": "fragment_contiguity", "label": label,
                                      "instance_ids": [int(instances[a].get("instance_id", instances[a].get("id"))),
                                                       int(instances[b].get("instance_id", instances[b].get("id")))],
                                      "adjacent_voxel_pairs": int(cnt),
                                      "min_adjacent_pairs": int(min_adjacent_pairs),
                                      "margin": int(cnt) - int(min_adjacent_pairs),
                                      "merged": contiguous})
                if not contiguous:
                    continue
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    groups = defaultdict(list)
    for k in range(len(instances)):
        groups[find(k)].append(k)
    absorbed = 0
    for root, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda k: (-len(instances[k].get("globalIndices") or []), k))
        keep = members[0]
        idx = set(instances[keep].get("globalIndices") or [])
        for k in members[1:]:
            idx |= set(instances[k].get("globalIndices") or [])
            instances[k]["_absorbed_into"] = keep
            if record is not None:
                record[int(instances[k].get("instance_id", instances[k].get("id")))] = {
                    "into": int(instances[keep].get("instance_id", instances[keep].get("id"))),
                    "into_label": instances[keep].get("label"),
                    "reason": "fragment",
                }
            absorbed += 1
        merged = sorted(idx)
        instances[keep]["globalIndices"] = merged
        instances[keep]["total_points"] = len(merged)
        print(f"[SegPipeline]   🧩 fragments: '{instances[keep].get('label')}' "
              f"#{instances[keep].get('id')} absorbed {len(members) - 1} contiguous "
              f"fragment(s) of the same label → {len(merged):,} pts")
    if absorbed:
        instances[:] = [i for i in instances if "_absorbed_into" not in i]
    return absorbed


def _attach_unsegmented(instances, xyz_display: np.ndarray,
                        attach_dist_m: float, rounds: int, votes: int,
                        block: int = 4_000_000):
    """Grow every instance into the unsegmented cloud around it.

    SAM3 runs on the KEYFRAMES, and the mask→cloud step only labels a point
    whose OWN origin frame carries a mask (exact match, on purpose: a mask
    from another timestamp maps background onto the wrong surface). A cloud
    built from 3,047 frames and segmented on 216 of them can therefore never
    exceed ~7% coverage no matter how good the masks are — pccr 2026-09-14
    measured 5.35%, test3 11.5%. The surface is the same surface; only its
    birth certificate differs.

    So the label travels through SPACE instead: a point within
    ``attach_dist_m`` of labelled points adopts the instance that the
    majority of its ``votes`` nearest labelled neighbours belong to (ties go
    to the nearest). ``rounds`` passes let a surface grow outwards a few
    centimetres at a time, each round starting from what the previous one
    labelled, so a wall is walked across rather than jumped across. Points
    nobody reaches stay unsegmented: nothing is invented.

    One KD-tree per round over ALL labelled points, not one per instance —
    the old shape was 105 trees × 26 M queries and was never affordable.

    Mutates ``instances`` in place; returns (n_attached, grown_positions).
    """
    from scipy.spatial import cKDTree

    N = len(xyz_display)
    owner = np.full(N, -1, dtype=np.int64)
    for k, inst in enumerate(instances):
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        owner[gi[(gi >= 0) & (gi < N)]] = k
    n_before = int((owner >= 0).sum())
    if not n_before:
        return 0, []
    votes = max(1, int(votes))
    rounds = max(1, int(rounds))

    for r in range(rounds):
        lab_idx = np.nonzero(owner >= 0)[0]
        un_idx = np.nonzero(owner < 0)[0]
        if not len(un_idx):
            break
        tree = cKDTree(xyz_display[lab_idx])
        lab_owner = owner[lab_idx]
        miss = len(lab_idx)
        new_owner = np.full(len(un_idx), -1, dtype=np.int64)
        for s in range(0, len(un_idx), block):
            sl = slice(s, min(s + block, len(un_idx)))
            _, nb = tree.query(xyz_display[un_idx[sl]], k=votes,
                               distance_upper_bound=float(attach_dist_m), workers=-1)
            if votes == 1:
                nb = nb[:, None]
            hit = nb < miss
            cand = np.where(hit, lab_owner[np.where(hit, nb, 0)], -1)
            # the instance most of the k nearest labelled neighbours belong to;
            # the columns are ordered by distance, so a tie falls to the nearest.
            # A single dissenting neighbour cannot drag a point across a boundary.
            if votes < 3:
                win = cand[:, 0]
            else:
                agree = np.zeros(cand.shape, dtype=np.int16)
                for c in range(votes):
                    agree[:, c] = (((cand == cand[:, c][:, None]) & hit).sum(1)
                                   * hit[:, c].astype(np.int16))
                win = cand[np.arange(len(cand)), agree.argmax(1)]
            new_owner[sl] = win
        got = new_owner >= 0
        if not got.any():
            break
        owner[un_idx[got]] = new_owner[got]
        print(f"[SegPipeline]     📎 round {r + 1}/{rounds}: "
              f"+{int(got.sum()):,} point(s) within {attach_dist_m * 100:.0f} cm")

    n_attached = int((owner >= 0).sum()) - n_before
    if n_attached <= 0:
        return 0, []
    # rebuild every instance's index list in ONE pass: 105 scans over 27 M
    # points is not affordable, one argsort is
    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    first = int(np.searchsorted(sorted_owner, 0))
    counts = np.bincount(sorted_owner[first:], minlength=len(instances))
    bounds = np.concatenate([[0], np.cumsum(counts)])
    grown = []
    for k in range(len(instances)):
        gi = np.sort(order[first + bounds[k]: first + bounds[k + 1]])
        was = int(instances[k].get("total_points") or 0)
        instances[k]["globalIndices"] = gi.tolist()
        instances[k]["total_points"] = int(len(gi))
        if len(gi) > was:
            grown.append(k)
    return n_attached, grown


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


def segmentation_result_is_stale(output_dir) -> tuple:
    """(stale, reason) — may output/segmentation_result.json be REUSED for this
    state of the session? Only on an IDENTICAL stamp (docs/plan_determinismo.md
    point 123): the sha256 of every input the projection reads (the cloud, the
    raw mask store and its metadata, the keyframe list, the poses, the session
    camera, the floor transform, the record-grid declaration), the code that
    decides and the configuration sections it reads. A result with no stamp
    (written before 2026-10-08) is stale; an mtime never decides."""
    from repro import check_stamp
    out = Path(output_dir)
    res = out / "segmentation_result.json"
    if not res.exists():
        return True, "no segmentation_result.json"
    try:
        doc = json.loads(res.read_text())
    except ValueError as e:
        return True, f"segmentation_result.json is unreadable ({e})"
    saved = doc.get("stamp") if isinstance(doc, dict) else None
    if not isinstance(saved, dict):
        return True, "segmentation_result.json carries no stamp (written before 2026-10-08)"
    if not doc.get("instances"):
        return True, "segmentation_result.json holds no instance"
    try:
        now = projection_stamp(out, out / "cleaned_cloud.ply")
    except FileNotFoundError as e:
        return True, f"an input of the projection is missing ({e})"
    diffs = check_stamp(saved, now)
    if diffs:
        return True, "stamp differs: " + "; ".join(diffs[:6]) + (" …" if len(diffs) > 6 else "")
    return False, "identical stamp (inputs, code and configuration)"


def _mask_frame_lookup(output_dir: Path, mask_frames, cloud_frames):
    """Map a cloud frame_global to the frame index the MASKS are keyed by.

    Thin wrapper over ``segmentation.mask_space`` — kept because eight call
    sites already speak this contract: it returns {cloud frame: mask frame},
    EMPTY when the identity is right.

    The two spaces: ``_prepare_valid_frames`` copies the keyframes into
    frames_valid/ renumbered 000000, 000001, … and SAM3 keys its masks by
    that POSITION, while the reconstruction stamps every point with the REAL
    video frame number (1, 60, 97, … 3026). Comparing them directly only ever
    matched the handful of keyframes whose video number happens to fall below
    the keyframe count — pccr 2026-09-14: 13 of 216, and those 13 took the
    mask of the wrong keyframe. Coverage was 5.35% for that reason alone, and
    the patchwork it produced is what shattered one floor into 48 instances.

    ``mask_frames`` / ``cloud_frames`` are no longer needed to DECIDE (the
    store declares its space, and a legacy store is measured against
    camera_frames.txt, which is the same list the poses were read from); they
    are still accepted, and the translation is restricted to the cloud frames
    the caller actually has, so the returned dict means what it always meant.
    """
    ms = mask_space.resolve(output_dir, log=lambda m: print(f"[SegPipeline]   {m}"))
    c2m = ms.cloud_to_mask()
    if not c2m:
        return {}
    have = {int(f) for f in cloud_frames} if cloud_frames is not None else None
    mask_set = ({int(f) for f in mask_frames}
                if mask_frames is not None else set(c2m.values()))
    out = {c: m for c, m in c2m.items()
           if (have is None or c in have) and (not mask_set or m in mask_set)}
    return out


def _mask_fates(metadata: dict, instances: list, absorbed_into: dict) -> dict:
    """What happened to every mask in ``segmentation.json`` that is NOT one of
    the instances the matching produced.

    A mask leaves the object list for one of four reasons, and until now only
    the first three were even written down — in the log, which nothing reads:

      space_dedupe  another instance occupies the same space (they ARE one
                    object under two names)
      fragment      same label, contiguous points: one physical surface SAM3
                    returned as many masks (pccr: 84 white_tiled_floor masks
                    for one floor)
      too_small     fewer points than ``min_instance_points``
      unmatched     no point of the cloud carries that mask — nothing was
                    fused, there is simply nothing there to show

    The survivors are the objects; everything here is provenance. Keeping the
    two apart is the whole point: the list endpoint must be able to tell a mask
    that was ABSORBED from one that is merely waiting for the next matching
    pass, and only the second belongs in the list.
    """
    survivors = {int(i.get("instance_id", i.get("id")))
                 for i in instances if i.get("instance_id", i.get("id")) is not None}
    fates = {}
    for raw in (metadata.get("instances") or []):
        iid = raw.get("instance_id", raw.get("id"))
        if iid is None:
            continue
        iid = int(iid)
        if iid in survivors:
            continue
        rec = absorbed_into.get(iid)
        fates[iid] = dict(rec) if rec else {
            "into": None, "into_label": None, "reason": "unmatched"}
        fates[iid].setdefault("label", raw.get("label"))
    # a keeper that was itself absorbed later (fragment chain) resolves to the
    # instance that actually survived, so the UI never points at a ghost — and
    # the label travels with it, or the row would name an object that is gone
    labels = {int(i.get("instance_id", i.get("id"))): i.get("label")
              for i in instances if i.get("instance_id", i.get("id")) is not None}
    for iid, rec in fates.items():
        seen = {iid}
        into = rec.get("into")
        while into is not None and into not in survivors and into not in seen:
            seen.add(into)
            into = (fates.get(into) or {}).get("into")
        rec["into"] = into if into in survivors else None
        rec["into_label"] = labels.get(rec["into"]) if rec["into"] is not None else None
    return fates


def _instance_id(inst: dict) -> int:
    """The id the class byte carries: the INSTANCE id, never the mask obj id
    (`id` = instance_id − 1 on SAM3 masklets — writing it painted each object
    with its neighbour's class, segmentation/republish.write_classification)."""
    return int(inst.get("instance_id", inst.get("id", 0)))


def _encode_classification(instances: list, classification: np.ndarray,
                           prev_map: Optional[dict]):
    """(classification, instance_id → byte, class_map) for the projection's writer.

    pccr 2026-09-30: this writer stored `min(obj id, 255)`, so every instance id
    above 254 — 50 objects there, the four drywalls, the backpack, beams,
    windows… — shared byte 255 and ONE viewer toggle switched all of them. The
    byte now comes from the same encoder as every other writer
    (`segmentation.republish._encode`: identity while ids fit, a compact 1..N with
    `class_map.json` when they do not). In incremental mode (`prev_map` given) the
    bytes already in `classification` are translated through the previous map
    into the new one, so a re-projected subset cannot shift the others."""
    from segmentation.republish import _encode
    ids = [_instance_id(i) for i in instances]
    prev_codes, prev_ids = [], []
    if prev_map is not None:
        inst_of = {int(k): int(v) for k, v in (prev_map.get("instance_of") or {}).items()}
        prev_codes = [int(c) for c in np.unique(classification) if int(c) > 0]
        prev_ids = [inst_of.get(c, c) for c in prev_codes]      # no map = identity
        ids = ids + prev_ids
    class_map = _encode(ids)
    code = {int(k): int(v) for k, v in class_map["class_of"].items()}
    if prev_codes:
        lut = np.zeros(256, dtype=np.uint8)
        for c, iid in zip(prev_codes, prev_ids):
            lut[c] = code.get(iid, 0)
        classification = lut[classification]
    return classification, code, class_map


def _frame_groups_of(frame_arr: np.ndarray) -> Dict[int, np.ndarray]:
    """{cloud frame: indices of the points born in it}, each sorted ascending."""
    order = np.argsort(frame_arr, kind="stable")
    sorted_f = frame_arr[order]
    uniq, starts = np.unique(sorted_f, return_index=True)
    ends = np.r_[starts[1:], len(sorted_f)]
    return {int(f): np.sort(order[s:e]) for f, s, e in zip(uniq.tolist(), starts, ends)}


def _area_variation_by_oid(areas: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """Per object, the MEASURED frame-to-frame variation of its mask area: the median
    |Δarea| between consecutive mask keyframes (point 117: the noise two near-equal
    areas are compared against); 0 for an object with one frame."""
    out = {}
    for oid, by_frame in areas.items():
        a = np.asarray([by_frame[f] for f in sorted(by_frame)], np.float64)
        out[int(oid)] = float(np.median(np.abs(np.diff(a)))) if len(a) > 1 else 0.0
    return out


def _match_masks_to_cloud(output_dir, ply_path=None) -> dict:
    """
    Core processing: match SAM3 masks against the PLY cloud — the mask→point
    assignment, the audit, the merges, the split, the attach, the OBBs, the
    class bytes, the octree — and return the result document.

    A PURE FUNCTION of its inputs (docs/plan_determinismo.md points 99 / 122):
    the cloud, the RAW mask store and its metadata (never a fused parent, never
    a previous result), the keyframe list, the poses, the session camera, the
    floor transform, the frozen configuration and this code. It starts from an
    empty result; every step is FATAL — a step that cannot run raises, names
    itself, and no result is written (the caller deletes a stale one). The
    result carries its ``stamp`` (point 123) and every decision's margin
    (``decisions``).

    This is CPU-intensive. Called once after segmentation to produce
    segmentation_result.json. Use apply_segmentation_to_cloud() for cached loading.
    """
    cfg, cfg_source = _projection_config(Path(output_dir))
    output_dir = Path(output_dir)
    error_factor, confidence, min_judges = _decision_params(cfg)
    decisions: Dict[str, object] = {
        "rule": ("USER 2026-10-07 (metric_lock.decide_change): a merge / split / link is applied "
                 "only when significant at the declared confidence, with >= min_judges judges, "
                 "by >= error_factor x the measured error; otherwise the simplest outcome"),
        "error_factor": error_factor, "confidence": confidence, "min_judges": min_judges}

    # Load metadata (the RAW parent: the store SAM3 wrote, plus the interactive manager's
    # masklets — never the fused view, which is what this very function derives)
    seg_path = output_dir / "segmentation.json"
    if not seg_path.exists():
        raise RuntimeError(f"{seg_path} does not exist — nothing to project")
    with open(seg_path) as f:
        metadata = json.load(f)

    # Load masks
    masks_path = output_dir / metadata.get("mask_file", "seg_masks.npz")
    if not masks_path.exists():
        raise RuntimeError(f"{masks_path} does not exist — the masks to project are missing")

    masks_data = np.load(masks_path)
    mask_keys = set(masks_data.files)
    obj_ids = sorted(int(o) for o in masks_data["obj_ids"].tolist())
    keyframes = masks_data["frames"].tolist()
    scaled_res = [int(x) for x in masks_data["scaled_res"].tolist()]

    # The scene cloud, or nothing. A chunk used to stand in for it when it was
    # missing, which cannot work: a chunk is one seventh of the scene and
    # carries no per-point provenance, so the match silently degraded to
    # instances with no points and the caller cached that as the session's
    # segmentation. Refusing is the only honest answer — the cloud does not
    # exist yet, so there is nothing to map onto.
    if ply_path is None:
        ply_path = output_dir / "cleaned_cloud.ply"
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise RuntimeError(f"no {ply_path.name} in {output_dir} — the cloud does not exist yet, "
                           f"nothing to map masks onto")

    # Load cloud origins
    origins = _load_ply_origins(ply_path)
    if origins is None:
        raise RuntimeError(f"{ply_path.name} carries no origin fields (frame_global / pixel_row / "
                           f"pixel_col) — the masks cannot be projected onto it")

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
        data = np.load(transform_path)
        s = float(data["s"])
        R = data["R"]
        t = data["t"]
        if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
            xyz_display = s * (xyz @ R.T) + t
            print(f"[SegPipeline]   Floor alignment loaded from {transform_path.name}")
    elif (output_dir / ".orientation_applied").exists():
        # reconstruction/orient.py baked +Y up and the floor at y=0 into the cloud
        # itself, measured from the camera-pose gravity over every frame. The raw
        # cloud IS the display frame — any further leveling would rotate it a second
        # time and _compute_obb's Y-up assumption would then hold in no frame at all.
        print("[SegPipeline]   Orientation baked from camera poses — display frame is identity")
    else:
        # Fallback: compute alignment (legacy sessions without a saved transform) —
        # seeded and keyed by the cloud (point 110)
        from alignment_manager import get_alignment_manager
        am = get_alignment_manager()
        s, R, t = am.compute_leveling_from_points(xyz)
        if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
            xyz_display = s * (xyz @ R.T) + t
            print(f"[SegPipeline]   Floor alignment computed (no saved transform)")

    # Group cloud points by frame for efficient lookup
    frame_arr = frame_global.astype(np.int32)
    frame_groups = _frame_groups_of(frame_arr)
    print(f"[SegPipeline]   {len(frame_groups)} unique frames in cloud")

    # the masks and the cloud index their frames differently — translate, EXACTLY
    # (point 121): a cloud frame that is not in camera_frames.txt FAILS the stage,
    # a keyframe without a mask contributes no point; the identity fallback that
    # read a video number as a position is gone. (A store declaring the keyframe
    # list it was segmented on is checked against camera_frames.txt here — point 114.)
    ms = mask_space.resolve(output_dir, masks=masks_data,
                            log=lambda m: print(f"[SegPipeline]   {m}"))
    mask_frame_of: Dict[int, int] = {}
    for cloud_frame in sorted(frame_groups):
        mf = ms.to_mask(cloud_frame)
        if mf is None:
            if ms.keyframes:
                raise RuntimeError(f"cloud frame {cloud_frame} is not in camera_frames.txt "
                                   f"({len(ms.keyframes)} keyframes) — the cloud and the keyframe "
                                   f"list are not one reconstruction")
            mf = cloud_frame                      # no keyframe list: the ordinal IS the frame
        mask_frame_of[cloud_frame] = int(mf)
    cloud_to_mask = {} if ms.is_identity else {f: m for f, m in mask_frame_of.items()}

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
    erosion_iterations = int((cfg.get("segmentation", {}) or {}).get("mask_erosion_iterations", 0))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ── THE GRIDS (point 108, DECIDIDO): the cloud's birth pixels live on the DECLARED
    # record grid (camera.json / corrected_cloud.json, `trace_grid`) and the masks on
    # their declared `scaled_res`; the exact grid maps of precision.camera take one to
    # the other (SessionProjection.record_to_mask — what the certification already
    # does). Nothing is inferred from the surviving pixels' maxima or from chunk
    # metadata; a session that declares no grid FAILS here.
    from correction.visit_drift import SessionProjection, trace_grid
    from precision.camera import CAMERA_JSON_NAME, grid_like, load_camera_json, mask_grid_for
    cam = load_camera_json(output_dir / CAMERA_JSON_NAME)
    Ht, Wt = trace_grid(output_dir)
    g = cam.omega_grid
    rec_grid = g if (int(g.w), int(g.h)) == (Wt, Ht) else grid_like(g, Wt, Ht, "record")
    mask_grid = mask_grid_for(cam.width, cam.height, (scaled_res[0], scaled_res[1]))
    proj = SessionProjection(cam, rec_grid, mask_grid,
                             source=f"{CAMERA_JSON_NAME} ({cam.source}, camera epoch {cam.camera_epoch})")
    if n_pts and (int(pixel_row.max()) >= Ht or int(pixel_col.max()) >= Wt
                  or int(pixel_row.min()) < 0 or int(pixel_col.min()) < 0):
        raise RuntimeError(f"the cloud's birth pixels span rows {int(pixel_row.min())}..{int(pixel_row.max())}, "
                           f"cols {int(pixel_col.min())}..{int(pixel_col.max())} outside the declared "
                           f"{Wt}x{Ht} record grid — the cloud and the camera do not describe one grid")
    rows_m, cols_m = proj.record_to_mask(pixel_row, pixel_col)        # -1 where the mask grid ends
    mask_h_ref, mask_w_ref = scaled_res[0], scaled_res[1]
    print(f"[SegPipeline]   Record grid: {Wt}x{Ht} ({rec_grid.name}), mask grid: {mask_w_ref}x{mask_h_ref} "
          f"— exact grid maps of {proj.source}")

    # ── Phase 1: Match all objects, track per-point best assignment ──
    # For deconfliction: each point goes to the object with the smallest mask area
    # (the user's rule, kept — point 117, DECIDIDO: the smaller area wins, a tie the
    # lower id; the runner-up is tracked so the NEAR-TIES can be counted and declared)
    point_obj_id = np.full(n_pts, -1, dtype=np.int32)     # winning obj_id per point
    point_mask_area = np.full(n_pts, np.inf, dtype=np.float64)  # smaller wins
    second_obj_id = np.full(n_pts, -1, dtype=np.int32)    # the runner-up and its area
    second_area = np.full(n_pts, np.inf, dtype=np.float64)

    obj_mask_areas = {}  # obj_id → average mask area (for priority)
    areas_by_oid_frame: Dict[int, Dict[int, float]] = {}

    # A LONG STAGE MUST SAY WHERE IT IS (USER 2026-09-22: *"etapas largas mudas
    # se puede y se debe solucionar ahora porque voy a lanzar desde ui y sino no
    # voy a recibir nada por mucho tiempo"*). This loop projects every masklet
    # over the whole cloud and printed nothing until it finished — 10 minutes of
    # silence on pccr, indistinguishable from a hang.
    _t_match = _time.time()
    _n_obj = len(obj_ids)
    _step = max(1, _n_obj // 20)          # ~20 lines, whatever the session size
    _done = 0
    for obj_id in obj_ids:
        _done += 1
        if _done == 1 or _done % _step == 0 or _done == _n_obj:
            _el = _time.time() - _t_match
            _eta = (_el / _done) * (_n_obj - _done)
            print(f"[SegPipeline]   matching {_done}/{_n_obj} masklets "
                  f"({100.0 * _done / max(_n_obj, 1):.0f}%, {_el:.0f}s elapsed"
                  + (f", ~{_eta:.0f}s left)" if _done < _n_obj else ")"),
                  flush=True)
        # Compute average mask area for this object (across all frames)
        frame_areas = []

        for cloud_frame in sorted(frame_groups):
            pt_indices = frame_groups[cloud_frame]
            # EXACT MATCH ONLY: if SAM3 didn't generate a mask for this exact frame, skip these points.
            # No fuzzy "nearest frame" matching, because that maps background points from unsegmented frames
            # to masks from completely different timestamps.
            mask_key = f"f{mask_frame_of[cloud_frame]}_o{obj_id}"
            if mask_key not in mask_keys:
                continue

            mask = masks_data[mask_key].astype(np.uint8)
            if mask.shape[0] != mask_h_ref or mask.shape[1] != mask_w_ref:
                raise RuntimeError(f"mask {mask_key} is {mask.shape[0]}x{mask.shape[1]} and the store "
                                   f"declares {mask_h_ref}x{mask_w_ref} — one store, one grid")

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
            areas_by_oid_frame.setdefault(int(obj_id), {})[int(cloud_frame)] = mask_area

            # Look up each point's pixel in the (eroded) mask through the exact grid maps
            r = rows_m[pt_indices]
            c = cols_m[pt_indices]
            ok = (r >= 0) & (c >= 0)
            in_mask = np.zeros(len(pt_indices), bool)
            in_mask[ok] = mask[r[ok], c[ok]]

            matched = pt_indices[in_mask]

            # ── Deconfliction: assign point to smallest-mask object ──
            # Vectorized: only update points where this mask_area is smaller;
            # the previous winner becomes the runner-up, a loser that beats the
            # runner-up replaces it (the near-tie count reads both)
            cur = point_mask_area[matched]
            wins = mask_area < cur
            winning_pts = matched[wins]
            second_obj_id[winning_pts] = point_obj_id[winning_pts]
            second_area[winning_pts] = cur[wins]
            point_obj_id[winning_pts] = obj_id
            point_mask_area[winning_pts] = mask_area
            losing = matched[~wins]
            better_second = mask_area < second_area[losing]
            second_obj_id[losing[better_second]] = obj_id
            second_area[losing[better_second]] = mask_area

        avg_area = np.mean(frame_areas) if frame_areas else 0
        obj_mask_areas[obj_id] = avg_area

    # the NEAR-TIES of the smaller-area rule (point 117): contested points whose two
    # smallest areas differ by less than error_factor x the measured frame-to-frame
    # variation of those masks' areas — counted and declared, the rule unchanged
    area_var = _area_variation_by_oid(areas_by_oid_frame)
    contested = np.flatnonzero(np.isfinite(second_area))
    if len(contested):
        var_best = np.asarray([area_var.get(int(o), 0.0) for o in point_obj_id[contested]])
        var_second = np.asarray([area_var.get(int(o), 0.0) for o in second_obj_id[contested]])
        gap = second_area[contested] - point_mask_area[contested]
        near = gap < error_factor * np.maximum(var_best, var_second)
        n_near = int(near.sum())
        exact = int((gap == 0).sum())
    else:
        n_near, exact = 0, 0
    decisions["mask_area_near_ties"] = {
        "rule": "a point inside two masks goes to the smaller area (tie: the lower id) — kept; "
                "near-ties = gap < error_factor x the masks' measured frame-to-frame area variation",
        "n_points_contested": int(len(contested)), "n_points_near_tie": n_near,
        "n_points_exact_tie": exact,
        "area_variation_median_px": float(np.median(list(area_var.values()))) if area_var else 0.0}
    if n_near:
        print(f"[SegPipeline]   ⚖ {n_near:,} of {len(contested):,} contested point(s) sit within the "
              f"masks' own area variation of the smaller-area rule — declared, rule unchanged")
    del second_area, second_obj_id

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
    # Where every mask of segmentation.json ended up. Without this the list
    # endpoint cannot tell a mask that was FUSED into another object from one
    # that is merely waiting for the next matching pass, so the fused ones sat
    # in the segmentation list forever showing 0 points (USER 2026-09-17:
    # "aparecen muchisimos en cero ... deben ser los que despues se
    # fusionaron, pero quedaron en cero y siguen apareciendo en la lista").
    absorbed_into: Dict[int, dict] = {}

    # ── AUDIT of the cloud against the masks (USER 2026-09-15) ──────────────
    # "no hay nada que cortar, es la auditoría de tu propia nube contra el
    # ground truth de la máscara". It MEASURES where each instance's mass falls
    # relative to its mask and removes nothing — the numbers travel to whoever
    # corrects the geometry. A configured audit that cannot be built FAILS the
    # projection (point 122) — it used to be skipped.
    mask_filter = None
    mf_cfg = (cfg.get("segmentation", {}) or {}).get("mask_filter", {}) or {}
    if mf_cfg.get("enabled"):
        from segmentation.mask_filter import MaskAudit
        mask_filter = MaskAudit(output_dir, Path(output_dir).parent, mf_cfg,
                                cloud_to_mask=cloud_to_mask,
                                log=lambda m: print(f"[SegPipeline] {m}"))

    # Marked, never removed. A point landing off its own mask in any view that
    # sees it is in the wrong PLACE; the certification may still move it there,
    # and only what is still wrong afterwards has no correction left. USER
    # 2026-09-15: "sé que ese punto debe estar, no es ruido, es un huérfano mal
    # ubicado, si no lo puedo corregir, lamentablemente lo voy a tener que
    # sacar" — the sacar is the SECOND pass, at the tail of the certification,
    # reading this file.
    out_of_place = np.zeros(n_pts, bool)
    judged_views = np.zeros(n_pts, np.uint16)           # point 119: the views that judged each point
    off_views = np.zeros(n_pts, np.uint16)              # and the ones that left it outside

    # The ShapeR description every instance inherits from its CONCEPT (USER 2026-10-01:
    # the VLM pass that named the SAM3 prompts also described each kind —
    # vlm_analysis.json `shape_descriptions`); the per-object one that replaces it is
    # attached after the certification (segmentation/object_captioner.py).
    from segmentation.object_captioner import concept_caption_lookup
    _concept_caption = concept_caption_lookup(output_dir)

    for iid in sorted(instance_groups):
        group_obj_ids = instance_groups[iid]
        # Merge all points assigned to any obj_id in this instance group
        all_matched = np.where(np.isin(point_obj_id, group_obj_ids))[0].astype(np.int64)

        if len(all_matched) == 0:
            continue

        if mask_filter is not None:
            _a = mask_filter.audit(int(iid), xyz[all_matched], frame_arr[all_matched])
            _o = _a.get("out_of_place")
            if _o is not None and len(_o) == len(all_matched):
                out_of_place[all_matched[_o]] = True
                judged_views[all_matched] = np.minimum(_a["judged_views"], 65535).astype(np.uint16)
                off_views[all_matched] = np.minimum(_a["off_views"], 65535).astype(np.uint16)

        # ── Per-instance DBSCAN outlier removal ──
        pre_filter_count = len(all_matched)
        voxel_mesh_data = []
        face_normals_data = []
        face_planes_data = []
        face_id_data = np.array([], dtype=np.int32)
        if pre_filter_count >= 20:
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
        _cap = _concept_caption(label)
        if _cap is not None:
            instance["shape_caption"] = _cap       # source 'concept', vlm_proposed

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

        instances.append(instance)
        removed = pre_filter_count - len(all_matched)
        filter_info = f" (filtered {removed} outliers)" if removed > 0 else ""
        print(f"[SegPipeline]   Object '{label}' #{iid}: "
              f"{len(all_matched):,} points{filter_info}")

    # ── Instance post-processing config ────────────────────────────────
    _dd = (cfg.get("segmentation", {}) or {})
    _merge_on = bool(_dd.get("merge_duplicates", False))

    # ── Phase 3: Cross-category Re-ID — merge instances with high 3D overlap ──
    # If VLM produced synonyms (e.g., "chair" + "wooden chair"), SAM3 may have
    # segmented the same physical object twice. Detect and merge by 3D point overlap.
    # The test is MUTUAL (segmentation.dedupe_mutual): both instances must be
    # mostly the intersection. `intersection / smaller` alone is CONTAINMENT and
    # lets a large instance absorb anything lying inside it.
    # (Phase 1 gives every point ONE obj id and instances are groups of obj ids, so
    # instance index sets are disjoint and this merge cannot fire — kept as the
    # recipe's step; the space dedupe below is the one that decides.)
    if _merge_on and len(instances) > 1:
        merge_threshold = float(_dd.get("dedupe_overlap", 0.8))
        _mutual_idx = bool(_dd.get("dedupe_mutual", True))
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

                overlap_ratio = intersection / min(len(set_i), len(set_j))
                if _mutual_idx:
                    overlap_ratio = intersection / max(len(set_i), len(set_j))

                if overlap_ratio >= merge_threshold:
                    # Merge: smaller into larger
                    if len(set_i) >= len(set_j):
                        # i absorbs j
                        set_i |= set_j
                        instances[i]["globalIndices"] = sorted(set_i)
                        instances[i]["total_points"] = len(set_i)
                        merged_away.add(j)
                        _record_overlap(absorbed_into, instances[j], instances[i],
                                        overlap_ratio)
                        print(f"[SegPipeline]   🔗 Merged '{instances[j]['label']}' #{instances[j]['id']} "
                              f"into '{instances[i]['label']}' #{instances[i]['id']} "
                              f"(overlap={overlap_ratio:.0%})")
                    else:
                        # j absorbs i
                        set_j |= set_i
                        instances[j]["globalIndices"] = sorted(set_j)
                        instances[j]["total_points"] = len(set_j)
                        merged_away.add(i)
                        _record_overlap(absorbed_into, instances[i], instances[j],
                                        overlap_ratio)
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
    _mutual = bool(_dd.get("dedupe_mutual", True))
    # NO MINIMUM (USER 2026-09-16: "no debe haber mínimo, mal"). It used to
    # drop every instance under 300 points as a mask sliver. A distant pipe with
    # 280 points is not a sliver, it is a pipe, and nothing downstream could
    # tell it had ever existed. 0 = keep everything.
    _min_pts = int(_dd.get("min_instance_points", 0))
    if not _merge_on:
        print("[SegPipeline]   Instance merging DISABLED "
              "(segmentation.merge_duplicates: false) — instances kept distinct")

    dedupe_decisions: list = []
    if _merge_on and len(instances) > 1 and _vox > 0:
        # per instance: its voxels (world-anchored, packed) and, per keyframe that
        # saw it, the voxels of the points born in that keyframe — the JUDGES of
        # point 104 are the keyframes that saw BOTH instances
        vox_all: List[np.ndarray] = []
        vox_by_kf: List[Dict[int, np.ndarray]] = []
        for inst in instances:
            idxs = np.asarray(inst["globalIndices"], dtype=np.int64)
            if len(idxs) == 0:
                vox_all.append(np.zeros(0, np.int64))
                vox_by_kf.append({})
                continue
            keys = _pack_voxels(np.floor(xyz_display[idxs] / _vox).astype(np.int64))
            vox_all.append(np.unique(keys))
            fr = frame_arr[idxs]
            per = {}
            order = np.argsort(fr, kind="stable")
            fs, starts = np.unique(fr[order], return_index=True)
            ends = np.r_[starts[1:], len(order)]
            for f_, s_, e_ in zip(fs.tolist(), starts, ends):
                per[int(f_)] = np.unique(keys[order[s_:e_]])
            vox_by_kf.append(per)
        absorbed = set()
        order = sorted(range(len(instances)), key=lambda k: (-len(vox_all[k]), k))
        for a_pos, i in enumerate(order):
            if i in absorbed or not len(vox_all[i]):
                continue
            for j in order[a_pos + 1:]:
                if j in absorbed or not len(vox_all[j]):
                    continue
                inter = int(np.isin(vox_all[j], vox_all[i], assume_unique=True).sum())
                if not inter:
                    continue
                share_i = inter / len(vox_all[i])
                share_j = inter / len(vox_all[j])
                # IDENTITY, not containment (USER 2026-09-14: "que ambas
                # instancias compartan mas del 80% de los puntos"). Dividing
                # only by the smaller asks "is B inside A", which every sign
                # resting on a wall answers yes — that is what collapsed 147
                # instances to 3. Asking BOTH to be mostly the intersection
                # separates "the same object under two names" from "one object
                # standing on another".
                # THE JUDGES (point 104, DECIDIDO): every keyframe that saw both —
                # its share of j's voxels inside i's space (and of i's inside j's,
                # the mutual test) — must exceed the user's bar significantly, with
                # >= min_judges judges, by >= error_factor x the judges' measured
                # spread; otherwise the two stay distinct and the margin is recorded
                common_kf = sorted(set(vox_by_kf[i]) & set(vox_by_kf[j]))
                judges = []
                for f_ in common_kf:
                    vi_f, vj_f = vox_by_kf[i][f_], vox_by_kf[j][f_]
                    sj = float(np.isin(vj_f, vox_all[i], assume_unique=True).sum()) / len(vj_f)
                    si = float(np.isin(vi_f, vox_all[j], assume_unique=True).sum()) / len(vi_f)
                    judges.append(min(si, sj) if _mutual else sj)
                d = _decide_above_bar(judges, _dup_thr, _robust_sigma(judges), factor=error_factor,
                                      confidence=confidence, min_judges=min_judges)
                dedupe_decisions.append({
                    "kind": "space_dedupe", "instance_ids": [
                        int(instances[i].get("instance_id", instances[i]["id"])),
                        int(instances[j].get("instance_id", instances[j]["id"]))],
                    "share": round(float(share_j), 3), "share_other": round(float(share_i), 3),
                    "n_judges": int(d["n_judges"]),
                    "median_judge_share": float(d["median_delta"] + _dup_thr),
                    "ci_margin": d["ci_margin"], "judges_margin": d["judges_margin"],
                    "error_margin": d["error_margin"], "merged": bool(d["improves"]),
                    "reason": d["reason"]})
                if d["improves"]:
                    # j is the same physical object as i → absorb
                    merged_idx = sorted(set(instances[i]["globalIndices"])
                                        | set(instances[j]["globalIndices"]))
                    instances[i]["globalIndices"] = merged_idx
                    instances[i]["total_points"] = len(merged_idx)
                    vox_all[i] = np.union1d(vox_all[i], vox_all[j])
                    for f_, v_ in vox_by_kf[j].items():
                        vox_by_kf[i][f_] = (np.union1d(vox_by_kf[i][f_], v_)
                                            if f_ in vox_by_kf[i] else v_)
                    absorbed.add(j)
                    absorbed_into[int(instances[j].get("instance_id", instances[j]["id"]))] = {
                        "into": int(instances[i].get("instance_id", instances[i]["id"])),
                        "into_label": instances[i].get("label"),
                        "reason": "space_dedupe",
                        "share": round(float(share_j), 3),
                        "share_other": round(float(share_i), 3),
                        "n_judges": int(d["n_judges"]),
                        "error_margin": d["error_margin"],
                    }
                    print(f"[SegPipeline]   🔗 Space-dedupe: '{instances[j]['label']}' "
                          f"#{instances[j]['id']} is the same object as "
                          f"'{instances[i]['label']}' #{instances[i]['id']} "
                          f"(they share {share_j:.0%} / {share_i:.0%} of their "
                          f"space; {d['n_judges']} keyframes judge, {d['reason']}) — merged")
                else:
                    print(f"[SegPipeline]   ↔ Space-dedupe: '{instances[j]['label']}' "
                          f"#{instances[j]['id']} vs '{instances[i]['label']}' #{instances[i]['id']} "
                          f"share {share_j:.0%} / {share_i:.0%} — kept distinct ({d['reason']})")
        if absorbed:
            pre = len(instances)
            instances = [inst for k, inst in enumerate(instances) if k not in absorbed]
            for inst in instances:
                m = np.array(inst["globalIndices"], dtype=np.int64)
                if len(m) >= 4:
                    inst["obb"] = _compute_obb(xyz_display[m])
            print(f"[SegPipeline]   Space-dedupe: {pre} → {len(instances)} instances")
    decisions["space_dedupe"] = dedupe_decisions

    # ── Same-label fragment consolidation ──────────────────────────────
    # SAM3 returns one mask per visually separable region, so one physical
    # surface arrives as many instances of ONE label (pccr 2026-09-14: 48
    # 'white_tiled_floor' for a single floor). Geometry decides: same label +
    # contiguous points = one object. Runs BEFORE the attach so the growth
    # starts from consolidated surfaces instead of 48 competing fragments.
    _frag_on = bool(_dd.get("merge_label_fragments", True))
    _frag_gap = float(_dd.get("fragment_gap_m", 0.10))
    frag_decisions: list = []
    if _frag_on and len(instances) > 1:
        pre = len(instances)
        n_abs = _merge_label_fragments(instances, xyz_display, gap_m=_frag_gap,
                                       record=absorbed_into, min_adjacent_pairs=min_judges,
                                       decisions=frag_decisions)
        if n_abs:
            for inst in instances:
                m = np.asarray(inst["globalIndices"], dtype=np.int64)
                if len(m) >= 4:
                    inst["obb"] = _compute_obb(xyz_display[m])
            print(f"[SegPipeline]   🧩 Fragment consolidation: {pre} → "
                  f"{len(instances)} instances (same label, contiguous "
                  f"within {_frag_gap*100:.0f} cm through >= {min_judges} adjacent voxel pairs)")
    decisions["fragments"] = frag_decisions

    # ── One instance, several objects seen together → split (USER 2026-09-30) ──
    split_decisions: list = []
    if bool(_dd.get("split_covisible", True)) and instances:
        _mp = int(cfg["correction"]["visit_drift"]["min_points"])
        _cc = None
        # camera centres, display frame — the poses of the live epoch keyed by
        # camera_frames.txt; a session without them splits nothing, declared
        _poses_p, _frames_p = output_dir / "camera_poses.txt", output_dir / "camera_frames.txt"
        if _poses_p.exists() and _frames_p.exists():
            _P = np.loadtxt(_poses_p).reshape(-1, 4, 4)
            _F = [int(float(x)) for x in _frames_p.read_text().split()]
            if len(_F) != len(_P):
                raise RuntimeError(f"camera_frames.txt lists {len(_F)} keyframes and camera_poses.txt "
                                   f"holds {len(_P)} poses — they are not one list")
            _C = s * (_P[:, :3, 3] @ R.T) + t
            _cc = {f: _C[k] for k, f in enumerate(_F)}
        else:
            print("[SegPipeline]   ✂ co-visible split: no camera poses / keyframe list — nothing split")
        # the raw store's highest id: the base the children's ids count from (point 105)
        _id_base = max([int(i.get("id", 0)) for i in (metadata.get("instances") or [])]
                       + [int(metadata.get("id_high_water") or 0)])
        n_split = _split_covisible_components(instances, xyz_display, frame_arr, gap_m=_frag_gap,
                                              min_points=_mp, covis_share=_dup_thr, cam_centre=_cc,
                                              min_walk_m=float(cfg["correction"]["visit_drift"]["min_walk_m"]),
                                              id_base=_id_base,
                                              decision=(error_factor, confidence, min_judges),
                                              record=split_decisions)
        if n_split:
            for inst in instances:
                m = np.asarray(inst["globalIndices"], dtype=np.int64)
                if len(m) >= 4:
                    inst["obb"] = _compute_obb(xyz_display[m])
            print(f"[SegPipeline]   ✂ Co-visible split: +{n_split} instance(s)")
    decisions["covisible_split"] = split_decisions

    # ── Geometric completion — "pegar los puntos al lugar correcto" (USER
    # 2026-08-29): SAM3 runs on the KEYFRAMES and the mask→cloud step only
    # labels a point whose own origin frame carries a mask, so coverage is
    # capped at keyframes/frames however good the masks are (pccr 2026-09-14:
    # 5.35% over 216 of 3,047 frames). A point within attach_dist_m of
    # labelled points adopts the instance most of its nearest labelled
    # neighbours belong to, over several rounds. Purely geometric and
    # conservative: points nobody reaches stay unsegmented (never invent).
    _att_on = bool(_dd.get("attach_unsegmented", True))
    _att_d = float(_dd.get("attach_dist_m", 0.03))
    _att_rounds = int(_dd.get("attach_rounds", 4))
    _att_votes = int(_dd.get("attach_votes", 3))
    if _att_on and instances:
        n_att, grown = _attach_unsegmented(instances, xyz_display,
                                           attach_dist_m=_att_d,
                                           rounds=_att_rounds, votes=_att_votes)
        for k in grown:   # OBBs must include the attached points
            m = np.asarray(instances[k]["globalIndices"], dtype=np.int64)
            if len(m) >= 4:
                instances[k]["obb"] = _compute_obb(xyz_display[m])
        if n_att:
            print(f"[SegPipeline]   📎 attach: {n_att:,} unsegmented points "
                  f"glued to their surfaces ({_att_rounds} round(s) of "
                  f"≤{_att_d*100:.0f} cm, {_att_votes}-neighbour vote)")

    # ── EXCLUSIVITY INVARIANT (USER 2026-08-31): every point belongs to ONE
    # instance or none — never two. Masks can claim the same points for
    # different segments; merges/attach/incremental unions could double-own.
    # Conflicts resolve to the SMALLEST instance (a point on a small object
    # belongs to the object, not the big surface behind it) and are reported.
    _n_dup = _enforce_exclusive_ownership(instances, n_pts)
    if _n_dup:
        print(f"[SegPipeline]   ⚠ exclusivity enforced: {_n_dup:,} "
              f"double-owned point(s) resolved")

    # ── Minimum object size — judged on the FINAL point count ────────────
    # It used to run before the attach, when it was a sliver filter reading a
    # mask's raw match. As an OBJECT filter (USER 2026-09-19: an object under
    # `min_instance_points` "que no quede segmentado") it has to judge what the
    # object ends up being: the attach glues on the points the keyframe masks
    # never reached, and exclusivity can take some back. A piece that grows
    # into a real surface is a real surface; one that does not is a sliver
    # either way. The USER's 1000 stays a strict count (point 116, DECIDIDO);
    # every object records its margin to the bar.
    if _min_pts > 0:
        _sz = {id(i): int(len(i.get("globalIndices") or ())) for i in instances}
        tiny = [inst for inst in instances if _sz[id(inst)] < _min_pts]
        if tiny:
            _tiny_desc = ", ".join("{}#{}({})".format(t["label"], t["id"], _sz[id(t)])
                                   for t in tiny[:10])
            print(f"[SegPipeline]   Dropped {len(tiny)} tiny instance(s) "
                  f"(<{_min_pts} pts): {_tiny_desc}{'...' if len(tiny) > 10 else ''}")
            for t in tiny:
                absorbed_into[int(t.get("instance_id", t["id"]))] = {
                    "into": None, "into_label": None, "reason": "too_small",
                    "label": t.get("label"),
                    "points": _sz[id(t)], "min_points": int(_min_pts),
                    "margin": int(_sz[id(t)] - _min_pts)}
            instances = [inst for inst in instances if _sz[id(inst)] >= _min_pts]
        for inst in instances:
            inst["min_points_margin"] = int(len(inst.get("globalIndices") or ()) - _min_pts)

    # ── Canonical instance store (scene_r.db) — THE single source of objects
    # for spatial Q&A (phase5), classification (phase2), findings (phase3) and
    # reports (phase6). Rebuilt from scratch on every segmentation, straight
    # from the CLEAN instances: points/OBBs in the DISPLAY frame (the exact
    # geometry the viewer renders and the user measures against).
    _write_instance_store(output_dir, instances, xyz_display)

    if mask_filter is not None:
        # The MARKS first, in their own block. They are the first moment of the
        # geometric cleanup cycle and the second moment cannot run without them.
        np.save(output_dir / "out_of_place.npy", out_of_place)
        write_npz_canonical(output_dir / OUT_OF_PLACE_VIEWS_NAME,
                            {"judged_views": judged_views, "off_views": off_views})
        print(f"[SegPipeline]    {int(out_of_place.sum()):,} point(s) marked out of "
              f"place (off their own mask in a view that sees them) — "
              f"kept, for the correction to move")
        rep = mask_filter.report()
        # every derived artifact carries the reconstruction it was measured on (point 118:
        # the identity of the reconstruction, not the epoch counter)
        from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
        rep[RECONSTRUCTION_ID_KEY] = reconstruction_id_or_none(output_dir)
        rep["points_out_of_place"] = int(out_of_place.sum())
        rep["points_out_of_place_by_one_view"] = int((out_of_place & (off_views == 1)).sum())
        rep["views_file"] = OUT_OF_PLACE_VIEWS_NAME
        atomic_write_json(output_dir / "mask_audit.json", rep, indent=1)
        byv = rep.get("instances_by_verdict") or {}
        frac = rep.get("on_mask_fraction")
        print(f"[SegPipeline] 🔍 mask audit: "
              + (f"{frac*100:.1f}% of the observations land on their mask; "
                 if frac is not None else "")
              + ", ".join(f"{len(v)} {k}" for k, v in sorted(byv.items()))
              + f" (of {rep['instances']} instances) — nothing removed")
        for k in ("drift_duplicate", "unsupported"):
            if byv.get(k):
                print(f"[SegPipeline]    {k}: instances {byv[k]}")

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
        # The fate of every mask segmentation.json lists. A mask is either one
        # of the instances above, or it is named here with where it went — it
        # is never simply absent, because "absent" is what the list endpoint
        # used to render as a zero-point row (USER 2026-09-17).
        "absorbed": {str(k): v for k, v in _mask_fates(metadata, instances,
                                                       absorbed_into).items()},
        "decisions": decisions,
    }
    _fates = result["absorbed"]
    if _fates:
        from collections import Counter as _C
        _why = _C(v["reason"] for v in _fates.values())
        print(f"[SegPipeline]   {len(_fates)} mask(s) of {len(metadata.get('instances') or [])} "
              f"are not separate objects: {dict(_why)} — recorded, not listed")

    print(f"[SegPipeline] ✅ {len(instances)} instances matched against {cloud_label}, "
          f"{total_segmented:,}/{n_pts:,} points ({coverage*100:.1f}% coverage)")

    # ── The per-point siblings of the cloud: the class byte + the 16-bit instance id
    # (ONE writer, segmentation.republish.write_classification — point 107)
    from repro import sha256_file
    from segmentation.republish import CLASSIFICATION, INSTANCE_IDS, write_classification
    write_classification(output_dir, instances, n_pts)

    # ── THE STAMP of this result (point 123): every input, the code, the configuration
    result["stamp"] = projection_stamp(output_dir, ply_path, cfg, cfg_source)

    # ── The octree: ONLY from the cloud the masks were projected on (point 120 — the
    # corrected_cloud.ply route is gone), rebuilt when its own stamp changed or it is
    # missing (point 123); a build that fails FAILS the projection (point 122)
    from potree_converter import POTREE_BIN, convert_ply_to_potree
    potree_dir = output_dir / "potree"
    oct_stamp = {"cloud_sha256": result["stamp"]["inputs"]["cloud"],
                 "classification_sha256": sha256_file(output_dir / CLASSIFICATION),
                 "instance_ids_sha256": sha256_file(output_dir / INSTANCE_IDS),
                 "converter_sha256": sha256_file(POTREE_BIN) if POTREE_BIN.exists() else None}
    stamp_p = output_dir / POTREE_STAMP_NAME
    saved_oct = None
    if stamp_p.exists():
        try:
            saved_oct = json.loads(stamp_p.read_text())
        except ValueError:
            saved_oct = None
    potree_missing = not (potree_dir / "metadata.json").exists()
    if potree_missing or saved_oct != oct_stamp:
        session_dir = output_dir.parent
        if stamp_p.exists():
            stamp_p.unlink()
        success = convert_ply_to_potree(session_dir, force=True, ply_override=ply_path)
        if not success:
            raise RuntimeError(f"the Potree octree could not be built from {ply_path.name} — "
                               f"the projection is not published without its octree")
        atomic_write_json(stamp_p, oct_stamp, indent=1, sort_keys=True)
        print(f"[SegPipeline] 🌲 Potree octree rebuilt from {ply_path.name}")
        result["reload_potree"] = True
    else:
        print("[SegPipeline] Potree untouched — same cloud, classification and converter "
              "(identical stamp, nothing to rebuild)")

    return result


def map_segmentation_to_cloud(output_dir) -> dict:
    """Deferred mask→cloud mapping (anchored pipeline order): run the FULL
    matching + per-instance cleaning ONCE against the (corrected, merged)
    cleaned cloud, then refresh the Phase R store's canonical OBBs so the
    assistant's boxes coincide exactly with the viewer's. Called by the
    cloudcompy stage when segmentation.json exists.

    ONCE means once: a `segmentation_result.json` whose STAMP (point 123 — the
    sha256 of every input, the code and the configuration) equals the one this
    state would produce is reused; any difference re-projects. An mtime never
    decides."""
    output_dir = Path(output_dir)
    if not (output_dir / "segmentation.json").exists():
        return {"error": "no segmentation.json", "instances": []}
    if not (output_dir / "cleaned_cloud.ply").exists():
        return {"error": "no cleaned_cloud.ply", "instances": []}
    _stale, _why = segmentation_result_is_stale(output_dir)
    if not _stale:
        _cached = json.loads((output_dir / "segmentation_result.json").read_text())
        if _cached.get("instances"):
            print(f"[SegPipeline] ♻ segmentation_result.json reused "
                  f"({len(_cached['instances'])} instance(s), {_why}) — the "
                  f"mask→cloud matching already ran on this state")
            return _cached
    else:
        print(f"[SegPipeline] projecting: {_why}")
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


def _merge_absorbed(prev, new, instances) -> dict:
    """The absorbed record across an incremental merge, keyed by instance_id.

    Newer entries win; a mask that is now one of ``instances`` is not absorbed
    by anything and is removed from the record.
    """
    out = {}
    for src in (prev or {}, new or {}):
        for k, v in src.items():
            try:
                out[str(int(k))] = dict(v)
            except (TypeError, ValueError):
                continue
    for inst in instances:
        iid = inst.get("instance_id", inst.get("id"))
        if iid is not None:
            out.pop(str(int(iid)), None)
    return out


def _match_and_save_result(output_dir, ply_path=None):
    """
    Run mask→cloud matching and save to segmentation_result.json.

    Takes the session's matching lock (`segmentation.match_lock`) — an OS
    lock, honoured ACROSS PROCESSES. It used to take none, so the pipeline's
    worker and the API process matched the same cloud at the same time
    (pccr 2026-09-21: four passes, four octrees, and the session's object
    count decided by arrival order).
    """
    from segmentation.match_lock import matching_lock
    with matching_lock(output_dir, log=lambda m: print(f"[SegPipeline]{m}")):
        return _match_and_save_result_locked(output_dir, ply_path)


def _match_and_save_result_locked(output_dir, ply_path=None):
    """The projection, FULL and PURE (docs/plan_determinismo.md point 99): it
    starts from an empty result, never merges a previous `segmentation_result.json`
    (no carried instances, no carried captions — the per-object descriptions
    are made again by the stage that makes them), and on an error or an empty
    projection it FAILS, deleting the previous result: a result on disk is
    always the product of the inputs beside it. The fusion map (point 100) is
    written beside the raw store before the result."""
    output_dir = Path(output_dir)
    result_path = output_dir / "segmentation_result.json"

    def _drop_previous(why: str):
        if result_path.exists():
            result_path.unlink()
            print(f"[SegPipeline] previous segmentation_result.json deleted — {why}")

    try:
        result = _match_masks_to_cloud(output_dir, ply_path)
        if "error" in result or not result.get("instances"):
            raise RuntimeError(f"the projection produced no instance for {output_dir} — "
                               f"{result.get('error') or 'no mask landed on the cloud'}")

        merged = result["instances"]
        total_pts = int(result.get("total_points") or 0)
        total_segmented = sum(inst.get("total_points", 0) for inst in merged)
        coverage = round(total_segmented / max(1, total_pts), 4)
        merged_result = {
            "type": "segmentation",
            "version": "3.0",
            "prompt": result.get("prompt", ""),
            "prompts": result.get("prompts", []),
            "cloud_source": result.get("cloud_source", ""),
            "total_points": total_pts,
            "segmented_points": total_segmented,
            "coverage": coverage,
            "instances": merged,
            "resolution": result.get("resolution", {}),
            # The fate of every mask that is NOT one of the instances above.
            # This dict is rebuilt key by key, so anything the matcher returns
            # has to be carried here explicitly or it never reaches disk — the
            # record printed to the log but vanished from the file on pccr
            # 2026-09-17.
            "absorbed": _merge_absorbed(None, result.get("absorbed"), merged),
            "decisions": result.get("decisions", {}),
            "stamp": result.get("stamp"),
        }
        # ── THE FUSION, beside the raw store (USER 2026-09-20 / point 100): the
        # matcher's verdict as fusion_map.json tied to the raw store's sha256; the
        # parent is never rewritten. Fatal like every step (point 122).
        from segmentation import fuse_parent
        fuse_parent.apply_fusion(output_dir, merged_result,
                                 log=lambda m: print(f"[SegPipeline]{m}"))

        atomic_write_json(result_path, merged_result)
        # a transient for the caller (the viewer reloads the octree once), never part of
        # the file: the same projection writes the same bytes whether the octree was
        # rebuilt or reused
        if result.get("reload_potree"):
            merged_result["reload_potree"] = True
    except BaseException as e:
        # a result on disk is always the product of the inputs beside it: none survives
        # a projection that did not finish (point 99)
        _drop_previous(f"the projection failed ({type(e).__name__}: {e})")
        raise
    print(f"[SegPipeline] 💾 Saved segmentation_result.json "
          f"({len(merged)} instances, {coverage*100:.1f}% coverage)")
    return merged_result


# Per-session lock to prevent parallel matching runs on the same output dir
import threading
# The per-session matching lock lives in `segmentation.match_lock`: an OS lock
# on the session directory, honoured ACROSS PROCESSES. What used to be here was
# a `threading.Lock`, which coordinates threads inside one process and is blind
# to the pipeline's `multiprocessing.spawn` worker — so on 2026-09-21 the
# viewer and the pipeline matched the same cloud at the same time. A dead guard
# that looks like a protection is worse than none.


def apply_segmentation_to_cloud(output_dir, ply_path=None) -> dict:
    """The segmentation the viewer receives — every instance carries its `class_byte`.

    pccr 2026-10-01 (USER: "cuando prendo o apago una segmentación además prende y apaga otros
    objetos"): the viewer keys its visibility texture by `inst.class_byte ?? instance_id`
    (ui Viewport.tsx). Only the /segments listing attached `class_byte`; the WebSocket payload built
    here did not, so with the COMPACT encoding (instance ids above 255 → bytes 1..N, class_map.json)
    every toggle switched the object whose byte equals the toggled instance id. The byte now rides on
    every path out of here, from the same map the octree was written with."""
    result = _apply_segmentation_to_cloud_impl(output_dir, ply_path)
    return with_class_bytes(Path(output_dir), result)


def with_class_bytes(output_dir: Path, result: dict) -> dict:
    """Attach each instance's class byte (`segmentation.republish.class_of`; no map = identity)."""
    from segmentation.republish import class_of
    code = class_of(output_dir)
    for inst in (result or {}).get("instances") or []:
        iid = inst.get("instance_id", inst.get("id"))
        if iid is not None:
            inst["class_byte"] = int(code.get(int(iid), int(iid))) if code else int(iid)
    return result


def _apply_segmentation_to_cloud_impl(output_dir, ply_path=None) -> dict:
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
    
    # ── Slow path ─────────────────────────────────────────────────────────
    # The lock is an OS lock on the session directory, so it is honoured by
    # the pipeline's worker PROCESS as well (`segmentation.match_lock`). The
    # old `threading.Lock` only saw other threads, so on 2026-09-21 the viewer
    # opening a session launched a full second matching alongside the
    # pipeline's — four passes over 22.7 M points, four octrees, and the
    # session's object count decided by arrival order.
    #
    # A caller that finds one in flight does NOT start its own: it waits and
    # reads what that pass wrote. That is the whole point — the redundant work
    # must not exist, not merely be serialised.
    from segmentation.match_lock import matching_lock
    with matching_lock(output_dir,
                       log=lambda m: print(f"[SegPipeline]{m}")) as waited:
        if waited and result_path.exists():
            try:
                with open(result_path) as f:
                    result = json.load(f)
                print(f"[SegPipeline] ⚡ Loaded the result the other matching "
                      f"just wrote ({len(result.get('instances') or [])} "
                      f"instances) — no second pass")
                return result
            except Exception:  # noqa: BLE001 — fall through and match
                pass
        return _apply_segmentation_slow(output_dir, ply_path, result_path)


def _apply_segmentation_slow(output_dir, ply_path, result_path):
    """The matching itself (the viewer's cold load of a session with masks and no
    result). The caller already holds the session lock. The same pure projection
    as the pipeline's, cached only when it produced objects (point 99: an empty
    projection is an error, never a cached result)."""
    print(f"[SegPipeline] No cached result, running full mask matching (will cache for next time)...")
    return _match_and_save_result_locked(Path(output_dir), ply_path)
