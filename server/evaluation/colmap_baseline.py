"""Phase 3 (h) — classical photogrammetry as the baseline: COLMAP 4.x (the GLOMAP global mapper
is `colmap global_mapper`) in its own conda env, never the pycolmap the chain uses.

One camera per session, features SIFT (h1 / h2) or the learned ALIKED + LightGlue (h3), sequential
matching with vocabulary-tree loop detection, two mappings over ONE database — incremental (h1)
and global (h2) — every disconnected model kept, then a dense cloud per model (image_undistorter →
patch_match_stereo → stereo_fusion). Same input frames as the reference (the chain's keyframes),
no learned model anywhere in h1 / h2, scale only from the gauge distances (the battery's phase 1).
Every command line goes into the workspace's manifest. GPU steps need the card for themselves
(the caller checks); COLMAP downloads its vocabulary trees / ONNX models into ~/.cache/colmap.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evaluation.colmap_io import list_models, read_model

FEATURE_TYPES = ("SIFT", "ALIKED_N16ROT", "ALIKED_N32")
MATCHER_TYPES = ("SIFT_BRUTEFORCE", "SIFT_LIGHTGLUE", "ALIKED_BRUTEFORCE", "ALIKED_LIGHTGLUE")
MAPPERS = ("incremental", "global")
MARKER = "matching.done.json"


class ColmapError(RuntimeError):
    pass


def _bool(v: Any) -> str:
    return "1" if bool(v) else "0"


def colmap_version(binary: str) -> str:
    try:
        r = subprocess.run([binary, "-h"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ColmapError(f"{binary}: cannot run ({e})") from e
    for line in (r.stdout + r.stderr).splitlines():
        if "COLMAP" in line:
            return line.strip().split("] ")[-1]
    return "unknown"


def _run(cmd: List[str], log_path: Path, log: Callable, timeout_s: float, env: Optional[dict] = None
         ) -> Dict[str, Any]:
    t0 = time.time()
    log(f"[colmap] $ {' '.join(cmd)}")
    with open(log_path, "ab") as lf:
        lf.write(("\n$ " + " ".join(cmd) + "\n").encode())
        lf.flush()
        try:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired as e:
            raise ColmapError(f"{cmd[1]} exceeded {timeout_s:g} s") from e
    rec = {"argv": cmd, "returncode": r.returncode, "seconds": round(time.time() - t0, 2)}
    if r.returncode != 0:
        raise ColmapError(f"{cmd[1]} failed with code {r.returncode} (log {log_path})")
    return rec


def workspace_key(features: str, matcher: str) -> str:
    return f"{features.lower()}__{matcher.lower()}"


def prepare_images(frames_dir: Path, keyframe_files: Sequence[str], ws: Path) -> Path:
    """The keyframe files, and only them, as the image set (hard links when possible)."""
    images = ws / "images"
    images.mkdir(parents=True, exist_ok=True)
    for name in keyframe_files:
        src = Path(frames_dir) / name
        if not src.exists():
            raise ColmapError(f"keyframe file {src} does not exist")
        dst = images / name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return images


def ensure_features_and_matches(ws: Path, features: str, matcher: str, ccfg: Dict[str, Any], log: Callable,
                                camera_params: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Feature extraction + sequential matching with loop detection into ``ws/database.db``
    (once per workspace: a marker records the commands; a later variant reuses them).
    ``camera_params``: the single camera's prior in native pixels, in the order of
    ``ccfg["camera_model"]`` (OPENCV: fx fy cx cy k1 k2 p1 p2) — COLMAP's ``ImageReader.camera_params``;
    None = COLMAP's own default (a focal of 1.2 × the long side, no prior). A workspace made with
    another prior is rebuilt: the two-view geometries depend on the camera."""
    if features not in FEATURE_TYPES:
        raise ColmapError(f"features {features!r} not in {FEATURE_TYPES}")
    if matcher not in MATCHER_TYPES:
        raise ColmapError(f"matcher {matcher!r} not in {MATCHER_TYPES}")
    want = [float(p) for p in camera_params] if camera_params is not None else None
    marker = ws / MARKER
    if marker.exists():
        rec = json.loads(marker.read_text())
        if rec.get("camera_params") == want:
            return rec
        log(f"[colmap] {ws}: the features and matches on disk were made with the camera prior "
            f"{rec.get('camera_params')}, {want} is asked — rebuilt")
        marker.unlink()
        for d in ("sparse_incremental", "sparse_global"):
            shutil.rmtree(ws / d, ignore_errors=True)
    binary = ccfg["binary"]
    db = ws / "database.db"
    if db.exists():
        db.unlink()
    log_path = ws / "colmap_features.log"
    cmds = []
    fx = [binary, "feature_extractor", "--database_path", str(db), "--image_path", str(ws / "images"),
          "--ImageReader.single_camera", "1", "--ImageReader.camera_model", str(ccfg["camera_model"]),
          "--FeatureExtraction.type", features, "--FeatureExtraction.use_gpu", _bool(ccfg["use_gpu"])]
    if features == "SIFT":
        fx += ["--SiftExtraction.max_num_features", str(int(ccfg["sift"]["max_num_features"]))]
    else:
        fx += ["--AlikedExtraction.max_num_features", str(int(ccfg["aliked"]["max_num_features"]))]
    if want is not None:
        fx += ["--ImageReader.camera_params", ",".join(f"{p:.10g}" for p in want)]
    cmds.append(_run(fx, log_path, log, float(ccfg["timeout_s"])))
    sq = ccfg["sequential"]
    tree_key = "sift" if features == "SIFT" else ("aliked_n16rot" if features == "ALIKED_N16ROT" else "aliked_n32")
    mx = [binary, "sequential_matcher", "--database_path", str(db),
          "--FeatureMatching.type", matcher, "--FeatureMatching.use_gpu", _bool(ccfg["use_gpu"]),
          "--SequentialMatching.overlap", str(int(sq["overlap"])),
          "--SequentialMatching.quadratic_overlap", _bool(sq["quadratic_overlap"]),
          "--SequentialMatching.loop_detection", _bool(sq["loop_detection"]),
          "--SequentialMatching.loop_detection_period", str(int(sq["loop_detection_period"])),
          "--SequentialMatching.loop_detection_num_images", str(int(sq["loop_detection_num_images"]))]
    if sq["loop_detection"]:
        mx += ["--SequentialMatching.vocab_tree_path", str(ccfg["vocab_tree"][tree_key])]
    # the vocabulary-tree retrieval (FAISS, OpenMP) dies by SIGSEGV now and then on this pod (pccr_audit3
    # 2026-10-31 and a CPU trial the same day; the next attempt ran through): ONE retry on a signal death,
    # recorded — a second death is the failure it is
    try:
        cmds.append(_run(mx, log_path, log, float(ccfg["timeout_s"])))
    except ColmapError as e:
        if "code -" not in str(e):
            raise
        log(f"[colmap] sequential_matcher died by a signal ({e}) — retried once (FAISS vocab-tree retrieval)")
        rec_retry = _run(mx, log_path, log, float(ccfg["timeout_s"]))
        rec_retry["attempt"] = 2
        rec_retry["first_attempt"] = str(e)
        cmds.append(rec_retry)
    rec = {"features": features, "matcher": matcher, "camera_model": ccfg["camera_model"], "camera_params": want,
           "commands": cmds, "colmap": colmap_version(binary), "n_images": len(list((ws / "images").iterdir()))}
    marker.write_text(json.dumps(rec, indent=1))
    return rec


def map_sparse(ws: Path, mapper: str, ccfg: Dict[str, Any], log: Callable) -> Dict[str, Any]:
    """h1 = `mapper` (incremental), h2 = `global_mapper` (GLOMAP): every model kept."""
    if mapper not in MAPPERS:
        raise ColmapError(f"mapper {mapper!r} not in {MAPPERS}")
    binary = ccfg["binary"]
    out = ws / f"sparse_{mapper}"
    marker = out / "mapping.done.json"
    if marker.exists():
        return json.loads(marker.read_text())
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    log_path = ws / f"colmap_{mapper}.log"
    cmd = [binary, "mapper" if mapper == "incremental" else "global_mapper", "--database_path", str(ws / "database.db"),
           "--image_path", str(ws / "images"), "--output_path", str(out)]
    extra = ccfg["mapper_options"] if mapper == "incremental" else ccfg["global_mapper_options"]
    for k, v in (extra or {}).items():
        cmd += [f"--{k}", str(v) if not isinstance(v, bool) else _bool(v)]
    rec_cmd = _run(cmd, log_path, log, float(ccfg["timeout_s"]))
    models = []
    for d in list_models(out):
        m = read_model(d)
        models.append({"dir": str(d.relative_to(ws)), "n_images": m.n_images, "n_points3d": m.n_points3d})
    rec = {"mapper": mapper, "command": rec_cmd, "models": models}
    marker.write_text(json.dumps(rec, indent=1))
    log(f"[colmap] {mapper}: {len(models)} model(s) — " + ", ".join(f"{m['n_images']} images" for m in models))
    return rec


def dense_for_model(ws: Path, model_dir: Path, ccfg: Dict[str, Any], log: Callable) -> Dict[str, Any]:
    """image_undistorter → patch_match_stereo → stereo_fusion for one sparse model."""
    binary = ccfg["binary"]
    rel = model_dir.relative_to(ws)
    dense = ws / "dense" / rel
    marker = dense / "dense.done.json"
    if marker.exists():
        return json.loads(marker.read_text())
    shutil.rmtree(dense, ignore_errors=True)
    dense.mkdir(parents=True)
    log_path = ws / "colmap_dense.log"
    d = ccfg["dense"]
    cmds = [_run([binary, "image_undistorter", "--image_path", str(ws / "images"), "--input_path", str(model_dir),
                  "--output_path", str(dense), "--output_type", "COLMAP",
                  "--max_image_size", str(int(d["max_image_size"]))], log_path, log, float(ccfg["timeout_s"]))]
    cmds.append(_run([binary, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
                      "--PatchMatchStereo.geom_consistency", _bool(d["geom_consistency"]),
                      "--PatchMatchStereo.max_image_size", str(int(d["max_image_size"]))],
                     log_path, log, float(ccfg["timeout_s"])))
    fused = dense / "fused.ply"
    cmds.append(_run([binary, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
                      "--input_type", "geometric", "--output_path", str(fused),
                      "--StereoFusion.min_num_pixels", str(int(d["fusion_min_num_pixels"]))],
                     log_path, log, float(ccfg["timeout_s"])))
    rec = {"model": str(rel), "commands": cmds, "fused_ply": str(fused.relative_to(ws)),
           "fused_bytes": fused.stat().st_size if fused.exists() else 0}
    marker.write_text(json.dumps(rec, indent=1))
    return rec


def run_colmap_variant(frames_dir: Path, keyframe_files: Sequence[str], scene_ws: Path, features: str, matcher: str,
                       mapper: str, ccfg: Dict[str, Any], log: Callable) -> Dict[str, Any]:
    """The whole baseline for one variant; returns its manifest (commands, models, dense clouds).
    Workspaces are shared between variants with the same features + matcher (h1 / h2 share
    one database, as the plan asks)."""
    ws = scene_ws / workspace_key(features, matcher)
    ws.mkdir(parents=True, exist_ok=True)
    prepare_images(frames_dir, keyframe_files, ws)
    man: Dict[str, Any] = {"workspace": str(ws), "features": features, "matcher": matcher, "mapper": mapper,
                           "config": {k: v for k, v in ccfg.items()},
                           "matching": ensure_features_and_matches(ws, features, matcher, ccfg, log)}
    man["mapping"] = map_sparse(ws, mapper, ccfg, log)
    min_images = int(ccfg["min_model_images"])
    dense = []
    for m in man["mapping"]["models"]:
        if m["n_images"] < min_images:
            dense.append({"model": m["dir"], "skipped": f"{m['n_images']} images < min_model_images {min_images}"})
            continue
        dense.append(dense_for_model(ws, ws / m["dir"], ccfg, log))
    man["dense"] = dense
    man["sparse_dir"] = str((ws / f"sparse_{mapper}").relative_to(ws))
    return man
