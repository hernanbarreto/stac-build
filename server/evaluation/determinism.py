"""Phase 6 — repeatability and determinism.

``evaluation.deterministic: true``: the chain's own deterministic regime (server/repro.py: strict
torch algorithms, cuBLAS workspace, cudnn benchmark off, every seed a config key, fixed
processing order) is what every variant runs under; the battery RECORDS every seed, the
environment variables it sets and, BY NAME, what has no deterministic version (the COLMAP
baselines: SiftGPU / PatchMatch). The run manifest holds the hash of the input video, of every
pinned checkpoint, the model revisions, library versions, driver, CUDA, card, the complete
effective configuration per variant and the seeds. Two runs of the same scene and configuration
are compared point by point, paired by provenance (frame, pixel): numbers only.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from evaluation.config import EvaluationConfig, repo_root

MANIFEST_VERSION = 1


# ── seeds, environment ───────────────────────────────────────────────────

def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}." if prefix or True else k))
        return {k.rstrip("."): v for k, v in out.items()} if prefix == "" else out
    out[prefix.rstrip(".")] = d
    return out


def seed_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Every configuration key that seeds something (…seed) plus the VLM's temperature."""
    flat = flatten(config)
    keys = {k: v for k, v in flat.items() if k.split(".")[-1] == "seed" or k.endswith("_seed")}
    for extra in ("semantic.generation.temperature", "semantic.generation.seed"):
        if extra in flat:
            keys[extra] = flat[extra]
    return dict(sorted(keys.items()))


def deterministic_regime(config: Dict[str, Any]) -> Dict[str, Any]:
    """What the deterministic mode asserts and sets: the chain's env (repro.deterministic_env),
    its torch flags, the seeds as configured. The values are not changed here — they ARE the
    chain's; a seed missing from the configuration would fail its loader."""
    from repro import CUBLAS_WORKSPACE_VALUE, deterministic_env
    env = deterministic_env({})
    return {"env": env, "cublas_workspace": CUBLAS_WORKSPACE_VALUE,
            "torch": {"use_deterministic_algorithms": True, "warn_only": False, "cudnn_benchmark": False,
                      "cudnn_deterministic": True, "tf32": False,
                      "source": "repro.enable_deterministic_torch (every GPU step of the chain)"},
            "seeds": seed_keys(config),
            "vlm": {"temperature": flatten(config).get("semantic.generation.temperature"),
                    "seed": flatten(config).get("semantic.generation.seed"),
                    "engine": "one vLLM per VLM stage from the frozen config, no prefix cache, one sequence, "
                              "batch invariance (semantic.service.job_engine)"},
            "processing_order": "fixed: keyframes in frame order, stages in DEFAULT_STAGE_ORDER, "
                                "variants in the configured order (bases before continuations)"}


def declared_non_deterministic(variant_kinds: Dict[str, str]) -> List[Dict[str, str]]:
    """Operations without a deterministic version, by name — never hidden."""
    out = []
    for name, kind in variant_kinds.items():
        if kind == "colmap":
            out.append({"variant": name, "operation": "colmap feature_extractor (SiftGPU) / sequential_matcher (GPU) / "
                                                      "patch_match_stereo", "why": "COLMAP's GPU kernels and multi-threaded "
                                                      "RANSAC are not bit-stable run to run; --default_random_seed fixes "
                                                      "the CPU PRNG only"})
    out.append({"variant": "*", "operation": "nvidia-smi sampling of the cost monitor",
                "why": "timing and memory samples vary run to run by construction (reported, never compared)"})
    return out


# ── hashes ───────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def video_identity(session_dir: Path) -> Dict[str, Any]:
    """The input video: frames/manifest.json's record when present, else the file's sha256."""
    man = Path(session_dir) / "frames" / "manifest.json"
    if man.exists():
        try:
            doc = json.loads(man.read_text())
            v = doc.get("video") or {}
            return {"source": "frames/manifest.json", "name": v.get("name"), "sha256": v.get("sha256"),
                    "size": v.get("size"), "n_frames": doc.get("n_frames"), "frames_sha256": doc.get("frames_sha256")}
        except (OSError, ValueError):
            pass
    vids = sorted(Path(session_dir).glob("source_video.*"))
    if vids:
        return {"source": "sha256 of the file", "name": vids[0].name, "sha256": sha256_file(vids[0]),
                "size": vids[0].stat().st_size}
    return {"source": "none", "name": None, "sha256": None}


def checkpoints(config: Dict[str, Any]) -> Dict[str, Any]:
    """Every pinned checkpoint the chain declares (model, revision, sha256 of its files)."""
    out: Dict[str, Any] = {}
    flat = flatten(config)
    model_id = flat.get("reconstruction.precision.gauge.model_id")
    try:
        from da3_weights import identity
        out["da3"] = identity(str(model_id))
    except Exception as e:  # noqa: BLE001 — recorded, not hidden
        out["da3"] = {"model_id": model_id, "error": f"{type(e).__name__}: {e}"}
    try:
        from semantic.serve import PINNED_WEIGHTS
        out["vlm"] = {k: {"revision": v.get("revision"), "sha256": v.get("sha256")} for k, v in PINNED_WEIGHTS.items()}
    except Exception as e:  # noqa: BLE001
        out["vlm"] = {"error": f"{type(e).__name__}: {e}"}
    wd = flat.get("reconstruction.precision.mono_detail.weights_dir")
    pd: Dict[str, Any] = {"weights_dir": wd, "files": {}}
    if wd:
        d = Path(wd)
        d = d if d.is_absolute() else repo_root() / d
        for p in sorted(d.glob("*.pth")) if d.exists() else []:
            side = p.with_suffix(p.suffix + ".sha256")
            pd["files"][p.name] = {"sha256": side.read_text().split()[0] if side.exists() else None,
                                   "name_prefix": p.stem.split("-")[-1], "bytes": p.stat().st_size}
    out["pointdit"] = pd
    out["vggsfm_tracker"] = {"url": flat.get("reconstruction.precision.tracks.tracker_weights_url"),
                             "sha256": flat.get("reconstruction.precision.tracks.tracker_weights_sha256")}
    out["sam3"] = {"version": flat.get("models.segmentation.version"),
                   "checkpoint_path": flat.get("models.segmentation.checkpoint_path"),
                   "note": "null = the vendor's auto-download; the vendor checkout's git state is in environment.git"}
    return out


def environment(gpu: bool) -> Dict[str, Any]:
    try:
        from repro import environment_record
        return environment_record(gpu=gpu)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def config_sha(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def run_manifest(run_id: str, scenes: Sequence[Any], variant_specs: Dict[str, Any], cfg: EvaluationConfig,
                 base_config: Dict[str, Any], effective_configs: Dict[str, Dict[str, Any]], *, gpu: bool,
                 argv: Sequence[str] = (), status: str = "running", repeat: int = 1) -> Dict[str, Any]:
    """The manifest of a run. Everything that identifies the run is here; the clock only in
    ``created`` / ``status`` (``stable_part`` strips them for a comparison)."""
    return {
        "manifest_version": MANIFEST_VERSION, "run_id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status, "argv": list(argv), "repeat": int(repeat),
        "scenes": [s.name for s in scenes],
        "scene_inputs": {s.name: {"session_dir": str(s.session_dir), "video": video_identity(s.session_dir)}
                         for s in scenes},
        "variants": list(variant_specs),
        "variant_specs": {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in variant_specs.items()},
        "deterministic": bool(cfg.deterministic),
        "regime": deterministic_regime(base_config) if cfg.deterministic else None,
        "declared_non_deterministic": declared_non_deterministic(
            {k: (v.kind if hasattr(v, "kind") else v.get("kind")) for k, v in variant_specs.items()}),
        "checkpoints": checkpoints(base_config),
        "environment": environment(gpu),
        "evaluation_config": cfg.raw,
        "base_config_sha256": config_sha(base_config),
        "effective_config_sha256": {k: config_sha(v) for k, v in effective_configs.items()},
        "effective_configs": effective_configs,
    }


def stable_part(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in manifest.items() if k not in ("created", "status")}


# ── two runs compared ────────────────────────────────────────────────────

def _key(data) -> np.ndarray:
    fg = np.asarray(data["frame_global"], np.int64)
    r = np.asarray(data["pixel_row"], np.int64)
    c = np.asarray(data["pixel_col"], np.int64)
    return (fg << 32) | ((r & 0xFFFF) << 16) | (c & 0xFFFF)


def compare_products(a, b) -> Dict[str, Any]:
    """Two StacProducts: points paired by provenance (frame, pixel); max / median / P95 position
    difference of the paired points, counts of the unpaired, object counts."""
    from evaluation import geometry as G
    ka, kb = _key(a.data), _key(b.data)
    oa, ob = np.argsort(ka, kind="stable"), np.argsort(kb, kind="stable")
    sa, sb = ka[oa], kb[ob]
    # pairs: equal keys (a key repeated inside one cloud pairs its first occurrence)
    ua, ia = np.unique(sa, return_index=True)
    ub, ib = np.unique(sb, return_index=True)
    common, ca, cb = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    ra, rb = oa[ia[ca]], ob[ib[cb]]
    d = np.linalg.norm(a.xyz[ra] - b.xyz[rb], axis=1) if len(common) else np.zeros(0)
    out = {"n_points_a": int(len(a.xyz)), "n_points_b": int(len(b.xyz)), "n_paired": int(len(common)),
           "n_only_a": int(len(ua) - len(common)), "n_only_b": int(len(ub) - len(common)),
           "n_duplicate_keys_a": int(len(ka) - len(ua)), "n_duplicate_keys_b": int(len(kb) - len(ub)),
           "position_diff_m": {"max": float(d.max()) if d.size else None, "median": float(np.median(d)) if d.size else None,
                               "p95": G.percentile(d, 95), "n_nonzero": int((d > 0).sum())},
           "bit_identical_cloud": bool(d.size and d.max() == 0 and len(ua) == len(ub) == len(common)
                                       and len(a.xyz) == len(b.xyz)),
           "keyframes_a": len(a.frames), "keyframes_b": len(b.frames),
           "poses_max_diff": float(np.abs(a.c2w - b.c2w).max()) if a.c2w.shape == b.c2w.shape else None}
    for name, p in (("a", a), ("b", b)):
        ids = getattr(p, "instance_ids", None)
        out[f"objects_{name}"] = int(len(np.unique(np.asarray(ids)[np.asarray(ids) > 0]))) if ids is not None else None
    try:
        out["cloud_sha256_a"] = sha256_file(a.cloud_path); out["cloud_sha256_b"] = sha256_file(b.cloud_path)
    except Exception:  # noqa: BLE001
        pass
    return out


def _scalars(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_scalars(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_scalars(v, f"{prefix}{i}."))
    elif isinstance(d, bool):
        out[prefix.rstrip(".")] = float(d)
    elif isinstance(d, (int, float)) and d == d:
        out[prefix.rstrip(".")] = float(d)
    return out


def compare_records(ra: Dict[str, Any], rb: Dict[str, Any]) -> Dict[str, Any]:
    """Every scalar of two variant records' metrics (phases 1, 2, coverage): a − b."""
    diffs = {}
    for section in ("summary", "coverage", "confidence_distance", "growth_models"):
        sa, sb = _scalars(ra.get(section) or {}), _scalars(rb.get(section) or {})
        for k in sorted(set(sa) | set(sb)):
            if k.endswith("product.name"):
                continue
            if k in sa and k in sb:
                if sa[k] != sb[k]:
                    diffs[f"{section}.{k}"] = {"a": sa[k], "b": sb[k], "diff": sa[k] - sb[k]}
            else:
                diffs[f"{section}.{k}"] = {"a": sa.get(k), "b": sb.get(k), "diff": None}
    return {"n_differences": len(diffs), "differences": diffs}


def repeatability(scene_dir: Path, name: str, rep_name: str, ra: Dict[str, Any], rb: Dict[str, Any],
                  log=print) -> Dict[str, Any]:
    out: Dict[str, Any] = {"variant": name, "repeat": rep_name, "status_a": ra.get("status"), "status_b": rb.get("status")}
    if ra.get("status") == "finished" and rb.get("status") == "finished":
        out["metrics"] = compare_records(ra, rb)
        if ra.get("session_dir") and rb.get("session_dir"):
            try:
                from evaluation.product import StacProduct
                pa = StacProduct(Path(ra["session_dir"]), name, log=lambda *a: None)
                pb = StacProduct(Path(rb["session_dir"]), rep_name, log=lambda *a: None)
                out["points"] = compare_products(pa, pb)
            except Exception as e:  # noqa: BLE001
                out["points"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            out["points"] = {"note": "no provenance (dense product): points not paired; counts and metrics only",
                             "n_points_a": (ra.get("product") or {}).get("n_points"),
                             "n_points_b": (rb.get("product") or {}).get("n_points")}
    else:
        out["note"] = "both runs must finish to be compared"
    (scene_dir / f"repeatability_{name}.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return out
