"""Omega's STAMPED completion marker (docs/plan_determinismo.md point 23, audit M-05).

The fork's own "already complete" test (camera_poses.txt AND chunk clouds on disk) was defeated
by the pipeline's cleanup: after a completed precision run ``_cleanup_recon_temps`` deletes
``maplong_run/pcd`` and ``_discard_previous_epochs`` deletes ``output/chunk_*.ply``, so every
replace-OFF Reconstruir of a finished session re-inferred Omega under stale state — rewriting
camera_poses.txt in Omega's raw frame while the scale / orientation markers still said "applied".

Now ``output/maplong_run/omega_complete.json`` is written by the map worker when the fork exited
0 and its products were post-processed, and it carries the STAMP of that launch
(:func:`launch_stamp`: the keyframes' bytes and order, the walk the plan was measured on, the
revisit reference, the Omega weights, the sky model, the fork's code and the whole fork config)
plus the sha256 of the products that are never transformed afterwards. The map worker skips the
fork ONLY when the stamp computed now matches the saved one and every recorded product is on
disk unchanged (:func:`check`); a stale marker means the products are of another launch — they
are wiped and the fork runs again. No cleanup deletes this file: it lives in maplong_run, which
only a NEW chunk plan wipes (then the products go with it, which is right).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MARKER_NAME = "omega_complete.json"
MARKER_VERSION = 1

# the products the marker vouches for: sha256 for the ones no later stage rewrites, existence
# for camera_poses.txt (scale_align and orient transform it right after, by design)
PRODUCTS_SHA256 = ("maplong_run/chunk_sim3.json", "maplong_run/frame_list.json",
                   "camera_frames.txt", "frame_list.json")
PRODUCTS_EXIST = ("maplong_run/camera_poses.txt", "camera_poses.txt")
# the fork config keys that are not part of what the fork computes (a log-only field of the
# resolution report) — everything else of the config enters the stamp
_VOLATILE_REPORT_KEYS = ("persisted",)


class OmegaCompleteError(RuntimeError):
    """The completion marker cannot be computed (an input the stamp needs is missing)."""


def marker_path(output_dir: Path) -> Path:
    return Path(output_dir) / "maplong_run" / MARKER_NAME


def fork_code_files() -> List[Path]:
    """The code whose bits make the fork's products — the SAME set the fork stamps itself with
    (vggt_long._stac_fork_code_files): the fork's STAC modules, the Omega package it wraps, the
    server modules it calls by name. A missing file FAILS (repro.stamp): nothing is stamped as
    present that is not."""
    import repro
    fork = repro.FORK_ROOT
    files = [fork / "vggt_long.py"]
    files += sorted((fork / "loop_utils").glob("*.py"))
    files += [fork / "base_models" / "base_model.py", fork / "base_models" / "vggtomega_adapter.py",
              fork / "LoopModels" / "LoopModel.py", fork / "LoopModels" / "calibration.py"]
    omega = fork.parent / "vggt-omega" / "vggt_omega"
    files += sorted(p for p in omega.rglob("*.py") if "__pycache__" not in p.parts)
    server = repro.REPO_ROOT / "server"
    files += [server / "repro.py", server / "reconstruction" / "loops" / "spatial_gate.py",
              server / "reconstruction" / "da3_anchor.py", server / "extract_da3_depth.py",
              server / "da3_weights.py"]
    return files


def _config_for_stamp(fork_config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(fork_config, default=str))
    rep = (cfg.get("Model") or {}).get("omega_resolution_report")
    if isinstance(rep, dict):
        for k in _VOLATILE_REPORT_KEYS:
            rep.pop(k, None)
    return cfg


def launch_stamp(session_dir: Path, frames_dir: Path, keyframes: Sequence[str],
                 fork_config: Mapping[str, Any]) -> Dict[str, Any]:
    """repro.stamp of a fork launch: INPUTS — every keyframe's bytes (keyed by name, so a copied
    session keeps its stamp), ``intake/walk.json`` (the I3 measurement the chunk plan and the
    anchors come from: every window's content sha256 and the DA3 identity), the SALAD revisit
    reference when present, the Omega weights and the sky model when the config names them;
    CODE — :func:`fork_code_files`; CONFIG — the whole fork config (minus log-only fields) and
    the keyframe order."""
    session_dir, frames_dir = Path(session_dir), Path(frames_dir)
    import repro
    inputs: Dict[str, Path] = {}
    for name in keyframes:
        p = frames_dir / name
        if not p.is_file():
            raise OmegaCompleteError(f"keyframe {p} does not exist — the launch cannot be stamped")
        inputs[f"frames/{name}"] = p
    walk = session_dir / "intake" / "walk.json"
    if not walk.is_file():
        raise OmegaCompleteError(f"{walk} does not exist — the chunk plan's measurement is an "
                                 f"input of every Omega launch")
    inputs["intake/walk.json"] = walk
    ref = session_dir / "output" / "salad_revisit_reference.json"
    if ref.is_file():
        inputs["salad_revisit_reference.json"] = ref
    w = (fork_config.get("Weights") or {})
    wpath = w.get("VGGTOmega") if w.get("model") == "VGGTOmega" else None
    if wpath and Path(str(wpath)).is_file():
        inputs["weights/VGGTOmega"] = Path(str(wpath))
    if (fork_config.get("Model") or {}).get("mask_sky", True):
        sky = repro.FORK_ROOT / "skyseg.onnx"
        if sky.is_file():
            inputs["skyseg.onnx"] = sky
    return repro.stamp(inputs=inputs, code=fork_code_files(),
                       config={"fork_config": _config_for_stamp(fork_config),
                               "frame_list": [str(k) for k in keyframes]})


def check(output_dir: Path, now: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """(complete, why_not): True only when the marker's stamp equals ``now`` and every product
    it recorded is on disk (unchanged where a sha256 was recorded). Every difference is named."""
    import repro
    output_dir = Path(output_dir)
    p = marker_path(output_dir)
    if not p.exists():
        return False, ["no completion marker (the fork never finished under this plan, or "
                       "the plan changed)"]
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return False, [f"{p.name} unreadable ({e})"]
    if not isinstance(doc, dict) or doc.get("version") != MARKER_VERSION:
        return False, [f"{p.name} is not a version-{MARKER_VERSION} marker"]
    diffs = list(repro.check_stamp(doc.get("stamp"), now))
    products = doc.get("products") if isinstance(doc.get("products"), dict) else {}
    if not products:
        diffs.append(f"{p.name} records no product")
    for rel, sha in sorted(products.items()):
        f = output_dir / rel
        if not f.is_file():
            diffs.append(f"product {rel} is gone")
        elif sha is not None and repro.sha256_file(f) != sha:
            diffs.append(f"product {rel} changed since the fork finished")
    return (not diffs), diffs


def write(output_dir: Path, now: Mapping[str, Any]) -> Path:
    """Record that the fork finished under ``now`` with the products on disk. Every product of
    PRODUCTS_EXIST must exist (the fork's poses are what the next stages read)."""
    import repro
    output_dir = Path(output_dir)
    products: Dict[str, Optional[str]] = {}
    for rel in PRODUCTS_EXIST:
        if not (output_dir / rel).is_file():
            raise OmegaCompleteError(f"the fork finished but {output_dir / rel} does not exist — "
                                     f"nothing to mark complete")
        products[rel] = None
    for rel in PRODUCTS_SHA256:
        f = output_dir / rel
        if f.is_file():
            products[rel] = repro.sha256_file(f)
    doc = {"version": MARKER_VERSION, "stamp": dict(now), "products": products}
    p = marker_path(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, p)
    return p


def clear(output_dir: Path) -> None:
    try:
        marker_path(output_dir).unlink()
    except FileNotFoundError:
        pass
