"""The one intake command: I0 quality features → I1 parallax keyframes +
witness frames → I2 VLM content tags + SAM3 exclusion masks, with a resume
marker (claude_stac.txt §4-F1; ``python -m intake.run --session <dir>``).

Resume by marker (conventions rule 6). ``<session>/intake/intake_state.json``
records, per step, the hash of the step's effective parameters and the
identity of the step's INPUTS — the frame inventory (names + sizes) for I0,
plus the I0 report for I1, plus the keyframe / witness lists for I2 — and the
step's own version (a change of the measurement re-runs it too). A step
whose entry matches (same parameters, same inputs, artifacts present and
readable) is skipped; anything else re-runs it and the log says why. Keying
on inputs rather than on "did the previous step run" means a deterministic
step that re-ran and produced the same output does NOT drag the steps after
it along: a reconstruction with "replace" deletes ``frames/selected_frames.
json`` (pipeline_manager.FRAMES_DIR_FILES), I1 re-measures the same frames
into the same keyframes, and I2 — the only step that needs the VLM and SAM3
— stays skipped.

I2's exclusion masks are consumed downstream (F4/F6/F7 exclude the
observations they cover); I1 does not re-read its keyframes under them.

This module decides nothing about frames: every threshold lives in the three
stage modules and in ``config.yaml intake:``. It runs on CPU except for what
I2 delegates to the semantic service / SAM3, and it constructs neither: the
production tagger / segmenter are built lazily by ``intake.content.
run_content`` only when ``intake.content.enabled`` and the step runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from intake import content as Cn
from intake import parallax as P
from intake import quality as Q
from intake.config import IntakeConfig, load_intake_config

INTAKE_DIRNAME = "intake"
FRAMES_DIRNAME = "frames"
STATE_NAME = "intake_state.json"
STATE_VERSION = 1
PROVENANCE = "tool_measured"
STEPS = ("quality", "parallax", "content")
LOG_TAG = "[intake.run]"

# Why a step ran / did not run — every decision names one of these (rule 5).
REASON_FORCE = "force"
REASON_NO_MARKER = "no_marker"
REASON_MARKER_UNREADABLE = "marker_unreadable"
REASON_MARKER_VERSION = "marker_version"
REASON_NOT_DONE = "marker_not_done"
REASON_PARAMS_CHANGED = "params_changed"
REASON_INPUTS_CHANGED = "inputs_changed"
REASON_ARTIFACT = "artifact"
REASON_MATCHES = "marker_matches"
REASON_SKIP_CONTENT = "skip_content"

Progress = Callable[[int, str], Any]


class IntakeRunError(RuntimeError):
    """Structural failures only: no frames directory, no frame, an artifact of
    a previous run that cannot be read where the marker says it can."""


# ── small helpers ────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    os.replace(tmp, path)
    return path


def _digest(obj: Any) -> str:
    """sha1 of the canonical JSON of ``obj`` (identity of an input, not a
    security primitive)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _jsonable_params(cfg: Any) -> Dict[str, Any]:
    """A step's dataclass as JSON-able dict (tuples → lists)."""
    return json.loads(json.dumps(asdict(cfg), default=list))


def params_hash(cfg: Any) -> str:
    """Hash of a step's effective parameters (its config dataclass)."""
    return _digest(_jsonable_params(cfg))


def _progress(progress: Optional[Progress], pct: int, msg: str) -> None:
    if progress is not None:
        progress(int(pct), msg)


def state_path(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / INTAKE_DIRNAME / STATE_NAME


def load_state(session_dir: os.PathLike) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(marker, problem)``. ``(None, None)`` when there is no marker;
    ``(None, "marker_unreadable: <why>")`` when a marker exists but cannot be
    read as a JSON object — every step then re-runs with THAT reason on record
    (the marker is only a resume hint: nothing is lost by re-measuring, and
    the rewrite replaces the broken file)."""
    p = state_path(session_dir)
    if not p.exists():
        return None, None
    try:
        with open(p) as f:
            st = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"{REASON_MARKER_UNREADABLE}: {p.name} ({type(e).__name__}: {e})"
    if not isinstance(st, dict):
        return None, (f"{REASON_MARKER_UNREADABLE}: {p.name} holds a "
                      f"{type(st).__name__}, not a JSON object")
    return st, None


def write_state(session_dir: os.PathLike, state: Dict[str, Any]) -> Path:
    state["updated_at"] = _now_iso()
    return _write_json_atomic(state_path(session_dir), state)


# ── input identities ─────────────────────────────────────────────────────

def frame_inventory(frames_dir: os.PathLike) -> Dict[str, Any]:
    """Identity of the frame inventory: count, first / last basename, total
    bytes and a digest over every (basename, size). Sizes rather than mtimes
    so a copied session keeps its marker. IntakeRunError when the directory
    holds no frame."""
    frames_dir = Path(frames_dir)
    try:
        paths = Q.list_frames(frames_dir)
    except Q.QualityError as e:
        raise IntakeRunError(str(e)) from e
    h = hashlib.sha1()
    bytes_total = 0
    for p in paths:
        size = p.stat().st_size
        bytes_total += size
        h.update(f"{p.name}:{size}\n".encode())
    return {
        "n_frames": len(paths),
        "first": paths[0].name,
        "last": paths[-1].name,
        "bytes_total": int(bytes_total),
        "digest": h.hexdigest(),
    }


def quality_identity(report: Dict[str, Any]) -> Dict[str, Any]:
    """What I1 consumes of the I0 report: the native size and every per-frame
    record (usable flag, sharp_rank, file)."""
    return {
        "n_frames": int(report["n_frames"]),
        "n_usable": int(report["n_usable"]),
        "digest": _digest({"native_w": report["native_w"], "native_h": report["native_h"],
                           "frames": report["frames"]}),
    }


def frame_list_identity(frames: Sequence[int]) -> Dict[str, Any]:
    """Identity of a keyframe / witness list."""
    frames = [int(f) for f in frames]
    return {
        "n": len(frames),
        "first": frames[0] if frames else None,
        "last": frames[-1] if frames else None,
        "digest": _digest(frames),
    }


# ── artifact checks (None = fine, else the problem, for the log) ─────────

def _quality_problem(frames_dir: Path) -> Optional[str]:
    try:
        Q.load_quality(frames_dir)
    except Q.QualityError as e:
        return str(e)
    return None


def _parallax_problem(session_dir: Path) -> Optional[str]:
    frames_dir = session_dir / FRAMES_DIRNAME
    try:
        P.load_selection(frames_dir)
    except P.ParallaxError as e:
        return str(e)
    p_w = frames_dir / P.WITNESS_FRAMES_NAME
    if not p_w.exists():
        return f"{p_w} does not exist"
    try:
        with open(p_w) as f:
            wit = json.load(f)
    except (OSError, ValueError) as e:
        return f"{p_w} is unreadable ({e})"
    if not isinstance(wit.get("frames"), list) or not isinstance(wit.get("selected_files"), list):
        return f"{p_w} lacks 'frames' / 'selected_files'"
    p_warn = session_dir / INTAKE_DIRNAME / P.COVERAGE_WARNINGS_NAME
    if not p_warn.exists():
        return f"{p_warn} does not exist"
    return None


def _content_problem(session_dir: Path) -> Optional[str]:
    try:
        report = Cn.load_content(session_dir)
    except Cn.ContentError as e:
        return str(e)
    masks = report.get("exclusion_masks")
    if not isinstance(masks, dict) or not isinstance(masks.get("dir"), str) \
            or not isinstance(masks.get("frames"), dict):
        return (f"{Cn.content_tags_path(session_dir)} lacks exclusion_masks.dir / "
                f"exclusion_masks.frames (always written by intake I2)")
    masks_dir = Path(masks["dir"])
    missing = [f for f in masks["frames"] if not Cn.mask_path(masks_dir, int(f)).exists()]
    if missing:
        return (f"{len(missing)} exclusion mask PNG(s) named by content_tags.json are "
                f"missing from {masks_dir} (first: {missing[0]})")
    return None


def ensure_legacy_shim(frames_dir: Path, quality: Dict[str, Any], log: Callable) -> Dict[str, Any]:
    """When I0 is skipped, ``frames/frame_quality.json`` must still be THIS
    intake's shim of THIS quality report: the legacy readers
    (frames.selector, segmentation.pipeline, the vendor's _stac_extra_frames)
    take their ``valid`` flags from it. A file that is missing (a
    reconstruction with replace deletes it), unreadable, written by another
    selector (the legacy blur analysis writes a percentile-culled one with
    its own ``method``) or not identical to the shim of the current report
    is rewritten from ``quality_features.json`` — nothing is re-measured —
    and the reason is logged and returned: ``{"action": "kept" | "rewritten",
    "reason"}``."""
    shim = frames_dir / Q.LEGACY_FRAME_QUALITY_NAME
    expected = json.loads(json.dumps(Q.legacy_frame_quality(quality)))
    reason = None
    if not shim.exists():
        reason = "missing (a reconstruction with replace deletes it)"
    else:
        try:
            with open(shim) as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            doc, reason = None, f"unreadable ({type(e).__name__}: {e})"
        if reason is None:
            method = doc.get("method") if isinstance(doc, dict) else None
            if method != Q.METHOD:
                reason = (f"foreign: written by {method!r}, not by the intake "
                          f"({Q.METHOD!r}) — its 'valid' flags are another selector's")
            elif doc != expected:
                reason = (f"stale: it is not the shim of the current "
                          f"{Q.QUALITY_FEATURES_NAME}")
    if reason is None:
        return {"action": "kept", "reason": "identical to the shim of the current report"}
    Q.write_legacy_frame_quality(frames_dir, quality)
    log(f"{LOG_TAG} quality: legacy {shim.name} {reason} — rewritten from "
        f"{Q.QUALITY_FEATURES_NAME}, nothing re-measured")
    return {"action": "rewritten", "reason": reason}


# ── the decision ─────────────────────────────────────────────────────────

def decide(state: Optional[Dict[str, Any]], step: str, p_hash: str, inputs: Dict[str, Any],
           force: bool, artifact_problem: Callable[[], Optional[str]],
           marker_problem: Optional[str] = None) -> Tuple[bool, str]:
    """(run, reason). A step runs when forced, when the marker is absent /
    unreadable (``marker_problem``, from :func:`load_state`) / of another
    version / not done for it, when its parameters or inputs differ from the
    recorded ones, or when its artifacts are missing or unreadable. Otherwise
    it is skipped with reason ``marker_matches``."""
    if force:
        return True, REASON_FORCE
    if marker_problem:
        return True, marker_problem
    if state is None:
        return True, REASON_NO_MARKER
    if state.get("version") != STATE_VERSION:
        return True, f"{REASON_MARKER_VERSION}:{state.get('version')!r}"
    entry = (state.get("steps") or {}).get(step)
    if not isinstance(entry, dict) or not entry.get("done"):
        return True, REASON_NOT_DONE
    if entry.get("params_hash") != p_hash:
        return True, REASON_PARAMS_CHANGED
    if entry.get("inputs") != inputs:
        return True, REASON_INPUTS_CHANGED
    problem = artifact_problem()
    if problem:
        return True, f"{REASON_ARTIFACT}: {problem}"
    return False, REASON_MATCHES


def _entry(p_hash: str, inputs: Dict[str, Any], artifacts: List[Path], t0: float,
           summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "done": True,
        "params_hash": p_hash,
        "inputs": inputs,
        "artifacts": [str(a) for a in artifacts],
        "summary": summary,
        "finished_at": _now_iso(),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


def _load_json(path: Path, what: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise IntakeRunError(f"{what} {path} is unreadable ({e})") from e


# ── the command ──────────────────────────────────────────────────────────

def run_intake(session_dir: os.PathLike, icfg: IntakeConfig, *, tagger: Optional[Cn.VLMTagger] = None,
               segmenter: Optional[Cn.Segmenter] = None, log: Callable = print,
               progress: Optional[Progress] = None, force: bool = False,
               skip_content: bool = False,
               before_content: Optional[Callable[[], Any]] = None,
               before_sam3: Optional[Callable[[], Any]] = None,
               cancelled: Q.Cancelled = None) -> Dict[str, Any]:
    """I0 → I1 → I2 on ``<session>/frames`` with the resume marker
    ``<session>/intake/intake_state.json``.

    ``force`` re-runs every step. ``skip_content`` leaves I2 out of this pass:
    nothing is tagged, no VLM or SAM3 is touched, ``content_tags.json`` is NOT
    written; a previous run's ``content_tags.json`` stays on disk untouched
    and the marker's ``content`` entry records ``done`` false, ``skipped``
    true and whether that file is now STALE (its inputs no longer match the
    current keyframes / witnesses). ``tagger`` / ``segmenter`` None → the
    production QwenTagger / Sam3Segmenter, constructed by
    :func:`intake.content.run_content` only when ``intake.content.enabled``
    and I2 actually runs. ``before_content`` (optional) is called right before
    I2 runs, and only when it runs with ``content.enabled`` — the place a
    caller makes sure the semantic service is reachable and fails with the
    reason when it is not. ``before_sam3`` (optional, default: nothing) is
    forwarded to :func:`intake.content.run_content`, which calls it after the
    last VLM tag and before the first SAM3 call — the place a caller hands the
    GPU over (the map_worker stops vLLM there: the two never share the card).
    ``cancelled()`` (optional) is polled inside every loop and raises
    ``intake.quality.IntakeCancelled`` naming the step. ``progress(pct, msg)``
    is optional. Every artifact is stamped with the session's epochs, read
    once here (``intake.quality.read_session_epochs``).

    Returns the run record: ``steps`` {step: {ran, reason, ...}}, the three
    stage reports as they stand on disk after the pass (``quality``,
    ``parallax`` {selected_frames, witness_frames, coverage_warnings},
    ``content`` or None), ``summary`` counts, ``artifacts`` paths and the
    marker path."""
    t_run = time.monotonic()
    session_dir = Path(session_dir)
    frames_dir = session_dir / FRAMES_DIRNAME
    if not frames_dir.is_dir():
        raise IntakeRunError(f"{frames_dir} is not a directory — a session holds its native "
                             f"frames in <session>/frames/<frame:06d>.jpg")
    hb = icfg.runtime.heartbeat_s
    epochs = Q.read_session_epochs(session_dir)
    inventory = frame_inventory(frames_dir)
    log(f"{LOG_TAG} session {session_dir}: {inventory['n_frames']} frame(s) "
        f"{inventory['first']}..{inventory['last']} ({inventory['bytes_total']} bytes), "
        f"force={force}, skip_content={skip_content}")

    old_state, marker_problem = load_state(session_dir)
    if marker_problem:
        log(f"{LOG_TAG} {marker_problem} — every step re-runs and the marker is rewritten")
    if old_state is not None and old_state.get("version") != STATE_VERSION:
        log(f"{LOG_TAG} marker {state_path(session_dir)} is version "
            f"{old_state.get('version')!r}, this module writes {STATE_VERSION} — every step "
            f"re-runs and the marker is rewritten")
    # the new marker starts from the old entries (a skipped step keeps its own) when the
    # old marker is of this version; a foreign version contributes nothing
    kept_steps: Dict[str, Any] = {}
    if old_state is not None and old_state.get("version") == STATE_VERSION:
        kept_steps = dict(old_state.get("steps") or {})
    state: Dict[str, Any] = {
        "version": STATE_VERSION,
        "provenance": PROVENANCE,
        **epochs,
        "session_dir": str(session_dir),
        "frames_dir": str(frames_dir),
        "steps": kept_steps,
    }
    steps_out: Dict[str, Dict[str, Any]] = {}

    # ── I0 quality ───────────────────────────────────────────────────────
    _progress(progress, 0, "intake I0: quality features")
    ph0 = params_hash(icfg.quality)
    # each step's identity carries its stage version: a change of the measurement
    # itself (not only of its parameters) re-runs the step
    in0 = {"frames": inventory, "stage_version": Q.QUALITY_VERSION}
    run0, why0 = decide(old_state, "quality", ph0, in0, force,
                        lambda: _quality_problem(frames_dir), marker_problem)
    if run0:
        log(f"{LOG_TAG} quality: running ({why0})")
        t0 = time.monotonic()
        quality = Q.run_quality(frames_dir, icfg.quality, log=log, heartbeat_s=hb,
                                cancelled=cancelled, **epochs)
        state["steps"]["quality"] = _entry(
            ph0, in0, [frames_dir / Q.QUALITY_FEATURES_NAME, frames_dir / Q.LEGACY_FRAME_QUALITY_NAME],
            t0, {"n_frames": quality["n_frames"], "n_usable": quality["n_usable"],
                 "rejected": quality["rejected"]})
        write_state(session_dir, state)
    else:
        log(f"{LOG_TAG} quality: skipped ({why0}: same parameters, same "
            f"{inventory['n_frames']} frames) — reusing {Q.QUALITY_FEATURES_NAME}")
        quality = Q.load_quality(frames_dir)
    shim_record = ensure_legacy_shim(frames_dir, quality, log) if not run0 else \
        {"action": "written", "reason": "I0 ran"}
    steps_out["quality"] = {"ran": run0, "reason": why0, "legacy_shim": shim_record}

    # ── I1 parallax ──────────────────────────────────────────────────────
    _progress(progress, 10, "intake I1: parallax keyframes + witness frames")
    ph1 = params_hash(icfg.parallax)
    in1 = {"frames": inventory, "quality": quality_identity(quality),
           "stage_version": P.PARALLAX_VERSION}
    run1, why1 = decide(old_state, "parallax", ph1, in1, force,
                        lambda: _parallax_problem(session_dir), marker_problem)
    if run1:
        log(f"{LOG_TAG} parallax: running ({why1})")
        t0 = time.monotonic()
        result = P.run_parallax(frames_dir, quality, icfg.parallax, log=log, heartbeat_s=hb,
                                cancelled=cancelled, **epochs)
        p_kf, p_w, p_warn = P.write_selection(session_dir, result, icfg.parallax)
        state["steps"]["parallax"] = _entry(
            ph1, in1, [p_kf, p_w, p_warn], t0,
            {"n_keyframes": result["n_keyframes"], "n_witness": result["n_witness"],
             "n_warnings": len(result["warnings"]), "n_lost": result["n_lost"]})
        write_state(session_dir, state)
    else:
        log(f"{LOG_TAG} parallax: skipped ({why1}: same parameters, same frames, same "
            f"quality report) — reusing {P.SELECTED_FRAMES_NAME} / {P.WITNESS_FRAMES_NAME}")
    selected = P.load_selection(frames_dir)
    witness = _load_json(frames_dir / P.WITNESS_FRAMES_NAME, "witness_frames.json")
    warnings_doc = _load_json(session_dir / INTAKE_DIRNAME / P.COVERAGE_WARNINGS_NAME,
                              "coverage_warnings.json")
    steps_out["parallax"] = {"ran": run1, "reason": why1}

    # ── I2 content ───────────────────────────────────────────────────────
    _progress(progress, 40, "intake I2: content tags + exclusion masks")
    keyframes, witnesses = Cn.read_i1_frames(session_dir)
    ph2 = params_hash(icfg.content)
    in2 = {"frames": inventory, "keyframes": frame_list_identity(keyframes),
           "witnesses": frame_list_identity(witnesses), "stage_version": Cn.CONTENT_VERSION}
    run2, why2 = decide(old_state, "content", ph2, in2, force,
                        lambda: _content_problem(session_dir), marker_problem)
    content: Optional[Dict[str, Any]] = None
    content_step: Dict[str, Any] = {"ran": False, "reason": why2, "skipped": False}
    if not run2:
        log(f"{LOG_TAG} content: skipped ({why2}: same parameters, same keyframes / "
            f"witnesses) — reusing {Cn.CONTENT_TAGS_NAME}")
        content = Cn.load_content(session_dir)
    elif skip_content:
        stale = Cn.content_tags_path(session_dir).exists()
        reason = (f"{REASON_SKIP_CONTENT}: I2 not run in this pass (it would have run: "
                  f"{why2})")
        log(f"{LOG_TAG} content: skipped ({reason})"
            + (f" — WARNING: the existing {Cn.CONTENT_TAGS_NAME} predates the current "
               f"keyframes / witnesses and is STALE" if stale else ""))
        state["steps"]["content"] = {
            "done": False, "skipped": True, "reason": reason, "params_hash": ph2,
            "inputs": in2, "stale_artifact": stale, "finished_at": _now_iso(),
        }
        write_state(session_dir, state)
        content_step = {"ran": False, "reason": REASON_SKIP_CONTENT, "skipped": True,
                        "would_have_run": why2, "stale_artifact": stale}
    else:
        log(f"{LOG_TAG} content: running ({why2}; enabled={icfg.content.enabled}, "
            f"backend={icfg.content.backend})")
        if icfg.content.enabled and before_content is not None:
            before_content()
        t0 = time.monotonic()
        content = Cn.run_content(session_dir, keyframes, witnesses, icfg.content,
                                 tagger=tagger, segmenter=segmenter, log=log, heartbeat_s=hb,
                                 before_sam3=before_sam3, cancelled=cancelled)
        state["steps"]["content"] = _entry(
            ph2, in2, [Cn.content_tags_path(session_dir)], t0,
            {"enabled": content["enabled"],
             "tagged": content["summary"]["tagged"],
             "parse_failures": content["parse_failures"],
             "n_exclusion_masks": content["exclusion_masks"]["n_frames_written"]})
        write_state(session_dir, state)
        content_step = {"ran": True, "reason": why2, "skipped": False}
    steps_out["content"] = content_step

    _progress(progress, 100, "intake done")
    summary = {
        "n_frames": int(quality["n_frames"]),
        "n_usable": int(quality["n_usable"]),
        "n_keyframes": int(selected["selected_count"]),
        "n_witness": len(witness["frames"]),
        "n_warnings": len(warnings_doc["warnings"]),
        "content_enabled": (bool(content["enabled"]) if content is not None else None),
        "n_exclusion_masks": (int(content["exclusion_masks"]["n_frames_written"])
                              if content is not None else None),
    }
    log(f"{LOG_TAG} done in {time.monotonic() - t_run:.1f} s: {summary['n_usable']}/"
        f"{summary['n_frames']} usable frames, {summary['n_keyframes']} keyframes, "
        f"{summary['n_witness']} witnesses, {summary['n_warnings']} coverage warning(s), "
        f"content {'not run' if content is None else summary['n_exclusion_masks']}"
        f"{'' if content is None else ' exclusion mask(s)'}; steps "
        + ", ".join(f"{k}={'ran' if v['ran'] else 'skipped'}" for k, v in steps_out.items()))
    return {
        "version": STATE_VERSION,
        "provenance": PROVENANCE,
        **epochs,
        "session_dir": str(session_dir),
        "force": bool(force),
        "skip_content": bool(skip_content),
        "steps": steps_out,
        "quality": quality,
        "parallax": {"selected_frames": selected, "witness_frames": witness,
                     "coverage_warnings": warnings_doc},
        "content": content,
        "summary": summary,
        "artifacts": {
            "quality_features": str(frames_dir / Q.QUALITY_FEATURES_NAME),
            "frame_quality": str(frames_dir / Q.LEGACY_FRAME_QUALITY_NAME),
            "selected_frames": str(frames_dir / P.SELECTED_FRAMES_NAME),
            "witness_frames": str(frames_dir / P.WITNESS_FRAMES_NAME),
            "coverage_warnings": str(session_dir / INTAKE_DIRNAME / P.COVERAGE_WARNINGS_NAME),
            "content_tags": (str(Cn.content_tags_path(session_dir)) if content is not None
                             else None),
            "exclusion_masks_dir": (str(Cn.exclusion_masks_dir(session_dir))
                                    if content is not None else None),
            "state": str(state_path(session_dir)),
        },
        "elapsed_s": round(time.monotonic() - t_run, 3),
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m intake.run",
        description="Intake: I0 quality features → I1 parallax keyframes + witness frames → "
                    "I2 VLM content tags + SAM3 exclusion masks, resumed by marker.")
    ap.add_argument("--session", required=True,
                    help="session directory (frames in <session>/frames/<frame:06d>.jpg)")
    ap.add_argument("--force", action="store_true",
                    help="re-run every step even when its marker matches")
    ap.add_argument("--skip-content", action="store_true",
                    help="leave I2 out of this pass (no VLM, no SAM3; content_tags.json not "
                         "written)")
    args = ap.parse_args(argv)
    icfg = load_intake_config()
    res = run_intake(Path(args.session), icfg, log=print, force=args.force,
                     skip_content=args.skip_content,
                     before_content=Cn.cli_before_content(print),
                     before_sam3=Cn.cli_before_sam3(print))
    s = res["summary"]
    print(f"{LOG_TAG} keyframes={s['n_keyframes']} witnesses={s['n_witness']} "
          f"warnings={s['n_warnings']} usable={s['n_usable']}/{s['n_frames']} "
          f"content={'skipped' if res['content'] is None else ('enabled' if s['content_enabled'] else 'disabled')}"
          f" exclusion_masks={s['n_exclusion_masks']}; marker {res['artifacts']['state']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
