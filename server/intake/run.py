"""The one intake command: I0 quality features → I1 parallax keyframes +
witness frames → I2 VLM content tags + SAM3 exclusion masks, with a resume
marker (claude_stac.txt §4-F1; ``python -m intake.run --session <dir>``).

Resume by STAMP (docs/plan_determinismo.md point 66, 2026-10-07; conventions
rule 6). ``<session>/intake/intake_state.json`` records, per step, the
:func:`repro.stamp` of everything its product depends on — the sha256 of
EVERY frame's bytes (keyed by its session-relative path), the step's upstream
products (I0's report for I1; I1's lists for I2), the intake and DA3-extractor
code (:data:`intake.stamps.INTAKE_CODE_FILES`), its parameters, the session
camera K and the focal probe's own stamp (I1), and the CPU environment
(library versions, CPU, BLAS core, JPEG decoders — points 74 / 79). A step
whose saved stamp matches the one computed now (artifacts present and
readable) is skipped; anything else re-runs it and the log names what
differs. Frames used to be identified by name + size and a step by a version
number bumped by hand: a resumed session kept keyframes computed by older
code, another numpy or another card. The frozen run configuration's sha256
(point 69) is RECORDED in the marker and in every step's entry — a product
says which configuration it was made under — and is not a key: the step's
own parameters are, exactly; a change elsewhere in config.yaml re-runs
nothing of the intake.

The frames are SEALED (points 67 / 76, :mod:`intake.frames_manifest`): the
intake refuses a scan whose frames are being written (a writer's claim or the
``frames.extracting/`` temp dir) or whose ``frames/manifest.json`` is not
complete, checks the frames on disk against the manifest (names and count,
then every frame's sha256 from I0's own stamp — no second hashing pass) and
fails naming the first difference. A ``frames/`` with no manifest predates
them and is ADOPTED once (origin ``adopted``, DECLARED in the log).

The marker holds NO wall clock, NO absolute path and NO session epoch (point
70): those go to the run record beside it, ``intake_state.timing.json``,
which is never a compared product. Every intake product stamps itself with
``geometry_epoch`` / ``camera_epoch`` = 0 (:data:`intake.quality.INTAKE_EPOCH`:
the intake precedes every reconstruction and feeds its epoch 0).

Keying on inputs rather than on "did the previous step run" means a
deterministic step that re-ran and produced the same output does NOT drag the
steps after it along: a reconstruction with "replace" deletes ``frames/
selected_frames.json`` (pipeline_manager.FRAMES_DIR_FILES), I1 re-measures
the same frames into the same keyframes, and I2 — the only step that needs
the VLM and SAM3 — stays skipped.

I2's exclusion masks are consumed downstream only through
``intake.content.valid_exclusion_masks`` (the inventory + its stamp); I1 does
not re-read its keyframes under them.

The configuration every step runs on is the job's FROZEN copy,
``output/run_config.yaml`` (point 69): the pipeline manager freezes it at job
start, a CLI freezes its own, and :func:`run_intake` refuses an intake
configuration that is not the frozen one.

This module decides nothing about frames: every threshold lives in the three
stage modules and in ``config.yaml intake:``. It runs on CPU except for what
I2 delegates to the semantic service / SAM3, and it constructs neither: the
production tagger / segmenter are built lazily by ``intake.content.
run_content`` only when ``intake.content.enabled`` and the step runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from intake import content as Cn
from intake import frames_manifest as FM
from intake import parallax as P
from intake import quality as Q
from intake import stamps as St
from intake.config import IntakeConfig, load_intake_config
from intake.run_config import (RunConfigError, cli_intake_config, freeze_run_config,
                               verify_intake_config)

INTAKE_DIRNAME = "intake"
FRAMES_DIRNAME = "frames"
STATE_NAME = "intake_state.json"
TIMING_NAME = "intake_state.timing.json"      # the run record: times, epochs, absolute paths
STATE_VERSION = 2                             # 2: stamps replace the (name, size, version) keys
PROVENANCE = "tool_measured"
STEPS = ("quality", "parallax", "content")
LOG_TAG = "[intake.run]"

# Why a step ran / did not run — every decision names one of these (rule 5).
REASON_FORCE = "force"
REASON_NO_MARKER = "no_marker"
REASON_MARKER_UNREADABLE = "marker_unreadable"
REASON_MARKER_VERSION = "marker_version"
REASON_NOT_DONE = "marker_not_done"
REASON_PARAMS_CHANGED = "params_changed"      # only the step's own parameters differ
REASON_CODE_CHANGED = "code_changed"          # a file of the intake / DA3 code differs
REASON_INPUTS_CHANGED = "inputs_changed"      # frames, upstream products, environment, K ...
REASON_ARTIFACT = "artifact"
REASON_MATCHES = "marker_matches"
REASON_SKIP_CONTENT = "skip_content"

Progress = Callable[[int, str], Any]


class IntakeRunError(RuntimeError):
    """Structural failures only: no frames directory, no frame, an artifact of
    a previous run that cannot be read where the marker says it can, an intake
    configuration that is not the frozen one."""


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


def _jsonable_params(cfg: Any) -> Dict[str, Any]:
    """A step's dataclass as JSON-able dict (tuples → lists)."""
    return json.loads(json.dumps(asdict(cfg), default=list))


def _progress(progress: Optional[Progress], pct: int, msg: str) -> None:
    if progress is not None:
        progress(int(pct), msg)


def state_path(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / INTAKE_DIRNAME / STATE_NAME


def timing_path(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / INTAKE_DIRNAME / TIMING_NAME


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
    """The marker — no time in it (the run record carries ``updated_at``)."""
    return _write_json_atomic(state_path(session_dir), state)


def write_timing(session_dir: os.PathLike, timing: Dict[str, Any]) -> Path:
    """The run record beside the marker: times, durations, the session's epochs at run time,
    absolute paths, the full environment — never a compared product (point 70)."""
    timing["updated_at"] = _now_iso()
    return _write_json_atomic(timing_path(session_dir), timing)


# ── input identities ─────────────────────────────────────────────────────

def frame_inventory(frames_dir: os.PathLike) -> Dict[str, Any]:
    """The frame inventory for the log and the summaries: count, first / last basename,
    total bytes. (The IDENTITY of the frames is their stamp — every frame's sha256,
    :func:`intake.stamps.frames_stamp`.) IntakeRunError when the directory holds no frame."""
    frames_dir = Path(frames_dir)
    try:
        paths = Q.list_frames(frames_dir)
    except Q.QualityError as e:
        raise IntakeRunError(str(e)) from e
    return {"n_frames": len(paths), "first": paths[0].name, "last": paths[-1].name,
            "bytes_total": int(sum(p.stat().st_size for p in paths))}


def frame_list_identity(frames: Sequence[int]) -> Dict[str, Any]:
    """Identity of a keyframe / witness list (for the log / summaries)."""
    frames = [int(f) for f in frames]
    return {"n": len(frames), "first": frames[0] if frames else None,
            "last": frames[-1] if frames else None}


def focal_probe_stamp_sha(session_dir: Path) -> Optional[str]:
    """The sha256 of the focal probe's own stamp (``intake/focal_probe.json`` — card, weights,
    dtype, DA3 identity, the probe frames' bytes) when the probe wrote one; None when the
    caller injected K another way (tests) — then K alone identifies the camera."""
    from intake.focal import FOCAL_NAME
    p = session_dir / INTAKE_DIRNAME / FOCAL_NAME
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    st = doc.get("stamp") if isinstance(doc, dict) else None
    return str(st["sha256"]) if isinstance(st, dict) and st.get("sha256") else None


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
            or not isinstance(masks.get("frames"), dict) or not isinstance(masks.get("stamp"), dict):
        return (f"{Cn.content_tags_path(session_dir)} lacks exclusion_masks.dir / "
                f"exclusion_masks.frames / exclusion_masks.stamp (always written by intake I2)")
    masks_dir = St.in_session(masks["dir"], session_dir)
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

def classify_differences(diffs: Sequence[str]) -> str:
    """The reason a stamp mismatch names: ``params_changed`` when ONLY the step's own
    parameters differ, ``code_changed`` when a code file does, ``inputs_changed`` for
    anything else (frames, upstream products, environment, K, the run configuration)."""
    kinds = set()
    for d in diffs:
        if d.startswith("code "):
            kinds.add("code")
        elif d.startswith("config 'params'"):
            kinds.add("params")
        else:
            kinds.add("inputs")
    if kinds == {"params"}:
        return REASON_PARAMS_CHANGED
    if "code" in kinds and "inputs" not in kinds:
        return REASON_CODE_CHANGED
    return REASON_INPUTS_CHANGED


def decide(state: Optional[Dict[str, Any]], step: str, stamp_now: Mapping[str, Any],
           force: bool, artifact_problem: Callable[[], Optional[str]],
           marker_problem: Optional[str] = None) -> Tuple[bool, str, List[str]]:
    """(run, reason, differences). A step runs when forced, when the marker is absent /
    unreadable (``marker_problem``, from :func:`load_state`) / of another version / not
    done for it, when its saved stamp differs from ``stamp_now`` (the differences are
    returned and the reason classifies them), or when its artifacts are missing or
    unreadable. Otherwise it is skipped with reason ``marker_matches``."""
    if force:
        return True, REASON_FORCE, []
    if marker_problem:
        return True, marker_problem, []
    if state is None:
        return True, REASON_NO_MARKER, []
    if state.get("version") != STATE_VERSION:
        return True, f"{REASON_MARKER_VERSION}:{state.get('version')!r}", []
    entry = (state.get("steps") or {}).get(step)
    if not isinstance(entry, dict) or not entry.get("done"):
        return True, REASON_NOT_DONE, []
    diffs = St.stamp_differences(entry.get("stamp"), stamp_now)
    if diffs:
        return True, classify_differences(diffs), diffs
    problem = artifact_problem()
    if problem:
        return True, f"{REASON_ARTIFACT}: {problem}", []
    return False, REASON_MATCHES, []


def _entry(stamp: Mapping[str, Any], artifacts: List[Path], session_dir: Path,
           summary: Dict[str, Any], run_config_sha256: str) -> Dict[str, Any]:
    return {
        "done": True,
        "stamp": dict(stamp),
        "run_config_sha256": run_config_sha256,     # recorded, not a key (point 69)
        "artifacts": [St.rel_to_session(a, session_dir) for a in artifacts],
        "summary": summary,
    }


def _timing_entry(t0: float) -> Dict[str, Any]:
    return {"finished_at": _now_iso(), "elapsed_s": round(time.monotonic() - t0, 3)}


def _load_json(path: Path, what: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise IntakeRunError(f"{what} {path} is unreadable ({e})") from e


def _said(log: Callable, step: str, reason: str, diffs: Sequence[str]) -> None:
    log(f"{LOG_TAG} {step}: running ({reason})"
        + ("" if not diffs else " — " + "; ".join(diffs[:8])
           + (f"; +{len(diffs) - 8} more" if len(diffs) > 8 else "")))


# ── the command ──────────────────────────────────────────────────────────

def run_intake(session_dir: os.PathLike, icfg: IntakeConfig, *, tagger: Optional[Cn.VLMTagger] = None,
               segmenter: Optional[Cn.Segmenter] = None, log: Callable = print,
               progress: Optional[Progress] = None, force: bool = False,
               skip_content: bool = False,
               before_content: Optional[Callable[[], Any]] = None,
               before_sam3: Optional[Callable[[], Any]] = None,
               focal: Optional[Callable[..., Any]] = None,
               cancelled: Q.Cancelled = None,
               run_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """I0 → I1 → I2 on ``<session>/frames`` with the resume marker
    ``<session>/intake/intake_state.json`` (stamps) and the run record
    ``intake_state.timing.json`` beside it.

    ``force`` re-runs every step. ``skip_content`` leaves I2 out of this pass:
    nothing is tagged, no VLM or SAM3 is touched, ``content_tags.json`` is NOT
    written; a previous run's ``content_tags.json`` stays on disk untouched
    and the marker's ``content`` entry records ``done`` false, ``skipped``
    true and whether that file is now STALE (its stamp no longer matches the
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
    ``focal(session_dir, quality, parallax_cfg, log, cancelled)`` returns the session
    camera's K (native px) that I1's rotation reference uses — default the DA3 probe
    of :mod:`intake.focal` (GPU, in this interpreter), reused on its own stamp; K and
    the probe's stamp enter I1's stamp, so a different K or card re-runs I1.
    ``cancelled()`` (optional) is polled inside every loop and raises
    ``intake.quality.IntakeCancelled`` naming the step. ``progress(pct, msg)``
    is optional.

    ``run_config``: the job's raw configuration when THIS call starts the job
    (a CLI) — it is frozen to ``output/run_config.yaml`` first; None when the
    job was frozen by the pipeline manager — then ``icfg`` must be the frozen
    configuration's intake section, or this fails (point 69). The marker and
    every step entry record the frozen configuration's sha256.

    Returns the run record: ``steps`` {step: {ran, reason, differences ...}},
    the three stage reports as they stand on disk after the pass
    (``quality``, ``parallax`` {selected_frames, witness_frames,
    coverage_warnings}, ``content`` or None), ``summary`` counts,
    ``artifacts`` paths (absolute, for the caller) and the marker path."""
    t_run = time.monotonic()
    session_dir = Path(session_dir)
    frames_dir = session_dir / FRAMES_DIRNAME
    # the frames are SEALED (points 67 / 76): never read while a writer is at work or its temp
    # dir exists, never through a manifest that is not complete; None = a frames/ that predates
    # manifests, adopted below once I0's stamp has hashed every frame
    try:
        manifest = FM.check_ready(session_dir)
    except FM.FramesManifestError as e:
        raise IntakeRunError(str(e)) from e
    if not frames_dir.is_dir():
        raise IntakeRunError(f"{frames_dir} is not a directory — a session holds its native "
                             f"frames in <session>/frames/<frame:06d>.jpg")
    hb = icfg.runtime.heartbeat_s
    # the configuration of this job, frozen (point 69): by the caller that starts the job,
    # or already by the pipeline manager — never config.yaml as it stands on disk
    if run_config is not None:
        frozen_sha = freeze_run_config(session_dir, run_config, log=log)["sha256"]
    try:
        frozen_sha = verify_intake_config(session_dir, icfg)
    except RunConfigError as e:
        raise IntakeRunError(str(e)) from e
    epochs = Q.read_session_epochs(session_dir)           # the run record only (point 70)
    inventory = frame_inventory(frames_dir)
    log(f"{LOG_TAG} session {session_dir}: {inventory['n_frames']} frame(s) "
        f"{inventory['first']}..{inventory['last']} ({inventory['bytes_total']} bytes), "
        f"force={force}, skip_content={skip_content}, run config {frozen_sha[:12]}")
    environment = St.cpu_environment_record()
    frame_inputs = St.frame_inputs(frames_dir, session_dir)
    if manifest is not None:                              # names and count first: no hashing yet
        try:
            FM.verify_names(manifest, [p.name for p in frame_inputs.values()],
                            where=f"{FRAMES_DIRNAME}/")
        except FM.FramesManifestError as e:
            raise IntakeRunError(str(e)) from e

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
        "run_config_sha256": frozen_sha,
        "steps": kept_steps,
    }
    old_timing = {}
    if timing_path(session_dir).exists():
        try:
            old_timing = json.loads(timing_path(session_dir).read_text())
        except (OSError, ValueError):
            old_timing = {}
    timing: Dict[str, Any] = {
        "version": STATE_VERSION,
        **epochs,
        "session_dir": str(session_dir),
        "frames_dir": str(frames_dir),
        "environment": environment,
        "steps": dict((old_timing.get("steps") or {}) if isinstance(old_timing, dict) else {}),
    }
    steps_out: Dict[str, Dict[str, Any]] = {}

    # ── I0 quality ───────────────────────────────────────────────────────
    _progress(progress, 0, "intake I0: quality features")
    # the frames are I0's INPUTS (each hashed on its own: a changed frame is named); the
    # later steps carry the same per-frame digests under their 'frames' config section
    st0 = St.step_stamp(frame_inputs, {"environment": environment,
                                       "params": _jsonable_params(icfg.quality)})
    # every frame's bytes against the manifest, from THIS stamp's digests (no second hashing pass);
    # a frames/ without one is adopted with them (DECLARED in the log)
    frame_shas = {k.split("/", 1)[1]: v for k, v in st0["inputs"].items()}
    try:
        if manifest is None:
            manifest = FM.adopt(session_dir, frame_shas, log=log)
        else:
            FM.verify_shas(manifest, frame_shas, where=f"{FRAMES_DIRNAME}/")
            log(f"{LOG_TAG} frames: every frame's sha256 matches {FRAMES_DIRNAME}/"
                f"{FM.MANIFEST_NAME} (origin {manifest.get('origin')}, {manifest.get('n_frames')} "
                f"frame(s), inventory {str(manifest.get('frames_sha256'))[:12]})")
    except FM.FramesManifestError as e:
        raise IntakeRunError(str(e)) from e
    frames_manifest = {"origin": manifest.get("origin"), "n_frames": manifest.get("n_frames"),
                       "frames_sha256": manifest.get("frames_sha256")}
    timing["frames_manifest"] = frames_manifest
    common = {"frames": st0["inputs"], "environment": environment}
    log(f"{LOG_TAG} frames stamped: {len(st0['inputs'])} file(s) by sha256; intake + DA3 code "
        f"{len(st0['code'])} file(s); numpy {environment['libs']['numpy']}, opencv "
        f"{environment['libs']['opencv-python']}, pillow {environment['libs']['pillow']}, "
        f"{environment['cpu_model']}")
    run0, why0, d0 = decide(old_state, "quality", st0, force,
                            lambda: _quality_problem(frames_dir), marker_problem)
    if run0:
        _said(log, "quality", why0, d0)
        t0 = time.monotonic()
        quality = Q.run_quality(frames_dir, icfg.quality, log=log, heartbeat_s=hb,
                                cancelled=cancelled, environment=environment)
        state["steps"]["quality"] = _entry(
            st0, [frames_dir / Q.QUALITY_FEATURES_NAME, frames_dir / Q.LEGACY_FRAME_QUALITY_NAME],
            session_dir,
            {"n_frames": quality["n_frames"], "n_usable": quality["n_usable"],
             "rejected": quality["rejected"]}, frozen_sha)
        timing["steps"]["quality"] = _timing_entry(t0)
        write_state(session_dir, state)
        write_timing(session_dir, timing)
    else:
        log(f"{LOG_TAG} quality: skipped ({why0}: same frames, parameters, code and "
            f"environment) — reusing {Q.QUALITY_FEATURES_NAME}")
        quality = Q.load_quality(frames_dir)
    shim_record = ensure_legacy_shim(frames_dir, quality, log) if not run0 else \
        {"action": "written", "reason": "I0 ran"}
    steps_out["quality"] = {"ran": run0, "reason": why0, "differences": d0,
                            "legacy_shim": shim_record}

    # ── I1 prerequisite: the session camera, measured ────────────────────
    _progress(progress, 8, "intake I1: the session camera (DA3 focal probe)")
    if focal is None:
        from intake.focal import default_probe
        focal = default_probe()
    K = np.asarray(focal(session_dir, quality, icfg.parallax, log, cancelled), np.float64)

    # ── I1 parallax ──────────────────────────────────────────────────────
    _progress(progress, 10, "intake I1: parallax keyframes + witness frames")
    st1 = St.step_stamp({f"{FRAMES_DIRNAME}/{Q.QUALITY_FEATURES_NAME}":
                         frames_dir / Q.QUALITY_FEATURES_NAME},
                        {**common, "params": _jsonable_params(icfg.parallax),
                         "K": [float(v) for v in K.ravel()],
                         "focal_probe_stamp_sha256": focal_probe_stamp_sha(session_dir)})
    run1, why1, d1 = decide(old_state, "parallax", st1, force,
                            lambda: _parallax_problem(session_dir), marker_problem)
    if run1:
        _said(log, "parallax", why1, d1)
        t0 = time.monotonic()
        result = P.run_parallax(frames_dir, quality, icfg.parallax, K, log=log, heartbeat_s=hb,
                                cancelled=cancelled, environment=environment)
        p_kf, p_w, p_warn = P.write_selection(session_dir, result, icfg.parallax)
        state["steps"]["parallax"] = _entry(
            st1, [p_kf, p_w, p_warn], session_dir,
            {"n_keyframes": result["n_keyframes"], "n_witness": result["n_witness"],
             "n_warnings": len(result["warnings"]), "n_lost": result["n_lost"]}, frozen_sha)
        timing["steps"]["parallax"] = _timing_entry(t0)
        write_state(session_dir, state)
        write_timing(session_dir, timing)
    else:
        log(f"{LOG_TAG} parallax: skipped ({why1}: same frames, quality report, K, parameters, "
            f"code and environment) — reusing {P.SELECTED_FRAMES_NAME} / "
            f"{P.WITNESS_FRAMES_NAME}")
    selected = P.load_selection(frames_dir)
    witness = _load_json(frames_dir / P.WITNESS_FRAMES_NAME, "witness_frames.json")
    warnings_doc = _load_json(session_dir / INTAKE_DIRNAME / P.COVERAGE_WARNINGS_NAME,
                              "coverage_warnings.json")
    steps_out["parallax"] = {"ran": run1, "reason": why1, "differences": d1}

    # ── I2 content ───────────────────────────────────────────────────────
    _progress(progress, 40, "intake I2: content tags + exclusion masks")
    keyframes, witnesses = Cn.read_i1_frames(session_dir)
    st2 = St.step_stamp({f"{FRAMES_DIRNAME}/{P.SELECTED_FRAMES_NAME}":
                         frames_dir / P.SELECTED_FRAMES_NAME,
                         f"{FRAMES_DIRNAME}/{P.WITNESS_FRAMES_NAME}":
                         frames_dir / P.WITNESS_FRAMES_NAME},
                        {**common, "params": _jsonable_params(icfg.content)})
    run2, why2, d2 = decide(old_state, "content", st2, force,
                            lambda: _content_problem(session_dir), marker_problem)
    content: Optional[Dict[str, Any]] = None
    content_step: Dict[str, Any] = {"ran": False, "reason": why2, "skipped": False,
                                    "differences": d2}
    if not run2:
        log(f"{LOG_TAG} content: skipped ({why2}: same keyframes / witnesses, parameters, "
            f"code and environment) — reusing {Cn.CONTENT_TAGS_NAME}")
        content = Cn.load_content(session_dir)
    elif skip_content:
        stale = Cn.content_tags_path(session_dir).exists()
        reason = (f"{REASON_SKIP_CONTENT}: I2 not run in this pass (it would have run: "
                  f"{why2})")
        log(f"{LOG_TAG} content: skipped ({reason})"
            + (f" — WARNING: the existing {Cn.CONTENT_TAGS_NAME} predates the current "
               f"keyframes / witnesses and is STALE" if stale else ""))
        state["steps"]["content"] = {
            "done": False, "skipped": True, "reason": reason, "stamp": dict(st2),
            "run_config_sha256": frozen_sha, "stale_artifact": stale,
        }
        timing["steps"]["content"] = {"finished_at": _now_iso(), "elapsed_s": 0.0,
                                      "skipped": True}
        write_state(session_dir, state)
        write_timing(session_dir, timing)
        content_step = {"ran": False, "reason": REASON_SKIP_CONTENT, "skipped": True,
                        "would_have_run": why2, "stale_artifact": stale, "differences": d2}
    else:
        _said(log, "content", why2, d2)
        log(f"{LOG_TAG} content: enabled={icfg.content.enabled}, backend={icfg.content.backend}")
        if icfg.content.enabled and before_content is not None:
            before_content()
        t0 = time.monotonic()
        content = Cn.run_content(session_dir, keyframes, witnesses, icfg.content,
                                 tagger=tagger, segmenter=segmenter, log=log, heartbeat_s=hb,
                                 before_sam3=before_sam3, cancelled=cancelled)
        state["steps"]["content"] = _entry(
            st2, [Cn.content_tags_path(session_dir)], session_dir,
            {"enabled": content["enabled"],
             "tagged": content["summary"]["tagged"],
             "parse_failures": content["parse_failures"],
             "n_exclusion_masks": content["exclusion_masks"]["n_frames_written"]}, frozen_sha)
        timing["steps"]["content"] = _timing_entry(t0)
        write_state(session_dir, state)
        write_timing(session_dir, timing)
        content_step = {"ran": True, "reason": why2, "skipped": False, "differences": d2}
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
    timing["elapsed_s"] = round(time.monotonic() - t_run, 3)
    timing["finished_at"] = _now_iso()
    write_timing(session_dir, timing)
    if state != old_state:            # e.g. only the recorded run configuration changed; an
        write_state(session_dir, state)   # identical pass leaves the marker's bytes alone
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
        "run_config_sha256": frozen_sha,
        "frames_manifest": frames_manifest,
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
            "timing": str(timing_path(session_dir)),
        },
        "elapsed_s": round(time.monotonic() - t_run, 3),
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m intake.run",
        description="Intake: I0 quality features → I1 parallax keyframes + witness frames → "
                    "I2 VLM content tags + SAM3 exclusion masks, resumed by stamp.")
    ap.add_argument("--session", required=True,
                    help="session directory (frames in <session>/frames/<frame:06d>.jpg)")
    ap.add_argument("--force", action="store_true",
                    help="re-run every step even when its marker matches")
    ap.add_argument("--skip-content", action="store_true",
                    help="leave I2 out of this pass (no VLM, no SAM3; content_tags.json not "
                         "written)")
    args = ap.parse_args(argv)
    # the configuration of this job: the session's frozen copy when there is one, else the
    # server's — frozen now, this command is the job's start (point 69)
    icfg, _sha = cli_intake_config(Path(args.session), log=print)
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
