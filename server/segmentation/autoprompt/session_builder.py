# STAC-Builder — Auto-prompter orchestrator (Phase 1).
#
# Turns raw keyframes into a pre-populated segmentation session the human
# reviews/corrects in the Segmentation Manager — replacing manual prompting as
# the default path. Also the headless masklet PRODUCER for Phase R.
#
# Flow: keyframes -> Qwen3-VL grounded detection -> geometric temporal
# association (BA poses) -> confidence gating (dubious -> review queue, never
# silently dropped) -> the (prompt, frame_map) contract the existing SAM3 batch
# pipeline consumes (written to output/vlm_analysis.json) -> optionally run SAM3
# to emit segmentation.json + seg_masks.npz.
#
# DETERMINISM (docs/plan_determinismo.md points 82, 86, 87, 88, 89, 94, 97, 159 —
# 2026-10-08): vlm_analysis.json carries the STAMP of everything it was derived from
# (repro.stamp: the bytes of every keyframe shown, the effective understanding prompt, the
# merge / bound / fallback keys, the sampling, the grouping rule, the served engine's
# identity, the code); when a later run of the session computes the SAME stamp the whole
# file is reused and the understanding is NOT re-run — when it differs, everything is
# derived again and the log names what changed. Every VLM call is recorded (the sha1 of the
# image sent and the encoder that made those bytes — the client's Pillow / libjpeg-turbo
# build and JPEG quality, point 94, also a section of the stamp —, finish_reason, tokens,
# what was rescued from a cut answer), every integer cut-off records its margin, the
# effective prompt text travels with its sha and origin, no clock enters the file (times go
# to vlm_analysis.timing.json), and the file's bytes never follow a hash seed. The VLM
# worker asks ``reuse_awaits_engine_identity`` / ``analysis_reusable`` BEFORE launching an
# engine: an analysis reusable in every stamped input but the engine's identity is reused
# under the identity computed without a launch (semantic.serve.expected_identity).
# SINCE 2026-10-09 the file also carries the UNDERSTANDING stamp (the part of the stamp the
# per-image calls depend on — :func:`understanding_stamp`): when the full stamp differs but
# the understanding stamp is identical (a change to the merge, its keys, the fallbacks …) and
# the file holds every per-image result, the per-image calls are NOT re-run — the saved
# results are aggregated again and only the merge pass and what follows it are derived
# ("understanding reused, merge re-derived", declared in the log, the file and the census).
# SINCE 2026-10-09 (USER: "Mandemos la mayor cantidad que podamos") the per-image calls and the
# merge's question chains go to the engine SEVERAL AT ONCE (semantic.parallel: as many as the
# engine's KV cache holds without preempting one — the first per-image call alone, to measure a
# call's prompt tokens), and their results are taken IN THE SEQUENTIAL ORDER: the records, the
# census, the logs and every decision are those of the one-at-a-time loop. A sample of the calls
# (the user's rule of judges, 5 at 0.95) is then re-asked ALONE and compared byte for byte; the
# bound, how it was derived and that comparison are in the census (``concurrency``,
# ``self_check``; the merge's in ``decisions.merge_strategy``). A mismatch is declared, never a
# failure.
#
# PROVENANCE: ours. Every label/box is vlm_proposed; masks come from SAM3;
# geometry/tools measure. Reuses the existing run_segmentation machinery.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .associate import Instance, associate_detections
from .detector import Detection, GroundedDetector
from .vocabulary import Vocabulary, load_vocabulary


def _is_structural(name: str) -> bool:
    from .consolidate_prompts import is_structural
    return is_structural(name)

VLM_ANALYSIS_NAME = "vlm_analysis.json"
VLM_TIMING_NAME = "vlm_analysis.timing.json"      # the run's clock, outside the compared file
CONCEPTS_NAME = "autoprompt_concepts.json"


@dataclass
class AutoPromptResult:
    n_keyframes: int
    n_detections: int
    n_instances: int
    n_accepted: int
    n_review: int
    prompt: str
    frame_map: dict[str, list[str]]
    vlm_analysis_path: str
    review_queue_path: str
    instances_path: str
    sam3_ran: bool = False
    per_class_counts: dict[str, int] = field(default_factory=dict)
    scene_type: str = ""
    reused: bool = False                 # vlm_analysis.json taken whole under its stamp

    def summary(self) -> str:
        return (
            f"scene='{self.scene_type}' keyframes={self.n_keyframes} "
            f"detections={self.n_detections} instances={self.n_instances} "
            f"(accepted={self.n_accepted}, review={self.n_review}) "
            f"classes={self.per_class_counts}"
        )


def _frame_num(filename: str) -> int:
    return int(os.path.splitext(os.path.basename(filename))[0])


def _write_json(path: Path, doc) -> None:
    """Every JSON product of this stage is written atomically (point 149)."""
    from atomic_io import atomic_write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, doc, indent=2, ensure_ascii=False)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def with_category(description: str, category: str) -> str:
    """The retry phrase SAM3 gets: the VLM's visual description WITH the category it
    describes — USER 2026-10-01 (pccr): the bare adjectives of a 'window' fallback
    ("dark, rectangular, closed") describe a door just as well, and SAM3 segmented one.
    Commas go (SAM3 reads a noun phrase, not a list); the category is appended unless the
    description already names it ("metal support column" stays as it is)."""
    words = [w for w in description.replace(",", " ").split() if w]
    cat_words = category.lower().split()
    text = " ".join(words)
    low = [w.lower() for w in words]
    if cat_words and all(w in low for w in cat_words):
        return text
    return f"{text} {category}".strip()


def build_fallback_prompts(understanding, phrases: list[str], synonyms: dict,
                           n_max: int, parts: dict | None = None) -> dict[str, list[str]]:
    """Per SAM3 prompt, its ORIGINS in the order SAM3 should try them when the bare
    category confirms nothing: the visual descriptions the VLM gave its objects (most
    frequent first, each carrying the CATEGORY — `with_category`), then the names merged
    into it (same name / synonym merge).

    A name the merge pass folded into its whole as a PART (``parts``: {part: whole} —
    USER 2026-10-09: "si tengo mil baldosas que forman el piso el objeto es piso no
    baldosa") is never a fallback, and neither is its description: retrying 'floor' as
    'tile' asks SAM3 for exactly the thousand objects the fold exists to prevent, and a
    wheel's 'black round rubber' describes the wheel, not the whole."""
    pset = set(phrases)
    parts = parts or {}
    descs: dict[str, Counter] = {}
    alias: dict[str, Counter] = {}
    for fu in understanding.per_frame:
        for o in dict.fromkeys(fu.objects):
            c0 = understanding.merged.get(o, o)
            c = synonyms.get(c0, c0)
            if c not in pset or c0 in parts:
                continue
            de = (getattr(fu, "descriptions", {}) or {}).get(o)
            if de and de != c:
                de = with_category(de, c)
                if de != c:
                    descs.setdefault(c, Counter())[de] += 1
            if o != c:
                alias.setdefault(c, Counter())[o] += 1
    out: dict[str, list[str]] = {}
    for c in phrases:
        cand = [d for d, _ in (descs.get(c) or Counter()).most_common()]
        cand += [a for a, _ in (alias.get(c) or Counter()).most_common() if a not in cand]
        if cand:
            out[c] = cand[:int(n_max)]
    return out


def build_shape_descriptions(understanding, phrases: list[str], synonyms: dict,
                             generated: str | None = None,
                             parts: dict | None = None) -> dict[str, dict]:
    """Per SAM3 prompt, its ShapeR description (``object_captioner.shape_caption``,
    source ``concept``) — USER 2026-10-01: the pass that prepares the prompts also
    prepares the ShapeR descriptions. Aggregated like the fallback prompts: every
    call's ``shape`` entry for a kind folded into the prompt it became (same name,
    then the merge pass), the MOST FREQUENT wording wins, ties go to the first
    seen. A prompt no call described has no entry — nothing is invented.
    ``generated`` is the caption's stable id (never a clock — point 86); None lets
    ``shape_caption`` derive it from the caption's own content. The shape of a name
    folded into its whole as a PART (``parts``: {part: whole}, USER 2026-10-09) never
    votes for the whole: a wheel's cylinder is not the shape of what it carries."""
    from segmentation.object_captioner import shape_caption
    pset = set(phrases)
    parts = parts or {}
    votes: dict[str, Counter] = {}
    fields_of: dict[tuple[str, str], dict] = {}
    first: dict[tuple[str, str], int] = {}
    n = 0
    for fu in understanding.per_frame:
        for o in dict.fromkeys(fu.objects):
            c0 = understanding.merged.get(o, o)
            c = synonyms.get(c0, c0)
            if c not in pset or c0 in parts:
                continue
            f = (getattr(fu, "shapes", {}) or {}).get(o)
            if not f:
                continue
            # the kind is named by its PROMPT ('columns' folded into 'column')
            f = dict(f, category=c)
            text = shape_caption(f, c, "concept", generated="-")["caption"]
            votes.setdefault(c, Counter())[text] += 1
            fields_of.setdefault((c, text), f)
            first.setdefault((c, text), n)
            n += 1
    out: dict[str, dict] = {}
    for c in phrases:
        cnt = votes.get(c)
        if not cnt:
            continue
        best = sorted(cnt, key=lambda t: (-cnt[t], first[(c, t)]))[0]
        out[c] = shape_caption(fields_of[(c, best)], c, "concept", generated=generated)
    return out


def image_encoding_record() -> dict:
    """How the client encodes the images it sends (point 94): the JPEG format and quality
    ``semantic.types._encode_image`` re-encodes a PIL image with, and the Pillow / libjpeg-
    turbo build doing it (``intake.stamps.pillow_record`` — the intake's own record of the
    same library, reused) — the pixels the VLM sees depend on them, so they enter the
    analysis' STAMP, every per-call record and the census. ONE definition for the two VLM
    consumers: ``segmentation.object_captioner.client_image_encoding_record``. A Pillow that
    cannot be identified RAISES (the stage fails naming it): the record is part of the stamp
    and a stamp of an unknown encoder says nothing."""
    from segmentation.object_captioner import client_image_encoding_record
    return client_image_encoding_record()


def image_encoder_summary(rec: dict) -> dict:
    """The compact form of :func:`image_encoding_record` every per-call record carries
    beside the sha1 of the image sent (point 94): format, quality, Pillow version and
    libjpeg-turbo build — enough to name the encoder of that call without the full record."""
    pil = rec.get("pillow") or {}
    return {"format": rec.get("format"), "quality": rec.get("quality"),
            "pillow": pil.get("version"), "libjpeg_turbo": pil.get("libjpeg_turbo")}


def vlm_stage_stamp(session_dir: Path, keyframes: list[str], config_sections: dict) -> dict:
    """The stamp of a VLM analysis (point 82): the BYTES of every keyframe the VLM is shown
    (and frames/selected_frames.json, the list they came from, when it exists), the code
    that turns answers into prompts and encodes the images, and every configuration
    section that decides (``config_sections``, named — the effective prompt by its sha256
    and origin, the sampling, the bounds, the merge keys, the served engine's identity)."""
    import repro
    from semantic import _parse as semantic_parse
    from semantic import backends as semantic_backends
    from semantic import parallel as semantic_parallel
    from semantic import types as semantic_types
    from segmentation import object_captioner
    from . import consolidate_prompts, coverage_sample, detector, scene_understanding
    from . import session_builder, vlm_sampling
    frames_dir = Path(session_dir) / "frames"
    inputs = {f"frames/{fn}": frames_dir / fn for fn in keyframes}
    sel = frames_dir / "selected_frames.json"
    if sel.exists():
        inputs["frames/selected_frames.json"] = sel
    # semantic.parallel (2026-10-09): the dispatcher that sends several calls at once and hands
    # their results back in the sequential order — the orchestration of this file, stamped
    # with it (and, like the loop, not part of the understanding stamp: see below)
    code = [scene_understanding, consolidate_prompts, session_builder, vlm_sampling,
            coverage_sample, detector, object_captioner, semantic_types, semantic_backends,
            semantic_parse, semantic_parallel]
    return repro.stamp(inputs=inputs, code=code, config=dict(config_sections))


# THE UNDERSTANDING STAMP (2026-10-09): the part of the stamp the PER-IMAGE calls depend on.
# A change to the merge (consolidate_prompts.py, its keys, the bound, the fallbacks, this
# orchestrator) used to re-run every per-image call — 295 on pccr, 1 h 45 — because the
# whole analysis was reused only under an identical stamp. The per-image results
# (scene_understanding.per_frame, census.calls, census.sampling) depend on: the keyframes'
# bytes (every input), the effective understanding prompt, the sampling, the understanding
# call's max_tokens and the cover keys ('understand'), the backend and the path, the served
# engine's identity, the client's image encoder, and the CODE of the per-image path — the
# parser and the aggregation (scene_understanding), the plan and the crops (vlm_sampling,
# coverage_sample), the salvage of a cut answer (detector), the shape entry's parser and the
# encoder record (object_captioner: fields_from_shape_entry, client_image_encoding_record),
# the client (semantic types / backends / _parse).
# DECLARED LIMIT: the loop that walks the plan's frames × views lives in this file
# (AutoPrompter.run), which also holds everything after the understanding; it is not part of
# the understanding stamp. What it sends is fixed by the stamped plan and crop code, and every
# saved call carries the sha1 of the image it sent. The same holds for the dispatcher that,
# since 2026-10-09, sends several of those calls at once (semantic.parallel — USER: "Mandemos la
# mayor cantidad que podamos"): it decides WHEN a call is sent, never WHAT; the engine is batch
# invariant (its identity, the 'service' section, is stamped) and every run re-asks a sample
# alone and records the byte-for-byte comparison in the census (``self_check``).
# The understanding stamp is a PROJECTION of the full stamp (same inputs, a subset of its code
# files and of its config sections, digested the same way as repro.stamp), so it is derived
# exactly from the full stamp of an analysis written before the split (pccr 2026-10-09 05:02).
UNDERSTANDING_SECTIONS = ("vlm_sampling", "understand", "understand_prompt", "backend",
                          "prompts_only", "max_keyframes", "service", "image_encoding")


def _understanding_code_keys() -> list[str]:
    """The repo-relative keys (as repro.stamp writes them) of the per-image path's code."""
    import repro
    from semantic import _parse as semantic_parse
    from semantic import backends as semantic_backends
    from semantic import types as semantic_types
    from segmentation import object_captioner
    from . import coverage_sample, detector, scene_understanding, vlm_sampling
    code = [scene_understanding, vlm_sampling, coverage_sample, detector, object_captioner,
            semantic_types, semantic_backends, semantic_parse]
    return sorted(repro.stamp(code=code)["code"])


def understanding_stamp(full: dict | None) -> dict | None:
    """The understanding stamp of a full analysis stamp (:func:`vlm_stage_stamp`): every input,
    the per-image path's code files and :data:`UNDERSTANDING_SECTIONS` — digested like a
    repro.stamp, so :func:`repro.check_stamp` compares two of them and names every difference.
    A code file or section the full stamp lacks is left out (check_stamp then names it as
    new). None when ``full`` is not a stamp."""
    import repro
    if not isinstance(full, dict) or not all(isinstance(full.get(k), dict)
                                             for k in ("inputs", "code", "config")):
        return None
    body = {"stamp_version": full.get("stamp_version"),
            "inputs": dict(full["inputs"]),
            "code": {k: full["code"][k] for k in _understanding_code_keys() if k in full["code"]},
            "config": {k: full["config"][k] for k in UNDERSTANDING_SECTIONS
                       if k in full["config"]}}
    body["sha256"] = repro.sha256_json(body)
    return body


# what one saved per-image result must carry to be aggregated again (FrameUnderstanding)
_PER_FRAME_KEYS = ("frame_id", "tile", "scene_type", "summary", "objects", "descriptions",
                   "shapes")


class AutoPrompter:
    def __init__(
        self,
        session_dir: str | Path,
        output_dir: str | Path,
        backend: str = "qwen_local",
        config: dict | None = None,
        vocab_path: str | Path | None = None,
    ):
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir)
        self.frames_dir = self.session_dir / "frames"
        cfg = (config or {}).get("autoprompt", {}) if config else {}
        self.cfg = cfg
        # SIMPLE pipeline flag: understanding-only prompts (rich phrases → SAM3),
        # no per-keyframe grounded detection, no boxes, no association.
        self.prompts_only = bool((((config or {}).get("reconstruction", {}) or {})
                                  .get("simple", {}) or {}).get("enabled", False))
        self.backend_name = cfg.get("backend", backend)
        self._config = config or {}
        # the session's own understanding prompt when the user saved one (USER
        # 2026-10-05, Autosegment window); the shipped prompt otherwise
        from segmentation.autoprompt.autosegment import vlm_prompt_for
        self.understand_prompt, self.understand_prompt_overridden = vlm_prompt_for(self.output_dir)
        self.understand_enabled = cfg.get("understand", True)
        self.understand_cover = bool(cfg.get("understand_cover", True))
        self.understand_cover_voxel_m = float(cfg.get("understand_cover_voxel_m", 0.10))
        self.understand_cover_overlap = float(cfg.get("understand_cover_overlap", 0.50))
        self.consolidate_prompts = bool(cfg.get("consolidate_prompts", True))
        self.consolidate_passes = int(cfg.get("consolidate_passes", 3))
        # USER 2026-09-29: the second VLM pass fuses the names that mean the same
        # thing BEFORE the SAM3 bound (strict key, no hidden default)
        if cfg and "merge_synonyms" not in cfg:
            raise KeyError("config.yaml is missing 'autoprompt.merge_synonyms' (true: a second "
                           "VLM pass fuses the names that mean the same thing)")
        self.merge_synonyms = bool(cfg.get("merge_synonyms")) if cfg else False
        # USER 2026-10-15: parts fold only into support surfaces, same kind = same head noun,
        # structure always (consolidate_prompts.merge_policy; every key required when cfg exists)
        from .consolidate_prompts import merge_policy
        self.merge_policy = merge_policy(cfg)
        # a merge key REMOVED from the configuration and still present fails naming it (a
        # leftover would look like a decision; the repo's convention, vlm_sampling.REMOVED_KEYS)
        from .consolidate_prompts import REMOVED_MERGE_KEYS
        for old, now in REMOVED_MERGE_KEYS.items():
            if cfg and old in cfg:
                raise KeyError(f"'autoprompt.{old}' was REMOVED (2026-10-09) — delete it from "
                               f"the configuration: {now}")
        self.reuse_vocabulary = bool(cfg.get("reuse_vocabulary", True))
        self.confidence_threshold = cfg.get("confidence_threshold", 0.5)
        self.iou_threshold = cfg.get("association_iou_threshold", 0.25)
        self.assumed_depth_m = cfg.get("assumed_depth_m", 5.0)
        self.use_depth = cfg.get("use_depth", True)
        self.max_keyframes = cfg.get("max_keyframes", 0)  # 0 = all
        self.vocab: Vocabulary = load_vocabulary(vocab_path or cfg.get("vocabulary_path"))
        # the client's image encoder record (point 94), read once by the first stamp
        self._image_encoding: dict | None = None

    # ── keyframe discovery ──────────────────────────────────────────
    def _keyframe_files(self, explicit: list[str] | None) -> list[str]:
        if explicit:
            return explicit
        sel = self.frames_dir / "selected_frames.json"
        if sel.exists():
            data = json.load(open(sel))
            files = data.get("selected_files") or []
            if files:
                out = sorted(files)
            else:
                out = sorted(f for f in os.listdir(self.frames_dir) if f.endswith(".jpg"))
        else:
            out = sorted(f for f in os.listdir(self.frames_dir) if f.endswith(".jpg"))
        if self.max_keyframes and len(out) > self.max_keyframes:
            idx = np.linspace(0, len(out) - 1, self.max_keyframes).astype(int)
            out = [out[i] for i in idx]
        return out

    # ── adaptive VLM sampling (camera-path coverage) ─────────────────
    def _adaptive_sample(self, files: list[str], cam) -> list[str]:
        """Pick the keyframes the VLM detector actually sees, ADAPTIVE to the
        scan instead of a fixed count: a frame is kept when the camera moved
        ≥ min_spacing_m or rotated ≥ min_rotation_deg since the last kept one
        (a new viewpoint = a new detection opportunity; 600 frames of the same
        wall add nothing but VLM latency). SAM3 propagates the masklets to
        every frame afterwards, so identity/coverage do not depend on the VLM
        seeing each image. Falls back to an even subsample when no poses.
        Scene understanding keeps its own (smaller) sample."""
        acfg = self.cfg.get("adaptive_sampling", {})
        if not acfg.get("enabled", True):
            return files
        min_frames = int(acfg.get("min_frames", 12))
        max_frames = int(acfg.get("max_frames", 0))          # 0 = no ceiling
        spacing_m = float(acfg.get("min_spacing_m", 0.4))
        rot_deg = float(acfg.get("min_rotation_deg", 18.0))
        no_pose_target = int(acfg.get("no_pose_target", 48))
        if len(files) <= min_frames:
            return files

        def _even(seq: list[str], n: int) -> list[str]:
            if len(seq) <= n:
                return seq
            idx = np.linspace(0, len(seq) - 1, n).astype(int)
            return [seq[i] for i in sorted(set(idx))]

        pose_map = cam.pose_map if cam is not None else {}
        posed = [(fn, pose_map.get(_frame_num(fn))) for fn in files]
        n_posed = sum(1 for _f, p in posed if p is not None)
        if n_posed < min_frames:
            out = _even(files, no_pose_target)
            print(f"[autoprompt] adaptive sampling: no usable poses — even "
                  f"subsample {len(files)} → {len(out)} keyframes")
            return out

        cos_thr = np.cos(np.radians(rot_deg))
        kept: list[str] = []
        last_t = last_R = None
        for fn, pose in posed:
            if pose is None:
                continue
            T = np.asarray(pose, float)
            t, R = T[:3, 3], T[:3, :3]
            if last_t is None:
                kept.append(fn); last_t, last_R = t, R
                continue
            moved = np.linalg.norm(t - last_t) >= spacing_m
            cos_a = (np.trace(last_R.T @ R) - 1.0) / 2.0
            turned = cos_a < cos_thr
            if moved or turned:
                kept.append(fn); last_t, last_R = t, R
        if files and files[-1] not in kept:
            kept.append(files[-1])           # always close the trajectory

        if len(kept) < min_frames:
            kept = _even(files, min_frames)
        if max_frames and len(kept) > max_frames:
            kept = _even(kept, max_frames)
        print(f"[autoprompt] adaptive sampling: {len(files)} keyframes → "
              f"{len(kept)} VLM frames (spacing {spacing_m} m / {rot_deg}°)")
        return kept

    # ── camera geometry (optional) ──────────────────────────────────
    def _load_camera(self):
        try:
            from segmentation.session_io import _load_camera_source
            return _load_camera_source(self.session_dir, self.output_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[autoprompt] no camera source (association falls back): {e}")
            return None

    def _depth_provider(self):
        if not self.use_depth:
            return None
        # DA3/VGGT per-frame depth, if present (omega_run / da3_run / results_output)
        candidates = [
            self.output_dir / "omega_run" / "results_output",
            self.output_dir / "da3_run" / "results_output",
            self.output_dir / "results_output",
        ]
        depth_dir = next((c for c in candidates if c.is_dir()), None)
        if depth_dir is None:
            return None

        def provider(fid: int):
            p = depth_dir / f"frame_{fid}.npz"
            if not p.exists():
                return None
            try:
                arr = np.load(p)
                key = "depth" if "depth" in arr else list(arr.keys())[0]
                from segmentation.session_io import correct_depth
                return (correct_depth(arr[key].astype(np.float32), fid,
                                      self.output_dir), None)
            except Exception:
                return None

        return provider

    # ── the stamp of this run's analysis (point 82) ──────────────────
    def _prompt_record(self) -> dict:
        """The effective understanding prompt, its sha256 and its origin (point 97)."""
        return {"text": self.understand_prompt,
                "sha256": prompt_sha256(self.understand_prompt),
                "source": "session" if self.understand_prompt_overridden else "shipped"}

    def _stamp(self, kf: list[str], service: dict | None) -> dict:
        from .scene_understanding import GROUPING_VERSION
        from .vlm_sampling import load_max_sam3_prompts, load_vlm_sampling
        from segmentation.object_captioner import load_object_captions
        prec = self._prompt_record()
        # the client's image encoder, read ONCE per run (point 94): it is a stamp section,
        # every per-call record names it and the census record carries it whole
        if self._image_encoding is None:
            self._image_encoding = image_encoding_record()
        sections = {
            "vlm_sampling": asdict(load_vlm_sampling(self._config)),
            "max_sam3_prompts": load_max_sam3_prompts(self._config),
            "understand": {"enabled": bool(self.understand_enabled),
                           "cover": bool(self.understand_cover),
                           "cover_voxel_m": self.understand_cover_voxel_m,
                           "cover_overlap": self.understand_cover_overlap,
                           "max_tokens": int(load_object_captions(self._config).understand_max_tokens)},
            "understand_prompt": {"sha256": prec["sha256"], "source": prec["source"]},
            # the merge answer's bound is no key since 2026-10-09: it is derived from the
            # served context (the 'service' section) by the stamped code (consolidate_prompts)
            "merge": {"merge_synonyms": bool(self.merge_synonyms),
                      "merge_max_phrases_per_call": self.cfg.get("merge_max_phrases_per_call")},
            "consolidate": {"consolidate_prompts": bool(self.consolidate_prompts),
                            "consolidate_passes": int(self.consolidate_passes)},
            "sam3_fallback_max": self.cfg.get("sam3_fallback_max"),
            "grouping": GROUPING_VERSION,
            "backend": self.backend_name,
            "prompts_only": bool(self.prompts_only),
            "max_keyframes": self.max_keyframes,
            # the engine that answers: weights, versions, flags, card (semantic.serve's
            # identity); None when no identity was handed in (a CLI without the stage's engine)
            "service": dict(service) if service else None,
            # the client that encodes what the engine sees (point 94): Pillow / libjpeg-turbo
            # build, JPEG quality — another build re-derives the analysis, declared
            "image_encoding": dict(self._image_encoding),
        }
        return vlm_stage_stamp(self.session_dir, kf, sections)

    # ── reuse without an engine (points 82 / 155 — the VLM worker's no-launch path) ──
    def reuse_awaits_engine_identity(self, keyframe_files: list[str] | None = None
                                     ) -> tuple[bool, str]:
        """Whether the session's vlm_analysis.json could be reused WHOLE once the identity
        of the engine this job would launch is known: every part of the stamp but the
        ``service`` section (the keyframes' bytes, the prompt, the sampling, the merge keys,
        the encoder, the code) already matches. The VLM worker asks this BEFORE computing
        that identity — the identity costs the weights' sha256 (minutes) and is computed
        only when it is the last thing standing between the session and a reuse. Returns
        (True, why) when only the engine's identity is still to be verified, (False, why)
        when the analysis is to be derived again whatever the engine is."""
        if not (self.prompts_only and self.understand_enabled):
            return False, "not the SIMPLE understanding path (nothing is reused there)"
        import repro
        doc, why = self._analysis_document()
        if doc is None:
            return False, why
        kf = self._keyframe_files(keyframe_files)
        # the stamp this job computes with the SAVED analysis' own engine section in place
        # of the one still to be verified: the structural difference (repro.check_stamp —
        # the same test ``run`` makes) then names exactly what differs BEYOND the engine.
        # Never a parse of the log line ``_reusable_analysis`` writes: a wording change
        # there would have turned every reuse into a launch without a failure.
        diffs = repro.check_stamp(doc.get("stamp"), self._stamp(kf, doc.get("service")))
        if diffs:
            return False, "its stamp differs beyond the engine: " + self._diffs_text(diffs)
        return True, "every stamped input but the engine's identity matches"

    def analysis_reusable(self, service: dict | None,
                          keyframe_files: list[str] | None = None) -> tuple[dict | None, str]:
        """The session's vlm_analysis.json when it would be reused WHOLE by :meth:`run`
        under the engine identity ``service`` (None, why) otherwise — the same test ``run``
        makes, without a client."""
        if not (self.prompts_only and self.understand_enabled):
            return None, "not the SIMPLE understanding path (nothing is reused there)"
        kf = self._keyframe_files(keyframe_files)
        return self._reusable_analysis(self._stamp(kf, service))

    def _analysis_document(self) -> tuple[dict | None, str]:
        """The session's vlm_analysis.json when a reuse is possible AT ALL — reuse_vocabulary
        on, the file there, readable, carrying a prompt — (None, why) otherwise. Whether its
        stamp matches is the next question (:meth:`_reusable_analysis`,
        :meth:`reuse_awaits_engine_identity`)."""
        p = self.output_dir / VLM_ANALYSIS_NAME
        if not self.reuse_vocabulary:
            return None, "autoprompt.reuse_vocabulary is false"
        if not p.exists():
            return None, "no vlm_analysis.json in the session"
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            return None, f"vlm_analysis.json unreadable ({e})"
        if not isinstance(doc, dict) or not str(doc.get("prompt") or "").strip():
            return None, "vlm_analysis.json carries no prompt"
        return doc, "on disk"

    @staticmethod
    def _diffs_text(diffs: list[str]) -> str:
        return "; ".join(diffs[:6]) + (f" (+{len(diffs) - 6} more)" if len(diffs) > 6 else "")

    def _reusable_analysis(self, stamp: dict) -> tuple[dict | None, str]:
        """The session's vlm_analysis.json when its stamp equals ``stamp`` — reused WHOLE
        (prompts, fallbacks, shape descriptions, census): point 82. Otherwise (None, why)."""
        import repro
        doc, why = self._analysis_document()
        if doc is None:
            return None, why
        diffs = repro.check_stamp(doc.get("stamp"), stamp)
        if diffs:
            return None, "its stamp differs: " + self._diffs_text(diffs)
        return doc, "stamp identical"

    # ── the per-image understanding reused when only what follows it changed (2026-10-09) ──
    @staticmethod
    def _saved_understanding_stamp(doc: dict) -> tuple[dict | None, str]:
        """The understanding stamp of a saved analysis: the one it recorded, or — written
        before the split — the projection of its full stamp (exact: the full stamp holds every
        part of it, file by file and section by section)."""
        rec = doc.get("understanding_stamp")
        if isinstance(rec, dict):
            return rec, "recorded"
        return understanding_stamp(doc.get("stamp")), "derived from the saved full stamp's parts"

    @staticmethod
    def _saved_understanding(doc: dict) -> tuple[dict | None, str]:
        """The per-image results a saved analysis holds, ready to be aggregated again —
        ``{"understanding": SceneUnderstanding, "calls": [...], "sampling": {...}}`` — or
        (None, why) when the document does not hold every one of them: the SIMPLE path's
        analysis, one ``scene_understanding.per_frame`` entry per parsed call with every field
        of a FrameUnderstanding, the call records and the sampling plan of the census, and a
        per-frame list that aggregates again into the saved list (``objects``,
        ``merged_same_name``) — a document that does not is not what its stamp says."""
        from .scene_understanding import FrameUnderstanding, aggregate
        if doc.get("source") != "qwen3vl_autoprompt_simple":
            return None, f"the saved analysis is not the SIMPLE path's ({doc.get('source')!r})"
        su = doc.get("scene_understanding")
        census = doc.get("census")
        if not isinstance(su, dict) or not isinstance(census, dict):
            return None, "the saved analysis holds no scene_understanding / census"
        per, calls, sampling = su.get("per_frame"), census.get("calls"), census.get("sampling")
        if not isinstance(per, list) or not isinstance(calls, list) or not isinstance(sampling, dict):
            return None, ("the saved analysis does not hold the per-image results "
                          "(scene_understanding.per_frame, census.calls, census.sampling)")
        if not all(isinstance(c, dict) for c in calls):
            return None, "a saved call record is not a record"
        fus = []
        for i, e in enumerate(per):
            if not isinstance(e, dict) or any(k not in e for k in _PER_FRAME_KEYS):
                return None, (f"saved per-frame result #{i} lacks a field of a "
                              f"FrameUnderstanding ({', '.join(_PER_FRAME_KEYS)})")
            if not (isinstance(e["objects"], list) and isinstance(e["descriptions"], dict)
                    and isinstance(e["shapes"], dict)):
                return None, f"saved per-frame result #{i} is malformed"
            fus.append(FrameUnderstanding(
                frame_id=int(e["frame_id"]), scene_type=str(e["scene_type"]),
                summary=str(e["summary"]), objects=[str(o) for o in e["objects"]],
                tile=e["tile"], descriptions=dict(e["descriptions"]), shapes=dict(e["shapes"])))
        n_parsed = sum(1 for c in calls if c.get("parsed"))
        if n_parsed != len(fus):
            return None, (f"the saved calls record {n_parsed} parsed answer(s) and the saved "
                          f"per-frame results are {len(fus)}")
        understanding = aggregate(fus)
        if (understanding.objects != su.get("objects")
                or dict(understanding.merged) != dict(su.get("merged_same_name") or {})):
            return None, ("the saved per-frame results do not aggregate again into the saved "
                          "list — the document is not what its stamp says")
        return {"understanding": understanding, "calls": [dict(c) for c in calls],
                "sampling": dict(sampling)}, (f"{len(calls)} call record(s), {len(fus)} "
                                              f"per-frame result(s)")

    def _reusable_understanding(self, stamp: dict) -> tuple[dict | None, str]:
        """The session's per-image understanding when its UNDERSTANDING stamp equals the one of
        ``stamp`` (:func:`understanding_stamp`) and the saved document holds every per-image
        result (:meth:`_saved_understanding`): the per-image calls are not re-run and only
        what follows them is derived again. Otherwise (None, why)."""
        import repro
        doc, why = self._analysis_document()
        if doc is None:
            return None, why
        saved, origin = self._saved_understanding_stamp(doc)
        diffs = repro.check_stamp(saved, understanding_stamp(stamp))
        if diffs:
            return None, f"its understanding stamp ({origin}) differs: " + self._diffs_text(diffs)
        parts, why = self._saved_understanding(doc)
        if parts is None:
            return None, why
        parts.update(origin=origin, saved_stamp=str((doc.get("stamp") or {}).get("sha256") or ""),
                     understanding_stamp=saved)
        return parts, f"understanding stamp identical ({origin}); {why}"

    def understanding_awaits_engine_identity(self, keyframe_files: list[str] | None = None
                                             ) -> tuple[bool, str]:
        """For the VLM worker's log when the analysis is NOT reusable whole: whether its
        per-image understanding would be, once the engine this job launches shows the identity
        the analysis was made with — every part of the understanding stamp but ``service``
        matches and the document holds every per-image result. The engine is launched either
        way (the merge pass needs it); :meth:`run` makes the decision on the live identity."""
        if not (self.prompts_only and self.understand_enabled):
            return False, "not the SIMPLE understanding path (nothing is reused there)"
        import repro
        doc, why = self._analysis_document()
        if doc is None:
            return False, why
        kf = self._keyframe_files(keyframe_files)
        saved, origin = self._saved_understanding_stamp(doc)
        now = understanding_stamp(self._stamp(kf, doc.get("service")))
        diffs = repro.check_stamp(saved, now)
        if diffs:
            return False, (f"its understanding stamp ({origin}) differs beyond the engine: "
                           + self._diffs_text(diffs))
        parts, why = self._saved_understanding(doc)
        if parts is None:
            return False, why
        return True, f"every input of the per-image calls but the engine's identity matches ({why})"

    # ── several calls in flight (USER 2026-10-09: "Mandemos la mayor cantidad que podamos") ──
    def _engine_kv(self, client) -> dict | None:
        """The engine's KV cache (semantic.parallel.engine_kv_cache — what bounds how many calls
        are in flight), read ONCE per run through whichever client asks first; None (declared:
        one call at a time) when the client has no such record."""
        if not getattr(self, "_kv_read", False):
            from semantic.parallel import engine_kv_cache
            self._kv = engine_kv_cache(client, log=lambda m: print(f"[autoprompt] {m}"))
            self._kv_read = True
        return getattr(self, "_kv", None)

    def _served_context(self, service: dict | None) -> tuple[int, str] | None:
        """The served context (consolidate_prompts.served_context_len: the engine's launch line,
        else the configuration) — the upper bound of a call nothing measured; None when it
        cannot be read (such a call then runs alone)."""
        from .consolidate_prompts import MergeError, served_context_len
        try:
            return served_context_len(service, self._config, self.backend_name)
        except MergeError:
            return None

    def _self_check_understanding(self, conc: dict | None, answers: dict, where: dict, client,
                                  max_tokens: int) -> dict:
        """THE SELF-CHECK of the understanding pass (USER 2026-10-09: the answers must be the
        sequential run's, byte for byte): a sample of the calls that ran with others in flight
        — the user's rule of judges, min_judge_closures(correction_graph.graph.
        heldout_confidence), spread evenly over the plan's order, the first call (made alone)
        excluded — is re-asked ALONE after the pass, nothing else in flight, and each answer's
        text compared with the pass's. A mismatch, or a re-ask that fails, is DECLARED (log +
        census), never a failure (USER: "que no falle, que lo reporte sin fallar")."""
        from semantic.parallel import (SELF_CHECK_SUFFIX, AnswerCapture, not_checked,
                                       sample_positions, self_check, self_check_judges)
        from .scene_understanding import understand_frame
        from .vlm_sampling import crop_for_vlm
        observed = (conc or {}).get("observed") or {}
        if conc is None or len(answers) <= 1:
            return not_checked("one call or none — no call ran beside another")
        if int(observed.get("max_in_flight") or 0) <= 1:
            return not_checked("every call ran alone (" + str(conc.get("one_at_a_time")
                                                            or "the bound admitted one at a time")
                               + ") — nothing to compare")
        n, judges = self_check_judges(self._config)
        if n is None:
            print(f"[autoprompt] [concurrency] ⚠ DECLARED: the self-check is not run — "
                  f"{judges['error']}")
            return not_checked(judges["error"], judges)
        sample = sample_positions(sorted(i for i in answers if i > 0), n)

        def _alone(i: int):
            fr, tid, box, sent_sha1 = where[i]
            img = Image.open(self.frames_dir / fr["file"]).convert("RGB")
            view = img if box is None else crop_for_vlm(img, box)
            rec: dict = {}
            cap = AnswerCapture(client, consumer_suffix=SELF_CHECK_SUFFIX)
            understand_frame(cap, view, fr["frame"], tile=tid, max_tokens=max_tokens,
                             prompt=self.understand_prompt, record=rec)
            if rec.get("image_sha1") != sent_sha1:
                raise RuntimeError(f"the image encoded for the re-ask (sha1 "
                                   f"{rec.get('image_sha1')}) is not the one the pass sent "
                                   f"({sent_sha1}) — nothing to compare")
            return cap.last

        return self_check(sample, answers, _alone,
                          describe=lambda i: {"call": i, "frame": where[i][0]["frame"],
                                              "tile": where[i][1]},
                          judges=judges, what="understanding",
                          log=lambda m: print(f"[autoprompt] {m}"))

    # ── main run ────────────────────────────────────────────────────
    def run(
        self,
        keyframe_files: list[str] | None = None,
        run_sam3: bool = False,
        on_progress: Callable[[int, str], None] | None = None,
        service: dict | None = None,
    ) -> AutoPromptResult:
        """``service`` is the verified identity of the engine answering (semantic.service.
        job_engine): it enters the stamp and the file, so an analysis made by another
        engine (other weights, flags, card) is never taken for this one."""
        from semantic.client import get_semantic_client

        def prog(pct, msg):
            if on_progress:
                on_progress(pct, msg)

        kf = self._keyframe_files(keyframe_files)
        stamp = self._stamp(kf, service)
        ustamp = understanding_stamp(stamp)
        timing: dict = {}
        reused_u: dict | None = None      # the per-image understanding, reused (2026-10-09)
        # ── the whole analysis, reused under an identical stamp (point 82) ─────
        if self.prompts_only and self.understand_enabled:
            doc, why = self._reusable_analysis(stamp)
            if doc is not None:
                phrases = [c.strip() for c in str(doc.get("prompt", "")).split(";") if c.strip()]
                su = doc.get("scene_understanding") or {}
                print(f"[autoprompt] ♻ vlm_analysis.json REUSED whole (stamp "
                      f"{stamp['sha256'][:12]}: {why}) — {len(phrases)} SAM3 prompt(s), "
                      f"the understanding is not re-run (autoprompt.reuse_vocabulary)")
                prog(100, f"prompts ready (reused): {len(phrases)} concepts → SAM3")
                return AutoPromptResult(
                    n_keyframes=len(kf), n_detections=0, n_instances=0,
                    n_accepted=0, n_review=0, prompt=";".join(phrases), frame_map={},
                    vlm_analysis_path=str(self.output_dir / VLM_ANALYSIS_NAME),
                    review_queue_path=str(self.output_dir / "autoprompt_review_queue.json"),
                    instances_path=str(self.output_dir / "autoprompt_instances.json"),
                    per_class_counts={p: 1 for p in phrases},
                    scene_type=str(su.get("scene_type") or ""), reused=True)
            # ── only what FOLLOWS the per-image calls changed: they are not re-run ──
            reused_u, why_u = self._reusable_understanding(stamp)
            if reused_u is not None:
                print(f"[autoprompt] ♻ understanding reused, merge re-derived — "
                      f"vlm_analysis.json is not reusable whole ({why}), but its "
                      f"{len(reused_u['calls'])} per-image call(s) are ({why_u}; understanding "
                      f"stamp {ustamp['sha256'][:12]}): they are NOT re-run; aggregate → "
                      f"same-name fold → merge pass → prompts → shape descriptions → census are "
                      f"derived again under stamp {stamp['sha256'][:12]}")
            else:
                print(f"[autoprompt] vlm_analysis.json NOT reused ({why}); the per-image "
                      f"understanding is not reusable either ({why_u}) — deriving everything "
                      f"again under stamp {stamp['sha256'][:12]}")

        # BA poses drive the adaptive sampling and the association of the
        # GROUNDED-DETECTION path only; the SIMPLE path never reads them (it used
        # to compute them anyway and log "(48 to the VLM)" for frames nobody sent)
        cam = None
        kf_vlm = kf
        if not self.prompts_only:
            cam = self._load_camera()
            kf_vlm = self._adaptive_sample(kf, cam)
            prog(2, f"auto-prompt: {len(kf)} keyframes ({len(kf_vlm)} to the detector)")

        client = get_semantic_client(backend=self.backend_name, consumer="phase1.autoprompt")
        detector = GroundedDetector(client, self.vocab)

        # ── Step 1: understand the scene (what is this? what's in it?) ──
        understanding = None
        targets: list[str] | None = None
        plan = None                       # which frames / crops the VLM was shown
        calls: list[dict] = []            # one entry per VLM call, parsed or not
        conc: dict | None = None          # how many understanding calls were in flight (2026-10-09)
        u_check: dict | None = None       # the byte-identity self-check of those calls
        self._kv_read, self._kv = False, None     # the engine's KV cache, read once per run
        if reused_u is not None:
            # the per-image results of the saved analysis, aggregated again (the same
            # aggregate, stamped): the calls, the plan they followed and what each answered
            # are the saved ones, declared in the file and the census
            understanding = reused_u["understanding"]
            calls = reused_u["calls"]
            plan = reused_u["sampling"]
            timing["understanding_reused"] = True
            prog(22, f"understanding reused: {len(calls)} per-image call(s) not re-run")
        elif self.understand_enabled:
            from .scene_understanding import understand_frame, aggregate
            from .vlm_sampling import (crop_for_vlm, load_vlm_sampling, plan_vlm_frames,
                                       tile_boxes, walk_chainage)
            # WHAT THE VLM NEVER SEES, IT CANNOT NAME — and in the SIMPLE
            # pipeline these phrases ARE the SAM3 prompts, so a frame left out
            # here is an object left out of the segmentation entirely.
            # The frames are spread UNIFORMLY ALONG THE WALK (vlm_sampling.py):
            # by walked chainage when intake/walk.json covers the keyframes, by
            # keyframe index otherwise, each optionally shown also as a grid of
            # enlarged crops so small objects get named. It used to be the
            # coverage cover with an 8-frame linspace fallback — and at the
            # intake there is no cloud, so pccr 2026-09-29 got the 8.
            from semantic.parallel import (AnswerCapture, Job, concurrency_plan, cost_of,
                                           plan_line, run_ordered)
            vcfg = load_vlm_sampling(self._config)
            preselected = None
            if self.understand_cover:
                from .coverage_sample import cover_keyframes
                preselected = cover_keyframes(
                    self.output_dir, self.session_dir, kf, _frame_num,
                    voxel_m=self.understand_cover_voxel_m,
                    max_overlap=self.understand_cover_overlap,
                    log=lambda m: print(f"[autoprompt] {m}"))
                if preselected is None:
                    print("[autoprompt] coverage unavailable — the walk-uniform "
                          "sampling decides the VLM frames")
            chainage, axis_reason = walk_chainage(self.session_dir, kf)
            plan = plan_vlm_frames(kf, vcfg, chainage=chainage, axis_reason=axis_reason,
                                   preselected=preselected)
            print(f"[autoprompt] {plan.summary()}")
            prog(2, f"auto-prompt: {len(kf)} keyframes → {len(plan.frames)} to the VLM, "
                    f"{plan.n_calls} call(s)")
            # The answer now carries one ShapeR shape entry per kind (USER 2026-10-01),
            # and a CUT answer keeps its complete objects (point 88). Its bound is
            # declared, never a literal.
            from segmentation.object_captioner import load_object_captions
            understand_max_tokens = int(load_object_captions(self._config).understand_max_tokens)
            fus = []
            t0 = time.time()
            # the encoder of every image sent, named in each call's record beside the
            # image's sha1 (point 94) — one summary, the same for the whole run
            encoder = image_encoder_summary(self._image_encoding)
            # SEVERAL CALLS IN FLIGHT (USER 2026-10-09: "Mandemos la mayor cantidad que
            # podamos"; semantic.parallel): the FIRST call goes alone and measures the prompt
            # tokens of a call (usage.prompt_tokens — exact for every view of the same pixel
            # size: the prompt text is the same for all), the rest are admitted in the plan's
            # order while their KV blocks fit the engine's cache; the results are delivered in
            # that order, so the records, the census, the progress lines and the aggregation are
            # the sequential loop's. Afterwards a sample is re-asked ALONE and compared byte for
            # byte (declared in the census, never a failure).
            answers: dict[int, str | None] = {}       # call index → the answer's text
            where: dict[int, tuple] = {}              # call index → (frame, tile, box, image sha1)

            def _views():
                """Every call of the plan, in the sequential loop's order: (frame record, tile,
                box, the view shown, the frame's number of views, last view of its frame)."""
                for fr in plan.frames:
                    img = Image.open(self.frames_dir / fr["file"]).convert("RGB")
                    views = [(None, None)] + tile_boxes(img.width, img.height, vcfg.tile_rows,
                                                        vcfg.tile_cols, vcfg.tile_overlap_frac)
                    for k, (tid, box) in enumerate(views):
                        view = img if box is None else crop_for_vlm(img, box)
                        yield fr, tid, box, view, len(views), k == len(views) - 1

            def _understand(chat_client, fr, tid, box, view):
                """One understanding call (on a worker thread when several are in flight):
                its census record, its parsed result and the answer's text."""
                rec = {"frame": fr["frame"], "file": fr["file"],
                       "keyframe_index": fr["keyframe_index"],
                       "position": fr["position"], "tile": tid,
                       "box": list(box) if box else None,
                       "image_encoding": encoder}
                cap = AnswerCapture(chat_client)
                fu = understand_frame(cap, view, fr["frame"], tile=tid,
                                      max_tokens=understand_max_tokens,
                                      prompt=self.understand_prompt, record=rec)
                rec.update(parsed=fu is not None, n_objects=len(fu.objects) if fu else 0,
                           # USER 2026-10-15: what the answer said about the envelope (the
                           # 'structure' field + the structural object names) — the census
                           # lists the keyframes that named none
                           n_structural=(len([o for o in fu.objects if _is_structural(o)]) if fu else 0),
                           structure=(list(fu.structure) if fu else []))
                return rec, fu, cap.last

            def _deliver(i, out):
                (fr, tid, box, n_views, last), (rec, fu, text) = out
                calls.append(rec)
                if fu:
                    fus.append(fu)
                answers[i], where[i] = text, (fr, tid, box, rec.get("image_sha1"))
                if last:
                    prog(2 + int(20 * len(calls) / max(1, plan.n_calls)),
                         f"understanding {fr['file']} ({n_views} view(s), {len(calls)}/"
                         f"{plan.n_calls} calls)")

            def _job(item, cost):
                fr, tid, box, view, n_views, last = item
                return Job(fn=lambda: ((fr, tid, box, n_views, last),
                                       _understand(client, fr, tid, box, view)), cost=cost)

            gen = _views()
            first = next(gen, None)
            if first is not None:
                fr0, tid0, box0, view0, nv0, last0 = first
                _deliver(0, ((fr0, tid0, box0, nv0, last0),
                             _understand(client, fr0, tid0, box0, view0)))
                ref_size = view0.size
                p0 = calls[0].get("prompt_tokens")
                p0 = int(p0) if isinstance(p0, int) and not isinstance(p0, bool) else None
                kv = self._engine_kv(client)
                ctx = self._served_context(service)
                rest = max(0, int(plan.n_calls) - 1)
                foot0 = (p0 + understand_max_tokens) if p0 is not None else (ctx[0] if ctx else None)
                conc = concurrency_plan(
                    kv, [foot0] if foot0 is not None else [], n_requests=rest,
                    footprint_source=(
                        f"measured: the first call's usage.prompt_tokens {p0} + max_tokens "
                        f"{understand_max_tokens} for a view of {ref_size[0]}x{ref_size[1]} px; "
                        f"a view of another size: the served context "
                        f"{ctx[0] if ctx else 'unknown — it runs alone'}"
                        if p0 is not None else
                        f"the served context {ctx[0] if ctx else 'unknown'} (the first call "
                        f"reported no prompt_tokens)"))

                def _rest():
                    for item in gen:
                        same = p0 is not None and item[3].size == ref_size
                        foot = (p0 + understand_max_tokens) if same else (ctx[0] if ctx else None)
                        yield _job(item, cost_of(conc, foot))

                stats = run_ordered(_rest(), capacity=conc["capacity_blocks"],
                                    threads=conc["threads"],
                                    on_result=lambda i, out: _deliver(i + 1, out))
                conc["observed"] = stats
                print(f"[autoprompt] {plan_line(conc, stats)}")
            timing["understanding_s"] = round(time.time() - t0, 1)
            n_bad = sum(1 for c in calls if not c["parsed"])
            n_cut = sum(1 for c in calls if c.get("truncated"))
            n_salv = sum(1 for c in calls if c.get("salvaged"))
            print(f"[autoprompt] VLM understanding: {len(calls)} call(s) in "
                  f"{timing['understanding_s']:.0f} s; {n_bad} answer(s) did not parse; "
                  f"{n_cut} cut at max_tokens, {n_salv} of them rescued (point 88)")
            # THE SELF-CHECK (USER 2026-10-09, byte identity): a sample of the calls that ran
            # with others in flight is re-asked ALONE; the sample is the user's rule of judges
            # (min_judge_closures(correction_graph.graph.heldout_confidence) — 5 at 0.95)
            t2 = time.time()
            u_check = self._self_check_understanding(
                conc, answers, where, client, understand_max_tokens)
            timing["self_check_s"] = round(time.time() - t2, 1)
            understanding = aggregate(fus)
        if understanding is not None:
            targets = understanding.objects
            if understanding.merged:
                print(f"[autoprompt] {len(understanding.merged)} phrase(s) folded into the "
                      f"SAME NAME (case / plural / article / punctuation only): "
                      f"{dict(sorted(understanding.merged.items()))}")
            _write_json(self.output_dir / "scene_understanding.json", understanding.to_dict())
            prog(22, f"scene: {understanding.scene_type} — {len(targets)} object types understood")

        # ── SIMPLE pipeline: understanding IS the whole VLM job ──────────
        # The scene pass already produced one RICH noun phrase per object type.
        # Those phrases go straight to SAM3 as SEPARATE text prompts (the ';'
        # join below is just the file format — the segmentation pipeline splits
        # it and runs ONE SAM3 concept session per phrase). No per-keyframe
        # grounded detection, no boxes, no association: SAM3 searches, labels
        # and TRACKS each concept itself — its tracking is the identity.
        if self.prompts_only:
            phrases = [p for p in (targets or []) if p and p.strip()]
            if not phrases:
                raise RuntimeError("scene understanding produced no objects — "
                                   "cannot build SAM3 prompts")
            # ── THE MERGE PASS (USER 2026-09-29): a second VLM pass over the NAMES
            # fuses the ones that mean the same thing, BEFORE the bound — the bound
            # then spends its budget on distinct objects, not on wordings (pccr, first
            # run with every keyframe × 5 views: 1 900 names, 150 prompts spent on
            # 'white wall' / 'white painted wall' / 'beige wall' …, 'black office
            # desk' never prompted). A call that fails fails the stage (point 88).
            # Since 2026-10-09 the same pass also folds a PART / component / building
            # unit / material INTO the listed whole it belongs to (USER: "los objetos deben
            # ser el más grande posible, si tengo mil baldosas que forman el piso el objeto
            # es piso no baldosa"): the fold is in ``synonyms`` like any merged name (the
            # prompt list, the bound, the second pass's 'covered' test read it) and in
            # ``part_of`` ({part: whole}), which the census reason, the SAM3 fallbacks and
            # the ShapeR descriptions read — a part is never retried as its whole.
            synonyms: dict[str, str] = {}
            part_of: dict[str, str] = {}
            merge_calls: list[dict] = []
            merge_exec: dict = {}         # how the merge's calls went (concurrency, self-check)
            merge_bound = int(self.cfg["merge_max_phrases_per_call"]) if self.merge_synonyms else None
            context = None                # (served context in tokens, where it was read)
            if self.merge_synonyms and len(phrases) > 1:
                from .consolidate_prompts import merge_synonyms, served_context_len
                from .scene_understanding import _head_noun
                prog(23, f"merging synonyms over {len(phrases)} names")
                # the merge answer may use all the room the served context leaves after its
                # prompt (2026-10-09: a fixed 8 192 cut pccr's answer before its folds)
                context = served_context_len(service, self._config, self.backend_name)
                print(f"[autoprompt] [merge] served context {context[0]} tokens ({context[1]}): "
                      f"each merge answer is bounded by it minus its own prompt; the HTTP call "
                      f"has no read timeout (the engine's max_tokens is the bound)")
                # the merge's own client: no HTTP read timeout. The call is non-streaming,
                # so service.request_timeout_s bounds the WHOLE answer — 600 s ≈ 15 k tokens
                # on pccr's engine (~25 tok/s, 2026-10-09), below the ~21.6 k the context
                # leaves: the answer would die as a ReadTimeout before its own bound. The
                # semantic config (and the engine identity that digests it) is untouched.
                merge_client = get_semantic_client(backend=self.backend_name,
                                                   consumer="phase1.merge_synonyms",
                                                   no_read_timeout=True)
                t1 = time.time()
                # ONE question per phrase (2026-10-09): the whole list in one answer looped
                # (JSON groups) or copied every name onto itself (one line per name) on pccr.
                # The phrases' question chains go to the engine SEVERAL AT ONCE (USER
                # 2026-10-09, semantic.parallel: bounded by the engine's KV cache); the
                # decisions are applied in the list's order, as before, and a sample of first
                # questions is re-asked alone (the user's rule of judges)
                from .consolidate_prompts import classify_phrases
                from semantic.parallel import self_check_judges
                synonyms = classify_phrases(
                    merge_client, understanding.scene_type if understanding else "", phrases,
                    log=lambda m: print(f"[autoprompt] {m}"), calls=merge_calls,
                    part_of=part_of, kv=self._engine_kv(merge_client),
                    judges=self_check_judges(self._config), record=merge_exec,
                    policy=self.merge_policy)
                timing["merge_s"] = round(time.time() - t1, 1)
            # proposals per name AFTER the folds (same name, then the merge pass): the
            # order of the prompts and the margin at the bound both read it
            _props: Counter = Counter()
            for fu in (understanding.per_frame if understanding else []):
                for o in dict.fromkeys(fu.objects):
                    c = understanding.merged.get(o, o)
                    _props[synonyms.get(c, c)] += 1
            if synonyms:
                _order = {p: i for i, p in enumerate(phrases)}
                phrases = sorted((p for p in phrases if p not in synonyms),
                                 key=lambda p: (-_props[p], _order[p]))
            structural_first: list[str] = []
            if self.merge_policy.structural_always:
                # USER 2026-10-15 ("no debe olvidarse lo estructural nunca"): the envelope goes
                # FIRST, so the bound on SAM3 prompts below never cuts a structural name
                structural_first = [p for p in phrases if _is_structural(p)]
                phrases = structural_first + [p for p in phrases if not _is_structural(p)]
            # ── the BOUND on SAM3 prompts (autoprompt.max_sam3_prompts) ────
            # `aggregate` keeps every distinct name and the consolidation removes
            # none, so the prompt count follows the VLM's phrasing — and SAM3
            # time is linear in it. Past the bound the names proposed by the
            # FEWEST calls are not prompted (the understanding's own order:
            # proposals incl. same-name variants, then first seen), each one
            # declared here, in autoprompt_concepts.json and in the census — with the
            # MARGIN at the cut (point 89): the proposals of the last name admitted and of
            # the first left out (equal = a tie decided by first-seen order).
            from .vlm_sampling import load_max_sam3_prompts
            max_prompts = load_max_sam3_prompts(self._config)
            overflow: list[str] = []
            n_names = len(phrases)
            if len(phrases) > max_prompts:
                overflow = phrases[max_prompts:]
                phrases = phrases[:max_prompts]
                print(f"[autoprompt] SAM3 prompt BOUND REACHED (autoprompt.max_sam3_prompts "
                      f"= {max_prompts}): {len(overflow)} of {n_names} "
                      f"name(s) NOT prompted, the least proposed: {overflow}")
            print(f"[autoprompt] {len(phrases)} SAM3 prompt(s) (bound {max_prompts}) — "
                  f"SAM3 time is linear in this count; its stage logs the measured s/prompt")
            last_in = phrases[-1] if phrases else None
            first_out = overflow[0] if overflow else None
            prompt_bound = {"max_sam3_prompts": max_prompts,
                            "n_names": n_names,
                            "bound_reached": bool(overflow), "not_prompted": overflow,
                            "last_admitted": ({"name": last_in, "n_proposals": _props[last_in]}
                                              if last_in else None),
                            "first_excluded": ({"name": first_out, "n_proposals": _props[first_out]}
                                               if first_out else None),
                            "margin_proposals": ((_props[last_in] - _props[first_out])
                                                 if (last_in and first_out) else None)}
            votes = list((getattr(understanding, "scene_type_votes", None) or {}).items())
            decisions = {
                "merge_strategy": ({"n_phrases": n_names if not synonyms else len(_props),
                                    "n_names_before_merge": len(targets or []),
                                    "merge_max_phrases_per_call": merge_bound,
                                    "one_call": (len(targets or []) <= merge_bound),
                                    "margin": merge_bound - len(targets or []),
                                    # the answer's bound: the served context minus each
                                    # call's prompt (per call in merge_calls); a cut list
                                    # is split in halves + a final pass (2026-10-09)
                                    "served_context_len": (context[0] if context else None),
                                    "served_context_source": (context[1] if context else None),
                                    "n_calls": sum(1 for c in merge_calls if c.get("asked")),
                                    "n_cut_split": sum(1 for c in merge_calls
                                                       if c.get("split_into")),
                                    # how many question chains were in flight and the
                                    # byte-identity self-check of a sample (USER 2026-10-09)
                                    "concurrency": merge_exec.get("concurrency"),
                                    "self_check": merge_exec.get("self_check")}
                                   if merge_bound is not None else None),
                "prompt_bound": prompt_bound,
                "scene_type": {"winner": (understanding.scene_type if understanding else None),
                               "votes": dict(votes),
                               "margin": ((votes[0][1] - votes[1][1]) if len(votes) > 1
                                          else (votes[0][1] if votes else 0))},
            }
            # ── the CONSOLIDATION pass (USER 2026-09-16) ─────────────────
            # The understanding ran frame by frame and no frame ever saw the
            # others' answers, so the union carries the same object under
            # several names and entries that are only PARTS of another. Each
            # phrase becomes one SAM3 session and one segment, so both cost a
            # duplicate or an unsegmentable fragment. Deciding which words name
            # one thing is a language judgement over the WHOLE list, which is
            # exactly what a per-frame prompt can never have.
            consolidation = None
            if self.consolidate_prompts and not self.merge_synonyms and len(phrases) > 1:
                from .consolidate_prompts import consolidate
                prog(23, f"consolidating {len(phrases)} concepts")
                consolidation = consolidate(
                    client, understanding.scene_type if understanding else "",
                    phrases, max_passes=self.consolidate_passes,
                    log=lambda m: print(f"[autoprompt] {m}"), policy=self.merge_policy)
                # THE CONSOLIDATION GROUPS, IT DOES NOT DELETE (USER 2026-09-22:
                # *"tampoco segmento una puerta, de entrada una locura"* ... *"no
                # lo fuerces sino que debe ser generico"*). Every phrase the
                # understanding produced still becomes a SAM3 prompt; what the
                # pass returns is the GROUPING, which travels in
                # prompt_consolidation.json for labelling and for the report.
                #
                # WHY: deciding that two WORDS name one thing, before anything
                # has been segmented, throws away the object itself when the
                # judgement is wrong — and it was wrong on its own evidence.
                # pccr 2026-09-22, verbatim from the run's own file:
                #     glass door        <- doorway
                #     white workbench   <- white table, desk
                #     black server rack <- white server cabinet
                #     white wall        <- red painted wall section
                #     white tiled floor <- dark gray tiled floor patch
                # Four of those five differ by COLOUR in their own names. The
                # structural rescue (`is_structural`) could not help: it only
                # protects a phrase filed as somebody's PART, never one filed as
                # somebody's ALIAS, so 'doorway' went with no warning.
                # The identity question is not a language question — it is
                # settled downstream on the GEOMETRY, by the mutual-overlap test
                # over the instances' own points (`segmentation.dedupe_overlap`,
                # measured on all 2,926 instance pairs of this very scan) and by
                # the fragment merge. Words propose; the cloud decides.
                # The cost is SAM3 time, linear in the number of concepts.
                _grouped = list(consolidation.objects)
                _write_json(self.output_dir / "prompt_consolidation.json",
                            consolidation.to_dict())
                _folded = [p for p in phrases if p not in set(_grouped)]
                if _folded:
                    print(f"[autoprompt] [consolidate] {len(_grouped)} group(s) for "
                          f"labelling; ALL {len(phrases)} concepts still go to SAM3 "
                          f"— kept: {_folded}")
                prog(25, f"{len(_grouped)} group(s), {len(phrases)} concepts to SAM3")
            # NOTHING CHANGES IN SILENCE, AND THE LIST IS REPRODUCIBLE
            # (USER 2026-09-22: *"vocabulario, no debe cambiar en silencio,
            # debe ser lo mas determinista posible, debe ser reproducible"*).
            # The record below is the session's vocabulary: the raw union, the
            # grouping, every pass and every failure — stamped (point 82 / 159) with
            # the same stamp as vlm_analysis.json, which is the ONE file reused.
            _rec = {
                "version": 2,
                "raw": list(targets or []),
                "prompts": list(phrases),
                "prompt_source": "vlm_proposed",
                "groups": (consolidation.to_dict() if consolidation else None),
                "passes": (consolidation.passes if consolidation else 0),
                "not_prompted_bound": overflow,
                "synonyms": synonyms,
                # the subset of ``synonyms`` that is a PART folded into its whole (USER
                # 2026-10-09), sorted: the file's bytes never follow the answer's order
                "parts_folded": dict(sorted(part_of.items())),
                # USER 2026-10-15: the policy the merge ran under, the pieces of a non-support
                # object kept as their own prompt, the structural names prompted first
                "merge_policy": self.merge_policy.to_dict(),
                "parts_kept_detail": sorted(c["phrase"] for c in merge_calls
                                            if isinstance(c, dict) and c.get("decision") == "detail_kept"),
                "structural_first": structural_first,
                "stamp": stamp["sha256"],
                "scene_type": (understanding.scene_type if understanding else None),
            }
            if consolidation is not None and consolidation.passes == 0:
                _rec["warning"] = ("the consolidation pass returned nothing "
                                   "parseable — the list is the raw union")
                print(f"[autoprompt] ⚠ {_rec['warning']}")
            _write_json(self.output_dir / CONCEPTS_NAME, _rec)
            prompt = ";".join(phrases)
            # THE FALLBACKS (USER 2026-09-30: "si hay un prompt que no produjo nada es que a
            # SAM3 le falta detalle — a ese se lo puede probar con los que se originó"): per
            # prompt, the visual descriptions the VLM gave its objects (most frequent first),
            # then the names merged into it; SAM3 tries them in order only when the bare
            # category confirms nothing (segmentation.pipeline._run_sam3_batched).
            fallback_prompts = (build_fallback_prompts(understanding, phrases, synonyms,
                                                       int(self.cfg["sam3_fallback_max"]),
                                                       parts=part_of)
                                if understanding is not None else {})
            # THE SHAPER DESCRIPTIONS (USER 2026-10-01: "el VLM, para preparar los prompts,
            # pasa SAM3 y las descripciones para ShapeR"): per prompt, the description the
            # understanding gave that kind — every projected instance of the concept
            # inherits it (segmentation/pipeline.py, source 'concept') until the per-object
            # pass after the certification refines it.
            shape_descriptions = (build_shape_descriptions(understanding, phrases, synonyms,
                                                           parts=part_of)
                                  if understanding is not None else {})
            _undescribed = [p for p in phrases if p not in shape_descriptions]
            print(f"[autoprompt] ShapeR descriptions: {len(shape_descriptions)} of "
                  f"{len(phrases)} prompt(s) described by the understanding"
                  + (f"; without one (concept only, no caption inherited): {_undescribed}"
                     if _undescribed else ""))
            vlm_analysis = {
                "source": "qwen3vl_autoprompt_simple",
                "backend": self.backend_name,
                # the engine that answered (semantic.serve identity, point 155) — None only
                # when the run had no engine of its own (a CLI)
                "service": (dict(service) if service else None),
                "understand_prompt": self._prompt_record(),
                "scene_understanding": understanding.to_dict() if understanding else None,
                "consolidation": consolidation.to_dict() if consolidation else None,
                "prompt": prompt,
                "prompt_source": "vlm_proposed",
                "fallback_prompts": fallback_prompts,
                "shape_descriptions": shape_descriptions,
                "frame_map": {},          # empty → SAM3 runs every phrase on ALL frames
                "boxes": {},              # NO box seeds, ever, in this mode
                "instances": [],
                "review_queue": [],
                "dubious_labels": [],
                "thresholds": {},
                # what the VLM looked at and what became of every phrase it
                # proposed — read by segmentation/census.py after SAM3
                "census": self._census_record(
                    plan, calls, self._concept_fates(
                        understanding, phrases, reused=False,
                        consolidation=consolidation, bounded_out=prompt_bound,
                        synonyms=synonyms, part_of=part_of),
                    reused=False, prompt_bound=prompt_bound, decisions=decisions,
                    merge_calls=merge_calls, image_encoding=self._image_encoding,
                    understanding_reused=reused_u is not None,
                    concurrency=conc, self_check=u_check),
                "stamp": stamp,
                # the part of the stamp the per-image calls depend on (2026-10-09): a later run
                # whose full stamp differs but whose understanding stamp is this one re-derives
                # only what follows the per-image calls
                "understanding_stamp": ustamp,
                # the per-image calls of THIS file were made by an earlier run (declared): the
                # analysis whose results were aggregated again, and how its stamp was read
                "understanding_reused": ({"from_stamp": reused_u["saved_stamp"],
                                          "understanding_stamp": ustamp["sha256"],
                                          "understanding_stamp_origin": reused_u["origin"],
                                          "n_calls_not_rerun": len(calls)}
                                         if reused_u is not None else None),
            }
            vlm_path = self.output_dir / VLM_ANALYSIS_NAME
            _write_json(vlm_path, vlm_analysis)
            _write_json(self.output_dir / VLM_TIMING_NAME,
                        dict(timing, n_calls=(0 if reused_u is not None else len(calls)),
                             n_merge_calls=sum(1 for c in merge_calls if c.get("asked"))))
            review_path = self.output_dir / "autoprompt_review_queue.json"
            _write_json(review_path, {"instances": []})
            inst_path = self.output_dir / "autoprompt_instances.json"
            _write_json(inst_path, {"accepted": [], "review": []})
            prog(100, f"prompts ready: {len(phrases)} rich concepts → SAM3")
            print(f"[autoprompt] SIMPLE: {len(phrases)} rich concept prompts for SAM3: "
                  f"{phrases}")
            return AutoPromptResult(
                n_keyframes=len(kf), n_detections=0, n_instances=0,
                n_accepted=0, n_review=0, prompt=prompt, frame_map={},
                vlm_analysis_path=str(vlm_path),
                review_queue_path=str(review_path),
                instances_path=str(inst_path),
                per_class_counts={p: 1 for p in phrases},
                scene_type=(understanding.scene_type if understanding else ""),
            )

        # ── Step 2: understanding-driven detection (segment everything) ──
        # only the adaptively-sampled keyframes go through the VLM; SAM3
        # propagates the resulting masklets to every frame afterwards
        detections_by_frame: dict[int, list[Detection]] = {}
        img_wh: dict[int, tuple[int, int]] = {}
        fid_to_file: dict[int, str] = {}
        n_det = 0
        for i, fn in enumerate(kf_vlm):
            img = Image.open(self.frames_dir / fn).convert("RGB")
            fid = _frame_num(fn)
            fid_to_file[fid] = fn
            img_wh[fid] = img.size
            dets = detector.detect(img, frame_id=fid, targets=targets)
            detections_by_frame[fid] = dets
            n_det += len(dets)
            prog(24 + int(40 * (i + 1) / max(1, len(kf_vlm))),
                 f"detected {len(dets)} in {fn} ({i + 1}/{len(kf_vlm)})")

        # geometric temporal association (reuses the poses loaded above)
        pose_map = cam.pose_map if cam else None
        K_for = (lambda f: cam.K_for(f)) if cam else None
        instances = associate_detections(
            detections_by_frame, pose_map, K_for,
            lambda f: img_wh.get(f, (1920, 1080)),
            depth_provider=self._depth_provider(),
            iou_threshold=self.iou_threshold,
            assumed_depth_m=self.assumed_depth_m,
        )
        prog(70, f"associated -> {len(instances)} instances")

        accepted, review = self._gate(instances)
        # Consolidate near-synonym labels BEFORE building the SAM3 contract: the VLM
        # freely emits "wheel"+"train_wheel", "ceiling_truss"+"ceiling_trusses", … and
        # each label becomes its own SAM3 concept pass — the same physical object then
        # gets segmented once per name (the "same tag N times" duplicates). Lexical
        # merge only (plural + shared last token); semantics untouched.
        merged = self._consolidate_labels(accepted + review)
        if merged:
            print(f"[autoprompt] vocabulary consolidated: "
                  f"{', '.join(f'{a}→{b}' for a, b in sorted(merged.items()))}")
        # Spec (item 5): dubious instances are NOT dropped — they are segmented
        # too (their masklets participate in the Phase R vote) but flagged so
        # they never generate pose residuals until validated. Labels whose
        # instances are ALL dubious are marked for the store.
        prompt, frame_map = self._build_contract(accepted + review, fid_to_file)
        accepted_labels = {i.label for i in accepted}
        dubious_labels = sorted({i.label for i in review} - accepted_labels)

        # persist the integration contract + audit artifacts
        vlm_analysis = {
            "source": "qwen3vl_autoprompt",
            "backend": self.backend_name,
            "service": (dict(service) if service else None),
            "understand_prompt": self._prompt_record(),
            "scene_understanding": understanding.to_dict() if understanding else None,
            "prompt": prompt,
            "prompt_source": "vlm_proposed",
            "frame_map": frame_map,
            # the understanding's descriptions, keyed by its own phrases; the detector's
            # labels only inherit one when they coincide with a phrase (declared)
            "shape_descriptions": (build_shape_descriptions(
                understanding, list(understanding.objects), {})
                if understanding is not None else {}),
            # extras beyond the InternVL3 contract (ignored by the SAM3 worker,
            # consumed by review UI / Phase R / audit):
            "boxes": self._boxes_by_label_frame(accepted + review, fid_to_file),
            "instances": [i.to_dict() for i in accepted],
            "review_queue": [i.to_dict() for i in review],
            # labels with ONLY dubious instances → Phase R marks them status=
            # 'dubious' (vote yes, pose residuals no, until validated)
            "dubious_labels": dubious_labels,
            "thresholds": {
                "confidence": self.confidence_threshold,
                "association_iou": self.iou_threshold,
            },
            "census": self._census_record(
                plan, calls, self._concept_fates(
                    understanding, [], reused=False, consolidation=None,
                    detection_path=True),
                reused=False, image_encoding=self._image_encoding),
            "stamp": stamp,
        }
        vlm_path = self.output_dir / VLM_ANALYSIS_NAME
        _write_json(vlm_path, vlm_analysis)
        _write_json(self.output_dir / VLM_TIMING_NAME, dict(timing, n_calls=len(calls)))

        review_path = self.output_dir / "autoprompt_review_queue.json"
        _write_json(review_path,
                    {"note": "dubious instances (below confidence threshold) — for human "
                             "review; NOT discarded. May vote in Phase R but generate no "
                             "pose residual until validated.",
                     "instances": [i.to_dict() for i in review]})

        inst_path = self.output_dir / "autoprompt_instances.json"
        _write_json(inst_path, {"accepted": [i.to_dict() for i in accepted],
                                "review": [i.to_dict() for i in review]})

        per_class: dict[str, int] = {}
        for inst in accepted:
            per_class[inst.label] = per_class.get(inst.label, 0) + 1

        sam3_ran = False
        if run_sam3 and prompt:
            prog(75, "running SAM3 to pre-populate masks...")
            self._run_sam3(prompt, frame_map, on_progress,
                           boxes_map=vlm_analysis["boxes"])
            sam3_ran = True

        prog(100, "auto-prompt complete")
        return AutoPromptResult(
            n_keyframes=len(kf), n_detections=n_det, n_instances=len(instances),
            n_accepted=len(accepted), n_review=len(review),
            prompt=prompt, frame_map=frame_map,
            vlm_analysis_path=str(vlm_path), review_queue_path=str(review_path),
            instances_path=str(inst_path), sam3_ran=sam3_ran,
            per_class_counts=per_class,
            scene_type=understanding.scene_type if understanding else "",
        )

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _concept_fates(understanding, prompts: list[str], *, reused: bool,
                       consolidation, detection_path: bool = False,
                       bounded_out: dict | None = None,
                       synonyms: dict | None = None,
                       part_of: dict | None = None) -> list[dict]:
        """What became of EVERY phrase the VLM proposed (USER 2026-09-29: no
        concept may disappear without a recorded reason). One entry per distinct
        phrase, with the calls that proposed it and exactly one fate:
          · ``prompt``        — it is a SAM3 prompt itself;
          · ``merged``        — folded into the SAME NAME (``same_name_key``), judged
                                the same thing by the merge pass, or — since
                                2026-10-09 — folded by it into its WHOLE as a part /
                                component / building unit / material (``part_of``:
                                {part: whole}; the entry then names the whole in
                                ``part_of``); the carrier is the prompt;
          · ``not_prompted``  — with the reason (the SAM3 prompt BOUND —
                                ``bounded_out``, the ``prompt_bound`` record —, a
                                reused session vocabulary, or the grounded-
                                detection path).
        The consolidation's grouping travels as evidence (``consolidation``:
        the group and whether the phrase is its name, an alias or a part) — it
        groups for labelling, it never removes a prompt."""
        if understanding is None:
            return []
        from .scene_understanding import _head_noun
        prompt_set = set(prompts)
        over = list((bounded_out or {}).get("not_prompted") or [])
        role: dict[str, dict] = {}
        if consolidation is not None:
            cd = consolidation.to_dict()
            for name in cd.get("objects") or []:
                role[name] = {"group": name, "as": "name"}
            for key, kind in (("merged", "alias"), ("parts", "part")):
                for name, members in (cd.get(key) or {}).items():
                    for m in members:
                        role[m] = {"group": name, "as": kind}
        proposed: dict[str, list] = {}
        for fu in understanding.per_frame:
            for o in dict.fromkeys(fu.objects):
                proposed.setdefault(o, []).append({"frame": fu.frame_id, "tile": fu.tile})
        out = []
        for phrase, by in proposed.items():
            carrier = understanding.merged.get(phrase, phrase)
            folded_into = (part_of or {}).get(carrier)
            same_as = (synonyms or {}).get(carrier)
            if same_as is not None:
                carrier = same_as
            rec = {"concept": phrase, "proposed_by": by, "n_proposals": len(by),
                   "head_noun": _head_noun(phrase)}
            if detection_path:
                rec.update(fate="not_prompted", prompt=None,
                           reason="grounded-detection path (reconstruction.simple off): "
                                  "the SAM3 prompts are the detector's labels, this "
                                  "phrase was one of its targets")
            elif carrier in prompt_set:
                rec.update(fate="prompt" if carrier == phrase else "merged", prompt=carrier,
                           reason=None if carrier == phrase else (
                               f"the merge pass (VLM) judged it a part, component, "
                               f"building unit or material of '{carrier}' — folded into "
                               f"that whole object (USER 2026-10-09: the object is the "
                               f"largest whole, never its pieces)"
                               if folded_into is not None else
                               f"the merge pass (VLM) judged it the same thing as "
                               f"'{carrier}'" if same_as is not None else
                               f"same name as '{carrier}' — differs only in case, "
                               f"spacing, punctuation, a leading article / count word "
                               f"or a plural ending (same_name_key)"))
                if folded_into is not None:
                    rec["part_of"] = carrier
            elif carrier in over:
                bound = int(bounded_out["max_sam3_prompts"])
                rank = bound + over.index(carrier) + 1
                rec.update(fate="not_prompted", prompt=None,
                           reason=(f"BOUND REACHED: autoprompt.max_sam3_prompts = {bound} "
                                   f"of {bounded_out.get('n_names')} names; ranked by the "
                                   f"VLM calls that proposed it, '{carrier}' is #{rank}"))
            elif reused:
                rec.update(fate="not_prompted", prompt=None,
                           reason="the session vocabulary was reused "
                                  "(autoprompt.reuse_vocabulary → output/"
                                  "vlm_analysis.json under its stamp): this run's "
                                  "understanding describes the scene, the list is the "
                                  "pinned one")
            else:
                rec.update(fate="unaccounted", prompt=None,
                           reason="no rule removed it and it is not a prompt — a bug")
            if carrier in role:
                rec["consolidation"] = role[carrier]
            out.append(rec)
        return out

    @staticmethod
    def _census_record(plan, calls: list[dict], fates: list[dict], *,
                       reused: bool, prompt_bound: dict | None = None,
                       decisions: dict | None = None,
                       merge_calls: list[dict] | None = None,
                       image_encoding: dict | None = None,
                       understanding_reused: bool = False,
                       concurrency: dict | None = None,
                       self_check: dict | None = None) -> dict:
        """The VLM half of output/segmentation_census.json, written INTO the
        contract SAM3 reads (vlm_analysis.json) so it can never describe
        another run's prompts. Since 2026-10-08 every call carries the sha1 of the
        image sent and the encoder that made those bytes, its finish_reason and what a
        cut answer still yielded (points 88 / 94), the merge calls are listed, and
        ``decisions`` records every integer cut-off with its margin (point 89).
        ``image_encoding``: the run's encoder record (the stamp's; read now when absent).
        ``plan``: the sampling plan, or its saved dict when the per-image understanding was
        reused (``understanding_reused``: the calls are the saved ones, declared).
        Since 2026-10-09 (USER: "Mandemos la mayor cantidad que podamos"): ``concurrency`` —
        how many understanding calls were in flight and how that bound was derived from the
        engine's KV cache (semantic.parallel.concurrency_plan, with what was observed) — and
        ``self_check`` — the sample re-asked ALONE and compared byte for byte, every mismatch
        listed (declared, never a failure). Both are None when no call was made (the
        understanding reused, or off). An execution record, like ``service``: the calls'
        records, their order and everything derived from them are those of the sequential
        loop."""
        return {
            "version": 2,
            "origin": "vlm_proposed",
            "sampling": (plan if isinstance(plan, dict) or plan is None else plan.to_dict()),
            "understanding_reused": bool(understanding_reused),
            "calls": calls,
            "n_calls": len(calls),
            "n_parsed": sum(1 for c in calls if c["parsed"]),
            "n_truncated": sum(1 for c in calls if c.get("truncated")),
            "n_salvaged": sum(1 for c in calls if c.get("salvaged")),
            "image_encoding": (dict(image_encoding) if image_encoding is not None
                               else image_encoding_record()),
            "merge_calls": list(merge_calls or []),
            "concurrency": concurrency,
            "self_check": self_check,
            "decisions": decisions,
            "concepts": fates,
            "vocabulary_reused": bool(reused),
            "prompt_bound": prompt_bound,
        }

    def _gate(self, instances: list[Instance]) -> tuple[list[Instance], list[Instance]]:
        accepted, review = [], []
        for inst in instances:
            thr = self.vocab.min_confidence_for(inst.label, self.confidence_threshold)
            # multi-view support is corroborating evidence: a 2+ view instance
            # is accepted at a slightly relaxed bar.
            eff_conf = inst.confidence + (0.05 if inst.n_views >= 2 else 0.0)
            (accepted if eff_conf >= thr else review).append(inst)
        return accepted, review

    @staticmethod
    def _consolidate_labels(instances: list[Instance]) -> dict[str, str]:
        """Merge NEAR-SYNONYM labels in place so each visual concept reaches SAM3
        exactly once. Two lexical rules only (semantics are never guessed):
          1. plural → singular when both exist ("ceiling_trusses" → "ceiling_truss")
          2. shared last token → the SHORTER (more generic) label wins for the
             SAM3 concept pass ("train_wheel"+"wheel" → "wheel"); the vocabulary
             stays a canonicalization overlay downstream, never a detection filter.
        Returns {old_label: new_label} for the merges applied."""
        labels = {i.label for i in instances}

        def _singular(s: str) -> str:
            # pragmatic English plural strip: trusses→truss, boxes→box, wheels→wheel
            for suf in ("sses", "xes", "ches", "shes"):
                if s.endswith(suf):
                    return s[:-2]
            return s[:-1] if s.endswith("s") and not s.endswith("ss") else s

        remap: dict[str, str] = {}
        # rule 1: plural collapses onto the existing singular
        for lb in sorted(labels):
            sg = _singular(lb)
            if sg != lb and sg in labels:
                remap[lb] = sg
        # rule 2: same last token (after plural strip) → shortest label wins
        by_token: dict[str, list[str]] = {}
        for lb in sorted(labels):
            eff = remap.get(lb, lb)
            by_token.setdefault(_singular(eff.split("_")[-1]), []).append(eff)
        for tok, group in by_token.items():
            group = sorted(set(group), key=len)
            if len(group) > 1 and tok == group[0] == _singular(group[0]):
                # only merge when the generic token itself is one of the labels
                # ("wheel" ⊂ "train_wheel"); unrelated same-suffix labels
                # ("door" vs "trapdoor" won't hit: token match is exact on "_" split)
                for other in group[1:]:
                    remap[other] = group[0]
        # resolve chains (plural → token merge)
        for k in list(remap):
            v = remap[k]
            while v in remap:
                v = remap[v]
            remap[k] = v
        if remap:
            for inst in instances:
                if inst.label in remap:
                    inst.label = remap[inst.label]
        return remap

    def _build_contract(
        self, accepted: list[Instance], fid_to_file: dict[int, str]
    ) -> tuple[str, dict[str, list[str]]]:
        frame_map: dict[str, list[str]] = {}
        for inst in accepted:
            files = frame_map.setdefault(inst.label, [])
            for m in inst.members:
                fn = fid_to_file.get(m.frame_id)
                if fn and fn not in files:
                    files.append(fn)
        for k in frame_map:
            frame_map[k] = sorted(frame_map[k])
        prompt = ";".join(sorted(frame_map.keys()))
        return prompt, frame_map

    def _boxes_by_label_frame(
        self, accepted: list[Instance], fid_to_file: dict[int, str]
    ) -> dict[str, dict[str, list]]:
        """Persist normalized xywh boxes per label per keyframe file (for
        box-seeding / review overlays). Dedup to best box per (instance,frame)."""
        out: dict[str, dict[str, list]] = {}
        for inst in accepted:
            per_frame_best: dict[int, tuple] = {}
            for m in inst.members:
                cur = per_frame_best.get(m.frame_id)
                if cur is None or m.confidence > cur[1]:
                    per_frame_best[m.frame_id] = (m.box, m.confidence)
            for fid, (box, _c) in per_frame_best.items():
                fn = fid_to_file.get(fid)
                if not fn:
                    continue
                x1, y1, x2, y2 = box
                xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]  # SAM3 xywh
                out.setdefault(inst.label, {}).setdefault(fn, []).append(
                    {"instance_id": inst.instance_id, "box_xywh": [round(c, 5) for c in xywh]}
                )
        return out

    def _run_sam3(self, prompt, frame_map, on_progress, boxes_map=None):
        from segmentation.pipeline import run_segmentation
        run_segmentation(
            str(self.frames_dir), str(self.output_dir),
            prompt=prompt, frame_map=frame_map,
            boxes_map=boxes_map,   # per-instance box seeding (SAM3 detector path)
            on_progress=(lambda pct, msg: on_progress(75 + int(pct * 0.24), msg))
            if on_progress else None,
        )
