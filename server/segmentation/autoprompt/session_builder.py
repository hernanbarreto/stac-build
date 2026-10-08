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
# image sent, finish_reason, tokens, what was rescued from a cut answer), every integer
# cut-off records its margin, the effective prompt text travels with its sha and origin, no
# clock enters the file (times go to vlm_analysis.timing.json), and the file's bytes never
# follow a hash seed.
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
                           n_max: int) -> dict[str, list[str]]:
    """Per SAM3 prompt, its ORIGINS in the order SAM3 should try them when the bare
    category confirms nothing: the visual descriptions the VLM gave its objects (most
    frequent first, each carrying the CATEGORY — `with_category`), then the names merged
    into it (same name / synonym merge)."""
    pset = set(phrases)
    descs: dict[str, Counter] = {}
    alias: dict[str, Counter] = {}
    for fu in understanding.per_frame:
        for o in dict.fromkeys(fu.objects):
            c0 = understanding.merged.get(o, o)
            c = synonyms.get(c0, c0)
            if c not in pset:
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
                             generated: str | None = None) -> dict[str, dict]:
    """Per SAM3 prompt, its ShapeR description (``object_captioner.shape_caption``,
    source ``concept``) — USER 2026-10-01: the pass that prepares the prompts also
    prepares the ShapeR descriptions. Aggregated like the fallback prompts: every
    call's ``shape`` entry for a kind folded into the prompt it became (same name,
    then the merge pass), the MOST FREQUENT wording wins, ties go to the first
    seen. A prompt no call described has no entry — nothing is invented.
    ``generated`` is the caption's stable id (never a clock — point 86); None lets
    ``shape_caption`` derive it from the caption's own content."""
    from segmentation.object_captioner import shape_caption
    pset = set(phrases)
    votes: dict[str, Counter] = {}
    fields_of: dict[tuple[str, str], dict] = {}
    first: dict[tuple[str, str], int] = {}
    n = 0
    for fu in understanding.per_frame:
        for o in dict.fromkeys(fu.objects):
            c0 = understanding.merged.get(o, o)
            c = synonyms.get(c0, c0)
            if c not in pset:
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
    ``semantic.types._encode_image`` re-encodes a PIL image with, and the Pillow / libjpeg
    build doing it (``intake.stamps.pillow_record``) — the pixels the VLM sees depend on
    them, so they are recorded with every run."""
    import inspect
    from semantic.types import _encode_image
    sig = inspect.signature(_encode_image).parameters
    rec = {"format": sig["fmt"].default, "quality": sig["quality"].default}
    try:
        from intake.stamps import pillow_record
        rec["pillow"] = pillow_record()
    except Exception as e:  # noqa: BLE001 — recorded as what it is: unreadable
        rec["pillow"] = {"error": f"{type(e).__name__}: {e}"}
    return rec


def vlm_stage_stamp(session_dir: Path, keyframes: list[str], config_sections: dict) -> dict:
    """The stamp of a VLM analysis (point 82): the BYTES of every keyframe the VLM is shown
    (and frames/selected_frames.json, the list they came from, when it exists), the code
    that turns answers into prompts and encodes the images, and every configuration
    section that decides (``config_sections``, named — the effective prompt by its sha256
    and origin, the sampling, the bounds, the merge keys, the served engine's identity)."""
    import repro
    from semantic import _parse as semantic_parse
    from semantic import backends as semantic_backends
    from semantic import types as semantic_types
    from segmentation import object_captioner
    from . import consolidate_prompts, coverage_sample, detector, scene_understanding
    from . import session_builder, vlm_sampling
    frames_dir = Path(session_dir) / "frames"
    inputs = {f"frames/{fn}": frames_dir / fn for fn in keyframes}
    sel = frames_dir / "selected_frames.json"
    if sel.exists():
        inputs["frames/selected_frames.json"] = sel
    code = [scene_understanding, consolidate_prompts, session_builder, vlm_sampling,
            coverage_sample, detector, object_captioner, semantic_types, semantic_backends,
            semantic_parse]
    return repro.stamp(inputs=inputs, code=code, config=dict(config_sections))


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
        self.reuse_vocabulary = bool(cfg.get("reuse_vocabulary", True))
        self.confidence_threshold = cfg.get("confidence_threshold", 0.5)
        self.iou_threshold = cfg.get("association_iou_threshold", 0.25)
        self.assumed_depth_m = cfg.get("assumed_depth_m", 5.0)
        self.use_depth = cfg.get("use_depth", True)
        self.max_keyframes = cfg.get("max_keyframes", 0)  # 0 = all
        self.vocab: Vocabulary = load_vocabulary(vocab_path or cfg.get("vocabulary_path"))

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
        sections = {
            "vlm_sampling": asdict(load_vlm_sampling(self._config)),
            "max_sam3_prompts": load_max_sam3_prompts(self._config),
            "understand": {"enabled": bool(self.understand_enabled),
                           "cover": bool(self.understand_cover),
                           "cover_voxel_m": self.understand_cover_voxel_m,
                           "cover_overlap": self.understand_cover_overlap,
                           "max_tokens": int(load_object_captions(self._config).understand_max_tokens)},
            "understand_prompt": {"sha256": prec["sha256"], "source": prec["source"]},
            "merge": {"merge_synonyms": bool(self.merge_synonyms),
                      "merge_max_phrases_per_call": self.cfg.get("merge_max_phrases_per_call"),
                      "merge_max_tokens": self.cfg.get("merge_max_tokens")},
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
        }
        return vlm_stage_stamp(self.session_dir, kf, sections)

    def _reusable_analysis(self, stamp: dict) -> tuple[dict | None, str]:
        """The session's vlm_analysis.json when its stamp equals ``stamp`` — reused WHOLE
        (prompts, fallbacks, shape descriptions, census): point 82. Otherwise (None, why)."""
        import repro
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
        diffs = repro.check_stamp(doc.get("stamp"), stamp)
        if diffs:
            return None, "its stamp differs: " + "; ".join(diffs[:6]) + (
                f" (+{len(diffs) - 6} more)" if len(diffs) > 6 else "")
        return doc, "stamp identical"

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
        timing: dict = {}
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
            print(f"[autoprompt] vlm_analysis.json NOT reused ({why}) — deriving everything "
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
        if self.understand_enabled:
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
            for fr in plan.frames:
                fn = fr["file"]
                img = Image.open(self.frames_dir / fn).convert("RGB")
                views = [(None, None)] + tile_boxes(img.width, img.height, vcfg.tile_rows,
                                                    vcfg.tile_cols, vcfg.tile_overlap_frac)
                for tid, box in views:
                    view = img if box is None else crop_for_vlm(img, box)
                    rec = {"frame": fr["frame"], "file": fn,
                           "keyframe_index": fr["keyframe_index"],
                           "position": fr["position"], "tile": tid,
                           "box": list(box) if box else None}
                    fu = understand_frame(client, view, fr["frame"], tile=tid,
                                          max_tokens=understand_max_tokens,
                                          prompt=self.understand_prompt, record=rec)
                    rec.update(parsed=fu is not None, n_objects=len(fu.objects) if fu else 0)
                    calls.append(rec)
                    if fu:
                        fus.append(fu)
                prog(2 + int(20 * len(calls) / max(1, plan.n_calls)),
                     f"understanding {fn} ({len(views)} view(s), {len(calls)}/"
                     f"{plan.n_calls} calls)")
            timing["understanding_s"] = round(time.time() - t0, 1)
            n_bad = sum(1 for c in calls if not c["parsed"])
            n_cut = sum(1 for c in calls if c.get("truncated"))
            n_salv = sum(1 for c in calls if c.get("salvaged"))
            print(f"[autoprompt] VLM understanding: {len(calls)} call(s) in "
                  f"{timing['understanding_s']:.0f} s; {n_bad} answer(s) did not parse; "
                  f"{n_cut} cut at max_tokens, {n_salv} of them rescued (point 88)")
            understanding = aggregate(fus)
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
            synonyms: dict[str, str] = {}
            merge_calls: list[dict] = []
            merge_bound = int(self.cfg["merge_max_phrases_per_call"]) if self.merge_synonyms else None
            if self.merge_synonyms and len(phrases) > 1:
                from .consolidate_prompts import merge_synonyms
                from .scene_understanding import _head_noun
                prog(23, f"merging synonyms over {len(phrases)} names")
                t1 = time.time()
                synonyms = merge_synonyms(
                    client, understanding.scene_type if understanding else "", phrases,
                    _head_noun, max_phrases_per_call=merge_bound,
                    max_tokens=int(self.cfg["merge_max_tokens"]),
                    log=lambda m: print(f"[autoprompt] {m}"), calls=merge_calls)
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
                                    "margin": merge_bound - len(targets or [])}
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
                    log=lambda m: print(f"[autoprompt] {m}"))
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
                                                       int(self.cfg["sam3_fallback_max"]))
                                if understanding is not None else {})
            # THE SHAPER DESCRIPTIONS (USER 2026-10-01: "el VLM, para preparar los prompts,
            # pasa SAM3 y las descripciones para ShapeR"): per prompt, the description the
            # understanding gave that kind — every projected instance of the concept
            # inherits it (segmentation/pipeline.py, source 'concept') until the per-object
            # pass after the certification refines it.
            shape_descriptions = (build_shape_descriptions(understanding, phrases, synonyms)
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
                        synonyms=synonyms),
                    reused=False, prompt_bound=prompt_bound, decisions=decisions,
                    merge_calls=merge_calls),
                "stamp": stamp,
            }
            vlm_path = self.output_dir / VLM_ANALYSIS_NAME
            _write_json(vlm_path, vlm_analysis)
            _write_json(self.output_dir / VLM_TIMING_NAME,
                        dict(timing, n_calls=len(calls), n_merge_calls=len(merge_calls)))
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
                reused=False),
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
                       synonyms: dict | None = None) -> list[dict]:
        """What became of EVERY phrase the VLM proposed (USER 2026-09-29: no
        concept may disappear without a recorded reason). One entry per distinct
        phrase, with the calls that proposed it and exactly one fate:
          · ``prompt``        — it is a SAM3 prompt itself;
          · ``merged``        — folded into the SAME NAME (``same_name_key``), the
                                carrier is the prompt;
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
                               f"the merge pass (VLM) judged it the same thing as "
                               f"'{carrier}'" if same_as is not None else
                               f"same name as '{carrier}' — differs only in case, "
                               f"spacing, punctuation, a leading article / count word "
                               f"or a plural ending (same_name_key)"))
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
                       merge_calls: list[dict] | None = None) -> dict:
        """The VLM half of output/segmentation_census.json, written INTO the
        contract SAM3 reads (vlm_analysis.json) so it can never describe
        another run's prompts. Since 2026-10-08 every call carries the sha1 of the
        image sent, its finish_reason and what a cut answer still yielded (points
        88 / 94), the merge calls are listed, and ``decisions`` records every integer
        cut-off with its margin (point 89)."""
        return {
            "version": 2,
            "origin": "vlm_proposed",
            "sampling": plan.to_dict() if plan is not None else None,
            "calls": calls,
            "n_calls": len(calls),
            "n_parsed": sum(1 for c in calls if c["parsed"]),
            "n_truncated": sum(1 for c in calls if c.get("truncated")),
            "n_salvaged": sum(1 for c in calls if c.get("salvaged")),
            "image_encoding": image_encoding_record(),
            "merge_calls": list(merge_calls or []),
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
