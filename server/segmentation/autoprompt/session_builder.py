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
# PROVENANCE: ours. Every label/box is vlm_proposed; masks come from SAM3;
# geometry/tools measure. Reuses the existing run_segmentation machinery.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .associate import Instance, associate_detections
from .detector import Detection, GroundedDetector
from .vocabulary import Vocabulary, load_vocabulary


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

    def summary(self) -> str:
        return (
            f"scene='{self.scene_type}' keyframes={self.n_keyframes} "
            f"detections={self.n_detections} instances={self.n_instances} "
            f"(accepted={self.n_accepted}, review={self.n_review}) "
            f"classes={self.per_class_counts}"
        )


def _frame_num(filename: str) -> int:
    return int(os.path.splitext(os.path.basename(filename))[0])


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
        self.understand_enabled = cfg.get("understand", True)
        self.understand_sample = cfg.get("understand_sample", 8)
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
                return (arr[key].astype(np.float32), None)
            except Exception:
                return None

        return provider

    # ── main run ────────────────────────────────────────────────────
    def run(
        self,
        keyframe_files: list[str] | None = None,
        run_sam3: bool = False,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> AutoPromptResult:
        from semantic.client import get_semantic_client

        def prog(pct, msg):
            if on_progress:
                on_progress(pct, msg)

        kf = self._keyframe_files(keyframe_files)
        # BA poses drive both the adaptive VLM sampling and the association
        cam = self._load_camera()
        kf_vlm = self._adaptive_sample(kf, cam)
        prog(2, f"auto-prompt: {len(kf)} keyframes ({len(kf_vlm)} to the VLM)")

        client = get_semantic_client(backend=self.backend_name, consumer="phase1.autoprompt")
        detector = GroundedDetector(client, self.vocab)

        # ── Step 1: understand the scene (what is this? what's in it?) ──
        understanding = None
        targets: list[str] | None = None
        if self.understand_enabled:
            from .scene_understanding import understand_frame, aggregate
            sample = kf
            if self.understand_sample and len(kf) > self.understand_sample:
                idx = np.linspace(0, len(kf) - 1, self.understand_sample).astype(int)
                sample = [kf[i] for i in idx]
            fus = []
            for j, fn in enumerate(sample):
                img = Image.open(self.frames_dir / fn).convert("RGB")
                fu = understand_frame(client, img, _frame_num(fn))
                if fu:
                    fus.append(fu)
                prog(2 + int(20 * (j + 1) / max(1, len(sample))), f"understanding {fn}")
            understanding = aggregate(fus)
            targets = understanding.objects
            (self.output_dir).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "scene_understanding.json").write_text(
                json.dumps(understanding.to_dict(), indent=2, ensure_ascii=False))
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
            prompt = ";".join(phrases)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            vlm_analysis = {
                "source": "qwen3vl_autoprompt_simple",
                "backend": self.backend_name,
                "scene_understanding": understanding.to_dict() if understanding else None,
                "prompt": prompt,
                "frame_map": {},          # empty → SAM3 runs every phrase on ALL frames
                "boxes": {},              # NO box seeds, ever, in this mode
                "instances": [],
                "review_queue": [],
                "dubious_labels": [],
                "thresholds": {},
            }
            vlm_path = self.output_dir / "vlm_analysis.json"
            vlm_path.write_text(json.dumps(vlm_analysis, indent=2, ensure_ascii=False))
            review_path = self.output_dir / "autoprompt_review_queue.json"
            review_path.write_text(json.dumps({"instances": []}, indent=2))
            inst_path = self.output_dir / "autoprompt_instances.json"
            inst_path.write_text(json.dumps({"accepted": [], "review": []}, indent=2))
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        vlm_analysis = {
            "source": "qwen3vl_autoprompt",
            "backend": self.backend_name,
            "scene_understanding": understanding.to_dict() if understanding else None,
            "prompt": prompt,
            "frame_map": frame_map,
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
        }
        vlm_path = self.output_dir / "vlm_analysis.json"
        vlm_path.write_text(json.dumps(vlm_analysis, indent=2, ensure_ascii=False))

        review_path = self.output_dir / "autoprompt_review_queue.json"
        review_path.write_text(json.dumps(
            {"note": "dubious instances (below confidence threshold) — for human "
                     "review; NOT discarded. May vote in Phase R but generate no "
                     "pose residual until validated.",
             "instances": [i.to_dict() for i in review]},
            indent=2, ensure_ascii=False))

        inst_path = self.output_dir / "autoprompt_instances.json"
        inst_path.write_text(json.dumps(
            {"accepted": [i.to_dict() for i in accepted],
             "review": [i.to_dict() for i in review]},
            indent=2, ensure_ascii=False))

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
