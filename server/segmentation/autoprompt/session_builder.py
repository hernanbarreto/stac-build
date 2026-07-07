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
        prog(2, f"auto-prompt: {len(kf)} keyframes")

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

        # ── Step 2: understanding-driven detection (segment everything) ──
        detections_by_frame: dict[int, list[Detection]] = {}
        img_wh: dict[int, tuple[int, int]] = {}
        fid_to_file: dict[int, str] = {}
        n_det = 0
        for i, fn in enumerate(kf):
            img = Image.open(self.frames_dir / fn).convert("RGB")
            fid = _frame_num(fn)
            fid_to_file[fid] = fn
            img_wh[fid] = img.size
            dets = detector.detect(img, frame_id=fid, targets=targets)
            detections_by_frame[fid] = dets
            n_det += len(dets)
            prog(24 + int(40 * (i + 1) / max(1, len(kf))), f"detected {len(dets)} in {fn}")

        # geometric temporal association
        cam = self._load_camera()
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
        prompt, frame_map = self._build_contract(accepted, fid_to_file)

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
            "boxes": self._boxes_by_label_frame(accepted, fid_to_file),
            "instances": [i.to_dict() for i in accepted],
            "review_queue": [i.to_dict() for i in review],
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
            self._run_sam3(prompt, frame_map, on_progress)
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

    def _run_sam3(self, prompt, frame_map, on_progress):
        from segmentation.pipeline import run_segmentation
        run_segmentation(
            str(self.frames_dir), str(self.output_dir),
            prompt=prompt, frame_map=frame_map,
            on_progress=(lambda pct, msg: on_progress(75 + int(pct * 0.24), msg))
            if on_progress else None,
        )
