# STAC-Builder — Phase 4A: capture QC pre-filter at ingestion.
#
# Runs BEFORE DA3/VGGT. Every sampled frame gets cheap classical metrics
# (Laplacian-variance blur + exposure). The expensive VLM is invoked ONLY on the
# ambiguous band — borderline blur or suspicious exposure — never on all frames
# (cost). On those cases Qwen3-VL judges blur, occlusion by people/equipment,
# exposure and general usefulness, and makes the keep/drop call.
#
# The filter NEVER deletes frames: it emits a manifest and a drop-log recording
# exactly which frames were rejected and why. It is fully configurable and can be
# disabled. The VLM proposes/judges usefulness; it does not measure geometry.
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from phase4_qc.metrics import frame_metrics

_SYSTEM = ("You are a capture-quality inspector deciding whether a photo is "
           "usable for 3D reconstruction. You judge image quality only. "
           "STRICT JSON only.")


def _prompt() -> str:
    return (
        "Assess this frame for 3D reconstruction. Return JSON:\n"
        '{"usable": <true|false>, '
        '"blur": "sharp|slight|severe", '
        '"occlusion": "none|partial|heavy", '  # people/equipment blocking the scene
        '"exposure": "ok|dark|bright", '
        '"reason": "<short>"}\n'
        "usable=false only for clearly unusable frames (severe blur, heavy "
        "occlusion, or exposure that hides the scene). Output ONLY the JSON."
    )


@dataclass
class FrameVerdict:
    frame_id: int
    keep: bool
    decided_by: str          # "classical" | "vlm"
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    vlm: dict | None = None


@dataclass
class QCManifest:
    n_frames: int = 0
    n_kept: int = 0
    n_dropped: int = 0
    n_escalated: int = 0
    frames: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"frames={self.n_frames} kept={self.n_kept} dropped={self.n_dropped} "
                f"vlm_escalated={self.n_escalated}")


def _parse(txt: str) -> dict | None:
    m = re.search(r"\{.*\}", txt or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class CaptureQC:
    def __init__(self, session_dir, config: dict | None = None,
                 backend: str = "qwen_local"):
        self.session_dir = Path(session_dir)
        self.frames_dir = self.session_dir / "frames"
        self.backend = backend
        cfg = (config or {}).get("phase4", {}) if config else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.sample_stride = int(cfg.get("sample_stride", 1))
        self.blur_keep_above = float(cfg.get("blur_keep_above", 120.0))
        self.blur_drop_below = float(cfg.get("blur_drop_below", 25.0))
        self.brightness_lo = float(cfg.get("brightness_lo", 40.0))
        self.brightness_hi = float(cfg.get("brightness_hi", 220.0))
        self.clip_frac_max = float(cfg.get("clip_frac_max", 0.25))
        self.escalate_to_vlm = bool(cfg.get("escalate_to_vlm", True))

    # ── classical triage: keep / drop / ambiguous ───────────────────
    def triage(self, m: dict) -> tuple[str, list[str]]:
        reasons = []
        exposure_bad = (m["brightness"] < self.brightness_lo
                        or m["brightness"] > self.brightness_hi
                        or m["clip_low"] > self.clip_frac_max
                        or m["clip_high"] > self.clip_frac_max)
        if m["brightness"] < self.brightness_lo:
            reasons.append("dark")
        if m["brightness"] > self.brightness_hi:
            reasons.append("bright")
        if m["clip_low"] > self.clip_frac_max or m["clip_high"] > self.clip_frac_max:
            reasons.append("clipping")
        if m["lap_var"] <= self.blur_drop_below:
            return "drop", ["blur"]
        if m["lap_var"] >= self.blur_keep_above and not exposure_bad:
            return "keep", []
        # borderline blur, or suspicious exposure -> ambiguous
        if m["lap_var"] < self.blur_keep_above:
            reasons.append("borderline_blur")
        return "ambiguous", reasons

    def _list_frames(self) -> list[int]:
        ids = []
        for p in sorted(self.frames_dir.glob("*.jpg")):
            try:
                ids.append(int(p.stem))
            except ValueError:
                continue
        return ids[:: max(1, self.sample_stride)]

    def run(self, on_progress=None) -> QCManifest:
        frames = self._list_frames()
        man = QCManifest(n_frames=len(frames))
        if not self.enabled:
            # spec: the pre-filter is deactivatable — a disabled QC keeps EVERY
            # frame and says so in the manifest (nothing dropped silently).
            for fid in frames:
                man.frames.append(FrameVerdict(
                    frame_id=fid, keep=True, decided_by="disabled",
                    reasons=["phase4.enabled=false"], metrics={}))
            man.n_kept = len(frames)
            return man

        client = None
        if self.escalate_to_vlm:
            from semantic.client import get_semantic_client
            client = get_semantic_client(backend=self.backend, consumer="phase4.qc")
        for i, fid in enumerate(frames):
            img = Image.open(self.frames_dir / f"{fid:06d}.jpg").convert("RGB")
            m = frame_metrics(img).as_dict()
            decision, reasons = self.triage(m)
            v = FrameVerdict(frame_id=fid, keep=True, decided_by="classical",
                             reasons=reasons, metrics=m)
            if decision == "keep":
                v.keep = True
            elif decision == "drop":
                v.keep = False
            else:  # ambiguous
                if client is not None:
                    man.n_escalated += 1
                    v.decided_by = "vlm"
                    from semantic.types import system, user
                    resp = client.chat([system(_SYSTEM), user(_prompt(), images=[img])],
                                       max_tokens=200, consumer="phase4.qc")
                    d = _parse(resp.content or "")
                    v.vlm = d
                    v.keep = bool(d.get("usable", True)) if d else True
                    if d:
                        v.reasons = [f"blur:{d.get('blur')}", f"occlusion:{d.get('occlusion')}",
                                     f"exposure:{d.get('exposure')}"]
                else:
                    # escalation disabled -> never silently drop: keep + flag
                    v.keep = True
                    v.reasons = reasons + ["ambiguous_kept"]
            man.frames.append(v)
            man.n_kept += int(v.keep)
            man.n_dropped += int(not v.keep)
            if on_progress:
                on_progress(int(100 * (i + 1) / max(1, len(frames))),
                            f"frame {fid}: {'KEEP' if v.keep else 'DROP'} ({v.decided_by})")
        return man

    def write(self, man: QCManifest, out_dir) -> dict:
        """Persist the manifest + a human-readable drop-log. Never deletes frames."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"frame_id": v.frame_id, "keep": v.keep, "decided_by": v.decided_by,
                 "reasons": v.reasons, "metrics": v.metrics, "vlm": v.vlm}
                for v in man.frames]
        (out_dir / "qc_manifest.json").write_text(json.dumps(
            {"summary": man.summary(), "frames": rows}, indent=2))
        dropped = [r for r in rows if not r["keep"]]
        lines = ["# Phase 4 QC drop-log — frames NOT deleted, only flagged",
                 f"# {man.summary()}", ""]
        for r in dropped:
            lines.append(f"frame {r['frame_id']:06d}  by={r['decided_by']}  "
                         f"reasons={','.join(r['reasons']) or 'n/a'}  "
                         f"lap_var={r['metrics'].get('lap_var', 0):.1f}")
        (out_dir / "qc_drop_log.txt").write_text("\n".join(lines) + "\n")
        return {"manifest": str(out_dir / "qc_manifest.json"),
                "drop_log": str(out_dir / "qc_drop_log.txt"), "n_dropped": len(dropped)}
