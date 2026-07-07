# STAC-Builder — Phase 3: 3D-anchored visual finding detection.
#
# Consumes the R.8 store. Over sampled keyframes an inspection prompt asks
# Qwen3-VL to report ONLY clearly visible defects (cracks, moisture/leakage,
# exposed steel, spalling, corrosion, out-of-norm provisional elements,
# obstructions) as strict JSON {type, box, severity, description, confidence}.
#
# Each 2D detection is anchored to 3D: box center -> depth at that pixel -> world
# point via the R-refined camera pose (geometry.unproject_pixel_to_world). The
# owning instance is the one whose mask covers the center pixel (fallback: the
# nearest instance cloud). Same finding seen in N frames is deduplicated to ONE
# by (type + 3D proximity). Where a finding falls on an instance with a geometric
# anomaly (onion / high vote entropy — a hook for full surface_fitting residual
# maps) its severity is raised and the correlation flagged.
#
# The VLM proposes/describes; it never measures. Every finding is born "proposed"
# and needs human validation before any deliverable. False positives destroy
# credibility, so the prompt is tuned to abstain and precision is reported
# honestly on a hand-annotated set (see eval.py).
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from phase_r.build_instances import _frame_size, _scaled_K
from phase_r.geometry import unproject_pixel_to_world
from phase_r.instance_store import InstanceStore

FINDING_TYPES = [
    "crack", "moisture", "exposed_steel", "spalling", "corrosion",
    "provisional_element", "obstruction", "other",
]
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

_SYSTEM = (
    "You are a civil-works inspection assistant reviewing ONE photo for a "
    "supervision report. Report ONLY defects that are clearly and unambiguously "
    "visible. When in doubt, do NOT report — false positives destroy the report's "
    "credibility. You describe what you see; you never measure. STRICT JSON only."
)


def _prompt() -> str:
    return (
        "List visible defects in this image. Allowed types: "
        + ", ".join(FINDING_TYPES) + ".\n"
        'Return JSON: {"findings": [{"type": "<one of the allowed types>", '
        '"box": [x, y, w, h], '  # normalized 0..1, top-left + size
        '"severity": "low|medium|high", '
        '"description": "<short, factual, what is visible>", '
        '"confidence": <0..1>}]}\n'
        "box is normalized [x,y,w,h] in 0..1 (top-left corner + width/height). "
        "Do NOT report as defects: tile grout lines, tile/panel joints, seams, "
        "expansion gaps, shadows, general dirt, dust or surface staining, or wear "
        "from normal use — these are NOT defects. Report a crack only if it clearly "
        "breaks through the material, and moisture only for evident wet/damp patches "
        "or active leakage (not mere discoloration). "
        "If there are NO clearly visible defects, return {\"findings\": []}. "
        "Do NOT invent defects. Output ONLY the JSON."
    )


@dataclass
class RawFinding:
    type: str
    box_xywh: tuple
    severity: str
    description: str
    confidence: float
    frame_id: int
    instance_id: int | None = None
    point3d: np.ndarray | None = None


@dataclass
class DetectStats:
    n_frames: int = 0
    n_raw: int = 0
    n_anchored: int = 0
    n_findings: int = 0
    n_correlated: int = 0
    per_type: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"frames={self.n_frames} raw={self.n_raw} anchored={self.n_anchored} "
                f"findings={self.n_findings} correlated={self.n_correlated} "
                f"per_type={self.per_type}")


def _parse_findings(txt: str) -> list[dict]:
    m = re.search(r"\{.*\}", txt or "", re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    items = obj.get("findings", obj if isinstance(obj, list) else [])
    return items if isinstance(items, list) else []


class FindingDetector:
    def __init__(self, store_path: str, session_dir, output_dir=None,
                 config: dict | None = None, backend: str = "qwen_local"):
        self.store = InstanceStore(store_path)
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir) if output_dir else self.session_dir / "output"
        self.frames_dir = self.session_dir / "frames"
        self.backend = backend
        cfg = (config or {}).get("phase3", {}) if config else {}
        self.sample_stride = int(cfg.get("sample_stride", 1))
        self.max_frames = int(cfg.get("max_frames", 60))
        self.min_confidence = float(cfg.get("min_confidence", 0.35))
        self.cluster_radius_m = float(cfg.get("cluster_radius_m", 0.30))
        self.residual_k = int(cfg.get("residual_k", 12))
        self.residual_thresh_m = float(cfg.get("residual_thresh_m", 0.02))
        self.max_depth = cfg.get("max_depth_m", None)
        self._masks = None
        self._id_to = None

    # ── inputs (mirror build_instances loaders) ─────────────────────
    def _load_masks(self):
        seg = json.load(open(self.output_dir / "segmentation.json"))
        self._id_to = {i["id"]: i for i in seg.get("instances", [])}
        npz = np.load(self.output_dir / seg.get("mask_file", "seg_masks.npz"))
        by_frame: dict[int, list[tuple[int, str]]] = {}
        for key in npz.files:
            if not key.startswith("f") or "_o" not in key:
                continue
            fpart, opart = key[1:].split("_o")
            by_frame.setdefault(int(fpart), []).append((int(opart), key))
        self._npz = npz
        self._by_frame = by_frame

    def _depth_provider(self):
        for c in ["omega_run/results_output", "da3_run/results_output", "results_output"]:
            d = self.output_dir / c
            if d.is_dir():
                def prov(fid, _d=d):
                    p = _d / f"frame_{fid}.npz"
                    if not p.exists():
                        return None
                    arr = np.load(p)
                    return arr["depth"].astype(np.float32) if "depth" in arr else None
                return prov
        return lambda fid: None

    def _keyframes(self) -> list[int]:
        # frames that actually contain segmented instances, subsampled
        frames = sorted(self._by_frame.keys())
        frames = frames[:: max(1, self.sample_stride)]
        if len(frames) > self.max_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_frames).round().astype(int)
            frames = [frames[i] for i in sorted(set(idx))]
        return frames

    # ── anchoring ───────────────────────────────────────────────────
    def _assign_instance(self, fid: int, cu_norm: float, cv_norm: float) -> int | None:
        """Instance whose mask covers the box-center pixel at this frame."""
        for oid, key in self._by_frame.get(fid, []):
            mask = self._npz[key]
            h, w = mask.shape[:2]
            ui, vi = int(cu_norm * w), int(cv_norm * h)
            if 0 <= ui < w and 0 <= vi < h and mask[vi, ui] > 0:
                return int(self._id_to.get(oid, {}).get("instance_id", oid))
        return None

    def _nearest_instance(self, pt: np.ndarray) -> int | None:
        best, best_d = None, self.cluster_radius_m * 3.0
        for inst in self.store.list_instances():
            pts = self.store.get_points(inst["instance_id"])
            if pts is None or not len(pts):
                continue
            d = float(np.min(np.linalg.norm(pts - pt[None, :], axis=1)))
            if d < best_d:
                best, best_d = int(inst["instance_id"]), d
        return best

    # ── multi-view dedup (union-find over type + 3D proximity) ──────
    def _dedup(self, raws: list[RawFinding]) -> list[RawFinding]:
        anchored = [r for r in raws if r.point3d is not None]
        n = len(anchored)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                if anchored[i].type != anchored[j].type:
                    continue
                if np.linalg.norm(anchored[i].point3d - anchored[j].point3d) <= self.cluster_radius_m:
                    parent[find(i)] = find(j)
        clusters: dict[int, list[RawFinding]] = {}
        for i, r in enumerate(anchored):
            clusters.setdefault(find(i), []).append(r)
        merged = []
        for members in clusters.values():
            best = max(members, key=lambda r: r.confidence)
            best.n_views = len(members)  # type: ignore[attr-defined]
            merged.append(best)
        return merged

    def _surface_fit_residual(self, instance_id: int, pt: np.ndarray) -> float | None:
        """REAL surface_fitting cross-check at the finding's location, when the
        artifacts exist (spec: cruce con surface_fitting cuando exista): local
        RMS out-of-surface residual over the k nearest points of the instance's
        deviation.ply. None when no artifact / point off the surface."""
        base = Path(self.store.path).parent / "surface_fit"
        if not base.is_dir():
            return None
        for d in sorted(base.glob(f"*_{instance_id}")):
            ply = d / "deviation.ply"
            if not ply.exists():
                continue
            try:
                import open3d as o3d
                xyz = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
                if len(xyz) < self.residual_k:
                    return None
                dist = np.linalg.norm(xyz - pt[None, :], axis=1)
                order = np.argsort(dist)[: self.residual_k]
                if dist[order[0]] > self.cluster_radius_m:
                    return None
                nb = xyz[order]
                from phase_r.depth_regularization import fit_plane
                n, dd = fit_plane(nb)
                return float(np.sqrt(np.mean((nb @ n + dd) ** 2)))
            except Exception:
                return None
        return None

    def _local_residual(self, instance_id: int | None, pt: np.ndarray) -> float | None:
        """Local geometric residual at the finding. Prefers the REAL
        surface_fitting deviation artifact when it exists; falls back to a
        plane-fit proxy over the k instance-cloud points nearest the finding's
        3D location — RMS out-of-plane residual (m). None if the point is off
        the cloud or too few neighbors. Unlike an instance-wide onion flag,
        this fires only where the surface is actually rough/non-planar AT the
        defect."""
        if instance_id is None:
            return None
        sf = self._surface_fit_residual(instance_id, pt)
        if sf is not None:
            return sf
        pts = self.store.get_points(instance_id)
        if pts is None or len(pts) < self.residual_k:
            return None
        dist = np.linalg.norm(pts - pt[None, :], axis=1)
        order = np.argsort(dist)[: self.residual_k]
        if dist[order[0]] > self.cluster_radius_m:
            return None  # finding point not on this instance's surface
        nb = pts[order]
        from phase_r.depth_regularization import fit_plane
        n, d = fit_plane(nb)
        return float(np.sqrt(np.mean((nb @ n + d) ** 2)))

    @staticmethod
    def _bump(sev: str) -> str:
        r = min(2, _SEVERITY_RANK.get(sev, 1) + 1)
        return {v: k for k, v in _SEVERITY_RANK.items()}[r]

    # ── run ─────────────────────────────────────────────────────────
    def run(self, on_progress=None) -> DetectStats:
        from semantic.client import get_semantic_client
        from semantic.types import system, user
        self._load_masks()
        cam = self._load_camera()
        depth_of = self._depth_provider()
        client = get_semantic_client(backend=self.backend, consumer="phase3.findings")

        frames = self._keyframes()
        stats = DetectStats(n_frames=len(frames))
        raws: list[RawFinding] = []
        for i, fid in enumerate(frames):
            img_path = self.frames_dir / f"{fid:06d}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            resp = client.chat([system(_SYSTEM), user(_prompt(), images=[img])],
                               max_tokens=768, consumer="phase3.findings")
            items = _parse_findings(resp.content or "")
            depth = depth_of(fid)
            c2w = cam.pose_map.get(fid) if cam else None
            K = cam.K_for(fid) if cam else None
            for it in items:
                r = self._make_raw(it, fid)
                if r is None:
                    continue
                stats.n_raw += 1
                if depth is not None and c2w is not None and K is not None:
                    self._anchor(r, depth, K, c2w)
                if r.point3d is not None:
                    stats.n_anchored += 1
                raws.append(r)
            if on_progress:
                on_progress(int(100 * (i + 1) / max(1, len(frames))),
                            f"frame {fid}: +{len(items)} raw")

        merged = self._dedup(raws)
        for r in merged:
            resid = self._local_residual(r.instance_id, r.point3d) if r.point3d is not None else None
            correlated = resid is not None and resid > self.residual_thresh_m
            severity = self._bump(r.severity) if correlated else r.severity
            self.store.add_finding(
                instance_id=r.instance_id, type=r.type, severity=severity,
                description=r.description, confidence=r.confidence, frame_id=r.frame_id,
                box_xywh=np.asarray(r.box_xywh, np.float32), point3d=r.point3d,
                correlated_residual=correlated, status="proposed", origin="vlm_proposed")
            stats.n_findings += 1
            stats.n_correlated += int(correlated)
            stats.per_type[r.type] = stats.per_type.get(r.type, 0) + 1
        self.store.close()
        return stats

    def _load_camera(self):
        from segmentation.session_io import _load_camera_source
        return _load_camera_source(self.session_dir, self.output_dir)

    def _make_raw(self, it: dict, fid: int) -> RawFinding | None:
        typ = str(it.get("type", "")).strip().lower().replace(" ", "_")
        if typ not in FINDING_TYPES:
            typ = "other"
        box = it.get("box") or it.get("box_xywh")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            return None
        conf = float(it.get("confidence", 0.0))
        if conf < self.min_confidence:
            return None
        box = tuple(float(np.clip(v, 0.0, 1.0)) for v in box)
        sev = str(it.get("severity", "medium")).strip().lower()
        if sev not in _SEVERITY_RANK:
            sev = "medium"
        return RawFinding(type=typ, box_xywh=box, severity=sev,
                          description=str(it.get("description", "")).strip(),
                          confidence=conf, frame_id=fid)

    def _anchor(self, r: RawFinding, depth, K, c2w) -> None:
        x, y, w, h = r.box_xywh
        cu_norm, cv_norm = x + w / 2.0, y + h / 2.0
        fw, fh = _frame_size(self.frames_dir, r.frame_id, depth.shape)
        dh, dw = depth.shape[:2]
        Kd = _scaled_K(np.asarray(K, float), (fw, fh), (dw, dh))
        pt = unproject_pixel_to_world(cu_norm * dw, cv_norm * dh, depth, Kd, c2w,
                                      max_depth=self.max_depth)
        if pt is None:
            return
        r.point3d = pt
        r.instance_id = (self._assign_instance(r.frame_id, cu_norm, cv_norm)
                         or self._nearest_instance(pt))
