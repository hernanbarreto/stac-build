# STAC-Builder — Phase 4B: post-reconstruction coverage report.
#
# After reconstruction, find under-sampled zones (occupied voxels in the low
# density percentile), recover the frames that view each zone (project its
# centroid through the R-refined poses), let Qwen3-VL describe what is missing,
# and emit a natural-language recapture checklist for the field operator.
#
# Geometry (voxel density, zone grouping, frame visibility) is deterministic and
# unit-tested; the VLM only phrases the recapture instruction — it never measures.
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phase_r.geometry import invert_rigid
from phase_r.instance_store import InstanceStore

_SYSTEM = ("You help a field operator recapture under-scanned areas. Look at the "
           "frame and say briefly what should be re-photographed and from where. "
           "One or two sentences, plain language.")


@dataclass
class Zone:
    zone_id: int
    centroid: np.ndarray
    n_points: int
    n_voxels: int
    frames: list = field(default_factory=list)
    description: str = ""


def voxel_density(points: np.ndarray, voxel: float):
    """Map points to a voxel grid; return (voxel_index_array, counts_per_point,
    unique_voxels, unique_counts)."""
    p = np.asarray(points, float)
    keys = np.floor(p / voxel).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return keys, counts[inv], uniq, counts


def _connected_components(occupied: set) -> list[list[tuple]]:
    """6-connected components over a set of integer voxel keys."""
    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    seen: set = set()
    comps = []
    for start in occupied:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            c = stack.pop()
            comp.append(c)
            for d in neigh:
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nb in occupied and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def low_density_zones(points: np.ndarray, voxel: float, low_pct: float = 20.0,
                      min_zone_points: int = 20, max_zone_fraction: float = 0.5) -> list[Zone]:
    """Under-sampled zones: connected components of occupied voxels whose TOTAL
    point count is in the bottom `low_pct` percentile of component sizes. This
    flags spatially isolated / thinly-scanned regions (few points describe that
    area) rather than the dense-cloud's tail voxels. `min_zone_points` is a floor
    that discards single-stray-point noise; a component holding more than
    `max_zone_fraction` of all points is the main reconstruction body, never an
    under-sampled zone (guards the single-blob degenerate case). Sparsest-first."""
    p = np.asarray(points, float)
    if len(p) == 0:
        return []
    keys, _pc, _uniq, _counts = voxel_density(p, voxel)
    key_tuples = [tuple(k) for k in keys]
    occupied = set(key_tuples)
    comps = _connected_components(occupied)
    if not comps:
        return []
    # total points per component
    comp_of = {}
    for ci, comp in enumerate(comps):
        for vk in comp:
            comp_of[vk] = ci
    comp_points = [0] * len(comps)
    for kt in key_tuples:
        comp_points[comp_of[kt]] += 1
    thr = np.percentile(comp_points, low_pct)
    body_cap = max_zone_fraction * len(p)
    zones: list[Zone] = []
    zid = 0
    for ci, comp in enumerate(comps):
        if comp_points[ci] > thr or comp_points[ci] < min_zone_points:
            continue
        if comp_points[ci] > body_cap:  # main reconstruction body, not a gap
            continue
        comp_set = set(comp)
        member = np.array([kt in comp_set for kt in key_tuples], bool)
        pts = p[member]
        zones.append(Zone(zone_id=zid, centroid=pts.mean(0), n_points=len(pts),
                          n_voxels=len(comp)))
        zid += 1
    zones.sort(key=lambda z: z.n_points)
    return zones


def frames_viewing(centroid: np.ndarray, pose_map: dict, k_for, wh: tuple[int, int],
                   max_frames: int = 5) -> list[int]:
    """Frames whose camera sees the zone centroid (projects in-bounds, z>0),
    ranked by how centered the projection is."""
    W, H = wh
    scored = []
    for fid, c2w in pose_map.items():
        K = k_for(fid)
        if K is None:
            continue
        pc = invert_rigid(np.asarray(c2w, float)) @ np.array([*centroid, 1.0])
        if pc[2] <= 1e-6:
            continue
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        if 0 <= u < W and 0 <= v < H:
            d = ((u - W / 2) / W) ** 2 + ((v - H / 2) / H) ** 2
            scored.append((d, fid))
    scored.sort()
    return [fid for _d, fid in scored[:max_frames]]


class CoverageReport:
    def __init__(self, store_path: str, session_dir, output_dir=None,
                 config: dict | None = None, backend: str = "qwen_local"):
        self.store = InstanceStore(store_path)
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir) if output_dir else self.session_dir / "output"
        self.frames_dir = self.session_dir / "frames"
        self.backend = backend
        cfg = (config or {}).get("phase4", {}) if config else {}
        self.voxel_size_m = float(cfg.get("voxel_size_m", 0.10))
        self.low_pct = float(cfg.get("low_density_pct", 20.0))
        self.min_zone_points = int(cfg.get("min_zone_points", 20))
        self.max_zone_fraction = float(cfg.get("max_zone_fraction", 0.5))
        self.max_zones = int(cfg.get("max_zones", 12))
        self.describe = bool(cfg.get("describe", True))

    def _pooled_cloud(self) -> np.ndarray:
        clouds = []
        for inst in self.store.list_instances():
            pts = self.store.get_points(inst["instance_id"])
            if pts is not None and len(pts):
                clouds.append(pts)
        return np.vstack(clouds) if clouds else np.empty((0, 3))

    def run(self, on_progress=None) -> dict:
        from segmentation.session_io import _load_camera_source
        cloud = self._pooled_cloud()
        zones = low_density_zones(cloud, self.voxel_size_m, self.low_pct,
                                  self.min_zone_points,
                                  self.max_zone_fraction)[: self.max_zones]
        cam = _load_camera_source(self.session_dir, self.output_dir)
        wh = self._frame_wh()
        client = None
        if self.describe and zones:
            from semantic.client import get_semantic_client
            client = get_semantic_client(backend=self.backend, consumer="phase4.coverage")

        checklist = []
        for i, z in enumerate(zones):
            if cam is not None:
                z.frames = frames_viewing(z.centroid, cam.pose_map, cam.K_for, wh)
            if client and z.frames:
                z.description = self._describe(client, z.frames[0])
            checklist.append({
                "zone_id": z.zone_id, "centroid": z.centroid.tolist(),
                "n_points": z.n_points, "n_voxels": z.n_voxels,
                "viewing_frames": z.frames,
                "instruction": z.description or self._fallback_instruction(z),
            })
            if on_progress:
                on_progress(int(100 * (i + 1) / max(1, len(zones))),
                            f"zone {z.zone_id}: {z.n_points} pts, {len(z.frames)} frames")
        self.store.close()
        return {"n_zones": len(zones), "voxel_size_m": self.voxel_size_m,
                "total_points": int(len(cloud)), "checklist": checklist}

    def _describe(self, client, fid: int) -> str:
        from semantic.types import system, user
        from PIL import Image
        p = self.frames_dir / f"{fid:06d}.jpg"
        if not p.exists():
            return ""
        img = Image.open(p).convert("RGB")
        resp = client.chat(
            [system(_SYSTEM),
             user("This area is under-scanned (few 3D points). What should the "
                  "operator re-capture here, and from what angle?", images=[img])],
            max_tokens=160, consumer="phase4.coverage")
        return (resp.content or "").strip()

    @staticmethod
    def _fallback_instruction(z: Zone) -> str:
        c = z.centroid
        return (f"Re-scan around ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) m: only "
                f"{z.n_points} points across {z.n_voxels} voxels — get closer, "
                f"slower passes with more overlap.")

    def write(self, report: dict, out_dir) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "coverage_report.json").write_text(json.dumps(report, indent=2))
        lines = ["# Phase 4 recapture checklist",
                 f"# {report['n_zones']} under-sampled zone(s), "
                 f"{report['total_points']} points, voxel={report['voxel_size_m']} m", ""]
        for c in report["checklist"]:
            lines.append(f"[ ] zone {c['zone_id']} — frames {c['viewing_frames']}")
            lines.append(f"    {c['instruction']}")
        (out_dir / "recapture_checklist.txt").write_text("\n".join(lines) + "\n")
        return {"report": str(out_dir / "coverage_report.json"),
                "checklist": str(out_dir / "recapture_checklist.txt")}

    def _frame_wh(self) -> tuple[int, int]:
        from PIL import Image
        for p in sorted(self.frames_dir.glob("*.jpg")):
            with Image.open(p) as im:
                return im.size
        return (1920, 1080)
