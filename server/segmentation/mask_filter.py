# STAC-Builder — geometric filter: a point belongs to its instance only if the
# FRAMES say so.
#
# USER 2026-09-14: "el filtrado de los segmentos es matemático, está bien, pero
# además quiero filtrado geométrico por reproyección de todas las máscaras ...
# los puntos que pasan son solo los que caen en la máscara y los de afuera son
# outliers o ruido ... eso debe correr antes de medir todo, porque su OBB debe
# estar perfectamente ajustado".
#
# The statistical filter (DBSCAN, the voxel-component extent) asks whether a
# point sits where the other points sit. This one asks the question that has a
# witness: project the point into the keyframes where SAM3 drew the instance
# and see whether it lands on the mask.
#
# Three rules make that safe:
#
#   · OCCLUSION. A point hidden behind another object lands off the mask and is
#     not noise. Only frames where the point is VISIBLE vote — hole_audit's
#     full-cloud Z-buffer for external occluders, the instance's own unfiltered
#     buffer for its own far side (shape_proposer._visible_in_frame).
#   · OWN VISIT ONLY. Two copies of one object left by drift are each supported
#     by the masks of their own pass. Judging a copy in the OTHER pass's frames
#     deletes it — which does not fix the duplicate, it HIDES it, and destroys
#     the closure the correction measures (the same signal that reads 0.40 →
#     0.87 agreement once the rigid shift is applied). So the instance's mask
#     keyframes are grouped into visits and every point is judged inside its
#     own.
#   · EVIDENCE, NEVER DELETION. Nothing leaves the cloud (§6 witnesses, and the
#     standing doctrine that the VGGT-Ω cloud is the truth). The point stops
#     belonging to the INSTANCE, so the OBB and everything measured from it
#     tighten; the measurement itself is untouched.
#
# Frame spaces (the bug class that cost two commits on 2026-09-14): the mask
# store is keyed by KEYFRAME POSITION, the camera poses by REAL global frame,
# and K lives on the TRACE grid while the masks live on the SAM3 grid. This
# module translates explicitly at every boundary and never assumes identity.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class MaskFilter:
    """Built once per session; judges one instance at a time.

    ``cloud_to_mask`` is the mapping the matcher already built
    (``_mask_frame_lookup``): cloud frame → mask keyframe position, EMPTY when
    the identity is correct.
    """

    def __init__(self, output_dir, session_dir, cfg: dict,
                 cloud_to_mask: Optional[Dict[int, int]] = None,
                 log: Callable[[str], None] = print):
        from reconstruction.surface_fit.hole_audit import _evidence
        from segmentation.erase import _mask_obj_by_iid

        self.log = log
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.ev = _evidence(self.output_dir, Path(session_dir))
        self.ok = bool(self.ev is not None and getattr(self.ev, "ok", False))
        self.obj_of: Dict[int, int] = {}
        self._m2c: Dict[int, int] = {}
        self.res: Optional[Tuple[int, int]] = None
        self.stats: List[dict] = []
        if not self.ok:
            self.log("[mask-filter] no mask/camera evidence — geometric filter skipped")
            return
        try:
            self.obj_of = _mask_obj_by_iid(self.output_dir)
        except Exception as e:  # noqa: BLE001 — declared, never silent
            self.log(f"[mask-filter] instance→oid map unavailable ({e}) — filter skipped")
            self.ok = False
            return
        c2m = dict(cloud_to_mask or {})
        self._m2c = {int(m): int(c) for c, m in c2m.items()}
        self.res = self._mask_res()
        if self.res is None:
            self.log("[mask-filter] no mask resolution — geometric filter skipped")
            self.ok = False

    # ── session-level lookups ────────────────────────────────────────────

    def _mask_res(self) -> Optional[Tuple[int, int]]:
        """(mh, mw) of the mask grid."""
        files = getattr(self.ev, "masks", None)
        if files is None:
            return None
        try:
            if "scaled_res" in files.files:
                r = np.asarray(files["scaled_res"]).ravel()
                if r.size >= 2:
                    return int(r[0]), int(r[1])
        except Exception:  # noqa: BLE001 — fall through to probing a mask
            pass
        for k in files.files:
            if re.match(r"^f\d+_o\d+$", k):
                m = files[k]
                return int(m.shape[0]), int(m.shape[1])
        return None

    def _cloud_frame(self, mask_idx: int) -> int:
        """Mask keyframe position → the real frame the poses are keyed by."""
        return self._m2c.get(int(mask_idx), int(mask_idx))

    # ── visits ───────────────────────────────────────────────────────────

    def _visits(self, mask_frames: Sequence[int]) -> List[List[int]]:
        """Split an instance's mask keyframes into passes of the camera. A gap
        of more than ``visit_gap_kf`` keyframes is the camera having left and
        come back — which is exactly what leaves two copies."""
        gap = int(self.cfg["visit_gap_kf"])
        out: List[List[int]] = []
        for f in sorted(int(x) for x in mask_frames):
            if out and f - out[-1][-1] <= gap:
                out[-1].append(f)
            else:
                out.append([f])
        return out

    def _assign_visits(self, fg: np.ndarray, visits: List[List[int]]) -> np.ndarray:
        """Each point → the index of the visit whose real-frame span is nearest
        its own origin frame."""
        if len(visits) == 1:
            return np.zeros(len(fg), np.int64)
        spans = []
        for v in visits:
            cf = [self._cloud_frame(m) for m in v]
            spans.append((float(min(cf)), float(max(cf))))
        f = fg.astype(np.float64)
        dist = np.stack([np.maximum(np.maximum(lo - f, f - hi), 0.0)
                         for lo, hi in spans], axis=1)
        return np.argmin(dist, axis=1)

    # ── the vote ─────────────────────────────────────────────────────────

    def _mask_of(self, mask_idx: int, oid: int) -> Optional[np.ndarray]:
        key = f"f{int(mask_idx)}_o{int(oid)}"
        try:
            if key not in self.ev.masks.files:
                return None
            m = np.asarray(self.ev.masks[key]) > 0
        except Exception:  # noqa: BLE001
            return None
        px = int(self.cfg["dilate_px"])
        if px > 0:
            # SAM3 silhouettes are not pixel-exact and the cloud is sparse at
            # mask resolution: without this the rim of every object is judged
            # an outlier by its own mask.
            try:
                from scipy import ndimage as ndi
                m = ndi.binary_dilation(m, iterations=px)
            except Exception:  # noqa: BLE001 — undilated is stricter, not wrong
                pass
        return m

    def judge(self, instance_id: int, pts_raw: np.ndarray,
              fg: np.ndarray) -> Tuple[np.ndarray, dict]:
        """(keep, stats) for one instance's points, given in the RAW frame (the
        frame the poses and cleaned_cloud.ply share) with their origin frames.

        ``keep`` is all-True whenever the instance cannot be judged: no oid, no
        masks, too few votes, or a trim so large it means the evidence is not
        about this instance. Never invent.
        """
        n = len(pts_raw)
        keep = np.ones(n, bool)
        st = {"instance_id": int(instance_id), "n_points": int(n), "judged": 0,
              "dropped": 0, "visits": 0, "frames": 0, "reason": None,
              "inside_frac_before": None, "inside_frac_after": None}
        if not self.ok or n == 0:
            st["reason"] = "no evidence"
            self.stats.append(st)
            return keep, st
        oid = self.obj_of.get(int(instance_id))
        if oid is None:
            st["reason"] = "instance has no mask object id"
            self.stats.append(st)
            return keep, st

        mask_frames = [f for f, _k in self.ev.frames_for(int(oid))]
        if not mask_frames:
            st["reason"] = "no masks for this instance"
            self.stats.append(st)
            return keep, st

        mh, mw = self.res
        visits = self._visits(mask_frames)
        vis_of = self._assign_visits(fg, visits)
        st["visits"] = len(visits)
        seen = np.zeros(n, np.int32)
        hit = np.zeros(n, np.int32)
        max_frames = int(self.cfg["max_frames_per_visit"])
        used_frames = 0

        for vi, vframes in enumerate(visits):
            sel = np.flatnonzero(vis_of == vi)
            if len(sel) < int(self.cfg["min_points_per_visit"]):
                continue
            pts = pts_raw[sel]
            # rank this visit's frames by how much of the object they show:
            # a frame with three pixels of mask judges nothing
            ranked = []
            for mf in vframes:
                m = self._mask_of(mf, oid)
                if m is not None and m.any():
                    ranked.append((int(m.sum()), int(mf), m))
            ranked.sort(key=lambda r: -r[0])
            zbuf_cache: Dict[int, Optional[np.ndarray]] = {}
            for _area, mf, m in ranked[:max_frames]:
                from segmentation.shape_proposer import _visible_in_frame
                pr = _visible_in_frame(
                    self.ev, self._cloud_frame(mf), mh, mw, pts,
                    inst_pts=pts, inst_zbuf_cache=zbuf_cache,
                    depth_tol=float(self.cfg["depth_tol_m"]),
                    self_tol=float(self.cfg["self_tol_m"]))
                if pr is None:
                    continue
                mu, mv, vis = pr
                if not vis.any():
                    continue
                used_frames += 1
                seen[sel[vis]] += 1
                inside = np.zeros(len(pts), bool)
                inside[vis] = m[mv[vis], mu[vis]]
                hit[sel[inside]] += 1

        st["frames"] = used_frames
        judged = seen >= int(self.cfg["min_votes"])
        st["judged"] = int(judged.sum())
        if not judged.any():
            st["reason"] = "no point reached min_votes"
            self.stats.append(st)
            return keep, st

        frac = np.zeros(n)
        frac[judged] = hit[judged] / np.maximum(seen[judged], 1)
        st["inside_frac_before"] = float(frac[judged].mean())
        outlier = judged & (frac < float(self.cfg["min_inside_frac"]))
        kept = n - int(outlier.sum())
        if kept < (1.0 - float(self.cfg["max_drop_frac"])) * n:
            # Dropping most of an instance is not a clean object with noise: it
            # is the evidence not being about this instance (a wrong oid, masks
            # from another pass). Declare it and change nothing.
            st["reason"] = (f"would drop {100 * (1 - kept / max(n, 1)):.0f}% "
                            f"(> max_drop_frac) — not applied")
            self.stats.append(st)
            return keep, st

        keep = ~outlier
        st["dropped"] = int(outlier.sum())
        kj = judged & keep
        st["inside_frac_after"] = float(frac[kj].mean()) if kj.any() else None
        self.stats.append(st)
        return keep, st

    # ── report ───────────────────────────────────────────────────────────

    def report(self) -> dict:
        applied = [s for s in self.stats if s["dropped"] > 0]
        tot_pts = int(sum(s["n_points"] for s in self.stats))
        tot_drop = int(sum(s["dropped"] for s in self.stats))
        befores = [s["inside_frac_before"] for s in self.stats
                   if s["inside_frac_before"] is not None]
        afters = [s["inside_frac_after"] for s in self.stats
                  if s["inside_frac_after"] is not None]
        return {"version": 1, "provenance": "tool_measured",
                "instances": len(self.stats), "instances_trimmed": len(applied),
                "points": tot_pts, "points_dropped": tot_drop,
                "drop_frac": (tot_drop / tot_pts) if tot_pts else 0.0,
                "inside_frac_before": float(np.mean(befores)) if befores else None,
                "inside_frac_after": float(np.mean(afters)) if afters else None,
                "per_instance": self.stats}
