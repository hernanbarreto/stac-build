# STAC-Builder — AUDIT of the cloud against the masks. Nothing is ever cut.
#
# USER 2026-09-15: "no hay nada que cortar, es la auditoría de tu propia nube
# contra el ground truth de la máscara, es eso".
#
# The mask is the ground truth. An instance's points are projected into the
# keyframes where SAM3 drew it, and the audit reports WHERE their mass falls:
#
#   · all of it ON the mask (one island or n) → the instance is correctly
#     placed, possibly incomplete. Confidence filtering cuts a surface into
#     islands, and a hole is not a position error;
#   · mass OFF the mask → geometry in the wrong place, and the audit says in
#     which visit's frames.
#
# Mass, not outline. USER: "si usas contorno, podrías tener apenas voladores
# que te dirían acá hay duplicado" — a handful of flyers stretch an outline but
# never carry weight, so the audit weighs the points instead of tracing them.
#
# The comparison ACROSS VISITS separates the two cases that matter, with no
# threshold anywhere: mass off the mask in one visit's frames and on it in
# another's is a DRIFT DUPLICATE, which the correction moves; mass off the mask
# in EVERY visit is attached junk, which is declared and left alone.
#
# What this replaced (2026-09-15) — the same module used to DELETE the points
# that fell off their mask, and the split gate used to cut an instance in two
# when its clusters sat farther apart than an invented multiple of an invented
# drift budget. On pccr that fragmented the floor into four pieces and the
# ceiling into six. Neither cuts anything now.
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


class MaskAudit:
    """Built once per session; audits one instance at a time.

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
            self.log("[mask-audit] no mask/camera evidence — audit skipped")
            return
        try:
            self.obj_of = _mask_obj_by_iid(self.output_dir)
        except Exception as e:  # noqa: BLE001 — declared, never silent
            self.log(f"[mask-audit] instance→oid map unavailable ({e}) — audit skipped")
            self.ok = False
            return
        c2m = dict(cloud_to_mask or {})
        self._m2c = {int(m): int(c) for c, m in c2m.items()}
        self.res = self._mask_res()
        if self.res is None:
            self.log("[mask-audit] no mask resolution — audit skipped")
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
            # mask resolution: without this the rim of every object reads as
            # mass off its own mask.
            try:
                from scipy import ndimage as ndi
                m = ndi.binary_dilation(m, iterations=px)
            except Exception:  # noqa: BLE001 — undilated is stricter, not wrong
                pass
        return m

    def audit(self, instance_id: int, pts_raw: np.ndarray,
              fg: np.ndarray) -> dict:
        """Audit one instance's points against its masks. Nothing is removed.

        Per VISIT, the points are projected into that visit's mask keyframes
        and their mass is split into what lands ON the instance's mask and what
        lands OFF it. WHERE the mass falls is the verdict; how many clusters it
        forms is not:

          · all on the mask (one island or n) → correct, possibly incomplete.
            Confidence filtering cuts a surface into islands and holes are not
            position errors;
          · mass off the mask → geometry in the wrong place, and the audit says
            in which visit's frames.

        The comparison across visits separates the two cases that matter, with
        no threshold: mass that is OFF the mask in one visit's frames and ON it
        in another's is a DRIFT DUPLICATE, to be corrected; mass that is off in
        EVERY visit is attached junk, to be declared. Neither is cut.
        """
        n = len(pts_raw)
        rec = {"instance_id": int(instance_id), "n_points": int(n),
               "visits": [], "on_mask": 0, "off_mask": 0, "reason": None}
        if not self.ok or n == 0:
            rec["reason"] = "no evidence"
            self.stats.append(rec)
            return rec
        oid = self.obj_of.get(int(instance_id))
        if oid is None:
            rec["reason"] = "instance has no mask object id"
            self.stats.append(rec)
            return rec
        mask_frames = [f for f, _k in self.ev.frames_for(int(oid))]
        if not mask_frames:
            rec["reason"] = "no masks for this instance"
            self.stats.append(rec)
            return rec

        mh, mw = self.res
        visits = self._visits(mask_frames)
        vis_of = self._assign_visits(fg, visits)
        max_frames = int(self.cfg["max_frames_per_visit"])

        # PER POINT, over EVERY view of the instance — not per visit and not by
        # majority. USER 2026-09-15: "si es correcto el objeto, no hay puntos
        # que caen bien en una toma y mal en otra". Landing off its own mask in
        # a SINGLE view that sees it unoccluded already proves the point sits in
        # the wrong place. It is marked here and removed nowhere: a drift orphan
        # falls badly today and correctly once the certification has moved it,
        # and removal is the last resort, after that chance
        # (`out_of_place` → the second pass decides).
        off_views = np.zeros(n, np.int32)
        seen_views = np.zeros(n, np.int32)

        for vi, vframes in enumerate(visits):
            sel = np.flatnonzero(vis_of == vi)
            pts = pts_raw[sel]
            vrec = {"visit": vi, "n_points": int(len(sel)), "frames": 0,
                    "on_mask": 0, "off_mask": 0, "seen": 0}
            if len(sel) == 0:
                rec["visits"].append(vrec)
                continue
            ranked = []
            for mf in vframes:
                m = self._mask_of(mf, oid)
                if m is not None and m.any():
                    ranked.append((int(m.sum()), int(mf), m))
            ranked.sort(key=lambda r: -r[0])
            zbuf: Dict[int, Optional[np.ndarray]] = {}
            for _area, mf, m in ranked[:max_frames]:
                from segmentation.shape_proposer import _visible_in_frame
                pr = _visible_in_frame(
                    self.ev, self._cloud_frame(mf), mh, mw, pts,
                    inst_pts=pts, inst_zbuf_cache=zbuf,
                    depth_tol=float(self.cfg["depth_tol_m"]),
                    self_tol=float(self.cfg["self_tol_m"]))
                if pr is None:
                    continue
                mu, mv, vis = pr
                if not vis.any():
                    continue
                vrec["frames"] += 1
                vrec["seen"] += int(vis.sum())
                inside = np.zeros(len(pts), bool)
                inside[vis] = m[mv[vis], mu[vis]]
                on = int(inside.sum())
                vrec["on_mask"] += on
                vrec["off_mask"] += int(vis.sum()) - on
                seen_views[sel[vis]] += 1
                off_views[sel[vis & ~inside]] += 1
            rec["visits"].append(vrec)
            rec["on_mask"] += vrec["on_mask"]
            rec["off_mask"] += vrec["off_mask"]

        # the comparison across visits, no threshold: a visit whose mass is
        # mostly off the mask is where this instance sits in the wrong place
        off = [v for v in rec["visits"] if v["seen"] and v["off_mask"] > v["on_mask"]]
        on = [v for v in rec["visits"] if v["seen"] and v["on_mask"] >= v["off_mask"]]
        rec["displaced_visits"] = [v["visit"] for v in off]
        rec["placed_visits"] = [v["visit"] for v in on]
        rec["verdict"] = ("correct" if not off else
                          "drift_duplicate" if on else "unsupported")
        # `judged` travels with the verdict: a point NO mask frame saw was not
        # found clean, it was not asked. The second moment of the cycle
        # (segmentation/geometric_cleanup) needs that distinction — it may only
        # call a marked point CURED when the masks looked at it again.
        rec["judged"] = judged = seen_views > 0
        rec["out_of_place"] = out = judged & (off_views > 0)
        rec["n_judged"] = int(judged.sum())
        rec["n_out_of_place"] = int(out.sum())
        self.stats.append(rec)
        return rec

    # ── report ───────────────────────────────────────────────────────────

    # the per-point marks are the CALLER's business, not the report's: they are
    # arrays as long as the instance, they do not serialize, and the counts
    # beside them (n_judged / n_out_of_place) say everything a report needs.
    # Leaving them in cost pccr its whole second moment on 2026-09-15 — the
    # report raised "Object of type ndarray is not JSON serializable", the
    # writer caught it as non-fatal, and out_of_place.npy was never saved.
    _PER_POINT = ("out_of_place", "judged")

    def report(self) -> dict:
        """What the audit found, per instance and in total. No verdict of the
        kind that removes anything: the numbers travel to whoever corrects."""
        on = int(sum(r.get("on_mask", 0) for r in self.stats))
        off = int(sum(r.get("off_mask", 0) for r in self.stats))
        by = {}
        per_instance = []
        for r in self.stats:
            by.setdefault(r.get("verdict", "unmeasured"), []).append(r["instance_id"])
            per_instance.append({k: v for k, v in r.items()
                                 if k not in self._PER_POINT})
        return {"version": 1, "provenance": "tool_measured",
                "instances": len(self.stats),
                "observations_on_mask": on, "observations_off_mask": off,
                "on_mask_fraction": (on / (on + off)) if (on + off) else None,
                "instances_by_verdict": {k: sorted(v) for k, v in by.items()},
                "per_instance": per_instance}
