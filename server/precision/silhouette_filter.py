"""The silhouette flyer filter of the corrected cloud (USER 2026-09-29).

F6 at maximum coverage lets every prior pixel in that no view contradicts; what the
contradiction vote cannot see is a point that is WRONG ALONG ITS OWN RAY where no other
keyframe measured a surface farther away — the tail a feed-forward depth leaves behind a
post, the rim smeared off an object's edge. The SAM3 masks can: such a point was born
inside the object's silhouette in its own keyframe and, seen from another keyframe with
enough parallax, it lands OUTSIDE that object's silhouette.

A point BELONGS to masklet m when its birth pixel (frame_global, pixel) lies inside m's
mask in its birth keyframe (``correction.visit_drift.points_of_masklets``, with the birth
pixel carried exactly onto the mask grid through the session camera: F6's undistorted
native pixel → the lens (F0's maps) → ``precision.camera.mask_grid_for``). A point that
belongs to no masklet is UNSEGMENTED and is never touched: silence is not a verdict.

Its ELIGIBLE views: every keyframe j ≠ its birth keyframe where m has a (non-empty)
mask, whose ray to the point makes an angle ≥ ``precision.refine.min_tri_deg`` with the
birth ray (a view along the birth ray cannot tell where on it the point is), where the
point projects inside the frame, and where it is NOT occluded according to view j's own
corrected depth (F6's): occluded when that depth is measured and the point lies deeper
than it by more than ``loops.witness.occlusion_tol_rel`` — the witness module's rule.
At most ``precision.cloud.silhouette_max_views`` views are tested per point (a COST cap):
per (masklet, birth keyframe) group, the views that frame the group's centroid first,
the widest triangulation angle at the centroid first, then the keyframe order —
deterministic, and the most informative: parallax is what exposes a tail.

    rule 1 (own silhouette)   with ≥ silhouette_min_votes eligible views, the point
                              leaves when the share of them where it lands inside its
                              LABEL's mask, dilated by segmentation.mask_filter.dilate_px
                              (SAM3 silhouettes are not pixel-exact), is under
                              silhouette_min_inside_frac;
    rule 2 (another's mask)   it leaves when, in its eligible views, it lands inside the
                              mask of a masklet of ANOTHER label and never inside its own.

Masklets with the SAME label never conflict and share one silhouette: the floor is
several masklets, and at this point of the chain there is no fused-object grouping yet
(segmentation_result.json belongs to another cloud or is pending). A point that belongs
to several masklets leaves only when EVERY one of its memberships says so — a point that
fits one of its objects is part of it.

Cost: one projection per (membership, tested view) pair — vectorised per view, all the
memberships that test it at once — on the device ``precision.cloud.silhouette_device``
declares (an unavailable device fails; nothing falls back). The report names the pairs
tested and the seconds.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_TAG = "[silhouette]"
KEEP, RULE_OWN, RULE_OTHER = 0, 1, 2
RULE_NAMES = {RULE_OWN: "own_silhouette", RULE_OTHER: "other_mask"}
_MASK_KEY = re.compile(r"^f(\d+)_o(\d+)$")


class SilhouetteError(RuntimeError):
    """The filter cannot run as declared — always with the exact reason."""


@dataclass(frozen=True)
class SilhouetteParams:
    max_views: int              # precision.cloud.silhouette_max_views (BOUND, cost)
    min_votes: int              # precision.cloud.silhouette_min_votes
    min_inside_frac: float      # precision.cloud.silhouette_min_inside_frac
    dilate_px: int              # segmentation.mask_filter.dilate_px
    occlusion_tol_rel: float    # loops.witness.occlusion_tol_rel
    min_tri_deg: float          # precision.refine.min_tri_deg
    device: str                 # precision.cloud.silhouette_device


def params_from(pcfg, raw_cfg: dict) -> SilhouetteParams:
    """Every parameter from its declared home; a missing one fails naming it."""
    from reconstruction.loops.config import load_loops_config
    mf = ((raw_cfg or {}).get("segmentation") or {}).get("mask_filter") or {}
    if "dilate_px" not in mf:
        raise SilhouetteError("segmentation.mask_filter.dilate_px is missing — the silhouette "
                              "filter dilates the SAM3 masks by that key, nothing is assumed")
    c = pcfg.cloud
    return SilhouetteParams(max_views=int(c.silhouette_max_views), min_votes=int(c.silhouette_min_votes),
                            min_inside_frac=float(c.silhouette_min_inside_frac),
                            dilate_px=int(mf["dilate_px"]),
                            occlusion_tol_rel=float(load_loops_config(raw_cfg).witness.occlusion_tol_rel),
                            min_tri_deg=float(pcfg.refine.min_tri_deg), device=str(c.silhouette_device))


def torch_device(name: str):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise SilhouetteError("precision.cloud.silhouette_device is cuda and no CUDA device is "
                              "available — the step fails instead of running somewhere else")
    return torch.device(name)


# ── the mask grid ─────────────────────────────────────────────────────────

def mask_lut(maps: Tuple[np.ndarray, np.ndarray], native_wh: Tuple[int, int],
             mask_hw: Tuple[int, int]) -> np.ndarray:
    """(H·W,) int64: the mask-grid linear index of every UNDISTORTED native pixel (F6's
    grid) — through the lens (F0's undistortion maps give the original frame's pixel)
    and the mask grid (a full-frame resize of the original frame, ``mask_grid_for``);
    −1 where the original frame, and so the mask, does not cover it."""
    from precision.camera import mask_grid_for, native_to_grid
    W, H = int(native_wh[0]), int(native_wh[1])
    if maps[0].shape != (H, W) or maps[1].shape != (H, W):
        raise SilhouetteError(f"the undistortion maps are {maps[0].shape}, the camera {H}x{W}")
    g = mask_grid_for(W, H, mask_hw)
    uv = native_to_grid(np.stack([maps[0], maps[1]], -1).astype(np.float64), g)
    c, r = np.rint(uv[..., 0]), np.rint(uv[..., 1])
    ok = np.isfinite(c) & np.isfinite(r) & (c >= 0) & (c < g.w) & (r >= 0) & (r < g.h)
    lin = np.where(ok, r * g.w + c, -1)
    return lin.astype(np.int64).reshape(-1)


def masks_by_keyframe(output_dir: Path, masks) -> Dict[int, List[Tuple[int, str]]]:
    """{keyframe position: [(oid, npz key)]} through the ONE frame-space translation
    (segmentation.mask_space) — the store may be keyed by position or by video frame."""
    from segmentation import mask_space
    space = mask_space.resolve(output_dir, masks=masks, log=lambda m: None)
    kfs = mask_space.keyframe_numbers(output_dir) or []
    kf_of_mask_frame = {}
    for k in range(len(kfs)):
        mf = space.from_keyframe(k)
        if mf is not None:
            kf_of_mask_frame[int(mf)] = k
    out: Dict[int, List[Tuple[int, str]]] = {}
    for key in masks.files:
        m = _MASK_KEY.match(key)
        if not m:
            continue
        kf = kf_of_mask_frame.get(int(m.group(1)))
        if kf is not None:
            out.setdefault(kf, []).append((int(m.group(2)), key))
    for v in out.values():
        v.sort()
    return out


# ── the verdict (pure: arrays in, keep mask out) ─────────────────────────

def _view_order(cent: np.ndarray, birth_c: np.ndarray, views: np.ndarray, w2c: np.ndarray,
                K: np.ndarray, hw: Tuple[int, int], birth: int, max_views: int) -> np.ndarray:
    """The views one (masklet, birth keyframe) group tests: those that frame its centroid
    first, the widest angle at the centroid first, then the keyframe order."""
    v = views[views != birth]
    if not len(v):
        return v
    H, W = hw
    T = w2c[v]
    P = np.einsum("vij,j->vi", T[:, :3, :3], cent) + T[:, :3, 3]
    z = P[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    u = K[0, 0] * P[:, 0] / zs + K[0, 2]
    vv = K[1, 1] * P[:, 1] / zs + K[1, 2]
    framed = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (vv >= 0) & (vv <= H - 1)
    C = -np.einsum("vji,vj->vi", T[:, :3, :3], T[:, :3, 3])          # view centres
    a = birth_c - cent
    b = C - cent
    cos = (b @ a) / np.maximum(np.linalg.norm(a) * np.linalg.norm(b, axis=1), 1e-12)
    ang = np.arccos(np.clip(cos, -1.0, 1.0))
    order = np.lexsort((v, -ang, ~framed))
    return v[order][:int(max_views)]


def silhouette_verdict(xyz: np.ndarray, birth: np.ndarray, members: Dict[int, np.ndarray],
                       labels: Dict[int, str], mask_keyframes: Dict[int, Sequence[int]],
                       w2c: np.ndarray, K: np.ndarray, hw: Tuple[int, int],
                       depth_of: Callable[[int], Optional[np.ndarray]],
                       masks_of: Callable[[int], Dict[int, np.ndarray]],
                       lut: np.ndarray, mask_hw: Tuple[int, int], params: SilhouetteParams,
                       log: Callable[[str], None] = print) -> Tuple[np.ndarray, dict]:
    """(keep bool (N,), report). ``birth``: keyframe POSITION per point; ``members``:
    {oid: point indices} (a point's birth pixel inside the oid's mask in its birth
    keyframe); ``labels``: {oid: label}; ``mask_keyframes``: {oid: positions with a
    non-empty mask}; ``w2c``: per position; ``K``/``hw``: F6's undistorted native camera
    and grid; ``depth_of(j)``: view j's corrected depth on that grid (≤ 0 = none);
    ``masks_of(j)``: {oid: mask on the mask grid}; ``lut``: native pixel → mask pixel."""
    import torch
    t0 = time.time()
    dev = torch_device(params.device)
    n = len(xyz)
    H, W = int(hw[0]), int(hw[1])
    Hm, Wm = int(mask_hw[0]), int(mask_hw[1])
    oids = sorted(int(o) for o, ix in members.items() if int(o) in labels and len(ix))
    label_names = sorted(set(labels.values()))
    lab_id = {name: i for i, name in enumerate(label_names)}
    # memberships, grouped by (oid, birth keyframe) — the unit that chooses its views
    pts, oid_of, lab_of, grp_of = [], [], [], []
    groups: List[Tuple[int, int]] = []
    group_views: List[np.ndarray] = []
    C_all = np.linalg.inv(w2c)[:, :3, 3]
    for o in oids:
        ix = np.unique(np.asarray(members[o], np.int64))
        ix = ix[(ix >= 0) & (ix < n)]
        b = birth[ix]
        ix, b = ix[b >= 0], b[b >= 0]
        if not len(ix):
            continue
        views = np.asarray(sorted(set(int(k) for k in mask_keyframes.get(o, ()))), np.int64)
        views = views[(views >= 0) & (views < len(w2c))]
        order = np.argsort(b, kind="stable")
        ix, b = ix[order], b[order]
        cuts = np.flatnonzero(np.diff(b)) + 1
        for seg in np.split(np.arange(len(ix)), cuts):
            bk = int(b[seg[0]])
            gi = len(groups)
            groups.append((o, bk))
            group_views.append(_view_order(xyz[ix[seg]].mean(0), C_all[bk], views, w2c, K, (H, W), bk,
                                           params.max_views))
            pts.append(ix[seg])
            oid_of.append(np.full(len(seg), o, np.int64))
            lab_of.append(np.full(len(seg), lab_id[labels[o]], np.int64))
            grp_of.append(np.full(len(seg), gi, np.int64))
    rep = {"version": 1, "provenance": "tool_measured", "params": asdict(params), "n_points": int(n),
           "n_masklets_with_points": len(oids), "n_groups": len(groups)}
    if not pts:
        rep.update({"n_memberships": 0, "n_segmented_points": 0, "n_unsegmented_points": int(n),
                    "dropped": 0, "dropped_by_rule": {v: 0 for v in RULE_NAMES.values()},
                    "dropped_by_label": {}, "pairs_tested": 0, "seconds": round(time.time() - t0, 2)})
        log(f"{LOG_TAG} no point belongs to a masklet — {n:,} unsegmented points, nothing judged")
        return np.ones(n, bool), rep
    mem_pt = np.concatenate(pts)
    mem_oid = np.concatenate(oid_of)
    mem_lab = np.concatenate(lab_of)
    mem_grp = np.concatenate(grp_of)
    n_mem = len(mem_pt)
    grp_start = np.searchsorted(mem_grp, np.arange(len(groups)), "left")
    grp_end = np.searchsorted(mem_grp, np.arange(len(groups)), "right")
    by_view: Dict[int, List[int]] = {}
    for gi, vs in enumerate(group_views):
        for j in vs:
            by_view.setdefault(int(j), []).append(gi)

    f64 = dict(dtype=torch.float64, device=dev)
    X_all = torch.as_tensor(np.asarray(xyz, np.float64), **f64)
    Cb_all = torch.as_tensor(C_all, **f64)
    t_pt = torch.as_tensor(mem_pt, device=dev)
    t_lab = torch.as_tensor(mem_lab, device=dev)
    t_birth = torch.as_tensor(np.asarray(birth, np.int64)[mem_pt], device=dev)
    t_lut = torch.as_tensor(np.asarray(lut, np.int64), device=dev)
    n_elig = torch.zeros(n_mem, dtype=torch.int32, device=dev)
    n_own = torch.zeros(n_mem, dtype=torch.int32, device=dev)
    n_oth = torch.zeros(n_mem, dtype=torch.int32, device=dev)
    cos_min = math.cos(math.radians(params.min_tri_deg))
    Kt = torch.as_tensor(np.asarray(K, np.float64), **f64)
    pairs = 0
    from correction.visit_drift import _dilated
    for j in sorted(by_view):
        sel = np.concatenate([np.arange(grp_start[g], grp_end[g]) for g in by_view[j]])
        pairs += len(sel)
        # the view's silhouettes, per LABEL (same-label masklets are one silhouette)
        present: Dict[int, np.ndarray] = {}
        for o, m in masks_of(j).items():
            if int(o) not in labels:
                continue
            m = np.asarray(m) > 0
            if m.shape != (Hm, Wm):
                raise SilhouetteError(f"mask of oid {o} in keyframe {j} is {m.shape}, the mask grid "
                                      f"is {Hm}x{Wm}")
            li = lab_id[labels[int(o)]]
            present[li] = present[li] | m if li in present else m
        if not present:
            continue
        lab_local = torch.full((len(label_names),), -1, dtype=torch.int64, device=dev)
        keys = sorted(present)
        for k, li in enumerate(keys):
            lab_local[li] = k
        und = torch.as_tensor(np.stack([present[li] for li in keys]).reshape(len(keys), -1), device=dev)
        dil = torch.as_tensor(np.stack([_dilated(present[li], params.dilate_px) > 0 for li in keys])
                              .reshape(len(keys), -1), device=dev)
        cnt = und.to(torch.int32).sum(0)
        d_np = depth_of(j)
        depth = (torch.as_tensor(np.asarray(d_np, np.float64).reshape(-1), **f64) if d_np is not None
                 else torch.zeros(H * W, **f64))
        if depth.numel() != H * W:
            raise SilhouetteError(f"the corrected depth of keyframe {j} is not on the {H}x{W} grid")
        idx = torch.as_tensor(sel, device=dev)
        X = X_all[t_pt[idx]]
        T = torch.as_tensor(np.asarray(w2c[j], np.float64), **f64)
        P = X @ T[:3, :3].T + T[:3, 3]
        z = P[:, 2]
        front = z > 1e-6
        zs = torch.where(front, z, torch.ones_like(z))
        ur = torch.round(Kt[0, 0] * P[:, 0] / zs + Kt[0, 2]).long()
        vr = torch.round(Kt[1, 1] * P[:, 1] / zs + Kt[1, 2]).long()
        inframe = front & (ur >= 0) & (ur < W) & (vr >= 0) & (vr < H)
        pix = torch.where(inframe, vr * W + ur, torch.zeros_like(ur))
        dj = depth[pix]
        occluded = (dj > 1e-6) & (z > dj * (1.0 + params.occlusion_tol_rel))
        a = Cb_all[t_birth[idx]] - X
        b = Cb_all[j] - X
        cos = (a * b).sum(1) / torch.clamp(a.norm(dim=1) * b.norm(dim=1), min=1e-12)
        mpix = t_lut[pix]
        elig = inframe & ~occluded & (cos <= cos_min) & (mpix >= 0)
        mp = torch.clamp(mpix, min=0)
        ll = lab_local[t_lab[idx]]
        if bool((ll < 0).any()):
            raise SilhouetteError(f"keyframe {j} is a view of a masklet with no mask there")
        own = dil[ll, mp]
        other = (cnt[mp] - und[ll, mp].to(torch.int32)) > 0
        n_elig[idx] += elig.to(torch.int32)
        n_own[idx] += (elig & own).to(torch.int32)
        n_oth[idx] += (elig & other).to(torch.int32)
    ne, no, nt = n_elig.cpu().numpy(), n_own.cpu().numpy(), n_oth.cpu().numpy()
    rule1 = (ne >= int(params.min_votes)) & (no < float(params.min_inside_frac) * ne)
    rule2 = (nt > 0) & (no == 0)
    reason = np.where(rule1, RULE_OWN, np.where(rule2, RULE_OTHER, KEEP)).astype(np.int64)
    # a point leaves only when EVERY one of its memberships says so
    n_memb = np.bincount(mem_pt, minlength=n)
    n_leave = np.bincount(mem_pt, weights=(reason > KEEP).astype(np.float64), minlength=n).astype(np.int64)
    leave = (n_memb > 0) & (n_leave == n_memb)
    keep = ~leave
    # the reason and label a dropped point is reported under: rule 1 before rule 2, the
    # smallest oid first (memberships are ordered by oid)
    by_label: Dict[str, Dict[str, int]] = {}
    by_rule = {v: 0 for v in RULE_NAMES.values()}
    dm = leave[mem_pt]
    if dm.any():
        o = np.lexsort((mem_oid[dm], reason[dm], mem_pt[dm]))
        mp_, rs_, lb_ = mem_pt[dm][o], reason[dm][o], mem_lab[dm][o]
        first = np.concatenate([[True], mp_[1:] != mp_[:-1]])
        for r, lb in zip(rs_[first], lb_[first]):
            name = RULE_NAMES[int(r)]
            by_rule[name] += 1
            d = by_label.setdefault(label_names[int(lb)], {v: 0 for v in RULE_NAMES.values()})
            d[name] += 1
    tested = np.array([len(v) for v in group_views], np.int64)[mem_grp]
    rep.update({
        "n_memberships": int(n_mem), "n_segmented_points": int((n_memb > 0).sum()),
        "n_unsegmented_points": int((n_memb == 0).sum()),
        "dropped": int(leave.sum()), "dropped_by_rule": by_rule,
        "dropped_by_label": dict(sorted(by_label.items(), key=lambda kv: -sum(kv[1].values()))),
        "memberships_by_rule": {"own_silhouette": int((reason == RULE_OWN).sum()),
                                "other_mask": int((reason == RULE_OTHER).sum())},
        "views_tested_per_membership": {"mean": float(tested.mean()), "max": int(tested.max())},
        "eligible_views_per_membership": {
            "mean": float(ne.mean()), "p10": float(np.percentile(ne, 10)),
            "median": float(np.median(ne)), "p90": float(np.percentile(ne, 90)), "max": int(ne.max()),
            "share_zero": float((ne == 0).mean()),
            "share_judged_rule1": float((ne >= int(params.min_votes)).mean())},
        "pairs_tested": int(pairs), "device": str(dev), "seconds": round(time.time() - t0, 2)})
    log(f"{LOG_TAG} {rep['n_segmented_points']:,} of {n:,} points belong to {len(oids)} masklet(s) "
        f"({n_mem:,} memberships, {len(groups):,} masklet×birth groups); {pairs:,} (membership, view) "
        f"pairs tested (≤ {params.max_views} views each), eligible views per membership median "
        f"{np.median(ne):.0f} (p10 {np.percentile(ne, 10):.0f}, p90 {np.percentile(ne, 90):.0f}), "
        f"{100 * rep['eligible_views_per_membership']['share_judged_rule1']:.1f} % with ≥ "
        f"{params.min_votes} (rule 1 can judge)")
    log(f"{LOG_TAG} {int(leave.sum()):,} leave — own silhouette {by_rule['own_silhouette']:,}, "
        f"another label's mask {by_rule['other_mask']:,}; {rep['n_unsegmented_points']:,} unsegmented "
        f"untouched ({rep['seconds']} s on {dev})")
    return keep, rep


# ── the step: both PLYs of the cloud stage, the same rows ───────────────

def run_filter(output_dir: Path, tmp: Path, frames: Dict[int, dict], kf: Sequence[int],
               kf_w2c: np.ndarray, K: np.ndarray, native_wh: Tuple[int, int],
               maps: Tuple[np.ndarray, np.ndarray], params: SilhouetteParams,
               log: Callable[[str], None] = print) -> dict:
    """Filter ``tmp/cleaned_cloud.ply`` and ``tmp/cleaned_cloud_raw.ply`` (same rows,
    pixel_row/col on F6's undistorted native grid) with the session's SAM3 masklets
    (``output_dir``/segmentation.json + its mask store). ``frames``: {frame: {depth}}
    — F6's corrected depth per keyframe; ``kf`` / ``kf_w2c``: the keyframes in
    camera_frames.txt order and their poses."""
    from correction import visit_drift as VD
    from correction.session import read_ply, write_ply
    from segmentation import mask_space
    output_dir, tmp = Path(output_dir), Path(tmp)
    seg = output_dir / "segmentation.json"
    if not seg.exists():
        log(f"{LOG_TAG} {seg} does not exist — no masklet, every point is unsegmented: nothing judged")
        return {"ran": False, "reason": f"{seg.name} absent — no masklets"}
    doc = json.loads(seg.read_text())
    masks_path = output_dir / str(doc.get("mask_file") or "seg_masks.npz")
    if not masks_path.exists():
        raise SilhouetteError(f"{masks_path} does not exist while {seg.name} names it")
    kfs = mask_space.keyframe_numbers(output_dir) or []
    if [int(x) for x in kfs] != [int(x) for x in kf]:
        raise SilhouetteError("the mask space's keyframe list is not the camera's keyframe list — "
                              "positions would name different keyframes")
    masklets = VD.masklet_visits(output_dir, log=lambda m: log(f"{LOG_TAG} {m}"))
    labels = {int(m.oid): str(m.label) for m in masklets}
    mask_kf = {int(m.oid): [int(k) for k in m.keyframes] for m in masklets}
    masks = np.load(masks_path)
    by_kf = masks_by_keyframe(output_dir, masks)
    probe = next((k for k in masks.files if _MASK_KEY.match(k)), None)
    if probe is None:
        log(f"{LOG_TAG} {masks_path.name} holds no mask — nothing judged")
        return {"ran": False, "reason": "no mask in the store"}
    mask_hw = tuple(int(v) for v in masks[probe].shape[:2])
    header, data = read_ply(tmp / "cleaned_cloud.ply")
    header_r, raw = read_ply(tmp / "cleaned_cloud_raw.ply")
    if len(raw) != len(data) or not np.array_equal(raw["frame_global"], data["frame_global"]) or \
            not np.array_equal(raw["pixel_row"], data["pixel_row"]) or \
            not np.array_equal(raw["pixel_col"], data["pixel_col"]):
        raise SilhouetteError("cleaned_cloud.ply and cleaned_cloud_raw.ply do not hold the same rows")
    W, H = int(native_wh[0]), int(native_wh[1])
    fg = np.asarray(data["frame_global"], np.int64)
    pr = np.asarray(data["pixel_row"], np.int64)
    pc = np.asarray(data["pixel_col"], np.int64)
    kf_arr = np.asarray(kf, np.int64)
    pos = np.full(int(kf_arr.max()) + 2, -1, np.int64)
    pos[kf_arr] = np.arange(len(kf_arr))
    birth = np.where((fg >= 0) & (fg < len(pos)), pos[np.clip(fg, 0, len(pos) - 1)], -1)
    lut = mask_lut(maps, (W, H), mask_hw)
    inb = (pr >= 0) & (pr < H) & (pc >= 0) & (pc < W)
    lin = np.where(inb, lut[np.clip(pr, 0, H - 1) * W + np.clip(pc, 0, W - 1)], -1)
    mr = np.where(lin >= 0, lin // mask_hw[1], -1)
    mc = np.where(lin >= 0, lin % mask_hw[1], -1)
    members = VD.points_of_masklets(output_dir, fg, pr, pc, log=lambda m: log(f"{LOG_TAG} {m}"),
                                    mask_pixels=(mr, mc))
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)

    def depth_of(j):
        fr = frames.get(int(kf[j]))
        return None if fr is None else fr["depth"]

    def masks_of(j):
        return {o: masks[key] for o, key in by_kf.get(int(j), [])}

    keep, rep = silhouette_verdict(xyz, birth, members, labels, mask_kf, np.asarray(kf_w2c, np.float64),
                                   np.asarray(K, np.float64), (H, W), depth_of, masks_of, lut, mask_hw,
                                   params, log=log)
    if not keep.all():
        write_ply(tmp / "cleaned_cloud.ply", header, data[keep])
        write_ply(tmp / "cleaned_cloud_raw.ply", header_r, raw[keep])
    rep.update({"ran": True, "kept": int(keep.sum()), "mask_grid": list(mask_hw)})
    return rep
