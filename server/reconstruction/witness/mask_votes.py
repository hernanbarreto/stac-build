"""§6.2 ``mask_votes`` / ``mask_conflicts`` on the cloud.

For every SAM3 instance, its points are projected (final poses) into every
keyframe that OBSERVES the instance (a mask of it exists there). Per point
and view the projection lands inside the SAME instance's mask (``inside`` →
mask_votes), inside ANOTHER instance's mask (``conflict`` → mask_conflicts)
or on background (no vote). The multi-view plurality vote r3d's build_scene
uses to ASSIGN points to objects, used here as a per-point witness only.

Masks are eroded by ``witness.mask_erosion_px`` (borders are never
penalised) and a projection that lands BEHIND the frame's own measured
surface by more than ``witness.occlusion_tol_rel`` is occluded in that view
(a point hidden by a closer object is not in conflict with that object).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class MaskStore:
    masks: dict                      # key f<frame>_o<obj> → uint8 (Hm,Wm)
    res: Tuple[int, int]             # (Hm, Wm) of the masks
    obj_of: Dict[int, int]           # instance_id → SAM3 obj id (mask key space)
    space: "object" = None           # segmentation.mask_space.MaskSpace

    def mask_key(self, cloud_frame: int, obj: int) -> Optional[str]:
        """The npz key of an object in a CLOUD/pose frame, or None when that
        frame is not a keyframe. The store's ``f<N>`` is the keyframe POSITION
        on every store the batch pipeline writes — never the frame number the
        poses, the depths and ``frame_global`` use."""
        if self.space is None:
            return f"f{int(cloud_frame)}_o{int(obj)}"
        return self.space.key(cloud_frame, obj)

    def cloud_frame(self, mask_frame: int) -> Optional[int]:
        if self.space is None:
            return int(mask_frame)
        return self.space.to_cloud(mask_frame)


def load_mask_store(output_dir) -> Optional[MaskStore]:
    """The session's SAM3 masks, or None when the session has no
    segmentation masks (then every point has zero mask witnesses)."""
    output_dir = Path(output_dir)
    p = output_dir / "seg_masks.npz"
    seg = output_dir / "segmentation.json"
    if not p.exists() or not seg.exists():
        return None
    z = np.load(p)
    meta = json.loads(seg.read_text())
    res = meta.get("resolution", {}).get("scaled")
    if "scaled_res" in z.files:
        sr = z["scaled_res"]
        res = [int(sr[0]), int(sr[1])]
    if not res:
        raise RuntimeError(f"{seg}: resolution.scaled missing — the mask grid is unknown")
    obj_of: Dict[int, int] = {}
    for e in meta.get("instances") or []:
        if e.get("id") is not None and e.get("instance_id") is not None:
            obj_of[int(e["instance_id"])] = int(e["id"])
    masks = {k: z[k] for k in z.files if k.startswith("f") and "_o" in k}
    from segmentation import mask_space
    space = mask_space.resolve(output_dir, masks=z)
    return MaskStore(masks=masks, res=(int(res[0]), int(res[1])), obj_of=obj_of,
                     space=space)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask.astype(bool), iterations=int(px))


def label_maps(store: MaskStore, instance_ids: List[int], erosion_px: int
               ) -> Tuple[Dict[int, np.ndarray], Dict[int, Set[int]]]:
    """Per keyframe an int32 label image (instance_id per pixel, 0 =
    background; smaller masks painted last so a nested object keeps its
    identity) and the set of instances observed in that frame.

    Keyed by the CLOUD/POSE frame, because that is what the caller's
    ``frames`` dict is keyed by. The mask keys are keyframe POSITIONS, and
    comparing the two directly is what limited the whole mask witness to the
    13 pccr keyframes whose video number happens to be below 216 — and those
    13 were matched to the WRONG keyframe's masks. The point ``status`` the
    kit paints green/yellow/red comes out of here."""
    per_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    dropped = 0
    for iid in instance_ids:
        obj = store.obj_of.get(int(iid))
        if obj is None:
            continue
        suffix = f"_o{obj}"
        for key, m in store.masks.items():
            if not key.endswith(suffix):
                continue
            mask_frame = int(key[1:key.index("_o")])
            frame = store.cloud_frame(mask_frame)
            if frame is None:            # a stray key of a mixed store
                dropped += 1
                continue
            per_frame.setdefault(int(frame), []).append((int(iid), m))
    if dropped:
        print(f"[witness] {dropped} mask key(s) have no keyframe — "
              f"{store.space.describe() if store.space else 'unknown frame space'}")
    maps: Dict[int, np.ndarray] = {}
    present: Dict[int, Set[int]] = {}
    Hm, Wm = store.res
    for frame, items in per_frame.items():
        lm = np.zeros((Hm, Wm), np.int32)
        items.sort(key=lambda it: -int(np.count_nonzero(it[1])))
        seen: Set[int] = set()
        for iid, m in items:
            if m.shape != (Hm, Wm):
                raise RuntimeError(f"mask {store.mask_key(frame, store.obj_of[iid])} has "
                                   f"shape {m.shape}, the store declares {store.res}")
            e = _erode(m, erosion_px)
            lm[e] = iid
            seen.add(iid)
        maps[frame] = lm
        present[frame] = seen
    return maps, present


def compute_mask_votes(xyz: np.ndarray, instances: List[dict], frames: Dict[int, dict],
                       store: MaskStore, erosion_px: int, occlusion_tol_rel: float,
                       device=None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point (mask_votes, mask_conflicts) uint8 (saturating). Points of no
    instance stay at zero."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n = len(xyz)
    votes = np.zeros(n, np.int32)
    conflicts = np.zeros(n, np.int32)
    ids = [int(i.get("instance_id", i.get("id"))) for i in instances]
    maps, present = label_maps(store, ids, erosion_px)
    Hm, Wm = store.res
    tol = float(occlusion_tol_rel)
    frame_t: Dict[int, dict] = {}
    for f, rec in frames.items():
        if f not in maps:
            continue
        T = np.asarray(rec["T"], np.float64)
        w2c = np.linalg.inv(T)
        d = np.asarray(rec["depth"], np.float32)
        Hd, Wd = d.shape
        frame_t[f] = {"w2c": torch.as_tensor(w2c, dtype=torch.float64, device=dev),
                      "K": torch.as_tensor(np.asarray(rec["K"], np.float64), dtype=torch.float64, device=dev),
                      "depth": torch.as_tensor(d, dtype=torch.float32, device=dev),
                      "lm": torch.as_tensor(maps[f], dtype=torch.int32, device=dev),
                      "sx": Wm / float(Wd), "sy": Hm / float(Hd), "hw": (Hd, Wd)}
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < n)]
        if len(gi) == 0:
            continue
        P = torch.as_tensor(np.asarray(xyz[gi], np.float64), dtype=torch.float64, device=dev)
        v_acc = torch.zeros(len(gi), dtype=torch.int32, device=dev)
        c_acc = torch.zeros(len(gi), dtype=torch.int32, device=dev)
        for f, ft in frame_t.items():
            if iid not in present.get(f, ()):
                continue
            X = P @ ft["w2c"][:3, :3].T + ft["w2c"][:3, 3]
            z = X[:, 2]
            ok = z > 1e-6
            zs = z.clamp(min=1e-6)
            K = ft["K"]
            u = K[0, 0] * X[:, 0] / zs + K[0, 2]
            v = K[1, 1] * X[:, 1] / zs + K[1, 2]
            Hd, Wd = ft["hw"]
            ud = torch.round(u).long(); vd = torch.round(v).long()
            inb = ok & (ud >= 0) & (ud < Wd) & (vd >= 0) & (vd < Hd)
            udc, vdc = ud.clamp(0, Wd - 1), vd.clamp(0, Hd - 1)
            dsurf = ft["depth"][vdc, udc]
            occluded = (dsurf > 1e-6) & (z.float() > dsurf * (1.0 + tol))
            um = torch.round(u * ft["sx"]).long().clamp(0, Wm - 1)
            vm = torch.round(v * ft["sy"]).long().clamp(0, Hm - 1)
            lab = ft["lm"][vm, um]
            use = inb & ~occluded
            v_acc += (use & (lab == iid)).to(torch.int32)
            c_acc += (use & (lab != 0) & (lab != iid)).to(torch.int32)
        votes[gi] = np.maximum(votes[gi], v_acc.cpu().numpy())
        conflicts[gi] = np.maximum(conflicts[gi], c_acc.cpu().numpy())
    return (np.minimum(votes, 255).astype(np.uint8), np.minimum(conflicts, 255).astype(np.uint8))
