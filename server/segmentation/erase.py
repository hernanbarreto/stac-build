# STAC-BUILD: Eraser tool core (USER 2026-08-29)
"""One erase gesture, one truth.

The user brushes a sphere over the scene (cloud OR mesh — same gesture): the
instance points inside leave their segment (become unsegmented) AND their
pixels are cleared from the instance's SAM3 mask in seg_masks.npz — the mask
stays the single source of truth, so any future re-match preserves the
deletion. Meshes derive from points: the caller (main.py) debounces a
per-instance re-fit for touched instances that have a published mesh, and
OBBs are recomputed from MESH vertices once a fresh mesh exists (user rule:
"una vez que haya malla, el bbox se obtiene de la malla"), from the remaining
points meanwhile.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_floor_transform(output_dir: Path):
    p = Path(output_dir) / "floor_transform.npz"
    if not p.exists():
        return 1.0, np.eye(3), np.zeros(3)
    d = np.load(p)
    return float(d["s"]), np.asarray(d["R"], float), np.asarray(d["t"], float)


def _display_sphere_to_raw(output_dir: Path, center, radius: float):
    s, R, t = _load_floor_transform(output_dir)
    c = np.asarray(center, dtype=np.float64)
    c_raw = R.T @ ((c - t) / s)
    return c_raw, float(radius) / s


def raw_to_display(output_dir: Path, pts: np.ndarray) -> np.ndarray:
    s, R, t = _load_floor_transform(output_dir)
    return s * (np.asarray(pts, np.float64) @ R.T) + t


def _atomic_savez(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    # the tmp name MUST end in ".npz": np.savez appends ".npz" to any other
    # name, so the replace() moved the EMPTY mkstemp file over seg_masks.npz
    # and truncated it to 0 bytes (bug 2026-08-29, recovered from the stray
    # data file it left behind)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _cube_hit(disp_pts: np.ndarray, c: np.ndarray, r: float,
              yaw: float = 0.0) -> np.ndarray:
    """Points inside a cube of half-side r centred at c, rotated ``yaw``
    radians about the vertical (Y) axis of the display frame — same
    convention as three.js ``rotation.y`` (user 2026-08-30: the cube must
    rotate to adapt to diagonal walls)."""
    d = np.asarray(disp_pts, np.float64) - c
    if abs(yaw) > 1e-9:
        ca, sa = np.cos(yaw), np.sin(yaw)
        x = ca * d[:, 0] - sa * d[:, 2]
        z = sa * d[:, 0] + ca * d[:, 2]
        d = np.column_stack([x, d[:, 1], z])
    return (np.abs(d) <= r).all(axis=1)


def _rewrite_ply_keep(ply_path: Path, keep: np.ndarray) -> bool:
    """Rewrite a binary-little-endian PLY keeping only ``keep`` rows —
    physical point deletion (user 2026-08-31: low-confidence unsegmented
    points must actually LEAVE the cloud). Header preserved verbatim except
    the vertex count. Atomic (tmp + replace). Returns False (untouched file)
    when any property type is unknown — never corrupt the cloud."""
    _ply_type = {
        'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
        'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
    }
    try:
        header: List[bytes] = []
        props = []
        n_pts = 0
        with open(ply_path, 'rb') as f:
            while True:
                line = f.readline()
                header.append(line)
                s = line.decode('ascii', 'ignore').strip()
                if s.startswith('element vertex'):
                    n_pts = int(s.split()[-1])
                elif s.startswith('property'):
                    parts = s.split()
                    if len(parts) < 3 or parts[1] not in _ply_type:
                        print(f"[Erase] delete aborted: unsupported PLY "
                              f"property '{s}'")
                        return False
                    props.append((parts[2], _ply_type[parts[1]]))
                elif s == 'end_header':
                    break
            if n_pts != len(keep):
                print(f"[Erase] delete aborted: keep mask {len(keep)} vs "
                      f"{n_pts} pts")
                return False
            data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n_pts)
        kept = data[keep]
        fd, tmp = tempfile.mkstemp(dir=str(ply_path.parent), suffix=".ply")
        os.close(fd)
        with open(tmp, 'wb') as f:
            for line in header:
                s = line.decode('ascii', 'ignore').strip()
                if s.startswith('element vertex'):
                    f.write(f"element vertex {len(kept)}\n".encode('ascii'))
                else:
                    f.write(line)
            f.write(kept.tobytes())
        os.replace(tmp, ply_path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] PLY rewrite failed ({e}) — cloud untouched")
        return False


def _load_confidence_norm(ply_path: Path) -> Optional[np.ndarray]:
    """Per-point confidence from the binary PLY, normalized to [0,1] with the
    array's own min/max — the SAME normalization the viewer applies (Potree
    metadata min/max come from these values), so a UI threshold means the
    same thing here. None when the cloud has no confidence field."""
    _ply_type = {
        'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
        'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
    }
    try:
        with open(ply_path, 'rb') as f:
            n_pts = 0
            props = []
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n_pts = int(line.split()[-1])
                elif line.startswith('property') and n_pts > 0:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] in _ply_type:
                        props.append((parts[2], _ply_type[parts[1]]))
                elif line == 'end_header':
                    break
            if n_pts == 0 or 'confidence' not in {p[0] for p in props}:
                return None
            data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n_pts)
            c = np.asarray(data['confidence'], dtype=np.float64)
            lo, hi = float(c.min()), float(c.max())
            return ((c - lo) / max(hi - lo, 1e-9)).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] confidence read failed ({e})")
        return None


def _mask_obj_by_iid(output_dir: Path) -> Dict[int, int]:
    """instance_id → seg_masks obj id (mesh_export convention: via
    segmentation.json, label fallback)."""
    out: Dict[int, int] = {}
    seg_json = output_dir / "segmentation.json"
    if not seg_json.exists():
        return out
    try:
        for e in (json.loads(seg_json.read_text()).get("instances") or []):
            if e.get("id") is not None and e.get("instance_id") is not None:
                out[int(e["instance_id"])] = int(e["id"])
    except Exception:  # noqa: BLE001
        pass
    return out


def published_mesh_path(output_dir: Path, label: str, iid: int) -> Optional[Path]:
    from segmentation.tsdf_export import _safe_label
    safe = _safe_label(label or "segment", int(iid))
    p = Path(output_dir) / "tsdf" / safe / f"{safe}.glb"
    return p if p.exists() else None


def _write_classification(output_dir: Path, instances: List[dict],
                          n_points: int) -> np.ndarray:
    """Rebuild classification.npy (per-point INSTANCE id; 0 = unsegmented) —
    it is THE color source the Potree converter bakes into the octree. Without
    this, an erase updated the indices but the rebuilt octree kept painting
    the removed points with their old segment color (user 2026-08-30)."""
    classification = np.zeros(n_points, dtype=np.uint8)
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < n_points)]
        # INSTANCE id, never the mask obj id: brush-created segments carry
        # id = mask oid (instance_id − 1), and writing THAT painted their
        # points with the PREVIOUS segment's class in the octree — the
        # "reassigned points answer to another segment's toggle" bug
        # (user 2026-08-31).
        classification[gi] = min(
            int(inst.get("instance_id", inst.get("id", 0))), 255)
    np.save(output_dir / "classification.npy", classification)
    return classification


def verify_octree_classification(output_dir: Path,
                                 sample: int = 5000) -> Optional[dict]:
    """Sample the ROOT node of the freshly-built Potree octree and check its
    per-point classification byte against classification.npy (matched by
    position on the conversion-source cloud). Closes the trace loop of a
    brush transaction: files → octree → what the viewer culls by
    (user 2026-08-31: 'cada transacción trazada y controlada').
    Returns {checked, matched, agreement} or None when unverifiable."""
    import struct
    output_dir = Path(output_dir)
    potree = output_dir / "potree" \
        if (output_dir / "potree").exists() else output_dir.parent / "potree"
    meta_p = potree / "metadata.json"
    cls_p = output_dir / "classification.npy"
    src = output_dir / "corrected_cloud.ply"
    if not src.exists():
        src = output_dir / "cleaned_cloud.ply"
    if not (meta_p.exists() and cls_p.exists() and src.exists()):
        return None
    try:
        meta = json.loads(meta_p.read_text())
        scale, offs = meta["scale"], meta["offset"]
        bpp = cls_off = 0
        cls_found = False
        for a in meta["attributes"]:
            if a["name"] == "classification":
                cls_off = bpp
                cls_found = True
            bpp += int(a["size"])
        if not cls_found or bpp <= 0:
            return None
        t, cm, npts, boff, bsz = struct.unpack(
            "<BBIQQ", (potree / "hierarchy.bin").read_bytes()[:22])
        with open(potree / "octree.bin", "rb") as f:
            f.seek(boff)
            buf = f.read(min(bsz, npts * bpp))
        n = min(int(npts), int(sample), len(buf) // bpp)
        if n < 100:
            return None
        rec = np.frombuffer(buf[:n * bpp], dtype=np.uint8).reshape(n, bpp)
        ixyz = rec[:, :12].copy().view("<i4").reshape(n, 3).astype(np.float64)
        pts = ixyz * np.asarray(scale) + np.asarray(offs)
        cls_oct = rec[:, cls_off].astype(np.int64)

        from segmentation.pipeline import _load_ply_origins
        origins = _load_ply_origins(src)
        if not origins:
            return None
        cloud = origins[0]
        gt = np.load(cls_p)
        if len(gt) != len(cloud):
            return {"checked": 0, "matched": 0, "agreement": 0.0,
                    "error": "classification/cloud size mismatch"}
        from scipy.spatial import cKDTree
        d, idx = cKDTree(cloud).query(pts, k=1)
        ok = d < 0.002    # 1 mm quantization → 2 mm tolerance
        if not ok.any():
            return {"checked": int(n), "matched": 0, "agreement": 0.0,
                    "error": "no positional matches — octree/cloud frames differ"}
        agree = float((cls_oct[ok] == gt[idx[ok]]).mean())
        return {"checked": int(ok.sum()),
                "matched": int((cls_oct[ok] == gt[idx[ok]]).sum()),
                "agreement": round(agree, 4)}
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] octree verification failed (non-fatal): {e}")
        return None


def erase_sphere(output_dir: Path, center_display, radius: float) -> dict:
    """Single-sphere compatibility wrapper over erase_spheres()."""
    return erase_spheres(output_dir,
                         [{"center": center_display, "radius": radius}])


_NEW_SEGMENT_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                       "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"]


def erase_spheres(output_dir: Path, spheres: List[dict],
                  target_iid: Optional[int] = None,
                  new_label: Optional[str] = None,
                  only_iids: Optional[List[int]] = None,
                  include_unsegmented: bool = True,
                  conf_below: Optional[float] = None) -> dict:
    """Apply ONE commit over the marked spheres (user 2026-08-29: mark zones
    first, then a single button applies everything — one mask edit, one OBB
    recompute, one octree rebuild).

    ``target_iid`` None → DELETE: points leave their segment (unsegmented).
    ``target_iid`` set → REASSIGN: every point inside the zones — owned by
    any other instance OR unsegmented — moves INTO the target segment (user:
    "no hay que borrarlos, son de otro segmento"). Mask pixels move with
    them (cleared at the source, set on the target) so the decision survives
    any re-match.
    ``new_label`` set → CREATE: a brand-new segment with that label is created
    from the zones and becomes the reassign target (user 2026-08-29: "se
    pueden asignar a un segmento existente o crear uno nuevo").
    ``only_iids`` (SAFETY, user 2026-08-30: "el selector está pasando por
    todos aunque no se vean — eso es peligroso"): when given, only these
    instance_ids can LOSE points — the UI sends the currently VISIBLE
    segments, so hidden ones are untouchable. The reassign target is exempt
    (chosen explicitly).
    ``conf_below`` (user 2026-08-31: brush confidence filter): points with
    normalized confidence BELOW this value are selected too — with no zones
    it selects globally. Delete-mode use: low-confidence points of the
    visible segments move to unsegmented. Same safety, ledger and undo.

    Returns {"touched": {iid: n_removed}, "total_removed": n, "reassigned": n,
    "mesh_instances": [...], "undo": {...}}."""
    from segmentation.pipeline import _load_ply_origins, _compute_obb

    output_dir = Path(output_dir)
    ply = output_dir / "cleaned_cloud.ply"
    res_path = output_dir / "segmentation_result.json"
    if not ply.exists() or not res_path.exists():
        raise FileNotFoundError("session has no cleaned cloud / segmentation")

    origins = _load_ply_origins(ply)
    if origins is None:
        raise RuntimeError("cloud has no origin fields (cannot edit masks)")
    xyz, fg, pr, pc = origins
    N = len(xyz)
    conf_norm = None
    if conf_below is not None:
        conf_norm = _load_confidence_norm(ply)
        if conf_norm is None or len(conf_norm) != N:
            raise ValueError("cloud has no usable confidence field")
        conf_below = float(conf_below)
    # zones live in the DISPLAY frame: a cube brush is axis-aligned in the
    # LEVELED frame the user sees (user 2026-08-30: sphere OR cube per zone),
    # which is NOT axis-aligned in raw — so all hit tests run on display pts
    s_ft, R_ft, t_ft = _load_floor_transform(output_dir)

    def _to_disp(p):
        return s_ft * (np.asarray(p, np.float64) @ R_ft.T) + t_ft

    zones = []
    for sp in spheres:
        kind = str(sp.get("shape") or "sphere").lower()
        if kind == "box":
            # GIZMO BOX (user 2026-08-30: "debe ser una caja, punto"): the
            # mesh is a 2x2x2 cube, so scale/rotation/position all live in the
            # matrix and the local test is simply |xyz| <= 1
            M = np.asarray(sp["matrix"], np.float64).reshape(4, 4, order="F")
            zones.append({"kind": "box", "inv": np.linalg.inv(M)})
        elif kind == "prism":
            # LASSO PRISM (user 2026-08-30): polygon (local XY) extruded along
            # local +Z by ``depth``; ``matrix`` maps local→display (three.js
            # toArray = column-major), gizmo edits (move/rotate/stretch) baked in
            M = np.asarray(sp["matrix"], np.float64).reshape(4, 4, order="F")
            zones.append({"kind": "prism", "inv": np.linalg.inv(M),
                          "poly": np.asarray(sp["polygon"], np.float64),
                          "depth": float(sp.get("depth") or 0.0)})
        else:
            zones.append({"kind": kind,
                          "c": np.asarray(sp["center"], np.float64),
                          "r": float(sp["radius"]),
                          "yaw": float(np.radians(float(sp.get("yaw_deg") or 0.0)))})
    if not zones and conf_below is None:
        return {"touched": {}, "total_removed": 0, "mesh_instances": [],
                "undo": None}

    def _zone_hit(disp_pts: np.ndarray) -> np.ndarray:
        h = np.zeros(len(disp_pts), dtype=bool)
        for z in zones:
            if z["kind"] == "box":
                ph = np.hstack([disp_pts, np.ones((len(disp_pts), 1))])
                local = (z["inv"] @ ph.T).T
                h |= (np.abs(local[:, :3]) <= 1.0).all(axis=1)
            elif z["kind"] == "prism":
                ph = np.hstack([disp_pts, np.ones((len(disp_pts), 1))])
                local = (z["inv"] @ ph.T).T
                zin = (local[:, 2] >= 0.0) & (local[:, 2] <= z["depth"])
                if zin.any():
                    from matplotlib.path import Path as _MplPath
                    inpoly = np.zeros(len(disp_pts), dtype=bool)
                    inpoly[zin] = _MplPath(z["poly"]).contains_points(
                        local[zin, :2])
                    h |= inpoly
            elif z["kind"] == "cube":
                h |= _cube_hit(disp_pts, z["c"], z["r"], z["yaw"])
            else:
                h |= ((disp_pts - z["c"]) ** 2).sum(axis=1) <= z["r"] * z["r"]
        return h

    result = json.loads(res_path.read_text())
    instances = result.get("instances") or []
    oid_map = _mask_obj_by_iid(output_dir)

    masks_path = output_dir / "seg_masks.npz"
    masks: Dict[str, np.ndarray] = {}
    if masks_path.exists():
        try:
            z = np.load(masks_path, allow_pickle=True)
            masks = {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001 — corrupt/empty npz: erase still
            # works on the point assignment; only the mask edit is skipped
            print(f"[Erase] seg_masks.npz unreadable ({e}) — erasing points "
                  "only (masks not edited)")
            masks = {}
    orig_h = float(pr.max() + 1)
    orig_w = float(pc.max() + 1)

    # user 2026-08-31: when the Unsegmented toggle is ON, its low-confidence
    # points have no "more unsegmented" to fall to — they are DELETED from
    # the cloud itself (irreversible; the ledger says so). Computed on the
    # PRE-commit ownership so points moved to unsegmented by THIS commit are
    # not swept along.
    pending_delete = None
    if conf_norm is not None and target_iid is None and include_unsegmented:
        _assigned_any = np.zeros(N, dtype=bool)
        for _inst in instances:
            _g = np.asarray(_inst.get("globalIndices") or [], dtype=np.int64)
            _assigned_any[_g[(_g >= 0) & (_g < N)]] = True
        pending_delete = np.nonzero((~_assigned_any)
                                    & (conf_norm < conf_below))[0]

    touched: Dict[int, int] = {}
    mesh_instances: List[int] = []
    undo_indices: Dict[str, List[int]] = {}
    undo_pixels: List[Tuple[str, List[int], List[int]]] = []

    created_iid = None
    if new_label:
        # CREATE a new segment and use it as the reassign target. It needs an
        # identity in BOTH spaces: instance_id (result/viewer) and mask obj id
        # (seg_masks/segmentation.json) so re-matching keeps it alive.
        new_iid = 1 + max(
            [int(i.get("instance_id", i.get("id", 0))) for i in instances],
            default=0)
        seg_json = output_dir / "segmentation.json"
        meta = {}
        try:
            meta = json.loads(seg_json.read_text()) if seg_json.exists() else {}
        except Exception:  # noqa: BLE001
            meta = {}
        meta_insts = meta.get("instances") or []
        new_oid = 1 + max([int(e.get("id", 0)) for e in meta_insts], default=0)
        color = _NEW_SEGMENT_COLORS[(new_iid - 1) % len(_NEW_SEGMENT_COLORS)]
        # result-file convention is id == instance_id (1-based); the mask obj
        # id (new_oid) lives ONLY in segmentation.json. Writing new_oid here
        # made every downstream .get("id")-first consumer (classification,
        # exports) treat the new segment as the PREVIOUS one.
        instances.append({
            "id": int(new_iid), "label": str(new_label),
            "instance_id": int(new_iid), "color": color,
            "total_points": 0, "globalIndices": [],
        })
        meta_insts.append({"id": int(new_oid), "instance_id": int(new_iid),
                           "label": str(new_label), "color": color})
        meta["instances"] = meta_insts
        try:
            seg_json.write_text(json.dumps(meta, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[Erase] segmentation.json update failed: {e}")
        if masks and "obj_ids" in masks:
            masks["obj_ids"] = np.append(
                np.asarray(masks["obj_ids"], dtype=np.int32),
                np.int32(new_oid))
        oid_map[int(new_iid)] = int(new_oid)
        target_iid = int(new_iid)
        created_iid = int(new_iid)
        print(f"[Erase] created new segment '{new_label}' "
              f"(instance {new_iid}, mask obj {new_oid})")

    target_inst = None
    assigned_before = None
    moved_parts: List[np.ndarray] = []
    if target_iid is not None:
        for inst in instances:
            if int(inst.get("instance_id", inst.get("id"))) == int(target_iid):
                target_inst = inst
                break
        if target_inst is None:
            raise ValueError(f"reassign target instance {target_iid} not found")
        assigned_before = np.zeros(N, dtype=bool)
        for inst in instances:
            gi0 = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
            assigned_before[gi0[(gi0 >= 0) & (gi0 < N)]] = True

    only_set = set(int(i) for i in only_iids) if only_iids is not None else None
    # Transaction ledger (user 2026-08-31: every brush transaction traced and
    # controlled): per-source breakdown, silently-protected hidden points,
    # conservation balance, post-write file verification.
    labels_by_iid = {
        int(i.get("instance_id", i.get("id"))): str(i.get("label", "segment"))
        for i in instances}
    assigned_points_before = sum(
        len(i.get("globalIndices") or []) for i in instances)
    protected_hidden: Dict[int, int] = {}
    for inst in instances:
        if target_inst is not None and inst is target_inst:
            continue   # the target only GAINS points in a reassign
        iid = int(inst.get("instance_id", inst.get("id")))
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < N)]
        if not len(gi):
            continue
        hit = _zone_hit(_to_disp(xyz[gi]))
        if conf_norm is not None:
            hit |= conf_norm[gi] < conf_below
        n_hit = int(hit.sum())
        if not n_hit:
            continue
        if only_set is not None and iid not in only_set:
            # hidden in the viewer → untouchable (safety), but never silent:
            # the ledger reports what stayed put inside the zones
            protected_hidden[iid] = n_hit
            continue
        removed = gi[hit]
        keep = gi[~hit]
        if target_inst is not None:
            moved_parts.append(removed)
        inst["globalIndices"] = keep.tolist()
        inst["total_points"] = int(len(keep))
        touched[iid] = n_hit
        undo_indices[str(iid)] = removed.tolist()
        # immediate OBB from the remaining points (display frame); once the
        # mesh regenerates, the caller replaces it with the mesh-derived OBB
        if len(keep) >= 4:
            inst["obb"] = _compute_obb(raw_to_display(output_dir, xyz[keep]))
        if published_mesh_path(output_dir, inst.get("label", ""), iid):
            mesh_instances.append(iid)
        # clear the removed points' pixels in this instance's masks — the
        # mask is the truth; re-matching must not resurrect the deletion
        oid = oid_map.get(iid)
        if oid is None or not masks:
            continue
        for f in np.unique(fg[removed]):
            key = f"f{int(f)}_o{int(oid)}"
            m = masks.get(key)
            if m is None:
                continue
            sel = removed[fg[removed] == f]
            mh, mw = m.shape[:2]
            rr = np.clip((pr[sel] * (mh / orig_h)).astype(np.int64), 0, mh - 1)
            cc = np.clip((pc[sel] * (mw / orig_w)).astype(np.int64), 0, mw - 1)
            on = m[rr, cc] > 0
            if not on.any():
                continue
            undo_pixels.append((key, rr[on].tolist(), cc[on].tolist()))
            m = m.copy()
            m[rr[on], cc[on]] = 0
            masks[key] = m

    # ── REASSIGN: the target gains every point in the zones — the ones taken
    # from other instances plus the previously-unsegmented ones
    n_reassigned = 0
    n_unseg_taken = 0
    target_pixels: List[Tuple[str, List[int], List[int]]] = []
    if target_inst is not None:
        # unsegmented capture honors the panel toggle (user 2026-08-30: los
        # unsegmented solo son seleccionables cuando Unsegmented está visible)
        un_idx = np.nonzero(~assigned_before)[0] if include_unsegmented \
            else np.empty(0, dtype=np.int64)
        if len(un_idx):
            hit_u = _zone_hit(_to_disp(xyz[un_idx]))
            if hit_u.any():
                moved_parts.append(un_idx[hit_u])
                n_unseg_taken = int(hit_u.sum())
        if moved_parts:
            added = np.unique(np.concatenate(moved_parts))
            tgi = np.asarray(target_inst.get("globalIndices") or [],
                             dtype=np.int64)
            merged = np.union1d(tgi, added)
            target_inst["globalIndices"] = merged.tolist()
            target_inst["total_points"] = int(len(merged))
            n_reassigned = int(len(added))
            if len(merged) >= 4:
                target_inst["obb"] = _compute_obb(
                    raw_to_display(output_dir, xyz[merged]))
            t_id = int(target_inst.get("instance_id", target_inst.get("id")))
            if published_mesh_path(output_dir,
                                   target_inst.get("label", ""), t_id):
                mesh_instances.append(t_id)
            # paint the moved points ONTO the target's masks so a re-match
            # keeps the reassignment (create the frame's mask if absent)
            toid = oid_map.get(t_id)
            if toid is not None and masks:
                try:
                    sr = masks.get("scaled_res")
                    mh0, mw0 = (int(sr[0]), int(sr[1])) if sr is not None \
                        else next(m.shape[:2] for k, m in masks.items()
                                  if k.startswith("f"))
                    for f in np.unique(fg[added]):
                        key = f"f{int(f)}_o{int(toid)}"
                        m = masks.get(key)
                        if m is None:
                            m = np.zeros((mh0, mw0), dtype=np.uint8)
                        sel = added[fg[added] == f]
                        mh, mw = m.shape[:2]
                        rr = np.clip((pr[sel] * (mh / orig_h)).astype(np.int64),
                                     0, mh - 1)
                        cc = np.clip((pc[sel] * (mw / orig_w)).astype(np.int64),
                                     0, mw - 1)
                        off = m[rr, cc] == 0
                        if not off.any():
                            continue
                        target_pixels.append((key, rr[off].tolist(),
                                              cc[off].tolist()))
                        m = m.copy()
                        m[rr[off], cc[off]] = 1
                        masks[key] = m
                except Exception as e:  # noqa: BLE001
                    print(f"[Erase] target mask paint failed (non-fatal): {e}")

    if not touched and not n_reassigned \
            and not (pending_delete is not None and len(pending_delete)):
        if created_iid is not None:
            # nothing landed in the new segment — roll its registration back
            try:
                seg_json = output_dir / "segmentation.json"
                meta = json.loads(seg_json.read_text())
                meta["instances"] = [e for e in (meta.get("instances") or [])
                                     if int(e.get("instance_id", -1)) != created_iid]
                seg_json.write_text(json.dumps(meta, indent=2))
            except Exception:  # noqa: BLE001
                pass
        return {"touched": {}, "total_removed": 0, "reassigned": 0,
                "mesh_instances": [], "undo": None,
                "ledger": {
                    "mode": "reassign" if target_iid is not None else "delete",
                    "target": ({"instance_id": int(target_iid),
                                "label": labels_by_iid.get(int(target_iid), "?")}
                               if target_iid is not None else None),
                    "moved_from": {}, "unsegmented_taken": 0,
                    "total_moved": 0,
                    "protected_hidden": {
                        f"{labels_by_iid.get(i, 'segment')}_{i}": n
                        for i, n in protected_hidden.items()},
                    "balance": None, "files_verified": None,
                }}

    if (undo_pixels or target_pixels) and masks_path.exists():
        _atomic_savez(masks_path, masks)

    # ── PHYSICAL DELETION of low-confidence unsegmented points ──
    n_deleted = 0
    if pending_delete is not None and len(pending_delete):
        keep = np.ones(N, dtype=bool)
        keep[pending_delete] = False
        ok_del = _rewrite_ply_keep(output_dir / "cleaned_cloud.ply", keep)
        corr = output_dir / "corrected_cloud.ply"
        if ok_del and corr.exists():
            ok_del = _rewrite_ply_keep(corr, keep)
        if ok_del:
            # every stored index shifts — remap all instances onto the new
            # cloud (deleted points were unsegmented, so instances only shift)
            new_idx = np.full(N, -1, dtype=np.int64)
            new_idx[keep] = np.arange(int(keep.sum()), dtype=np.int64)
            for inst in instances:
                _g = np.asarray(inst.get("globalIndices") or [],
                                dtype=np.int64)
                _g = new_idx[_g[(_g >= 0) & (_g < N)]]
                _g = _g[_g >= 0]
                inst["globalIndices"] = _g.tolist()
                inst["total_points"] = int(len(_g))
            n_deleted = int(len(pending_delete))
            N = int(keep.sum())
            result["total_points"] = N
            xyz = xyz[keep]
            # index remap invalidates this commit's undo — it is irreversible
            undo_indices.clear()
            undo_pixels.clear()
            print(f"[Erase] 🗑 {n_deleted:,} low-confidence unsegmented "
                  f"point(s) permanently removed from the cloud "
                  f"({N:,} remain)")

    result["segmented_points"] = sum(
        int(i.get("total_points") or 0) for i in instances)
    result["coverage"] = round(
        result["segmented_points"] / max(1, int(result.get("total_points") or N)), 4)
    res_path.write_text(json.dumps(result))
    classification = _write_classification(output_dir, instances, N)

    # ── FILE VERIFICATION: what we just wrote must balance exactly —
    # classification.npy per-class counts == each instance's index count, and
    # points are conserved (assigned_after − assigned_before == what left /
    # entered unsegmented). Any mismatch is reported, never swallowed.
    counts = np.bincount(classification, minlength=256)
    files_verified = True
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        if iid > 255:
            continue
        if int(counts[iid]) != len(inst.get("globalIndices") or []):
            files_verified = False
            print(f"[Erase] ⚠ VERIFY FAILED: class {iid} has "
                  f"{int(counts[iid])} pts in classification.npy vs "
                  f"{len(inst.get('globalIndices') or [])} in result json")
    assigned_points_after = sum(
        len(i.get("globalIndices") or []) for i in instances)
    # EXCLUSIVITY INVARIANT (user 2026-08-31): every point owned by exactly
    # one segment — verified on EVERY transaction, mismatches never silent.
    _all_gi = [np.asarray(i.get("globalIndices") or [], dtype=np.int64)
               for i in instances if i.get("globalIndices")]
    _cat = np.concatenate(_all_gi) if _all_gi else np.empty(0, dtype=np.int64)
    overlap_points = int(len(_cat) - len(np.unique(_cat)))
    if overlap_points:
        print(f"[Erase] ⚠ EXCLUSIVITY VIOLATION: {overlap_points:,} point(s) "
              f"owned by more than one segment")
    if target_inst is not None:
        expected_delta = n_unseg_taken          # net gain = unsegmented drawn in
    else:
        expected_delta = -int(sum(touched.values()))   # deletions leave the set
    balance = {
        "cloud_points": int(N),
        "assigned_before": int(assigned_points_before),
        "assigned_after": int(assigned_points_after),
        "expected_delta": int(expected_delta),
        "consistent": (assigned_points_after - assigned_points_before
                       == expected_delta),
    }
    if not balance["consistent"]:
        print(f"[Erase] ⚠ BALANCE MISMATCH: {balance}")

    t_id_led = (int(target_inst.get("instance_id", target_inst.get("id")))
                if target_inst is not None else None)
    ledger = {
        "mode": "reassign" if target_inst is not None else "delete",
        "target": ({"instance_id": t_id_led,
                    "label": str(target_inst.get("label", "segment"))}
                   if target_inst is not None else None),
        "moved_from": {f"{labels_by_iid.get(i, 'segment')}_{i}": int(n)
                       for i, n in touched.items()},
        "unsegmented_taken": int(n_unseg_taken),
        "total_moved": int(n_reassigned) if target_inst is not None
                       else int(sum(touched.values())),
        "protected_hidden": {f"{labels_by_iid.get(i, 'segment')}_{i}": int(n)
                             for i, n in protected_hidden.items()},
        "balance": balance,
        "files_verified": files_verified,
        "exclusive": overlap_points == 0,
        "overlap_points": overlap_points,
        "conf_below": conf_below,
        "deleted_points": n_deleted,
        "irreversible": n_deleted > 0,
    }
    print(f"[Erase] ledger: {json.dumps(ledger)}")

    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] store rebuild failed (non-fatal): {e}")

    if n_deleted:
        # physical deletion shifted every index — this commit cannot be undone
        undo = None
        return {"touched": {int(k): v for k, v in touched.items()},
                "total_removed": int(sum(touched.values())),
                "reassigned": int(n_reassigned),
                "mesh_instances": sorted(set(mesh_instances)),
                "undo": None,
                "ledger": ledger}

    undo = {"indices": undo_indices, "pixels": undo_pixels}
    if target_inst is not None and n_reassigned:
        undo["target"] = int(target_inst.get("instance_id",
                                             target_inst.get("id")))
        undo["target_added"] = np.unique(
            np.concatenate(moved_parts)).tolist() if moved_parts else []
        undo["target_pixels"] = target_pixels
        if created_iid is not None:
            undo["created"] = created_iid
            undo["created_oid"] = oid_map.get(created_iid)
    return {"touched": {int(k): v for k, v in touched.items()},
            "total_removed": int(sum(touched.values())),
            "reassigned": int(n_reassigned),
            "mesh_instances": sorted(set(mesh_instances)),
            "undo": undo,
            "ledger": ledger}


def undo_erase(output_dir: Path, undo: dict) -> dict:
    """Restore one erase stroke (indices back into their instances, mask
    pixels back on). Returns {restored: n}."""
    from segmentation.pipeline import _load_ply_origins, _compute_obb

    output_dir = Path(output_dir)
    res_path = output_dir / "segmentation_result.json"
    result = json.loads(res_path.read_text())
    by_iid = {int(i.get("instance_id", i.get("id"))): i
              for i in (result.get("instances") or [])}
    origins = _load_ply_origins(output_dir / "cleaned_cloud.ply")
    xyz = origins[0] if origins else None

    restored = 0
    for iid_s, idxs in (undo.get("indices") or {}).items():
        inst = by_iid.get(int(iid_s))
        if inst is None:
            continue
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        merged = np.union1d(gi, np.asarray(idxs, dtype=np.int64))
        inst["globalIndices"] = merged.tolist()
        inst["total_points"] = int(len(merged))
        restored += len(idxs)
        if xyz is not None and len(merged) >= 4:
            inst["obb"] = _compute_obb(raw_to_display(output_dir, xyz[merged]))

    # reassign rollback: the target gives the gained points back; a CREATED
    # segment is removed entirely (result + segmentation.json + its masks)
    t_iid = undo.get("target")
    created = undo.get("created")
    if created is not None:
        result["instances"] = [
            i for i in (result.get("instances") or [])
            if int(i.get("instance_id", i.get("id"))) != int(created)]
        try:
            seg_json = output_dir / "segmentation.json"
            meta = json.loads(seg_json.read_text())
            meta["instances"] = [e for e in (meta.get("instances") or [])
                                 if int(e.get("instance_id", -1)) != int(created)]
            seg_json.write_text(json.dumps(meta, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[Erase] undo: segmentation.json rollback failed: {e}")
    elif t_iid is not None:
        inst = by_iid.get(int(t_iid))
        added = np.asarray(undo.get("target_added") or [], dtype=np.int64)
        if inst is not None and len(added):
            gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
            keep = np.setdiff1d(gi, added)
            inst["globalIndices"] = keep.tolist()
            inst["total_points"] = int(len(keep))
            if xyz is not None and len(keep) >= 4:
                inst["obb"] = _compute_obb(
                    raw_to_display(output_dir, xyz[keep]))

    masks_path = output_dir / "seg_masks.npz"
    if (undo.get("pixels") or undo.get("target_pixels")) and masks_path.exists():
        try:
            z = np.load(masks_path, allow_pickle=True)
            masks = {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001
            print(f"[Erase] undo: seg_masks.npz unreadable ({e}) — "
                  "points restored, masks untouched")
            masks = None
        if masks is not None:
            for key, rr, cc in (undo.get("pixels") or []):
                m = masks.get(key)
                if m is None:
                    continue
                m = m.copy()
                m[np.asarray(rr, np.int64), np.asarray(cc, np.int64)] = 1
                masks[key] = m
            for key, rr, cc in (undo.get("target_pixels") or []):
                m = masks.get(key)
                if m is None:
                    continue
                m = m.copy()
                m[np.asarray(rr, np.int64), np.asarray(cc, np.int64)] = 0
                masks[key] = m
            c_oid = undo.get("created_oid")
            if undo.get("created") is not None and c_oid is not None:
                for key in [k for k in masks
                            if k.endswith(f"_o{int(c_oid)}")]:
                    del masks[key]
                if "obj_ids" in masks:
                    arr = np.asarray(masks["obj_ids"], dtype=np.int32)
                    masks["obj_ids"] = arr[arr != np.int32(c_oid)]
            _atomic_savez(masks_path, masks)

    result["segmented_points"] = sum(
        int(i.get("total_points") or 0) for i in (result.get("instances") or []))
    result["coverage"] = round(
        result["segmented_points"] / max(1, int(result.get("total_points") or 1)), 4)
    res_path.write_text(json.dumps(result))
    if xyz is not None:
        _write_classification(output_dir, result.get("instances") or [],
                              len(xyz))
    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] store rebuild failed (non-fatal): {e}")
    return {"restored": restored}


def crop_glb_sphere(glb_path: Path, output_dir: Path, center_display,
                    radius: float, shape: str = "sphere",
                    yaw_deg: float = 0.0) -> bool:
    """Best-effort INSTANT visual crop: drop the published GLB's faces whose
    vertices fall inside the erase zone (sphere or display-axis-aligned cube).
    The definitive mesh comes from the debounced re-fit; this only keeps the
    screen honest meanwhile. Returns False (file untouched) on any trouble."""
    try:
        import trimesh
        s_ft, R_ft, t_ft = _load_floor_transform(output_dir)
        c = np.asarray(center_display, np.float64)
        r = float(radius)
        scene = trimesh.load(str(glb_path), force="scene")
        changed = False
        for name, geom in list(scene.geometry.items()):
            v = s_ft * (np.asarray(geom.vertices) @ R_ft.T) + t_ft
            if str(shape).lower() == "cube":
                inside = _cube_hit(v, c, r, float(np.radians(yaw_deg)))
            else:
                inside = ((v - c) ** 2).sum(axis=1) <= r * r
            if not inside.any():
                continue
            faces = np.asarray(geom.faces)
            keepf = ~inside[faces].any(axis=1)
            if keepf.all():
                continue
            geom.update_faces(keepf)
            geom.remove_unreferenced_vertices()
            changed = True
        if changed:
            scene.export(str(glb_path))
        return changed
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] instant GLB crop skipped ({glb_path.name}): {e}")
        return False


def obb_from_mesh(glb_path: Path, output_dir: Path) -> Optional[dict]:
    """OBB from the published mesh's vertices, in DISPLAY frame (user rule:
    once a mesh exists, the bbox comes from the mesh — flatter, tighter)."""
    try:
        import trimesh
        from segmentation.pipeline import _compute_obb
        scene = trimesh.load(str(glb_path), force="scene")
        vs = [np.asarray(g.vertices) for g in scene.geometry.values()
              if len(getattr(g, "vertices", []))]
        if not vs:
            return None
        v = np.concatenate(vs)
        if len(v) < 4:
            return None
        return _compute_obb(raw_to_display(output_dir, v))
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] mesh OBB failed ({glb_path.name}): {e}")
        return None
