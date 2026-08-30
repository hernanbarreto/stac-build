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
                          n_points: int) -> None:
    """Rebuild classification.npy (per-point mask-obj id; 0 = unsegmented) —
    it is THE color source the Potree converter bakes into the octree. Without
    this, an erase updated the indices but the rebuilt octree kept painting
    the removed points with their old segment color (user 2026-08-30)."""
    classification = np.zeros(n_points, dtype=np.uint8)
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < n_points)]
        classification[gi] = min(int(inst.get("id", 0)), 255)
    np.save(output_dir / "classification.npy", classification)


def erase_sphere(output_dir: Path, center_display, radius: float) -> dict:
    """Single-sphere compatibility wrapper over erase_spheres()."""
    return erase_spheres(output_dir,
                         [{"center": center_display, "radius": radius}])


_NEW_SEGMENT_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                       "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"]


def erase_spheres(output_dir: Path, spheres: List[dict],
                  target_iid: Optional[int] = None,
                  new_label: Optional[str] = None,
                  only_iids: Optional[List[int]] = None) -> dict:
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
    # zones live in the DISPLAY frame: a cube brush is axis-aligned in the
    # LEVELED frame the user sees (user 2026-08-30: sphere OR cube per zone),
    # which is NOT axis-aligned in raw — so all hit tests run on display pts
    s_ft, R_ft, t_ft = _load_floor_transform(output_dir)

    def _to_disp(p):
        return s_ft * (np.asarray(p, np.float64) @ R_ft.T) + t_ft

    zones = []
    for sp in spheres:
        kind = str(sp.get("shape") or "sphere").lower()
        zones.append((kind, np.asarray(sp["center"], np.float64),
                      float(sp["radius"])))
    if not zones:
        return {"touched": {}, "total_removed": 0, "mesh_instances": [],
                "undo": None}

    def _zone_hit(disp_pts: np.ndarray) -> np.ndarray:
        h = np.zeros(len(disp_pts), dtype=bool)
        for kind, c, r in zones:
            if kind == "cube":
                h |= (np.abs(disp_pts - c) <= r).all(axis=1)
            else:
                h |= ((disp_pts - c) ** 2).sum(axis=1) <= r * r
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
        instances.append({
            "id": int(new_oid), "label": str(new_label),
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
    for inst in instances:
        if target_inst is not None and inst is target_inst:
            continue   # the target only GAINS points in a reassign
        iid = int(inst.get("instance_id", inst.get("id")))
        if only_set is not None and iid not in only_set:
            continue   # hidden in the viewer → untouchable (safety)
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < N)]
        if not len(gi):
            continue
        hit = _zone_hit(_to_disp(xyz[gi]))
        n_hit = int(hit.sum())
        if not n_hit:
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
    target_pixels: List[Tuple[str, List[int], List[int]]] = []
    if target_inst is not None:
        un_idx = np.nonzero(~assigned_before)[0]
        if len(un_idx):
            hit_u = _zone_hit(_to_disp(xyz[un_idx]))
            if hit_u.any():
                moved_parts.append(un_idx[hit_u])
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

    if not touched and not n_reassigned:
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
                "mesh_instances": [], "undo": None}

    if (undo_pixels or target_pixels) and masks_path.exists():
        _atomic_savez(masks_path, masks)
    result["segmented_points"] = sum(
        int(i.get("total_points") or 0) for i in instances)
    result["coverage"] = round(
        result["segmented_points"] / max(1, int(result.get("total_points") or N)), 4)
    res_path.write_text(json.dumps(result))
    _write_classification(output_dir, instances, N)

    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] store rebuild failed (non-fatal): {e}")

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
            "undo": undo}


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
                    radius: float, shape: str = "sphere") -> bool:
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
                inside = (np.abs(v - c) <= r).all(axis=1)
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
