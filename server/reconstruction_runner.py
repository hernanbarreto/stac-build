"""
Reconstruction v2 runner — orchestrate classify → reconstruct → assemble for a
whole session, write per-element GLBs + a `scene.json`.

Works on the cleaned point cloud (raw capture frame — the viewer re-applies the
floor transform), the SAM3 `segmentation_result.json`, the camera poses, and any
ShapeR GLBs already produced for the free-form objects. It is **additive**: it
does not touch the existing `/shape/export` flow; a separate endpoint triggers it
and the viewer can load `scene.json` instead of the raw shape list.

Output layout: ``<output_dir>/scene/<safe_label>_<id>.glb`` + ``.meta.json`` and
``<output_dir>/scene/scene.json``.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Hard caps so a single run stays inside the host's RAM (12 GB on the CPU-only
# box). Without these, a 12.5M-pt LiDAR cloud + 300+ camera views is enough to
# run the machine into swap; two of these in parallel froze it outright
# (2026-05-12 — two `[ReconV2] start` lines in the log → OOM → host hang). The
# full cloud is still streamed to the viewer via Potree; these caps only bound
# the geometry fits / raycasting / silhouette rasterisation.
MAX_INSTANCE_PTS = 150_000   # per-instance sub-cloud cap (RANSAC / PCA / hull / shrink-wrap)
MAX_HC_PTS = 80_000          # high-confidence sub-cloud cap (opening detection, wrap target)
DEFAULT_MAX_VIEWS = 150      # camera views fed to occlusion + visual hull (was 500)


def _emit(cb: Optional[Callable[[dict], None]], **kw) -> None:
    """Best-effort progress callback — never let a UI hook break the run."""
    if cb is None:
        return
    try:
        cb(dict(kw))
    except Exception:
        pass

from reconstruction import (high_confidence_subcloud, load_ply_xyz_conf,
                            classify_instance, reconstruct_element,
                            reconstruct_mesh_from_glb, write_element_glb,
                            CameraView, assemble_scene, write_scene_json)
from reconstruction.elements import MeshElement, json_safe


def _safe_label(label: str, iid: int) -> str:
    return f"{str(label).replace(' ', '_').replace('/', '_')[:30]}_{int(iid)}"


def _world_up_vec(s: str) -> np.ndarray:
    s = (s or "+y").strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    axis = s[-1]
    v = np.zeros(3)
    v["xyz".index(axis)] = sign
    return v


def _selected_frame_indices(frames_dir: Path):
    """Indices of the keyframes that were actually used to build the cloud — the
    ones that passed the blur + cosine-novelty filters (``selected_frames.json``,
    falling back to ``frame_quality.json``). Returns None if no filter file → use all."""
    fd = Path(frames_dir)
    names = None
    sel = fd / "selected_frames.json"
    fq = fd / "frame_quality.json"
    try:
        if sel.exists():
            try:
                from frame_selector import load_selected_frames
                names = load_selected_frames(str(fd))
            except Exception:
                names = json.loads(sel.read_text())
        elif fq.exists():
            data = json.loads(fq.read_text())
            if isinstance(data, dict):
                names = [k for k, v in data.items() if (v if isinstance(v, bool) else v.get("ok", True))] \
                    if data else None
            elif isinstance(data, list):
                names = data
    except Exception:
        names = None
    if not names:
        return None
    out = set()
    for x in names:
        try:
            if isinstance(x, (int, np.integer)):
                out.add(int(x))
            else:
                out.add(int(Path(str(x)).stem))
        except Exception:
            pass
    return out or None


def _load_cameras(session_dir: Path, output_dir: Path):
    """The camera source (pose_map / K_for / source_resolution) or None."""
    try:
        from segmentation.shaper_export import _load_camera_source
    except Exception:
        return None
    try:
        return _load_camera_source(Path(session_dir), Path(output_dir))
    except Exception:
        return None


def _cam_res(cam) -> tuple:
    res = getattr(cam, "source_resolution", None)
    H = int(res[0]) if res else 1080
    W = int(res[1]) if res else 1920
    return H, W


def _c2w4(c2w) -> np.ndarray:
    c4 = np.eye(4, dtype=np.float64)
    c2w = np.asarray(c2w, dtype=np.float64)
    c4[:c2w.shape[0], :c2w.shape[1]] = c2w
    return c4


def _build_camera_views(cam, frames_dir: Path, max_views: int = 150) -> List[CameraView]:
    """A representative spread of `CameraView`s (``mask=None``) over the keyframes
    the cloud was built from — used by the occlusion reasoning (cloud + ray-cast,
    which doesn't need masks). Mask-aware views (for the visual-hull carve of mesh
    elements) are per-instance — see `_instance_camera_views`."""
    if cam is None:
        return []
    H, W = _cam_res(cam)
    sel = _selected_frame_indices(frames_dir)
    fids = sorted(cam.pose_map.keys())
    if sel:
        kept = [fi for fi in fids if fi in sel]
        if kept:
            fids = kept
    if len(fids) > max_views:
        step = len(fids) / float(max_views)
        fids = [fids[int(i * step)] for i in range(max_views)]
    views: List[CameraView] = []
    for fi in fids:
        c2w = cam.pose_map.get(fi)
        K = cam.K_for(fi)
        if c2w is None or K is None:
            continue
        views.append(CameraView(c2w=_c2w4(c2w), K=np.asarray(K, dtype=np.float64),
                                width=W, height=H, mask=None))
    return views


def _load_mask_store(output_dir: Path):
    """Open ``seg_masks.npz`` (lazily) → dict with the NpzFile + scaled_res, or None.
    Keys in the NPZ are ``f{frame_idx}_o{instance_id}`` → uint8 (H,W) mask."""
    p = Path(output_dir) / "seg_masks.npz"
    if not p.exists():
        return None
    try:
        data = np.load(p)
        sres = data["scaled_res"].tolist() if "scaled_res" in data.files else None
        return {"data": data, "scaled_res": sres, "files": set(data.files)}
    except Exception:
        return None


def _instance_camera_views(cam, mask_store, iid: int, frames_dir: Path,
                           max_views: int = 320, max_dim: int = 512) -> List[CameraView]:
    """`CameraView`s for ONE instance, each carrying that instance's SAM3 mask, so
    `carve_to_visual_hull` can trim over-grown ShapeR geometry and `silhouette_iou`
    can score the result. Uses **every** keyframe the instance is visible in (more
    views ⇒ tighter visual hull) — ``max_views`` is just a safety valve for a
    pathologically large scan. Masks are downscaled (K scaled to match) so memory
    stays bounded. Returns [] if no masks for this instance."""
    if cam is None or mask_store is None:
        return []
    data = mask_store["data"]
    files = mask_store["files"]
    iid = int(iid)
    frame_ids = []
    for f in files:
        if not (f.startswith("f") and "_o" in f):
            continue
        body = f[1:]
        if "_o" not in body:
            continue
        fidx, oidx = body.split("_o", 1)
        if fidx.isdigit() and oidx.isdigit() and int(oidx) == iid:
            frame_ids.append(int(fidx))
    frame_ids = sorted(set(frame_ids) & set(cam.pose_map.keys()))
    sel = _selected_frame_indices(frames_dir)
    if sel:
        kept = [fi for fi in frame_ids if fi in sel]
        if kept:
            frame_ids = kept
    if not frame_ids:
        return []
    if max_views and len(frame_ids) > max_views:        # only kicks in on huge scans
        step = len(frame_ids) / float(max_views)
        frame_ids = [frame_ids[int(i * step)] for i in range(max_views)]
    src_H, src_W = _cam_res(cam)
    try:
        import cv2
    except Exception:
        cv2 = None
    views: List[CameraView] = []
    for fi in frame_ids:
        c2w = cam.pose_map.get(fi)
        K = cam.K_for(fi)
        if c2w is None or K is None:
            continue
        try:
            m = np.asarray(data[f"f{fi}_o{iid}"])
        except Exception:
            continue
        if m.ndim != 2 or m.size == 0:
            continue
        mh, mw = int(m.shape[0]), int(m.shape[1])
        ds = min(1.0, float(max_dim) / float(max(mh, mw, 1)))
        out_h, out_w = max(1, int(round(mh * ds))), max(1, int(round(mw * ds)))
        if cv2 is not None and (out_h != mh or out_w != mw):
            mm = cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mm = m.astype(bool)
            out_h, out_w = mh, mw
        # K is at the camera source resolution; the mask is at (mw, mh); scale K
        # from source → mask-full → downscaled.
        sx = (mw / float(src_W)) * (out_w / float(mw))
        sy = (mh / float(src_H)) * (out_h / float(mh))
        Kk = np.asarray(K, dtype=np.float64).copy()
        Kk[0, 0] *= sx; Kk[0, 2] *= sx
        Kk[1, 1] *= sy; Kk[1, 2] *= sy
        views.append(CameraView(c2w=_c2w4(c2w), K=Kk, width=out_w, height=out_h, mask=mm))
    return views


def _load_tsdf_mesh(output_dir: Path, label: str, iid: int):
    """The TSDF mesh produced for this instance (``output/tsdf/<folder>/<folder>.glb``),
    if any — a dense, *smooth* surface that's a safe shrink-wrap target for the
    ShapeR mesh (and the `tsdf_divergence` cross-check). Returns a trimesh or None."""
    folder = _safe_label(label, iid)
    cand = Path(output_dir) / "tsdf" / folder / f"{folder}.glb"
    if not cand.exists():
        return None
    try:
        import trimesh
        m = trimesh.load(str(cand), force="mesh")
        if m is not None and getattr(m, "vertices", None) is not None and len(m.vertices) >= 8:
            return m
    except Exception:
        pass
    return None


def _detect_world_up(cam_positions: np.ndarray, points: np.ndarray) -> np.ndarray:
    try:
        from segmentation.shaper_export import _detect_world_up as _du
        return _world_up_vec(_du(np.asarray(cam_positions, dtype=np.float64),
                                 np.asarray(points, dtype=np.float64)))
    except Exception:
        # fallback: axis of least variance in camera positions, signed toward the cloud
        if len(cam_positions) >= 3:
            v = np.var(cam_positions, axis=0)
            ax = int(np.argmin(v))
            sign = 1.0 if (np.mean(cam_positions[:, ax]) - np.mean(points[:, ax])) >= 0 else -1.0
            u = np.zeros(3); u[ax] = sign
            return u
        return np.array([0.0, 1.0, 0.0])


def _placeholder_mesh_from_cloud(xyz: np.ndarray):
    """Convex hull of a point cloud → (verts, faces). Used when an object has no
    ShapeR GLB yet, so the scene isn't missing it."""
    try:
        import trimesh
        m = trimesh.points.PointCloud(np.asarray(xyz, dtype=np.float64)).convex_hull
        return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)
    except Exception:
        return None, None


def run_reconstruction(session_id: str, output_dir, frames_dir,
                       session_dir=None, only_obj_ids: Optional[List[int]] = None,
                       use_shaper_glbs: bool = True,
                       max_views: int = DEFAULT_MAX_VIEWS,
                       max_instance_pts: int = MAX_INSTANCE_PTS,
                       progress_cb: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
    """Run reconstruction v2 for a session. Returns a summary dict.

    Args:
        output_dir: the session's ``output/`` directory.
        frames_dir: the RGB frames directory.
        session_dir: parent of ``frames_dir`` (autodetected if None).
        only_obj_ids: restrict to these instance ids (None = all).
        use_shaper_glbs: if True, free-form objects use the ShapeR GLB the
            subprocess produced (under ``output/shape/<folder>/<folder>.glb``);
            otherwise a convex-hull placeholder is used.
        max_views: cap on camera views fed to occlusion / visual-hull reasoning.
        max_instance_pts: cap on points per instance before the geometry fits
            (RANSAC / PCA / hull / shrink-wrap) — the rest is randomly dropped.
        progress_cb: optional callback receiving ``{phase, ...}`` dicts so the
            caller can surface live progress (runs in this thread).
    """
    t0 = time.time()
    rng = np.random.default_rng(0)
    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)
    session_dir = Path(session_dir) if session_dir else frames_dir.parent
    print(f"[ReconV2] start  session={session_id}  output_dir={output_dir}")

    # cloud + confidence
    ply_path = output_dir / "cleaned_cloud.ply"
    if not ply_path.exists():
        for alt in ("corrected_cloud.ply", "cleaned_cloud_symlink.ply", "merged.ply"):
            if (output_dir / alt).exists():
                ply_path = output_dir / alt
                break
    loaded = load_ply_xyz_conf(ply_path)
    if loaded is None:
        return {"ok": False, "error": f"cannot read cloud {ply_path}"}
    xyz, conf, rgb = loaded
    n_pts = len(xyz)
    print(f"[ReconV2] cloud: {n_pts:,} pts, confidence={'yes' if conf is not None else 'no'}, rgb={'yes' if rgb is not None else 'no'}")
    _emit(progress_cb, phase="loaded_cloud", n_points=int(n_pts))

    # instances
    seg_path = output_dir / "segmentation_result.json"
    if not seg_path.exists():
        return {"ok": False, "error": "no segmentation_result.json"}
    seg = json.loads(seg_path.read_text())
    instances = seg.get("instances", [])
    if not instances:
        return {"ok": False, "error": "no instances"}

    # cameras + world-up + SAM3 masks (for the per-instance visual-hull carve)
    cam = _load_cameras(session_dir, output_dir)
    views = _build_camera_views(cam, frames_dir, max_views=max_views)
    mask_store = _load_mask_store(output_dir)
    cam_pos = np.array([np.asarray(v.c2w, dtype=np.float64)[:3, 3] for v in views]) \
        if views else np.zeros((0, 3))
    world_up = _detect_world_up(cam_pos, xyz)
    scene_up_centroid = float(np.median(xyz @ world_up)) if n_pts else None
    print(f"[ReconV2] {len(views)} camera views, world_up≈{np.round(world_up, 2).tolist()}, "
          f"masks={'yes' if mask_store else 'no'}")
    n_todo = sum(1 for inst in instances
                 if int(inst.get("id", inst.get("instance_id", -1))) >= 0
                 and (not only_obj_ids or int(inst.get("id", inst.get("instance_id", -1))) in only_obj_ids))
    _emit(progress_cb, phase="classifying", n_views=len(views), total=n_todo, done=0)

    shape_dir = output_dir / "shape"
    scene_dir = output_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    elements = []
    hc_clouds: Dict[int, np.ndarray] = {}
    per_inst: List[Dict] = []
    done = 0
    for inst in instances:
        iid = int(inst.get("id", inst.get("instance_id", -1)))
        if iid < 0:
            continue
        if only_obj_ids and iid not in only_obj_ids:
            continue
        label = inst.get("label", f"object_{iid}")
        done += 1
        _emit(progress_cb, phase="classifying", done=done, total=n_todo, current=str(label))
        gi = np.asarray(inst.get("globalIndices", []), dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < n_pts)]
        if len(gi) < 12:
            per_inst.append({"id": iid, "label": label, "skipped": "too few points"})
            continue
        # Cap the per-instance sub-cloud so RANSAC / PCA / hull / shrink-wrap stay
        # bounded regardless of how many cloud points landed on this instance
        # (a scanned wall can be millions of points; ~150k is plenty for a fit).
        if len(gi) > max_instance_pts:
            gi = np.sort(rng.choice(gi, size=max_instance_pts, replace=False))
        xyz_inst = xyz[gi]
        hc_idx = high_confidence_subcloud(conf, gi)
        xyz_hc = xyz[hc_idx]
        if len(xyz_hc) > MAX_HC_PTS:
            xyz_hc = xyz_hc[np.sort(rng.choice(len(xyz_hc), size=MAX_HC_PTS, replace=False))]
        hc_clouds[iid] = xyz_hc
        # caption fields from the ShapeR PKL (which now carries them) or meta.json
        cap_fields: Dict[str, str] = {}
        caption = label
        folder = _safe_label(label, iid)
        for cand in (shape_dir / folder / f"{folder}.meta.json",):
            if cand.exists():
                try:
                    md = json.loads(cand.read_text())
                    cf = md.get("caption_fields")
                    if isinstance(cf, dict):
                        cap_fields = {k: cf.get(k, "") for k in ("category", "shape", "material", "detail")}
                    caption = md.get("caption", label) or label
                except Exception:
                    pass
        if not cap_fields:
            for cand in (shape_dir / folder / f"{folder}.pkl",):
                if cand.exists():
                    try:
                        import pickle
                        with open(cand, "rb") as f:
                            pk = pickle.load(f)
                        cf = pk.get("caption_fields")
                        if isinstance(cf, dict):
                            cap_fields = {k: cf.get(k, "") for k in ("category", "shape", "material", "detail")}
                        caption = pk.get("caption", label) or label
                    except Exception:
                        pass
        # fall back to the SAM3 label/category as the caption category so the
        # classifier still gets a role hint (floor vs ceiling, pipe vs wall, ...)
        if not cap_fields.get("category"):
            cap_fields = {"category": str(inst.get("label") or inst.get("category") or ""),
                          "shape": cap_fields.get("shape", ""),
                          "material": cap_fields.get("material", ""),
                          "detail": cap_fields.get("detail", "")}
        obb = inst.get("obb")

        # Load the TSDF mesh (if any) for this instance up-front: the classifier
        # uses it as a second source for the wall-footprint fit (fused with the
        # cloud's geodesic polyline), and the same mesh is reused later by
        # reconstruct_mesh_from_glb. Cheap to load once and reuse.
        inst_tsdf_mesh = _load_tsdf_mesh(output_dir, label, iid)
        try:
            cls = classify_instance(xyz_inst, xyz_hc=xyz_hc, caption_fields=cap_fields,
                                    world_up=world_up, obb=obb,
                                    scene_up_centroid=scene_up_centroid,
                                    tsdf_mesh=inst_tsdf_mesh)
        except Exception as e:
            print(f"[ReconV2] classify #{iid}: {e}")
            per_inst.append({"id": iid, "label": label, "error": f"classify: {e}"})
            continue

        el = None
        try:
            if cls.geometry_class != "volumetric_mesh":
                el = reconstruct_element(cls, instance_id=iid, label=label, xyz=xyz_inst,
                                         xyz_hc=xyz_hc, world_up=world_up, caption=caption,
                                         caption_fields=cap_fields, source_indices=gi, obb=obb)
            if el is None:   # volumetric_mesh, or a parametric fit fell through → ShapeR / placeholder
                glb = shape_dir / folder / f"{folder}.glb"
                if use_shaper_glbs and glb.exists():
                    # double-bound regime: visual-hull carve against this instance's
                    # SAM3 masks (upper bound) + shrink-wrap onto its TSDF surface
                    # (lower bound / smooth dense detail), if either is available.
                    inst_views = _instance_camera_views(cam, mask_store, iid, frames_dir)
                    tsdf_mesh = inst_tsdf_mesh   # already loaded above for the classifier
                    if inst_views or tsdf_mesh is not None:
                        print(f"[ReconV2]   #{iid} {label}: ShapeR mesh + "
                              f"{len(inst_views)} masked views + "
                              f"{'TSDF' if tsdf_mesh is not None else 'no TSDF'}")
                    el = reconstruct_mesh_from_glb(glb, instance_id=iid, label=label,
                                                   xyz_hc=xyz_hc, tsdf_mesh=tsdf_mesh,
                                                   views=(inst_views or views), caption=caption,
                                                   caption_fields=cap_fields, role=cls.role,
                                                   source_indices=gi)
                if el is None:
                    pv, pf = _placeholder_mesh_from_cloud(xyz_hc if len(xyz_hc) >= 8 else xyz_inst)
                    if pv is not None:
                        el = MeshElement(instance_id=iid, label=label, geometry_class="volumetric_mesh",
                                         caption=caption, caption_fields=cap_fields, source_indices=gi,
                                         vertices=pv, faces=pf, observed=np.ones(len(pv), dtype=bool),
                                         quality_flag="placeholder_convex_hull")
                        el.meta["role"] = cls.role
        except Exception as e:
            print(f"[ReconV2] reconstruct #{iid}: {e}")
            traceback.print_exc()
            per_inst.append({"id": iid, "label": label, "error": f"reconstruct: {e}",
                             "geometry_class": cls.geometry_class})
            continue
        if el is None:
            per_inst.append({"id": iid, "label": label, "skipped": "no element",
                             "geometry_class": cls.geometry_class})
            continue
        elements.append(el)
        per_inst.append({"id": iid, "label": label, "geometry_class": cls.geometry_class,
                         "role": cls.role, "is_structure": cls.is_structure,
                         "n_points": int(len(gi)), "n_high_conf": int(len(xyz_hc))})

    if not elements:
        return {"ok": False, "error": "no elements reconstructed", "instances": per_inst}

    # assemble
    _emit(progress_cb, phase="assembling", n_elements=len(elements))
    try:
        scene = assemble_scene(elements, session_id=session_id, views=views, hc_clouds=hc_clouds)
    except Exception as e:
        print(f"[ReconV2] assemble: {e}")
        traceback.print_exc()
        from reconstruction import Scene
        scene = Scene(session_id=session_id, elements=elements, adjacency=[])

    # write GLBs + per-element meta + scene.json (with PBR material + cloud-painted
    # colour/texture + IFC metadata baked into each GLB)
    _emit(progress_cb, phase="writing", n_elements=len(scene.elements))
    inst_color_by_id = {int(i.get("id", i.get("instance_id", -1))): i.get("color") for i in instances}
    n_glb = 0
    for el in scene.elements:
        folder = _safe_label(el.label, el.instance_id)
        gp = scene_dir / f"{folder}.glb"
        try:
            seg_color = inst_color_by_id.get(int(el.instance_id))
            if seg_color and not el.meta.get("color"):
                el.meta["color"] = seg_color
            if write_element_glb(el, gp, cloud_xyz=xyz, cloud_rgb=rgb, seg_color=seg_color) is not None:
                n_glb += 1
                (scene_dir / f"{folder}.meta.json").write_text(json.dumps(json_safe(el.to_meta_dict()), indent=2))
        except Exception as e:
            print(f"[ReconV2] write GLB #{el.instance_id}: {e}")
    write_scene_json(scene, scene_dir / "scene.json")

    elapsed = time.time() - t0
    summary = {
        "ok": True, "session_id": session_id, "elapsed_s": round(elapsed, 1),
        "n_elements": len(scene.elements), "n_glb": n_glb,
        "n_adjacency": len(scene.adjacency), "n_camera_views": len(views),
        "scene_json": str((scene_dir / "scene.json")),
        "instances": per_inst,
        "by_class": {},
    }
    for el in scene.elements:
        gc = el.geometry_class
        summary["by_class"][gc] = summary["by_class"].get(gc, 0) + 1
    print(f"[ReconV2] done in {elapsed:.1f}s: {len(scene.elements)} elements "
          f"({summary['by_class']}), {n_glb} GLBs, {len(scene.adjacency)} adjacency edges")
    return summary
