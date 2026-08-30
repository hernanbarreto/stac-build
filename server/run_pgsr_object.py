#!/usr/bin/env python3
"""Per-object PGSR (USER 2026-08-30): train PGSR on ONE segmented instance
using ONLY its SAM3 masks ("con las máscaras eh!!!"), then integrate the
rendered depths into a per-object TSDF and publish it next to the Poisson
mesh (tsdf/<label>_<id>_pgsr/) so both appear in the viewer for comparison.

Per instance:
  1. session PGSR scene scaffold (idempotent export_scene: images + COLMAP)
  2. object scene: shared images/cameras, OWN seed (the instance's cloud
     points) and OWN masks — inverted SAM3: 0 = the object (supervised),
     255 = everything else (excluded from the photometric loss). Frames
     without the object's mask are fully excluded.
  3. run_pgsr.sh (pgsr env) with the validated max-quality regime
  4. TSDF from the object's own renders → GLB publish

Runs in the SERVER (da3) env; the trainer subprocess manages the pgsr env.
Protocol (stdout): [PGSROBJ-PROGRESS]{...} / [PGSROBJ-RESULT]{...}
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
except Exception:  # noqa: BLE001
    pass

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass


def _progress(iid, phase, detail=""):
    print("[PGSROBJ-PROGRESS]" + json.dumps(
        {"instance_id": int(iid), "phase": phase, "detail": detail}),
        flush=True)


def _build_object_scene(output_dir: Path, scene: Path, obj_scene: Path,
                        obj_pts: np.ndarray, obj_cols, masked_frames: set,
                        mask_arrays: dict, max_seed_pts: int) -> None:
    """Object scene = shared images + cameras, own seed + own inverted masks."""
    import open3d as o3d
    from PIL import Image
    from reconstruction.pgsr_export import _write_points3d_ply

    if obj_scene.exists():
        shutil.rmtree(obj_scene)
    (obj_scene / "sparse").mkdir(parents=True)
    os.symlink(str(scene / "images"), str(obj_scene / "images"))
    for f in ("cameras.txt", "images.txt"):
        shutil.copy2(scene / "sparse" / f, obj_scene / "sparse" / f)

    # seed: the instance's own cloud points (raw frame == pose frame)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
    if obj_cols is not None and len(obj_cols) == len(obj_pts):
        pcd.colors = o3d.utility.Vector3dVector(obj_cols)
    tmp = obj_scene / "sparse" / "_seed_tmp.ply"
    o3d.io.write_point_cloud(str(tmp), pcd, write_ascii=False)
    n_seed = _write_points3d_ply(tmp, obj_scene / "sparse" / "points3D.ply",
                                 max_seed_pts)
    tmp.unlink(missing_ok=True)

    # masks: 255 everywhere (excluded), 0 on the object (supervised).
    meta = json.loads((scene / "scene_meta.json").read_text())
    Wn, Hn = meta["native_wh"]
    masks_dir = obj_scene / "masks"
    masks_dir.mkdir()
    n_sup = 0
    for img in sorted((scene / "images").iterdir()):
        fnum = int(os.path.splitext(img.name)[0])
        full = np.full((Hn, Wn), 255, dtype=np.uint8)
        if fnum in masked_frames:
            m = mask_arrays[fnum]
            up = np.array(Image.fromarray((m > 0).astype(np.uint8) * 255)
                          .resize((Wn, Hn), Image.NEAREST))
            full[up > 0] = 0
            n_sup += 1
        # vendor loader: image_name = basename.split('.')[0] → mask file must
        # be "<stem>.png" ("000123.png"); "<name>.jpg.png" loaded ZERO masks
        # and the first train run went unmasked (caught+killed 2026-08-30)
        Image.fromarray(full).save(masks_dir / f"{Path(img.name).stem}.png")
    print(f"[pgsr-obj] scene: {n_seed:,} seed pts, {n_sup} supervised "
          f"keyframes of {len(list((scene / 'images').iterdir()))}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--instance-id", type=int, action="append", required=True)
    ap.add_argument("--iterations", type=int, default=15000)
    ap.add_argument("--min-mask-frames", type=int, default=3)
    ap.add_argument("--max-seed-pts", type=int, default=400_000)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    frames_dir = Path(args.frames_dir).resolve()
    session_dir = Path(args.session_dir).resolve()

    import open3d as o3d
    import yaml
    from reconstruction.pgsr_export import SCENE_DIRNAME, export_scene
    from segmentation.erase import _mask_obj_by_iid
    from segmentation.tsdf_export import _safe_label, export_tsdf_meshes

    cfg = {}
    cfg_path = _SERVER_DIR / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    pcfg = ((cfg.get("reconstruction") or {}).get("pgsr") or {})

    seg = json.loads((output_dir / "segmentation_result.json").read_text())
    by_iid = {int(i.get("instance_id", i.get("id"))): i
              for i in seg.get("instances") or []}
    oid_map = _mask_obj_by_iid(output_dir)
    masks_npz = np.load(output_dir / "seg_masks.npz", allow_pickle=True)

    pc = o3d.io.read_point_cloud(str(output_dir / "cleaned_cloud.ply"))
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors) if len(pc.colors) else None

    # session scaffold (idempotent, shared by every object)
    scene = output_dir / SCENE_DIRNAME
    if not (scene / "scene_meta.json").exists():
        _progress(0, "scene", "building session PGSR scene")
        export_scene(output_dir, frames_dir,
                     max_seed_pts=int(pcfg.get("max_seed_pts", 1_500_000)),
                     log=lambda m: print(f"[pgsr-obj] {m}", flush=True))

    written, skipped = [], []
    for iid in args.instance_id:
        t0 = time.time()
        inst = by_iid.get(int(iid))
        if inst is None:
            skipped.append({"instance_id": iid, "reason": "unknown instance"})
            continue
        label = inst.get("label", "segment")
        safe = _safe_label(label, int(iid))
        oid = oid_map.get(int(iid))
        if oid is None:
            skipped.append({"instance_id": iid, "reason": "no mask obj id"})
            continue
        mask_arrays = {}
        for k in masks_npz.files:
            if k.startswith("f") and k.endswith(f"_o{oid}"):
                mask_arrays[int(k.split("_o")[0][1:])] = masks_npz[k]
        if len(mask_arrays) < int(args.min_mask_frames):
            skipped.append({"instance_id": iid,
                            "reason": f"only {len(mask_arrays)} masked "
                                      f"frames (<{args.min_mask_frames})"})
            print(f"[pgsr-obj] {safe}: SKIP — {len(mask_arrays)} masked "
                  f"frames", flush=True)
            continue
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < len(pts))]
        if len(gi) < 1000:
            skipped.append({"instance_id": iid,
                            "reason": f"only {len(gi)} points"})
            continue

        obj_root = output_dir / "pgsr_obj" / safe
        _progress(iid, "scene", f"{len(mask_arrays)} masked frames")
        _build_object_scene(output_dir, scene, obj_root / "scene",
                            pts[gi], cols[gi] if cols is not None else None,
                            set(mask_arrays), mask_arrays,
                            int(args.max_seed_pts))

        # OBJECT DEPTH ANCHOR (2026-08-30): the instance's OWN cloud points
        # rasterized per view — the sparse-view depth prior (Sparse2DGS et al.)
        # anchored to the validated truth
        anchor_dir = None
        try:
            from reconstruction.pgsr_export import export_cloud_anchor_depths
            _progress(iid, "anchor", "rasterizing object cloud")
            anchor_dir = export_cloud_anchor_depths(
                output_dir, frames_dir, point_indices=gi,
                dst_dir=obj_root / "scene" / "cloud_anchor",
                log=lambda m: print(f"[pgsr-obj] {m}", flush=True))
        except Exception as e:  # noqa: BLE001
            print(f"[pgsr-obj] {safe}: object anchor failed ({e}) — "
                  "training without depth prior", flush=True)

        # trainer — the validated max-quality regime, masked to the object
        cmd = ["bash", str(_SERVER_DIR / "run_pgsr.sh"),
               "--scene", str(obj_root / "scene"),
               "--model_dir", str(obj_root / "model"),
               "--render_dir", str(obj_root / "render"),
               "--iterations", str(int(args.iterations)),
               "--resolution", str(int(pcfg.get("resolution", 1))),
               "--ncc_scale", str(float(pcfg.get("ncc_scale", 0.5))),
               "--densify_abs_grad_threshold",
               str(float(pcfg.get("densify_abs_grad_threshold", 0.00015))),
               "--opacity_cull_threshold",
               str(float(pcfg.get("opacity_cull_threshold", 0.05))),
               "--max_abs_split_points",
               str(int(pcfg.get("max_abs_split_points", 50000)))]
        if bool(pcfg.get("use_depth_filter", False)):
            cmd.append("--use_depth_filter")
        if bool(pcfg.get("exposure_compensation", True)):
            cmd.append("--exposure_compensation")
        # OBJECT MODE v2 (2026-08-30): background transparency + cloud-anchored
        # depth + 3D bbox prune — the three fixes from the masked-GS research
        sfc = ((cfg.get("surface_fit") or {}))
        cmd += ["--object_bg_weight",
                str(float(sfc.get("pgsr_object_bg_weight", 1.0))),
                "--object_prune_margin",
                str(float(sfc.get("pgsr_object_prune_margin", 0.5)))]
        if anchor_dir is not None:
            cmd += ["--cloud_anchor_dir", str(anchor_dir),
                    "--cloud_anchor_weight",
                    str(float(sfc.get("pgsr_object_anchor_weight", 1.0))),
                    "--cloud_anchor_band",
                    str(float(sfc.get("pgsr_object_anchor_band_m", 0.02)))]
        _progress(iid, "train", f"{args.iterations} iters")
        print(f"[pgsr-obj] {safe}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"[pgsr-obj:{safe}] {line}", flush=True)
        proc.wait()
        if proc.returncode != 0:
            skipped.append({"instance_id": iid,
                            "reason": f"trainer exit {proc.returncode}"})
            continue
        render_dir = obj_root / "render"
        if not list(render_dir.glob("frame_*.npz")):
            skipped.append({"instance_id": iid, "reason": "no renders"})
            continue

        _progress(iid, "tsdf", "integrating object renders")
        try:
            glbs = export_tsdf_meshes(
                output_dir, frames_dir, seg, session_dir=session_dir,
                obj_ids=[int(iid)], depth_trunc=12.0,
                pgsr_render_dir=render_dir, name_suffix="_pgsr")
        except Exception as e:  # noqa: BLE001
            skipped.append({"instance_id": iid, "reason": f"tsdf: {e}"})
            print(f"[pgsr-obj] {safe}: TSDF failed: {e}", flush=True)
            continue
        # texrecon after the TSDF (USER 2026-08-30): the PGSR mesh ships
        # textured like the Poisson one; failure keeps the untextured mesh
        for g in glbs:
            try:
                _progress(iid, "texture", Path(g).name)
                from reconstruction.texture_bake import bake_object_glb
                if bake_object_glb(Path(g), session_dir, output_dir):
                    mp = Path(g).with_suffix(".meta.json")
                    if mp.exists():
                        m = json.loads(mp.read_text())
                        m["textured"] = True
                        mp.write_text(json.dumps(m, indent=2))
            except Exception as e:  # noqa: BLE001
                print(f"[pgsr-obj] {safe}: texture failed ({e}) — "
                      "untextured mesh kept", flush=True)
        for g in glbs:
            written.append(str(g))
        _progress(iid, "done", f"{time.time() - t0:.0f}s")
        print(f"[pgsr-obj] {safe}: done in {time.time() - t0:.0f}s "
              f"→ {[Path(g).name for g in glbs]}", flush=True)

    print("[PGSROBJ-RESULT]" + json.dumps(
        {"written": written, "skipped": skipped}), flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
