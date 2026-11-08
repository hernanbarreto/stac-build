"""AMB3R-SLAM's run → a session the viewer shows WITHOUT corrections (env da3).

AMB3R-SLAM (HengyiWang/amb3r-slam, 2026: DA3-Small tracking, a heavy model — DA3 Giant or VGGT-Ω — building submaps of
24 views with each frame in about three of them, long-range Sim(3) edges, verified loops, an offline pose graph) writes
the camera of every frame it read (traj_<name>.npz: camera-to-world) and its map (map_<name>.ply: every kept pixel of
each saved frame through its submap's final Sim(3)). The session takes
  - the map as output/chunk_0.ply (the cloud stage reads it; no per-point frame: the witness filter cannot run on it);
  - the cameras of the frames the map was saved from (every ``save_every`` x frame_stride-th frame) as the session's
    keyframes: output/camera_poses.txt (OpenCV axes, as AMB3R's), camera_frames.txt, frames/ (hard links);
  - output/amb3r_run.json: the model, the stride, AMB3R's keyframes, loops and per-submap metric scales;
  - --align-to <run dir>: a run with no metric scale (VGGT-Ω's) brought into another run's frame and size by ONE
    similarity (Umeyama on the camera centres of the same frames; its residual recorded) — to look at both side by side
    in metres, not a measurement;
  - when the run holds frames/ (AMB3R's --stac_dump: one record per saved frame), the session's keyframes are those
    frames and their depth records are written for the precision chain: depth along the world camera's axis, the
    pass's pinhole K, carried onto the grid F0 reads (output/vggt_omega_config.yaml: max_size 832; bilinear, a pixel
    valid where its four neighbours are) when AMB3R ran at another size, Ω's or DA3's confidence; F5's gauge is the
    scatter of AMB3R's per-submap metric scales (1.4826 MAD of their log) of this run, or of the --align-to run when
    this one has none (Ω).
  python amb3r_session.py <run dir> <frames dir> <source src_default> <new project> --stride 2 --every 8 [--replace]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER))
PROJECTS = Path(os.environ.get("STAC_PROJECTS_DIR", "/workspace/stac-build/server/projects"))


def write_ply(path, xyz, rgb, conf, fg, rows, cols):
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("confidence", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")])
    v = np.empty(len(xyz), dt)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    v["confidence"], v["frame_global"], v["pixel_row"], v["pixel_col"] = conf, fg, rows, cols
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(v)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property float confidence\nproperty int frame_global\n"
            "property short pixel_row\nproperty short pixel_col\nend_header\n")
    tmp = Path(str(path) + ".part")
    with open(tmp, "wb") as fh:
        fh.write(head.encode("ascii"))
        v.tofile(fh)
    tmp.replace(path)


def read_ply_xyz_rgb(path):
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        off = f.tell()
    n = int(next(ln.split()[2] for ln in header.decode().splitlines() if ln.startswith("element vertex")))
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.fromfile(path, dtype=dt, count=n, offset=off)
    return np.stack([v["x"], v["y"], v["z"]], -1), np.stack([v["red"], v["green"], v["blue"]], -1)


def write_records(recs, out, files, align, resolution=832):
    """The depth records of AMB3R's dumped frames on the grid F0 reads; returns their (aligned) camera-to-world."""
    from PIL import Image
    from precision.camera import omega_grid_for
    from tools.amb3r.records import K_between, full_frame_grid, resample
    rd = out / "omega_run" / "results_output"
    rd.mkdir(parents=True, exist_ok=True)
    W, H = Image.open(files[0]).size
    g_r = omega_grid_for(W, H, "max_size", int(resolution)).to_dict()
    (out / "record_grid.json").write_text(json.dumps(g_r, indent=1))
    (out / "vggt_omega_config.yaml").write_text(f"Model:\n  omega_mode: max_size\n  omega_resolution: {int(resolution)}\n")
    poses, intr, kres = [], [], []
    for r in recs:
        z = np.load(r)
        f = int(z["frame"])
        depth, conf, K, c2w = (np.asarray(z["depth"], np.float64), np.asarray(z["conf"], np.float64),
                               np.asarray(z["K"], np.float64), np.asarray(z["c2w"], np.float64).copy())
        if align is not None:
            c2w[:3, :3] = align["R"] @ c2w[:3, :3]
            c2w[:3, 3] = align["scale"] * align["R"] @ c2w[:3, 3] + align["t"]
            depth = depth * align["scale"]
        g_a = full_frame_grid(W, H, depth.shape[1], depth.shape[0], name="amb3r")
        if (g_a["w"], g_a["h"]) == (g_r["w"], g_r["h"]) and all(g_a[k] == g_r[k] for k in ("crop_x", "crop_y", "crop_w", "crop_h")):
            d, c, Kr = depth.astype(np.float32), conf.astype(np.float32), K
        else:
            valid = depth > 0
            d, ok = resample(depth, valid, g_a, g_r)
            c, _ = resample(conf, valid, g_a, g_r)
            Kr = K_between(K, g_a, g_r)
        frame_no = int(files[f].stem)
        np.savez(rd / f"frame_{frame_no}.npz", depth=d.astype(np.float32), conf=c.astype(np.float32), K_omega=Kr)
        poses.append(c2w)
        intr.append([Kr[0, 0], Kr[1, 1], Kr[0, 2], Kr[1, 2]])
        kres.append(float(z["k_residual_px"]))
    np.savetxt(out / "intrinsic.txt", np.array(intr))
    print(f"[amb3r-session] {len(recs)} depth records on the {g_r['w']}x{g_r['h']} grid (AMB3R's "
          f"{depth.shape[1]}x{depth.shape[0]}); K fit residual median {np.median(kres):.3f} px", flush=True)
    return np.array(poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("frames")
    ap.add_argument("source")
    ap.add_argument("name")
    ap.add_argument("--stride", type=int, required=True, help="AMB3R's --stride: input frame i is the i*stride-th file")
    ap.add_argument("--every", type=int, required=True, help="the map's saved frames: every n-th input frame")
    ap.add_argument("--label", default="AMB3R-SLAM")
    ap.add_argument("--align-to", default=None)
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args()
    run, fdir, src = Path(a.run), Path(a.frames), Path(a.source)
    t0 = time.time()
    traj = sorted(run.glob("traj_*.npz"))
    maps = [p for p in sorted(run.glob("map_*.ply")) if not p.name.endswith("_stride.ply")]
    if len(traj) != 1 or len(maps) != 1:
        raise SystemExit(f"{run}: expected one traj_*.npz and one map_*.ply, found {len(traj)} and {len(maps)}")
    z = np.load(traj[0])
    pred = np.asarray(z["pred"], np.float64)
    files = sorted(p for p in fdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[::a.stride]
    if len(files) < len(pred):
        raise SystemExit(f"{len(pred)} cameras but {len(files)} frames at stride {a.stride} in {fdir}")
    align = None
    if a.align_to:
        ref = np.asarray(np.load(sorted(Path(a.align_to).glob("traj_*.npz"))[0])["pred"], np.float64)
        if len(ref) != len(pred):
            raise SystemExit(f"--align-to: {len(ref)} cameras there, {len(pred)} here — not the same frames")
        A, B = pred[:, :3, 3], ref[:, :3, 3]
        ma, mb = A.mean(0), B.mean(0)
        U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A))
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt
        sc = float(np.trace(np.diag(S) @ D) / (A - ma).var(0).sum())
        t = mb - sc * R @ ma
        res = np.linalg.norm(sc * A @ R.T + t - B, axis=1)
        pred = pred.copy()
        pred[:, :3, :3] = R @ pred[:, :3, :3]
        pred[:, :3, 3] = sc * A @ R.T + t
        align = {"to": str(a.align_to), "scale": sc, "R": R, "t": t, "residual_median_m": float(np.median(res)),
                 "residual_p90_m": float(np.percentile(res, 90)), "residual_max_m": float(res.max())}
        print(f"[amb3r-session] aligned to {a.align_to}: scale {sc:.4f}, camera residual median "
              f"{np.median(res):.3f} m, p90 {np.percentile(res, 90):.3f} m", flush=True)
    dump = run / "frames"
    recs = sorted(dump.glob("frame_*.npz")) if dump.is_dir() else []
    sel = ([int(np.load(r)["frame"]) for r in recs] if recs else list(range(0, len(pred), a.every)))
    scan = src.parent.name
    proj = PROJECTS / a.name
    ses = proj / "scans" / scan / "src_default"
    out = ses / "output"
    if proj.exists():
        if not a.replace:
            raise SystemExit(f"{proj} exists (--replace to overwrite it)")
        shutil.rmtree(proj)
    out.mkdir(parents=True)
    (ses / "frames").mkdir(parents=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    (proj / "project.json").write_text(json.dumps({
        "name": a.name, "slug": a.name, "location": "", "created_at": now, "ifcs": [], "scan_days": [],
        "scans": [{"key": f"{scan}/default", "label": a.label, "added_at": now, "date": scan, "source": "default"}],
        "active_scan": f"{scan}/default"}, indent=2))
    frames = []
    for i in sel:
        f = files[i]
        os.link(f, ses / "frames" / f.name)
        frames.append(int(f.stem))
    (ses / "frames" / "selected_frames.json").write_text(json.dumps({"selected_files": [files[i].name for i in sel]}))
    if recs:
        poses = write_records(recs, out, files, align)
    else:
        poses = pred[sel]
    np.savetxt(out / "camera_poses.txt", poses.reshape(len(sel), 16))
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    (out / "frame_list.json").write_text(json.dumps([files[i].name for i in sel]))
    if recs:
        gauge_run = run
        sc_ = np.asarray(z["submap_scale"])[:, 0]
        if np.allclose(sc_, 1.0) and a.align_to:
            gauge_run = Path(a.align_to)
            sc_ = np.asarray(np.load(sorted(gauge_run.glob("traj_*.npz"))[0])["submap_scale"])[:, 0]
        ls = np.log(sc_[sc_ > 0])
        sigma = float(1.4826 * np.median(np.abs(ls - np.median(ls))))
        (out / "gauge.json").write_text(json.dumps({
            "applied_instrument": "amb3r_submap_metric", "sigma_by_instrument": {"amb3r_submap_metric": sigma},
            "source": f"tools/amb3r/amb3r_session.py: 1.4826 MAD of the log of AMB3R's per-submap metric scales "
                      f"({gauge_run}) — the F2 quantity F5 reads; F2 itself does not run in this pipeline"}, indent=1))
    xyz, rgb = read_ply_xyz_rgb(maps[0])
    if align is not None:
        xyz = (align["scale"] * xyz.astype(np.float64) @ R.T + t).astype(np.float32)
    n = len(xyz)
    write_ply(out / "chunk_0.ply", xyz.astype(np.float32), rgb, np.ones(n, np.float32), np.zeros(n, np.int32),
              np.zeros(n, np.int16), np.zeros(n, np.int16))
    walk = float(np.linalg.norm(np.diff(pred[:, :3, 3], axis=0), axis=1).sum())
    res = json.loads((run / "results.json").read_text())
    (out / "amb3r_run.json").write_text(json.dumps({
        "run": str(run), "label": a.label, "stride": a.stride, "every": a.every, "cameras": len(pred),
        "session_keyframes": len(sel), "amb3r_keyframes": np.asarray(z["kf"]).tolist(),
        "submap_metric_scale": np.asarray(z["submap_scale"]).tolist(), "walk": walk, "points": n,
        "align": None if align is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in align.items()},
        "depth_records": len(recs),
        "results": res}, indent=1))
    print(f"[amb3r-session] {ses}: {len(sel)} cameras of {len(pred)}, {n:,} map points, walk {walk:.1f} "
          f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
