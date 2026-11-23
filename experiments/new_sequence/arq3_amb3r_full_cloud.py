"""AMB3R-SLAM's cloud AT FULL POWER (USER 2026-11-22: "siempre usalo con su potencia total, con el filtro de confianza
al 10%" / "corre asi … y pisa le bufferstop de amb3r salad que tenemos ahora"): from AMB3R's own per-frame dump
(run.py --stac_dump: every frame it mapped, ALL its pixels — depth along the world camera's axis, the pass's K, the
final camera-to-world, its confidence; the sky already removed by AMB3R), in metres by the run's metre (make_metric),
filtered by OUR confidence floor alone: min + 10 % of the range of the pass's valid confidences
(precision.depth_on_f5.confidence_floor's rule, over the whole pass). Nothing else. Written over the session's cloud
and its octree rebuilt. The dump is left as it is.

    python arq3_amb3r_full_cloud.py <scan dir> <amb3r_video.npz in metres> <dump dir> <session name>"""
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
CONF_MIN_NORM = 0.10


def main() -> int:
    src, npz, dump, name = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    every = int(sys.argv[5]) if len(sys.argv) > 5 else 1     # one record in `every` (by frame number)
    from PIL import Image
    from intake.quality import frame_file
    from potree_converter import convert_ply_to_potree
    from reconstruction.amb3r_probe import load_video
    metre = float(load_video(npz)["metric_scale"])
    files = [f for f in sorted(dump.glob("frame_*.npz")) if int(f.stem.split("_")[1]) % every == 0]
    if not files:
        raise SystemExit(f"no frame records in {dump}")
    lo, hi = np.inf, -np.inf
    for f in files:
        with np.load(f) as z:
            d, c = np.asarray(z["depth"]), np.asarray(z["conf"], np.float64)
            v = c[(d > 0) & np.isfinite(c)]
            if v.size:
                lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
    thr = lo + CONF_MIN_NORM * (hi - lo)
    print(f"[full] {len(files)} frames; confidence {lo:.4g}–{hi:.4g} → floor {thr:.4g}; metre {metre:.4f} m per unit",
          flush=True)
    seg = []
    n_valid = 0
    for f in files:
        with np.load(f) as z:
            d = np.asarray(z["depth"], np.float64)
            c = np.asarray(z["conf"], np.float64)
            K = np.asarray(z["K"], np.float64)
            c2w = np.asarray(z["c2w"], np.float64)
            fr = int(z["frame"])
        H, W = d.shape
        valid = (d > 0) & np.isfinite(c)
        n_valid += int(valid.sum())
        keep = valid & (c >= thr)
        vv, uu = np.nonzero(keep)
        dz = d[keep]
        cam = np.stack([(uu - K[0, 2]) / K[0, 0] * dz, (vv - K[1, 2]) / K[1, 1] * dz, dz], 1)
        X = (cam @ c2w[:3, :3].T + c2w[:3, 3]) * metre
        img = np.asarray(Image.open(frame_file(src / "frames", fr)).convert("RGB"))
        Hn, Wn = img.shape[:2]
        col = np.clip(np.rint(uu * Wn / W), 0, Wn - 1).astype(np.int64)
        row = np.clip(np.rint(vv * Hn / H), 0, Hn - 1).astype(np.int64)
        a = np.empty(len(X), np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                                       ("blue", "u1"), ("confidence", "<f4"), ("frame_global", "<i4"),
                                       ("pixel_row", "<i2"), ("pixel_col", "<i2")]))
        a["x"], a["y"], a["z"] = X[:, 0], X[:, 1], X[:, 2]
        rgb = img[row, col]
        a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        a["confidence"] = c[keep]
        a["frame_global"], a["pixel_row"], a["pixel_col"] = fr, row, col
        seg.append(a)
    a = np.concatenate(seg)
    del seg
    proj = Path(SERVER) / "projects" / name
    scans = sorted(proj.glob("scans/*/src_*"))
    if len(scans) != 1:
        raise SystemExit(f"{proj}: {len(scans)} scans — the session to write over is not this script's test session")
    tscan = scans[0]
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(a)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property float confidence\nproperty int frame_global\nproperty short pixel_row\n"
            "property short pixel_col\nend_header\n")
    out = tscan / "output" / "cleaned_cloud.ply"
    tmp = out.with_name("cleaned_cloud.ply.part")
    with open(tmp, "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(a.tobytes())
    tmp.replace(out)
    ok = convert_ply_to_potree(tscan, force=True)
    print(f"[full] {name}: {len(a):,} points of {n_valid:,} valid pixels over the 10 % floor, from {len(files)} frames "
          f"at {W}x{H} → {out}; octree {'built' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
