"""AMB3R-SLAM's own cloud, as a test session for his eye (USER 2026-11-22: "publicame esta nube si ya esta"): every
frame AMB3R reconstructed (its record: depth on the pixel-stride grid, its camera), back-projected in metres, up to the
range (15 m), nothing else filtered; each point's colour and origin (frame_global, pixel_row, pixel_col) from the frame
it came from. Published as a NEW project (arq_session.py's copy of the scan, the cloud as output/cleaned_cloud.ply,
its octree by the server's own converter).

    python arq3_publish_amb3r.py <source scan dir> <amb3r_video.npz (in metres)> <new project name>"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
RANGE_M = float("inf")   # USER 2026-11-22: "no la recortes no hace falta"


def main() -> int:
    src, npz, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    from PIL import Image
    from intake.quality import frame_file
    from potree_converter import convert_ply_to_potree
    from reconstruction.amb3r_probe import load_video
    v = load_video(npz)
    if "metric_scale" not in v:
        raise SystemExit(f"{npz} is not in metres")
    s = int(v["stride"])
    fno = np.asarray(v["frame_no"], np.int64)
    rec, c2w, K, D = (np.asarray(v[k]) for k in ("rec_index", "rec_c2w", "rec_K", "rec_depth"))
    gw, gh = (int(x) for x in v["grid_wh"])
    Hs, Ws = D.shape[1:]
    vv, uu = np.mgrid[0:Hs, 0:Ws]
    L = {k: [] for k in ("xyz", "rgb", "fg", "pr", "pc")}
    for r in range(len(rec)):
        d = D[r].astype(np.float64)
        Kr = K[r].astype(np.float64).copy()
        Kr[:2] /= s                                            # the model grid's K on the pixel-stride grid
        x = (uu - Kr[0, 2]) / Kr[0, 0] * d
        y = (vv - Kr[1, 2]) / Kr[1, 1] * d
        m = (d > 0) & (np.sqrt(x * x + y * y + d * d) <= RANGE_M)
        cam = np.stack([x[m], y[m], d[m]], 1)
        X = cam @ c2w[r][:3, :3].T + c2w[r][:3, 3]
        f = int(fno[rec[r]])
        img = np.asarray(Image.open(frame_file(src / "frames", f)).convert("RGB"))
        Hn, Wn = img.shape[:2]
        col = np.clip(np.rint(uu[m] * s * Wn / gw), 0, Wn - 1).astype(np.int64)
        row = np.clip(np.rint(vv[m] * s * Hn / gh), 0, Hn - 1).astype(np.int64)
        L["xyz"].append(X.astype(np.float32)); L["rgb"].append(img[row, col])
        L["fg"].append(np.full(len(X), f, np.int32)); L["pr"].append(row.astype(np.int16)); L["pc"].append(col.astype(np.int16))
    xyz = np.concatenate(L["xyz"])
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("confidence", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")])
    a = np.empty(len(xyz), dt)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rgb = np.concatenate(L["rgb"])
    a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    a["confidence"] = 1.0
    a["frame_global"], a["pixel_row"], a["pixel_col"] = np.concatenate(L["fg"]), np.concatenate(L["pr"]), np.concatenate(L["pc"])
    proj = Path(SERVER) / "projects" / name
    old = sorted(proj.glob("scans/*/src_*"))
    if proj.exists() and len(old) == 1 and not (old[0] / "output" / "cleaned_cloud.ply").exists():
        tscan = old[0]                                   # this script's own copy, left without its cloud
    else:
        r = subprocess.run([sys.executable, str(HERE / "arq_session.py"), str(src), name], cwd=SERVER, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(r.stdout[-2000:], flush=True)
            return 1
        tscan = Path(r.stdout.strip().splitlines()[-1])
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(a)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property float confidence\nproperty int frame_global\nproperty short pixel_row\n"
            "property short pixel_col\nend_header\n")
    with open(tscan / "output" / "cleaned_cloud.ply", "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(a.tobytes())
    ok = convert_ply_to_potree(tscan, force=True)
    print(f"[publish] {name}: {len(a):,} points from {len(rec)} reconstructed frames (every {s} px, ≤ {RANGE_M:g} m) "
          f"→ {tscan}; octree {'built' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
