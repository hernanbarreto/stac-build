"""Depth records on the grid F0 reads: grid maps (precision.camera's GridMap as a dict), resampling between grids, the
raw cloud from the records. numpy only, so any environment can import it."""
import json
from pathlib import Path

import numpy as np


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


def full_frame_grid(native_w, native_h, w, h, name="full_frame"):
    """The grid of a whole frame resized to (w, h) (cv2.resize: pixel centres map as (x + 0.5) s - 0.5)."""
    return {"name": name, "w": int(w), "h": int(h), "content_w": int(w), "content_h": int(h), "pad_left": 0,
            "pad_top": 0, "crop_x": 0, "crop_y": 0, "crop_w": int(native_w), "crop_h": int(native_h),
            "native_w": int(native_w), "native_h": int(native_h)}


def _scales(g):
    return g["crop_w"] / float(g["content_w"]), g["crop_h"] / float(g["content_h"])


def grid_to_native(uv, g):
    """precision.camera.grid_to_native on a grid dict: grid pixel -> native pixel (centres at integers)."""
    uv = np.asarray(uv, np.float64)
    sx, sy = _scales(g)
    out = np.empty_like(uv)
    out[..., 0] = (uv[..., 0] - g["pad_left"] + 0.5) * sx - 0.5 + g["crop_x"]
    out[..., 1] = (uv[..., 1] - g["pad_top"] + 0.5) * sy - 0.5 + g["crop_y"]
    return out


def native_to_grid(uv, g):
    """The exact inverse of grid_to_native."""
    uv = np.asarray(uv, np.float64)
    sx, sy = _scales(g)
    out = np.empty_like(uv)
    out[..., 0] = (uv[..., 0] - g["crop_x"] + 0.5) / sx - 0.5 + g["pad_left"]
    out[..., 1] = (uv[..., 1] - g["crop_y"] + 0.5) / sy - 0.5 + g["pad_top"]
    return out


def K_between(K, g_from, g_to):
    """A pinhole K (pixel centres) on grid g_from, the same camera on grid g_to (through the native frame)."""
    sf, tf = _scales(g_from), _scales(g_to)
    c_native = grid_to_native(np.array([K[0, 2], K[1, 2]]), g_from)
    c_to = native_to_grid(c_native, g_to)
    out = np.eye(3)
    out[0, 0] = K[0, 0] * sf[0] / tf[0]
    out[1, 1] = K[1, 1] * sf[1] / tf[1]
    out[0, 2], out[1, 2] = c_to
    return out


def resample(img, valid, g_from, g_to):
    """``img`` on g_from read at every pixel of g_to: bilinear, a pixel valid only where its four neighbours are.
    Returns (values float32 0 where invalid, validity)."""
    h, w = img.shape
    r, c = np.mgrid[0:g_to["h"], 0:g_to["w"]]
    xy = native_to_grid(grid_to_native(np.stack([c, r], -1).astype(np.float64), g_to), g_from)
    x, y = xy[..., 0], xy[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    inb = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    ax, ay = x - x0c, y - y0c
    ok = inb & valid[y0c, x0c] & valid[y0c, x0c + 1] & valid[y0c + 1, x0c] & valid[y0c + 1, x0c + 1]
    v = ((1 - ax) * (1 - ay) * img[y0c, x0c] + ax * (1 - ay) * img[y0c, x0c + 1]
         + (1 - ax) * ay * img[y0c + 1, x0c] + ax * ay * img[y0c + 1, x0c + 1])
    return np.where(ok, v, 0.0).astype(np.float32), ok


def raw_cloud(out, frames_dir, stride=1):
    """chunk_0.ply + chunk_0_origins.npz from the session's depth records, poses and record grid: every valid record
    pixel (every ``stride``-th along each axis) unprojected with its record K and pose, coloured from its frame.
    Returns the number of points."""
    from PIL import Image
    out, frames_dir = Path(out), Path(frames_dir)
    g = json.loads((out / "record_grid.json").read_text())
    frames = [int(x) for x in (out / "camera_frames.txt").read_text().split()]
    P = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    rd = out / "omega_run" / "results_output"
    xyz, rgb, cf, fg, rr, cc = [], [], [], [], [], []
    st = int(stride)
    for f, c2w in zip(frames, P):
        with np.load(rd / f"frame_{f}.npz") as z:
            d, conf, Kg = np.asarray(z["depth"]), np.asarray(z["conf"], np.float32), np.asarray(z["K_omega"], np.float64)
        r_, c_ = np.nonzero(d[::st, ::st] > 0)
        r_, c_ = r_ * st, c_ * st
        z_ = d[r_, c_].astype(np.float64)
        pc = np.stack([(c_ - Kg[0, 2]) / Kg[0, 0] * z_, (r_ - Kg[1, 2]) / Kg[1, 1] * z_, z_], -1)
        xyz.append((pc @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32))
        uv = np.rint(grid_to_native(np.stack([c_, r_], -1).astype(np.float64), g)).astype(int)
        pix = np.asarray(Image.open(frames_dir / f"{f:06d}.jpg").convert("RGB"))
        rgb.append(pix[np.clip(uv[:, 1], 0, pix.shape[0] - 1), np.clip(uv[:, 0], 0, pix.shape[1] - 1)])
        cf.append(conf[r_, c_])
        fg.append(np.full(len(z_), f, np.int32))
        rr.append(r_.astype(np.int16))
        cc.append(c_.astype(np.int16))
    XYZ = np.concatenate(xyz)
    CF, FG, RR, CC = np.concatenate(cf), np.concatenate(fg), np.concatenate(rr), np.concatenate(cc)
    write_ply(out / "chunk_0.ply", XYZ, np.concatenate(rgb), CF, FG, RR, CC)
    np.savez(out / "chunk_0_origins.npz", frame_global=FG, pixel_row=RR, pixel_col=CC, confidence=CF)
    return int(len(XYZ))
