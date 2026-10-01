"""DA3 pose-conditioned depth, v2 (USER 2026-09-30: fixes 1, 2, 3 after v1's floor came out layered).
Omega was checked first: its forward takes images only (vggt_omega.py:35), no camera input.

  1. OVERLAPPING windows (WIN keyframes, 50 % overlap). Each window is Sim(3)-aligned to F5's poses
     on its own (DA3's align_to_input_ext_scale), so each carries its own depth scale; the frames two
     windows share MEASURE the ratio between them (median d_a/d_b per shared frame) and one
     least-squares solve over all windows brings them to a common scale. Gauge: mean log-scale = 0,
     i.e. F5's metric on average, as the Umeyama fits gave it. Each keyframe then takes its depth
     from the window where it sits most central.
  2. NO confidence gate. Only pixels with no real depth leave: non-finite, <= 0, and DA3's own sky
     pixels (it overwrites them with the scene's p99 depth).
  3. NATIVE resolution: process_res = the frames' long side.

Depth maps are saved (output/da3_posed_depth/) so a multi-view fusion can run later without DA3.
Result: output/_epoch_8/ with a manifest, selectable in the viewer; v1 stays as epoch 7.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
O = S / "output"
DST = O / "_epoch_8"
TMP = O / "_tx_da3_posed_v2"
DEPTH_DIR = O / "da3_posed_depth"
WIN = int(os.environ.get("DA3_WIN", "48"))
STEP = WIN // 2
SKY_THR = 0.3                      # DA3's own non-sky threshold (utils/alignment.compute_sky_mask)
sys.path.insert(0, "/workspace/stac-build/server")
sys.path.insert(0, "/workspace/stac-build/vendor/depth-anything-3/src")


def log(m):
    print(f"[da3-v2 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def main() -> int:
    import torch
    from PIL import Image
    t0 = time.time()
    frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
    c2w = np.loadtxt(O / "precision" / "f5_camera_poses.txt").reshape(-1, 4, 4)
    cam = json.loads((O / "camera.json").read_text())
    fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]
    W0, H0 = int(cam["width"]), int(cam["height"])
    K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    N = len(frames)
    assert len(c2w) == N, (len(c2w), N)
    starts = list(range(0, max(N - WIN, 0), STEP)) + [max(N - WIN, 0)]
    log(f"{N} keyframes, F5 camera fx {fx:.1f} ({W0}x{H0}), {len(starts)} windows of {WIN} "
        f"at 50 % overlap, process_res {max(W0, H0)}")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda").eval()
    for d in (TMP, DST, DEPTH_DIR):
        if d.exists():
            shutil.rmtree(d)
    (TMP / "chunks").mkdir(parents=True)
    DEPTH_DIR.mkdir()
    copies = {}                                    # (window, i) -> (depth, conf, valid, K)
    for w, s0 in enumerate(starts):
        idx = list(range(s0, min(s0 + WIN, N)))
        imgs = [str(S / "frames" / f"{frames[i]:06d}.jpg") for i in idx]
        ext = np.stack([np.linalg.inv(c2w[i]) for i in idx]).astype(np.float32)      # w2c
        ixt = np.stack([K0] * len(idx)).astype(np.float32)
        with torch.no_grad():
            pred = model.inference(imgs, extrinsics=ext, intrinsics=ixt, align_to_input_ext_scale=True,
                                   process_res=max(W0, H0))
        d, c, k = _np(pred.depth), _np(pred.conf), _np(pred.intrinsics)
        sky = _np(pred.sky) if pred.sky is not None else None
        for j, i in enumerate(idx):
            valid = np.isfinite(d[j]) & (d[j] > 0)
            if sky is not None:
                valid &= sky[j] < SKY_THR
            copies[(w, i)] = (d[j].astype(np.float32), c[j].astype(np.float16), valid, k[j].astype(np.float64))
        log(f"window {w + 1}/{len(starts)}: kf {idx[0]}-{idx[-1]}, depth {d.shape[1:]}, "
            f"sky {0 if sky is None else float((sky >= SKY_THR).mean()) * 100:.2f} %, "
            f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB ({time.time() - t0:.0f} s)")
    del model
    torch.cuda.empty_cache()

    # 1. one scale per window from the frames two windows share
    rows, rhs, pair_log = [], [], []
    for (wa, i) in copies:
        for wb in range(wa + 1, len(starts)):
            if (wb, i) in copies:
                da, _, va, _ = copies[(wa, i)]
                db, _, vb, _ = copies[(wb, i)]
                m = va & vb
                if m.sum() < 1000:
                    continue
                lr = float(np.median(np.log(da[m]) - np.log(db[m])))
                r = np.zeros(len(starts)); r[wa], r[wb] = 1.0, -1.0
                rows.append(r); rhs.append(-lr); pair_log.append((wa, wb, lr))
    gauge = np.ones(len(starts)) / len(starts)
    A, b = np.vstack(rows + [gauge]), np.array(rhs + [0.0])
    logs = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = A[:-1] @ logs - b[:-1]
    scale = np.exp(logs)
    by_pair = {}
    for (wa, wb, lr) in pair_log:
        by_pair.setdefault((wa, wb), []).append(lr)
    spread = [float(np.std(v)) for v in by_pair.values() if len(v) > 1]
    log(f"window scales vs each other: {', '.join(f'{s:.3f}' for s in scale)}")
    log(f"shared-frame ratios: {len(pair_log)} measured, raw |log| median {np.median(np.abs([p[2] for p in pair_log])) * 100:.2f} %, "
        f"after solve |resid| median {np.median(np.abs(resid)) * 100:.2f} % p90 {np.percentile(np.abs(resid), 90) * 100:.2f} %, "
        f"within one pair's frames std median {np.median(spread) * 100 if spread else 0:.2f} %")

    # each keyframe from the window where it is most central, scaled
    owner = {}
    for i in range(N):
        cands = [w for w in range(len(starts)) if (w, i) in copies]
        owner[i] = min(cands, key=lambda w: abs(i - (starts[w] + (min(starts[w] + WIN, N) - 1)) / 2))
    from precision.epoch0_cloud import _write_ply_xyzrgb
    n_tot, n_pix = 0, 0
    for s0 in range(0, N, STEP):
        idx = list(range(s0, min(s0 + STEP, N)))
        xyz_l, rgb_l, fg_l, pr_l, pc_l, cf_l = [], [], [], [], [], []
        for i in idx:
            w = owner[i]
            d, c, valid, K = copies[(w, i)]
            d = d * scale[w]
            H, W = d.shape
            np.savez(DEPTH_DIR / f"kf_{i:04d}.npz", depth=d.astype(np.float32), conf=c, valid=valid,
                     K=K, c2w=c2w[i], frame=frames[i], window=w, window_scale=scale[w])
            rr, cc = np.nonzero(valid)
            n_pix += H * W
            z = d[rr, cc].astype(np.float64)
            Xc = np.stack([(cc - K[0, 2]) / K[0, 0] * z, (rr - K[1, 2]) / K[1, 1] * z, z], 1)
            Xw = Xc @ c2w[i][:3, :3].T + c2w[i][:3, 3]
            img = np.asarray(Image.open(S / "frames" / f"{frames[i]:06d}.jpg").convert("RGB").resize((W, H)))
            xyz_l.append(Xw.astype(np.float32)); rgb_l.append(img[rr, cc])
            fg_l.append(np.full(len(rr), frames[i], np.int32))
            pr_l.append(np.round((rr + 0.5) * H0 / H - 0.5).astype(np.int16))
            pc_l.append(np.round((cc + 0.5) * W0 / W - 0.5).astype(np.int16))
            cf_l.append(c[rr, cc].astype(np.float32))
        xyz = np.concatenate(xyz_l)
        name = f"chunk_{s0 // STEP:03d}"
        _write_ply_xyzrgb(TMP / "chunks" / f"{name}.ply", xyz, np.concatenate(rgb_l).astype(np.uint8))
        np.savez(TMP / "chunks" / f"{name}_origins.npz", frame_global=np.concatenate(fg_l),
                 pixel_row=np.concatenate(pr_l), pixel_col=np.concatenate(pc_l),
                 confidence=np.concatenate(cf_l))
        n_tot += len(xyz)
    copies.clear()
    log(f"{n_tot:,} raw points of {n_pix:,} pixels ({n_tot / n_pix * 100:.1f} % kept: only no-depth + sky out)")
    from config import cfg
    from precision.epoch0_cloud import clean_cmd
    cleaned = TMP / "cleaned_cloud.ply"
    r = subprocess.run(clean_cmd(cfg, TMP / "chunks", cleaned), cwd="/workspace/stac-build/server",
                       capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if "✅" in ln:
            log(ln.strip())
    if r.returncode != 0 or not cleaned.exists():
        log(f"❌ cleaner failed: {r.stderr[-800:]}")
        return 1
    from potree_converter import convert_ply_to_potree
    if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
        log("❌ octree failed")
        return 1
    DST.mkdir()
    shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply"))
    shutil.move(str(TMP / "potree"), str(DST / "potree"))
    shutil.copy(O / "precision" / "f5_camera_poses.txt", DST / "camera_poses.txt")
    (DST / "_manifest.json").write_text(json.dumps({
        "epoch": 8, "epoch_from": 8, "epoch_to": 8, "kind": "new_cloud",
        "note": "DA3 depth conditioned on F5 camera + poses, v2: overlapping windows on one scale, "
                "no confidence gate, native resolution (outside the pipeline, 2026-09-30)",
        "artifacts": [{"rel": "cleaned_cloud.ply", "existed_before": True},
                      {"rel": "potree", "existed_before": True},
                      {"rel": "camera_poses.txt", "existed_before": True}]}))
    shutil.rmtree(TMP, ignore_errors=True)
    n = json.loads((DST / "potree" / "metadata.json").read_text())["points"]
    log(f"DONE: {n:,} pts → {DST} (selectable as epoch 8) in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
