"""F6 — COLMAP PatchMatch as the A/B reference of the plane sweep (claude_stac.txt §4-F6).

Comparison only: nothing downstream reads these depths. COLMAP sees what the
sweep saw — F5's camera (the undistorted PINHOLE with K, F0's maps), F5's poses,
the same undistorted native images, and for every keyframe the same consistency
neighbours the sweep used (``depth_native/report.json``). Its per-image depth
range comes from the landmarks (F4's tracks triangulated with F5) and from
``prior_points_per_image`` samples of the keyframe's own prior (single-image
tracks): COLMAP's ``ComputeDepthRanges`` reads both.

Declared: the reference runs on KEYFRAMES only — COLMAP's geometric pass needs
every source image's depth map, and witnesses have none; the sweep's photometric
views also include the localised witnesses.

Needs the COLMAP binary built WITH CUDA (``precision.depth.colmap.binary``; the
env ``colmap`` carries one). Outputs ``output/depth_colmap/frame_<n>.npz``
{depth f32} and the ``colmap_ab`` section of ``output/depth_native/report.json``.

CLI: ``python -m precision.depth_colmap --session <dir> [--keep-workspace]``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

COLMAP_DIRNAME = "depth_colmap"
WORKSPACE_DIRNAME = "_colmap_ws"
LOG_TAG = "[depth-colmap]"


class ColmapError(RuntimeError):
    """The reference cannot run — with the exact reason."""


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix — COLMAP's images.txt convention."""
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.asarray(q)
    return q if q[0] >= 0 else -q


def qvec_to_rotmat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def image_name(frame: int) -> str:
    return f"{int(frame):06d}.png"


def write_sparse(sparse_dir: Path, K: np.ndarray, wh, frames: Sequence[int],
                 w2c: np.ndarray, points: Sequence[Dict[str, Any]]) -> None:
    """COLMAP text model. ``points``: {"X": (3,), "obs": [(image index, (u, v))]} with
    undistorted pixel coordinates (COLMAP's pixel centre convention: +0.5)."""
    sparse_dir.mkdir(parents=True, exist_ok=True)
    W, H = wh
    (sparse_dir / "cameras.txt").write_text(
        f"1 PINHOLE {W} {H} {K[0, 0]:.10g} {K[1, 1]:.10g} {K[0, 2] + 0.5:.10g} {K[1, 2] + 0.5:.10g}\n")
    p2d: List[List[str]] = [[] for _ in frames]
    tracks: List[List[str]] = []
    for pid, pt in enumerate(points, start=1):
        tr = []
        for i, (u, v) in pt["obs"]:
            tr.append(f"{i + 1} {len(p2d[i])}")
            p2d[i].append(f"{u + 0.5:.4f} {v + 0.5:.4f} {pid}")
        X = pt["X"]
        tracks.append(f"{pid} {X[0]:.10g} {X[1]:.10g} {X[2]:.10g} 128 128 128 0 " + " ".join(tr))
    with open(sparse_dir / "images.txt", "w") as fh:
        for i, f in enumerate(frames):
            q = rotmat_to_qvec(w2c[i][:3, :3])
            t = w2c[i][:3, 3]
            fh.write(f"{i + 1} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                     f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} 1 {image_name(f)}\n")
            fh.write(" ".join(p2d[i]) + "\n")
    (sparse_dir / "points3D.txt").write_text("\n".join(tracks) + ("\n" if tracks else ""))


def write_patch_match_cfg(stereo_dir: Path, sources: Dict[int, Sequence[int]]) -> None:
    """Each reference image and its explicit source images."""
    stereo_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for f, src in sources.items():
        lines += [image_name(f), ", ".join(image_name(s) for s in src)]
    (stereo_dir / "patch-match.cfg").write_text("\n".join(lines) + "\n")


def read_colmap_array(path: Path) -> np.ndarray:
    """COLMAP's dense .bin (``width&height&channels&`` + float32, column-major)."""
    with open(path, "rb") as fh:
        head = b""
        amps = 0
        while amps < 3:
            c = fh.read(1)
            if not c:
                raise ColmapError(f"{path}: truncated header")
            head += c
            amps += c == b"&"
        w, h, ch = (int(x) for x in head.decode().split("&")[:3])
        a = np.fromfile(fh, np.float32)
    if a.size != w * h * ch:
        raise ColmapError(f"{path}: {a.size} values for {w}×{h}×{ch}")
    return np.transpose(a.reshape((w, h, ch), order="F"), (1, 0, 2)).squeeze()


def write_colmap_array(path: Path, a: np.ndarray) -> None:
    """Inverse of ``read_colmap_array`` (tests)."""
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    h, w, ch = a.shape
    with open(path, "wb") as fh:
        fh.write(f"{w}&{h}&{ch}&".encode())
        np.transpose(a, (1, 0, 2)).reshape(-1, order="F").astype(np.float32).tofile(fh)


def compare(sweep_npz: Path, colmap_depth: np.ndarray) -> Dict[str, Any]:
    """Tier-0 sweep vs COLMAP on the pixels both measured; the prior-fill tier too."""
    from precision.depth_sweep import SOURCE_PRIOR_FILL, SOURCE_SWEEP
    with np.load(sweep_npz) as z:
        d, src = z["depth"], z["source"]
    c = colmap_depth
    out: Dict[str, Any] = {"colmap_coverage": float((c > 0).mean()),
                           "sweep_tier0_coverage": float((src == SOURCE_SWEEP).mean())}
    for name, code in (("tier0", SOURCE_SWEEP), ("tier1", SOURCE_PRIOR_FILL)):
        m = (src == code) & (c > 0) & (d > 0)
        out[f"{name}_n_common"] = int(m.sum())
        out[f"{name}_median_rel"] = float(np.median(np.abs(d[m] / c[m] - 1))) if m.any() else None
        out[f"{name}_p90_rel"] = float(np.percentile(np.abs(d[m] / c[m] - 1), 90)) if m.any() else None
    return out


def run_colmap(session_dir: Path, pcfg, keep_workspace: bool = False,
               log: Callable = print) -> Dict[str, Any]:
    import cv2
    from intake.content import frame_file
    from precision import depth_sweep as DS
    session_dir = Path(session_dir)
    out = session_dir / "output"
    ccfg = pcfg.depth.colmap
    rep_path = out / DS.DEPTH_DIRNAME / DS.REPORT_NAME
    if not rep_path.exists():
        raise ColmapError(f"{rep_path} is missing — the reference compares against the sweep "
                          f"(python -m precision.depth_sweep --session <dir>)")
    rep = json.loads(rep_path.read_text())
    inp = DS.load_inputs(session_dir, pcfg)
    if any(rep.get(k) != v for k, v in inp.epochs.items()):
        raise ColmapError(f"the sweep was measured on {({k: rep.get(k) for k in inp.epochs})}, "
                          f"the session is at {inp.epochs} — re-run the sweep first")
    binary = Path(ccfg.binary)
    if not binary.exists():
        raise ColmapError(f"{binary} does not exist — COLMAP built with CUDA is required "
                          f"(precision.depth.colmap.binary)")
    probe = subprocess.run([str(binary), "help"], capture_output=True, text=True, timeout=60)
    if "with CUDA" not in (probe.stdout + probe.stderr):
        raise ColmapError(f"{binary} is not built with CUDA (its banner: "
                          f"{(probe.stdout or probe.stderr).splitlines()[:1]}) — PatchMatch "
                          f"stereo needs it")
    idx = {f: i for i, f in enumerate(inp.kf)}
    sources = {int(f): [int(s) for s in pf["consistency_views"] if int(s) in idx]
               for f, pf in rep["per_frame"].items()}
    sources = {f: s for f, s in sources.items() if s}
    if not sources:
        raise ColmapError("no keyframe of the sweep report has a consistency neighbour")
    ws = out / "precision" / WORKSPACE_DIRNAME
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "images").mkdir(parents=True)
    used = sorted(set(sources) | {s for v in sources.values() for s in v})
    t0 = time.time()
    for f in used:
        bgr = cv2.imread(str(frame_file(inp.frames_dir, f)), cv2.IMREAD_COLOR)
        und = cv2.remap(bgr, inp.maps[0], inp.maps[1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(ws / "images" / image_name(f)), und)
    # depth range per image: landmarks (real tracks) + the prior's own samples
    sub = {f: i for i, f in enumerate(used)}
    X, groups = DS.triangulated_landmarks(session_dir, pcfg, inp)
    from precision.camera import undistort_points, undistort_solver
    solver = undistort_solver(pcfg.camera)
    points = []
    for t, P in X.items():
        obs = [(sub[inp.kf[i]], p) for i, p in groups[t] if inp.kf[i] in sub]
        if len(obs) >= 2:
            uv = undistort_points(np.array([p for _, p in obs], np.float64), inp.cam, **solver)
            points.append({"X": P, "obs": [(o[0], tuple(q)) for o, q in zip(obs, uv)]})
    rng = np.random.default_rng(int(pcfg.depth.seed))
    for f in used:
        s_k = rep["per_frame"].get(str(f), {}).get("s_k")
        rec = inp.records_dir / f"frame_{f}.npz"
        if s_k is None or not rec.exists():
            continue
        with np.load(rec) as zf:
            rd = zf["depth"]
            rc = zf["conf"] if "conf" in zf.files else np.full(rd.shape, np.nan, np.float32)
        gd = cv2.imread(str(frame_file(inp.frames_dir, f)), cv2.IMREAD_GRAYSCALE).astype(np.float32) / DS.GRAY_MAX
        z0, _ = DS.prior_native(rd, rc, s_k, inp.cam, inp.maps, gd)
        vv, uu = np.nonzero(z0 > 0)
        if vv.size == 0:
            continue
        pick = rng.choice(vv.size, size=min(int(ccfg.prior_points_per_image), vv.size), replace=False)
        i = idx[f]
        c2w = np.linalg.inv(inp.kf_w2c[i])
        for u, v in zip(uu[pick], vv[pick]):
            z = float(z0[v, u])
            Pc = np.array([(u - inp.K[0, 2]) / inp.K[0, 0] * z, (v - inp.K[1, 2]) / inp.K[1, 1] * z, z])
            points.append({"X": c2w[:3, :3] @ Pc + c2w[:3, 3], "obs": [(sub[f], (float(u), float(v)))]})
    write_sparse(ws / "sparse", inp.K, inp.wh, used, inp.kf_w2c[[idx[f] for f in used]], points)
    write_patch_match_cfg(ws / "stereo", sources)
    for d in ("depth_maps", "normal_maps", "consistency_graphs"):
        (ws / "stereo" / d).mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "patch_match_stereo", "--workspace_path", str(ws),
           "--workspace_format", "COLMAP",
           "--PatchMatchStereo.max_image_size", "-1",
           "--PatchMatchStereo.window_radius", str(ccfg.window_radius),
           "--PatchMatchStereo.num_iterations", str(ccfg.num_iterations),
           "--PatchMatchStereo.geom_consistency", "1" if ccfg.geom_consistency else "0",
           "--PatchMatchStereo.gpu_index", "0"]
    log(f"{LOG_TAG} {len(sources)} reference keyframe(s), {len(used)} image(s), "
        f"{len(points):,} depth-range point(s) → {' '.join(cmd[:2])}")
    logf = ws / "colmap.log"
    with open(logf, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = logf.read_text().splitlines()[-15:]
        raise ColmapError(f"COLMAP patch_match_stereo exited with {rc}: " + " | ".join(tail))
    kind = "geometric" if ccfg.geom_consistency else "photometric"
    cdir = out / COLMAP_DIRNAME
    cdir.mkdir(parents=True, exist_ok=True)
    per_frame, missing, dms = {}, [], {}
    for f in sources:
        p = ws / "stereo" / "depth_maps" / f"{image_name(f)}.{kind}.bin"
        if not p.exists():
            missing.append(f)
            continue
        dm = read_colmap_array(p).astype(np.float32)
        dm = np.where(np.isfinite(dm) & (dm > 0), dm, 0.0).astype(np.float32)
        dms[f] = dm
        sw = out / DS.DEPTH_DIRNAME / f"frame_{f}.npz"
        if sw.exists():
            per_frame[str(f)] = compare(sw, dm)
    # TIER-2 EVIDENCE (USER 2026-09-29): COLMAP's depth judged by the sweep's own rule —
    # confirmations and contradictions against the neighbours' COLMAP depths, at the
    # sweep's τ_rel / τ_px — so F7 can admit it exactly as it admits tier 0
    tau_rel, tau_px = float(rep["tau_rel"]), float(rep["tau_px"])
    kf_index = {int(f): i for i, f in enumerate(inp.kf)}
    dev = DS._device()
    n_t2 = 0
    for f, dm in dms.items():
        nb = [g for g in sources[f] if g in dms]
        if nb:
            n_c, r_c, n_b = DS.consistency(dm, inp.K, inp.kf_w2c[kf_index[f]], [dms[g] for g in nb],
                                           [inp.kf_w2c[kf_index[g]] for g in nb], tau_rel, tau_px,
                                           device=dev)
        else:
            n_c = np.zeros(dm.shape, np.uint8); n_b = np.zeros(dm.shape, np.uint8)
            r_c = np.full(dm.shape, np.nan, np.float32)
        n_t2 += int(((dm > 0) & (n_c >= int(pcfg.fuse.min_witness_views)) & (n_b < n_c)).sum())
        np.savez_compressed(cdir / f"frame_{f}.npz", depth=dm, n_consistent=n_c, n_contradict=n_b,
                            residual_rel=r_c.astype(np.float32))
    agg = {}
    for key in ("tier0_median_rel", "tier1_median_rel", "colmap_coverage", "sweep_tier0_coverage"):
        v = [r[key] for r in per_frame.values() if r.get(key) is not None]
        agg[key] = float(np.median(v)) if v else None
    rep["colmap_ab"] = {"colmap_binary": str(binary), "depth_kind": kind,
                        "params": {k: getattr(ccfg, k) for k in ccfg.__dataclass_fields__},
                        "keyframes_only": True, "n_frames": len(per_frame),
                        "tier2_candidates": n_t2, "tau_rel": tau_rel, "tau_px": tau_px,
                        "missing_depth_maps": missing, "median_over_frames": agg,
                        "per_frame": per_frame, "seconds": round(time.time() - t0, 1)}
    rep_path.write_text(json.dumps(rep, indent=1, default=float))
    if not keep_workspace:
        shutil.rmtree(ws, ignore_errors=True)
    log(f"{LOG_TAG} {len(per_frame)} frame(s) compared: sweep tier 0 vs COLMAP median "
        f"{(agg['tier0_median_rel'] or float('nan')) * 100:.2f} % (median over frames), "
        f"COLMAP coverage {(agg['colmap_coverage'] or 0) * 100:.1f} %; tier-2 candidates "
        f"(consistent, ≥ {pcfg.fuse.min_witness_views} views) {n_t2:,} px → {rep_path}")
    return rep["colmap_ab"]


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.depth_colmap",
                                 description="COLMAP PatchMatch reference for the F6 sweep.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--keep-workspace", action="store_true")
    args = ap.parse_args(argv)
    run_colmap(Path(args.session), load_precision_config(), keep_workspace=args.keep_workspace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
