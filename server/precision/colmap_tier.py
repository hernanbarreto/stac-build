"""The photometric reference of the chain — COLMAP PatchMatch stereo with F5's camera and poses, as
TIER 0 of the owner fusion (audit 2026-10-23, phase 3; USER: "que la fotogrametría clásica y la IA
queden perfectamente fusionadas para lograr la mejor nube para ingeniería").

Classical multi-view stereo measures depth by photometric consistency across views: where a
surface has texture it is accurate to a fraction of a pixel of disparity (pccr 2026-10-17: OpenMVS
with OUR poses agreed with the certified cloud to p50 0.6 cm where it measured) and where it has
none it measures nothing. The learned depth (Omega) measures everything with a relative error of
the order of a percent. The two are complementary, and the chain now treats them as instruments of
different accuracy, never as rivals:
  * this step runs COLMAP PatchMatch (geometric consistency on) over the keyframes with F5's
    camera (the undistorted PINHOLE K on F0's maps), F5's poses and the same undistorted images the
    bend works on; each keyframe's source images are its most COVISIBLE keyframes, measured on
    Omega's record (loops and revisits included, never an index window); its depth range comes
    from F5's landmarks and from samples of Omega's record (COLMAP's ComputeDepthRanges reads both);
  * every COLMAP depth is then judged by the sweep's own consistency rule against the other COLMAP
    maps — confirmations (n_consistent) and contradictions (n_contradict: a view sees free space
    through the point) at the instrument's OWN tolerance, tau_rel measured as the p-quantile of the
    disagreement between COLMAP maps (never a declared number) and tau_px = F5's held-out RMS × k;
  * depth_on_f5 reads the result: the bend may fit Omega to COLMAP's dense depth (judged on F5's
    held-out landmarks, which COLMAP never saw), and the owner fusion gets the consistent COLMAP
    pixels as tier 0 with their own tau — a photometric observation outranks every learned one of
    the same element and never averages with it.

Needs the COLMAP binary built WITH CUDA (reconstruction.precision.depth.colmap.binary). Outputs
``output/depth_colmap_f5/frame_<n>.npz`` {depth f32, n_consistent u8, n_contradict u8, residual_rel
f32} and ``output/depth_colmap_f5/report.json`` stamped with the session's epochs and reconstruction
id — depth_on_f5 refuses a report of another geometry. The directory is the tier's own: the sweep
chain's ``output/depth_colmap`` (F6 COLMAP A/B, its F7's tier 2) is measured with another geometry
and neither chain may read the other's maps.

CLI: ``python -m precision.colmap_tier --session <dir> [--keep-workspace]``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from precision.depth_colmap import (ColmapError, image_name, read_colmap_array,  # noqa: F401
                                    write_patch_match_cfg, write_sparse)

TIER_DIRNAME = "depth_colmap_f5"          # F5's geometry; never the sweep chain's depth_colmap
REPORT_NAME = "report.json"
WORKSPACE_DIRNAME = "_colmap_tier_ws"
LOG_TAG = "[colmap-tier]"
TIER_VERSION = 1


def report_path(out: Path) -> Path:
    return Path(out) / TIER_DIRNAME / REPORT_NAME


def _records(inp, frames: Sequence[int], log: Callable):
    """Omega's record depth per keyframe on the bend's grid (the undistorted native frame; carried
    through the lens and the record grid when they differ — depth_on_f5's own convention), with a
    validity mask (finite, positive, not sky)."""
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import SKY_CONF
    W, H = int(inp.wh[0]), int(inp.wh[1])
    lens = bool(np.any(inp.cam.dist()))
    with np.load(inp.records_dir / f"frame_{frames[0]}.npz") as z:
        rec_hw = tuple(int(x) for x in np.asarray(z["depth"]).shape)
    carry = lens or rec_hw != (H, W)
    depth, valid = {}, {}
    for f in frames:
        with np.load(inp.records_dir / f"frame_{f}.npz") as z:
            d = np.asarray(z["depth"], np.float32); cf = np.asarray(z["conf"], np.float32)
        if carry:
            d = CC.record_on_native(d, inp.cam, inp.maps); cf = CC.record_on_native(cf, inp.cam, inp.maps)
        valid[f] = np.isfinite(d) & (d > 0) & np.isfinite(cf) & (cf > SKY_CONF)
        depth[f] = np.where(valid[f], d, 0.0).astype(np.float32)
    if carry:
        log(f"{LOG_TAG} Omega's record {rec_hw[1]}x{rec_hw[0]}" + (" + lens" if lens else "")
            + f" carried onto the undistorted native grid {W}x{H}")
    return depth, valid


def sources_by_covisibility(frames: Sequence[int], depth: Dict[int, np.ndarray], valid: Dict[int, np.ndarray],
                            K: np.ndarray, c2w: Dict[int, np.ndarray], stride: int, min_frac: float,
                            max_sources: int) -> Dict[int, List[int]]:
    """Each keyframe's source images: its most covisible keyframes on Omega's record (owner_fusion's
    covisibility — the frustum overlap measured on the depth maps)."""
    from precision.owner_fusion import candidates_of, covisibility
    frames = [int(f) for f in frames]
    cov = covisibility(frames, depth, valid, K, c2w, int(stride))
    cands = candidates_of(cov, float(min_frac), int(max_sources))
    return {f: [frames[j] for j in cands[i]] for i, f in enumerate(frames)}


def depth_range_points(session_dir: Path, pcfg, inp, frames: Sequence[int], used: Sequence[int],
                       depth: Dict[int, np.ndarray], samples_per_image: int, seed: int) -> List[Dict[str, Any]]:
    """COLMAP's depth-range points: F5's landmarks (real tracks, undistorted pixels) and
    ``samples_per_image`` samples of Omega's record per keyframe (single-image tracks)."""
    from precision import depth_sweep as DS
    from precision.camera import undistort_points, undistort_solver
    sub = {int(f): i for i, f in enumerate(used)}
    idx = {int(f): i for i, f in enumerate(inp.kf)}
    X, groups = DS.triangulated_landmarks(session_dir, pcfg, inp)
    solver = undistort_solver(pcfg.camera)
    points: List[Dict[str, Any]] = []
    for t, P in X.items():
        obs = [(sub[int(inp.kf[i])], p) for i, p in groups[t] if int(inp.kf[i]) in sub]
        if len(obs) >= 2:
            uv = undistort_points(np.array([p for _, p in obs], np.float64), inp.cam, **solver)
            points.append({"X": np.asarray(P, np.float64), "obs": [(o[0], tuple(float(x) for x in q)) for o, q in zip(obs, uv)]})
    rng = np.random.default_rng(int(seed))
    for f in used:
        d = depth.get(int(f))
        if d is None:
            continue
        vv, uu = np.nonzero(d > 0)
        if vv.size == 0:
            continue
        pick = rng.choice(vv.size, size=min(int(samples_per_image), vv.size), replace=False)
        c2w = np.linalg.inv(inp.kf_w2c[idx[int(f)]])
        for u, v in zip(uu[pick], vv[pick]):
            z = float(d[v, u])
            Pc = np.array([(u - inp.K[0, 2]) / inp.K[0, 0] * z, (v - inp.K[1, 2]) / inp.K[1, 1] * z, z])
            points.append({"X": c2w[:3, :3] @ Pc + c2w[:3, 3], "obs": [(sub[int(f)], (float(u), float(v)))]})
    return points


def measured_tau_rel(frames: Sequence[int], dms: Dict[int, np.ndarray], sources: Dict[int, List[int]],
                     K: np.ndarray, c2w: Dict[int, np.ndarray], quantile_pct: float, stride: int = 3) -> float:
    """The instrument's own agreement tolerance: the ``quantile_pct`` percentile of |z_f→g − d_g| / d_g
    over the pixels COLMAP measured in both of a (reference, source) pair — the construction of the
    vote's tau, on COLMAP's maps."""
    import torch
    from precision import f6_torch as FT
    from precision.owner_fusion import _frame_geometry, carry
    samp = []
    fr = [int(f) for f in frames if int(f) in dms]
    for f in fr[::max(1, int(stride))]:
        H, W = dms[f].shape
        ray_w, C = _frame_geometry(f, H, W, K, c2w)
        z_f = FT._t(dms[f]); v_f = z_f > 0
        for g in sources.get(f, []):
            if g not in dms:
                continue
            z_g = FT._t(dms[g])
            # a tolerance of 1 (every landing counts) returns the raw disagreement through `t`
            agree, t, _, _ = carry(z_f, v_f, ray_w, C, np.linalg.inv(c2w[g]), z_g, z_g > 0, K, 1.0, False)
            m = agree & torch.isfinite(t) & (t > 0)
            if bool(m.any()):
                samp.append(torch.abs(t[m] - z_f[m]) / z_f[m])
    if not samp:
        raise ColmapError("no two COLMAP depth maps share a pixel — the instrument's tolerance cannot be measured")
    return float(FT.quantile(torch.cat(samp), [float(quantile_pct) / 100.0])[0])


def run_colmap_tier(session_dir: Path, pcfg, keep_workspace: bool = False, log: Callable = print,
                    runner: Optional[Callable[[List[str], Path], int]] = None) -> Dict[str, Any]:
    """The step. ``runner(cmd, log_file) -> returncode`` replaces the COLMAP subprocess (tests)."""
    import cv2
    from intake.quality import frame_file, read_session_epochs
    from precision import depth_sweep as DS
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    session_dir = Path(session_dir)
    out = session_dir / "output"
    fu = pcfg.fusion
    ccfg = pcfg.depth.colmap
    t0 = time.time()
    inp = DS.load_inputs(session_dir, pcfg)
    frames = [int(f) for f in inp.kf]
    idx = {f: i for i, f in enumerate(frames)}
    c2w = {f: np.linalg.inv(inp.kf_w2c[i]) for f, i in idx.items()}
    K = np.asarray(inp.K, np.float64)
    depth, valid = _records(inp, frames, log)
    sources = sources_by_covisibility(frames, depth, valid, K, c2w, fu.covis_stride, fu.min_covis_frac,
                                      fu.colmap_max_sources)
    sources = {f: s for f, s in sources.items() if s}
    if not sources:
        raise ColmapError("no keyframe is covisible with another on Omega's record — PatchMatch has no source images")
    binary = Path(ccfg.binary)
    if runner is None:
        if not binary.exists():
            raise ColmapError(f"{binary} does not exist — COLMAP built with CUDA is required "
                              f"(precision.depth.colmap.binary)")
        probe = subprocess.run([str(binary), "help"], capture_output=True, text=True, timeout=60)
        if "with CUDA" not in (probe.stdout + probe.stderr):
            raise ColmapError(f"{binary} is not built with CUDA — PatchMatch stereo needs it")
    ws = out / "precision" / WORKSPACE_DIRNAME
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "images").mkdir(parents=True)
    used = sorted(set(sources) | {s for v in sources.values() for s in v})
    for f in used:
        bgr = cv2.imread(str(frame_file(inp.frames_dir, f)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ColmapError(f"frame image {frame_file(inp.frames_dir, f)} is unreadable")
        und = cv2.remap(bgr, inp.maps[0], inp.maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(ws / "images" / image_name(f)), und)
    points = depth_range_points(session_dir, pcfg, inp, frames, used, depth, ccfg.prior_points_per_image,
                                pcfg.depth.seed)
    write_sparse(ws / "sparse", K, inp.wh, used, inp.kf_w2c[[idx[f] for f in used]], points)
    write_patch_match_cfg(ws / "stereo", sources)
    for d in ("depth_maps", "normal_maps", "consistency_graphs"):
        (ws / "stereo" / d).mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "patch_match_stereo", "--workspace_path", str(ws), "--workspace_format", "COLMAP",
           "--PatchMatchStereo.max_image_size", "-1",
           "--PatchMatchStereo.window_radius", str(ccfg.window_radius),
           "--PatchMatchStereo.num_iterations", str(ccfg.num_iterations),
           "--PatchMatchStereo.geom_consistency", "1" if ccfg.geom_consistency else "0",
           "--PatchMatchStereo.gpu_index", "0"]
    n_src = [len(v) for v in sources.values()]
    log(f"{LOG_TAG} {len(sources)} reference keyframe(s), sources per keyframe min {min(n_src)} median "
        f"{int(np.median(n_src))} max {max(n_src)} (covisibility), {len(points):,} depth-range point(s) → "
        f"{' '.join(cmd[:2])}")
    logf = ws / "colmap.log"
    if runner is not None:
        rc = int(runner(cmd, logf))
    else:
        with open(logf, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = logf.read_text().splitlines()[-15:] if logf.exists() else []
        raise ColmapError(f"COLMAP patch_match_stereo exited with {rc}: " + " | ".join(tail))
    kind = "geometric" if ccfg.geom_consistency else "photometric"
    dms: Dict[int, np.ndarray] = {}
    missing = []
    for f in sources:
        p = ws / "stereo" / "depth_maps" / f"{image_name(f)}.{kind}.bin"
        if not p.exists():
            missing.append(f)
            continue
        dm = read_colmap_array(p).astype(np.float32)
        if dm.shape != (int(inp.wh[1]), int(inp.wh[0])):
            raise ColmapError(f"COLMAP's depth map of keyframe {f} is {dm.shape[1]}x{dm.shape[0]}, the camera grid "
                              f"{inp.wh[0]}x{inp.wh[1]} — PatchMatch must run at the native size (max_image_size -1)")
        dms[f] = np.where(np.isfinite(dm) & (dm > 0), dm, 0.0).astype(np.float32)
    if not dms:
        raise ColmapError("COLMAP wrote no depth map")
    # the instrument's own tolerance, then every map judged by the sweep's consistency rule
    tau_rel = measured_tau_rel(frames, dms, sources, K, c2w, fu.colmap_tau_quantile)
    tau_px = float(inp.tau_px)
    dev = DS._device()
    cdir = out / TIER_DIRNAME
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("frame_*.npz"):
        old.unlink()
    per_frame: Dict[str, Any] = {}
    n_tier0 = n_meas = 0
    for f, dm in dms.items():
        nb = [g for g in sources[f] if g in dms]
        if nb:
            n_c, r_c, n_b = DS.consistency(dm, K, inp.kf_w2c[idx[f]], [dms[g] for g in nb],
                                           [inp.kf_w2c[idx[g]] for g in nb], tau_rel, tau_px, device=dev)
        else:
            n_c = np.zeros(dm.shape, np.uint8); n_b = np.zeros(dm.shape, np.uint8)
            r_c = np.full(dm.shape, np.nan, np.float32)
        t0m = (dm > 0) & (n_c >= int(fu.colmap_min_views)) & (n_b < n_c)
        n_tier0 += int(t0m.sum()); n_meas += int((dm > 0).sum())
        per_frame[str(f)] = {"measured": float((dm > 0).mean()), "tier0": float(t0m.mean()), "sources": nb}
        np.savez_compressed(cdir / f"frame_{f}.npz", depth=dm, n_consistent=np.asarray(n_c, np.uint8),
                            n_contradict=np.asarray(n_b, np.uint8), residual_rel=np.asarray(r_c, np.float32))
    H, W = int(inp.wh[1]), int(inp.wh[0])
    epochs = read_session_epochs(session_dir)
    rep = {"version": TIER_VERSION, "stage": "colmap_tier", "provenance": "tool_measured", **epochs,
           RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(out),
           "colmap_binary": str(binary), "depth_kind": kind, "grid": [W, H],
           "params": {k: getattr(ccfg, k) for k in ccfg.__dataclass_fields__},
           "fusion_params": {"colmap_min_views": int(fu.colmap_min_views), "colmap_max_sources": int(fu.colmap_max_sources),
                             "covis_stride": int(fu.covis_stride), "min_covis_frac": float(fu.min_covis_frac),
                             "colmap_tau_quantile": float(fu.colmap_tau_quantile)},
           "tau_rel": tau_rel, "tau_px": tau_px, "n_frames": len(dms), "missing_depth_maps": missing,
           "measured_share": (n_meas / float(len(dms) * H * W)) if dms else None,
           "tier0_share": (n_tier0 / float(len(dms) * H * W)) if dms else None,
           "tier0_pixels": n_tier0, "per_frame": per_frame, "seconds": round(time.time() - t0, 1)}
    report_path(out).write_text(json.dumps(rep, indent=1, default=float))
    if not keep_workspace:
        shutil.rmtree(ws, ignore_errors=True)
    log(f"{LOG_TAG} {len(dms)} depth map(s): measured {rep['measured_share'] * 100:.1f} % of the pixels, tier 0 "
        f"(>= {fu.colmap_min_views} consistent views, fewer contradictions) {rep['tier0_share'] * 100:.1f} %; "
        f"tau_rel {tau_rel * 100:.2f} % (p{fu.colmap_tau_quantile:g} of the maps' own disagreement), tau_px {tau_px:.2f}; "
        f"{rep['seconds']} s → {report_path(out)}")
    return rep


def load_colmap_tier(out: Path, frames: Sequence[int], epochs: Dict[str, int], rid: Optional[str],
                     min_views: int, wh: Tuple[int, int]) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], dict]:
    """depth_on_f5's reader: ({frame: (depth f32, tier0 mask)} for the keyframes COLMAP measured, the
    report). The report must be THIS geometry's (its epochs and reconstruction id) — another's is
    refused by name, never used."""
    from correction.epoch import same_reconstruction
    rp = report_path(out)
    if not rp.exists():
        raise ColmapError(f"{rp} is missing — run python -m precision.colmap_tier (the chain's f6_colmap_tier step) first")
    rep = json.loads(rp.read_text())
    for k, v in epochs.items():
        if rep.get(k) != v:
            raise ColmapError(f"{rp} was measured at {k} {rep.get(k)}, the session is at {v} — re-run the COLMAP tier "
                              f"on this geometry")
    taken, why = same_reconstruction(rep, rid)
    if not taken:
        raise ColmapError(f"{rp}: {why} — the COLMAP tier belongs to another reconstruction")
    W, H = int(wh[0]), int(wh[1])
    maps: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    cdir = Path(out) / TIER_DIRNAME
    for f in frames:
        p = cdir / f"frame_{int(f)}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            d = np.asarray(z["depth"], np.float32); n_c = np.asarray(z["n_consistent"]); n_b = np.asarray(z["n_contradict"])
        if d.shape != (H, W):
            raise ColmapError(f"{p} is {d.shape[1]}x{d.shape[0]}, the bend's grid {W}x{H}")
        tier0 = (d > 0) & (n_c >= int(min_views)) & (n_b < n_c)
        maps[int(f)] = (np.where(tier0, d, 0.0).astype(np.float32), tier0)
    return maps, rep


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.colmap_tier",
                                 description="COLMAP PatchMatch with F5's geometry — tier 0 of the owner fusion.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--keep-workspace", action="store_true")
    args = ap.parse_args(argv)
    run_colmap_tier(Path(args.session), load_precision_config(session_dir=Path(args.session)),
                    keep_workspace=args.keep_workspace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
