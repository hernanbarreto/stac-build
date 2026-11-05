"""The structure probe and the revisited zones in the reconstruction (USER 2026-11-05: "una pasada de omega ahí ves
cómo se distribuye todo … es recto … o es un tramo con muchas vueltas"; "cada zona revisitada es un chunk, con todos sus
pedazos del recorrido").

1. ``structure_probe``: ONE VGGT-Ω pass over every keyframe at the resolution where they all fit the card
   (vendor/VGGT-Long/stac_frames_pass.py, with SALAD), then ``walk_structure.analyse``: who sees whom, the revisits,
   the zones — the revisits the plan's chain of chunks does not already join. The pass is stamped and reused
   (intake/structure_probe.npz + .stamp.json); the analysis is written with a figure (intake/structure.json, .png).
2. ``apply_zones`` — after the chunked pass has been aligned and its Omega records written: each zone (all its pieces
   of the walk) is inferred TOGETHER in one Omega pass with its turn frames, at the run's resolution; a Sim(3) is fitted
   from the zone to the aligned walk on EXACT correspondences (the same keyframe, the same pixel, every keyframe of the
   zone — the zone's seams with the chunks that hold them), and the zone becomes the OWNER of its keyframes: their
   Omega records (depth, confidence, K, pose) and their camera poses are the zone's, so a revisited place is one joint
   reconstruction instead of two copies the graph had to close. The zone's records carry their own chunk id (N + z):
   its confidence is its own pass's, floored per chunk like any other. Unchanged around it: the chunks keep their turn
   frames and their witness pass (the F6 witnesses of a zone's keyframes are carried by their anchor keyframe, as
   always), the scale, the orientation and F0-F6 read the records as before. A zone whose views do not fit one pass
   on the card at the run's resolution is NOT inferred: the chain and its closures hold it, declared."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

FORK = Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"
PASS_SCRIPT = FORK / "stac_frames_pass.py"


def _fork_python() -> str:
    py = os.environ.get("STAC_PYTHON_MAPANYTHING") or "/workspace/miniforge3/envs/mapanything/bin/python"
    if not os.access(py, os.X_OK):
        raise FileNotFoundError(f"the fork's interpreter {py} does not exist (STAC_PYTHON_MAPANYTHING)")
    return py


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_frames_pass(paths: Sequence[str], resolution: int, out_npz: Path, cfg_v: dict, *, sky_cache: Optional[Path],
                    salad: bool, log: Callable[[str], None]) -> Path:
    """One VGGT-Ω inference over ``paths`` (the fork's interpreter, repro's deterministic environment)."""
    import da3_weights
    import repro
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    lst = out_npz.with_suffix(".frames.txt")
    lst.write_text("\n".join(str(p) for p in paths) + "\n")
    cfgp = out_npz.with_suffix(".config.yaml")
    with open(cfgp, "w") as f:
        yaml.dump(cfg_v, f, default_flow_style=False)
    cmd = [_fork_python(), str(PASS_SCRIPT), "--config", str(cfgp), "--frames", str(lst),
           "--resolution", str(int(resolution)), "--out", str(out_npz)]
    if salad:
        cmd.append("--salad")
    if sky_cache is not None:
        cmd += ["--sky-cache", str(sky_cache)]
    env = repro.deterministic_env()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "8"
    env["MKL_CBWR"] = "COMPATIBLE"
    env["HF_HOME"] = da3_weights.HF_HOME
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                         cwd=str(FORK))
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-30:]
            if "[frames-pass]" in line or "Error" in line or "Traceback" in line or "[STAC sky]" in line:
                log(line)
    if p.wait() != 0 or not out_npz.exists():
        raise RuntimeError(f"the frames pass over {len(paths)} frame(s) failed:\n" + "\n".join(tail[-15:]))
    for q in (lst, cfgp):
        q.unlink(missing_ok=True)
    return out_npz


def structure_probe(scan_dir: Path, frames_dir: Path, sel_files: Sequence[str], cfg_v: dict, *, resolution: int,
                    ranges: Sequence[Sequence[int]], sky_cache: Optional[Path], log: Callable[[str], None]) -> Dict:
    """The structure probe and its analysis against the plan's chunks (``ranges``). The pass is reused when the
    keyframes, the resolution, the pass's code, the weights and SALAD's settings and revisit reference are the same
    (its stamp beside the npz); the analysis is cheap and always runs. SALAD's bar is the loop detector's own
    (stac_frames_pass.py)."""
    from reconstruction import walk_structure as WS
    intake = Path(scan_dir) / "intake"
    npz, doc_p, png = intake / "structure_probe.npz", intake / "structure.json", intake / "structure.png"
    stamp_p = intake / "structure_probe.stamp.json"
    sal = (cfg_v.get("Loop") or {}).get("SALAD") or {}
    ref = sal.get("revisit_reference")
    stamp = {"keyframes": hashlib.sha256("\n".join(sel_files).encode()).hexdigest(), "resolution": int(resolution),
             "pass_code": _sha_file(PASS_SCRIPT), "weights": str((cfg_v.get("Weights") or {}).get("VGGTOmega")),
             "salad_threshold": sal.get("similarity_threshold"), "salad_min_gap": sal.get("min_gap"),
             "revisit_reference": _sha_file(Path(ref)) if ref and Path(ref).exists() else None}
    reuse = False
    if npz.exists() and stamp_p.exists():
        try:
            reuse = json.loads(stamp_p.read_text()) == stamp
        except Exception:  # noqa: BLE001
            reuse = False
    if reuse:
        log(f"[structure] the probe's pass reused ({npz.name}, same keyframes, resolution, code and SALAD)")
    else:
        stamp_p.unlink(missing_ok=True)
        paths = [str(Path(frames_dir) / f) for f in sel_files]
        log(f"[structure] the probe: ONE Omega pass over all {len(paths)} keyframes at {resolution}")
        run_frames_pass(paths, resolution, npz, cfg_v, sky_cache=sky_cache, salad=True, log=log)
        tmp = stamp_p.with_name(stamp_p.name + ".tmp")
        tmp.write_text(json.dumps(stamp, indent=1))
        os.replace(tmp, stamp_p)
    res = WS.analyse(npz, ranges=ranges, log=log)
    z = np.load(npz)
    WS.figure(res, z["c2w"], png, Path(scan_dir).parent.parent.name)
    res["pass_stamp"] = stamp
    WS.save(res, doc_p)
    return {k: v for k, v in res.items() if k != "covisibility"}


def zone_pass_frames(zone: Dict, sel_files: Sequence[str], turn_files: Sequence[str]) -> List[str]:
    """The zone's views: the keyframes of its runs and the turn frames strictly inside each run, in frame order."""
    kf = [sel_files[k] for k in zone["keyframes"]]
    spans = [(int(Path(sel_files[a]).stem), int(Path(sel_files[b]).stem)) for a, b in zone["runs"]]
    ctx = [f for f in turn_files if any(lo < int(Path(f).stem) < hi for lo, hi in spans)]
    return sorted(set(kf) | set(ctx), key=lambda f: int(Path(f).stem))


def _world(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    d = depth.astype(np.float64)
    cam = np.stack([(uu - K[0, 2]) * d / K[0, 0], (vv - K[1, 2]) * d / K[1, 1], d], -1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def apply_zones(scan_dir: Path, frames_dir: Path, sel_files: Sequence[str], turn_files: Sequence[str], zones: List[Dict],
                cfg_v: dict, *, resolution: int, capacity: int, n_chunks: int, sky_cache: Optional[Path],
                log: Callable[[str], None]) -> List[Dict]:
    """Each zone inferred together and made the owner of its keyframes (module docstring). ``capacity``: the views
    one Omega pass holds on the card at ``resolution`` (the run's)."""
    from loop_utils.loop_bridges import robust_sim3          # the fork's exact-correspondence Sim(3)
    from reconstruction.walk_structure import _floor_mask
    out = Path(scan_dir) / "output"
    rec_dir = out / "omega_run" / "results_output"
    stems = [int(Path(f).stem) for f in sel_files]
    reports = []
    new_pose = {}
    for zi, zone in enumerate(zones):
        views = zone_pass_frames(zone, sel_files, turn_files)
        if len(views) > int(capacity):
            reports.append({"zone": zi, "runs": zone["runs"], "keyframes": len(zone["keyframes"]),
                            "views": len(views), "inferred": False,
                            "why": f"{len(views)} views > {int(capacity)} one pass holds at {resolution}"})
            log(f"[zones] ⚠ zone {zi + 1}/{len(zones)} (runs {zone['runs']}): {len(views)} views do not fit one "
                f"Omega pass at {resolution} ({int(capacity)}) — NOT inferred together; the chain and its closures "
                f"hold it (declared)")
            continue
        npz = out / "maplong_run" / "zones" / f"zone_{zi:02d}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        run_frames_pass([str(Path(frames_dir) / f) for f in views], resolution, npz, cfg_v, sky_cache=sky_cache,
                        salad=False, log=log)
        z = np.load(npz)
        slot = {int(Path(f).stem): i for i, f in enumerate(views)}
        src, dst, keys = [], [], []
        recs = {}
        for g in zone["keyframes"]:
            n = stems[g]
            rp = rec_dir / f"frame_{n}.npz"
            if not rp.exists():
                raise RuntimeError(f"[zones] zone {zi}: keyframe {n} has no Omega record")
            r = dict(np.load(rp))
            recs[g] = r
            i = slot[n]
            dz, dg = z["depth"][i], r["depth"]
            if dz.shape != dg.shape:
                raise RuntimeError(f"[zones] zone {zi}: keyframe {n} grid {dz.shape} vs the record's {dg.shape} — "
                                   f"not the run's resolution")
            ok = _floor_mask(z["conf"][i], dz) & _floor_mask(r.get("conf", np.ones_like(dg)), dg)
            Xz = _world(dz, z["K"][i], z["c2w"][i])[ok]
            Xg = _world(dg, r["K_omega"], r["pose_c2w"])[ok]
            vv, uu = np.nonzero(ok)
            src.append(Xz); dst.append(Xg)
            keys.append((np.int64(n) << 40) | (vv.astype(np.int64) << 20) | uu.astype(np.int64))
        fit = robust_sim3(np.concatenate(src), np.concatenate(dst), keys=np.concatenate(keys))
        if fit is None:
            raise RuntimeError(f"[zones] zone {zi}: too few exact correspondences for the zone's Sim(3)")
        s, R, t, res, n_used = fit
        cid = int(n_chunks) + zi
        for g, r in recs.items():
            i = slot[stems[g]]
            c2w_z = z["c2w"][i].astype(np.float64)
            c2w = np.eye(4)
            c2w[:3, :3] = R @ c2w_z[:3, :3]
            c2w[:3, 3] = s * (R @ c2w_z[:3, 3]) + t
            rec = {"depth": (s * z["depth"][i]).astype(np.float32), "pose_c2w": c2w,
                   "chunk": np.int64(cid), "frame_global": np.int64(g),
                   "conf": z["conf"][i].astype(np.float32), "K_omega": z["K"][i].astype(np.float64)}
            tmp = rec_dir / f"frame_{stems[g]}.part.npz"
            np.savez_compressed(tmp, **rec)
            os.replace(tmp, rec_dir / f"frame_{stems[g]}.npz")
            new_pose[stems[g]] = c2w
        rep = {"zone": zi, "chunk": cid, "runs": zone["runs"], "keyframes": len(zone["keyframes"]),
               "views": len(views), "inferred": True, "sim3_scale": float(s),
               "fit_median_residual": float(res), "correspondences": int(n_used),
               "resolution": int(resolution)}
        reports.append(rep)
        log(f"[zones] zone {zi + 1}/{len(zones)} (runs {zone['runs']}, {len(views)} views): inferred together, "
            f"Sim(3) to the aligned walk on {n_used:,} exact correspondences — scale {s:.4f}, median residual "
            f"{res:.4f} (the chain's units, before the metric scale); it owns its {len(zone['keyframes'])} "
            f"keyframes (chunk {cid})")
    # the camera poses of record follow the records (both copies the later stages read)
    for base in (out, out / "maplong_run"):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if not (pp.exists() and fp.exists()):
            continue
        nums = [int(float(x)) for x in fp.read_text().split()]
        lines = [l for l in pp.read_text().splitlines() if l.strip()]
        if len(nums) != len(lines):
            raise RuntimeError(f"[zones] {pp}: {len(lines)} poses for {len(nums)} frames")
        lines = [(" ".join(repr(float(v)) for v in new_pose[n].reshape(-1)) if n in new_pose else l)
                 for n, l in zip(nums, lines)]
        tmp = pp.with_name(pp.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, pp)
    doc = {"zones": reports, "chunks_before_zones": int(n_chunks)}
    tmp = out / "zones.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out / "zones.json")
    return reports
