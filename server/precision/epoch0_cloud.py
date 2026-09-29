"""The Omega cloud of EPOCH 0, built AFTER the precision core for comparison only.

USER 2026-09-29: "conservamos [la época 0] mientras validamos, deben poder
seleccionarse desde la UI" — and, the same day, no working cloud is merged or
filtered BEFORE F7 (the core needs none). So the Omega cloud is built at the END,
from the raw chunk PLYs Omega left (or, for a session whose chunks are gone, from
Omega's per-keyframe records), through the cleaning recipe the CloudCompy stage
always ran (reconstruction/gpu_cloud_clean: voxel + SOR at the postprocessing
parameters; the witness filter needs the epoch's own poses and is left out), the
scene consolidation, and its Potree octree — into ``_epoch_0/`` with the files
REGISTERED in its manifest, so ``correction.apply.select_epoch(…, 0)`` restores
them and the fused cloud goes to the stored dir of the live epoch.

Records: ``omega_run/results_output/frame_<f>.npz`` = {depth, conf, pose_c2w,
K_omega, frame_global, chunk}. Their poses are ``camera_poses.txt.prescale`` (Omega
after the metric lock, before scale_align and the Y-up orientation); epoch 0 is that
geometry under ONE similarity (scale_align × orient) — measured on pccr 2026-08-31:
step ratio 0.9707 everywhere, rigid after the scale. The similarity is solved from the
prescale poses to the epoch-0 poses (Umeyama) and applied to the unprojected points.
The confidence gate is the fork's (`_stac_conf_threshold`): the bottom
`conf_percentile` % of the chunk's VALID points (conf > 1e-5) out, and the pipeline's
one floor `conf_min_norm` (min-max fraction of the chunk's valid confidences) on top.

CLI: ``python -m precision.epoch0_cloud --session <dir> [--from-omega-npz]``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_TAG = "[epoch0-cloud]"
TMP_DIRNAME = "_tx_epoch0"
EPOCH0_DIRNAME = "_epoch_0"
SKY_CONF = 1e-5            # the fork's sky mask: conf ≤ this is 'no geometry', not a low score


class Epoch0Error(RuntimeError):
    """The comparison cloud could not be built — with the exact reason."""


# ── geometry helpers ────────────────────────────────────────────────────

def similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) with dst ≈ s·R·src + t (Umeyama, least squares over the rows)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise Epoch0Error(f"similarity needs two equal (N≥3,3) point sets, got {src.shape} / {dst.shape}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def conf_threshold(confs: np.ndarray, conf_percentile: Optional[float], conf_min_norm: float) -> float:
    """The fork's `_stac_conf_threshold`: percentile of the VALID confidences, and the
    min-max floor on top — the stricter wins. -1 when nothing is valid."""
    c = np.asarray(confs, np.float64).reshape(-1)
    valid = c[c > SKY_CONF]
    if valid.size == 0:
        return -1.0
    thr = float(np.percentile(valid, float(conf_percentile))) if conf_percentile is not None else -1.0
    if conf_min_norm and conf_min_norm > 0:
        lo, hi = float(valid.min()), float(valid.max())
        if hi > lo:
            thr = max(thr, lo + float(conf_min_norm) * (hi - lo))
    return thr


# ── the records → chunk PLYs the cleaner reads ───────────────────────────

def _read_prescale_and_epoch0(out: Path) -> Tuple[List[int], np.ndarray, np.ndarray]:
    frames = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
    pre = out / "camera_poses.txt.prescale"
    e0 = out / EPOCH0_DIRNAME / "camera_poses.txt"
    if not e0.exists():
        e0 = out / "camera_poses.txt"        # nothing published yet: the live poses ARE epoch 0
    for p in (pre, e0):
        if not p.exists():
            raise Epoch0Error(f"{p} is missing — the similarity from Omega's records to epoch 0 "
                              f"needs both pose files")
    P_pre = np.loadtxt(pre).reshape(-1, 4, 4)
    P_e0 = np.loadtxt(e0).reshape(-1, 4, 4)
    if len(P_pre) != len(frames) or len(P_e0) != len(frames):
        raise Epoch0Error(f"pose rows {len(P_pre)} / {len(P_e0)} vs {len(frames)} keyframes")
    return frames, P_pre, P_e0


def _write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment STAC epoch-0 rebuild from Omega records (precision/epoch0_cloud.py)\n"
              f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def chunks_from_omega_records(session_dir: Path, tmp: Path, *, conf_percentile: Optional[float],
                              conf_min_norm: float, log: Callable = print) -> Dict[str, object]:
    """Rebuild Omega's raw cloud from the per-keyframe records into ``tmp`` as
    chunk_<k>.ply + chunk_<k>_origins.npz (the layout gpu_cloud_clean reads), in
    epoch-0 coordinates. Returns counts and the similarity used."""
    import cv2
    session_dir = Path(session_dir)
    out = session_dir / "output"
    rec = out / "omega_run" / "results_output"
    frames, P_pre, P_e0 = _read_prescale_and_epoch0(out)
    s, R, t = similarity_umeyama(P_pre[:, :3, 3], P_e0[:, :3, 3])
    resid = np.linalg.norm((s * (P_pre[:, :3, 3] @ R.T) + t) - P_e0[:, :3, 3], axis=1)
    log(f"{LOG_TAG} similarity prescale → epoch 0: scale {s:.5f}, camera residual median "
        f"{np.median(resid) * 1000:.2f} mm max {resid.max() * 1000:.2f} mm")
    by_chunk: Dict[int, List[int]] = {}
    for f in frames:
        p = rec / f"frame_{f}.npz"
        if not p.exists():
            raise Epoch0Error(f"{p} is missing — Omega's record of keyframe {f}")
        with np.load(p) as z:
            by_chunk.setdefault(int(z["chunk"]), []).append(f)
    tmp.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for k, fs in sorted(by_chunk.items()):
        confs = []
        for f in fs:
            with np.load(rec / f"frame_{f}.npz") as z:
                confs.append(np.asarray(z["conf"], np.float32).reshape(-1))
        thr = conf_threshold(np.concatenate(confs), conf_percentile, conf_min_norm)
        xyz_l, rgb_l, fg_l, pr_l, pc_l, cf_l = [], [], [], [], [], []
        for f in fs:
            with np.load(rec / f"frame_{f}.npz") as z:
                d, c, K, c2w = (np.asarray(z["depth"], np.float64), np.asarray(z["conf"], np.float32),
                                np.asarray(z["K_omega"], np.float64), np.asarray(z["pose_c2w"], np.float64))
            H, W = d.shape
            keep = np.isfinite(d) & (d > 0) & (c > SKY_CONF) & (c >= thr)
            rr, cc = np.nonzero(keep)
            if not rr.size:
                continue
            zc = d[rr, cc]
            Xc = np.stack([(cc - K[0, 2]) / K[0, 0] * zc, (rr - K[1, 2]) / K[1, 1] * zc, zc], 1)
            Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]
            Xe = s * (Xw @ R.T) + t
            img = cv2.imread(str(session_dir / "frames" / f"{f:06d}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                raise Epoch0Error(f"frame {f:06d}.jpg unreadable")
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            rgb = img[rr, cc][:, ::-1]
            xyz_l.append(Xe.astype(np.float32)); rgb_l.append(rgb.astype(np.uint8))
            fg_l.append(np.full(rr.size, f, np.int32)); pr_l.append(rr.astype(np.int16))
            pc_l.append(cc.astype(np.int16)); cf_l.append(c[rr, cc].astype(np.float32))
        if not xyz_l:
            continue
        xyz = np.concatenate(xyz_l); rgb = np.concatenate(rgb_l)
        _write_ply_xyzrgb(tmp / f"chunk_{k:03d}.ply", xyz, rgb)
        np.savez(tmp / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(fg_l),
                 pixel_row=np.concatenate(pr_l), pixel_col=np.concatenate(pc_l),
                 confidence=np.concatenate(cf_l))
        n_total += len(xyz)
        log(f"{LOG_TAG} chunk {k}: {len(fs)} keyframes, conf threshold {thr:.4f} → {len(xyz):,} pts")
    if n_total == 0:
        raise Epoch0Error("no point passed the confidence gate — nothing to rebuild")
    return {"n_points": n_total, "n_chunks": len(by_chunk), "similarity": {"scale": s},
            "camera_residual_max_m": float(resid.max())}


# ── the recipe and the registration ──────────────────────────────────────

def clean_cmd(config: dict, input_dir: Path, output_ply: Path) -> List[str]:
    """The cloud stage's cleaner command (workers/cloudcompy_worker.py), without the
    witness filter: it needs the epoch's own per-frame depth and poses."""
    postproc = config.get("postprocessing", {}) or {}
    cmd = [sys.executable, "-m", "reconstruction.gpu_cloud_clean",
           "--input-dir", str(input_dir), "--output", str(output_ply),
           "--voxel-size", str(postproc.get("voxel_size", 0.001)),
           "--sor-knn", str(postproc.get("sor_knn", 6)),
           "--sor-sigma", str(postproc.get("sor_sigma", 1.0)),
           "--noise-radius", str(postproc.get("noise_radius", 0.01)),
           "--noise-sigma", str(postproc.get("noise_sigma", 1.0)),
           "--conf-min-norm", str(postproc.get("conf_min_norm", 0.0))]
    if int(postproc.get("max_points", 0) or 0) > 0:
        cmd += ["--max-points", str(postproc["max_points"])]
    for flag in ("skip_duplicates", "skip_sor", "skip_noise", "skip_normals"):
        if postproc.get(flag, False):
            cmd.append(f"--{flag.replace('_', '-')}")
    return cmd


def register_in_manifest(epoch_dir: Path, rels: Sequence[str]) -> Path:
    """Add ``rels`` to the stored epoch's manifest artifacts (existed_before: the swap
    back to this epoch restores them and moves the live ones to the current epoch's
    stored dir)."""
    mp = epoch_dir / "_manifest.json"
    if not mp.exists():
        raise Epoch0Error(f"{mp} is missing — {epoch_dir.name} is not a stored epoch")
    m = json.loads(mp.read_text())
    have = {a["rel"] for a in m.get("artifacts", [])}
    for rel in rels:
        if rel not in have:
            m.setdefault("artifacts", []).append({"rel": rel, "existed_before": True})
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, mp)
    return mp


def build_epoch0_cloud(session_dir: Path, config: dict, *, input_dir: Optional[Path] = None,
                       from_records: bool = False, log: Callable = print,
                       progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, object]:
    """Build epoch 0's cloud (from the chunk PLYs in ``input_dir`` — output/ by default —
    or from Omega's records with ``from_records``), clean it with the recipe,
    consolidate, build its octree, move it into `_epoch_0/` and register it."""
    session_dir = Path(session_dir)
    out = session_dir / "output"
    e0 = out / EPOCH0_DIRNAME
    if not (e0 / "_manifest.json").exists():
        raise Epoch0Error(f"{e0} is not a stored epoch (no _manifest.json) — the core must have "
                          f"published at least one epoch")
    tmp = out / TMP_DIRNAME
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    t0 = time.time()
    rep: Dict[str, object] = {"from_records": bool(from_records)}
    try:
        if from_records:
            simple = (config.get("reconstruction") or {}).get("simple") or {}
            rep["records"] = chunks_from_omega_records(
                session_dir, tmp, conf_percentile=simple.get("conf_percentile"),
                conf_min_norm=float(simple.get("conf_min_norm", 0.0) or 0.0), log=log)
            src = tmp
        else:
            src = Path(input_dir) if input_dir else out
            if not list(src.glob("chunk_*.ply")):
                raise Epoch0Error(f"no chunk_*.ply in {src} — Omega's raw cloud is gone (use "
                                  f"--from-omega-npz to rebuild it from the records)")
        if progress:
            progress(20.0, "epoch 0: cleaning Omega's cloud with the postprocessing recipe")
        cleaned = tmp / "cleaned_cloud.ply"
        cmd = clean_cmd(config, src, cleaned)
        server_dir = Path(__file__).resolve().parent.parent
        log(f"{LOG_TAG} {' '.join(cmd[1:6])} …")
        proc = subprocess.Popen(cmd, cwd=str(server_dir), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if line and ("✅" in line or "Step" in line or "❌" in line or "pts" in line):
                log(f"{LOG_TAG} {line}")
        if proc.wait() != 0 or not cleaned.exists():
            raise Epoch0Error(f"the cleaner failed (exit {proc.returncode}) — epoch 0 not built")
        sc = (config.get("postprocessing") or {}).get("scene_consolidate") or {}
        if sc.get("enabled", True):
            if progress:
                progress(60.0, "epoch 0: consolidating (normal-aware MLS)")
            from reconstruction.surface_fit.consolidate import scene_consolidate
            from reconstruction.loops.config import load_loops_config
            stats = scene_consolidate(
                tmp, radius_m=sc.get("radius_m"), min_radius_m=float(sc.get("min_radius_m", 0.02)),
                max_radius_m=float(sc.get("max_radius_m", 0.06)), iterations=int(sc.get("iterations", 2)),
                normal_gate=float(sc.get("normal_gate", 0.25)),
                excluded_statuses=load_loops_config(config).witness.mls_excluded_statuses)
            rep["consolidate"] = {k: stats[k] for k in ("n_points", "radius_m", "mean_move_mm")
                                  if k in stats}
        if progress:
            progress(80.0, "epoch 0: Potree octree")
        from potree_converter import convert_ply_to_potree
        ok = convert_ply_to_potree(session_dir, force=True, ply_override=cleaned,
                                   potree_dir_override=tmp / "potree")
        if not ok or not (tmp / "potree" / "metadata.json").exists():
            raise Epoch0Error("the Potree build of epoch 0 failed — nothing registered")
        rels = ["cleaned_cloud.ply"]
        for rel in ("cleaned_cloud_raw.ply",):
            if (tmp / rel).exists():
                rels.append(rel)
        rels.append("potree")
        for rel in rels:
            dst = e0 / rel
            if dst.is_dir():
                shutil.rmtree(dst)
            elif dst.exists():
                dst.unlink()
            shutil.move(str(tmp / rel), str(dst))
        register_in_manifest(e0, rels)
        rep.update({"epoch_dir": str(e0), "artifacts": rels,
                    "n_points": int(json.loads((e0 / "potree" / "metadata.json").read_text()).get("points", 0)),
                    "seconds": round(time.time() - t0, 1)})
        log(f"{LOG_TAG} epoch 0 = {rep['n_points']:,} pts in {e0} ({rep['seconds']} s), "
            f"registered: {rels}")
        return rep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m precision.epoch0_cloud",
                                 description="Build epoch 0's Omega cloud for comparison.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--from-omega-npz", action="store_true",
                    help="rebuild Omega's raw cloud from omega_run/results_output (chunks gone)")
    args = ap.parse_args(argv)
    from config import cfg
    build_epoch0_cloud(Path(args.session), cfg, from_records=args.from_omega_npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
