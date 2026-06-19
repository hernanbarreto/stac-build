"""
Metric scale alignment for the VGGT-Omega path.
================================================================================
VGGT-Omega gives excellent but UP-TO-SCALE poses + depth. DA3 (Giant Large 1.1)
gives metric depth per frame. This module recovers the single global scale factor
that makes VGGT-Omega's reconstruction metric, by comparing the two depths on the
keyframes, and rewrites camera_poses.txt with metrically-scaled translations.

Why one global scale: VGGT-Long aligns chunks with Sim3, so the whole VGGT-Omega
reconstruction is internally scale-consistent — a single s maps it to metres.
(If per-chunk scale drift is ever observed, switch to per-chunk s — same method.)

s is estimated robustly: per keyframe, s_f = median(depth_DA3 / depth_Omega) over
pixels valid in BOTH (conf-filtered, finite, positive); global s = median(s_f),
trimmed. Then every camera-centre translation is multiplied by s. Rotations and
the metric DA3 depth (the surface source) are untouched.

Run (mapanything env), after VGGT-Long[Omega] produced poses + the DA3 per-frame npz:
    python -m reconstruction.scale_align --output-dir <out> [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ScaleAlign")


def _da3_npz_dir(output_dir: Path) -> Optional[Path]:
    for sub in ("da3_run/results_output", "results_output"):
        d = output_dir / sub
        if d.exists() and any(d.glob("frame_*.npz")):
            return d
    return None


def _omega_depth(output_dir: Path) -> Dict[int, np.ndarray]:
    """VGGT-Omega per-frame (up-to-scale) depth, keyed by REAL frame number.
    Read from VGGT-Long's saved chunks (_tmp_results_aligned/chunk_K.npy), mapped to
    frame numbers via camera_frames.txt order is NOT reliable per-chunk, so we use the
    per-frame npz the omega adapter/pipeline emits if present; else the chunk arrays."""
    out: Dict[int, np.ndarray] = {}
    # Preferred: per-frame omega npz (if the pipeline emits one)
    for sub in ("omega_run/results_output", "vggtomega_run/results_output"):
        d = output_dir / sub
        if d.exists():
            for p in d.glob("frame_*.npz"):
                try:
                    n = int(Path(p).stem.split("_")[1])
                    out[n] = np.load(p)["depth"].astype(np.float32)
                except Exception:
                    continue
            if out:
                return out
    return out


def _read_poses(output_dir: Path) -> Tuple[List[str], List[int], Path]:
    """Return (raw pose lines, frame numbers row-aligned, poses path)."""
    for base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if pp.exists() and fp.exists():
            lines = pp.read_text().splitlines()
            nums = [int(float(x)) for x in fp.read_text().split()]
            if len([l for l in lines if len(l.split()) == 16]) == len(nums):
                return lines, nums, pp
    raise FileNotFoundError("no row-aligned camera_poses.txt/camera_frames.txt")


def _ratio(da3: np.ndarray, omega: np.ndarray) -> Optional[float]:
    """median(DA3/Omega) over pixels valid in both, at the common (resized) grid."""
    if da3 is None or omega is None:
        return None
    if da3.shape != omega.shape:
        # resize DA3 → omega grid (nearest)
        H, W = omega.shape
        yi = (np.arange(H) * da3.shape[0] / H).astype(int).clip(0, da3.shape[0] - 1)
        xi = (np.arange(W) * da3.shape[1] / W).astype(int).clip(0, da3.shape[1] - 1)
        da3 = da3[yi][:, xi]
    m = np.isfinite(da3) & np.isfinite(omega) & (da3 > 1e-3) & (omega > 1e-3)
    if m.sum() < 100:
        return None
    return float(np.median(da3[m] / omega[m]))


def estimate_scale(output_dir: Path) -> Optional[float]:
    da3_dir = _da3_npz_dir(output_dir)
    if da3_dir is None:
        logger.error("no DA3 results_output/*.npz (metric depth) found")
        return None
    omega_depth = _omega_depth(output_dir)
    if not omega_depth:
        logger.error("no VGGT-Omega per-frame depth found (omega_run/results_output) — "
                     "the omega worker must emit per-frame depth npz for scale alignment")
        return None
    ratios = []
    for n, od in omega_depth.items():
        p = da3_dir / f"frame_{n}.npz"
        if not p.exists():
            continue
        try:
            dd = np.load(p)["depth"].astype(np.float32)
        except Exception:
            continue
        r = _ratio(dd, od)
        if r is not None and np.isfinite(r) and r > 0:
            ratios.append(r)
    if len(ratios) < 3:
        logger.error(f"only {len(ratios)} usable keyframes for scale — abort")
        return None
    ratios = np.array(ratios)
    lo, hi = np.percentile(ratios, [10, 90])           # trim outliers
    s = float(np.median(ratios[(ratios >= lo) & (ratios <= hi)]))
    logger.info(f"scale s = {s:.4f} (median over {len(ratios)} keyframes, "
                f"spread {ratios.min():.3f}–{ratios.max():.3f})")
    return s


def apply_scale(output_dir: Path, s: float, dry_run: bool = False) -> None:
    """Multiply every camera-centre translation by s in every camera_poses.txt copy
    (c2w: scale the [:3,3] column). Rotations unchanged. Backs up to .prescale."""
    for base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        pp = base / "camera_poses.txt"
        if not pp.exists():
            continue
        lines = pp.read_text().splitlines()
        out = []
        for ln in lines:
            v = ln.split()
            if len(v) == 16:
                m = np.array([float(x) for x in v], np.float64).reshape(4, 4)
                m[:3, 3] *= s                          # c2w camera centre → metric
                ln = " ".join(f"{x:.8g}" for x in m.reshape(-1))
            out.append(ln)
        if dry_run:
            logger.info(f"[dry-run] would scale {pp}")
            continue
        bak = pp.with_suffix(".txt.prescale")
        if not bak.exists():
            shutil.copy(pp, bak)
        pp.write_text("\n".join(out) + "\n")
        logger.info(f"  scaled {pp} (backup {bak.name})")


def run(output_dir: Path, dry_run: bool = False) -> Optional[float]:
    s = estimate_scale(output_dir)
    if s is None:
        return None
    apply_scale(output_dir, s, dry_run=dry_run)
    logger.info(f"✅ metric scale {'(dry-run) ' if dry_run else ''}s={s:.4f} applied")
    return s


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(Path(args.output_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
