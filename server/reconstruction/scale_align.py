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


_PLY_NP = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
           "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
           "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
           "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}


def _scale_ply_xyz(path: Path, s: float) -> bool:
    """Multiply the x,y,z of every vertex of a PLY by s, in place. Handles
    binary_little/big_endian and ascii. Other properties (rgb, confidence…) untouched.
    Assumes the vertex element has no list properties (true for point clouds). Returns
    False if the layout is unexpected (no x/y/z) so the caller can abort."""
    raw = path.read_bytes()
    hend = raw.find(b"end_header")
    if hend < 0:
        return False
    nl = raw.find(b"\n", hend)
    header = raw[:nl + 1].decode("ascii", "replace")
    body = raw[nl + 1:]

    fmt = None
    props: List[Tuple[str, str]] = []   # (name, ply_type) for the vertex element
    n = 0
    in_vertex = False
    for ln in header.splitlines():
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            in_vertex = (t[1] == "vertex")
            if in_vertex:
                n = int(t[2])
        elif t[0] == "property" and in_vertex:
            props.append((t[-1], t[1]))    # property <type> <name>

    names = [p[0] for p in props]
    if not all(c in names for c in ("x", "y", "z")):
        return False

    if fmt == "ascii":
        xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
        out_lines = []
        for bl in body.decode("ascii", "replace").splitlines()[:n]:
            v = bl.split()
            if len(v) >= len(names):
                v[xi] = f"{float(v[xi]) * s:.8g}"
                v[yi] = f"{float(v[yi]) * s:.8g}"
                v[zi] = f"{float(v[zi]) * s:.8g}"
            out_lines.append(" ".join(v))
        path.write_bytes(raw[:nl + 1] + ("\n".join(out_lines) + "\n").encode("ascii"))
        return True

    endian = "<" if (fmt and "little" in fmt) else ">"
    try:
        dt = np.dtype([(nm, endian + _PLY_NP[ty]) for nm, ty in props])
    except KeyError:
        return False
    nbytes = n * dt.itemsize
    arr = np.frombuffer(body[:nbytes], dtype=dt).copy()
    for c in ("x", "y", "z"):
        arr[c] = (arr[c].astype(np.float64) * s).astype(arr[c].dtype)
    path.write_bytes(raw[:nl + 1] + arr.tobytes() + body[nbytes:])
    return True


def _scale_aligned_depth(output_dir: Path, s: float, dry_run: bool = False) -> None:
    """Scale the per-frame OMEGA depth the TSDF reads (maplong_run/_tmp_results_aligned/
    chunk_*.npy, key 'depth') by s. Without this, scale_align leaves this depth UP-TO-SCALE
    while the poses are metric → the TSDF integrates omega keyframe depth ~s× too small at
    metric poses → collapsed/garbage mesh (the "TSDF espantoso"). Scales 'depth' and
    'world_points'; intrinsics / conf / images are untouched (K does not change with metric
    scale). Safe to call standalone to retro-fix a run whose scale_align predates this."""
    aligned = output_dir / "maplong_run" / "_tmp_results_aligned"
    if not aligned.exists():
        return
    for p in sorted(aligned.glob("chunk_*.npy")):
        if dry_run:
            logger.info(f"[dry-run] would scale aligned depth {p.name} ×{s:.4f}")
            continue
        try:
            d = np.load(str(p), allow_pickle=True).item()
        except Exception as e:
            logger.warning(f"  could not load {p.name}: {e}")
            continue
        if d.get("depth") is not None:
            d["depth"] = (np.asarray(d["depth"], np.float32) * s).astype(np.float32)
        if d.get("world_points") is not None:
            d["world_points"] = (np.asarray(d["world_points"], np.float32) * s).astype(np.float32)
        np.save(str(p), d)
        logger.info(f"  scaled aligned depth {p.name} ×{s:.4f}")


def apply_scale(output_dir: Path, s: float, dry_run: bool = False) -> None:
    """Apply s as a GLOBAL SIMILARITY to the whole reconstruction: every chunk_*.ply
    point XYZ ×s, the per-frame omega depth the TSDF reads (_tmp_results_aligned) ×s,
    AND every camera-centre translation ×s. Rotations and the metric DA3 depth are
    untouched.

    Why both: scaling ONLY the camera centres (the old behaviour) left the chunk clouds
    in up-to-scale units → cameras de-synced from geometry → the BA reprojection / TSDF
    shifted each chunk (the per-chunk "ghosting"), and the 5mm CloudCompy voxel — huge
    relative to the tiny up-to-scale cloud — over-decimated it (~290k pts). Scaling BOTH
    keeps it consistent AND restores density (≈ s³ more voxels filled).

    Clouds are scaled FIRST and a failure RAISES before any pose is touched, so a partial
    failure degrades safely to the consistent up-to-scale state (cameras never de-sync)."""
    # ── 1) the cloud (chunk PLYs CloudCompy will merge) ──
    for ply in sorted(output_dir.glob("chunk_*.ply")):
        if dry_run:
            logger.info(f"[dry-run] would scale {ply.name} XYZ ×{s:.4f}")
            continue
        if not _scale_ply_xyz(ply, s):
            raise RuntimeError(f"could not scale {ply.name} (unexpected PLY layout) — "
                               f"aborting so cameras and cloud never de-sync")
        logger.info(f"  scaled cloud {ply.name} XYZ ×{s:.4f}")

    # ── 2) the per-frame omega depth the TSDF integrates ──
    _scale_aligned_depth(output_dir, s, dry_run=dry_run)

    # ── 3) the camera centres (every camera_poses.txt copy) ──
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
    # Idempotency guard: a metric scale is a one-shot global similarity. If a resume
    # re-enters here after it already ran, scaling again would DOUBLE it (cloud and
    # poses ×s²). The marker records it was applied → skip.
    marker = output_dir / ".metric_scale_applied"
    if marker.exists() and not dry_run:
        logger.info(f"metric scale already applied ({marker.read_text().strip()}) — skipping")
        return None
    s = estimate_scale(output_dir)
    if s is None:
        return None
    apply_scale(output_dir, s, dry_run=dry_run)
    if not dry_run:
        marker.write_text(f"s={s:.6f}\n")
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
