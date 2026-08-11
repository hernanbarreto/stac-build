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

────────────────────────────────────────────────────────────────────────────────
v2 (precision task, Phase A). The single-median estimator above stays the
production baseline (mode "global_median", bit-identical). On top of it:
  - configurable estimator modes "affine_robust" / "depth_dependent"
    (reconstruction/scale_model.py — evidence-gated, auto-degrading);
  - an optional VIO scale source (reconstruction/vio_scale.py): when the session
    carries a VIO trajectory (docs/VIO_FORMAT.md), VIO SETS the scale and DA3
    becomes the cross-check; the VIO↔DA3 agreement is always reported;
  - a persisted per-session diagnostic (output/scale_diagnostics.json): per-anchor
    ratios, dispersion, s along the walk, residual vs depth, model selection
    verdicts, VIO agreement, and an aggregate scale_confidence health metric.
The applied correction remains ONE global similarity s in every mode (omega's
only gauge freedom — see scale_model.py docstring); fail-hard rules unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ScaleAlign")

DIAGNOSTICS_NAME = "scale_diagnostics.json"


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


def _ratio(da3: np.ndarray, omega: np.ndarray, conf: Optional[np.ndarray] = None,
           near_frac: float = 0.25, conf_top_frac: float = 0.0,
           restrict: Optional[np.ndarray] = None) -> Optional[float]:
    """median(DA3/Omega) over the NEAR-BAND pixels (closest `near_frac` by depth) valid in
    both, at the common (resized) grid — optionally intersected with the top
    `conf_top_frac` most-confident DA3 pixels and/or a boolean `restrict` mask
    (e.g. eroded structural-instance masks; any resolution, resized to the omega grid).

    Why the near band and not all pixels: monocular metric depth (DA3) compresses the far
    range → distant pixels systematically bias the DA3/omega ratio LOW, under-scaling the
    whole reconstruction. The rails — our metric target for BIM — sit in the near field,
    where DA3 is most accurate. Restricting the ratio to the closest 25% of pixels removes
    that far-range bias. Validated on test3 (230 keyframes): near-25% lifts s by +8.25% vs
    all-pixels, which would carry test2's track gauge from 1328 mm (−7.5%) to ~1438 mm
    (+0.2% vs the 1435 mm standard gauge) — and test2 independently needs +8.06%. Agnostic:
    no known pattern / no manual measurement, just "weight the near field where DA3 is good".

    The confidence gate stacks the same idea on the model's own uncertainty: only the
    top fraction of DA3-confidence pixels vote. Fallback ladder when a filter starves
    (<50 px): near∩conf → conf-only → near-only → all valid.
    """
    if da3 is None or omega is None:
        return None
    H, W = omega.shape

    def _fit(a):
        # resize any per-pixel companion array → omega grid (nearest)
        if a is None or a.shape[:2] == (H, W):
            return a
        yi = (np.arange(H) * a.shape[0] / H).astype(int).clip(0, a.shape[0] - 1)
        xi = (np.arange(W) * a.shape[1] / W).astype(int).clip(0, a.shape[1] - 1)
        return a[yi][:, xi]

    da3 = _fit(da3)
    conf = _fit(conf)
    restrict = _fit(restrict)
    m = np.isfinite(da3) & np.isfinite(omega) & (da3 > 1e-3) & (omega > 1e-3)
    if restrict is not None:
        m &= restrict.astype(bool)
    if m.sum() < 100:
        return None
    rr = da3[m] / omega[m]
    od = omega[m]                       # depth proxy; per-chunk Sim3 preserves intra-frame order
    near = od <= np.percentile(od, float(near_frac) * 100.0)
    if conf is not None and conf_top_frac and conf_top_frac > 0:
        cf = conf[m].astype(np.float32)
        top = cf >= np.percentile(cf, 100.0 * (1.0 - float(conf_top_frac)))
        both = near & top
        if both.sum() >= 50:
            return float(np.median(rr[both]))
        if top.sum() >= 50:             # near band too thin → confidence-only
            return float(np.median(rr[top]))
    if near.sum() >= 50:
        return float(np.median(rr[near]))
    return float(np.median(rr))         # too few near pixels → fall back to all valid


def estimate_scale(output_dir: Path, log=None,
                   near_frac: float = 0.25, conf_top_frac: float = 0.0) -> Optional[float]:
    # `log` (if given) mirrors every message to the caller's sink (e.g. the worker pipe →
    # server log file AND the UI) so WHAT HAPPENED is always visible, not just in this
    # module's logger (which the worker does not forward).
    _log = log if log is not None else (lambda m: logger.info(m))
    da3_dir = _da3_npz_dir(output_dir)
    if da3_dir is None:
        _log("FAIL: no DA3 results_output/*.npz (metric depth) found → cannot estimate scale")
        return None
    omega_depth = _omega_depth(output_dir)
    if not omega_depth:
        _log("FAIL: no VGGT-Omega per-frame depth (omega_run/results_output) found "
             "→ cannot estimate scale")
        return None
    import glob as _g
    n_da3 = len(_g.glob(str(da3_dir / "frame_*.npz")))
    matched = 0
    bad_ratio = 0
    ratios = []
    for n, od in omega_depth.items():
        p = da3_dir / f"frame_{n}.npz"
        if not p.exists():
            continue
        matched += 1
        try:
            npz = np.load(p)
            dd = npz["depth"].astype(np.float32)
            cc = npz["conf"].astype(np.float32) if "conf" in npz.files else None
        except Exception:
            continue
        r = _ratio(dd, od, conf=cc, near_frac=near_frac, conf_top_frac=conf_top_frac)
        if r is not None and np.isfinite(r) and r > 0:
            ratios.append(r)
        else:
            bad_ratio += 1
    _log(f"inputs: DA3 frames={n_da3}, omega frames={len(omega_depth)}, "
         f"matched frame#={matched}, usable ratios={len(ratios)}, rejected ratios={bad_ratio}")
    if len(ratios) < 3:
        _log(f"FAIL: only {len(ratios)} usable keyframes (need ≥3) → abort, poses stay up-to-scale "
             f"(matched {matched}/{len(omega_depth)} omega frames to DA3)")
        return None
    ratios = np.array(ratios)
    lo, hi = np.percentile(ratios, [10, 90])           # trim outliers
    s = float(np.median(ratios[(ratios >= lo) & (ratios <= hi)]))
    _filters = f"near-{near_frac:.0%} band" + (
        f" ∩ conf top-{conf_top_frac:.0%}" if conf_top_frac and conf_top_frac > 0 else "")
    _log(f"OK: metric scale s={s:.4f} ({_filters} ratio, median over {len(ratios)} "
         f"keyframes, spread {ratios.min():.3f}–{ratios.max():.3f})")
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


def _kf_centres_and_times(output_dir: Path, session_dir: Path):
    """Keyframe camera centres (up-to-scale, pre-apply) + timestamps (s) for the
    VIO segment ratio. Timestamps = frame_number / video fps. Raises with the
    exact reason when fps cannot be established (VIO needs a time base)."""
    from reconstruction.vio_scale import video_fps
    lines, nums, _ = _read_poses(output_dir)
    mats = [np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
            for ln in lines if len(ln.split()) == 16]
    centres = np.stack([m[:3, 3] for m in mats])
    fps = video_fps(session_dir)
    if not fps:
        raise RuntimeError("VIO present but the video fps could not be read "
                           "(no source_video.* or unreadable) — VIO timestamps cannot "
                           "be matched to frames. Fix or remove the VIO file.")
    t = np.asarray(nums, np.float64) / float(fps)
    return t, centres


def estimate_v2(output_dir: Path, log=None, near_frac: float = 0.25,
                conf_top_frac: float = 0.0, cfg: Optional[dict] = None,
                session_dir: Optional[Path] = None) -> Tuple[Optional[float], dict]:
    """Phase-A estimator + diagnostics. Returns (s_applied, diagnostics dict).

    Scale source priority: VIO (when detected and enabled) SETS s; the DA3 model
    is always estimated as well — as the applied source when no VIO, as the
    cross-check when VIO is present. s_applied None ⇒ genuine failure (fatal
    upstream). VIO present-but-unusable RAISES (fail-hard, no silent fallback)."""
    from reconstruction import scale_model as sm
    _log = log if log is not None else (lambda m: logger.info(m))
    cfg = cfg or {}
    mode = str(cfg.get("mode", "global_median"))
    session_dir = Path(session_dir) if session_dir else output_dir.parent

    diag: dict = {"version": 1, "generated_by": "scale_align_v2",
                  "mode_requested": mode, "near_frac": near_frac,
                  "conf_top_frac": conf_top_frac}

    # ── DA3 model (baseline s always computed via the untouched production path) ──
    s_baseline = estimate_scale(output_dir, log=log, near_frac=near_frac,
                                conf_top_frac=conf_top_frac)
    frames = sm.collect_frame_samples(output_dir, near_frac=near_frac,
                                      conf_top_frac=conf_top_frac)
    s_da3: Optional[float] = s_baseline
    sel = None
    if mode != "global_median" and frames:
        sel = sm.select_model(frames, mode)
        s_model = sm.applied_gain(sel["model"], frames)
        if s_model is not None and np.isfinite(s_model) and s_model > 0:
            s_da3 = float(s_model)
        _log(f"mode {mode} → used {sel['mode_used']}"
             + (f" (DEGRADED: {sel['degrade_reason']})" if sel["degraded"] else "")
             + (f" — s={s_da3:.4f} (baseline global_median {s_baseline:.4f})"
                if s_da3 is not None and s_baseline is not None else ""))
        diag.update({"mode_used": sel["mode_used"], "degraded": sel["degraded"],
                     "degrade_reason": sel["degrade_reason"],
                     "model": {"kind": sel["model"]["kind"], "params": sel["model"]["params"]}
                     if sel["model"] else None,
                     "cv_loss": sel["cv"], "bic": sel["bic"]})
    else:
        diag.update({"mode_used": "global_median", "degraded": False,
                     "degrade_reason": None, "model": None,
                     "cv_loss": {}, "bic": {}})
    diag["s_da3_global_median"] = s_baseline
    diag["s_da3"] = s_da3

    # ── VIO source (optional; VIO commands when present) ──
    vio_info = None
    s_vio: Optional[float] = None
    if bool(cfg.get("vio", True)):
        from ingestors.vio_detector import detect_vio_data
        det = detect_vio_data(session_dir)
        if det["has_vio"]:
            from reconstruction.vio_scale import (load_vio_trajectory,
                                                  estimate_vio_scale)
            _log(f"VIO trajectory detected: {det['vio_path'].name} — VIO sets the "
                 f"scale, DA3 becomes the cross-check")
            vt, vp, _fps_hint = load_vio_trajectory(det["vio_path"])
            kf_t, kf_c = _kf_centres_and_times(output_dir, session_dir)
            vio_info = estimate_vio_scale(
                vt, vp, kf_t, kf_c,
                segment_s=float(cfg.get("vio_segment_s", 5.0)),
                min_segments=int(cfg.get("vio_min_segments", 8)),
                min_coverage=float(cfg.get("vio_min_coverage", 0.5)))
            vio_info["file"] = det["vio_path"].name
            vio_info.pop("segments", None)      # keep the JSON compact; ratios summarized
            s_vio = vio_info["s_vio"]
            _log(f"VIO scale s={s_vio:.4f} (median over {vio_info['n_segments']} segments, "
                 f"MAD {100 * (vio_info['mad_rel'] or 0):.1f}%, coverage "
                 f"{vio_info['coverage_frac']:.0%})")

    # ── source priority + agreement ──
    if s_vio is not None:
        s_applied, source = s_vio, "vio"
        if s_da3 is not None:
            agreement = 100.0 * (s_da3 / s_vio - 1.0)
            vio_info["agreement_pct"] = agreement
            _log(f"VIO↔DA3 agreement: DA3 is {agreement:+.2f}% vs VIO "
                 f"(DA3 s={s_da3:.4f}, VIO s={s_vio:.4f}) — DA3 scale-bias diagnostic")
        else:
            vio_info["agreement_pct"] = None
            _log("VIO↔DA3 agreement: DA3 estimate unavailable — no cross-check")
    else:
        s_applied, source = s_da3, "da3"
    diag["scale_source"] = source
    diag["vio"] = vio_info
    diag["s_applied"] = s_applied

    # ── per-anchor diagnostics + confidence ──
    if frames:
        ratios = np.array([f["s_f"] for f in frames])
        med = float(np.median(ratios))
        mad_rel = float(np.median(np.abs(ratios - med)) / med) if med else None
        diag["anchors"] = {
            "count": len(frames),
            "mad_rel": mad_rel,
            "spread": [float(ratios.min()), float(ratios.max())],
            "frames": [{"num": f["num"], "s_f": f["s_f"], "n_px": f["n_px"],
                        "z_median_omega": f["z_median"], "d_median_m": f["d_median"]}
                       for f in frames],
            "s_over_walk": [{"num": f["num"], "s_f": f["s_f"]} for f in frames],
        }
        diag["jackknife"] = sm.jackknife_s(frames)
        if s_applied is not None and source == "da3":
            diag["residual_vs_depth"] = sm.residual_depth_profile(frames, s_applied)
        elif s_da3 is not None:
            diag["residual_vs_depth"] = sm.residual_depth_profile(frames, s_da3)
        else:
            diag["residual_vs_depth"] = []
        heldout = (sel or {}).get("cv", {}).get("scale_only") if sel else \
            sm._cv_frame_loss(frames, "scale_only")
        diag["heldout_cv_loss_scale_only"] = heldout
        conf, terms = sm.scale_confidence(
            mad_rel, len(frames), heldout,
            (vio_info or {}).get("agreement_pct") if vio_info else None)
    else:
        diag["anchors"] = {"count": 0, "mad_rel": None, "spread": None,
                           "frames": [], "s_over_walk": []}
        diag["jackknife"] = None
        diag["residual_vs_depth"] = []
        diag["heldout_cv_loss_scale_only"] = None
        conf, terms = sm.scale_confidence(
            None, 0, None, (vio_info or {}).get("agreement_pct") if vio_info else None)
    diag["scale_confidence"] = conf
    diag["confidence_terms"] = terms
    if s_applied is not None:
        _log(f"scale diagnostics: source={source}, s={s_applied:.4f}, "
             f"anchors={diag['anchors']['count']}, "
             f"MAD {100 * (diag['anchors']['mad_rel'] or 0):.1f}%, "
             f"confidence {conf:.2f}")
    return s_applied, diag


def run(output_dir: Path, dry_run: bool = False, log=None,
        near_frac: float = 0.25, conf_top_frac: float = 0.0,
        cfg: Optional[dict] = None, session_dir: Optional[Path] = None
        ) -> Optional[float]:
    """Estimate + apply the metric scale. Returns the scale s (float) on success OR when
    it was ALREADY applied (marker present → output is already metric). Returns None ONLY
    when the scale could not be estimated (genuine failure) — the caller MUST treat None
    as fatal (a non-metric reconstruction is useless for BIM comparison)."""
    _log = log if log is not None else (lambda m: logger.info(m))
    # Idempotency guard: a metric scale is a one-shot global similarity. If a resume
    # re-enters here after it already ran, scaling again would DOUBLE it (cloud + poses
    # ×s²). The marker records it was applied. (Replace clears the marker → re-scales.)
    marker = output_dir / ".metric_scale_applied"
    if marker.exists() and not dry_run:
        txt = marker.read_text().strip()
        try:
            s_prev = float(txt.split("=")[-1])
        except Exception:
            s_prev = float("nan")
        _log(f"ALREADY METRIC: marker present ({txt}) → output already scaled, not re-scaling "
             f"(Replace clears the marker to re-scale)")
        return s_prev          # non-None → caller knows it IS metric (not a failure)
    s, diag = estimate_v2(output_dir, log=log, near_frac=near_frac,
                          conf_top_frac=conf_top_frac, cfg=cfg, session_dir=session_dir)
    diag["dry_run"] = bool(dry_run)
    try:
        (output_dir / DIAGNOSTICS_NAME).write_text(json.dumps(diag, indent=2))
        _log(f"scale diagnostics persisted → {DIAGNOSTICS_NAME}")
    except Exception as e:  # noqa: BLE001 — diagnostics must never mask the scale itself
        _log(f"WARNING: could not write {DIAGNOSTICS_NAME}: {e}")
    if s is None:
        return None            # genuine failure → caller makes it FATAL
    apply_scale(output_dir, s, dry_run=dry_run)
    if not dry_run:
        marker.write_text(f"s={s:.6f}\n")
    _log(f"✅ metric scale {'(dry-run) ' if dry_run else ''}s={s:.4f} applied "
         f"(source={diag.get('scale_source', 'da3')})")
    return s


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", default="global_median",
                    choices=["global_median", "affine_robust", "depth_dependent"])
    ap.add_argument("--no-vio", action="store_true",
                    help="ignore a VIO trajectory even if the session has one")
    args = ap.parse_args()
    run(Path(args.output_dir), dry_run=args.dry_run,
        cfg={"mode": args.mode, "vio": not args.no_vio})


if __name__ == "__main__":
    main()
