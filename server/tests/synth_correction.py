"""Synthetic correction scenes with known ground truth (no GPU).

Builds a full on-disk mini-session (the real layout: cleaned_cloud.ply with
provenance, cleaned_cloud_raw.ply, camera_poses.txt + camera_frames.txt,
segmentation_result.json, scale_diagnostics.json, optional chunk_plan.json
and omega_run npz) with a trajectory, a floor (flat / ramp / step), objects
of the three observability classes (plane wall, cylinder column, asymmetric
box) and TWO VISITS to the same objects; then injects controlled errors:
(a) progressive rigid drift over the revisit, (b) depth compression k,
(c) both.

Conventions follow test_scale_v2 / test_pose_refine: seeded RNG, tmp_path
sessions, assertions on persisted artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from correction.config import load_correction_config  # noqa: E402

FRAME_STEP = 10          # real frame number = keyframe * FRAME_STEP


def make_correction_raw_cfg(**overrides) -> dict:
    """A COMPLETE correction: config dict scaled for small synthetic clouds.
    Every mandatory key present; overrides go per dotted path."""
    raw = {"correction": {
        "evidence": {
            "obb_margin_m": 0.05, "obb_core_pct": 98.0, "min_object_points_solve": 30,
            "min_object_points_fingerprint": 30, "min_baseline_m": 0.30,
            "min_objects_for_depth": 2, "visit_gap_kf": 2,
        },
        "observability": {
            "pca_ratio_planar": 0.05, "pca_ratio_cylindrical": 0.15,
            "bounded_extent_tol": 0.20, "yaw_anisotropy_max": 0.60,
        },
        "solve": {
            "icp_iters": 60, "icp_trim": 0.7, "icp_sample": 6000,
            "eval_sample": 4000, "icp_min_corr": 30,
            "icp_converge_deg": 0.001, "icp_converge_m": 0.00001,
            "plane_ransac_tol_m": 0.02, "plane_ransac_iters": 150,
            "plane_ransac_sample": 5000, "depth_compress_tol": 0.03,
            "seed": 0,
        },
        "gates": {
            "mode": "veto",
            "max_object_residual_m": 0.15, "residual_improvement_ratio": 0.5,
            "collapse_floor_m": 0.03,
            "max_rot_deg": 10.0, "max_translation_m": 3.0,
            "heldout_floor_tol_m": 0.05, "heldout_floor_abs_m": 0.08,
            "max_step_mm": 60.0, "max_step_deg": 1.5,
            "heldout_object_tol_m": 0.10,
            "scale_agree_tol": 0.05, "scale_mad_tol": 0.03,
        },
        "floor": {
            "model_default": "plane", "band_m": 0.5, "low_band_pct": 5.0,
            "min_tilt_deg": 1.0,
            "max_tilt_deg": 10.0, "min_inliers": 60,
            "min_inlier_ratio": 0.5, "ransac_tol_m": 0.02,
            "ransac_refit_band_m": 0.03, "ransac_iters": 200,
            "ransac_sample": 20000, "smooth_window_kf": 5,
            "step_demote_m": 0.15, "reference_span_kf": 8,
        },
        "revisit": {
            "sample_per_kf": 400, "min_gap_kf": 8, "min_covis": 0.25,
            "depth_min_m": 0.3, "depth_max_m": 12.0, "zbuffer_cell_px": 8,
            "same_surface_tol_m": 0.10, "same_surface_tol_rel": 0.10,
            "region_cell_m": 1.0,
            "region_sample": 4000, "offset_min_m": 0.05, "image_max_px": 320,
        },
        "consistency": {"cell_m": 1.0, "block_min_points": 200,
                        "block_sample": 5000, "writer_min_points": 50,
                        "writer_query_sample": 500, "viewer_probe": 20,
                        "viewer_min_frac": 0.5, "near_kf": 5},
        "kfgraph": {"pair_src_sample": 500, "pair_tgt_sample": 1500,
                    "normal_k": 8, "gn_iters": 10, "gn_converge": 1e-4,
                    "damping": 1e-3, "damping_min": 1e-9, "lm_factor": 10.0, "lm_tries": 8,
                    "floor_m": 0.03},
        "photo": {"candidate_sample": 300, "candidate_min": 30,
                  "pair_min_points": 50, "points_per_pair": 100,
                  "splat_radius": 1, "outer_iters": 1, "huber_px": 4.0},
        "posegraph": {"loop_weight": 100.0, "rot_lever_m": 3.0,
                      "min_blocks_improved": 1.0},
        "apply": {"depth_correction_mode": "sidecar",
                  "potree_rebuild": False},
        "runtime": {"workers": 2},
    }}
    for dotted, value in overrides.items():
        node = raw["correction"]
        parts = dotted.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = value
    return raw


def make_correction_cfg(**overrides):
    return load_correction_config(make_correction_raw_cfg(**overrides))


# ── geometry builders ────────────────────────────────────────────────────

def _wall(rng, n=2500):
    """Planar wall at z=2.0, x∈[1,3], y∈[0.2,2]."""
    x = rng.uniform(1.0, 3.0, n)
    y = rng.uniform(0.2, 2.0, n)
    z = np.full(n, 2.0) + rng.normal(0, 0.004, n)
    return np.stack([x, y, z], 1)


def _box(rng, n=2500, center=(5.0, 0.6, 0.8)):
    """Compact asymmetric box 0.9×0.5×0.35 (distinct extents → full SE(3)
    observability under the yaw solver)."""
    c = np.asarray(center)
    p = rng.uniform(-1, 1, (n, 3)) * np.array([0.45, 0.25, 0.175])
    # asymmetric bump on one corner so no reflection symmetry survives
    bump = rng.uniform(-1, 1, (n // 4, 3)) * 0.1 + np.array([0.5, 0.3, 0.2])
    return np.concatenate([c + p, c + bump]) + rng.normal(0, 0.003,
                                                          (n + n // 4, 3))


def _cylinder(rng, n=2500, center=(7.0, 0.0, 0.8)):
    """Vertical column r=0.09, h=2.2 (linear/elongated class)."""
    th = rng.uniform(0, 2 * np.pi, n)
    r = 0.09 + rng.normal(0, 0.003, n)
    y = rng.uniform(0.0, 2.2, n)
    c = np.asarray(center)
    return np.stack([c[0] + r * np.cos(th), y, c[2] + r * np.sin(th)], 1)


def _floor_patch(rng, x0, x1, n, y_fn):
    x = rng.uniform(x0, x1, n)
    z = rng.uniform(-1.5, 1.5, n)
    y = y_fn(x) + rng.normal(0, 0.004, n)
    return np.stack([x, y, z], 1)


# ── PLY writing (real header, real provenance) ───────────────────────────

_PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("frame_global", "<i4"), ("pixel_row", "<i4"), ("pixel_col", "<i4"),
    ("confidence", "<f4"),
])

_HEADER_TXT = """ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int frame_global
property int pixel_row
property int pixel_col
property float confidence
end_header
"""


def write_cloud(path: Path, xyz: np.ndarray, fg: np.ndarray,
                rng: np.random.Generator) -> None:
    n = len(xyz)
    data = np.zeros(n, dtype=_PLY_DTYPE)
    data["x"], data["y"], data["z"] = (xyz[:, 0].astype(np.float32),
                                       xyz[:, 1].astype(np.float32),
                                       xyz[:, 2].astype(np.float32))
    data["red"] = rng.integers(0, 255, n)
    data["green"] = rng.integers(0, 255, n)
    data["blue"] = rng.integers(0, 255, n)
    data["frame_global"] = fg.astype(np.int32)
    data["pixel_row"] = rng.integers(0, 384, n)
    data["pixel_col"] = rng.integers(0, 688, n)
    data["confidence"] = rng.uniform(0.5, 1.0, n).astype(np.float32)
    header = _HEADER_TXT.format(n=n)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


class SynthScene:
    """One built session with its ground truth kept for assertions."""

    def __init__(self, output_dir: Path, xyz_true, fg, ks, poses_true,
                 frames, instances, floor_mask, gt):
        self.output_dir = output_dir
        self.xyz_true = xyz_true
        self.fg = fg
        self.ks = ks
        self.poses_true = poses_true
        self.frames = frames
        self.instances = instances
        self.floor_mask = floor_mask
        self.gt = gt


def build_scene(tmp: Path, *, n_kf: int = 30, seed: int = 7,
                drift_yaw_deg: float = 0.0, drift_t=(0.0, 0.0, 0.0),
                depth_c: float = 1.0,
                ref_kfs=range(0, 10), revisit_kfs=range(20, 30),
                floor: str = "flat", floor_slope: float = 0.04,
                da3_agrees_with_truth: bool = True,
                extra_unmarked: bool = True,
                write_raw: bool = True,
                write_omega_npz: bool = False,
                chunk_plan: Optional[dict] = None,
                floor_pts_per_kf: int = 260,
                drift_model: str = "rate") -> SynthScene:
    """Build a session under tmp/output. Ground-truth errors injected on the
    REVISIT keyframes: rigid drift ramps linearly from identity at
    ref end → full (yaw, t) at the revisit start and beyond; depth
    compression c contracts each revisit point toward its own camera
    (the corrector must recover k = 1/c)."""
    rng = np.random.default_rng(seed)
    output = tmp / "output"
    output.mkdir(parents=True, exist_ok=True)

    frames = [k * FRAME_STEP for k in range(n_kf)]
    # trajectory: straight walk along +x, 0.5 m per keyframe
    centers = np.stack([np.arange(n_kf) * 0.5,
                        np.full(n_kf, 1.6),
                        np.zeros(n_kf)], 1)
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    poses[:, :3, 3] = centers

    if floor == "flat":
        y_fn = lambda x: np.zeros_like(x)                      # noqa: E731
    elif floor == "ramp":
        y_fn = lambda x: floor_slope * x                       # noqa: E731
    elif floor == "step":
        y_fn = lambda x: np.where(x > 7.0, 0.5, 0.0)           # noqa: E731
    else:
        raise ValueError(floor)

    ref_kfs = list(ref_kfs)
    revisit_kfs = list(revisit_kfs)

    parts: List[np.ndarray] = []
    part_fg: List[np.ndarray] = []
    instances: List[dict] = []
    floor_flags: List[np.ndarray] = []

    def _add(pts, kf_pool, is_floor=False):
        kfs = rng.choice(kf_pool, len(pts))
        parts.append(pts)
        part_fg.append(np.array([frames[k] for k in kfs]))
        floor_flags.append(np.full(len(pts), is_floor))
        start = sum(len(p) for p in parts[:-1])
        return np.arange(start, start + len(pts))

    # objects seen in BOTH visits (two independent samplings = two copies)
    obj_defs = [("wall1", _wall), ("box1", _box), ("col1", _cylinder)]
    obj_indices: Dict[str, List[np.ndarray]] = {}
    for name, fn in obj_defs:
        i_ref = _add(fn(rng), ref_kfs)
        i_rev = _add(fn(rng), revisit_kfs)
        obj_indices[name] = [i_ref, i_rev]
    if extra_unmarked:
        shelf = lambda r: _box(r, n=1200, center=(2.0, 1.4, -1.0))  # noqa: E731
        i_ref = _add(shelf(rng), ref_kfs)
        i_rev = _add(shelf(rng), revisit_kfs)
        obj_indices["shelf1"] = [i_ref, i_rev]

    # floor: every keyframe contributes points around its camera x
    for k in range(n_kf):
        x0 = centers[k, 0] - 1.0
        _add(_floor_patch(rng, x0, x0 + 2.0, floor_pts_per_kf, y_fn),
             [k], is_floor=True)

    xyz_true = np.concatenate(parts)
    fg = np.concatenate(part_fg)
    floor_mask = np.concatenate(floor_flags)
    kf_of_frame = {f: k for k, f in enumerate(frames)}
    ks = np.array([kf_of_frame[f] for f in fg])

    # ── inject the observed (drifted) geometry ──────────────────────────
    theta = np.radians(drift_yaw_deg)
    t_full = np.asarray(drift_t, dtype=np.float64)
    ramp_start = ref_kfs[-1]
    ramp_end = revisit_kfs[0]

    walk = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
        np.diff(centers, axis=0), axis=1))])

    def drift_of_kf(k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """USER 2026-09-09 drift model ('rate'): the accumulated error grows
        linearly with the distance walked from the start (E(0)=0), the
        injected (yaw, t) being the error at the END of the walk. 'step'
        keeps the older ramp-then-constant injection."""
        if drift_model == "rate":
            w = float(walk[k] / walk[-1])
        else:
            if k <= ramp_start:
                return np.eye(3), np.zeros(3), 1.0
            w = min(1.0, (k - ramp_start) / max(ramp_end - ramp_start, 1))
        th = theta * w
        R = np.array([[np.cos(th), 0, np.sin(th)],
                      [0, 1, 0],
                      [-np.sin(th), 0, np.cos(th)]])
        c_here = 1.0 + (depth_c - 1.0) * (1.0 if k >= ramp_end else 0.0)
        return R, t_full * w, c_here

    xyz_obs = xyz_true.copy()
    poses_obs = poses.copy()
    for k in range(n_kf):
        R, t, c = drift_of_kf(k)
        sel = np.where(ks == k)[0]
        if not len(sel):
            continue
        cam = centers[k]
        if c != 1.0:
            xyz_obs[sel] = cam + (xyz_obs[sel] - cam) * c
        xyz_obs[sel] = xyz_obs[sel] @ R.T + t
        poses_obs[k][:3, :3] = R @ poses_obs[k][:3, :3]
        poses_obs[k][:3, 3] = R @ poses_obs[k][:3, 3] + t

    # ── persist the session ─────────────────────────────────────────────
    write_cloud(output / "cleaned_cloud.ply", xyz_obs, fg, rng)
    if write_raw:
        write_cloud(output / "cleaned_cloud_raw.ply", xyz_obs, fg,
                    np.random.default_rng(seed))  # same order, own colors
    (output / "camera_frames.txt").write_text(
        "\n".join(str(f) for f in frames) + "\n")
    (output / "camera_poses.txt").write_text("\n".join(
        " ".join(f"{v:.9f}" for v in P.reshape(-1)) for P in poses_obs)
        + "\n")

    for iid, (name, _fn) in enumerate(
            [(n, f) for n, f in obj_defs]
            + ([("shelf1", None)] if extra_unmarked else []), start=1):
        gidx = np.concatenate(obj_indices[name])
        instances.append({"instance_id": iid, "id": iid - 1, "label": name,
                          "color": "#ffcc00",
                          "globalIndices": [int(i) for i in gidx],
                          "total_points": int(len(gidx))})
    (output / "segmentation_result.json").write_text(json.dumps(
        {"type": "segmentation", "version": "3.0", "instances": instances}))

    # scale diagnostics: 6 anchors spread over the walk. If DA3 agrees with
    # the TRUTH, compressed frames show s_f = 1/c (ratio DA3/omega);
    # otherwise DA3 "confirms" the observed depth (s_f = 1).
    anchor_kfs = [2, 7, 13, 21, 24, 28]
    anchor_frames = []
    for k in anchor_kfs:
        c_here = drift_of_kf(k)[2]
        s_f = (1.0 / c_here) if da3_agrees_with_truth else 1.0
        anchor_frames.append({"num": frames[k], "s_f": round(s_f, 6),
                              "n_px": 5000, "z_median_omega": 2.5,
                              "d_median_m": 2.5})
    (output / "scale_diagnostics.json").write_text(json.dumps({
        "version": 1, "generated_by": "synth_correction",
        "mode_used": "global_median", "s_applied": 1.0,
        "scale_source": "da3",
        "anchors": {"count": len(anchor_frames), "mad_rel": 0.01,
                    "frames": anchor_frames},
    }, indent=1))

    if chunk_plan is not None:
        (output / "chunk_plan.json").write_text(json.dumps(chunk_plan))

    if write_omega_npz:
        d = output / "omega_run" / "results_output"
        d.mkdir(parents=True, exist_ok=True)
        H, W = 24, 32
        for k in range(n_kf):
            z = np.full((H, W), 2.0 + 0.01 * k, np.float32)
            np.savez(d / f"frame_{frames[k]}.npz", depth=z)

    gt = {"yaw_deg": drift_yaw_deg, "t": t_full, "depth_c": depth_c,
          "drift_model": drift_model, "walk": walk,
          "k_expected": (1.0 / depth_c), "ramp_start": ramp_start,
          "ramp_end": ramp_end, "revisit_kfs": revisit_kfs,
          "ref_kfs": ref_kfs, "anchor_kfs": anchor_kfs}
    return SynthScene(output, xyz_true, fg, ks, poses, frames, instances,
                      floor_mask, gt)


def session_files_snapshot(output_dir: Path) -> Dict[str, bytes]:
    """Byte snapshot of every file in the session (for untouched-on-reject
    assertions)."""
    snap = {}
    for p in sorted(Path(output_dir).rglob("*")):
        if p.is_file():
            snap[str(p.relative_to(output_dir))] = p.read_bytes()
    return snap
