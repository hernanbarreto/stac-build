"""scale_align v2 (precision task, Phase A) — synthetic unit tests.

Covers: estimator modes (recovery + evidence-gated degradation), bit-identical
global_median baseline, outlier robustness, jackknife, confidence bounds,
diagnostics persistence, depth-coverage anchor top-up, and the VIO source
(parsing, per-segment robustness under drift, fail-hard gates, priority over
DA3 + agreement reporting). No GPU, no real sessions."""
import json
from pathlib import Path

import numpy as np
import pytest

from reconstruction import scale_align, scale_model as sm, vio_scale
from ingestors.vio_detector import detect_vio_data


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic session builder
# ──────────────────────────────────────────────────────────────────────────────

H, W = 64, 48


def _depth_field(z_lo, z_hi, seed):
    rng = np.random.default_rng(seed)
    base = np.linspace(z_lo, z_hi, H)[:, None] * np.ones((1, W))
    return (base + 0.01 * (z_hi - z_lo) * rng.standard_normal((H, W))).astype(np.float32)


def build_session(tmp: Path, n_frames=12, s_true=2.0, b_true=0.0, a1_true=0.0,
                  noise=0.01, outlier_frac=0.0, seed=0):
    """Session with omega depth z and DA3 depth d = s·z + a1·z² + b + noise.
    Frames span different depth bands (identifiability across the range)."""
    out = tmp / "output"
    om = out / "omega_run" / "results_output"
    da = out / "da3_run" / "results_output"
    om.mkdir(parents=True); da.mkdir(parents=True)
    rng = np.random.default_rng(seed + 1)
    nums = []
    for i in range(n_frames):
        n = 10 * i
        nums.append(n)
        z_lo = 0.5 + 0.25 * i
        z = _depth_field(z_lo, z_lo + 3.0, seed + i)
        d = (s_true * z + a1_true * z * z + b_true
             + noise * rng.standard_normal(z.shape) * (s_true * z)).astype(np.float32)
        if outlier_frac > 0:
            m = rng.random(z.shape) < outlier_frac
            d[m] = d[m] * rng.uniform(2.0, 5.0, int(m.sum())).astype(np.float32)
        conf = np.ones_like(z)
        np.save(om / "_tmp.npy", z)   # noop to keep layout simple
        np.savez(om / f"frame_{n}.npz", depth=z)
        np.savez(da / f"frame_{n}.npz", depth=d, conf=conf)
    (om / "_tmp.npy").unlink()
    # row-aligned poses (identity rotations, centres on a line) + frames
    lines = []
    for i, n in enumerate(nums):
        T = np.eye(4)
        T[0, 3] = 0.1 * i
        lines.append(" ".join(f"{x:.8g}" for x in T.reshape(-1)))
    (out / "camera_poses.txt").write_text("\n".join(lines) + "\n")
    (out / "camera_frames.txt").write_text(" ".join(str(n) for n in nums) + "\n")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────────────

def test_global_median_bit_identical(tmp_path):
    out = build_session(tmp_path, s_true=2.0, noise=0.02)
    s_prod = scale_align.estimate_scale(out)
    frames = sm.collect_frame_samples(out)
    assert s_prod is not None
    assert sm.trimmed_median_s(frames) == pytest.approx(s_prod, abs=0)
    s2, diag = scale_align.estimate_v2(out, cfg={"mode": "global_median", "vio": False})
    assert s2 == pytest.approx(s_prod, abs=0)
    assert diag["scale_source"] == "da3"
    assert diag["mode_used"] == "global_median"


def test_affine_recovers_gain_and_offset(tmp_path):
    out = build_session(tmp_path, s_true=2.0, b_true=0.30, noise=0.01)
    frames = sm.collect_frame_samples(out)
    sel = sm.select_model(frames, "affine_robust")
    assert not sel["degraded"], sel["degrade_reason"]
    p = sel["model"]["params"]
    assert p["s"] == pytest.approx(2.0, rel=0.03)
    assert p["b"] == pytest.approx(0.30, rel=0.15)
    # the applied similarity is the de-biased GAIN, not the median ratio
    s_applied = sm.applied_gain(sel["model"], frames)
    assert s_applied == pytest.approx(2.0, rel=0.03)
    # the biased baseline would fold b into s
    assert abs(sm.trimmed_median_s(frames) - 2.0) > abs(s_applied - 2.0)


def test_affine_degrades_without_offset(tmp_path):
    out = build_session(tmp_path, s_true=2.0, b_true=0.0, noise=0.02)
    frames = sm.collect_frame_samples(out)
    sel = sm.select_model(frames, "affine_robust")
    assert sel["degraded"]
    assert sel["mode_used"] == "scale_only_robust"
    assert "did not improve" in sel["degrade_reason"]


def test_depth_dependent_recovers_structure(tmp_path):
    out = build_session(tmp_path, s_true=2.0, a1_true=0.15, noise=0.01)
    frames = sm.collect_frame_samples(out)
    sel = sm.select_model(frames, "depth_dependent")
    assert not sel["degraded"], sel["degrade_reason"]
    assert sel["mode_used"].startswith("depth_dependent:")


def test_depth_dependent_degrades_on_pure_scale(tmp_path):
    out = build_session(tmp_path, s_true=2.0, noise=0.02)
    frames = sm.collect_frame_samples(out)
    sel = sm.select_model(frames, "depth_dependent")
    assert sel["degraded"]
    assert sel["mode_used"] in ("scale_only_robust", "affine_robust")


def test_huber_robust_to_outliers(tmp_path):
    out = build_session(tmp_path, s_true=2.0, noise=0.01, outlier_frac=0.10)
    frames = sm.collect_frame_samples(out)
    z, d, w = sm._pooled(frames)
    model = sm._fit_model("scale_only", z, d, w)
    assert model["params"]["s"] == pytest.approx(2.0, rel=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def test_jackknife_and_confidence(tmp_path):
    out = build_session(tmp_path, s_true=2.0, noise=0.02)
    frames = sm.collect_frame_samples(out)
    jk = sm.jackknife_s(frames)
    assert jk is not None and jk["n"] == len(frames)
    assert jk["max_dev_rel"] < 0.05
    c_good, terms = sm.scale_confidence(0.01, 12, 0.001, None)
    c_bad, _ = sm.scale_confidence(0.20, 4, 0.05, None)
    assert 0.0 <= c_bad < c_good <= 1.0
    c_vio, t_vio = sm.scale_confidence(0.01, 12, 0.001, 2.0)
    assert "vio_agreement" in t_vio and 0.0 <= c_vio <= 1.0


def test_run_writes_diagnostics_dry(tmp_path):
    out = build_session(tmp_path, s_true=2.0, noise=0.02)
    s = scale_align.run(out, dry_run=True, cfg={"mode": "global_median", "vio": False})
    assert s == pytest.approx(scale_align.estimate_scale(out), abs=0)
    diag = json.loads((out / scale_align.DIAGNOSTICS_NAME).read_text())
    assert diag["dry_run"] is True
    assert diag["s_applied"] == pytest.approx(s)
    assert diag["anchors"]["count"] == 12
    assert diag["residual_vs_depth"]
    assert 0.0 <= diag["scale_confidence"] <= 1.0
    assert not (out / ".metric_scale_applied").exists()   # dry run never marks


def test_residual_profile_flags_depth_structure(tmp_path):
    out = build_session(tmp_path, s_true=2.0, a1_true=0.15, noise=0.005)
    frames = sm.collect_frame_samples(out)
    s = sm.trimmed_median_s(frames)
    prof = sm.residual_depth_profile(frames, s)
    assert len(prof) >= 4
    r = [p["median_residual_pct"] for p in prof]
    assert max(r) - min(r) > 2.0      # the un-modeled gain slope shows as structure


# ──────────────────────────────────────────────────────────────────────────────
# Anchor depth-coverage top-up
# ──────────────────────────────────────────────────────────────────────────────

def test_topup_plans_uncovered_depth_bins(tmp_path):
    out = build_session(tmp_path, n_frames=12, s_true=2.0)
    da = out / "da3_run" / "results_output"
    # keep DA3 only on the 4 SHALLOWEST frames → deep bins uncovered
    for i in range(4, 12):
        (da / f"frame_{10 * i}.npz").unlink()
    sel_files = [f"{10 * i}.jpg" for i in range(12)]
    plan = sm.plan_depth_coverage_topup(out, sel_files, max_topup=8)
    assert plan, "expected top-up extractions for uncovered deep bins"
    nums = {int(f.split(".")[0]) // 10 for f in plan}
    assert all(i >= 4 for i in nums)  # planned frames are the deep ones
    # full coverage → nothing to plan
    out2 = build_session(tmp_path / "full", n_frames=12, s_true=2.0)
    assert sm.plan_depth_coverage_topup(out2, sel_files, max_topup=8) == []


# ──────────────────────────────────────────────────────────────────────────────
# VIO
# ──────────────────────────────────────────────────────────────────────────────

def _vio_csv(path: Path, t, p, header=True):
    lines = ["timestamp,x,y,z"] if header else []
    for ti, pi in zip(t, p):
        lines.append(f"{ti:.3f},{pi[0]:.5f},{pi[1]:.5f},{pi[2]:.5f}")
    path.write_text("\n".join(lines) + "\n")


def test_vio_detector(tmp_path):
    assert detect_vio_data(tmp_path)["has_vio"] is False
    t = np.arange(0, 3, 0.1)
    _vio_csv(tmp_path / "vio_trajectory.csv", t, np.zeros((len(t), 3)))
    det = detect_vio_data(tmp_path)
    assert det["has_vio"] and det["format"] == "csv"


def test_vio_csv_and_json_parse(tmp_path):
    t = np.arange(0, 10, 0.1)
    p = np.stack([t * 0.5, np.zeros_like(t), np.zeros_like(t)], 1)
    _vio_csv(tmp_path / "a.csv", t, p, header=True)
    ta, pa, _ = vio_scale.load_vio_trajectory(tmp_path / "a.csv")
    assert len(ta) == len(t) and pa.shape == (len(t), 3)
    _vio_csv(tmp_path / "b.csv", t, p, header=False)
    tb, pb, _ = vio_scale.load_vio_trajectory(tmp_path / "b.csv")
    assert np.allclose(pa, pb)
    (tmp_path / "c.json").write_text(json.dumps(
        {"video_fps": 30.0,
         "samples": [{"t": float(ti), "p": [float(x) for x in pi]}
                     for ti, pi in zip(t, p)]}))
    tc, pc, fps = vio_scale.load_vio_trajectory(tmp_path / "c.json")
    assert fps == 30.0 and np.allclose(pc, pa)
    (tmp_path / "bad.csv").write_text("timestamp,x,y,z\n1.0,2.0,oops,4.0\n" * 30)
    with pytest.raises(RuntimeError, match="malformed"):
        vio_scale.load_vio_trajectory(tmp_path / "bad.csv")


def _walk(s_true=40.0, dur=60.0, speed=1.0, drift_amp=0.0):
    """Metric VIO walk + omega keyframes (units = meters / s_true)."""
    vt = np.arange(0.0, dur, 1.0 / 10)
    x = speed * vt
    drift = drift_amp * np.sin(2 * np.pi * vt / dur)
    vp = np.stack([x, drift, np.zeros_like(x)], 1)
    kt = np.arange(0.0, dur, 1.0)
    kc = np.stack([speed * kt / s_true, np.zeros_like(kt), np.zeros_like(kt)], 1)
    return vt, vp, kt, kc


def test_vio_segment_scale_recovers_s_under_drift():
    vt, vp, kt, kc = _walk(s_true=40.0, drift_amp=0.5)
    res = vio_scale.estimate_vio_scale(vt, vp, kt, kc)
    assert res["s_vio"] == pytest.approx(40.0, rel=0.03)
    assert res["n_segments"] >= 8
    assert res["coverage_frac"] > 0.9


def test_vio_fail_hard_gates():
    vt, vp, kt, kc = _walk()
    with pytest.raises(RuntimeError, match="overlap"):
        vio_scale.estimate_vio_scale(vt + 1000.0, vp, kt, kc)
    with pytest.raises(RuntimeError, match="segments"):
        vio_scale.estimate_vio_scale(vt, np.zeros_like(vp), kt, kc)  # static VIO


def test_vio_short_file_fails(tmp_path):
    _vio_csv(tmp_path / "v.csv", [0.0, 0.1], [[0, 0, 0], [0.1, 0, 0]])
    with pytest.raises(RuntimeError, match="samples"):
        vio_scale.load_vio_trajectory(tmp_path / "v.csv")


def test_vio_priority_and_agreement(tmp_path, monkeypatch):
    # session with DA3 s≈2.0; VIO says the walk is 5% longer than omega·2.0
    out = build_session(tmp_path, n_frames=12, s_true=2.0, noise=0.01)
    dur = 110.0                                # nums 0..110 at fps=1 → t = num
    vt = np.arange(0.0, dur + 1, 0.1)
    s_vio_true = 2.1
    vp = np.stack([0.01 * s_vio_true * vt, np.zeros_like(vt), np.zeros_like(vt)], 1)
    # omega centres move 0.1/frame-index = 0.01/frame-num (nums = 10·i)
    _vio_csv(tmp_path / "vio_trajectory.csv", vt, vp)
    monkeypatch.setattr("reconstruction.vio_scale.video_fps", lambda _s: 1.0)
    s, diag = scale_align.estimate_v2(
        out, cfg={"mode": "global_median", "vio": True,
                  "vio_segment_s": 10.0, "vio_min_segments": 5,
                  "vio_min_coverage": 0.5},
        session_dir=tmp_path)
    assert diag["scale_source"] == "vio"
    assert s == pytest.approx(2.1, rel=0.03)
    ag = diag["vio"]["agreement_pct"]
    assert ag == pytest.approx(100.0 * (diag["s_da3"] / s - 1.0))
    assert abs(ag) < 10.0
    assert 0.0 <= diag["scale_confidence"] <= 1.0
