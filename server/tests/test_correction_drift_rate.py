"""USER 2026-09-09 drift-rate model: E(0)=0 at the start of the walk, a
duplicate at chainage d_j pins the line (or a knot of the curve), every
keyframe is corrected by −E(d_k) — backwards (little near the start) and
forwards (extrapolated). Nothing is smeared over keyframes that were right."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction import distribute                                 # noqa: E402
from correction.ledger import load_epoch_npz                      # noqa: E402
from correction.run import run_objects                            # noqa: E402
from correction.session import read_ply                           # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def _poses_line(n, step=0.5):
    P = np.tile(np.eye(4), (n, 1, 1))
    P[:, 0, 3] = np.arange(n) * step
    return P


def test_line_through_start_and_one_closure():
    n = 31
    d = distribute.chainage(_poses_line(n))          # 0..15 m
    # reference at 2 m, displaced copy at 12 m, closure (displaced→ref) −1 m
    sol = [{"anchor_kf": 24, "kf_span": [24, 30], "R": np.eye(3),
            "t": np.array([-1.0, 0.0, 0.0]), "k": 1.0}]
    R, t, k, rep = distribute.distribute(n, d, 4, sol)
    eps = 1.0 / (12.0 - 2.0)                         # 0.1 m per metre
    assert np.allclose(t[0], 0)                      # start exact
    assert abs(t[4][0] + eps * 2.0) < 1e-9           # reference moves too
    assert abs(t[24][0] + eps * 12.0) < 1e-9         # closure knot
    assert abs(t[30][0] + eps * 15.0) < 1e-9         # extrapolated forward
    assert np.allclose(np.diff(t[:, 0]), -eps * 0.5) # constant rate
    assert rep["mode"] == "drift_rate_per_metre"
    assert abs(rep["drift_rate_mm_per_m"] - 100.0) < 1e-6


def test_two_closures_make_a_piecewise_curve():
    n = 31
    d = distribute.chainage(_poses_line(n))
    sols = [{"anchor_kf": 12, "kf_span": [12, 14], "R": np.eye(3),
             "t": np.array([-0.2, 0, 0]), "k": 1.0},
            {"anchor_kf": 24, "kf_span": [24, 30], "R": np.eye(3),
             "t": np.array([-1.0, 0, 0]), "k": 1.0}]
    R, t, k, rep = distribute.distribute(n, d, 0, sols)
    assert abs(t[12][0] + 0.2) < 1e-9 and abs(t[24][0] + 1.0) < 1e-9
    # slope changes at the first knot: steeper afterwards
    s1 = -(t[12][0] - t[0][0]) / 6.0
    s2 = -(t[24][0] - t[12][0]) / 6.0
    assert s2 > s1
    assert len(rep["knots"]) == 3


def test_rate_drift_is_recovered_along_the_whole_walk(tmp_path):
    """Injected rate drift (grows with the walk) → after the correction the
    error is small EVERYWHERE, including the middle of the walk that no
    marked object observed."""
    scene = build_scene(tmp_path, drift_yaw_deg=1.0,
                        drift_t=(0.30, 0.0, 0.12), drift_model="rate")
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg(**{"gates.mode": "advisory"}))
    assert rep["status"] == "pending", rep.get("rejection_reason")
    _, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    err = np.linalg.norm(xyz - scene.xyz_true, axis=1)
    mid = np.isin(scene.ks, list(range(12, 18)))     # unobserved middle
    assert float(np.median(err[mid])) < 0.03, float(np.median(err[mid]))
    rev = np.isin(scene.ks, list(scene.gt["revisit_kfs"])) & ~scene.floor_mask
    assert float(np.median(err[rev])) < 0.03
