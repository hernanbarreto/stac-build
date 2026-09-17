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
    assert rep["status"] == "applied", rep.get("rejection_reason")
    _, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    err = np.linalg.norm(xyz - scene.xyz_true, axis=1)
    mid = np.isin(scene.ks, list(range(12, 18)))     # unobserved middle
    assert float(np.median(err[mid])) < 0.03, float(np.median(err[mid]))
    rev = np.isin(scene.ks, list(scene.gt["revisit_kfs"])) & ~scene.floor_mask
    assert float(np.median(err[rev])) < 0.03


# ── the curve moves along the SCREW, not by lerping the translation ──────
# USER 2026-09-17. The model spread the correction as C(f) = (exp(f·r), f·τ):
# the rotation on the manifold, the translation scaled LINEARLY as if the
# rotation were not there. That is exact only while the rotation is small.
# pccr desk#201: a closure of 148.5° that lands its copy on its twin 60.6 cm
# away was spread as displacements of metres in the middle of the walk — the
# greedy loop measured "its own copies 60.8 → 783.2 cm" and the pose graph,
# fed 19 edges like it, closed 0% and left the session at epoch 0.

def _copies_after(deg, n=216, walk=19.3, ref_kf=10, anchor_kf=214):
    """Median separation of two copies of one object after the distributed
    field is applied — each copy seen from its OWN stretch of keyframes, which
    is where a torn field shows up."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    d = np.linspace(0.0, walk, n)
    axis = np.array([0.2, 0.95, 0.24]); axis /= np.linalg.norm(axis)
    R = Rotation.from_rotvec(np.deg2rad(deg) * axis).as_matrix()
    B = np.array([4.0, 1.2, 7.8]) + rng.normal(0, 0.35, (3000, 3))
    c = B.mean(0)
    t = np.array([0.55, 0.05, 0.25]) + c - R @ c          # moves the copy ~60 cm
    A = (B - t) @ np.linalg.inv(R).T
    sol = [{"anchor_kf": anchor_kf, "kf_span": [200, 215], "R": R, "t": t, "k": 1.0}]
    R_kf, t_kf, _k, _rep = distribute.distribute(n, d, ref_kf, sol)
    kA = rng.integers(200, 216, len(A)); kB = rng.integers(4, 21, len(B))
    wA = np.einsum("nij,nj->ni", R_kf[kA], A) + t_kf[kA]
    wB = np.einsum("nij,nj->ni", R_kf[kB], B) + t_kf[kB]
    before = float(np.median(np.linalg.norm(A - B, axis=1)))
    return before, float(np.median(np.linalg.norm(wA - wB, axis=1)))


def test_a_large_rotation_closure_still_brings_its_copies_together():
    """The regression: at 148.5° the separable curve left 83.6 cm of a 96.8 cm
    separation — it barely closed anything."""
    before, after = _copies_after(148.5)
    assert before > 0.9
    assert after < 0.10, f"{before*100:.1f} cm → {after*100:.1f} cm"


def test_the_closure_is_honoured_across_the_whole_rotation_range():
    """Flat, not degrading with the angle: 90° used to leave 29.7 cm."""
    for deg in (0.5, 5.7, 11.3, 33.3, 90.8, 148.5):
        _before, after = _copies_after(deg)
        assert after < 0.10, f"{deg}°: {after*100:.1f} cm left"


def test_a_small_rotation_is_unchanged_by_the_screw_model():
    """What already worked must keep working — below ~30° the two models agree,
    so floor corrections and near-pure translations are untouched."""
    for deg in (0.5, 5.7, 11.3):
        _b, after = _copies_after(deg)
        assert after < 0.035, f"{deg}°: {after*100:.1f} cm"


def test_screw_exp_and_log_are_inverses_as_transforms():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(64):
        T = np.eye(4)
        T[:3, :3] = Rotation.random(random_state=int(rng.integers(10 ** 6))).as_matrix()
        T[:3, 3] = rng.normal(0, 8, 3)
        worst = max(worst, float(np.abs(distribute.screw_exp(distribute.screw_log(T)) - T).max()))
    assert worst < 1e-9, worst
    # the degenerate case the left Jacobian has to handle: no rotation at all
    T = np.eye(4); T[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(distribute.screw_exp(distribute.screw_log(T)), T)


def test_a_pure_translation_closure_is_exactly_the_old_straight_line():
    """With no rotation the screw IS the straight line, so the drift-rate
    arithmetic of the first test must survive to the digit."""
    n = 31
    d = distribute.chainage(_poses_line(n))
    sol = [{"anchor_kf": 24, "kf_span": [24, 30], "R": np.eye(3),
            "t": np.array([-1.0, 0.0, 0.0]), "k": 1.0}]
    _R, t, _k, rep = distribute.distribute(n, d, 4, sol)
    eps = 1.0 / (12.0 - 2.0)
    assert np.allclose(t[0], 0)
    assert abs(t[24][0] + eps * 12.0) < 1e-9
    assert abs(t[30][0] + eps * 15.0) < 1e-9
    assert np.allclose(np.diff(t[:, 0]), -eps * 0.5)
    assert abs(rep["drift_rate_mm_per_m"] - 100.0) < 1e-6
