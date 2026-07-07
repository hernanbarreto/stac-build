# STAC-Builder — Phase R.2/R.4/R.5/R.6/R.9 unit tests (synthetic).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.vote import vote_entropy, vote_point, View  # noqa: E402
from phase_r.residuals import (  # noqa: E402
    Sim3, sim3_from_obb_pair, optimize_window_graph, WindowEdge,
)
from phase_r.metric_hierarchy import umeyama, MetricAuthority  # noqa: E402
from phase_r.depth_regularization import fit_plane, regularize_depth_to_plane  # noqa: E402
from phase_r.failsafe import compare_and_gate  # noqa: E402

rng = np.random.default_rng(0)
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])


# ── R.2 vote ────────────────────────────────────────────────────────
def test_vote_entropy_bounds():
    assert vote_entropy({1: 10}) == 0.0
    assert vote_entropy({1: 5, 2: 5}) > 0.6  # ln2 ~ 0.693


def test_vote_point_assigns_correct_instance():
    # two views, a point that lands in instance 7's mask in both
    H, W = 480, 640
    m7 = np.zeros((H, W), bool); m7[200:300, 300:400] = True
    m9 = np.zeros((H, W), bool); m9[0:50, 0:50] = True
    c2w = np.eye(4)
    views = [View(c2w, K, {7: m7, 9: m9}, (W, H)),
             View(c2w, K, {7: m7, 9: m9}, (W, H))]
    # point at ~ (0,0,4) projects to image centre (320,240) -> inside m7
    a, counts, ent = vote_point(np.array([0.0, 0.06, 4.0]), views)
    assert a == 7, (a, counts)
    assert ent == 0.0  # unanimous


# ── R.4 Sim3 + pose graph ───────────────────────────────────────────
def test_sim3_roundtrip():
    v = np.array([0.1, 0.2, -0.1, 0.3, 1.0, -2.0, 0.5])
    S = Sim3.exp(v)
    assert np.allclose(S.log(), v, atol=1e-8)
    I = S.compose(S.inverse())
    assert np.allclose(I.R, np.eye(3), atol=1e-8) and np.allclose(I.t, 0, atol=1e-8)
    assert abs(I.s - 1) < 1e-8


def _obb(T, ext):
    aabb = np.array([-ext[0]/2, ext[0]/2, -ext[1]/2, ext[1]/2, -ext[2]/2, ext[2]/2])
    return T, aabb


def test_sim3_from_obb_pair_recovers_drift():
    T0 = np.eye(4); ext0 = np.array([1.0, 2.0, 0.5])
    D = Sim3.exp(np.array([0.05, 0.1, 0.0, 0.0, 0.2, -0.1, 0.05]))
    T1 = np.eye(4); T1[:3, :3] = D.R; T1[:3, 3] = D.apply(T0[:3, 3])
    ext1 = D.s * ext0
    M = sim3_from_obb_pair(*_obb(T0, ext0), *_obb(T1, ext1))
    assert abs(M.s - D.s) < 1e-6
    assert np.allclose(M.R, D.R, atol=1e-6) and np.allclose(M.t, D.t, atol=1e-6)


def test_pose_graph_reduces_residual():
    # 3 windows; instance drifts by D1 (win0->1) and D2 (win0->2)
    D1 = Sim3.exp(np.array([0.03, 0.05, 0.0, 0.0, 0.15, -0.05, 0.02]))
    D2 = Sim3.exp(np.array([0.06, 0.0, 0.08, 0.0, -0.1, 0.2, -0.04]))
    edges = [WindowEdge(0, 1, D1), WindowEdge(0, 2, D2), WindowEdge(1, 2, D1.inverse().compose(D2))]
    corr, stats = optimize_window_graph(3, edges)
    assert stats["cost"] < 0.02 * stats["cost0"] + 1e-9, stats
    # recovered correction for window 1 should ~ D1^-1
    err = corr[1].compose(D1).magnitude()
    assert err < 0.05, err


# ── R.6 metric hierarchy ────────────────────────────────────────────
def test_umeyama_recovers_similarity():
    src = rng.normal(0, 1, (50, 3))
    s, R = 2.5, umeyama(rng.normal(0, 1, (3, 3)), rng.normal(0, 1, (3, 3)))[1]  # any R
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec([0.2, -0.3, 0.1]).as_matrix()
    t = np.array([1.0, -2.0, 0.5])
    dst = (s * (R @ src.T).T) + t
    se, Re, te = umeyama(src, dst)
    assert abs(se - s) < 1e-6 and np.allclose(Re, R, atol=1e-6) and np.allclose(te, t, atol=1e-5)


def test_metric_authority_marker_wins():
    auth = MetricAuthority(marker_scale=1.0, tolerance_frac=0.02)
    assert auth.resolve_scale(1.005) == 1.005          # within tol, anchor kept
    assert auth.resolve_scale(1.10, window_id=3) == 1.0  # beyond tol, marker wins
    assert len(auth.log.conflicts) == 1 and auth.log.conflicts[0]["resolution"] == "marker_wins"


# ── R.5 depth regularization ────────────────────────────────────────
def test_depth_regularization_flattens_whitelisted_only():
    H, W = 60, 80
    # ground-truth frontal plane at z=3, camera K
    plane = (np.array([0.0, 0.0, 1.0]), -3.0)  # z=3
    depth = np.full((H, W), 3.0) + rng.normal(0, 0.1, (H, W))
    mask = np.ones((H, W), bool)
    reg = regularize_depth_to_plane(depth, mask, plane, K, weight=0.8, label="wall")
    assert np.std(reg) < np.std(depth)             # flatter
    # non-whitelisted class untouched
    same = regularize_depth_to_plane(depth, mask, plane, K, weight=0.8, label="cable")
    assert np.allclose(same, depth)


def test_fit_plane():
    pts = np.column_stack([rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500),
                           np.full(500, 2.0) + rng.normal(0, 0.001, 500)])
    n, d = fit_plane(pts)
    assert abs(abs(n[2]) - 1.0) < 1e-2 and abs(abs(d) - 2.0) < 1e-2


# ── R.9 fail-safe ───────────────────────────────────────────────────
def test_failsafe_keeps_when_better():
    base = {"seam_error_median": 13.2, "vote_entropy_mean": 0.5, "scale_S_std": 0.1}
    refined = {"seam_error_median": 4.0, "vote_entropy_mean": 0.3, "scale_S_std": 0.05}
    g = compare_and_gate(base, refined)
    assert g.use_anchoring and not g.regressions


def test_failsafe_rejects_when_worse():
    base = {"seam_error_median": 13.2, "vote_entropy_mean": 0.5}
    refined = {"seam_error_median": 4.0, "vote_entropy_mean": 0.9}  # entropy regressed
    g = compare_and_gate(base, refined)
    assert not g.use_anchoring and any(r["metric"] == "vote_entropy_mean" for r in g.regressions)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all phase-R anchoring tests passed")
