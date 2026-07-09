# STAC-Builder — Phase R refinement-fix tests (synthetic).
#
# Covers the load-bearing fixes: R.5 world->camera plane transform, R.2
# vectorized vote + per-region entropy, R.3 onion heatmap, R.4 iterative loop
# with edge recompute, R.6 scale prior in the solve.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.depth_regularization import (  # noqa: E402
    fit_plane, plane_world_to_camera, regularize_depth_to_plane,
)
from phase_r.vote import View, region_entropy, vote_points_batch  # noqa: E402
from phase_r.onion import onion_heatmap  # noqa: E402
from phase_r.residuals import Sim3, WindowEdge, optimize_window_graph  # noqa: E402

rng = np.random.default_rng(1)
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])


# ── R.5: world-frame plane must be moved into the camera frame ──────
def test_plane_world_to_camera_identity():
    n, d = plane_world_to_camera((np.array([0.0, 0.0, 1.0]), -3.0), np.eye(4))
    assert np.allclose(n, [0, 0, 1]) and abs(d + 3.0) < 1e-9


def test_plane_world_to_camera_regularizes_correctly():
    """Camera rotated+translated; plane fit on WORLD points. The old code fed
    the world plane straight to the camera-frame predictor — depth diverged.
    With the transform, the predicted depth matches the true render."""
    from scipy.spatial.transform import Rotation
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_rotvec([0.1, 0.4, -0.2]).as_matrix()
    c2w[:3, 3] = [0.5, -1.0, 2.0]

    # true geometry: a wall plane in world; render its depth for this camera
    n_w = np.array([0.2, 0.1, 0.95]); n_w /= np.linalg.norm(n_w)
    d_w = -4.0
    H, W = 48, 64
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_c = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1],
                       np.ones_like(us, float)], -1)
    R, t = c2w[:3, :3], c2w[:3, 3]
    denom = dirs_c @ (R.T @ n_w)
    depth_true = -(n_w @ t + d_w) / denom
    assert (depth_true > 0).all()

    # world points sampled from that plane (what the store holds)
    pts_c = dirs_c.reshape(-1, 3) * depth_true.reshape(-1, 1)
    pts_w = (R @ pts_c.T).T + t
    plane_fit = fit_plane(pts_w)

    cam_plane = plane_world_to_camera(plane_fit, c2w)
    noisy = depth_true + rng.normal(0, 0.05, depth_true.shape)
    reg = regularize_depth_to_plane(noisy, np.ones((H, W), bool), cam_plane, K,
                                    weight=1.0, label="wall")
    err = np.abs(reg - depth_true)
    assert np.median(err) < 1e-3, float(np.median(err))  # snapped to the true wall


# ── R.2: vectorized vote + region entropy ───────────────────────────
def test_vote_points_batch_matches_pointwise():
    from phase_r.vote import vote_point
    H, W = 480, 640
    m7 = np.zeros((H, W), bool); m7[200:300, 300:400] = True
    m9 = np.zeros((H, W), bool); m9[180:260, 280:360] = True  # overlapping voter
    views = [View(np.eye(4), K, {7: m7, 9: m9}, (W, H)),
             View(np.eye(4), K, {7: m7}, (W, H))]
    pts = np.array([[0.0, 0.06, 4.0], [0.0, -0.5, 4.0], [-0.35, -0.25, 4.0]])
    a_b, e_b, s_b = vote_points_batch(pts, views)
    for i, p in enumerate(pts):
        a, counts, e = vote_point(p, views)
        assert (a if a is not None else -1) == a_b[i]
        assert abs(e - e_b[i]) < 1e-9
    assert (s_b[a_b >= 0] > 0).all()


def test_region_entropy_aggregates_by_cell():
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.3, 0.4],    # cell 0_0_0
                    [1.5, 0.1, 0.1]])                     # cell 1_0_0
    ents = np.array([0.2, 0.4, 0.9])
    reg = region_entropy(pts, ents, cell_m=1.0)
    assert abs(reg["0_0_0"]["mean_entropy"] - 0.3) < 1e-9
    assert reg["0_0_0"]["n_points"] == 2
    assert abs(reg["1_0_0"]["mean_entropy"] - 0.9) < 1e-9


# ── R.3: per-instance heatmap flags the doubled half only ───────────
def test_onion_heatmap_localizes_doubling():
    # wall in the XY plane of its OBB, doubled (onion) only for x > 0
    n = 4000
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    z = rng.normal(0, 0.003, n)
    doubled = x > 0
    z[doubled] += np.where(rng.random(int(doubled.sum())) < 0.5, 0.0, 0.08)
    pts = np.column_stack([x, y, z])
    T = np.eye(4)
    aabb = np.array([-1, 1, -1, 1, -0.1, 0.1])
    hm = onion_heatmap(pts, T, aabb, grid=4, min_cell_points=50)
    cells = np.array(hm["cells"])
    left = cells[:2, :]    # x<0 half (axis_u = 0)
    right = cells[2:, :]   # x>0 half
    assert right.max() > 0.05, right    # doubling detected ~0.08 m
    assert left.max() < 0.02, left      # clean half stays clean


# ── R.4: iterating with edge recompute converges (old loop was a no-op) ──
def test_refine_loop_recompute_converges():
    D1 = Sim3.exp(np.array([0.03, 0.05, 0.0, 0.0, 0.15, -0.05, 0.02]))
    D2 = Sim3.exp(np.array([0.06, 0.0, 0.08, 0.0, -0.1, 0.2, -0.04]))
    # per-window OBBs for one instance seen in 3 windows
    T0 = np.eye(4); aabb = np.array([-0.5, 0.5, -1.0, 1.0, -0.25, 0.25])
    obbs = {0: (T0, aabb)}
    for w, D in ((1, D1), (2, D2)):
        Tw = np.eye(4); Tw[:3, :3] = D.R @ T0[:3, :3]; Tw[:3, 3] = D.apply(T0[:3, 3])
        obbs[w] = (Tw, aabb * D.s)

    from phase_r.residuals import sim3_from_obb_pair

    def edges_of(obbs):
        out = []
        for a in range(3):
            for b in range(a + 1, 3):
                out.append(WindowEdge(a, b, sim3_from_obb_pair(*obbs[a], *obbs[b])))
        return out

    totals = [Sim3.identity()] * 3
    step = None
    for _ in range(2):
        corr, stats = optimize_window_graph(3, edges_of(obbs))
        step = max(M.magnitude() for M in corr)
        totals = [c.compose(t) for c, t in zip(corr, totals)]
        new = {}
        for w, (T, ab) in obbs.items():
            M = corr[w]
            T2 = T.copy(); T2[:3, :3] = M.R @ T2[:3, :3]; T2[:3, 3] = M.apply(T2[:3, 3])
            new[w] = (T2, ab * M.s)
        obbs = new
        if step < 1e-3:
            break
    # second pass sees already-corrected OBBs -> its correction is ~identity
    assert step < 1e-3, step
    # and the composed totals undo the original drift
    assert totals[1].compose(D1).magnitude() < 0.05
    assert totals[2].compose(D2).magnitude() < 0.05


# ── R.6: scale prior pulls the solve toward the DA3 prior ───────────
def test_scale_prior_regularizes_scale():
    # one weak edge suggesting a 20% rescale of window 1
    M = Sim3(1.2, np.eye(3), np.zeros(3))
    edges = [WindowEdge(0, 1, M, weight=0.1)]
    # min_window_edges=1: this test exercises the PRIOR math on a single edge;
    # the production underdetermination guard (default 2) would lock the window
    corr_free, _ = optimize_window_graph(2, edges, min_window_edges=1)
    corr_prior, _ = optimize_window_graph(2, edges, scale_priors={1: 1.0},
                                          scale_prior_weight=50.0,
                                          min_window_edges=1)
    # without the prior the solver rescales; with it the scale stays ~1
    assert abs(np.log(corr_prior[1].s)) < abs(np.log(corr_free[1].s))
    assert abs(corr_prior[1].s - 1.0) < 0.03


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all phase-R refinement tests passed")


# ── canonical seg-OBB ingestion + robust trimmed fit ────────────────
def test_trimmed_obb_rejects_depth_noise():
    from phase_r.geometry import fit_gravity_aligned_obb
    # thin floor slab + 0.5% far outliers along Y (depth noise); the 1%-trim
    # percentile must land inside the slab, not in the outlier tail
    n = 5000
    pts = np.column_stack([rng.uniform(-3, 3, n), rng.normal(0, 0.03, n),
                           rng.uniform(-3, 3, n)])
    out_idx = rng.choice(n, n // 200, replace=False)
    pts[out_idx, 1] += rng.uniform(0.3, 0.5, len(out_idx))
    T0, aabb0, _ = fit_gravity_aligned_obb(pts)                    # raw min/max
    T1, aabb1, _ = fit_gravity_aligned_obb(pts, trim_percent=1.0)  # robust
    thick0 = aabb0[3] - aabb0[2]
    thick1 = aabb1[3] - aabb1[2]
    assert thick0 > 0.3, thick0        # raw fit inflated by outliers
    assert thick1 < 0.2, thick1        # trimmed fit ~ true ±3σ thickness


def test_seg_obb_display_roundtrip(tmp_path):
    """Seg OBB (display frame) -> raw store frame -> SpatialTools display
    read must reproduce the original box exactly (the assistant's highlight
    coincides with the box the viewer already draws)."""
    import json
    from scipy.spatial.transform import Rotation
    from phase_r.build_instances import load_seg_display_obbs
    from phase_r.instance_store import InstanceStore
    from phase5_qa.tools import SpatialTools, load_display_transform

    s = 1.0
    R = Rotation.from_rotvec([0, 0.4, 0]).as_matrix()
    t = np.array([0.0, 2.448, 0.0])
    np.savez(tmp_path / "floor_transform.npz", s=np.float64(s), R=R, t=t)
    center_d = [1.2, 0.5, -3.0]
    half_d = [0.4, 1.1, 0.05]
    rot_d = Rotation.from_rotvec([0, 0.2, 0]).as_matrix()
    (tmp_path / "segmentation_result.json").write_text(json.dumps({
        "instances": [{"id": 7, "instance_id": 7, "label": "door",
                       "obb": {"center": center_d, "half_extents": half_d,
                               "rotation": rot_d.tolist()}}]}))

    raw = load_seg_display_obbs(tmp_path)
    assert 7 in raw
    T_raw, aabb_raw = raw[7]

    st = InstanceStore(tmp_path / "scene_r.db")
    st.upsert_instance(7, "door", n_views=3)
    st.set_obb(7, T_raw, aabb_raw, T_raw[:3, 3])
    tools = SpatialTools(st, display_transform=load_display_transform(st.path))
    T2, aabb2, pos2 = tools._obb(7)
    assert np.allclose(pos2, center_d, atol=1e-5), pos2
    h2 = [(aabb2[1] - aabb2[0]) / 2, (aabb2[3] - aabb2[2]) / 2, (aabb2[5] - aabb2[4]) / 2]
    assert np.allclose(h2, half_d, atol=1e-5), h2
    assert np.allclose(T2[:3, :3], rot_d, atol=1e-5)
    st.close()
