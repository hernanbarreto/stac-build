"""F5's torch bundle adjustment (precision/ba_torch.py): analytic Jacobians against finite differences, a
synthetic scene recovered from a perturbed start under both prior models, bit-identical twice, and —
where pycolmap is present — the same problem solved by Ceres through precision.refine.run_rung agrees
with it to the solver tolerance. CPU only (device 'cpu'); the GPU path is the same code."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from precision import ba_torch as BA  # noqa: E402
from precision.refine import RelativePrior  # noqa: E402

TB = SimpleNamespace(function_tolerance=1e-6, parameter_tolerance=1e-8, gradient_tolerance=1e-8, min_relative_decrease=1e-3,
                     lambda_init=1e-4, lm_diag_min=1e-6, lm_diag_max=1e32)       # config.yaml refine.torch_ba (Ceres')
CFG = SimpleNamespace(huber_px=2.0, max_iterations=200, ceres_threads=1, seed=0, torch_ba=TB)
CAM = [400.0, 402.0, 231.5, 415.5, 0.02, -0.01, 0.0005, -0.0003]
WH = (464, 832)


def _rot(axis, deg):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(deg)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _scene(n_cam=10, n_pts=300, noise_px=0.2, seed=0):
    """Cameras on a gentle arc 3 m from a wall of points, looking at it; OPENCV projections + noise."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(-2, 2, n_pts), rng.uniform(-1.2, 1.2, n_pts), rng.uniform(2.5, 4.0, n_pts)])
    w2c = []
    for k in range(n_cam):
        c2w = np.eye(4)
        c2w[:3, :3] = _rot([0, 1, 0], -12 + 24 * k / (n_cam - 1))
        c2w[:3, 3] = [-1.5 + 3.0 * k / (n_cam - 1), 0.05 * math.sin(k), 0.0]
        w2c.append(np.linalg.inv(c2w))
    w2c = np.asarray(w2c)
    import cv2
    groups, X = {}, {}
    for j in range(n_pts):
        obs = []
        for i in range(n_cam):
            Xc = w2c[i, :3, :3] @ pts[j] + w2c[i, :3, 3]
            if Xc[2] <= 0.3:
                continue
            rv, _ = cv2.Rodrigues(w2c[i, :3, :3])
            uv, _ = cv2.projectPoints(pts[j].reshape(1, 1, 3), rv, w2c[i, :3, 3], BA_K(CAM), np.asarray(CAM[4:8]))
            u, v = uv.reshape(2)
            if 0 <= u < WH[0] and 0 <= v < WH[1]:
                obs.append((i, np.array([u, v]) + rng.normal(0, noise_px, 2)))
        if len(obs) >= 2:
            groups[j] = obs; X[j] = pts[j].copy()
    return w2c, pts, groups, X


def BA_K(p):
    return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])


def _perturb(w2c, X, rng, t_m=0.03, rot_deg=0.6, x_m=0.04):
    w = w2c.copy()
    for i in range(len(w)):
        if i == 0:
            continue
        w[i, :3, :3] = _rot(rng.normal(size=3), rng.normal(0, rot_deg)) @ w[i, :3, :3]
        w[i, :3, 3] += rng.normal(0, t_m, 3)
    Xp = {t: x + rng.normal(0, x_m, 3) for t, x in X.items()}
    return w, Xp


def test_the_analytic_jacobians_match_finite_differences():
    rng = np.random.default_rng(1)
    Xc = torch.tensor(np.column_stack([rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(1, 4, 50)]), dtype=torch.float64)
    cam = torch.tensor(np.tile(CAM, (50, 1)), dtype=torch.float64)
    free = (0, 1, 2, 3, 4, 5, 6, 7)
    uv, J_X, J_cam, front = BA.project_jac(Xc, cam, free)
    assert bool(front.all())
    h = 1e-6
    for a in range(3):
        d = torch.zeros_like(Xc); d[:, a] = h
        num = (BA.project_jac(Xc + d, cam, free)[0] - BA.project_jac(Xc - d, cam, free)[0]) / (2 * h)
        assert torch.allclose(num, J_X[:, :, a], atol=1e-5, rtol=1e-6)
    for k, a in enumerate(free):
        d = torch.zeros_like(cam); d[:, a] = h
        num = (BA.project_jac(Xc, cam + d, free)[0] - BA.project_jac(Xc, cam - d, free)[0]) / (2 * h)
        assert torch.allclose(num, J_cam[:, :, k], atol=1e-5, rtol=1e-6)


def test_the_segment_sum_and_the_pairs_are_exact_and_ordered():
    seg = torch.tensor([0, 0, 1, 3, 3, 3])
    vals = torch.tensor([1.0, 2.0, 5.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert BA.segment_sum(vals, seg, 5).tolist() == [3.0, 5.0, 0.0, 3.0, 0.0]
    a, b = BA.pair_indices(torch.tensor([2, 1]))
    assert a.tolist() == [0, 0, 1, 1, 2] and b.tolist() == [0, 1, 0, 1, 2]
    t = torch.zeros(4, dtype=torch.float64)
    BA.sorted_scatter_add(t, torch.tensor([2, 0, 2, 3]), torch.tensor([1.0, 1.0, 2.0, 4.0], dtype=torch.float64))
    assert t.tolist() == [1.0, 0.0, 3.0, 4.0]


@pytest.mark.parametrize("model", ["absolute", "relative"])
def test_a_perturbed_scene_is_recovered_and_the_solve_is_bit_identical(model):
    w2c, pts, groups, X = _scene()
    rng = np.random.default_rng(7)
    w0, X0 = _perturb(w2c, X, rng)
    if model == "absolute":
        kw = dict(sigmas=np.full(len(w2c), 0.5), prior_w2c=w0)
    else:
        kw = dict(sigmas=None, rel_prior=RelativePrior(w2c, sigma_rel=0.02, sigma_lat=0.05, sigma_rot=math.radians(0.5)))
    r1 = BA.run_rung_torch("R0", w0, CAM, WH, groups, X0, cfg=CFG, device="cpu", **kw)
    r2 = BA.run_rung_torch("R0", w0, CAM, WH, groups, X0, cfg=CFG, device="cpu", **kw)
    assert r1.termination == "CONVERGENCE" and r1.final_cost < r1.initial_cost * 0.05
    assert np.array_equal(r1.w2c, r2.w2c) and all(np.array_equal(r1.X[t], r2.X[t]) for t in r1.X)
    # the recovered poses against the truth, up to the gauge: relative poses keyframe 0 → k
    def rel(W):
        return np.stack([W[k] @ np.linalg.inv(W[0]) for k in range(len(W))])
    err_t = np.linalg.norm(rel(r1.w2c)[:, :3, 3] - rel(w2c)[:, :3, 3], axis=1)
    assert np.median(err_t) < 0.01, err_t
    assert r1.fit_rms_px < 0.5


def test_the_camera_rungs_recover_an_injected_focal_error():
    w2c, pts, groups, X = _scene(noise_px=0.1)
    cam_wrong = list(CAM); cam_wrong[0] *= 1.02; cam_wrong[1] *= 1.02
    r = BA.run_rung_torch("R1", w2c, cam_wrong, WH, groups, X, sigmas=np.full(len(w2c), 0.05), cfg=CFG,
                          prior_w2c=w2c, device="cpu")
    fx, fy = r.params_by_block[0][:2]
    assert abs(fx - CAM[0]) < 1.5 and abs(fy - CAM[1]) < 1.5 and r.termination == "CONVERGENCE"


def test_ceres_and_torch_agree_on_the_same_problem():
    pycolmap = pytest.importorskip("pycolmap")
    from precision.refine import run_rung
    w2c, pts, groups, X = _scene()
    rng = np.random.default_rng(3)
    w0, X0 = _perturb(w2c, X, rng, t_m=0.01, rot_deg=0.2, x_m=0.02)
    # 2 cm priors: the gauge (the metre above all — reprojection alone is scale-free) is pinned, so the two
    # solvers' solutions are comparable point for point; with soft priors they differed by a 0.13 % scale
    sig = np.full(len(w2c), 0.02)
    rc = run_rung("R0", w0, CAM, WH, groups, X0, sig, CFG, prior_w2c=w0)
    rt = BA.run_rung_torch("R0", w0, CAM, WH, groups, X0, sig, CFG, prior_w2c=w0, device="cpu")
    assert abs(rt.final_cost - rc.final_cost) <= 0.02 * rc.final_cost + 1e-9, (rt.final_cost, rc.final_cost)
    # the two solvers reach stationary points of the same robust cost by different paths (Ceres' corrector
    # vs IRLS); with 20 cm priors the gauge is soft, so the comparison is the SHAPE: relative poses to kf 0
    def rel(W):
        return np.stack([W[k] @ np.linalg.inv(W[0]) for k in range(len(W))])
    d_rel = np.linalg.norm(rel(rt.w2c)[:, :3, 3] - rel(rc.w2c)[:, :3, 3], axis=1)
    d_abs = np.linalg.norm(rt.w2c[:, :3, 3] - rc.w2c[:, :3, 3], axis=1)
    assert np.max(d_rel) < 2e-3 and np.max(d_abs) < 5e-3, (d_rel, d_abs)
    assert abs(rt.fit_rms_px - rc.fit_rms_px) < 0.02, (rt.fit_rms_px, rc.fit_rms_px)
