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


def _scene(n_cam=10, n_pts=300, noise_px=0.2, seed=0, curve=False):
    """Cameras on a gentle arc 3 m from a wall of points, looking at it; OPENCV projections + noise.
    ``curve``: the centres leave the line (y and z swing) — COLMAP's Sim3 alignment to the priors needs
    non-collinear centres, or it drops every prior and refine.run_rung refuses the solve."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(-2, 2, n_pts), rng.uniform(-1.2, 1.2, n_pts), rng.uniform(2.5, 4.0, n_pts)])
    w2c = []
    for k in range(n_cam):
        c2w = np.eye(4)
        c2w[:3, :3] = _rot([0, 1, 0], -12 + 24 * k / (n_cam - 1))
        c2w[:3, 3] = ([-1.5 + 3.0 * k / (n_cam - 1), 0.4 * math.sin(1.3 * k), 0.3 * math.cos(0.7 * k)] if curve
                      else [-1.5 + 3.0 * k / (n_cam - 1), 0.05 * math.sin(k), 0.0])
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


def _umeyama(src, dst):
    """src aligned to dst by the least-squares similarity (scale, rotation, translation)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / (S ** 2).sum(1).mean()
    return s * (src - mu_s) @ R.T + mu_d


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


def test_the_full_residual_jacobian_matches_the_step_apply_makes():
    """The pose / landmark / camera Jacobians of _Problem.residuals against finite differences of the
    SAME perturbation _Problem.apply makes (R' = exp(δω) R, t' = t + δt, X' = X + δ, cam' = cam + δ) —
    the convention the LM step relies on (ρ ≈ 1 near a quadratic optimum)."""
    w2c, pts, groups, X = _scene(n_cam=4, n_pts=60)
    tracks = [t for t in X if t in groups and len(groups[t]) >= 2]
    cam_idx = np.concatenate([np.array([i for i, _ in groups[t]]) for t in tracks])
    pt_idx = np.concatenate([np.full(len(groups[t]), k) for k, t in enumerate(tracks)])
    uv = np.concatenate([np.array([p for _, p in groups[t]]) for t in tracks])
    X0 = np.array([X[t] for t in tracks])
    prob = BA._Problem(w2c, [list(CAM)], np.zeros(len(w2c), int), cam_idx, pt_idx, uv, X0, 2.0, (0, 1, 2, 3),
                       device="cpu", sigmas=np.full(len(w2c), 0.5), prior_centres=-np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3]),
                       torch_ba=TB)
    r0, front, J_pose, J_pt, J_cam = prob.residuals()
    assert bool(front.all())
    h = 1e-6
    n_pose = 6 * len(w2c)
    for col in range(6):                                       # one pose parameter at a time, every camera
        dc = torch.zeros(prob.n_c, dtype=torch.float64)
        dc[col:n_pose:6] = h
        dp = torch.zeros_like(prob.X)
        R1, t1, cam1, X1 = prob.apply(dc, dp)
        R2, t2, cam2, X2 = prob.apply(-dc, dp)
        num = (prob.residuals(R1, t1, cam1, X1, with_jac=False)[0] - prob.residuals(R2, t2, cam2, X2, with_jac=False)[0]) / (2 * h)
        assert torch.allclose(num, J_pose[:, :, col], atol=1e-4, rtol=1e-5), col
    for a in range(3):                                         # the landmarks
        dp = torch.zeros_like(prob.X); dp[:, a] = h
        dc = torch.zeros(prob.n_c, dtype=torch.float64)
        R1, t1, cam1, X1 = prob.apply(dc, dp)
        R2, t2, cam2, X2 = prob.apply(dc, -dp)
        num = (prob.residuals(R1, t1, cam1, X1, with_jac=False)[0] - prob.residuals(R2, t2, cam2, X2, with_jac=False)[0]) / (2 * h)
        assert torch.allclose(num, J_pt[:, :, a], atol=1e-4, rtol=1e-5), a
    for k in range(4):                                         # the free camera parameters (fx fy cx cy)
        dc = torch.zeros(prob.n_c, dtype=torch.float64); dc[n_pose + k] = h
        dp = torch.zeros_like(prob.X)
        R1, t1, cam1, X1 = prob.apply(dc, dp)
        R2, t2, cam2, X2 = prob.apply(-dc, dp)
        num = (prob.residuals(R1, t1, cam1, X1, with_jac=False)[0] - prob.residuals(R2, t2, cam2, X2, with_jac=False)[0]) / (2 * h)
        assert torch.allclose(num, J_cam[:, :, k], atol=1e-4, rtol=1e-5), k
    # the prior rows too
    e0, Jp = prob.prior_terms(prob.R, prob.t)
    for col in range(6):
        dc = torch.zeros(prob.n_c, dtype=torch.float64); dc[col:n_pose:6] = h
        R1, t1, _, _ = prob.apply(dc, torch.zeros_like(prob.X))
        R2, t2, _, _ = prob.apply(-dc, torch.zeros_like(prob.X))
        num = (prob.prior_terms(R1, t1)[0] - prob.prior_terms(R2, t2)[0]) / (2 * h)
        assert torch.allclose(num, (Jp[:, col:n_pose:6]).sum(1), atol=1e-4, rtol=1e-5), col


def test_the_segment_sum_and_the_pairs_are_exact_and_ordered():
    seg = torch.tensor([0, 0, 1, 3, 3, 3])
    vals = torch.tensor([1.0, 2.0, 5.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert BA.segment_sum(vals, seg, 5).tolist() == [3.0, 5.0, 0.0, 3.0, 0.0]
    a, b = BA.pair_indices(torch.tensor([2, 1]))
    assert a.tolist() == [0, 0, 1, 1, 2] and b.tolist() == [0, 1, 0, 1, 2]
    t = torch.zeros(4, dtype=torch.float64)
    BA.sorted_scatter_add(t, torch.tensor([2, 0, 2, 3]), torch.tensor([1.0, 1.0, 2.0, 4.0], dtype=torch.float64))
    assert t.tolist() == [1.0, 0.0, 3.0, 4.0]


def test_a_perturbed_scene_is_recovered_and_the_solve_is_bit_identical():
    w2c, pts, groups, X = _scene()
    rng = np.random.default_rng(7)
    w0, X0 = _perturb(w2c, X, rng)
    kw = dict(sigmas=np.full(len(w2c), 0.5), prior_w2c=w0)
    r1 = BA.run_rung_torch("R0", w0, CAM, WH, groups, X0, cfg=CFG, device="cpu", **kw)
    r2 = BA.run_rung_torch("R0", w0, CAM, WH, groups, X0, cfg=CFG, device="cpu", **kw)
    assert r1.termination == "CONVERGENCE" and r1.final_cost < r1.initial_cost * 0.05
    assert np.array_equal(r1.w2c, r2.w2c) and all(np.array_equal(r1.X[t], r2.X[t]) for t in r1.X)
    # the recovered poses against the truth UP TO THE GAUGE (reprojection alone is similarity-free and
    # 50 cm priors pin it loosely): the camera centres after a similarity alignment, the relative
    # rotations keyframe 0 → k as they are
    def centres(W):
        return -np.einsum("nji,nj->ni", W[:, :3, :3], W[:, :3, 3])
    err_c = np.linalg.norm(_umeyama(centres(r1.w2c), centres(w2c)) - centres(w2c), axis=1)
    assert np.median(err_c) < 0.005, err_c
    rel_rot = [r1.w2c[k, :3, :3] @ r1.w2c[0, :3, :3].T @ (w2c[k, :3, :3] @ w2c[0, :3, :3].T).T for k in range(len(w2c))]
    err_r = np.degrees([math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))) for R in rel_rot])
    assert np.median(err_r) < 0.05, err_r
    assert r1.fit_rms_px < 0.5


def _ceres_or_skip():
    """The Ceres path runs through pycolmap 4.x (the chain's env mapanything); an older pycolmap (env da3
    has 3.13) lacks the API precision.refine.run_rung uses — the comparison is skipped there, named."""
    pycolmap = pytest.importorskip("pycolmap")
    if not hasattr(pycolmap.Camera, "create_from_model_name"):
        pytest.skip(f"pycolmap {pycolmap.__version__}: no Camera.create_from_model_name — the Ceres path needs 4.x")
    return pycolmap


def test_the_camera_rungs_recover_an_injected_focal_error():
    """R1 from a 2 % focal error: the solve lands at the noisy problem's own optimum — the bar is
    Ceres' answer on the same problem (within the solver tolerance), not the truth, which the 0.1 px
    noise and the frontal scene place ~2 px away for both solvers."""
    w2c, pts, groups, X = _scene(noise_px=0.1, curve=True)
    cam_wrong = list(CAM); cam_wrong[0] *= 1.02; cam_wrong[1] *= 1.02
    sig = np.full(len(w2c), 0.05)
    r = BA.run_rung_torch("R1", w2c, cam_wrong, WH, groups, X, sigmas=sig, cfg=CFG, prior_w2c=w2c, device="cpu")
    fx, fy = r.params_by_block[0][:2]
    assert r.termination == "CONVERGENCE", r.termination
    _ceres_or_skip()
    from precision.refine import run_rung
    rc = run_rung("R1", w2c, cam_wrong, WH, groups, X, sig, CFG, prior_w2c=w2c)
    fxc, fyc = rc.params_by_block[0][:2]
    assert abs(fx - fxc) < 0.1 and abs(fy - fyc) < 0.1, ((fx, fy), (fxc, fyc), CAM[:2])
    assert abs(r.fit_rms_px - rc.fit_rms_px) < 0.01, (r.fit_rms_px, rc.fit_rms_px)


def test_ceres_and_torch_agree_on_the_same_problem():
    _ceres_or_skip()
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


@pytest.mark.parametrize("damping", ["hessian", "schur"])
@pytest.mark.parametrize("cam_free,blocks", [((), 1), ((0, 1, 2, 3), 1), ((0, 1), 3)])
def test_the_step_is_the_dense_damped_system(cam_free, blocks, damping, monkeypatch):
    """_Problem.step(λ) against the DENSE solve of the same linearization — the whole Schur reduction
    (the multi-block camera coupling included: until 2026-11-01 a pose's coupling to ANOTHER block's
    focal landed in its own block's columns and station's F5 R3 — removed 2026-11-10 — ran out of iterations) and the
    back-substitution, to 1e-9. "hessian": Ceres' system (JᵀJ + λ·D) δ = −Jᵀr, D = diag(JᵀJ) clamped
    to [lm_diag_min, lm_diag_max] for every parameter. "schur" (the default): the landmarks damped the
    same, the cameras by λ·diag of their damped Schur complement."""
    monkeypatch.setattr(BA, "CAMERA_DAMPING", damping)
    w2c, pts, groups, X = _scene(n_cam=6, n_pts=80, noise_px=0.3)
    tracks = [t for t in X if t in groups and len(groups[t]) >= 2]
    cam_idx = np.concatenate([np.array([i for i, _ in groups[t]]) for t in tracks])
    pt_idx = np.concatenate([np.full(len(groups[t]), k) for k, t in enumerate(tracks)])
    uv = np.concatenate([np.array([p for _, p in groups[t]]) for t in tracks])
    X0 = np.array([X[t] for t in tracks]) + np.random.default_rng(1).normal(0, 0.02, (len(tracks), 3))
    block_of = (np.arange(len(w2c)) * blocks // len(w2c)).astype(int)
    prob = BA._Problem(w2c, [list(CAM)] * blocks, block_of, cam_idx, pt_idx, uv, X0, 1.0, cam_free,
                       device="cpu", sigmas=np.full(len(w2c), 0.05),
                       prior_centres=-np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3]) + 0.01, torch_ba=TB)
    L = prob.linearize()
    M, P, nc = int(L["r_w"].shape[0]), int(prob.P), int(prob.n_c)
    cols = torch.cat([prob.pose_idx, prob.cam_cols], 1) if prob.nf else prob.pose_idx      # (N, nco)
    J = torch.zeros(2 * M + int(L["e_p"].shape[0]), nc + 3 * P, dtype=torch.float64)
    r = torch.zeros(J.shape[0], dtype=torch.float64)
    rows = torch.arange(2 * M).reshape(M, 2)
    for m in range(M):
        J[rows[m][:, None], cols[prob.cam_idx[m]][None, :]] = L["J_c"][m]
        J[rows[m][:, None], (nc + 3 * prob.pt_idx[m] + torch.arange(3))[None, :]] = L["J_pt"][m]
        r[rows[m]] = L["r_w"][m]
    if L["e_p"].shape[0]:
        J[2 * M:, :nc] = L["J_p"]
        r[2 * M:] = L["e_p"]
    keep = torch.cat([prob.free, torch.ones(3 * P, dtype=torch.bool)])
    Jk = J[:, keep]
    A = Jk.T @ Jk
    D = torch.diagonal(A).clamp(TB.lm_diag_min, TB.lm_diag_max)
    nf_ = int(prob.free.sum())
    for lam in (1e-4, 1e-1, 3.0):
        if damping == "hessian":
            x = torch.linalg.solve(A + lam * torch.diag(D), -(Jk.T @ r))
        else:
            Ad = A + lam * torch.diag(torch.cat([torch.zeros(nf_, dtype=torch.float64), D[nf_:]]))
            Acc, Acp, App = Ad[:nf_, :nf_], Ad[:nf_, nf_:], Ad[nf_:, nf_:]
            g_all = Jk.T @ r
            Sd = Acc - Acp @ torch.linalg.solve(App, Acp.T)
            gr = g_all[:nf_] - Acp @ torch.linalg.solve(App, g_all[nf_:])
            Sl = Sd + lam * torch.diag(torch.diagonal(Sd).clamp(TB.lm_diag_min, TB.lm_diag_max))
            xc = torch.linalg.solve(Sl, -gr)
            xp = torch.linalg.solve(App, -g_all[nf_:] - Acp.T @ xc)
            x = torch.cat([xc, xp])
        (dc, dp), _cost, _pred, why = prob.step(lam)
        assert why == ""
        got = torch.cat([dc[prob.free], dp.reshape(-1)])
        assert torch.allclose(got, x, rtol=1e-9, atol=1e-12 * float(x.abs().max())), lam


def _flat_problem(n_cam=8, n_pts=300, seed=0):
    w2c, pts, groups, X = _scene(n_cam=n_cam, n_pts=n_pts, noise_px=0.3, seed=seed)
    tracks = [t for t in X if t in groups and len(groups[t]) >= 2]
    cam_idx = np.concatenate([np.array([i for i, _ in groups[t]]) for t in tracks])
    pt_idx = np.concatenate([np.full(len(groups[t]), k) for k, t in enumerate(tracks)])
    uv = np.concatenate([np.array([p for _, p in groups[t]]) for t in tracks])
    X0 = np.array([X[t] for t in tracks]) + np.random.default_rng(seed + 1).normal(0, 0.02, (len(tracks), 3))
    return w2c, cam_idx, pt_idx, uv, X0


@pytest.mark.parametrize("cam_free,blocks", [((), 1), ((0, 1, 2, 3, 4, 5, 6, 7), 1), ((0, 1), 3)])
def test_the_chunked_problem_is_the_one_piece_problem(cam_free, blocks):
    """THE CARD'S MEMORY (2026-11-10, pccr 2026-08-24: 101.6 M residual blocks did not fit the 80 GB card in one
    piece): the problem in chunks of landmarks gives the one-piece cost, linearisation and LM step to the rounding
    of the summation order — and with ONE chunk, the same bits."""
    w2c, cam_idx, pt_idx, uv, X0 = _flat_problem()
    block_of = (np.arange(len(w2c)) * blocks // len(w2c)).astype(int)
    kw = dict(device="cpu", sigmas=np.full(len(w2c), 0.05),
              prior_centres=-np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3]) + 0.01, torch_ba=TB)
    args = (w2c, [list(CAM)] * blocks, block_of, cam_idx, pt_idx, uv, X0, 1.0, cam_free)
    one = BA._Problem(*args, **kw)
    many = BA._ChunkedProblem(*args, chunk_obs=97, **kw)
    single = BA._ChunkedProblem(*args, chunk_obs=10 ** 9, **kw)
    assert len(many.chunks) > 5 and len(single.chunks) == 1
    r1, f1 = one.residuals(with_jac=False)
    r2, f2 = many.residuals(with_jac=False)
    assert torch.equal(r1, r2) and torch.equal(f1, f2)
    assert abs(one.total_cost() - many.total_cost()) <= 1e-12 * one.total_cost()
    L1, L2 = one.linearize(), many.linearize()
    for k in ("Hpp", "gp", "H", "g"):
        assert torch.allclose(L1[k], L2[k], rtol=1e-10, atol=1e-10 * float(L1[k].abs().max())), k
    for lam in (1e-4, 1e-1, 3.0):
        (dc1, dp1), c1, p1, w1 = one.step(lam)
        (dc2, dp2), c2, p2, w2 = many.step(lam)
        (dc3, dp3), c3, p3, w3 = single.step(lam)
        assert w1 == w2 == w3 == ""
        # the summation order's rounding (H agrees to 4e-15), amplified by the damped system's conditioning:
        # measured ≤ 2e-10 of the step's norm at λ 1e-4, ≤ 2e-13 at 0.1
        assert float((dc1 - dc2).norm()) <= 1e-8 * float(dc1.norm()), lam
        assert float((dp1 - dp2).norm()) <= 1e-8 * float(dp1.norm()), lam
        assert abs(p1 - p2) <= 1e-9 * abs(p1) and abs(c1 - c2) <= 1e-12 * c1
        assert torch.equal(dc1, dc3) and torch.equal(dp1, dp3) and p1 == p3 and c1 == c3, lam


def test_a_rung_solved_in_chunks_lands_where_the_one_piece_solve_does(monkeypatch):
    """run_rung_torch above CHUNKED_ABOVE_OBS: the same rung, the same optimum (the LM path may differ by the
    summation order's rounding); deterministic run to run."""
    w2c, pts, groups, X = _scene()
    rng = np.random.default_rng(7)
    w0, X0 = _perturb(w2c, X, rng)
    kw = dict(sigmas=np.full(len(w2c), 0.5), prior_w2c=w0)
    r1 = BA.run_rung_torch("R1", w0, CAM, WH, groups, X0, cfg=CFG, device="cpu", **kw)
    monkeypatch.setattr(BA, "CHUNKED_ABOVE_OBS", 0)
    monkeypatch.setattr(BA, "CHUNK_OBS", 211)
    r2 = BA.run_rung_torch("R1", w0, CAM, WH, groups, X0, cfg=CFG, device="cpu", **kw)
    r3 = BA.run_rung_torch("R1", w0, CAM, WH, groups, X0, cfg=CFG, device="cpu", **kw)
    assert r1.termination == r2.termination == "CONVERGENCE"
    assert np.array_equal(r2.w2c, r3.w2c) and all(np.array_equal(r2.X[t], r3.X[t]) for t in r2.X)
    assert abs(r1.final_cost - r2.final_cost) <= 1e-6 * r1.final_cost, (r1.final_cost, r2.final_cost)
    assert np.max(np.abs(r1.w2c - r2.w2c)) < 1e-6 and np.max(np.abs(np.subtract(r1.params_by_block, r2.params_by_block))) < 1e-4
    assert abs(r1.fit_rms_px - r2.fit_rms_px) < 1e-6
