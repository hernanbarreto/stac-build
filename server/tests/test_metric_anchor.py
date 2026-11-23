"""F5's metric anchor (USER 2026-11-22, MetricTrail's concept: "usamos el concepto"): the depth rows' Jacobians
against finite differences, the anchor holding the depth where the reprojections are silent, its rows kept out
of the fit statistic, and its construction (the nearest keyframe, the per-keyframe scale, the measured weight).
CPU only."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from precision import ba_torch as BA  # noqa: E402
from precision import refine as RF  # noqa: E402

TB = SimpleNamespace(function_tolerance=1e-6, parameter_tolerance=1e-8, gradient_tolerance=1e-8, min_relative_decrease=1e-3,
                     lambda_init=1e-4, lm_diag_min=1e-6, lm_diag_max=1e32)
CFG = SimpleNamespace(huber_px=2.0, max_iterations=200, ceres_threads=1, seed=0, torch_ba=TB)
CAM = [400.0, 402.0, 231.5, 415.5, 0.0, 0.0, 0.0, 0.0]
WH = (464, 832)


def _rot(axis, deg):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(deg)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _project(X, w2c):
    Xc = w2c[:3, :3] @ X + w2c[:3, 3]
    return np.array([CAM[0] * Xc[0] / Xc[2] + CAM[2], CAM[1] * Xc[1] / Xc[2] + CAM[3]]), Xc[2]


def _scene(n_cam=6, n_pts=200, baseline=0.6, noise_px=0.2, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(-1.5, 1.5, n_pts), rng.uniform(-1.0, 1.0, n_pts), rng.uniform(2.5, 4.5, n_pts)])
    w2c = []
    for k in range(n_cam):
        c2w = np.eye(4)
        c2w[:3, :3] = _rot([0, 1, 0], -4 + 8 * k / (n_cam - 1))
        c2w[:3, 3] = [-baseline / 2 + baseline * k / (n_cam - 1), 0.02 * math.sin(k), 0.0]
        w2c.append(np.linalg.inv(c2w))
    w2c = np.asarray(w2c)
    groups, X = {}, {}
    for j in range(n_pts):
        obs = []
        for i in range(n_cam):
            uv, z = _project(pts[j], w2c[i])
            if z > 0.3 and 0 <= uv[0] < WH[0] and 0 <= uv[1] < WH[1]:
                obs.append((i, uv + rng.normal(0, noise_px, 2)))
        if len(obs) >= 2:
            groups[j] = obs; X[j] = pts[j].copy()
    return w2c, pts, groups, X


def _depth_problem(w2c, groups, X, kappa=3.0, prior_scale=1.0, cubes=None, cam_free=(), blocks=1, huber=None,
                   prior_noise=0.0, seed=0):
    """A _Problem of the scene's reprojections plus one depth row per landmark (its first keyframe), the prior
    = the TRUE depth × prior_scale; ``cubes`` (keyframe → cube) makes each cube's log-scale an unknown."""
    tracks = list(X)
    pos = {t: k for k, t in enumerate(tracks)}
    cam_idx, pt_idx, uv = [], [], []
    for t in tracks:
        for i, p in groups[t]:
            cam_idx.append(i); pt_idx.append(pos[t]); uv.append(p)
    n_rep = len(uv)
    d_log = []
    rng_p = np.random.default_rng(seed)
    for k, t in enumerate(tracks):
        i0, p0 = groups[t][k % len(groups[t])]
        z = (w2c[i0, :3, :3] @ X[t] + w2c[i0, :3, 3])[2]
        cam_idx.append(i0); pt_idx.append(pos[t]); uv.append(p0)
        d_log.append(math.log(z * prior_scale) + (rng_p.normal(0, prior_noise) if prior_noise else 0.0))
    is_d = np.r_[np.zeros(n_rep, bool), np.ones(len(tracks), bool)]
    dl = np.r_[np.zeros(n_rep), np.asarray(d_log)]
    X0 = np.asarray([X[t] for t in tracks])
    depth = (is_d, dl, kappa)
    if cubes is not None:
        cubes = np.asarray(cubes, np.int64)
        depth = (is_d, dl, kappa, cubes, np.linspace(-0.03, 0.04, int(cubes.max()) + 1))
    block_of = (np.arange(len(w2c)) * blocks // len(w2c)).astype(int)
    return BA._Problem(w2c, [CAM] * blocks, block_of, np.asarray(cam_idx), np.asarray(pt_idx), np.asarray(uv),
                       X0, CFG.huber_px if huber is None else huber, cam_free, device="cpu",
                       sigmas=np.full(len(w2c), 0.5),
                       prior_centres=-np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3]) + 0.01, torch_ba=TB,
                       depth=depth)


def test_the_depth_rows_jacobians_match_finite_differences():
    w2c, pts, groups, X = _scene()
    rng = np.random.default_rng(3)
    Xp = {t: x * (1 + rng.normal(0, 0.05)) for t, x in X.items()}        # depths off the prior
    prob = _depth_problem(w2c, groups, Xp)
    r, front, J_pose, J_pt, J_cam = prob.residuals()
    dm = prob.is_depth
    assert bool(dm.any()) and bool(front[dm].all())
    h = 1e-6
    for a in range(3):
        d = torch.zeros_like(prob.X); d[:, a] = h
        num = (prob.residuals(X=prob.X + d, with_jac=False)[0] - prob.residuals(X=prob.X - d, with_jac=False)[0]) / (2 * h)
        assert torch.allclose(num[dm], J_pt[dm][:, :, a], atol=1e-6, rtol=1e-6)
    for a in range(6):
        dw = torch.zeros(prob.N, 6, dtype=torch.float64); dw[:, a] = h
        def state(sign):
            Rr = BA.so3_exp(sign * dw[:, :3]) @ prob.R
            return Rr, prob.t + sign * dw[:, 3:]
        Rp, tp = state(1); Rm, tm = state(-1)
        num = (prob.residuals(R=Rp, t=tp, with_jac=False)[0] - prob.residuals(R=Rm, t=tm, with_jac=False)[0]) / (2 * h)
        assert torch.allclose(num[dm], J_pose[dm][:, :, a], atol=1e-6, rtol=1e-6)
    assert float(J_cam[dm].abs().max() if J_cam.numel() else 0.0) == 0.0


def test_the_anchor_holds_the_shape_where_the_reprojections_are_silent():
    # a 6 cm baseline at 2.5-4.5 m: the reprojections hardly see depth; the landmarks start with a SHAPE error
    # (alternately 8 % too far and too near) and each cube's record is in its own unit (× 1.3 / × 0.7): the
    # cubes' scales are unknowns, so the anchor must hold the shape, whatever the record's unit
    w2c, pts, groups, X = _scene(baseline=0.06, noise_px=0.3, seed=1)
    tracks = list(X)
    Xs = {t: X[t] * (1.08 if k % 2 else 0.92) for k, t in enumerate(tracks)}
    cubes = np.asarray([0 if i < len(w2c) // 2 else 1 for i in range(len(w2c))])
    unit = {0: 1.3, 1: 0.7}
    ref = [groups[t][k % len(groups[t])] for k, t in enumerate(tracks)]    # rows spread over both cubes
    anchor = {"track": np.asarray(tracks), "i_ref": np.asarray([i for i, _ in ref]),
              "uv": np.asarray([p for _, p in ref]),
              "log_d": np.asarray([math.log((w2c[i, :3, :3] @ X[t] + w2c[i, :3, 3])[2] / unit[int(cubes[i])])
                                   for (i, _), t in zip(ref, tracks)]),
              "kappa": 0.3 / 0.01,                                      # σ_px 0.3 px / σ_log 1 %
              "cube_of_cam": cubes, "s0": np.zeros(2)}
    kw = dict(sigmas=np.full(len(w2c), 0.5), prior_w2c=w2c)
    free = BA.run_rung_torch("R0", w2c, CAM, WH, groups, Xs, cfg=CFG, device="cpu", **kw)
    held = BA.run_rung_torch("R0", w2c, CAM, WH, groups, Xs, cfg=CFG, device="cpu", anchor=anchor, **kw)

    def depth_err(r):
        return np.median([abs(math.log(np.linalg.norm(r.X[t]) / np.linalg.norm(X[t]))) for t in r.X])
    assert held.termination == "CONVERGENCE"
    print(f"depth error: free {depth_err(free):.4f}, anchored {depth_err(held):.4f}")
    assert depth_err(held) < 0.01 < depth_err(free)
    # each cube's scale found: the record's unit
    np.testing.assert_allclose(np.exp(held.anchor_state["s"]), [unit[0], unit[1]], rtol=0.01)
    # the fit statistic is the reprojections' alone: comparable with and without the anchor
    assert held.fit_rms_px < 1.0


def test_build_metric_anchor_nearest_keyframe_scale_and_weight(monkeypatch):
    import precision.camera as PC
    monkeypatch.setattr(PC, "native_to_grid", lambda uv, g: np.asarray(uv, np.float64))
    w2c, pts, groups, X = _scene(n_cam=5, n_pts=150, baseline=1.2, noise_px=0.2, seed=2)
    scale_rec = {i: 0.8 for i in range(len(w2c))}                    # one cube: one record unit
    H, W = WH[1], WH[0]
    recs = {}
    for i in range(len(w2c)):
        rec = np.zeros((H, W), np.float32)
        for t, x in X.items():
            for k, p in groups[t]:
                if k == i:
                    z = (w2c[i, :3, :3] @ x + w2c[i, :3, 3])[2]
                    rec[int(round(p[1])) if p[1] < H - 0.5 else H - 1, int(round(p[0])) if p[0] < W - 0.5 else W - 1] = z / scale_rec[i]
        recs[i] = rec
    fit_g = RF.group_tracks(np.concatenate([[t] * len(groups[t]) for t in groups]).astype(np.int64),
                            np.concatenate([[i for i, _ in groups[t]] for t in groups]).astype(np.int64),
                            np.concatenate([[p for _, p in groups[t]] for t in groups]),
                            list(range(len(w2c))), lists=False)
    out = RF.build_metric_anchor(fit_g, X, w2c, CAM, lambda i: (recs[i], None, 0), 0.05, 5, log=lambda m: None)
    assert out is not None
    for t, i in zip(out["track"].tolist(), out["i_ref"].tolist()):
        z = [(w2c[k, :3, :3] @ X[t] + w2c[k, :3, 3])[2] for k, _ in groups[t]]
        assert i == groups[t][int(np.argmin(z))][0]                     # the nearest keyframe
    # the record's unit recovered as the cube's start scale: log z − (log_d + s0) ≈ 0
    assert len(out["s0"]) == 1 and abs(math.exp(out["s0"][0]) - 0.8) < 0.01
    dev = [math.log((w2c[i, :3, :3] @ X[t] + w2c[i, :3, 3])[2]) - ld - out["s0"][0]
           for t, i, ld in zip(out["track"].tolist(), out["i_ref"].tolist(), out["log_d"].tolist())]
    assert np.median(np.abs(dev)) < 0.02
    assert np.isfinite(out["kappa"]) and out["kappa"] > 0


def test_the_scale_jacobian_matches_finite_differences():
    w2c, pts, groups, X = _scene()
    rng = np.random.default_rng(4)
    Xp = {t: x * (1 + rng.normal(0, 0.05)) for t, x in X.items()}
    cubes = np.asarray([i % 2 for i in range(len(w2c))])
    prob = _depth_problem(w2c, groups, Xp, cubes=cubes)
    r, front, J_pose, J_pt, J_cam = prob.residuals()
    dm = prob.is_depth
    h = 1e-6
    for c in range(prob.C):
        d = torch.zeros(prob.C, dtype=torch.float64); d[c] = h
        num = (prob.residuals(with_jac=False, s=prob.s + d)[0] - prob.residuals(with_jac=False, s=prob.s - d)[0]) / (2 * h)
        on = prob.cube_of_cam[prob.cam_idx] == c
        assert torch.allclose(num[:, 0], torch.where(on, J_cam[:, 0, -1], torch.zeros_like(num[:, 0])), atol=1e-6)
    assert float(J_cam[~dm][:, :, -1].abs().max()) == 0.0


@pytest.mark.parametrize("cam_free,blocks", [((), 1), ((0, 1, 2, 3), 1), ((0, 1), 3)])
def test_the_step_with_cube_scales_is_the_dense_damped_system(cam_free, blocks):
    """_Problem.step(λ) with the anchor's depth rows and the cubes' log-scales as unknowns, against the DENSE
    damped solve of the same linearization (the cameras damped by their Schur complement's diagonal), to 1e-9."""
    w2c, pts, groups, X = _scene(n_cam=6, n_pts=60, noise_px=0.3)
    rng = np.random.default_rng(2)
    Xp = {t: x + rng.normal(0, 0.02, 3) for t, x in X.items()}
    cubes = np.asarray([0, 0, 1, 1, 2, 2])
    prob = _depth_problem(w2c, groups, Xp, kappa=20.0, cubes=cubes, cam_free=cam_free, blocks=blocks, huber=1.0)
    L = prob.linearize()
    M, P, nc = int(L["r_w"].shape[0]), int(prob.P), int(prob.n_c)
    cols = prob._cols_of_obs(prob.nco)                                  # (M, nco)
    J = torch.zeros(2 * M + int(L["e_p"].shape[0]), nc + 3 * P, dtype=torch.float64)
    r = torch.zeros(J.shape[0], dtype=torch.float64)
    rows = torch.arange(2 * M).reshape(M, 2)
    for m in range(M):
        J[rows[m][:, None], cols[m][None, :]] += L["J_c"][m]
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


def test_the_weight_and_its_bootstrap_error_at_a_state():
    """anchor_dispersions: σ_px, σ_log, their ratio and the ratio's keyframe-bootstrap standard error — the same
    bits for the same seed, a finite positive error, and a smaller one with more data."""
    w2c, pts, groups, X = _scene(n_cam=8, n_pts=200, noise_px=0.3, seed=5)
    rng = np.random.default_rng(6)
    Xp = {t: x * (1 + rng.normal(0, 0.02)) for t, x in X.items()}
    prob = _depth_problem(w2c, groups, Xp, kappa=15.0, cubes=np.asarray([i // 4 for i in range(len(w2c))]),
                          prior_noise=0.03, seed=1)
    r, front = prob.residuals(with_jac=False)
    a = BA.anchor_dispersions(prob, r, front, 300, 0)
    b = BA.anchor_dispersions(prob, r, front, 300, 0)
    assert a == b
    assert a["sigma_px"] > 0 and a["sigma_log"] > 0
    assert a["kappa"] == pytest.approx(a["sigma_px"] / a["sigma_log"])
    assert np.isfinite(a["kappa_se"]) and 0 < a["kappa_se"] < a["kappa"]
    w2c2, _, groups2, X2 = _scene(n_cam=8, n_pts=800, noise_px=0.3, seed=5)
    Xp2 = {t: x * (1 + rng.normal(0, 0.02)) for t, x in X2.items()}
    prob2 = _depth_problem(w2c2, groups2, Xp2, kappa=15.0, cubes=np.asarray([i // 4 for i in range(len(w2c2))]),
                           prior_noise=0.03, seed=1)
    r2, f2 = prob2.residuals(with_jac=False)
    c = BA.anchor_dispersions(prob2, r2, f2, 300, 0)
    assert c["kappa_se"] / c["kappa"] < a["kappa_se"] / a["kappa"]
