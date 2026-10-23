"""Phase 3b (audit 2026-10-23): drift measured in metres at the loops, the odometry prior model
of F5 against the absolute one on a walk with a known truth, the metre locked to the gauge's step
lengths.

A synthetic walk around a 6 x 4 m rectangle that overlaps its own first side (keyframes 40-43
stand where 0-3 stood): the loop window is {0..4} ∪ {39..43}. The initial poses carry a yaw
drift that accumulates to ~0.5 m at the return; the gauge priors (σ_rel 0.01) hold that drift.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pycolmap = pytest.importorskip("pycolmap")
if not hasattr(pycolmap.Camera, "create_from_model_name"):
    pytest.skip("pycolmap 4 (the mapanything env) is required", allow_module_level=True)

from precision import loop_gap as LG                              # noqa: E402
from precision import refine as R                                 # noqa: E402
from precision import scale_lock as SL                            # noqa: E402
from precision.camera import undistort_solver                     # noqa: E402
from precision.config import load_precision_config                # noqa: E402

PC = load_precision_config()
CFG = PC.refine
SOLVER = undistort_solver(PC.camera)
WH = (640, 480)
GT = [500.0, 500.0, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0]
N_KF = 44
LOOP_A = list(range(0, 5))
LOOP_B = list(range(39, 44))


def _error_factor() -> float:
    from config import cfg as raw_cfg
    from reconstruction.loops.config import improvement_error_factor
    return float(improvement_error_factor(raw_cfg))


FAC = _error_factor()


def _rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _look(direction):
    """c2w rotation with the camera's +z along ``direction`` (world y up)."""
    z = np.asarray(direction, float); z = z / np.linalg.norm(z)
    up = np.array([0.0, -1.0, 0.0])                     # image y down
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]


def rectangle_walk(n=N_KF, step=0.5):
    """Perimeter positions and headings of a 6 x 4 rectangle in (x, z), y = 0, overlapping its
    first side: keyframe k stands at perimeter length 0.5·k mod 20, with the 2 cm bob of a
    handheld camera (exactly coplanar centres are a degenerate case for COLMAP's prior alignment —
    refine refuses that solve; a real walk never is)."""
    corners = [np.array([0.0, 0.0, 0.0]), np.array([6.0, 0.0, 0.0]), np.array([6.0, 0.0, 4.0]),
               np.array([0.0, 0.0, 4.0])]
    c2w = []
    for k in range(n):
        s = (step * k) % 20.0
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            L = np.linalg.norm(b - a)
            if s <= L + 1e-9:
                d = (b - a) / L
                T = np.eye(4); T[:3, :3] = _look(d); T[:3, 3] = a + d * s + np.array([0.0, 0.02 * np.sin(1.7 * s), 0.0])
                c2w.append(T)
                break
            s -= L
    return np.array(c2w)


def room_points(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-5, 11, n), rng.uniform(-2.5, 2.5, n), rng.uniform(-5, 9, n)]


def drifted(c2w, yaw_deg_at_end=10.0):
    """Integrate the true steps with a yaw error growing linearly along the walk — the shape of a
    monocular odometry's drift; the heading carries the same error."""
    n = len(c2w)
    out = c2w.copy()
    p = c2w[0, :3, 3].copy()
    for k in range(1, n):
        th = np.radians(yaw_deg_at_end) * k / (n - 1)
        p = p + _rot_y(th) @ (c2w[k, :3, 3] - c2w[k - 1, :3, 3])
        out[k, :3, 3] = p
        out[k, :3, :3] = _rot_y(th) @ c2w[k, :3, :3]
    return out


def visible(X, w2c, params):
    z = (X @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    p = R.project(X, w2c, params)
    ok = (z > 0.5) & (z < 12.0) & (p[:, 0] >= 0) & (p[:, 0] < WH[0]) & (p[:, 1] >= 0) & (p[:, 1] < WH[1])
    return ok, p


def observe(X, c2w, params, noise=0.3, seed=0, win=8, overlap=4, loop=(LOOP_A, LOOP_B)):
    """F4-like observations: sequential windows of ``win`` keyframes overlapping by ``overlap``
    plus ONE loop window; every point seen from ≥ 2 keyframes of a window is a track of that
    window. Returns track, frame, uv, window_id, is_loop (per observation) with kf = 0..N-1."""
    rng = np.random.default_rng(seed)
    w2c = np.linalg.inv(c2w)
    n = len(c2w)
    windows = []
    s = 0
    while True:
        windows.append((list(range(s, min(n, s + win))), False))
        if s + win >= n:
            break
        s += win - overlap
    windows.append((sorted(loop[0]) + sorted(loop[1]), True))
    vis = {}
    for i in range(n):
        vis[i] = visible(X, w2c[i], params)
    track, frame, uv, wid, isl = [], [], [], [], []
    tid = 0
    for w, (frames, is_loop) in enumerate(windows):
        seen = np.zeros(len(X), int)
        for i in frames:
            seen += vis[i][0]
        for j in np.flatnonzero(seen >= 2):
            for i in frames:
                if vis[i][0][j]:
                    track.append(tid); frame.append(i); uv.append(vis[i][1][j] + rng.normal(0, noise, 2))
                    wid.append(w); isl.append(is_loop)
            tid += 1
    return (np.array(track), np.array(frame), np.array(uv), np.array(wid, np.int32),
            np.array(isl, bool))


def split_of(track, frac=0.2, seed=0):
    ids = np.unique(track)
    rng = np.random.default_rng(seed)
    held = set(ids[rng.random(len(ids)) < frac].tolist())
    return np.array([1 if t in held else 0 for t in track], np.int8)


def true_error_m(w2c, c2w_true):
    C = SL.centres_of(w2c)
    Rk, t = LG.kabsch(C, c2w_true[:, :3, 3])
    return float(np.sqrt(np.mean(np.sum((C @ Rk.T + t - c2w_true[:, :3, 3]) ** 2, axis=1))))


@pytest.fixture(scope="module")
def walk():
    c2w = rectangle_walk()
    assert len(c2w) == N_KF
    # the overlap: keyframes 40..43 stand where 0..3 stood, heading alike
    assert np.allclose(c2w[40:44], c2w[0:4], atol=1e-9)
    X = room_points()
    track, frame, uv, wid, isl = observe(X, c2w, GT)
    return {"c2w": c2w, "X": X, "track": track, "frame": frame, "uv": uv, "wid": wid, "isl": isl,
            "split": split_of(track)}


def _measure(walk, w2c, held_only=True):
    kf = list(range(N_KF))
    sel = walk["split"] == 1 if held_only else np.ones(len(walk["track"]), bool)
    held_g = R.group_tracks(walk["track"][sel], walk["frame"][sel], walk["uv"][sel], kf)
    window_of = {int(t): int(w) for t, w in zip(walk["track"], walk["wid"])}
    loops = sorted(set(int(w) for w in walk["wid"][walk["isl"]]))
    return LG.measure_loops(held_g, window_of, loops, w2c, GT, SOLVER, CFG.min_tri_deg, error_factor=FAC,
                            confidence=CFG.heldout_confidence, n_boot=200, seed=0)


def test_loop_sides_split_at_the_widest_gap_and_refuse_one_run():
    assert LG.loop_sides([0, 1, 2, 3, 4, 39, 40, 41]) == ({0, 1, 2, 3, 4}, {39, 40, 41})
    assert LG.loop_sides([5, 6, 7, 8]) is None
    assert LG.loop_sides([3]) is None


def test_true_poses_read_closed_and_drifted_poses_read_open_and_coherent(walk):
    c2w = walk["c2w"]
    closed = _measure(walk, np.linalg.inv(c2w))
    w = closed["windows"]
    assert len(w) == 1 and closed["summary"]["n_judged"] == 1
    (rep,) = w.values()
    assert rep["status"] == "judged" and rep["n_tracks"] >= 20
    assert rep["side_a"] == LOOP_A and set(rep["side_b"]) <= set(LOOP_B) and len(rep["side_b"]) >= 4
    assert not rep["open"], rep["open_verdict"]
    # the control measures the triangulation noise of the same tracks (a forward walk: centimetres
    # at 0.3 px, the parallax of the points ahead is small) and the gap sits inside it
    assert rep["control_median_m"] < 0.15 and rep["gap_median_m"] < 1.5 * rep["control_median_m"]
    d = drifted(c2w)
    opened = _measure(walk, np.linalg.inv(d))
    (rep_d,) = opened["windows"].values()
    assert rep_d["open"] and rep_d["coherent"], (rep_d["open_verdict"], rep_d["residual_open_verdict"])
    drift = float(np.linalg.norm(d[41, :3, 3] - c2w[41, :3, 3]))
    assert drift > 0.5
    assert rep_d["gap_median_m"] > 0.5 * drift
    # the rigid fit reads the offset of the surface between the visits and the heading drift (10°)
    fit = rep_d["rigid_fit"]
    assert abs(fit["translation_m"] - rep_d["gap_median_m"]) < 0.3 * rep_d["gap_median_m"], fit
    assert abs(fit["rotation_deg"] - 10.0) < 3.0, fit
    assert fit["residual_median_m"] < 0.3 * rep_d["gap_median_m"]
    report = LG.loop_report(opened, closed)
    assert report["verdict_by_window"][list(w)[0]] == "closed_by_f5"
    assert report["summary"]["open_before"] == 1 and report["summary"]["open_after"] == 0


def test_tracks_that_disagree_read_open_and_incoherent(walk):
    """Side B's observations permuted between the loop tracks: every gap is large, no rigid
    offset explains them."""
    c2w = walk["c2w"]
    w2 = dict(walk)
    rng = np.random.default_rng(5)
    uv = walk["uv"].copy()
    loop_b = walk["isl"] & np.isin(walk["frame"], LOOP_B) & (walk["split"] == 1)
    idx = np.flatnonzero(loop_b)
    uv[idx] = uv[rng.permutation(idx)]
    w2["uv"] = uv
    rep = list(_measure(w2, np.linalg.inv(c2w))["windows"].values())[0]
    assert rep["status"] == "judged"
    assert rep["open"] and not rep["coherent"], (rep["open_verdict"], rep["residual_open_verdict"])
    assert rep["rigid_fit"]["residual_median_m"] > 3.0 * rep["control_median_m"]


def test_the_gap_verdict_is_paired_and_clustered_by_window(walk):
    c2w = walk["c2w"]
    before = _measure(walk, np.linalg.inv(drifted(c2w)))
    after = _measure(walk, np.linalg.inv(c2w))
    v = LG.judge_gap_change(before, after, error_factor=FAC, confidence=CFG.heldout_confidence, n_boot=200, seed=0)
    # one loop window = one judge: the rule asks for min_judge_closures (5) — the clusters refuse
    assert v["n_judges"] == 1 and not v["enough_judges"] and v["unit"] == "m"
    assert v["median_delta"] > 0.3
    same = LG.judge_gap_change(after, after, error_factor=FAC, confidence=CFG.heldout_confidence, n_boot=200, seed=0)
    assert not same["improves"] and not same["worsens"]


def test_scale_lock_reads_the_gauge_metre_from_the_step_lengths():
    c2w = rectangle_walk()
    cg = c2w[:, :3, 3]
    rng = np.random.default_rng(0)
    # the solve's trajectory: the gauge's shape with a yaw drift (lengths kept) and 3 % too large
    cs = drifted(c2w, 3.0)[:, :3, 3] * 1.03 + rng.normal(0, 0.002, cg.shape)
    m = SL.measure(cg, cs, error_factor=FAC, confidence=0.95, n_boot=300, seed=0)
    assert m["apply"] and abs(m["s"] - 1 / 1.03) < 0.004, m
    assert m["n_steps"] == N_KF - 1 and not m["trend_per_100m"]["significant"]
    # within the noise: the solve's metre stands
    m0 = SL.measure(cg, cg + rng.normal(0, 0.002, cg.shape), error_factor=FAC, confidence=0.95, n_boot=300, seed=0)
    assert not m0["apply"] and abs(m0["log_s"]) < 0.01, m0
    # a scale drift along the walk is a reported trend
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cg, axis=0), axis=1))])
    cs2 = np.cumsum(np.r_[cg[:1], np.diff(cg, axis=0) * (1.0 + 0.004 * L[1:, None])], axis=0)
    m2 = SL.measure(cg, cs2, error_factor=FAC, confidence=0.95, n_boot=300, seed=0)
    assert m2["trend_per_100m"]["significant"] and m2["trend_per_100m"]["slope"] < 0
    # the similarity keeps every reprojection
    w2c = np.linalg.inv(c2w)
    X = {0: np.array([1.0, 0.2, 3.0]), 1: np.array([4.0, -0.5, 2.0])}
    w2c_s, X_s = SL.apply_similarity(w2c, X, 1.03, cg.mean(axis=0))
    for t, P in X.items():
        for i in (0, 7, 20):
            np.testing.assert_allclose(R.project(P[None], w2c[i], GT), R.project(X_s[t][None], w2c_s[i], GT), atol=1e-9)
    assert np.allclose(w2c_s[:, :3, :3], w2c[:, :3, :3])


def test_the_odometry_priors_recover_the_true_shape_to_millimetres(walk):
    """THE PRIOR MODEL (phase 3b): the same drifted start, the same tracks. Absolute priors hold
    Omega's drifted shape (the loop 'closes' at the ends, the interior stays bent); the odometry
    priors — the metre on every step, the first keyframe fixed — give the tracks the shape."""
    c2w = walk["c2w"]
    w2c0 = np.linalg.inv(drifted(c2w))
    kw = {"window_id": walk["wid"], "is_loop": walk["isl"]}
    err0 = true_error_m(w2c0, c2w)
    out = {}
    for model in ("relative", "absolute"):
        cfg = replace(CFG, permutations=200, prior_model=model, prior_sigma_from_loops=(model == "relative"))
        core = R.refine_core(w2c0, GT, WH, walk["track"], walk["frame"], walk["uv"], walk["split"],
                             list(range(N_KF)), 0.01, cfg, SOLVER, error_factor=FAC, log=lambda *a: None, **kw)
        out[model] = core
    rel, ab = out["relative"], out["absolute"]
    assert err0 > 0.1
    err_rel, err_ab = true_error_m(rel["best"].w2c, c2w), true_error_m(ab["best"].w2c, c2w)
    assert err_rel < 0.005, err_rel                       # millimetres
    assert err_ab > 0.05 and err_rel < 0.1 * err_ab, (err_ab, err_rel)
    # the odometry model: R0 uninformative, then R0s with the σ measured against the tracks — the
    # measured rotation scatter is the synthetic's heading drift per step (10° over 43 steps)
    od = rel["odometry"]
    assert od["model"] == "relative" and od["prior"]["measured_taken"] is True
    sc = od["scatter_vs_uninformative"]
    assert sc["n_steps"] == N_KF - 1 and abs(sc["rot_median_deg"] - 10.0 / (N_KF - 1)) < 0.05
    assert rel["rungs"]["R0s"]["taken"] and rel["best"].name in ("R0s", "R1", "R2")
    assert rel["rungs"]["R0s"]["loop_gap_verdict_vs_R0"]["unit"] == "m"
    assert "skipped" in rel["prior_sigma_ladder"] and "odometry" in rel["prior_sigma_ladder"]["skipped"]
    # the metre: the step-length lock reads the solve's scale within 1e-3 of the gauge's, not applied
    lock = rel["scale_lock"]
    assert lock["measured"] and lock["n_steps"] == N_KF - 1 and abs(lock["log_s"]) < 1e-3 and not lock["apply"]
    # the loop instrument on both: open before, closed after
    for core in (rel, ab):
        sm = core["loops"]["summary"]
        assert sm["n_judged"] == 1 and sm["open_before"] == 1 and sm["open_after"] == 0, sm
    assert "scatter_vs_uninformative" not in ab["odometry"] and ab["odometry"]["model"] == "absolute"


def test_without_loop_columns_the_instrument_declares_itself_absent(walk):
    c2w = walk["c2w"]
    w2c0 = np.linalg.inv(drifted(c2w))
    cfg = replace(CFG, permutations=200, prior_model="absolute", prior_sigma_from_loops=True)
    core = R.refine_core(w2c0, GT, WH, walk["track"], walk["frame"], walk["uv"], walk["split"],
                         list(range(N_KF)), 0.01, cfg, SOLVER, error_factor=FAC, log=lambda *a: None)
    lad = core["prior_sigma_ladder"]
    assert not core["loops"]["measured"]
    assert "skipped" in lad and lad["rungs"] == {} and lad["applied_factor"] == 1.0
    assert core["best"].name in ("R0", "R1", "R2")


def test_the_sigma_relaxation_of_the_absolute_model_is_measured_on_the_open_loops(walk):
    """The absolute model's relaxation: tried only when a loop stays open after the camera ladder,
    the factor measured on that loop, taken by the loop-gap rule; reported either way."""
    c2w = walk["c2w"]
    w2c0 = np.linalg.inv(drifted(c2w))
    cfg = replace(CFG, permutations=200, prior_model="absolute", prior_sigma_from_loops=True)
    core = R.refine_core(w2c0, GT, WH, walk["track"], walk["frame"], walk["uv"], walk["split"],
                         list(range(N_KF)), 0.01, cfg, SOLVER, error_factor=FAC, log=lambda *a: None,
                         window_id=walk["wid"], is_loop=walk["isl"])
    lad = core["prior_sigma_ladder"]
    assert lad["enabled"] and "measured" in lad
    assert lad["measured"]["method"].startswith("median over the open loop windows")
    for name, rep in lad["rungs"].items():
        assert "loop_gap_verdict_vs_best" in rep and rep["loops"]["n_judged"] == 1
        assert rep["nested_cost_check"]["applicable"] is False and rep["prior_sigma_factor"] > 1.0
    if lad["measured"]["n_open"] == 0:
        assert "skipped" in lad and lad["applied_factor"] == 1.0
