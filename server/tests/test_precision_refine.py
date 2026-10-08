"""F5 — joint refinement judged by held-out reprojection (claude_stac.txt §4-F5), on a
synthetic Brown camera with known poses: a focal 3 % off with real distortion is
recovered and the ladder takes the distortion rung; with a perfect camera and no
distortion it stays at R0; witness frames localise against the fixed landmarks
within the keyframes' own held-out error."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pycolmap = pytest.importorskip("pycolmap")
if not hasattr(pycolmap.Camera, "create_from_model_name"):
    pytest.skip("pycolmap 4 (the mapanything env) is required", allow_module_level=True)

from precision import refine as R                                # noqa: E402
from precision.camera import undistort_solver                   # noqa: E402
from precision.config import load_precision_config              # noqa: E402

PC = load_precision_config()
CFG = PC.refine
SOLVER = undistort_solver(PC.camera)
WH = (640, 480)


def _error_factor() -> float:
    from config import cfg as raw_cfg
    from reconstruction.loops.config import improvement_error_factor
    return float(improvement_error_factor(raw_cfg))


FAC = _error_factor()


def _rot(yaw_deg):
    a = np.radians(yaw_deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _scene(n_kf=14, n_pts=1500, seed=0):
    """An OBSERVABLE camera: points 2-15 m around the path, the camera turning ±30°
    and pitching while it walks — a focal is only measurable when the rays span
    rotation and depth (a narrow straight walk leaves focal and depth
    interchangeable, and the held-out rightly refuses to pick one)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-np.pi, np.pi, n_pts)
    r = rng.uniform(2, 15, n_pts)
    X = np.c_[r * np.sin(ang) * 0.6, rng.uniform(-2.5, 2.5, n_pts), 4 + r * np.abs(np.cos(ang))]
    c2w = []
    for k in range(n_kf):
        T = np.eye(4)
        pitch = np.radians(8 * np.sin(k))
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        T[:3, :3] = _rot(-30 + 60 * k / max(n_kf - 1, 1)) @ Rx
        T[:3, 3] = [0.35 * k, 0.1 * np.sin(k), 0.15 * k]
        c2w.append(T)
    return X, np.array(c2w)


def _observe(X, c2w, params, noise, seed, frames):
    rng = np.random.default_rng(seed)
    w2c = np.linalg.inv(c2w)
    track, frame, uv = [], [], []
    for i, T in enumerate(w2c):
        z = (X @ T[:3, :3].T + T[:3, 3])[:, 2]
        p = R.project(X, T, params)
        ok = (z > 0) & (p[:, 0] >= 0) & (p[:, 0] < WH[0]) & (p[:, 1] >= 0) & (p[:, 1] < WH[1])
        for j in np.flatnonzero(ok):
            track.append(j)
            frame.append(frames[i])
            uv.append(p[j] + rng.normal(0, noise, 2))
    return np.array(track), np.array(frame), np.array(uv)


def _perturb(c2w, rot_deg, trans_m, seed):
    rng = np.random.default_rng(seed)
    out = c2w.copy()
    for T in out[1:]:
        T[:3, :3] = T[:3, :3] @ _rot(rng.normal(0, rot_deg))
        T[:3, 3] += rng.normal(0, trans_m, 3)
    return out


def _split(track, frac=0.2, seed=0):
    ids = np.unique(track)
    rng = np.random.default_rng(seed)
    held = set(ids[rng.random(len(ids)) < frac].tolist())
    return np.array([1 if t in held else 0 for t in track], np.int8)


def test_a_wrong_focal_and_real_distortion_are_recovered_by_the_distortion_rung():
    X, c2w = _scene()
    gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
    kf = list(range(len(c2w)))
    track, frame, uv = _observe(X, c2w, gt, 0.3, 1, kf)
    init = [515.0, 515.0, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0]
    core = R.refine_core(np.linalg.inv(_perturb(c2w, 0.2, 0.02, 2)), init, WH, track, frame, uv,
                         _split(track), kf, 0.01, CFG, SOLVER, error_factor=FAC, log=lambda *a: None)
    best = core["best"]
    assert best.name == "R2", core["rungs"]
    fx = best.params_by_block[0][0]
    assert abs(fx - 500.0) / 500.0 < 0.01, fx
    assert abs(best.params_by_block[0][4] - (-0.05)) < 0.02
    held_init = np.median(list(core["held"]["init"].values()))
    held_best = np.median(list(core["held"]["R2"].values()))
    assert held_best < held_init


def test_a_perfect_camera_without_distortion_stays_at_R0():
    X, c2w = _scene(seed=3)
    gt = [500.0, 500.0, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0]
    kf = list(range(len(c2w)))
    track, frame, uv = _observe(X, c2w, gt, 0.3, 4, kf)
    core = R.refine_core(np.linalg.inv(_perturb(c2w, 0.1, 0.01, 5)), gt, WH, track, frame, uv,
                         _split(track), kf, 0.01, CFG, SOLVER, error_factor=FAC, log=lambda *a: None)
    assert core["best"].name == "R0", core["rungs"]
    assert not core["rungs"]["R1"]["taken"] and not core["rungs"]["R2"]["taken"]


def test_witnesses_localise_within_the_keyframes_error():
    X, c2w = _scene(seed=6)
    gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
    kf = list(range(0, 2 * len(c2w), 2))                 # keyframes 0, 2, 4, …
    track, frame, uv = _observe(X, c2w, gt, 0.3, 7, kf)
    core = R.refine_core(np.linalg.inv(c2w), gt, WH, track, frame, uv, _split(track), kf, 0.01,
                         CFG, SOLVER, error_factor=FAC, log=lambda *a: None)
    # witnesses halfway between keyframes (odd frame numbers)
    wc2w = []
    for k in range(len(c2w) - 1):
        T = c2w[k].copy()
        T[:3, 3] = 0.5 * (c2w[k, :3, 3] + c2w[k + 1, :3, 3])
        wc2w.append(T)
    wframes = [2 * k + 1 for k in range(len(wc2w))]
    wt, wf, wuv = _observe(X, np.array(wc2w), gt, 0.3, 8, wframes)
    bound = R.heldout_bound(R.heldout_leave_one_view_out(core["held_groups"], core["best"],
                                                         SOLVER, CFG), CFG)
    loc = R.localize_witnesses(core["best"], core["X"], wt, wf, wuv, wframes, WH, bound, CFG)
    assert all(r["localized"] for r in loc.values()), loc
    for k, f in enumerate(wframes):
        c = np.asarray(loc[f]["c2w"])[:3, 3]
        assert np.linalg.norm(c - wc2w[k][:3, 3]) < 0.02


def _bits(core):
    b = core["best"]
    held = core["held"][b.name]
    return (b.name, b.w2c.tobytes(), np.asarray(b.params_by_block).tobytes(), b.fit_rms_px,
            list(held.keys()), np.array(list(held.values())).tobytes(),
            {t: X.tobytes() for t, X in core["X"].items()})


def test_the_ladder_and_the_witnesses_are_bit_identical_run_to_run():
    """Identical inputs → identical bits, whatever ran before: the same ladder (every
    rung a pose-prior BA whose alignment RANSAC is seeded, Ceres on one thread) and the
    same witness PnP (LO-RANSAC seeded) twice, with a different allocation history and
    COLMAP's PRNG advanced in between."""
    X, c2w = _scene(seed=6)
    gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
    kf = list(range(0, 2 * len(c2w), 2))
    track, frame, uv = _observe(X, c2w, gt, 0.3, 7, kf)
    init = [510.0, 510.0, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0]
    w2c0 = np.linalg.inv(_perturb(c2w, 0.2, 0.02, 9))
    wc2w = c2w[:-1].copy()
    wc2w[:, :3, 3] = 0.5 * (c2w[:-1, :3, 3] + c2w[1:, :3, 3])
    wframes = [2 * k + 1 for k in range(len(wc2w))]
    wt, wf, wuv = _observe(X, wc2w, gt, 0.3, 8, wframes)
    runs = []
    for rep in range(2):
        junk = [np.random.default_rng(rep).random(1000 * (rep + 1) + 17) for _ in range(5)]
        pycolmap.set_random_seed(12345 + rep)             # someone else used the PRNG
        core = R.refine_core(w2c0, init, WH, track, frame, uv, _split(track), kf, 0.01, CFG,
                             SOLVER, error_factor=FAC, log=lambda *a: None)
        loc = R.localize_witnesses(core["best"], core["X"], wt, wf, wuv, wframes, WH, 1e9, CFG)
        runs.append((_bits(core), {f: (np.asarray(r["c2w"]).tobytes(), r["rms_px"])
                                   for f, r in loc.items()}))
        del junk
    assert runs[0][0] == runs[1][0]
    assert runs[0][1] == runs[1][1]
    assert all(np.isfinite(r[1]) for r in runs[0][1].values())


def test_prior_sigmas_grow_with_the_walk():
    c = np.cumsum(np.ones((6, 3)) * [1.0, 0.0, 0.0], axis=0)
    s = R.prior_sigmas(c, 0.01)
    assert np.all(np.diff(s) >= 0) and s[0] > 0
    assert abs(s[-1] - 0.01 * np.sqrt(5)) < 1e-12


# the F5 ladder on this file's scene, in a fresh interpreter: prints the sha256 of every solved
# quantity (plan point 58 — run under the runner's own step environment)
_SOLVE_ONCE = r"""
import hashlib, json, sys
sys.path.insert(0, %(server)r); sys.path.insert(0, %(tests)r)
import numpy as np
import threadpoolctl
import test_precision_refine as T
R = T.R
X, c2w = T._scene(seed=6)
gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
kf = list(range(0, 2 * len(c2w), 2))
track, frame, uv = T._observe(X, c2w, gt, 0.3, 7, kf)
init = [510.0, 510.0, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0]
w2c0 = np.linalg.inv(T._perturb(c2w, 0.2, 0.02, 9))
core = R.refine_core(w2c0, init, T.WH, track, frame, uv, T._split(track), kf, 0.01, T.CFG, T.SOLVER,
                     error_factor=T.FAC, log=lambda *a: None)
b = core["best"]
h = hashlib.sha256()
h.update(b.name.encode()); h.update(b.w2c.tobytes()); h.update(np.asarray(b.params_by_block).tobytes())
h.update(np.float64(b.fit_rms_px).tobytes())
for name, held in sorted(core["held"].items()):
    keys = sorted(held)
    h.update(name.encode()); h.update(np.array(keys).tobytes()); h.update(np.array([held[k] for k in keys]).tobytes())
for t in sorted(core["X"]):
    h.update(core["X"][t].tobytes())
print(json.dumps({"sha": h.hexdigest(), "best": b.name,
                  "blas": sorted((str(d.get("internal_api")), str(d.get("architecture")), int(d.get("num_threads") or 0))
                                 for d in threadpoolctl.threadpool_info() if d.get("user_api") == "blas")}))
"""


def _solve_in_a_fresh_interpreter(env):
    import json
    import subprocess
    tests_dir = Path(__file__).resolve().parent
    code = _SOLVE_ONCE % {"server": str(tests_dir.parent), "tests": str(tests_dir)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_solve_is_bit_identical_under_the_runners_blas_threads(monkeypatch):
    """Plan point 58: Ceres runs one thread, but CHOLMOD's BLAS inside pycolmap sees the step env's
    OMP / OPENBLAS / MKL thread count. Two F5 ladder solves in two fresh interpreters under the
    runner's F5 environment must hash identically — and identically to the one-thread solve. They
    did on 2026-10-07 (a8790622…, AMD EPYC 7763, pycolmap 4.0.4): that is why F5 keeps
    runner.threads (precision.runner.step_threads). If this ever fails, F5 is pinned to 1 thread."""
    from precision import runner as RN
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    f5 = next(s for s in RN.STEPS if s.key == "f5_refine")
    threads = RN.step_threads(f5, PC.runner)
    env = RN.step_env(threads)
    a = _solve_in_a_fresh_interpreter(env)
    b = _solve_in_a_fresh_interpreter(env)
    one = _solve_in_a_fresh_interpreter(RN.step_env(1))
    assert a["blas"] and all(arch == "Haswell" for api, arch, _n in a["blas"] if api == "openblas")
    assert a == b, (a, b)
    assert a["sha"] == one["sha"], (a, one)
    assert a["best"] == "R2"
