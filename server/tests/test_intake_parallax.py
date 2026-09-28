"""I1 — keyframes by parallax (claude_stac.txt §4-F1), as the spec asks:

  * a pure rotation earns no keyframe beyond the first and is warned
    pure_rotation;
  * a translation earns keyframes at parallax intervals — each keyframe's
    reading lies in the quantum band, the sharpest frame of the band wins;
  * a light change without motion earns no keyframe beyond the first;
  * witnesses dedup against the last CHOSEN witness and include every keyframe;
  * the written selection satisfies the v2 contract, two runs are identical;
  * the reference is the pure rotation of any pinhole: a rotation reads ~0
    with the intrinsics unknown, a forward walk reads its baseline (a
    homography would not).

Every sequence is rendered by tests/synth_precision.py with sensor noise."""

import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from tests import synth_precision as S                          # noqa: E402
from intake import quality as Q                                 # noqa: E402
from intake import parallax as P                                # noqa: E402
from intake.config import ParallaxConfig, QualityConfig, load_intake_config   # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
QCFG = QualityConfig(luma_lo=40.0, luma_hi=220.0, clip_frac_max=0.25, clip_lo=5, clip_hi=250,
                     fft_max_side=640, diff_max_side=320)
# The production values except the tracking scale, the grid and the LK window,
# sized for 320x240 test frames — test_test_config_mirrors_production pins that.
SIZED_FOR_TEST = {"grid_side", "process_scale", "lk_win"}
PCFG = ParallaxConfig(grid_side=24, process_scale=0.5, ransac_px=1.0, parallax_quantum_px=12.0,
                      witness_min_parallax_px=1.5, min_tracks=40, lk_win=15, lk_levels=3,
                      fb_max_px=1.0, warn_static_disp_px=0.5, warn_rotation_min_disp_px=4.0,
                      warn_min_run_frames=5, parallax_quantile=0.9, keyframe_band_frac=0.25,
                      reference_max_eval=200, reference_ftol=1.0e-3)
NOISE_SIGMA = 2.0
QUIET = (lambda *a, **k: None)
WALK = dict(start=S.room_start_pose(x_m=1.2), direction="right")


def _run(session_dir, scene, cam, poses, cfg=PCFG, **kw):
    sess = S.write_session(session_dir, scene, cam, poses, noise_sigma=NOISE_SIGMA, **kw)
    quality = Q.analyze_frames(sess.frames_dir, QCFG, log=QUIET, heartbeat_s=1e-6)
    result = P.run_parallax(sess.frames_dir, quality, cfg, log=QUIET, heartbeat_s=1e-6)
    return sess, quality, result


def _kinds(result):
    return [w["kind"] for w in result["warnings"]]


@pytest.fixture(scope="module")
def scene():
    return S.room_scene(seed=0)


@pytest.fixture(scope="module")
def cam():
    return S.default_camera()


@pytest.fixture(scope="module")
def rotation(tmp_path_factory, scene, cam):
    poses = S.walk_poses("rotate", 30, start=S.room_start_pose(), yaw_deg_per_frame=1.5)
    return _run(tmp_path_factory.mktemp("rotation"), scene, cam, poses)


@pytest.fixture(scope="module")
def translation(tmp_path_factory, scene, cam):
    poses = S.walk_poses("translate", 50, step_m=0.05, **WALK)
    return _run(tmp_path_factory.mktemp("translation"), scene, cam, poses)


@pytest.fixture(scope="module")
def light_change(tmp_path_factory, scene, cam):
    poses = S.walk_poses("still", 30, start=S.room_start_pose())
    return _run(tmp_path_factory.mktemp("light"), scene, cam, poses,
                light=list(np.linspace(0.6, 1.4, 30)))


# ── configuration ────────────────────────────────────────────────────────

def test_test_config_mirrors_production():
    prod = load_intake_config().parallax
    for f in fields(ParallaxConfig):
        if f.name not in SIZED_FOR_TEST:
            assert getattr(PCFG, f.name) == getattr(prod, f.name), f.name


def test_production_yaml_has_exactly_the_dataclass_keys():
    doc = yaml.safe_load((SERVER / "config.yaml").read_text())
    assert set(doc["intake"]["parallax"]) == {f.name for f in fields(ParallaxConfig)}


# ── the reference ────────────────────────────────────────────────────────

def _pinhole(f, w=640, h=480):
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])


def _project(K, R, t, X):
    x = (K @ (R @ X.T + t[:, None])).T
    return x[:, :2] / x[:, 2:3]


def _cloud(rng, n=600):
    # points 2 – 8 m in front of the camera, spread over the view
    z = rng.uniform(2.0, 8.0, n)
    return np.c_[rng.uniform(-0.5, 0.5, n) * z, rng.uniform(-0.4, 0.4, n) * z, z]


def _reading(a, b):
    T = P.image_normaliser((640, 480))
    x, H, _cost = P.fit_rotation(a, b, T, PCFG)
    return float(np.median(P.symmetric_transfer_error(H, P._inv3(H), a, b)))


def test_inv3_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        M = rng.normal(size=(3, 3))
        assert np.allclose(P._inv3(M), np.linalg.inv(M))
    assert P._inv3(np.zeros((3, 3))) is None


def test_singular_intrinsics_are_non_finite_not_an_exception():
    """A focal length the solver drives to exactly 0 must price itself out of
    the fit, not abort the run (it did, at frame 10,091 of pccr 2026-08-24)."""
    for x in ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]):
        with np.errstate(all="ignore"):
            assert not np.all(np.isfinite(P.rotation_any_k(np.array(x))))
    rng = np.random.default_rng(3)
    X = _cloud(rng)
    K = _pinhole(500.0)
    a = _project(K, np.eye(3), np.zeros(3), X)
    b = _project(K, np.eye(3), np.array([0.05, 0.0, 0.0]), X)
    T = P.image_normaliser((640, 480))
    # started ON the singularity: the fit completes (a result or None), it never raises
    with np.errstate(all="ignore"):
        P.fit_rotation(a, b, T, PCFG, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_rotation_reads_zero_whatever_the_focal_length():
    rng = np.random.default_rng(1)
    X = _cloud(rng)
    for f in (350.0, 600.0, 1100.0):           # the fit is started at the half-diagonal (400)
        K = _pinhole(f)
        R, _ = cv2.Rodrigues(np.array([0.02, 0.08, 0.01]))
        a = _project(K, np.eye(3), np.zeros(3), X)
        b = _project(K, R, np.zeros(3), X)
        assert _reading(a, b) < 0.05, f


def test_forward_walk_reads_its_baseline():
    """A homography explains much of a forward walk's expansion; the rotation
    family cannot (K·I·K⁻¹ = I for any K): its reading is nearer the ground
    truth — with no rotation, the whole image motion — at every baseline, and
    grows with the distance walked."""
    rng = np.random.default_rng(2)
    X = _cloud(rng)
    K = _pinhole(500.0)
    a = _project(K, np.eye(3), np.zeros(3), X)
    readings = []
    for dz in (0.1, 0.25, 0.5, 1.0):
        b = _project(K, np.eye(3), np.array([0.0, 0.0, -dz]), X)
        truth = float(np.median(np.hypot(*(b - a).T)))
        H, _ = cv2.findHomography(a, b, 0)
        hom = float(np.median(P.symmetric_transfer_error(H, np.linalg.inv(H), a, b)))
        rot = _reading(a, b)
        assert abs(truth - rot) < abs(truth - hom), dz
        readings.append(rot)
    assert readings == sorted(readings)


# ── the sequences ────────────────────────────────────────────────────────

def test_pure_rotation_earns_no_keyframe_and_is_warned(rotation):
    _sess, _q, res = rotation
    assert res["n_keyframes"] == 1
    assert "pure_rotation" in _kinds(res)


def test_translation_keyframes_at_parallax_intervals(translation):
    _sess, quality, res = translation
    kfs = res["keyframes"]
    assert len(kfs) >= 3
    q, band = PCFG.parallax_quantum_px, PCFG.keyframe_band_frac
    ranks = {int(r["frame"]): r["sharp_rank"] for r in quality["frames"]}
    for k in kfs[1:]:
        if k["closed_by"] == "quantum":
            assert q * (1 - band) <= k["parallax_px"] <= q * (1 + band), k
    frames = [k["frame"] for k in kfs]
    assert frames == sorted(frames) and len(set(frames)) == len(frames)
    # the window's keyframe is the sharpest of the frames that read inside the band
    by_frame = {r["frame"]: r for r in res["frames"]}
    for w in res["windows"]:
        if w["closed_by"] != "quantum":
            continue
        in_band = [f for f, r in by_frame.items()
                   if r["anchor"] == w["anchor"] and not r["lost"]
                   and q * (1 - band) <= r["parallax_px"] <= q * (1 + band)]
        if in_band:
            assert ranks[w["keyframe"]] >= max(ranks[f] for f in in_band) - 1e-12
    assert "pure_rotation" not in _kinds(res)


def test_light_change_without_motion_earns_no_keyframe(light_change):
    _sess, _q, res = light_change
    assert res["n_keyframes"] == 1
    assert "static" in _kinds(res)


def test_witnesses_dedup_and_include_every_keyframe(translation):
    _sess, _q, res = translation
    wit = res["witnesses"]
    frames = [w["frame"] for w in wit]
    assert frames == sorted(frames) and len(set(frames)) == len(frames)
    assert {k["frame"] for k in res["keyframes"]} <= set(frames)
    for w in wit[1:]:
        if w["reason"] == "parallax":
            assert w["parallax_px"] >= PCFG.witness_min_parallax_px
    # dedup: fewer witnesses than frames on a 5 cm/frame walk, more than keyframes
    assert res["n_keyframes"] < len(wit) <= res["n_usable_frames"]


def test_written_selection_satisfies_the_v2_contract(translation):
    sess, _q, res = translation
    p_kf, p_w, p_warn = P.write_selection(sess.session_dir, res, PCFG)
    sel = json.loads(p_kf.read_text())
    assert sel["version"] == "2.0" and sel["method"].startswith(P.METHOD)
    assert sel["selected_count"] == len(sel["selected_files"]) == res["n_keyframes"]
    assert sel["total_frames"] == res["n_frames"]
    assert all((sess.frames_dir / f).exists() for f in sel["selected_files"])
    assert P.load_selection(sess.frames_dir)["selected_files"] == sel["selected_files"]
    wit = json.loads(p_w.read_text())
    assert [w["frame"] for w in wit["frames"]] == [w["frame"] for w in res["witnesses"]]
    warn = json.loads(p_warn.read_text())
    assert warn["warning_kinds"] == list(P.WARNING_KINDS)
    for doc in (sel, wit, warn):
        assert doc["provenance"] == "tool_measured"
        assert "geometry_epoch" in doc and "camera_epoch" in doc and "params" in doc


def test_two_runs_are_identical(tmp_path, scene, cam):
    poses = S.walk_poses("translate", 20, step_m=0.05, **WALK)
    _s1, _q1, r1 = _run(tmp_path / "a", scene, cam, poses)
    _s2, _q2, r2 = _run(tmp_path / "b", scene, cam, poses)
    assert r1["keyframes"] == r2["keyframes"]
    assert r1["witnesses"] == r2["witnesses"]
    assert r1["warnings"] == r2["warnings"]


def test_lost_tracks_invent_no_keyframe(tmp_path, scene, cam):
    """Every frame lost (min_tracks above the grid): no baseline was measured,
    so no keyframe beyond the first is invented and the run is warned."""
    poses = S.walk_poses("translate", 12, step_m=0.05, **WALK)
    cfg = replace(PCFG, min_tracks=PCFG.grid_side ** 2 + 1)
    _s, _q, res = _run(tmp_path, scene, cam, poses, cfg=cfg)
    assert res["n_keyframes"] == 1                  # nothing measurable: no keyframe invented
    assert "tracking_lost" in _kinds(res)
