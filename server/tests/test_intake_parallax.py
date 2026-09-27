"""I1 — parallax measured against the ANCHOR (the last keyframe), on the rigid
tracks, every window matched to the anchor at every frame:

  * the tracker: a track's error does not grow with the links behind it; a
    window that loses its anchor's patch (occlusion, aliasing) leaves;
  * a pure rotation earns no keyframe and its whole run is flagged
    pure_rotation — at 1–2°/frame, 0.1°/frame over 200 frames, past the field
    of view (180°: coverage breaks, no keyframe), through rolling bands, with
    a person crossing and the principal point 2 % off-centre;
  * a translation earns keyframes that follow the ground-truth parallax of
    the baseline anchor → keyframe (sideways; forward down a corridor at the
    production scale), inside the symmetric keyframe band, and is never
    flagged as a rotation; the same walk gives the same keyframes with or
    without a still prefix, at 1× vs 2× frame density and at 60 / 30 / 15 fps
    (the declared bound where one frame step is wider than the band);
  * an object moving on its own — during a pure rotation, in front of a still
    camera, crossing during a still prefix — earns no keyframe and is
    reported as dynamic_content; one crossing ALONG the epipolar lines of a
    walk is the declared limit, measured, and the exclusion audit reports it
    once I2's masks exist;
  * a still camera on a weak texture earns one keyframe and a static warning;
    an anchor the tracker cannot hold is warned tracking_lost; a textureless
    stretch is a hard break with no keyframe inside it;
  * a light change without motion earns one keyframe and a static warning;
    witnesses dedup against the last CHOSEN witness (keyframes included); the
    written selection satisfies the v2 contract; two runs are identical; the
    declared limits (a single plane reads as a rotation, a distorted lens
    leaks a little parallax on a rotation, a window straddling a depth edge)
    are measured, not hidden.

Every sequence is rendered by tests/synth_precision.py with sensor noise; where
the anchor matters the recording starts still, as a real one does."""

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
from intake import parallax as P                                # noqa: E402
from intake import quality as Q                                 # noqa: E402
from intake.config import ParallaxConfig, QualityConfig, load_intake_config   # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
QCFG = QualityConfig(luma_lo=40.0, luma_hi=220.0, clip_frac_max=0.25, clip_lo=5, clip_hi=250,
                     fft_max_side=640, diff_max_side=320)
# The production values except the tracking scale, the grid and the LK window,
# sized for 320x240 test frames (production frames are 1080p and more) —
# test_test_config_mirrors_production pins that.
SIZED_FOR_TEST = {"grid_side", "process_scale", "lk_win"}
PCFG = ParallaxConfig(grid_side=24, process_scale=0.5, ransac_px=1.0, parallax_quantum_px=12.0,
                      witness_min_parallax_px=1.5, min_tracks=40, lk_win=15, lk_levels=3,
                      fb_max_px=1.0, warn_static_disp_px=0.5, warn_rotation_min_disp_px=4.0,
                      warn_min_run_frames=5, parallax_quantile=0.9, rotation_floor_factor=1.5,
                      keyframe_band_frac=0.25, rigidity_confidence=0.95, rigidity_bootstrap=200,
                      refine_max_iter=30, refine_eps_px=0.01)
NOISE_SIGMA = 2.0          # sensor noise in grey levels
STILL_PREFIX = 8           # frames the camera is held still before it moves
QUIET = (lambda *a, **k: None)
WALL_POSE = S.look_at_pose((0.0, 1.5, 6.5), (0.0, 1.5, 7.5))     # 1.5 m from the back wall
PERSON_TEX = S.make_texture("noise", 256, seed=77)


def _with_still(poses, n=STILL_PREFIX):
    return np.concatenate([np.repeat(poses[:1], n, axis=0), poses])


def _run(session_dir, scene, cam, poses, cfg=PCFG, scene_fn=None, **kw):
    """Render (optionally a per-frame scene), measure I0, run I1.
    Returns (session, quality, result)."""
    if scene_fn is None:
        sess = S.write_session(session_dir, scene, cam, poses, noise_sigma=NOISE_SIGMA, **kw)
    else:
        renders = [S.render(scene_fn(i, c), cam, c, noise_sigma=NOISE_SIGMA, seed=i,
                            frame_index=i) for i, c in enumerate(poses)]
        frames_dir = Path(session_dir) / "frames"
        files = S.write_frames(frames_dir, renders)
        (Path(session_dir) / "output").mkdir(parents=True, exist_ok=True)
        sess = S.SynthSession(session_dir=Path(session_dir), frames_dir=frames_dir,
                              output_dir=Path(session_dir) / "output", cam=cam, poses=poses,
                              frame_numbers=list(range(len(poses))), renders=renders,
                              scene=scene, files=files)
    quality = Q.analyze_frames(sess.frames_dir, QCFG, log=QUIET, heartbeat_s=1e-6)
    result = P.run_parallax(sess.frames_dir, quality, cfg, log=QUIET, heartbeat_s=1e-6)
    return sess, quality, result


def _person_scene(scene, offset_fn):
    """The verifiers' moving object: a textured slab 0.6 x 1.0 m held 1.5 m in
    front of the camera, sliding along the camera's x axis by ``offset_fn(i)``
    (None: absent) — a person the camera pans past."""
    def fn(i, c2w):
        xc = offset_fn(i)
        if xc is None:
            return scene
        pc = c2w[:3, :3] @ np.array([xc, 0.0, 1.5]) + c2w[:3, 3]
        yaw = np.degrees(np.arctan2(c2w[0, 2], c2w[2, 2]))
        box = S.Box(pc, (0.6, 1.0, 0.05), yaw, PERSON_TEX, "person", "person")
        return S.Scene(quads=list(scene.quads), name="dyn", boxes=list(scene.boxes) + [box])
    return fn


def _seeds_native(cam, cfg=PCFG):
    """The anchor seeds the tracker uses (the grid inset by half an LK window)."""
    small, scale = P.downscale(np.zeros((cam.height, cam.width), np.uint8), cfg.process_scale)
    return P.to_native(P.seed_grid(small.shape, cfg.grid_side, margin=cfg.lk_win // 2), scale)


# ── an independent ground truth (numpy only — none of intake.parallax) ───

def _dlt_homography(a, b):
    """Least-squares homography a → b by the normalised DLT (Hartley),
    written here independently of the module under test."""
    def norm(p):
        c = p.mean(axis=0)
        s = np.sqrt(2.0) / np.mean(np.linalg.norm(p - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    Ta, Tb = norm(a), norm(b)
    ah = (np.c_[a, np.ones(len(a))] @ Ta.T)
    bh = (np.c_[b, np.ones(len(b))] @ Tb.T)
    rows = []
    for (x, y, w), (u, v, t) in zip(ah, bh):
        rows.append([0, 0, 0, -t * x, -t * y, -t * w, v * x, v * y, v * w])
        rows.append([t * x, t * y, t * w, 0, 0, 0, -u * x, -u * y, -u * w])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    return H / H[2, 2]


def _transfer(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def _rigid_seeds(sess, i, j, cfg=PCFG):
    """The anchor-i seeds (native px) the parallax of frame j is read on: the
    chain re-run on the session's own frames, its trusted survivors, and
    their F-inliers (the rigid set of an ``epipolar`` verdict)."""
    small = [P.downscale(Q.read_gray(sess.frame_path(sess.frame_numbers[k])), cfg.process_scale)
             for k in range(i, j + 1)]
    chain = P.AnchorChain(sess.frame_numbers[i], small[0][0], small[0][1], cfg,
                          native_wh=(sess.cam.width, sess.cam.height))
    for k in range(1, len(small)):
        chain.step(sess.frame_numbers[i + k], small[k][0])
    a, b = chain.positions_native(chain.trusted())
    fit = P.fit_two_view(a, b, cfg)
    return a[fit["in_f"]]


def _gt_parallax(sess, i, j, cfg=PCFG, seeds=None):
    """The parallax of the baseline i → j on NOISE-FREE correspondences: the
    seeds of render i (default: the whole grid) unprojected with the
    ground-truth depth, projected into pose j; q-quantile of the symmetric
    transfer error w.r.t. the least-squares homography of all of them."""
    cam = sess.cam
    uv = _seeds_native(cam, cfg) if seeds is None else seeds
    r = np.clip(np.round(uv[:, 1]).astype(int), 0, cam.height - 1)
    c = np.clip(np.round(uv[:, 0]).astype(int), 0, cam.width - 1)
    z = sess.renders[i].depth[r, c].astype(np.float64)
    hit = z > 0
    X = S.unproject(cam, sess.poses[i], uv[hit], z[hit])
    uvb, _, front = S.project(cam, sess.poses[j], X)
    inside = (front & (uvb[:, 0] >= 0) & (uvb[:, 0] <= cam.width - 1)
              & (uvb[:, 1] >= 0) & (uvb[:, 1] <= cam.height - 1))
    a, b = uv[hit][inside], uvb[inside]
    H = _dlt_homography(a, b)
    Hi = np.linalg.inv(H)
    sym = (np.linalg.norm(_transfer(H, a) - b, axis=1) + np.linalg.norm(_transfer(Hi, b) - a, axis=1)) / 2
    return float(np.quantile(sym, cfg.parallax_quantile))


def _warnings(result, kind):
    return [w for w in result["warnings"] if w["kind"] == kind]


def _moving(result, start=STILL_PREFIX):
    return [f for f in result["frames"] if f["frame"] > start]


def _covered(result, kind, frames):
    cov = set()
    for w in _warnings(result, kind):
        cov |= set(range(w["frame_start"], w["frame_end"] + 1))
    return sum(1 for f in frames if f["frame"] in cov)


# ── sequences (rendered once per module) ─────────────────────────────────

@pytest.fixture(scope="module")
def scene():
    return S.room_scene(seed=0)


@pytest.fixture(scope="module")
def cam():
    return S.default_camera()


@pytest.fixture(scope="module")
def rotation(tmp_path_factory, scene, cam):
    poses = _with_still(S.walk_poses("rotate", 30, start=S.room_start_pose(), yaw_deg_per_frame=2.0))
    return _run(tmp_path_factory.mktemp("rotation"), scene, cam, poses)


@pytest.fixture(scope="module")
def rotation_slow(tmp_path_factory, scene, cam):
    poses = _with_still(S.walk_poses("rotate", 60, start=S.room_start_pose(), yaw_deg_per_frame=1.0))
    return _run(tmp_path_factory.mktemp("rotation_slow"), scene, cam, poses)


WALK = dict(start=S.room_start_pose(x_m=1.2), direction="right")


@pytest.fixture(scope="module")
def translation(tmp_path_factory, scene, cam):
    walk = S.walk_poses("translate", 50, step_m=0.05, **WALK)
    return _run(tmp_path_factory.mktemp("translation"), scene, cam, _with_still(walk))


@pytest.fixture(scope="module")
def short_walk(tmp_path_factory, scene, cam):
    """A short walk (3 still frames, 14 × 5 cm) for the tests that re-run the
    stage on the same frames (determinism, stamps, logs)."""
    walk = S.walk_poses("translate", 14, step_m=0.05, **WALK)
    return _run(tmp_path_factory.mktemp("short_walk"), scene, cam, _with_still(walk, 3))


@pytest.fixture(scope="module")
def translation_bare(tmp_path_factory, scene, cam):
    walk = S.walk_poses("translate", 50, step_m=0.05, **WALK)
    return _run(tmp_path_factory.mktemp("translation_bare"), scene, cam, walk)


@pytest.fixture(scope="module")
def density(tmp_path_factory, scene, cam):
    """One 1.6 m sideways walk rendered at 2.5 cm per frame; 1x = every other
    frame (5 cm), 2x = every frame — the same poses, twice the frames."""
    base = S.walk_poses("translate", 65, step_m=0.025, **WALK)
    one = _run(tmp_path_factory.mktemp("density_1x"), scene, cam, _with_still(base[::2], 3))
    two = _run(tmp_path_factory.mktemp("density_2x"), scene, cam, _with_still(base, 3))
    return one, two


ROT_DYN_START = 4          # still frames before the pan
ROT_DYN_N = 16             # frames of the 2°/frame pan with the person crossing


@pytest.fixture(scope="module")
def rotation_with_person(tmp_path_factory, scene, cam):
    rot = _with_still(S.walk_poses("rotate", ROT_DYN_N, start=S.room_start_pose(),
                                   yaw_deg_per_frame=2.0), ROT_DYN_START)
    # the person walks across the view (−0.3 m → +0.15 m in camera x, 3 cm per
    # frame) while the camera pans 2° per frame
    fn = _person_scene(scene, lambda i: (-0.3 + 0.03 * (i - ROT_DYN_START))
                       if i >= ROT_DYN_START else None)
    return _run(tmp_path_factory.mktemp("rot_person"), scene, cam, rot, scene_fn=fn)


@pytest.fixture(scope="module")
def still_with_person(tmp_path_factory, scene, cam):
    poses = S.walk_poses("still", 20, start=S.room_start_pose())
    fn = _person_scene(scene, lambda i: -0.8 + 0.05 * i)
    return _run(tmp_path_factory.mktemp("still_person"), scene, cam, poses, scene_fn=fn)


@pytest.fixture(scope="module")
def light_change(tmp_path_factory, scene, cam):
    poses = S.walk_poses("still", 30, start=S.room_start_pose())
    return _run(tmp_path_factory.mktemp("light"), scene, cam, poses,
                light=list(np.linspace(0.6, 1.4, 30)))


@pytest.fixture(scope="module")
def wall_and_box(scene):
    """The back wall filling the view 1.5 m away and a textured box floating
    0.7 m from the camera in front of it (the room's own boxes left out)."""
    box = S.Box((0.2, 1.5, 7.4), (0.4, 0.4, 0.4), 0.0,
                S.make_texture("checker", 256, seed=11, contrast=0.8), "front_box", "box")
    return S.Scene(quads=list(scene.quads), name="wall_and_box", boxes=[box])


@pytest.fixture(scope="module")
def dominant_plane(tmp_path_factory, wall_and_box, cam):
    """Sideways walk, 2 cm per frame over 20 frames, that keeps the box in view."""
    walk = S.walk_poses("translate", 20, start=WALL_POSE, step_m=0.02, direction="right")
    return _run(tmp_path_factory.mktemp("dominant"), wall_and_box, cam, _with_still(walk))


# ── the test config is the production one ────────────────────────────────

def test_test_config_mirrors_production():
    """Every bound the tests exercise is the value config.yaml ships; only the
    tracker's scale / grid / window are sized for 320x240 frames."""
    with open(SERVER / "config.yaml") as f:
        prod = load_intake_config(yaml.safe_load(f)).parallax
    for f in fields(ParallaxConfig):
        if f.name not in SIZED_FOR_TEST:
            assert getattr(PCFG, f.name) == getattr(prod, f.name), f.name


# ── unit pieces ──────────────────────────────────────────────────────────

def test_seed_grid_downscale_and_native_round_trip():
    seeds = P.seed_grid((60, 80), 4)
    assert seeds.shape == (16, 2) and seeds.dtype == np.float32
    assert np.allclose(seeds[0], [9.5, 7.0]) and np.allclose(seeds[-1], [69.5, 52.0])
    gray = np.zeros((240, 320), dtype=np.uint8)
    small, scale = P.downscale(gray, 0.5)
    assert small.shape == (120, 160) and scale == (0.5, 0.5)
    same, unit = P.downscale(gray, 1.0)
    assert same is gray and unit == (1.0, 1.0)
    nat = P.to_native(np.array([[79.5, 59.5]]), scale)
    assert np.allclose(nat, [[159.5, 119.5]])
    odd, sc = P.downscale(np.zeros((241, 321), dtype=np.uint8), 0.25)
    assert odd.shape == (60, 80) and sc == (80 / 321, 60 / 241)
    A = P.small_to_native_matrix(sc)
    pts = np.array([[0.0, 0.0], [10.5, 3.25], [79.0, 59.0]])
    assert np.allclose(P._apply_h(A, pts), P.to_native(pts, sc))
    H = np.array([[1.01, 0.02, 3.0], [-0.01, 0.99, -2.0], [1e-5, -2e-5, 1.0]])
    Hs = P.homography_to_process(H, sc)
    assert np.allclose(P._apply_h(A, P._apply_h(Hs, pts)), P._apply_h(H, P._apply_h(A, pts)))


def test_symmetric_transfer_error_against_a_hand_computation():
    """(|H·a − b| + |H⁻¹·b − a|) / 2, pinned on three points computed by hand
    for H = scale 2 + shift (1, −1): exact point → 0; a2=(1,1)→b2=(6,5):
    forward |(3,1)−(6,5)| = 5, backward |(2.5,3)−(1,1)| = 2.5 → 3.75;
    a3=(2,−1)→b3=(5,−1): forward |(5,−3)−(5,−1)| = 2, backward |(2,0)−(2,−1)|
    = 1 → 1.5. Then a projective H against an independent numpy computation."""
    H = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]])
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
    b = np.array([[1.0, -1.0], [6.0, 5.0], [5.0, -1.0]])
    got = P.symmetric_transfer_error(H, np.linalg.inv(H), a, b)
    assert np.allclose(got, [0.0, 3.75, 1.5], atol=1e-12)
    Hp = np.array([[1.02, 0.01, 4.0], [-0.03, 0.97, 2.0], [2e-4, -1e-4, 1.0]])
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 320, (50, 2))
    b = _transfer(Hp, a) + rng.normal(0, 1.5, (50, 2))
    fwd = np.linalg.norm(_transfer(Hp, a) - b, axis=1)
    bwd = np.linalg.norm(_transfer(np.linalg.inv(Hp), b) - a, axis=1)
    assert np.allclose(P.symmetric_transfer_error(Hp, np.linalg.inv(Hp), a, b), (fwd + bwd) / 2)


def test_least_squares_reference_is_the_independent_dlt():
    """The parallax reference (least squares over all rigid tracks) agrees with
    the test's own DLT, and its reading is the q-quantile of the residuals."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 320, (200, 2))
    H = np.array([[1.01, 0.02, 3.0], [-0.01, 0.99, -2.0], [1e-4, -5e-5, 1.0]])
    b = _transfer(H, a) + rng.normal(0, 0.3, (200, 2))
    Href, sym = P.least_squares_reference(a, b)
    Hd = _dlt_homography(a, b)
    assert np.abs(_transfer(Href / Href[2, 2], a) - _transfer(Hd, a)).max() < 0.05
    Hi = np.linalg.inv(Hd)
    ind = (np.linalg.norm(_transfer(Hd, a) - b, axis=1) + np.linalg.norm(_transfer(Hi, b) - a, axis=1)) / 2
    assert np.quantile(sym, 0.9) == pytest.approx(np.quantile(ind, 0.9), rel=0.05)


def test_pair_discriminant_tells_a_rotation_from_a_plane_translation():
    """K-free: a pinhole rotation's homography has a complex eigenvalue pair
    (D = −sin²θ); a plane under translation (homology / elation) has all-real
    eigenvalues (D ≥ 0); conjugation by any K changes nothing."""
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    th = np.radians(3.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    Hr = K @ R @ np.linalg.inv(K)
    assert P.pair_discriminant(P._det_normalised(Hr)) == pytest.approx(-np.sin(th) ** 2, rel=1e-6)
    t, n, d = np.array([0.1, 0, 0.05]), np.array([0, 0, 1.0]), 2.0
    Hh = K @ (np.eye(3) + np.outer(t, n) / d) @ np.linalg.inv(K)
    assert P.pair_discriminant(P._det_normalised(Hh)) >= 0.0
    He = K @ (np.eye(3) + np.outer([0.1, 0, 0], n) / d) @ np.linalg.inv(K)      # n·t = 0
    assert abs(P.pair_discriminant(P._det_normalised(He))) < 1e-12


def _synthetic_pair(kind, rng, n_static=400, n_blob=100, noise=0.05):
    """Correspondences a → b (320x240 px, f 228) for the rigidity unit test:
    a static room-like scene (depths 2–7 m) seen under ``kind`` + a coherent
    blob moving 14 px sideways on its own."""
    K = np.array([[228.5, 0, 159.5], [0, 228.5, 119.5], [0, 0, 1.0]])
    a = rng.uniform([10, 10], [310, 230], (n_static, 2))
    z = rng.uniform(2.0, 7.0, n_static)
    X = np.c_[(a[:, 0] - 159.5) / 228.5, (a[:, 1] - 119.5) / 228.5, np.ones(n_static)] * z[:, None]
    if kind == "rotation":
        th = np.radians(4.0)
        R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        Xb = X @ R.T
    elif kind == "still":
        Xb = X.copy()
    else:                                     # sideways translation of 15 cm
        Xb = X + np.array([0.15, 0.0, 0.0])
    b = (Xb @ K.T)
    b = b[:, :2] / b[:, 2:3]
    ab = rng.uniform([120, 60], [200, 180], (n_blob, 2))
    bb = ab + np.array([14.0, 0.0]) if kind != "rotation" else \
        _transfer(K @ R @ np.linalg.inv(K), ab) + np.array([14.0, 0.0])
    a = np.r_[a, ab] + rng.normal(0, noise, (n_static + n_blob, 2))
    b = np.r_[b, bb] + rng.normal(0, noise, (n_static + n_blob, 2))
    return a, b, np.r_[np.zeros(n_static, bool), np.ones(n_blob, bool)]


@pytest.mark.parametrize("kind,verdict", [("rotation", "rotation"), ("still", "still"),
                                          ("translation", "epipolar")])
def test_rigidity_verdict_on_known_correspondences(kind, verdict):
    """A coherent blob moving by itself on 20 % of the tracks: under a pure
    rotation or a still camera the majority testifies the camera did not
    translate and the blob leaves the rigid set; under a translation of a
    multi-depth scene the verdict is epipolar (F fixed by the majority)."""
    rng = np.random.default_rng(7)
    a, b, blob = _synthetic_pair(kind, rng)
    # the floor a tracker with this noise measures: the q-quantile of the symmetric
    # transfer error of noise-only tracks (0.05 px per coordinate on a and on b —
    # a 2-D Gaussian of sigma 0.05·√2, whose q-quantile radius is sigma·√(−2 ln(1 − q)))
    floor = 0.05 * np.sqrt(2.0) * np.sqrt(-2.0 * np.log(1.0 - PCFG.parallax_quantile))
    m = P.measure_tracks(a, b, PCFG, pp=(159.5, 119.5), floor_px=floor, seed=(0, 1))
    assert not m["lost"] and m["verdict"] == verdict, m["rigidity"]
    if verdict in ("rotation", "still"):
        assert not np.any(m["rigid"] & blob)                  # the blob is out
        assert m["parallax_px"] < PCFG.witness_min_parallax_px
    if verdict == "rotation":
        rot = m["rigidity"]["rotation"]
        assert rot["evident"] and rot["consistent"]
        assert rot["f_px"] == pytest.approx(228.5, rel=0.05)
        assert rot["angle_deg"] == pytest.approx(4.0, abs=0.2)
        assert np.allclose(rot["pp_px"], (159.5, 119.5), atol=2.0)     # fitted, not assumed
    if verdict == "epipolar":
        assert m["parallax_px"] > 10 * PCFG.witness_min_parallax_px / 5   # real parallax is read
        assert m["rigidity"]["why"].startswith("the off-homography F-inliers are not a minority")
    # a rotation fit above the measured floor is not a rotation
    if kind == "rotation":
        tight = P.measure_tracks(a + rng.normal(0, 1.0, a.shape), b, PCFG, pp=(159.5, 119.5),
                                 floor_px=1e-3, seed=(0, 1))
        assert tight["verdict"] != "rotation"


def test_track_step_and_measure_a_pure_image_shift():
    """A frame and its own copy shifted by 3 px: the tracks read the shift, a
    shift IS a homography (parallax at the noise); too few tracks → lost with
    its reason."""
    tex = S.make_texture("noise", 256, seed=3)
    a = cv2.resize(tex, (320, 240), interpolation=cv2.INTER_AREA)
    M = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]])
    b = cv2.warpAffine(a, M, (320, 240), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a_s, scale = P.downscale(a, PCFG.process_scale)
    b_s, _ = P.downscale(b, PCFG.process_scale)
    seeds = P.seed_grid(a_s.shape, PCFG.grid_side)
    pts_a, pts_b, ok, fb = P.track_step(a_s, b_s, seeds, PCFG, scale)
    assert pts_a.shape == pts_b.shape == (PCFG.grid_side ** 2, 2)
    assert ok.sum() > 0.8 * len(ok) and np.all(np.isfinite(fb[ok]))
    na, nb = P.to_native(pts_a[ok], scale), P.to_native(pts_b[ok], scale)
    m = P.measure_tracks(na, nb, PCFG, pp=(159.5, 119.5), floor_px=0.1)
    assert not m["lost"] and abs(m["disp_px"] - 3.0) < 0.15
    assert m["parallax_px"] < 0.3 and m["n_h_inliers"] > 0.9 * len(na)
    lost = P.measure_tracks(na[:10], nb[:10], PCFG, pp=(159.5, 119.5), floor_px=0.1)
    assert lost["lost"] and lost["reason"] == "too_few_tracks"
    assert "too_few_rigid_tracks" in P.LOST_REASONS


def test_twin_is_a_zero_parallax_warp():
    tex = cv2.resize(S.make_texture("noise", 256, seed=3), (160, 120), interpolation=cv2.INTER_AREA)
    H = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    shifted = cv2.warpAffine(tex, H[:2], (160, 120), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    assert np.array_equal(P.twin_image(tex, H), shifted)


# ── (a) pure rotation: no keyframe, the whole run flagged ────────────────

@pytest.mark.parametrize("which,n_rot", [("rotation", 30), ("rotation_slow", 60)])
def test_pure_rotation_earns_no_keyframe_and_is_flagged(which, n_rot, request):
    """2°/frame x 30 and 1°/frame x 60 after a still prefix: one keyframe, ≥ 90 %
    of the rotated frames flagged pure_rotation AND covered by warnings (the
    anchor-based reading stays at its floor while the displacement grows); the
    majority's verdict is a rotation of a pinhole whose fitted focal length is
    the camera's."""
    sess, _, r = request.getfixturevalue(which)
    assert r["n_keyframes"] == 1 and r["keyframes"][0]["frame"] == 0
    assert r["keyframes"][0]["reason"] == "first_usable_frame"
    rot = _moving(r)
    n = n_rot - 1                  # the first rotated pose repeats the still one
    assert len(rot) == n
    flagged = sum(f["pure_rotation"] for f in rot)
    assert flagged >= 0.9 * n
    assert _covered(r, "pure_rotation", rot) >= 0.9 * n
    assert all(f["parallax_px"] < PCFG.witness_min_parallax_px for f in rot)
    verd = [f["verdict"] for f in rot]
    assert verd.count("rotation") >= 0.9 * n
    fs = [f["rigidity"]["rotation"]["f_px"] for f in rot if f["verdict"] == "rotation"]
    assert np.median(fs) == pytest.approx(sess.cam.fx, rel=0.03)
    assert all(f["floor_source"] == "twin+fb" for f in rot)
    assert "at its own noise floor" in _warnings(r, "pure_rotation")[0]["detail"]
    static = _warnings(r, "static")
    assert len(static) == 1 and static[0]["frame_start"] == 1
    assert static[0]["frame_end"] == STILL_PREFIX
    assert not _warnings(r, "dynamic_content") and r["n_witness"] == 1


def test_rotation_on_a_distorted_lens_declared_limit(tmp_path, scene, rotation):
    """A homography does not explain a rotation seen through a distorted lens:
    the residual reads as a little parallax. Measured: it stays far under a
    quantum (no keyframe) and above the undistorted reading."""
    poses = _with_still(S.walk_poses("rotate", 30, start=S.room_start_pose(), yaw_deg_per_frame=2.0))
    _, _, rd = _run(tmp_path, scene, S.default_camera(distortion=True), poses)
    _, _, ru = rotation
    assert rd["n_keyframes"] == 1
    med_d = np.median([f["parallax_px"] for f in _moving(rd)])
    med_u = np.median([f["parallax_px"] for f in _moving(ru)])
    assert med_d > med_u                                  # the leak is visible in the numbers
    assert max(f["parallax_px"] for f in _moving(rd)) < PCFG.parallax_quantum_px / 2


# ── (a') a person crossing during a pure rotation / in front of a still camera

def test_moving_object_during_a_rotation_earns_no_keyframe(rotation_with_person):
    """The acceptance probe: a textured slab on ~20 % of the anchor's tracks,
    sliding across the view during a 2°/frame pan. F alone accepts it (an
    object moving along a line satisfies a degenerate F); the majority's
    rotation verdict leaves it out of the rigid set: no keyframe, the pan
    still flagged as a rotation, the run reported as dynamic_content — and
    counted, its tracks would have read as parallax."""
    sess, _, r = rotation_with_person
    uv = np.round(_seeds_native(sess.cam)).astype(int)
    ids = sess.renders[ROT_DYN_START].quad_id[uv[:, 1], uv[:, 0]]
    labels = np.array([q.label for q in _person_scene(sess.scene, lambda i: 0.0)(0, sess.poses[0]).all_quads()])
    share = float(np.mean((ids >= 0) & (labels[np.clip(ids, 0, None)] == "person")))
    assert 0.15 <= share <= 0.3, share                     # ~20 % of the tracks
    assert r["n_keyframes"] == 1 and r["n_witness"] == 1
    pan = [f for f in r["frames"] if f["frame"] > ROT_DYN_START]
    n = ROT_DYN_N - 1              # the first rotated pose repeats the still one
    assert len(pan) == n
    assert sum(f["verdict"] == "rotation" for f in pan) >= 0.9 * n
    assert max(f["parallax_px"] for f in pan) < PCFG.witness_min_parallax_px
    assert max(f["parallax_all_px"] for f in pan) >= PCFG.witness_min_parallax_px
    assert _covered(r, "pure_rotation", pan) >= 0.9 * n
    assert _covered(r, "dynamic_content", pan) >= 0.6 * n
    assert max(f["rigidity"]["n_dynamic_marked"] for f in pan) > 0.1 * len(uv)
    assert "moving on its own" in _warnings(r, "dynamic_content")[0]["detail"]


def test_moving_object_in_front_of_a_still_camera(still_with_person):
    """A still camera and a person walking across: every verdict is still, the
    reading stays at the floor, and the whole run is reported as
    dynamic_content — the windows the person covers stop showing the
    anchor's patch while the camera does not translate (occlusion is dynamic
    evidence under a no-translation verdict), and the person's own tracks
    move on their own."""
    sess, _, r = still_with_person
    assert r["n_keyframes"] == 1 and r["n_witness"] == 1
    assert all(f["verdict"] == "still" for f in r["frames"])
    assert max(f["parallax_px"] for f in r["frames"]) < PCFG.witness_min_parallax_px
    assert _covered(r, "dynamic_content", r["frames"]) >= 0.9 * len(r["frames"])
    assert max(f["rigidity"]["n_occluded_as_dynamic"] for f in r["frames"]) > 0
    assert _warnings(r, "static")


# ── (b) translation ──────────────────────────────────────────────────────

def test_translation_keyframes_follow_ground_truth_parallax(translation):
    """Each keyframe closes a window whose measured parallax from its anchor
    follows the independent ground truth of that baseline; nothing is flagged
    as a rotation; the windows tile the walk."""
    sess, quality, r = translation
    idx = {f: i for i, f in enumerate(sess.frame_numbers)}
    moving = _moving(r)
    assert not any(f["lost"] for f in moving)
    assert r["n_pure_rotation"] == 0 and not _warnings(r, "pure_rotation")
    assert all(f["verdict"] == "epipolar" for f in moving if f["disp_px"] > 2.0)
    kfs = r["keyframes"]
    assert len(kfs) >= 4
    by_frame = {f["frame"]: f for f in quality["frames"]}
    for k in kfs[1:]:
        assert k["reason"] == "quantum"
        w = k["window"]
        lo = (1 - PCFG.keyframe_band_frac) * PCFG.parallax_quantum_px
        hi = (1 + PCFG.keyframe_band_frac) * PCFG.parallax_quantum_px
        assert lo <= k["parallax_from_anchor_px"] <= hi
        assert k["parallax_over_quantum"] == pytest.approx(k["parallax_from_anchor_px"] / PCFG.parallax_quantum_px)
        assert w["max_parallax_px"] >= PCFG.parallax_quantum_px and w["band_px"] == [lo, hi]
        assert w["choice"] == "sharpest_in_band" and k["anchor_is_keyframe"]
        i, j = idx[k["anchor"]], idx[k["frame"]]
        gt = _gt_parallax(sess, i, j, seeds=_rigid_seeds(sess, i, j))
        assert k["parallax_from_anchor_px"] == pytest.approx(gt, rel=0.15)
        # the sharpest of the band
        band = [f for f in r["frames"] if f["anchor"] == w["anchor"]
                and w["frame_start"] <= f["frame"] <= w["frame_end"]
                and f["parallax_px"] is not None and lo <= f["parallax_px"] <= hi]
        if band:        # frames after the keyframe were re-measured from it
            assert by_frame[k["frame"]]["sharp_rank"] >= max(
                by_frame[f["frame"]]["sharp_rank"] for f in band if f["frame"] <= k["frame"])
    assert [k["anchor"] for k in kfs[1:]] == [k["frame"] for k in kfs[:-1]]
    tail = [w for w in r["windows"] if w["closed_by"] == "end"]
    assert len(tail) <= 1 and all(w["keyframe"] is None and w["reason"] for w in tail)
    assert r["n_still"] == STILL_PREFIX and len(_warnings(r, "static")) == 1
    assert not _warnings(r, "dynamic_content")


def test_same_walk_same_keyframes_with_or_without_a_still_prefix(translation, translation_bare):
    """No session floor any more: a recording that starts still and one that
    does not read the same walk into the same keyframe count (±1)."""
    _, _, a = translation
    _, _, b = translation_bare
    assert abs(a["n_keyframes"] - b["n_keyframes"]) <= 1
    assert b["n_pure_rotation"] == 0 and not _warnings(b, "pure_rotation")


def test_same_walk_same_keyframes_at_twice_the_frame_density(density):
    """The acceptance probe: one walk at 1x and 2x frame density → the same
    keyframe count (±1) and the same number of windows closed by the quantum
    (anchor-based parallax does not depend on how many links chained it)."""
    (s1, _, one), (s2, _, two) = density
    assert len(s2.frame_numbers) > 1.9 * (len(s1.frame_numbers) - 3)
    assert one["n_keyframes"] >= 3
    assert abs(one["n_keyframes"] - two["n_keyframes"]) <= 1
    for r in (one, two):
        assert r["n_pure_rotation"] == 0 and not _warnings(r, "pure_rotation")
        assert not _warnings(r, "static")[1:]


# ── (b') a translation across a DOMINANT plane ───────────────────────────

def test_dominant_plane_translation_is_not_a_rotation(dominant_plane, wall_and_box):
    """The wall holds ≥ 70 % of the tracks, the box the rest. The homography
    majority is an elation (no rotation evidence), so the verdict stays
    epipolar and the box's parallax is read: several keyframes, no
    pure-rotation warning, the keyframes' parallax follows the ground truth."""
    sess, quality, r = dominant_plane
    uv = np.round(_seeds_native(sess.cam)).astype(int)
    wall = wall_and_box.quad_index("wall_back")
    box = {wall_and_box.quad_index(f"front_box_{f}") for f in
           ("bottom", "top", "front", "back", "left", "right")}
    for rd in sess.renders:
        ids = rd.quad_id[uv[:, 1], uv[:, 0]]
        assert set(np.unique(ids)) <= {wall} | box
        assert np.mean(ids == wall) >= 0.70
        assert np.mean(np.isin(ids, list(box))) > 1.0 - PCFG.parallax_quantile
    moving = _moving(r)
    assert not any(f["lost"] for f in moving)
    assert all(f["verdict"] == "epipolar" for f in moving if f["disp_px"] > 2.0)
    assert r["n_keyframes"] > 1 and not _warnings(r, "pure_rotation")
    idx = {f: i for i, f in enumerate(sess.frame_numbers)}
    for k in r["keyframes"][1:]:
        i, j = idx[k["anchor"]], idx[k["frame"]]
        gt = _gt_parallax(sess, i, j, seeds=_rigid_seeds(sess, i, j))
        assert k["parallax_from_anchor_px"] == pytest.approx(gt, rel=0.2)


# ── (c) light change without motion ──────────────────────────────────────

def test_light_change_without_motion(light_change):
    sess, quality, r = light_change
    assert quality["n_usable"] == 30
    assert r["n_keyframes"] == 1 and r["n_witness"] == 1
    assert r["n_still"] == r["n_measured"] == 29
    assert all(f["verdict"] == "still" and f["disp_px"] < PCFG.warn_static_disp_px
               for f in r["frames"])
    static = _warnings(r, "static")
    assert len(static) == 1 and (static[0]["frame_start"], static[0]["frame_end"]) == (1, 29)
    assert static[0]["n_frames"] == 29 and "did not move" in static[0]["detail"]
    assert r["warnings"] == static
    assert len(r["windows"]) == 1 and r["windows"][0]["closed_by"] == "end"


# ── (d) witnesses ────────────────────────────────────────────────────────

def test_witnesses_dedup_against_the_last_chosen_witness(translation):
    """Every witness past the first is measured from the previous CHOSEN
    witness (keyframes included — they reset the anchor); a dedup witness
    reached the bar from it; keyframes ⊂ witnesses."""
    _, _, r = translation
    kf = [k["frame"] for k in r["keyframes"]]
    wit = r["witnesses"]
    frames = [w["frame"] for w in wit]
    assert frames == sorted(frames) and len(set(frames)) == len(frames)
    assert set(kf) <= set(frames)
    assert all(w["is_keyframe"] == (w["frame"] in kf) for w in wit)
    assert all(w["reason"] in P.WITNESS_REASONS for w in wit)
    assert len(wit) > len(kf) + 5
    assert wit[0]["frame"] == 0 and wit[0]["reason"] == "first_usable_frame"
    for prev, cur in zip(wit, wit[1:]):
        if cur["reason"] == "dedup":
            assert cur["anchor"] == prev["frame"]
            assert cur["parallax_from_anchor_px"] >= PCFG.witness_min_parallax_px
        elif cur["reason"] == "keyframe":
            assert cur["is_keyframe"]


# ── declared limit: a single plane ───────────────────────────────────────

def test_single_plane_reads_as_rotation_declared_limit(tmp_path, scene, translation):
    """Facing the back wall alone and walking sideways 0.87 m: every track lies
    on one plane, a homography explains all of it, the parallax reads its
    floor, no keyframe is added and the run is flagged as a pure rotation."""
    walk = S.walk_poses("translate", 30, start=WALL_POSE, step_m=0.03, direction="right")
    sess, _, r = _run(tmp_path, scene, S.default_camera(), _with_still(walk))
    assert set(np.unique(sess.renders[0].quad_id)) == {scene.quad_index("wall_back")}
    assert S.chainage(sess.poses)[-1] > 0.8
    assert r["n_keyframes"] == 1
    moving = [f for f in _moving(r) if f["disp_px"] >= PCFG.warn_rotation_min_disp_px]
    assert sum(f["pure_rotation"] for f in moving) >= 0.9 * len(moving)
    assert _warnings(r, "pure_rotation")[-1]["frame_end"] == sess.frame_numbers[-1]


# ── the tracker: every frame matched to the anchor, no chained drift ──────

def test_tracks_are_matched_to_the_anchor_not_chained(tmp_path, scene, cam):
    """A 2°/frame pan: every track's position at every frame is its window
    refined against the ANCHOR's own window — the error does not grow with
    the number of links (chained LK drifted 0.24 → 2.9 px median over 51
    links); checked against the ground-truth projection of every seed."""
    poses = S.walk_poses("rotate", 26, start=S.room_start_pose(), yaw_deg_per_frame=2.0)
    sess = S.write_session(tmp_path, scene, cam, poses, noise_sigma=NOISE_SIGMA)
    small = [P.downscale(Q.read_gray(sess.frame_path(f)), PCFG.process_scale) for f in sess.frame_numbers]
    chain = P.AnchorChain(0, small[0][0], small[0][1], PCFG, native_wh=(cam.width, cam.height))
    uv = chain.seeds_native
    r_ = np.clip(np.round(uv[:, 1]).astype(int), 0, cam.height - 1)
    c_ = np.clip(np.round(uv[:, 0]).astype(int), 0, cam.width - 1)
    X = S.unproject(cam, sess.poses[0], uv, sess.renders[0].depth[r_, c_].astype(np.float64))
    med = []
    for k in range(1, len(small)):
        chain.step(k, small[k][0])
        a, b = chain.positions_native(chain.valid)
        gt, _, _ = S.project(cam, sess.poses[k], X[chain.valid])
        med.append(float(np.median(np.linalg.norm(b - gt, axis=1))))
    assert max(med) < 0.35, med                      # native px, 50° of pan from the anchor
    assert np.mean(med[-5:]) < 2.0 * np.mean(med[:5]) + 0.1, med   # no growth with the links


def test_track_loss_windows_shorter_than_the_run_are_warned():
    """An anchor whose tracks are lost within warn_min_run_frames frames — a
    texture the tracker cannot hold re-anchors every frame or two, each
    re-measurement succeeds — is a coverage event: its frames are flagged
    tracking_lost and a run of them is warned (pure over the records)."""
    def rec(frame, closed_by, n_frames, lost=False):
        return {"frame": frame, "lost": lost, "reason": "too_few_tracks" if lost else None,
                "disp_px": 0.2, "parallax_px": 0.3, "floor_px": 0.3, "n_surviving": 45,
                "verdict": "still", "dynamic_moving_share": 0.0, "parallax_all_px": 0.3,
                "window_closed_by": closed_by, "window_n_frames": n_frames,
                "losses": [{"anchor": frame - 1, "reason": "too_few_rigid_tracks",
                            "n_surviving": 45, "n_rigid": 38}]}
    rows = [{"frame": f, "usable": True} for f in range(12)]
    recs = [rec(f, "coverage_break", 1) for f in range(1, 12)]
    flags = [P.frame_flags(r_, PCFG) for r_ in recs]
    assert all(f["tracking_lost"] for f in flags)
    w = [x for x in P.coverage_warnings(recs, rows, PCFG) if x["kind"] == "tracking_lost"]
    assert len(w) == 1 and w[0]["n_frames"] == 11 and "cannot hold an anchor" in w[0]["detail"]
    held = [rec(f, "coverage_break", PCFG.warn_min_run_frames) for f in range(1, 12)]
    assert not any(P.frame_flags(r_, PCFG)["tracking_lost"] for r_ in held)
    quantum = [rec(f, "quantum", 1) for f in range(1, 12)]
    assert not any(P.frame_flags(r_, PCFG)["tracking_lost"] for r_ in quantum)


def test_still_camera_on_a_marginal_texture(tmp_path, scene, cam):
    """A still camera facing a low-contrast wall (the verifiers' texture at
    contrast 0.06: the previous tracker gave several witnesses and no
    warning; at 0.03 one keyframe per frame): one keyframe, one witness, and
    the run warned as static — the displacement sits at the tracker's own
    measured floor."""
    wall = S.room_scene(seed=0)
    q = wall.quads[1]
    wall.quads[1] = S.TexturedQuad(q.origin, q.edge_u, q.edge_v,
                                   S.make_texture("noise", 256, seed=2, contrast=0.06),
                                   "wall_back", "wall")
    poses = S.walk_poses("still", 24, start=S.look_at_pose((-0.5, 1.5, 7.3), (-0.5, 1.5, 8.3)))
    sess, _, r = _run(tmp_path, wall, cam, poses)
    assert set(np.unique(sess.renders[0].quad_id)) == {wall.quad_index("wall_back")}
    assert r["n_keyframes"] == 1 and r["n_witness"] == 1
    assert not [w for w in r["windows"] if w["keyframe"] is not None]
    static = _warnings(r, "static")
    assert static and sum(w["n_frames"] for w in static) >= 0.9 * len(r["frames"])


# ── (a'') pure rotation past the field of view: coverage breaks, no keyframe

def test_half_turn_pan_earns_no_keyframe(tmp_path, scene, cam):
    """A 180° pan (5°/frame): the anchor's tracks leave the view several
    times; each loss is a COVERAGE BREAK — no keyframe without a measured
    baseline (the previous rule made one every ~50° at 0.7 px of parallax) —
    and the measurement continues from a pending anchor. One keyframe, the
    pan flagged as a pure rotation."""
    poses = _with_still(S.walk_poses("rotate", 36, start=S.room_start_pose(), yaw_deg_per_frame=5.0), 4)
    sess, _, r = _run(tmp_path, scene, cam, poses)
    assert r["n_keyframes"] == 1 and r["n_witness"] == 1
    breaks = [w for w in r["windows"] if w["closed_by"] in ("coverage_break", "tracking_lost")]
    assert len(breaks) >= 2 and all(w["keyframe"] is None for w in breaks)
    assert all("no keyframe" in w["reason"] for w in breaks)
    pan = [f for f in r["frames"] if f["frame"] > 4]
    assert _covered(r, "pure_rotation", pan) >= 0.9 * len(pan)
    assert max(f["parallax_px"] or 0.0 for f in pan) < PCFG.witness_min_parallax_px


def test_slow_pan_stays_at_its_floor(tmp_path, scene, cam):
    """A slow pan (0.1°/frame over 200 frames — 60 fps footage of a slow
    turn): the reading of a window matched to its anchor does not grow with
    the chain, so it stays at the measured floor: ≥ 90 % of the pan flagged
    and covered, no witness, no dynamic_content."""
    poses = _with_still(S.walk_poses("rotate", 201, start=S.room_start_pose(), yaw_deg_per_frame=0.1), 4)
    sess, _, r = _run(tmp_path, scene, cam, poses)
    assert r["n_keyframes"] == 1 and r["n_witness"] == 1
    moving = [f for f in r["frames"] if f["disp_px"] is not None
              and f["disp_px"] >= PCFG.warn_rotation_min_disp_px]
    assert len(moving) > 150
    assert sum(f["pure_rotation"] for f in moving) >= 0.9 * len(moving)
    assert _covered(r, "pure_rotation", moving) >= 0.9 * len(moving)
    assert not _warnings(r, "dynamic_content")


# ── illumination that varies across the frame ────────────────────────────

def _banded(renders, amplitude):
    """Rolling-shutter flicker: a gain 1 + A·sin(2π(row / (H/3) + 0.37·i))
    rolling down the frame (the verifiers' probe)."""
    out = []
    for i, r_ in enumerate(renders):
        rows = np.arange(r_.rgb.shape[0])[:, None, None]
        g = 1.0 + amplitude * np.sin(2 * np.pi * (rows / (r_.rgb.shape[0] / 3.0) + 0.37 * i))
        rgb = np.clip(np.round(r_.rgb.astype(float) * g), 0, 255).astype(np.uint8)
        out.append(S.Render(rgb=rgb, depth=r_.depth, normal=r_.normal, quad_id=r_.quad_id,
                            dynamic_mask=r_.dynamic_mask))
    return out


def _run_renders(session_dir, renders):
    frames_dir = Path(session_dir) / "frames"
    S.write_frames(frames_dir, renders)
    quality = Q.analyze_frames(frames_dir, QCFG, log=QUIET, heartbeat_s=1e-6)
    return P.run_parallax(frames_dir, quality, PCFG, log=QUIET, heartbeat_s=1e-6)


def test_rolling_bands_are_not_parallax(tmp_path, scene, cam):
    """±10 % bands rolling down the frame (a lamp flickering under a rolling
    shutter): every window is matched on its own mean and spread, so a gain
    that varies across the frame is not read as motion — a still camera
    earns one keyframe and one witness, a pure rotation no keyframe and its
    pan is flagged (the previous tracker: a quantum keyframe at 18 px on the
    rotation, two on the still camera)."""
    still = S.walk_poses("still", 20, start=S.room_start_pose())
    rs = _run_renders(tmp_path / "still", _banded(
        [S.render(scene, cam, c, noise_sigma=NOISE_SIGMA, seed=i) for i, c in enumerate(still)], 0.1))
    assert rs["n_keyframes"] == 1 and rs["n_witness"] == 1
    assert max(f["parallax_px"] or 0.0 for f in rs["frames"]) < PCFG.witness_min_parallax_px
    rot = _with_still(S.walk_poses("rotate", 24, start=S.room_start_pose(), yaw_deg_per_frame=2.0), 4)
    rr = _run_renders(tmp_path / "rot", _banded(
        [S.render(scene, cam, c, noise_sigma=NOISE_SIGMA, seed=i) for i, c in enumerate(rot)], 0.1))
    assert rr["n_keyframes"] == 1
    pan = [f for f in rr["frames"] if f["frame"] > 4]
    assert _covered(rr, "pure_rotation", pan) >= 0.9 * len(pan)


# ── the rotation verdict fits the principal point ────────────────────────

def test_person_during_a_pan_with_an_off_centre_principal_point(tmp_path, scene):
    """The principal point 2 % of the image off-centre (phones, stabilised
    crops) and a person crossing a 2°/frame pan: the pinhole-rotation fit
    finds the principal point instead of assuming the image centre, so the
    majority's verdict stays a rotation and the person no keyframe (the
    centre-pinned fit fell back to epipolar: a quantum keyframe at 11 px)."""
    c = S.default_camera()
    cam = S.BrownCamera(c.width, c.height, c.fx, c.fy, c.cx + 0.02 * c.width, c.cy + 0.02 * c.height)
    rot = _with_still(S.walk_poses("rotate", 30, start=S.room_start_pose(), yaw_deg_per_frame=2.0))
    fn = _person_scene(scene, lambda i: (-0.6 + 0.04 * (i - STILL_PREFIX)) if i >= STILL_PREFIX else None)
    sess, _, r = _run(tmp_path, scene, cam, rot, scene_fn=fn)
    assert r["n_keyframes"] == 1
    pan = [f for f in r["frames"] if f["frame"] > STILL_PREFIX]
    assert sum(f["verdict"] == "rotation" for f in pan) >= 0.9 * len(pan)
    assert sum(f["pure_rotation"] for f in pan) >= 0.9 * len(pan)
    pps = np.array([f["rigidity"]["rotation"]["pp_px"] for f in pan if f["verdict"] == "rotation"])
    assert np.allclose(np.median(pps, axis=0), (cam.cx, cam.cy), atol=0.02 * c.width)
    assert abs(np.median(pps[:, 0]) - (c.width - 1) / 2.0) > 0.01 * c.width   # not the centre


# ── objects moving on their own during a translation (B1 / B2) ───────────

@pytest.fixture(scope="module")
def walk_with_person(tmp_path_factory, scene, cam):
    """B1: the person crosses during the still prefix only; B2: the person
    crosses sideways during the walk (the verifiers' adv2 probe)."""
    walk = _with_still(S.walk_poses("translate", 50, step_m=0.05, **WALK))
    b1 = _run(tmp_path_factory.mktemp("b1"), scene, cam, walk,
              scene_fn=_person_scene(scene, lambda i: (-0.6 + 0.08 * i) if i <= STILL_PREFIX else None))
    b2 = _run(tmp_path_factory.mktemp("b2"), scene, cam, walk,
              scene_fn=_person_scene(scene, lambda i: (-0.8 + 0.03 * (i - STILL_PREFIX))
                                     if STILL_PREFIX <= i < STILL_PREFIX + 40 else None))
    return b1, b2


def test_person_during_the_still_prefix_leaves_the_walk_alone(walk_with_person, translation):
    """B1: the walk keeps its clean keyframes (±1) and the still prefix is
    reported as dynamic_content (the person covers the anchor's windows
    while the camera does not translate)."""
    (_, _, b1), _ = walk_with_person
    _, _, clean = translation
    assert abs(b1["n_keyframes"] - clean["n_keyframes"]) <= 1
    prefix = [f for f in b1["frames"] if f["frame"] <= STILL_PREFIX]
    assert _covered(b1, "dynamic_content", prefix) >= 0.8 * len(prefix)
    assert not _warnings(b1, "pure_rotation")


def _person_masks(sess, frames):
    """Exclusion masks as I2 would write them: the person's pixels (ground
    truth) of each frame, 255 = excluded."""
    labels = None
    out = {}
    for f in frames:
        i = sess.frame_numbers.index(f)
        sc = _person_scene(sess.scene, lambda k: 0.0)(0, sess.poses[0])
        labels = np.array([q.label for q in sc.all_quads()])
        ids = sess.renders[i].quad_id
        person_ids = np.flatnonzero(labels == "person")
        out[f] = np.isin(ids, person_ids)
    return out


def test_person_crossing_a_walk_is_declared_and_audited(walk_with_person, translation, tmp_path):
    """B2, the declared limit: a person crossing sideways during a sideways
    walk moves along the epipolar lines of the camera's own motion — F-
    consistent, read as structure, and it adds keyframes (measured here).
    I1 cannot tell it from the images; once I2 has masked it, the exclusion
    audit re-reads every keyframe without the masked tracks and warns
    (excluded_parallax) on the windows that closed on the person."""
    _, (sess, _, b2) = walk_with_person
    _, _, clean = translation
    assert b2["n_keyframes"] >= clean["n_keyframes"] + 2          # the declared inflation
    assert not _warnings(b2, "pure_rotation")
    P.write_selection(sess.session_dir, b2, PCFG)
    frames = sorted({k["frame"] for k in b2["keyframes"]} | {k["anchor"] for k in b2["keyframes"]
                                                             if k["anchor"] is not None})
    masks_dir = sess.session_dir / "intake" / "exclusion_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for f, m in _person_masks(sess, frames).items():
        if m.any():
            cv2.imwrite(str(masks_dir / f"{f:06d}.png"), np.where(m, 255, 0).astype(np.uint8))
    audit = P.audit_exclusions(sess.session_dir, masks_dir, PCFG, log=QUIET)
    rests = [k for k in audit["keyframes"] if k["verdict"] in ("rests_on_excluded", "too_few_tracks_left")]
    assert len(rests) >= 2
    assert all(k["n_excluded"] > 0 for k in rests)
    assert all(k["reading_without_excluded_px"] is None
               or k["reading_without_excluded_px"] < k["bar_px"] for k in rests)
    cov = json.loads((sess.session_dir / "intake" / "coverage_warnings.json").read_text())
    ex = [w for w in cov["warnings"] if w["kind"] == "excluded_parallax"]
    assert len(ex) == len(rests) and cov["exclusion_audit"]["n_rests_on_excluded"] == len(rests)
    doc = json.loads((sess.session_dir / "intake" / "exclusion_audit.json").read_text())
    assert doc["provenance"] == "tool_measured" and doc["inputs"]["masks"]
    # the audit is idempotent: a second pass replaces its own warnings
    P.audit_exclusions(sess.session_dir, masks_dir, PCFG, log=QUIET)
    cov2 = json.loads((sess.session_dir / "intake" / "coverage_warnings.json").read_text())
    assert [w for w in cov2["warnings"] if w["kind"] == "excluded_parallax"] == ex


# ── a hall: far texture at the sampling limit, a near box ────────────────

def test_hall_tracks_read_their_ground_truth(tmp_path):
    """The verifiers' hall (a 40 m room, a box 2.5 m ahead, 2 cm/frame): the
    chained tracker read 2× the geometric parallax — occlusion-boundary and
    aliasing tracks drifting along the epipolar lines. Every track is now
    matched to its anchor window and the appearance test removes what is not
    the anchor's patch: over the tracks whose anchor window does not straddle
    the box's silhouette, the reading follows the ground truth on the same
    tracks within the translation test's tolerance. DECLARED: a window that
    straddles the silhouette follows the side with the stronger texture (the
    near box's parallax at a far seed); with the box on ≈ (1 − q) of the
    tracks those windows move the q-quantile itself — measured, bounded
    below."""
    cam = S.default_camera()
    hall = S.room_scene(seed=0, width_m=40.0, depth_m=40.0, height_m=20.0, boxes=0)
    box = S.Box((0.3, 1.5, 3.5), (0.7, 0.7, 0.7), 0.0,
                S.make_texture("checker", 256, seed=11, contrast=0.8), "near_box", "box")
    sc = S.Scene(quads=list(hall.quads), name="hall_box", boxes=[box])
    poses = S.walk_poses("translate", 12, start=S.look_at_pose((0.0, 1.5, 1.0), (0.0, 1.5, 2.0)),
                         step_m=0.02, direction="right")
    sess = S.write_session(tmp_path, sc, cam, poses, noise_sigma=NOISE_SIGMA)
    small = [P.downscale(Q.read_gray(sess.frame_path(f)), PCFG.process_scale) for f in sess.frame_numbers]
    chain = P.AnchorChain(0, small[0][0], small[0][1], PCFG, native_wh=(cam.width, cam.height))
    for k in range(1, len(small)):
        m = chain.step(k, small[k][0])
    assert not m.lost and m.verdict == "epipolar"
    a, b = chain.positions_native(chain.trusted())
    # ground truth of the same seeds; windows straddling the box silhouette apart
    boxq = [i for i, q in enumerate(sc.all_quads()) if q.name.startswith("near_box")]
    half = (PCFG.lk_win // 2 + 1) / PCFG.process_scale
    straddle = []
    for x, y in a:
        ids = sess.renders[0].quad_id[int(max(0, y - half)):int(min(cam.height, y + half + 1)),
                                      int(max(0, x - half)):int(min(cam.width, x + half + 1))]
        inb = np.isin(ids, boxq)
        straddle.append(bool(inb.any() and (~inb).any()))
    straddle = np.array(straddle)
    r_ = np.clip(np.round(a[:, 1]).astype(int), 0, cam.height - 1)
    c_ = np.clip(np.round(a[:, 0]).astype(int), 0, cam.width - 1)
    X = S.unproject(cam, sess.poses[0], a, sess.renders[0].depth[r_, c_].astype(np.float64))
    gt, _, _ = S.project(cam, sess.poses[-1], X)
    keep = ~straddle
    assert np.median(np.linalg.norm(b[keep] - gt[keep], axis=1)) < 0.2
    _H, meas = P.least_squares_reference(a[keep], b[keep])
    Hg = _dlt_homography(a[keep], gt[keep])
    Hi = np.linalg.inv(Hg)
    ref = (np.linalg.norm(_transfer(Hg, a[keep]) - gt[keep], axis=1)
           + np.linalg.norm(_transfer(Hi, gt[keep]) - a[keep], axis=1)) / 2
    q = PCFG.parallax_quantile
    assert np.quantile(meas, q) == pytest.approx(np.quantile(ref, q), rel=0.15)
    # the declared straddling effect, measured: bounded by twice the ground truth
    assert m.parallax_px <= 2.0 * np.quantile(ref, q) + PCFG.witness_min_parallax_px


# ── frame-rate independence (the same walk at 60 / 30 / 15 fps) ──────────

def _render_pool(scene, cam, poses):
    """Render in parallel (fork pool; the renderer is deterministic)."""
    import multiprocessing as mp
    import os
    global _POOL_ARGS
    _POOL_ARGS = (scene, cam, poses)
    with mp.get_context("fork").Pool(max(1, min(16, (os.cpu_count() or 1)))) as pool:
        return pool.map(_render_one, range(len(poses)), chunksize=2)


_POOL_ARGS = None


def _render_one(i):
    scene, cam, poses = _POOL_ARGS
    return S.render(scene, cam, poses[i], noise_sigma=NOISE_SIGMA, seed=i, frame_index=i)


def test_same_walk_at_60_30_15_fps(tmp_path):
    """One sideways walk past a shelf 0.5 m away (1.9 m, ≥ 30 keyframes) rendered at
    1 cm per frame — 60 fps at 0.6 m/s — and read at 1×, ½× and ¼× the frame
    rate (the same renders, strided frame numbers). A keyframe is the
    sharpest frame whose reading lies in the SYMMETRIC band [(1 − b), (1 + b)]
    × quantum around the quantum (the window is followed until the reading
    leaves the band) — or, when the sampling skips the band, the frame whose
    reading is closest to the quantum — so the baseline a keyframe keeps does
    not depend on how many frames sampled the window (the one-sided band
    [0.75 q, reading] drifted toward 0.875 q at 60 fps: 63 / 57 / 50
    keyframes). Asserted: every keyframe inside the band; the walk's total
    measured baseline the same at every rate within one quantum; the counts
    within ±1 where the band is sampled, ±2 where one frame step is wider than
    the whole band (measured: 32 / 32 / 30 keyframes, total baseline 376 /
    372 / 372 px)."""
    cam = S.default_camera()
    room = S.room_scene(seed=0, boxes=0)
    shelf = S.Box((0.0, 0.7, 1.5), (6.0, 1.4, 0.2), 0.0, S.make_texture("noise", 1024, seed=21),
                  "shelf", "box")
    sc = S.Scene(quads=list(room.quads), name="shelf", boxes=[shelf])
    length = 1.9
    start = S.look_at_pose((-length / 2, 1.5, 1.0), (-length / 2, 1.5, 2.0))
    poses = _with_still(S.walk_poses("translate", int(round(length / 0.01)) + 1, start=start,
                                     step_m=0.01, direction="right"), 4)
    renders = _render_pool(sc, cam, poses)
    q = PCFG.parallax_quantum_px
    lo, hi = (1 - PCFG.keyframe_band_frac) * q, (1 + PCFG.keyframe_band_frac) * q
    counts, totals = {}, {}
    for stride in (1, 2, 4):
        idx = list(range(0, len(poses), stride))
        frames_dir = tmp_path / f"x{stride}" / "frames"
        S.write_frames(frames_dir, [renders[i] for i in idx], idx)
        quality = Q.analyze_frames(frames_dir, QCFG, log=QUIET, heartbeat_s=1e-6)
        r = P.run_parallax(frames_dir, quality, PCFG, log=QUIET, heartbeat_s=1e-6)
        kfs = r["keyframes"][1:]
        assert all(k["reason"] == "quantum" for k in kfs)
        # inside the band — or, where the sampling stepped over the whole band
        # (≈ 15 px per frame at 15 fps here), the reading closest to the quantum
        assert all(lo <= k["parallax_from_anchor_px"] <= hi for k in kfs
                   if k["window"]["choice"] == "sharpest_in_band"), stride
        assert all(k["window"]["n_candidates"] == 0 for k in kfs
                   if k["window"]["choice"] == "closest_to_quantum"), stride
        tail = r["windows"][-1]
        totals[stride] = (sum(k["parallax_from_anchor_px"] for k in kfs)
                          + (tail["max_parallax_px"] or 0.0 if tail["closed_by"] == "end" else 0.0))
        counts[stride] = r["n_keyframes"]
        assert not _warnings(r, "pure_rotation")
    assert min(counts.values()) >= 30, counts
    # the walk's measured baseline does not depend on the frame rate
    assert max(totals.values()) - min(totals.values()) <= q, totals
    # while the band is sampled (60 and 30 fps here: ≥ 1 frame per band) the
    # counts agree within one keyframe; at 15 fps one frame step (~16 px) is
    # wider than the whole band (6 px) and a keyframe keeps the frame closest
    # to the quantum — the declared bound: one more keyframe of difference
    assert abs(counts[1] - counts[2]) <= 1, counts
    assert abs(counts[1] - counts[4]) <= 2, counts


# ── a forward walk at the production scale ───────────────────────────────

def test_forward_walk_at_production_scale(tmp_path):
    """The verifiers' forward-walk failure at production scale (1280×720,
    process_scale 0.25, LK window 21): chained tracks drifted with the link
    count, the rigid set collapsed, the window never closed and the walk was
    flagged a pure rotation. A corridor walk (4 cm/frame, 2.4 m): every
    keyframe's reading follows the ground truth of its baseline over the
    windows a tracker can match — the seeds whose window lies whole in both
    frames, off the aperture-limited stripe texture — the keyframe count is
    that ground truth's ±1, and nothing is flagged a rotation. (Production
    values throughout except a sparser seed grid, for the test's time.)"""
    with open(SERVER / "config.yaml") as f:
        prod = load_intake_config(yaml.safe_load(f)).parallax
    cfg = replace(prod, grid_side=16)
    cam = S.default_camera(1280, 720)
    sc = S.room_scene(seed=4, width_m=4.0, depth_m=30.0, height_m=3.0, boxes=6)
    poses = _with_still(S.walk_poses("translate", 61, start=S.look_at_pose((0.0, 1.5, 1.0),
                                                                           (0.0, 1.5, 2.0)),
                                     step_m=0.04, direction="forward"), 4)
    renders = _render_pool(sc, cam, poses)
    frames_dir = tmp_path / "frames"
    S.write_frames(frames_dir, renders)
    quality = Q.analyze_frames(frames_dir, QCFG, log=QUIET, heartbeat_s=1e-6)
    r = P.run_parallax(frames_dir, quality, cfg, log=QUIET, heartbeat_s=1e-6)
    small, scale = P.downscale(np.zeros((cam.height, cam.width), np.uint8), cfg.process_scale)
    uv = P.to_native(P.seed_grid(small.shape, cfg.grid_side, margin=cfg.lk_win // 2), scale)
    names = [qq.name for qq in sc.all_quads()]
    stripes = np.array([nm.startswith(("box1", "box3", "box5")) for nm in names])
    margin = (cfg.lk_win // 2 + 1) / cfg.process_scale

    def gt(i, j):
        r_ = np.clip(np.round(uv[:, 1]).astype(int), 0, cam.height - 1)
        c_ = np.clip(np.round(uv[:, 0]).astype(int), 0, cam.width - 1)
        qid = renders[i].quad_id[r_, c_]
        ok = (qid >= 0) & ~stripes[np.clip(qid, 0, None)]
        X = S.unproject(cam, poses[i], uv[ok], renders[i].depth[r_, c_][ok].astype(np.float64))
        b, _, front = S.project(cam, poses[j], X)
        ins = (front & (b[:, 0] >= margin) & (b[:, 0] <= cam.width - 1 - margin)
               & (b[:, 1] >= margin) & (b[:, 1] <= cam.height - 1 - margin))
        a, b = uv[ok][ins], b[ins]
        H = _dlt_homography(a, b)
        Hi = np.linalg.inv(H)
        sym = (np.linalg.norm(_transfer(H, a) - b, axis=1)
               + np.linalg.norm(_transfer(Hi, b) - a, axis=1)) / 2
        return float(np.quantile(sym, cfg.parallax_quantile))

    anchor, n_gt = 4, 1
    for j in range(5, len(poses)):
        if gt(anchor, j) >= cfg.parallax_quantum_px:
            anchor, n_gt = j, n_gt + 1
    assert n_gt >= 4
    assert abs(r["n_keyframes"] - n_gt) <= 1, (r["n_keyframes"], n_gt)
    for k in r["keyframes"][1:]:
        assert k["parallax_from_anchor_px"] == pytest.approx(gt(k["anchor"], k["frame"]), rel=0.15)
    assert r["n_pure_rotation"] == 0 and not _warnings(r, "pure_rotation")


# ── lost tracking and exposure gaps ──────────────────────────────────────

def test_tracking_lost_is_a_hard_break(tmp_path):
    """Translation, then eight frames staring at a textureless wall from 30 cm,
    then translation again: the textureless stretch loses the tracks and is
    warned; the walk before and the walk after each earn their own keyframes;
    no keyframe is made where no baseline was measured — the measurement
    restarts from a PENDING anchor (not a keyframe) and the walk after earns
    its keyframes by the quantum from it."""
    scene = S.room_scene(seed=0, flat_wall=True)
    p1 = S.walk_poses("translate", 20, start=S.room_start_pose(x_m=1.0), step_m=0.05, direction="right")
    flat = S.walk_poses("still", 8, start=S.look_at_pose((0.0, 1.5, 7.7), (0.0, 1.5, 8.5)))
    p3 = S.walk_poses("translate", 20, start=p1[-1], step_m=0.05, direction="right")
    poses = _with_still(np.concatenate([p1, flat, p3]))
    sess, quality, r = _run(tmp_path, scene, S.default_camera(), poses)
    lost = [f for f in r["frames"] if f["lost"]]
    assert len(lost) >= PCFG.warn_min_run_frames
    assert all(f["reason"] in P.LOST_REASONS and f["parallax_px"] is None for f in lost)
    warns = _warnings(r, "tracking_lost")
    assert len(warns) >= 1 and "min_tracks" in warns[0]["detail"]
    kfr = {k["frame"]: k for k in r["keyframes"]}
    flat_frames = set(range(STILL_PREFIX + 20, STILL_PREFIX + 28))
    assert [f for f in kfr if f < min(flat_frames)]                 # the walk before
    after = [k for k in r["keyframes"] if k["frame"] > max(flat_frames)]
    assert after and all(k["reason"] == "quantum" for k in after)   # the walk after
    assert not [f for f in kfr if f in flat_frames]                 # nothing without a baseline
    breaks = [w for w in r["windows"] if w["closed_by"] == "tracking_lost"]
    assert breaks and all(w["keyframe"] is None and w["next_anchor"] is not None for w in breaks)
    assert any(k["anchor_is_keyframe"] is False for k in after)     # measured from a pending anchor
    assert set(P.KEYFRAME_REASONS) == {"first_usable_frame", "quantum", "track_loss"}
    assert r["n_lost"] == len(lost)


def test_exposure_gap_is_bridged_and_warned(tmp_path, scene):
    walk = S.walk_poses("translate", 40, start=S.room_start_pose(x_m=1.0), step_m=0.05, direction="right")
    poses = _with_still(walk)
    light = [1.0] * len(poses)
    dark = list(range(25, 31))
    for k in dark:
        light[k] = 0.05                                        # luma ≈ 6 < luma_lo → "dark"
    sess, quality, r = _run(tmp_path, scene, S.default_camera(), poses, light=light)
    assert quality["rejected"]["dark"] == len(dark)
    warns = _warnings(r, "exposure")
    assert len(warns) == 1 and (warns[0]["frame_start"], warns[0]["frame_end"]) == (dark[0], dark[-1])
    assert warns[0]["n_frames"] == len(dark) and "dark=6" in warns[0]["detail"]
    usable = {f["frame"] for f in quality["frames"] if f["usable"]}
    assert all(f["frame"] in usable for f in r["frames"])
    after = [f for f in r["frames"] if f["frame"] == dark[-1] + 1][0]
    assert not after["lost"]                                   # the chain bridges the gap
    assert all(k["frame"] in usable for k in r["keyframes"]) and r["n_keyframes"] >= 2
    assert all(w["frame"] in usable for w in r["witnesses"])


# ── warnings are pure over the records ───────────────────────────────────

def test_warning_bars_rejudge_the_same_records(rotation):
    """coverage_warnings re-judges stored frame records without re-tracking:
    a longer run requirement drops the warning, a higher displacement bar
    unflags the frames, a tighter floor factor flags fewer."""
    _, quality, r = rotation
    rows = sorted(quality["frames"], key=lambda f: f["frame"])
    recs = r["frames"]
    assert P.coverage_warnings(recs, rows, PCFG) == r["warnings"]
    long_run = P.coverage_warnings(recs, rows, replace(PCFG, warn_min_run_frames=40))
    assert not [w for w in long_run if w["kind"] == "pure_rotation"]
    high = replace(PCFG, warn_rotation_min_disp_px=1000.0)
    assert not any(P.frame_flags(f, high)["pure_rotation"] for f in recs)
    tight = replace(PCFG, rotation_floor_factor=1.0)
    loose = replace(PCFG, rotation_floor_factor=10.0)
    n_t = sum(P.frame_flags(f, tight)["pure_rotation"] for f in recs)
    n_l = sum(P.frame_flags(f, loose)["pure_rotation"] for f in recs)
    assert n_t <= r["n_pure_rotation"] <= n_l == len(_moving(r))


# ── (e) artifacts and consumers ──────────────────────────────────────────

def _strict_json(path):
    def _no_nan(c):
        raise ValueError(f"{path} carries {c}")
    return json.loads(Path(path).read_text(), parse_constant=_no_nan)


def test_write_selection_v2_contract_and_consumers(translation):
    sess, quality, r = translation
    p_kf, p_w, p_warn = P.write_selection(sess.session_dir, r, PCFG)
    assert p_kf == sess.frames_dir / "selected_frames.json"
    assert p_w == sess.frames_dir / "witness_frames.json"
    assert p_warn == sess.session_dir / "intake" / "coverage_warnings.json"
    assert not list(sess.frames_dir.glob("*.tmp")) and not list(p_warn.parent.glob("*.tmp"))
    sel = _strict_json(p_kf)
    assert sel["version"] == "2.0" and sel["method"] == "parallax_lk_12"
    assert sel["total_frames"] == len(sess.frame_numbers)
    assert sel["selected_count"] == len(sel["selected_files"]) == r["n_keyframes"]
    assert sel["selected_files"] == sorted(sel["selected_files"], key=lambda f: int(Path(f).stem))
    assert all((sess.frames_dir / f).exists() for f in sel["selected_files"])
    assert sel["selected_files"] == [k["file"] for k in r["keyframes"]]
    assert sel["parallax_quantum_px"] == 12.0 and sel["parallax_quantile"] == 0.9
    assert sel["keyframe_band_frac"] == 0.25
    assert sel["n_witness"] == r["n_witness"] and sel["keyframes"] == json.loads(json.dumps(r["keyframes"]))
    assert 0.0 <= sel["reduction"] < 1.0
    wit = _strict_json(p_w)
    assert wit["version"] == P.PARALLAX_VERSION and wit["method"] == "parallax_lk"
    assert wit["witness_min_parallax_px"] == 1.5
    assert wit["selected_files"] == sorted((w["file"] for w in r["witnesses"]), key=lambda f: int(Path(f).stem))
    assert set(sel["selected_files"]) <= set(wit["selected_files"])
    assert wit["selected_count"] == len(wit["selected_files"]) and wit["total_frames"] == sel["total_frames"]
    cov = _strict_json(p_warn)
    assert cov["version"] == P.PARALLAX_VERSION
    assert cov["warnings"] == r["warnings"] and cov["n_lost"] == 0
    assert len(cov["frames"]) == r["n_measured"] + r["n_lost"]
    assert cov["floor"]["median_px"] > 0 and set(cov["verdicts"]) == set(P.VERDICTS)
    assert set(cov["warning_kinds"]) == set(P.WARNING_KINDS)
    for doc in (sel, wit, cov):
        assert doc["provenance"] == "tool_measured"
        assert doc["geometry_epoch"] == 0 and doc["camera_epoch"] == 0
        assert doc["params"] == json.loads(json.dumps(r["params"]))
        assert doc["inputs"]["n_frames"] == len(sess.frame_numbers)
    assert P.load_selection(sess.frames_dir) == sel
    meta, kft = P.load_keyframe_tracks(sess.session_dir)       # what the exclusion audit reads
    assert meta["provenance"] == "tool_measured" and meta["native_w"] == sess.cam.width
    assert sorted(kft) == [k["frame"] for k in r["keyframes"][1:]]
    assert all(v["anchor"] == k["anchor"] and len(v["a"]) == len(v["b"]) >= PCFG.min_tracks
               for v, k in zip((kft[f] for f in sorted(kft)), r["keyframes"][1:]))
    try:
        from frames.selector import load_selected_frames
    except ImportError as e:                          # pragma: no cover - environment gap
        if "torch" in str(e):
            raise
        pytest.skip(f"frames.selector not importable here: {e}")
    assert load_selected_frames(str(sess.frames_dir)) == sel["selected_files"]
    from segmentation.scene_analyzer import _load_keyframes      # the VLM stage's strict reader
    paths = _load_keyframes(sess.frames_dir)
    assert [Path(p).name for p in paths] == sel["selected_files"]


def test_epochs_are_stamped_as_given(short_walk, tmp_path):
    sess, quality, _ = short_walk
    r = P.run_parallax(sess.frames_dir, quality, PCFG, log=QUIET, heartbeat_s=1e-6,
                       geometry_epoch=3, camera_epoch=1)
    assert (r["geometry_epoch"], r["camera_epoch"]) == (3, 1)
    (tmp_path / "frames").mkdir()
    for p in P.write_selection(tmp_path, r, PCFG):
        doc = json.loads(p.read_text())
        assert (doc["geometry_epoch"], doc["camera_epoch"]) == (3, 1), p.name


def test_load_selection_refuses_another_selector(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    with pytest.raises(P.ParallaxError, match="does not exist"):
        P.load_selection(frames_dir)
    (frames_dir / "selected_frames.json").write_text(json.dumps(
        {"version": "2.0", "method": "motion_60", "total_frames": 1, "selected_count": 1,
         "selected_files": ["000000.jpg"]}))
    with pytest.raises(P.ParallaxError, match="motion_60"):
        P.load_selection(frames_dir)


# ── (f) determinism ──────────────────────────────────────────────────────

def test_determinism(short_walk, tmp_path):
    sess, quality, r = short_walk
    again = P.run_parallax(sess.frames_dir, quality, PCFG, log=QUIET, heartbeat_s=1e-6)
    rest = {k: v for k, v in r.items() if k != "keyframe_tracks"}
    assert json.dumps({k: v for k, v in again.items() if k != "keyframe_tracks"},
                      sort_keys=True) == json.dumps(rest, sort_keys=True)
    assert sorted(again["keyframe_tracks"]) == sorted(r["keyframe_tracks"])
    for k, v in r["keyframe_tracks"].items():
        w = again["keyframe_tracks"][k]
        assert w["anchor"] == v["anchor"]
        assert np.array_equal(w["a"], v["a"]) and np.array_equal(w["b"], v["b"])
    d1, d2 = tmp_path / "one", tmp_path / "two"
    for d in (d1, d2):
        (d / "frames").mkdir(parents=True)
        P.write_selection(d, r, PCFG)
    for name in ("frames/selected_frames.json", "frames/witness_frames.json",
                 "intake/coverage_warnings.json"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes()


# ── structure, logs, cancel, CLI ─────────────────────────────────────────

def test_structural_failures_carry_the_reason(translation, tmp_path):
    sess, quality, r = translation
    with pytest.raises(P.ParallaxError, match="heartbeat_s"):
        P.run_parallax(sess.frames_dir, quality, PCFG, log=QUIET, heartbeat_s=0.0)
    with pytest.raises(TypeError, match="heartbeat_s"):             # no default, no global read
        P.run_parallax(sess.frames_dir, quality, PCFG, log=QUIET)
    other = dict(quality, frames=quality["frames"][:-1], n_frames=quality["n_frames"] - 1)
    with pytest.raises(P.ParallaxError, match="another frame inventory"):
        P.run_parallax(sess.frames_dir, other, PCFG, log=QUIET, heartbeat_s=1e-6)
    stale = dict(quality, version=0)
    with pytest.raises(P.ParallaxError, match="version"):
        P.run_parallax(sess.frames_dir, stale, PCFG, log=QUIET, heartbeat_s=1e-6)
    none_usable = dict(quality, frames=[dict(f, usable=False, reject_reason="dark")
                                        for f in quality["frames"]],
                       rejected={"dark": quality["n_frames"], "bright": 0, "clipped": 0})
    with pytest.raises(P.ParallaxError, match="no usable frame"):
        P.run_parallax(sess.frames_dir, none_usable, PCFG, log=QUIET, heartbeat_s=1e-6)
    empty = tmp_path / "frames"
    empty.mkdir()
    with pytest.raises(P.ParallaxError, match="no frame"):
        P.run_parallax(empty, quality, PCFG, log=QUIET, heartbeat_s=1e-6)
    with pytest.raises(P.ParallaxError, match="not a directory"):
        P.write_selection(tmp_path / "nowhere", r, PCFG)


def test_cancel_stops_the_loop_naming_where(translation):
    sess, quality, _ = translation
    calls = []

    def cancel_on_third():
        calls.append(1)
        return len(calls) >= 3

    with pytest.raises(Q.IntakeCancelled, match=r"intake I1 \(parallax, keyframes\) at frame 3/"):
        P.run_parallax(sess.frames_dir, quality, PCFG, log=QUIET, heartbeat_s=1e-6,
                       cancelled=cancel_on_third)
    assert len(calls) == 3


def test_single_usable_frame_is_the_keyframe(tmp_path, scene, cam):
    sess = S.write_session(tmp_path, scene, cam, S.walk_poses("still", 1, start=S.room_start_pose()))
    quality = Q.analyze_frames(sess.frames_dir, QCFG, log=QUIET, heartbeat_s=1e-6)
    r = P.run_parallax(sess.frames_dir, quality, PCFG, log=QUIET, heartbeat_s=1e-6)
    assert r["n_measured"] == 0 and r["n_keyframes"] == 1 and r["n_witness"] == 1
    assert r["floor"]["median_px"] is None and r["frames"] == []
    assert r["warnings"] == [] and r["windows"] == []
    P.write_selection(sess.session_dir, r, PCFG)
    assert P.load_selection(sess.frames_dir)["selected_files"] == ["000000.jpg"]


def test_heartbeat_lines_carry_the_rate(short_walk):
    sess, quality, _ = short_walk
    logs = []
    P.run_parallax(sess.frames_dir, quality, PCFG, log=logs.append, heartbeat_s=1e-6)
    assert any("frames/s" in m for m in logs)
    assert any("floor median" in m and "rigidity" in m for m in logs)
    assert any("keyframe(s)" in m and "witness(es)" in m for m in logs)
    assert all(m.startswith(P.LOG_TAG) for m in logs)


def test_cli_runs_with_the_real_config(tmp_path, scene, cam):
    """python -m intake.parallax --session <dir> with the production intake
    block (I0 written first); only the contract is asserted here — the
    production tracking scale is sized for real frames, not 320x240."""
    poses = _with_still(S.walk_poses("translate", 12, start=S.room_start_pose(x_m=1.0), step_m=0.05,
                                     direction="right"))
    sess = S.write_session(tmp_path, scene, cam, poses, noise_sigma=NOISE_SIGMA)
    Q.run_quality(sess.frames_dir, QCFG, log=QUIET, heartbeat_s=1e-6)
    (sess.output_dir / "geometry_epoch.json").write_text(json.dumps({"epoch": 2}))
    assert P.main(["--session", str(sess.session_dir)]) == 0
    sel = _strict_json(sess.frames_dir / "selected_frames.json")
    assert sel["version"] == "2.0" and sel["method"].startswith("parallax_lk_")
    assert sel["selected_files"][0] == "000000.jpg"
    assert sel["geometry_epoch"] == 2 and sel["camera_epoch"] == 0     # read from the session
    assert (sess.session_dir / "intake" / "coverage_warnings.json").exists()
    assert (sess.frames_dir / "witness_frames.json").exists()
