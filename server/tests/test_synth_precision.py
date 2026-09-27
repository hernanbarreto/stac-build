"""The shared synthetic scene (tests/synth_precision.py) tells the truth about
itself: ground-truth depth round-trips through the Brown camera to the quad
planes and back to the pixel, renders are deterministic, dynamic blobs are
flagged with their own depth, light scales the image, the pose generators do
what their names say, and a written session passes the native-frame guard of
precision.camera."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import synth_precision as S   # noqa: E402
from precision import camera as C        # noqa: E402


@pytest.fixture(scope="module")
def scene():
    return S.room_scene(seed=0)


@pytest.fixture(scope="module")
def start():
    return S.room_start_pose()


# ── camera ───────────────────────────────────────────────────────────────

def test_default_camera_and_precision_camera():
    cam = S.default_camera(320, 240, fov_deg=70.0, distortion=True)
    assert (cam.width, cam.height) == (320, 240)
    assert np.isclose(cam.fx, 0.5 * 320 / np.tan(np.radians(35.0))) and cam.fx == cam.fy
    assert (cam.cx, cam.cy) == (159.5, 119.5)                # pixel-centre convention
    assert cam.K().shape == (3, 3) and cam.dist().shape == (4,)
    assert tuple(cam.dist()) == (-0.12, 0.03, 0.001, -0.0005)
    pc = cam.to_precision_camera()
    assert isinstance(pc, C.CameraModel) and pc.source == "synthetic" and pc.camera_epoch == 0
    assert pc.params == cam.params() and pc.width == 320 and pc.height == 240
    assert (pc.omega_grid.native_w, pc.omega_grid.native_h) == (320, 240)
    assert pc.mask_grid is not None
    assert np.allclose(pc.K(), cam.K()) and np.array_equal(pc.dist(), cam.dist())
    plain = S.default_camera()
    assert not np.any(plain.dist())
    g = C.omega_grid_for(320, 240, "max_size", 1024)
    assert S.default_camera().to_precision_camera(omega_grid=g).omega_grid == g


# ── ground truth round trips ─────────────────────────────────────────────

@pytest.mark.parametrize("distortion", [False, True])
def test_depth_gt_round_trip(scene, start, distortion):
    """A hit pixel, undistorted through precision.camera and pushed along its
    ray by the rendered depth, lands on its quad's plane (< 1e-6 m) and
    projects back onto the very pixel (< 1e-6 px); project() also returns
    the rendered depth as z."""
    cam = S.default_camera(distortion=distortion)
    c2w = S.walk_poses("arc", 4, start=start, step_m=0.4)[3]     # an oblique pose
    r = S.render(scene, cam, c2w)
    quads = scene.all_quads()
    rng = np.random.default_rng(7)
    rows = rng.integers(0, cam.height, 2000)
    cols = rng.integers(0, cam.width, 2000)
    ok = r.quad_id[rows, cols] >= 0
    rows, cols = rows[ok], cols[ok]
    assert ok.sum() > 1000
    uv = np.stack([cols, rows], axis=1).astype(np.float64)
    z = r.depth[rows, cols].astype(np.float64)
    assert np.all(z > 0)
    pc = cam.to_precision_camera()
    und = C.undistort_points(uv, pc, **S.undistort_solver())
    d = np.stack([(und[:, 0] - cam.cx) / cam.fx, (und[:, 1] - cam.cy) / cam.fy,
                  np.ones(len(uv))], axis=1)
    X = (d * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
    qid = r.quad_id[rows, cols]
    resid = np.array([abs((X[k] - quads[q].origin) @ quads[q].normal()) for k, q in enumerate(qid)])
    assert resid.max() < 1e-6, resid.max()
    # and inside the rectangle of that quad
    for k, q in enumerate(qid[:200]):
        quad = quads[q]
        rel = X[k] - quad.origin
        a = rel @ quad.edge_u / (quad.edge_u @ quad.edge_u)
        b = rel @ quad.edge_v / (quad.edge_v @ quad.edge_v)
        assert -1e-6 <= a <= 1 + 1e-6 and -1e-6 <= b <= 1 + 1e-6
    uv2, z2, in_front = S.project(cam, c2w, X)
    assert in_front.all()
    assert np.abs(uv2 - uv).max() < 1e-6, np.abs(uv2 - uv).max()
    assert np.abs(z2 - z).max() < 1e-9
    assert np.abs(S.unproject(cam, c2w, uv, z) - X).max() < 1e-9
    if distortion:
        # the lens really moves pixels: the undistorted grid is not the grid
        assert np.abs(und - uv).max() > 1.0
    # normals are unit and face the camera
    n = r.normal[rows, cols].astype(np.float64)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    view = X - c2w[:3, 3]
    assert np.all(np.einsum("ij,ij->i", n, view) < 0)


def test_project_behind_camera_flagged(start):
    cam = S.default_camera()
    behind = start[:3, 3] - 2.0 * start[:3, 2]
    ahead = start[:3, 3] + 2.0 * start[:3, 2]
    uv, z, in_front = S.project(cam, start, np.stack([behind, ahead]))
    assert list(in_front) == [False, True]
    assert np.allclose(uv[1], [cam.cx, cam.cy]) and np.isclose(z[1], 2.0)


def test_render_shapes_and_room_is_closed(scene, start):
    cam = S.default_camera()
    r = S.render(scene, cam, start)
    assert r.rgb.shape == (240, 320, 3) and r.rgb.dtype == np.uint8
    assert r.depth.shape == (240, 320) and r.depth.dtype == np.float32
    assert r.normal.shape == (240, 320, 3) and r.normal.dtype == np.float32
    assert r.quad_id.shape == (240, 320) and r.quad_id.dtype == np.int32
    assert r.dynamic_mask.shape == (240, 320) and r.dynamic_mask.dtype == np.bool_
    assert np.all(r.quad_id >= 0) and np.all(r.depth > 0)      # a closed room: every ray hits
    assert not r.dynamic_mask.any()
    names = [q.name for q in scene.all_quads()]
    seen = {names[i] for i in np.unique(r.quad_id)}
    assert {"floor", "wall_back", "ceiling", "wall_left", "wall_right"} <= seen
    assert any(n.startswith("box") for n in seen)
    assert np.isclose(r.depth[120, 160], 7.0, atol=1e-3)         # back wall straight ahead
    # the rendered texture is not flat: it carries gradients a tracker can use
    assert r.rgb[..., 0].astype(np.float64).std() > 20.0


def test_flat_wall_is_textureless():
    scene = S.room_scene(seed=0, flat_wall=True)
    cam = S.default_camera()
    r = S.render(scene, cam, S.room_start_pose())
    back = scene.quad_index("wall_back")
    m = r.quad_id == back
    assert m.sum() > 1000
    assert r.rgb[m].std() == 0.0 and r.rgb[m][0, 0] == 128


# ── determinism, blobs, light ────────────────────────────────────────────

def test_render_determinism(scene, start):
    cam = S.default_camera(distortion=True)
    a = S.render(scene, cam, start, noise_sigma=4.0, seed=11)
    b = S.render(scene, cam, start, noise_sigma=4.0, seed=11)
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.depth, b.depth)
    assert np.array_equal(a.quad_id, b.quad_id) and np.array_equal(a.normal, b.normal)
    c = S.render(scene, cam, start, noise_sigma=4.0, seed=12)
    assert not np.array_equal(a.rgb, c.rgb)
    assert np.array_equal(a.depth, c.depth)                     # noise touches colour only
    clean = S.render(scene, cam, start)
    diff = a.rgb.astype(np.float64) - clean.rgb.astype(np.float64)
    assert 3.0 < diff.std() < 5.0                               # ≈ the requested sigma


def test_dynamic_blob_pixels_flagged_with_own_depth(scene, start):
    cam = S.default_camera()
    blob = S.MovingBlob(center_px0=(160.0, 120.0), velocity_px=(2.0, -1.0), radius_px=10.0,
                        depth_m=1.0, color=(255, 0, 0))
    r0 = S.render(scene, cam, start, dynamic=[blob], frame_index=0)
    m = r0.dynamic_mask
    assert abs(m.sum() - np.pi * 100.0) < 0.15 * np.pi * 100.0
    yy, xx = np.nonzero(m)
    assert abs(xx.mean() - 160.0) < 0.5 and abs(yy.mean() - 120.0) < 0.5
    assert np.all(r0.depth[m] == np.float32(1.0)) and np.all(r0.quad_id[m] == -1)
    assert np.all(r0.rgb[m] == np.array([255, 0, 0], np.uint8))
    assert np.all(r0.depth[~m] > 1.0)                           # the wall is behind the blob
    assert np.all(r0.quad_id[~m] >= 0)
    # outside the disc nothing changed
    clean = S.render(scene, cam, start)
    assert np.array_equal(r0.rgb[~m], clean.rgb[~m])
    # it moves with the video frame index
    r5 = S.render(scene, cam, start, dynamic=[blob], frame_index=5)
    yy5, xx5 = np.nonzero(r5.dynamic_mask)
    assert abs(xx5.mean() - 170.0) < 0.5 and abs(yy5.mean() - 115.0) < 0.5
    # a blob behind the surface it would cover is hidden
    far = S.MovingBlob((160.0, 120.0), (0.0, 0.0), 10.0, 20.0, (0, 255, 0))
    assert not S.render(scene, cam, start, dynamic=[far]).dynamic_mask.any()
    with pytest.raises(ValueError):
        S.MovingBlob((0, 0), (0, 0), 0.0, 1.0, (0, 0, 0))


def test_light_scales_rgb(scene, start):
    cam = S.default_camera()
    full = S.render(scene, cam, start, light=1.0).rgb.astype(np.float64)
    half = S.render(scene, cam, start, light=0.5).rgb.astype(np.float64)
    assert np.abs(half - 0.5 * full).max() <= 1.0                # rounding only
    assert half.mean() < 0.55 * full.mean()
    bright = S.render(scene, cam, start, light=1.8).rgb
    assert bright.max() == 255 and bright.mean() > full.mean()


# ── poses ────────────────────────────────────────────────────────────────

def test_look_at_pose_is_opencv_camera():
    M = S.look_at_pose((0.0, 1.5, 1.0), (0.0, 1.5, 5.0))
    R = M[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], [0, 0, 1])                       # +z forward
    assert np.allclose(R[:, 1], [0, -1, 0])                      # +y down (world −y)
    assert np.allclose(M[:3, 3], [0.0, 1.5, 1.0])
    uv, z, ok = S.project(S.default_camera(), M, np.array([[0.0, 1.5, 5.0]]))
    assert ok[0] and np.allclose(uv[0], [159.5, 119.5]) and np.isclose(z[0], 4.0)


def test_walk_poses_kinds_and_chainage(start):
    n = 60
    rot = S.walk_poses("rotate", n, start=start, yaw_deg_per_frame=1.5)
    assert rot.shape == (n, 4, 4)
    assert np.abs(rot[:, :3, 3] - start[:3, 3]).max() == 0.0     # centre fixed
    for i in (1, 30, 59):
        Rrel = rot[i, :3, :3] @ start[:3, :3].T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        assert np.isclose(ang, 1.5 * i, atol=1e-9)
        assert np.allclose(Rrel @ np.array([0, 1, 0]), [0, 1, 0])   # about world up
    assert np.all(S.chainage(rot) == 0.0)

    tr = S.walk_poses("translate", n, start=start, step_m=0.05)
    steps = np.linalg.norm(np.diff(tr[:, :3, 3], axis=0), axis=1)
    assert np.allclose(steps, 0.05, atol=1e-12)
    assert np.allclose(tr[:, :3, :3], start[:3, :3])              # rotation fixed
    assert np.allclose(tr[-1, :3, 3] - start[:3, 3], 59 * 0.05 * start[:3, 2])   # along forward
    ch = S.chainage(tr)
    assert ch[0] == 0.0 and np.allclose(ch, 0.05 * np.arange(n)) and np.isclose(ch[-1], 2.95)
    side = S.walk_poses("translate", 3, start=start, step_m=0.1, direction="right")
    assert np.allclose(side[1, :3, 3] - start[:3, 3], 0.1 * start[:3, 0])
    world = S.walk_poses("translate", 3, start=start, step_m=0.1, direction=(0.0, 2.0, 0.0))
    assert np.allclose(world[2, :3, 3] - start[:3, 3], [0.0, 0.2, 0.0])

    still = S.walk_poses("still", n, start=start)
    assert np.all(still == start[None]) and np.all(S.chainage(still) == 0.0)

    arc = S.walk_poses("arc", n, start=start, step_m=0.05, radius_m=2.0)
    steps = np.linalg.norm(np.diff(arc[:, :3, 3], axis=0), axis=1)
    assert np.allclose(steps, 2 * 2.0 * np.sin(0.05 / 2.0 / 2.0))        # the chord of the arc step
    assert np.allclose(arc[:, 1, 3], start[1, 3])                        # horizontal
    centre = start[:3, 3] + 2.0 * (S._rot_y(90.0) @ start[:3, 2])
    assert np.allclose(np.linalg.norm(arc[:, :3, 3] - centre, axis=1), 2.0)
    for i in (1, 20, 59):                                                # heading follows the tangent
        Rrel = arc[i, :3, :3] @ start[:3, :3].T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        assert np.isclose(ang, np.degrees(i * 0.05 / 2.0), atol=1e-9)
    with pytest.raises(ValueError):
        S.walk_poses("spiral", 3, start=start)
    with pytest.raises(ValueError):
        S.walk_poses("translate", 3, start=start, direction="sideways")


def test_perturb_poses_noise_and_drift(start):
    poses = S.walk_poses("translate", 40, start=start, step_m=0.1)
    same = S.perturb_poses(poses)
    assert np.array_equal(same, poses) and same is not poses
    noisy = S.perturb_poses(poses, rot_deg_sigma=0.5, trans_sigma_m=0.02, seed=3)
    assert np.array_equal(noisy, S.perturb_poses(poses, rot_deg_sigma=0.5, trans_sigma_m=0.02, seed=3))
    R = noisy[:, :3, :3]
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3)[None], atol=1e-12)
    dt = np.linalg.norm(noisy[:, :3, 3] - poses[:, :3, 3], axis=1)
    assert 0.0 < dt.mean() < 0.1
    angs = [np.degrees(np.arccos(np.clip((np.trace(R[i] @ poses[i, :3, :3].T) - 1) / 2, -1, 1)))
            for i in range(40)]
    assert 0.0 < np.mean(angs) < 3.0
    drift = S.perturb_poses(poses, drift_per_m=0.02)
    ch = S.chainage(poses)
    err = drift[:, :3, 3] - poses[:, :3, 3]
    assert np.allclose(np.linalg.norm(err, axis=1), 0.02 * ch)             # ∝ chainage
    assert np.allclose(err[-1], 0.02 * ch[-1] * start[:3, 2])              # along the walk
    assert np.array_equal(drift[:, :3, :3], poses[:, :3, :3])
    up = S.perturb_poses(poses, drift_per_m=0.05, drift_dir=(0, 1, 0))
    assert np.allclose(up[:, 1, 3] - poses[:, 1, 3], 0.05 * ch)
    still = S.walk_poses("still", 5, start=start)
    assert np.array_equal(S.perturb_poses(still, drift_per_m=1.0), still)  # no walk, no drift


def test_scale_drift_along_walk(scene, start):
    cam = S.default_camera(160, 120)
    poses = S.walk_poses("translate", 4, start=start, step_m=0.5)
    depths = [S.render(scene, cam, p).depth for p in poses]
    ch = S.chainage(poses)
    out = S.scale_drift_along_walk(depths, ch, eps_per_m=0.02)
    assert len(out) == 4 and all(o.dtype == np.float32 for o in out)
    assert np.array_equal(out[0], depths[0])                               # start exact
    for k in range(1, 4):
        assert np.allclose(out[k], depths[k] * np.float32(1.0 + 0.02 * ch[k]), rtol=1e-6)
    z = depths[1].copy(); z[0, 0] = 0.0
    assert S.scale_drift_along_walk([z], [1.0], 0.5)[0][0, 0] == 0.0       # no hit stays no hit
    with pytest.raises(ValueError):
        S.scale_drift_along_walk(depths, ch[:2], 0.01)


# ── scene building blocks ────────────────────────────────────────────────

def test_make_texture_kinds():
    for kind in ("noise", "checker", "stripes", "flat"):
        t = S.make_texture(kind, 128, seed=1)
        assert t.shape == (128, 128) and t.dtype == np.uint8
    assert S.make_texture("flat").std() == 0.0
    assert set(np.unique(S.make_texture("checker", 64))) == {1, 255}
    st = S.make_texture("stripes", 64)
    assert np.all(st == st[0:1, :])                                        # vertical stripes
    assert st.std() > 0
    lo = S.make_texture("noise", 128, seed=2, contrast=0.2)
    hi = S.make_texture("noise", 128, seed=2, contrast=1.0)
    assert lo.std() < hi.std() and np.array_equal(hi, S.make_texture("noise", 128, seed=2))
    assert not np.array_equal(hi, S.make_texture("noise", 128, seed=3))
    with pytest.raises(ValueError):
        S.make_texture("marble")


def test_box_and_scene():
    box = S.Box(center=(1.0, 0.25, 4.0), size=(0.6, 0.5, 0.8), yaw_deg=30.0,
                texture=S.make_texture("checker", 64), name="crate", label="box")
    faces = box.quads()
    assert len(faces) == 6 and {f.name for f in faces} == {
        f"crate_{s}" for s in ("bottom", "top", "front", "back", "left", "right")}
    corners = np.concatenate([f.corners() for f in faces])
    assert np.isclose(corners[:, 1].min(), 0.0) and np.isclose(corners[:, 1].max(), 0.5)
    assert np.allclose(corners.mean(axis=0), box.center)
    # every face is a rectangle with the right area, edges orthogonal
    areas = sorted(np.linalg.norm(np.cross(f.edge_u, f.edge_v)) for f in faces)
    assert np.allclose(areas, sorted([0.48, 0.48, 0.3, 0.3, 0.4, 0.4]))
    assert all(abs(f.edge_u @ f.edge_v) < 1e-12 for f in faces)
    assert box.quads() == faces or all(a.name == b.name for a, b in zip(box.quads(), faces))
    box.yaw_deg = 60.0                                                     # the cache follows the fields
    assert not np.allclose(box.quads()[2].edge_u, faces[2].edge_u)
    scene = S.room_scene(seed=3, boxes=3)
    quads = scene.all_quads()
    assert len(quads) == 6 + 18 and len(scene.boxes) == 3
    assert [q.label for q in quads[:6]] == ["floor", "wall", "wall", "wall", "wall", "ceiling"]
    assert scene.quad_index("box2_top") == 6 + 12 + 1
    for b in scene.boxes:                                                  # resting on the floor, inside
        assert np.isclose(b.center[1], b.size[1] / 2) and abs(b.center[0]) < 3.0 and 0 < b.center[2] < 8.0
    with pytest.raises(KeyError):
        scene.quad_index("nothing")
    with pytest.raises(ValueError):
        S.TexturedQuad((0, 0, 0), (1, 0, 0), (2, 0, 0), S.make_texture("flat", 8), "bad", "x")
    with pytest.raises(ValueError):
        S.TexturedQuad((0, 0, 0), (1, 0, 0), (0, 1, 0), np.zeros((8, 8), np.float32), "bad", "x")


# ── session layout ───────────────────────────────────────────────────────

def test_write_session_native_frames_and_guard(tmp_path, scene, start):
    cam = S.default_camera(distortion=True)
    poses = S.walk_poses("translate", 60, start=start, step_m=0.05)
    blob = S.MovingBlob((40.0, 60.0), (3.0, 0.0), 8.0, 1.5, (250, 250, 250))
    light = list(np.linspace(1.0, 0.6, 60))
    t0 = time.time()
    sess = S.write_session(tmp_path / "s", scene, cam, poses, frame_numbers=list(range(0, 180, 3)),
                           light=light, dynamic=[blob], noise_sigma=1.5, seed=5)
    elapsed = time.time() - t0
    # a measurement, not a verdict: compute time is not a design criterion and a
    # wall-clock bound flakes on a loaded box (run with -s to see it)
    print(f"write_session: 60 frames 320x240 in {elapsed:.2f} s")
    assert sess.session_dir == tmp_path / "s" and sess.frames_dir == tmp_path / "s" / "frames"
    assert sess.output_dir.is_dir() and not any(sess.output_dir.iterdir())
    assert sess.frame_numbers == list(range(0, 180, 3)) and len(sess.renders) == 60
    assert sess.files == [f"{f:06d}.jpg" for f in range(0, 180, 3)]
    assert sorted(p.name for p in sess.frames_dir.glob("*.jpg")) == sess.files
    assert sess.frame_path(177).exists() and sess.light == pytest.approx(light)
    import cv2
    img = cv2.imread(str(sess.frame_path(3)))
    assert img.shape == (240, 320, 3)
    # JPEG q95 of the rendered image, channels back in RGB order
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    assert np.abs(rgb - sess.renders[1].rgb.astype(np.float64)).mean() < 4.0
    rep = C.verify_native_frames(sess.session_dir)
    assert (rep["native_w"], rep["native_h"], rep["n_frames"]) == (320, 240, 60)
    # the blob moved per VIDEO frame: frame 3 shows it 9 px right of frame 0
    y0, x0 = np.nonzero(sess.renders[0].dynamic_mask)
    y1, x1 = np.nonzero(sess.renders[1].dynamic_mask)
    assert abs(x1.mean() - x0.mean() - 9.0) < 0.5 and abs(y1.mean() - y0.mean()) < 0.5
    # light really dimmed the sequence
    assert sess.renders[-1].rgb.mean() < 0.75 * sess.renders[0].rgb.mean()
    gt = np.load(sess.session_dir / S.GT_NAME)
    assert np.array_equal(gt["poses"], poses) and np.array_equal(gt["K"], cam.K())
    assert list(gt["frame_numbers"]) == sess.frame_numbers and np.allclose(gt["light"], light)
    # a second identical write is byte-identical (determinism through the JPEG)
    sess2 = S.write_session(tmp_path / "t", scene, cam, poses, frame_numbers=list(range(0, 180, 3)),
                            light=light, dynamic=[blob], noise_sigma=1.5, seed=5)
    assert (sess.frame_path(30).read_bytes() == sess2.frame_path(30).read_bytes())
    with pytest.raises(ValueError):
        S.write_session(tmp_path / "u", scene, cam, poses, frame_numbers=[0] * 60)
    with pytest.raises(ValueError):
        S.write_session(tmp_path / "v", scene, cam, poses, light=[1.0, 2.0])


def test_write_frames_default_numbering(tmp_path, scene, start):
    cam = S.default_camera(64, 48)
    renders = [S.render(scene, cam, p) for p in S.walk_poses("still", 3, start=start)]
    names = S.write_frames(tmp_path / "frames", renders)
    assert names == ["000000.jpg", "000001.jpg", "000002.jpg"]
    assert C.frame_sizes(tmp_path / "frames") == {(64, 48): 3}
    with pytest.raises(ValueError):
        S.write_frames(tmp_path / "f2", renders, frame_numbers=[0, 1])
