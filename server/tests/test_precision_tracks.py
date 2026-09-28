"""F4 — native-pixel tracks (claude_stac.txt §4-F4), on a rendered scene with a
ground-truth tracker standing in for the network: every observation lands in
native pixels with the error the tracker had (RMS vs ground truth reported), the
held-out split is deterministic, I2's exclusion masks keep their pixels out of
the queries, witness frames are tracked between their keyframes, loop pairs get
their windows and SALAD's index file is read through its own image list."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cv2 = pytest.importorskip("cv2")

from tests import synth_precision as S                          # noqa: E402
from precision import tracks as T                                # noqa: E402
from precision.camera import grid_full_frame_resize, grid_to_native, native_to_grid   # noqa: E402
from precision.config import load_precision_config              # noqa: E402

TCFG = load_precision_config().tracks
NOISE_GRID_PX = 0.3
KF = [0, 3, 6, 9, 11]


@pytest.fixture(scope="module")
def sess(tmp_path_factory):
    poses = S.walk_poses("translate", 12, start=S.room_start_pose(x_m=1.0), step_m=0.08,
                         direction="right")
    s = S.write_session(tmp_path_factory.mktemp("trk"), S.room_scene(seed=0), S.default_camera(),
                        poses, noise_sigma=1.0)
    fr = s.frames_dir
    (fr / "selected_frames.json").write_text(json.dumps(
        {"selected_files": [f"{k:06d}.jpg" for k in KF]}))
    (fr / "witness_frames.json").write_text(json.dumps(
        {"frames": [{"frame": k} for k in range(12)]}))
    return s


def _gmap(s):
    w, h = s.cam.width, s.cam.height
    tw, th = T.tracker_grid(w, h, TCFG.tracker_long_side, TCFG.tracker_stride)
    return grid_full_frame_resize(w, h, tw, th, "tracker")


def _truth_fn(s, noise=NOISE_GRID_PX, seed=0):
    """The ground truth through the tracker's interface: query → its surface point
    (rendered depth) → every frame's pixel; occluded / out of view = not visible."""
    g = _gmap(s)
    rng = np.random.default_rng(seed)

    def fn(images, queries, frames):
        f0 = frames[0]
        nat = grid_to_native(queries.astype(np.float64), g)
        d0 = s.renders[f0].depth
        u = np.clip(np.rint(nat[:, 0]).astype(int), 0, d0.shape[1] - 1)
        v = np.clip(np.rint(nat[:, 1]).astype(int), 0, d0.shape[0] - 1)
        z0 = d0[v, u].astype(np.float64)
        X = S.unproject(s.cam, s.poses[f0], nat, np.where(z0 > 0, z0, 1.0))
        uv = np.zeros((len(frames), len(queries), 2))
        vis = np.zeros((len(frames), len(queries)))
        for k, f in enumerate(frames):
            p, z, front = S.project(s.cam, s.poses[f], X)
            H, W = s.renders[f].depth.shape
            inb = front & (p[:, 0] >= 0) & (p[:, 0] <= W - 1) & (p[:, 1] >= 0) & (p[:, 1] <= H - 1)
            pu = np.clip(np.rint(p[:, 0]).astype(int), 0, W - 1)
            pv = np.clip(np.rint(p[:, 1]).astype(int), 0, H - 1)
            dz = np.abs(s.renders[f].depth[pv, pu] - z) / np.maximum(z, 1e-9)
            vis[k] = (inb & (z0 > 0) & (dz < 0.02)).astype(float)
            uv[k] = native_to_grid(p, g) + rng.normal(0, noise, (len(queries), 2))
        return uv, vis, np.full(vis.shape, noise)
    return fn


def _gt_native(s, tr):
    """Every observation's ground-truth native pixel (its track's query surface point)."""
    g = _gmap(s)
    q_of = dict(zip(tr["track_query_id"].tolist(), zip(tr["track_query_frame"].tolist(),
                                                       tr["track_query_uv"])))
    out = np.zeros((len(tr["obs_track"]), 2))
    for i, (t, f) in enumerate(zip(tr["obs_track"], tr["obs_frame"])):
        qf, quv = q_of[int(t)]
        nat = grid_to_native(np.asarray(quv, np.float64)[None], g)
        d0 = s.renders[qf].depth
        z0 = float(d0[int(round(nat[0, 1])), int(round(nat[0, 0]))])
        X = S.unproject(s.cam, s.poses[qf], nat, np.array([z0]))
        out[i] = S.project(s.cam, s.poses[int(f)], X)[0][0]
    return out


def test_observations_land_in_native_pixels_with_the_trackers_error(sess):
    rep = T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
    tr = T.load_tracks_v2(sess.session_dir)
    err = np.linalg.norm(tr["obs_uv_native"] - _gt_native(sess, tr), axis=1)
    moved = tr["obs_frame"] != tr["track_query_frame"][np.searchsorted(tr["track_query_id"], tr["obs_track"])]
    g = _gmap(sess)
    expected = NOISE_GRID_PX * 0.5 * (g.native_w / g.w + g.native_h / g.h) * np.sqrt(2)
    rms = float(np.sqrt(np.mean(err[moved] ** 2)))
    assert 0.5 * expected < rms < 1.5 * expected, (rms, expected)       # RMS px vs GT
    assert np.allclose(tr["obs_uv_sigma"], NOISE_GRID_PX * 0.5 * (g.native_w / g.w + g.native_h / g.h))
    assert rep["keyframes_covered"] == len(KF)
    meta = json.loads(str(tr["meta"]))
    assert meta["version"] == 2 and meta["provenance"] == "tool_measured"
    for k in ("obs_track", "obs_frame", "obs_uv", "obs_vis", "obs_score", "track_query_id",
              "track_query_frame", "track_query_uv", "res_h", "res_w"):
        assert k in tr, k                                                # the v1 keys stay


def test_identical_runs_write_identical_bits_and_say_which_tracker(sess):
    T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
    a = T.load_tracks_v2(sess.session_dir)
    rep = T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
    b = T.load_tracks_v2(sess.session_dir)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes(), k
    assert rep["tracker_weights"] == {"injected_track_fn": True}


def test_tracker_weights_are_refused_unless_their_content_matches(tmp_path, monkeypatch):
    """The weights are pinned by CONTENT: a file under the same name that hashes otherwise
    (the moving 'main' revision, a corrupt copy) is refused — nothing is downloaded when
    the file is present."""
    import hashlib
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "w.pt").write_bytes(b"weights at the pinned revision")
    url = "https://example.invalid/repo/resolve/0123456789abcdef/w.pt"
    sha = hashlib.sha256(b"weights at the pinned revision").hexdigest()
    assert T.verified_checkpoint(url, sha, log=lambda *a: None) == ck / "w.pt"
    (ck / "w.pt").write_bytes(b"weights after 'main' moved")
    with pytest.raises(T.TracksError, match="hashes to sha256"):
        T.verified_checkpoint(url, sha, log=lambda *a: None)


def test_deterministic_torch_is_scoped():
    torch = pytest.importorskip("torch")
    before = torch.are_deterministic_algorithms_enabled()
    with T.deterministic_torch(TCFG.seed):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        x = torch.rand(4)
    with T.deterministic_torch(TCFG.seed):
        assert torch.equal(torch.rand(4), x)                 # seeded
    assert torch.are_deterministic_algorithms_enabled() == before


def test_the_split_is_deterministic_and_per_track(sess):
    T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
    a = T.load_tracks_v2(sess.session_dir)["track_split"].copy()
    T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
    b = T.load_tracks_v2(sess.session_dir)["track_split"]
    assert np.array_equal(a, b)
    assert abs(a.mean() - TCFG.heldout_frac) < 0.05


def test_exclusion_masks_keep_their_pixels_out_of_the_queries(sess):
    mdir = sess.session_dir / "intake" / "exclusion_masks"
    mdir.mkdir(parents=True, exist_ok=True)
    H, W = sess.renders[0].depth.shape
    for f in KF:                               # the left half of every keyframe is excluded
        m = np.zeros((H, W), np.uint8)
        m[:, : W // 2] = 255
        cv2.imwrite(str(mdir / f"{f:06d}.png"), m)
    try:
        rep = T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
        tr = T.load_tracks_v2(sess.session_dir)
        qn = grid_to_native(tr["track_query_uv"].astype(np.float64), _gmap(sess))
        kf_q = np.isin(tr["track_query_frame"], KF)
        assert (qn[kf_q, 0] >= W // 2 - 0.5).all()
        assert rep["n_queries_dropped_excluded_or_edge"] > 0
    finally:
        for p in mdir.glob("*.png"):
            p.unlink()


def test_witnesses_are_tracked_between_their_keyframes(sess):
    T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
    tr = T.load_tracks_v2(sess.session_dir)
    wit = sorted(set(range(12)) - set(KF))
    seen = set(tr["obs_frame"][tr["frame_kind"] == T.KIND_WITNESS].tolist())
    assert seen == set(wit)
    assert set(tr["obs_frame"][tr["frame_kind"] == T.KIND_KEYFRAME].tolist()) == set(KF)


def test_salad_indices_are_read_through_their_own_image_list(tmp_path, sess):
    lc = tmp_path / "loop_closures.txt"
    lc.write_text("# Loop Detection Results (index1, index2, similarity)\n\n# Loop pairs:\n"
                  "0, 4, 0.7100\n\n# Image path list:\n"
                  + "".join(f"# {i}: /x/frames/{f:06d}.jpg\n" for i, f in enumerate(KF)))
    assert T.salad_pairs(lc) == [(0, 11)]
    wins = T.loop_windows(KF, [(0, 4)], 1)
    assert wins[0].kind == "loop" and wins[0].frames == [0, 3, 9, 11]
    assert wins[0].query_frames == [0, 11]


def test_loop_windows_mark_their_observations(sess):
    d = sess.session_dir / "output" / "maplong_run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "loop_closures.txt").write_text(
        "0, 4, 0.7\n" + "".join(f"# {i}: {f:06d}.jpg\n" for i, f in enumerate(KF)))
    try:
        rep = T.run_tracks(sess.session_dir, TCFG, track_fn=_truth_fn(sess), log=lambda *a: None)
        tr = T.load_tracks_v2(sess.session_dir)
        assert rep["n_windows_by_kind"]["loop"] == 1 and tr["is_loop_pair"].any()
    finally:
        (d / "loop_closures.txt").unlink()


def test_witness_windows_keep_both_keyframes_when_cut():
    wins = T.witness_windows([0, 30], list(range(1, 30)), window_frames=10)
    assert all(w.frames[0] == 0 and w.frames[-1] == 30 for w in wins)
    assert sorted(f for w in wins for f in w.frames[1:-1]) == list(range(1, 30))


def test_depth_edges_flag_a_step():
    d = np.full((10, 10), 2.0)
    d[:, 5:] = 4.0
    e = T.depth_edges(d, 0.05)
    assert e[:, 4].all() and not e[:, :4].any() and not e[:, 6:].any()
