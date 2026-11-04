"""The turn frames as F6 WITNESSES (USER 2026-11-04, "hacé 1 y 2"): the fork keeps each context
view's Omega depth, confidence and pose relative to the keyframe before it
(vendor/VGGT-Long/loop_utils/context_views.context_records); F6 carries the anchor's scale, bend,
floor and refined pose onto it (precision/depth_on_f5.load_turn_witnesses) and lets it vote on the
keyframes near its anchor (f6_torch.two_sided_vote ``extra``) — the points stay the keyframes'.
CPU only, synthetic."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "vendor" / "VGGT-Long"))

from loop_utils.context_views import CONTEXT_DIR, context_records  # noqa: E402


def _pose(x=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    M = np.eye(4)
    M[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M[:3, 3] = [x, 0.0, 0.0]
    return M


def test_context_records_carry_depth_conf_and_the_pose_relative_to_the_keyframe_before():
    S, H, W = 4, 6, 8
    depth = np.stack([np.full((H, W), 2.0 + k) for k in range(S)])
    conf = np.ones((S, H, W)); conf[0, 0, 0] = 0.0
    ext = np.stack([_pose(0.1 * k, 0.05 * k) for k in range(S)])
    K = np.stack([np.eye(3)] * S)
    pred = {"depth": depth, "world_points_conf": conf, "extrinsic": ext, "intrinsic": K}
    views = ["/f/000010.jpg", "/f/000013.jpg", "/f/000020.jpg", "/f/000024.jpg"]
    recs = context_records(pred, views, keep=[0, 2], keyframe_global=[7, 8])
    assert [(r["frame"], r["anchor_frame"], r["anchor_index"]) for r in recs] == [(13, 10, 7), (24, 20, 8)]
    assert np.allclose(recs[0]["rel_pose"], np.linalg.inv(ext[0]) @ ext[1])
    assert recs[0]["anchor_depth_median"] == 2.0 and recs[1]["anchor_depth_median"] == 4.0
    assert recs[1]["depth"].dtype == np.float32 and float(recs[1]["depth"][0, 0]) == 5.0


def _cam(H=32, W=32):
    return np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1.0]])


def test_an_extra_witness_votes_like_a_neighbour_and_none_changes_nothing():
    from precision import f6_torch as FT
    H = W = 32
    K = _cam(H, W)
    plane = np.full((H, W), 3.0, np.float32)
    frames = [100, 200]
    c2w = {100: _pose(0.0), 200: _pose(0.6), 150: _pose(0.3)}
    w2c = {f: np.linalg.inv(m) for f, m in c2w.items()}

    def depth_of(M):                                   # a wall at z = 3 in the world
        return np.full((H, W), 3.0, np.float32)
    dep = {f: depth_of(c2w[f]) for f in c2w}
    judge = {f: np.ones((H, W), bool) for f in c2w}
    cand = np.ones((H, W), bool)
    base = FT.two_sided_vote(0, frames, dep, judge, cand, K, c2w, w2c, [1], 0.02)
    same = FT.two_sided_vote(0, frames, dep, judge, cand, K, c2w, w2c, [1], 0.02, extra=())
    for k in ("agree", "contra", "zmed"):
        assert FT._np(base[k], np.float64).tolist() == FT._np(same[k], np.float64).tolist()
    more = FT.two_sided_vote(0, frames, dep, judge, cand, K, c2w, w2c, [1], 0.02, extra=[150])
    a0, a1 = FT._np(base["agree"], np.int64), FT._np(more["agree"], np.int64)
    assert (a1 >= a0).all() and (a1 - a0).max() == 1 and (a1 > a0).sum() > 100
    assert more["splats"].shape[0] == base["splats"].shape[0] + 1


def _write_view(d, chunk, frame, anchor, stamp, depth, rel, am, conf=None, prefix="c"):
    np.savez(d / f"{prefix}{chunk:03d}_{frame:06d}.npz", chunk=chunk, fork_stamp=stamp, frame=frame,
             frame_file=f"{frame:06d}.jpg", anchor_frame=anchor, anchor_index=0, depth=depth,
             conf=conf if conf is not None else np.full(depth.shape, 9.0, np.float32), K=np.eye(3),
             rel_pose=rel, anchor_depth_median=am)


def test_f6_carries_the_anchors_scale_bend_floor_and_pose_onto_each_witness(tmp_path):
    from precision.depth_on_f5 import design, load_turn_witnesses
    out = tmp_path / "output"
    d = out / "maplong_run" / CONTEXT_DIR
    d.mkdir(parents=True)
    (out / "maplong_run" / "fork_stamp.json").write_text(json.dumps({"sha256": "S"}))
    H, W = 4, 5
    frames = [10, 20, 30, 40]
    rel = _pose(0.05, 0.1)
    _write_view(d, 1, 15, 10, "S", np.full((H, W), 2.0, np.float32), rel, 2.0)
    _write_view(d, 0, 15, 10, "S", np.full((H, W), 9.0, np.float32), rel, 2.0)     # not the anchor's chunk
    _write_view(d, 1, 25, 20, "OLD", np.full((H, W), 2.0, np.float32), rel, 2.0)  # another run
    _write_view(d, 1, 35, 31, "S", np.full((H, W), 2.0, np.float32), rel, 2.0)    # anchor not a keyframe
    chunk = {10: 1, 20: 1, 30: 2, 40: 2}
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Dm = design(uu.ravel(), vv.ravel(), W, H)
    coefs = {i: np.array([1.1, 0.0, 0.0]) for i in range(4)}
    c2w = {f: _pose(float(f)) for f in frames}
    dep, valid, passed, cw, extra, rep = load_turn_witnesses(
        out, frames, chunk, {1: 5.0, 2: 5.0}, {10: 3.0, 20: 3.0, 30: 3.0, 40: 3.0}, coefs, c2w,
        None, False, (H, W), Dm, H, W, [-1, 1], log=lambda m: None)
    assert list(dep) == [15] and rep["used"] == 1
    assert rep["ignored"] == {"another fork run's": 1, "anchor not a keyframe of this run": 1}
    # depth: 2.0 raw × (zmed 3.0 / anchor median 2.0) × bend 1.1
    assert np.allclose(dep[15], 2.0 * 1.5 * 1.1) and passed[15].all()
    T = rel.copy(); T[:3, 3] *= 1.5 * 1.1
    assert np.allclose(cw[15], c2w[10] @ T)
    assert extra == {10: [15], 20: [15]}                  # anchor 10 is keyframe 0: window [-1, 1]
    none = load_turn_witnesses(tmp_path / "nothing", frames, chunk, {1: 5.0}, {}, coefs, c2w, None,
                               False, (H, W), Dm, H, W, [-1, 1], log=lambda m: None)
    assert none[0] == {} and "no context views" in none[5]["why_none"]


def test_the_witnesses_enter_the_vote_after_the_mono_detail_and_never_its_keyframes():
    src = (ROOT / "server" / "precision" / "depth_on_f5.py").read_text()
    body = src[src.index("def compute("):src.index("def publish_cloud(")]
    i_md = body.index("apply_mono_detail(")
    i_w = body.index("load_turn_witnesses(")
    i_v = body.index("edge_keeping_vote(frames, dep, valid, passed, K, c2w_vote,")
    assert i_md < i_w < i_v
    assert "extra=t_extra" in body and "turn_witnesses=t_rep" in body
    assert "frames=frames" in body                       # the cloud is built over the keyframes


def test_the_interval_witnesses_of_their_own_pass_are_read_like_the_turn_frames(tmp_path):
    """USER 2026-11-04 ("que solo entren para afirmar o no en el F6"): the interval frames come
    from a pass of their own after the chunk (w<chunk>_<frame>.npz) — F6 reads them with the same
    carry as the turn frames (c<chunk>_<frame>.npz) and they vote on the keyframes near their
    anchor."""
    from precision.depth_on_f5 import design, load_turn_witnesses
    out = tmp_path / "output"
    d = out / "maplong_run" / CONTEXT_DIR
    d.mkdir(parents=True)
    (out / "maplong_run" / "fork_stamp.json").write_text(json.dumps({"sha256": "S"}))
    H, W = 4, 5
    frames = [10, 20, 30, 40]
    rel = _pose(0.05, 0.1)
    _write_view(d, 1, 15, 10, "S", np.full((H, W), 2.0, np.float32), rel, 2.0)              # turn
    _write_view(d, 1, 25, 20, "S", np.full((H, W), 2.0, np.float32), rel, 2.0, prefix="w")  # interval
    chunk = {10: 1, 20: 1, 30: 2, 40: 2}
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Dm = design(uu.ravel(), vv.ravel(), W, H)
    coefs = {i: np.array([1.1, 0.0, 0.0]) for i in range(4)}
    c2w = {f: _pose(float(f)) for f in frames}
    dep, valid, passed, cw, extra, rep = load_turn_witnesses(
        out, frames, chunk, {1: 5.0, 2: 5.0}, {10: 3.0, 20: 3.0, 30: 3.0, 40: 3.0}, coefs, c2w,
        None, False, (H, W), Dm, H, W, [-1, 1], log=lambda m: None)
    assert sorted(dep) == [15, 25] and rep["used"] == 2
    assert np.allclose(dep[25], 2.0 * 1.5 * 1.1)
    assert extra == {10: [15, 25], 20: [15, 25], 30: [25]}
