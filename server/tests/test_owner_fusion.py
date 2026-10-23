"""Owner fusion (precision/owner_fusion.py, audit 2026-10-23): one point per surface element. A wall
seen by seven keyframes with seven different scale errors came out as seven layers; after the
fusion every element of the wall is published once, on the ray of its nearest confirmed observer,
at the consensus depth — and a drifted revisit is NOT merged. Synthetic, no GPU."""
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import f6_torch as FT                                   # noqa: E402
from precision import owner_fusion as OF                               # noqa: E402

H, W = 48, 64
K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
UU, VV = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
CFG = dict(covis_stride=4, min_covis_frac=0.05, max_candidates=12, evidence_cap=2, interpolate=True,
           nearest_tie_rel=0.02)


def _cam(x, z=0.0):
    c = np.eye(4); c[0, 3] = x; c[2, 3] = z; return c


def _wall(c2w, slope=0.15, z0=4.0):
    """Camera depth of the plane z = z0 + slope·x (world), per pixel of a camera at c2w (axis-aligned)."""
    d = np.stack([(UU - K[0, 2]) / K[0, 0], (VV - K[1, 2]) / K[1, 1], np.ones_like(UU)], -1) @ c2w[:3, :3].T
    C = c2w[:3, 3]
    return (z0 + slope * C[0] - C[2]) / (d[..., 2] - slope * d[..., 0])


def _points(frames, c2w, depth, mask):
    P = []
    for f in frames:
        m = mask[f]
        zz = depth[f][m].astype(np.float64)
        X = np.stack([(UU[m] - K[0, 2]) / K[0, 0] * zz, (VV[m] - K[1, 2]) / K[1, 1] * zz, zz], 1)
        P.append(X @ c2w[f][:3, :3].T + c2w[f][:3, 3])
    return np.concatenate(P) if P else np.zeros((0, 3))


def _plane_rms(P, slope=0.15, z0=4.0):
    n = np.array([-slope, 0.0, 1.0]); n /= np.linalg.norm(n)
    return float(np.sqrt(np.mean(((P @ n) - z0 * n[2]) ** 2)))


def _scene(biases, xs=None, zs=None):
    """A walk TOWARDS the wall (z from −1.5 m to 0) with a little side step: the nearest camera sees
    the wall's centre at the finest footprint, the farthest sees the widest patch."""
    frames = list(range(len(biases)))
    n = len(biases)
    xs = xs if xs is not None else [0.1 * (i - n / 2) for i in frames]
    zs = zs if zs is not None else [-1.5 * (n - 1 - i) / max(n - 1, 1) for i in frames]
    c2w = {f: _cam(x, z) for f, x, z in zip(frames, xs, zs)}
    depth = {f: (_wall(c2w[f]) * b).astype(np.float32) for f, b in zip(frames, biases)}
    valid = {f: np.ones((H, W), bool) for f in frames}
    return frames, c2w, depth, valid


def test_seven_layers_become_one_surface_and_nothing_is_lost():
    biases = [1.007, 0.994, 1.003, 0.997, 1.006, 0.995, 1.001]      # a span of 1.3 %, inside tau
    frames, c2w, depth, valid = _scene(biases)
    tau = 0.02
    before = _points(frames, c2w, depth, valid)
    res = OF.fuse(frames, depth, valid, K, c2w, tau, log=lambda *a: None, **CFG)
    after = _points(frames, c2w, res.depth, res.owner)
    # the layers: every keyframe's bias in metres; the fused cloud: the consensus
    assert _plane_rms(before) > 0.015, "the seven layers must be there before the fusion"
    assert _plane_rms(after) < 0.6 * _plane_rms(before), f"fused rms {_plane_rms(after) * 100:.2f} cm"
    # the CONFIRMED owners (>= 3 members) sit at the consensus: under 6 mm of the true plane
    conf = {f: res.owner[f] & (res.members[f] >= 3) for f in frames}
    assert _plane_rms(_points(frames, c2w, res.depth, conf)) < 0.006
    n_single = max(int(valid[f].sum()) for f in frames)
    assert len(after) < 2.0 * n_single and len(after) < 0.4 * len(before), (len(after), len(before))
    rep = res.report
    assert rep["owners"] == len(after) and rep["owners"] + rep["consumed"] == rep["observations"]
    # every consumed observation is represented: its chain of winners ends at an owner whose fused
    # point stands within a few footprints of where the observation stood
    rp = rep["representation"]
    assert rp["sampled"] > 100 and rp["resolved_share"] == 1.0 and rp["distance_footprints"]["p95"] < 2.0, rp
    # the unconfirmed owners are the far keyframe's periphery, which nobody else sees
    assert rep["members_of_owners"]["p50"] >= 2 and rep["members_of_owners"]["share_zero"] < 0.1
    # every element of the wall the cameras share is represented: sample the true plane inside the
    # middle camera's view and find a fused point within two pixel footprints
    f = frames[6]
    zt = _wall(c2w[f])
    sel = (np.arange(H)[:, None] % 3 == 0) & (np.arange(W)[None, :] % 3 == 0)
    sel &= (UU > 8) & (UU < W - 8)
    zz = zt[sel]
    S = np.stack([(UU[sel] - K[0, 2]) / K[0, 0] * zz, (VV[sel] - K[1, 2]) / K[1, 1] * zz, zz], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
    d2 = ((S[:, None, :] - after[None, :, :]) ** 2).sum(-1).min(1)
    foot = 2.0 * zz / K[0, 0]
    assert (np.sqrt(d2) <= foot).mean() > 0.99
    # the owners are the NEAREST confirmed observers: the last keyframe (at the wall) owns the centre
    # it sees at the finest footprint; the first (farthest) owns only what nobody nearer sees
    own_counts = {f: int(res.owner[f].sum()) for f in frames}
    assert own_counts[6] > own_counts[0] and own_counts[6] > 0.5 * int(valid[6].sum())


def test_a_box_in_front_of_the_wall_never_fuses_with_it():
    frames = [0, 1, 2]
    c2w = {f: _cam(x) for f, x in zip(frames, (-0.3, 0.0, 0.3))}
    depth, valid = {}, {}
    for f in frames:
        z = _wall(c2w[f], slope=0.0)
        X2 = (UU - K[0, 2]) / K[0, 0] * 2.0 + c2w[f][0, 3]; Y2 = (VV - K[1, 2]) / K[1, 1] * 2.0
        box = (np.abs(X2) <= 0.4) & (np.abs(Y2) <= 0.3)
        z = np.where(box, 2.0, z) * (1 + 0.004 * (f - 1))
        depth[f] = z.astype(np.float32); valid[f] = np.ones((H, W), bool)
    res = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, **CFG)
    for f in frames:
        own = res.owner[f]
        zf = res.depth[f][own]; z0 = depth[f][own]
        near = z0 < 3.0
        assert np.all(np.abs(zf[near] - 2.0) < 0.05) and np.all(np.abs(zf[~near] - 4.0) < 0.1)
    assert res.report["observations"] == 3 * H * W


def test_a_drifted_revisit_is_not_merged_and_is_declared_unconfirmed():
    biases = [1.0, 1.0, 1.0, 1.0]
    frames, c2w, depth, valid = _scene(biases, xs=[-0.3, 0.0, 0.3, 0.0], zs=[0.0, 0.0, 0.0, 0.0])
    c2w[3] = _cam(0.0, z=0.3)                            # the revisit's pose drifted 30 cm forward …
    depth[3] = (_wall(_cam(0.0)) ).astype(np.float32)   # … but it measured the wall from where it really was
    res = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, **CFG)
    # keyframe 3's wall sits 30 cm off the others' in the world: nothing agrees, so its pixels stay
    # their own owners (unconfirmed) and the others' are untouched by it
    assert res.owner[3].mean() > 0.95 and (res.members[3][res.owner[3]] == 0).all()
    # the three consistent views publish their wall ONCE between them (the union of their views is
    # ~1.1 views wide): no duplicates, the ties of equal distance decided by the index
    share = sum(res.owner[f].mean() for f in frames[:3])
    assert 0.95 < share < 1.3, share
    P = _points(frames, c2w, res.depth, res.owner)
    assert _plane_rms(P) > 0.1, "the drift must remain visible in the cloud, never hidden"


def test_the_fusion_is_deterministic_and_independent_of_the_keyframe_order():
    biases = [1.01, 0.99, 1.0, 1.005, 0.995]
    frames, c2w, depth, valid = _scene(biases)
    a = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, **CFG)
    b = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, **CFG)
    for f in frames:
        assert np.array_equal(a.owner[f], b.owner[f]) and np.array_equal(a.depth[f], b.depth[f])
    perm = [3, 0, 4, 1, 2]
    c = OF.fuse(perm, depth, valid, K, c2w, 0.02, log=lambda *a: None, **CFG)
    for f in frames:
        assert np.array_equal(a.owner[f], c.owner[f]) and np.array_equal(a.depth[f], c.depth[f])
        assert np.array_equal(a.members[f], c.members[f])


def test_the_rejected_file_names_every_consumed_pixel_with_its_winner(tmp_path):
    biases = [1.0, 1.0, 1.0]
    frames, c2w, depth, valid = _scene(biases, xs=[-0.2, 0.0, 0.2], zs=[0.0, 0.0, 0.0])
    res = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, **CFG)
    n = OF.write_rejected(tmp_path / "rejected_points.npz", frames, res, valid)
    with np.load(tmp_path / "rejected_points.npz") as z:
        assert int(z["version"]) == 3 and str(z["stage"]) == "owner_fusion"
        assert len(z["frame_global"]) == n == res.report["consumed"]
        assert (z["reason"] == OF.REJECT_FUSED).all() and (z["winner_frame"] >= 0).all()
        # the winner of a consumed pixel is a valid observation of another keyframe
        for k in range(0, n, max(1, n // 50)):
            g, r, c = int(z["winner_frame"][k]), int(z["winner_row"][k]), int(z["winner_col"][k])
            assert g != int(z["frame_global"][k]) and valid[g][r, c]
    # the carry: an exact landing on a flat patch reads the neighbour's depth bilinearly, a patch
    # across a step at the nearest pixel — never across the step
    z_g = np.full((H, W), 4.0, np.float32); z_g[:, W // 2:] = 2.0
    ray_w, C = OF._frame_geometry(0, H, W, K, c2w)
    agree, t, qr, qc = OF.carry(FT._t(depth[0]), FT._t(valid[0], FT.torch.bool), ray_w, C, np.linalg.inv(c2w[1]),
                                FT._t(z_g), FT._t(np.ones((H, W), bool), FT.torch.bool), K, 0.02, True)
    tt = FT._np(t)
    assert np.all(np.isnan(tt) | (np.abs(tt - 4.0) < 0.2) | (np.abs(tt - 2.0) < 0.2))
    assert OF.candidates_of(np.zeros((2, 2)), 0.05, 12) == {0: [], 1: []}   # no overlap: no candidate
