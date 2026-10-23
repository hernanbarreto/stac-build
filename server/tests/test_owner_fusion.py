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
        assert int(z["version"]) == 4 and str(z["stage"]) == "owner_fusion"
        assert len(z["frame_global"]) == n == res.report["consumed"] + res.report["free_space"]["rejected"] \
            + res.report["unverified"]["rejected"]
        fused = z["reason"] == OF.REJECT_FUSED
        assert fused.all() and (z["winner_frame"] >= 0).all() and len(z["n_agree"]) == len(z["n_contradict"]) == n
        assert res.report["free_space"]["rejected"] == 0 and (z["n_contradict"] == 0).all()
        # the winner of a consumed pixel is a valid observation of another keyframe
        for k in range(0, n, max(1, n // 50)):
            g, r, c = int(z["winner_frame"][k]), int(z["winner_row"][k]), int(z["winner_col"][k])
            assert g != int(z["frame_global"][k]) and valid[g][r, c]
    # the carry: an exact landing on a flat patch reads the neighbour's depth bilinearly, a patch
    # across a step at the nearest pixel — never across the step
    z_g = np.full((H, W), 4.0, np.float32); z_g[:, W // 2:] = 2.0
    ray_w, C = OF._frame_geometry(0, H, W, K, c2w)
    agree, t, qr, qc, _c = OF.carry(FT._t(depth[0]), FT._t(valid[0], FT.torch.bool), ray_w, C, np.linalg.inv(c2w[1]),
                                FT._t(z_g), FT._t(np.ones((H, W), bool), FT.torch.bool), K, 0.02, True)
    tt = FT._np(t)
    assert np.all(np.isnan(tt) | (np.abs(tt - 4.0) < 0.2) | (np.abs(tt - 2.0) < 0.2))
    assert OF.candidates_of(np.zeros((2, 2)), 0.05, 12) == {0: [], 1: []}   # no overlap: no candidate


def test_the_photometric_tier_decides_where_it_measures_and_the_learned_depth_fills_the_rest():
    """Phase 3: Omega's depth of the wall carries a +1.5 % bias in every keyframe (the bend cannot
    remove what every keyframe shares); COLMAP PatchMatch measured the textured middle stripe
    exactly (tier 0) in three of the five keyframes. The fused cloud is at the photometric depth
    on the stripe — a tier-0 member never averages with the biased tier-1 ones — and at Omega's
    elsewhere; the owners of the stripe are tier-0 observations."""
    biases = [1.015, 1.015, 1.015, 1.015, 1.015]
    frames, c2w, depth, valid = _scene(biases)
    tier = {f: np.ones((H, W), np.int8) for f in frames}
    stripe = (VV > H * 0.35) & (VV < H * 0.65)
    for f in (1, 2, 3):
        depth[f] = np.where(stripe, _wall(c2w[f]), depth[f]).astype(np.float32)   # exact where COLMAP measured
        tier[f][stripe] = 0
    res = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, tier=tier, tau_by_tier=(0.005, 0.02), **CFG)
    P_stripe, P_rest = [], []
    for f in frames:
        own = res.owner[f]
        t0 = own & (res.fused_tier[f] == 0); t1 = own & (res.fused_tier[f] == 1)
        P_stripe.append(_points([f], c2w, res.depth, {f: t0})); P_rest.append(_points([f], c2w, res.depth, {f: t1}))
    P_stripe = np.concatenate(P_stripe); P_rest = np.concatenate(P_rest)
    assert len(P_stripe) > 0.5 * int(stripe.sum()), "the stripe is owned by photometric observations"
    assert _plane_rms(P_stripe) < 0.003, f"stripe rms {_plane_rms(P_stripe) * 100:.2f} cm"
    assert _plane_rms(P_rest) > 0.03, "outside the stripe only Omega measured: its bias stays, declared by the tier"
    rep = res.report["tiers"]
    assert rep["owners_by_fused_tier"]["0"] == len(P_stripe) and rep["tau_by_tier"] == [0.005, 0.02]
    # a tier-0 pixel of keyframe 2 is owned by a tier-0 observation (its own or a nearer one), never
    # consumed by a biased tier-1 one
    wf, wr, wc = res.winner[2]
    consumed0 = (tier[2] == 0) & ~res.owner[2] & (wf >= 0)
    for g in np.unique(wf[consumed0]):
        sel = consumed0 & (wf == g)
        assert (tier[int(g)][wr[sel].astype(int), wc[sel].astype(int)] == 0).all()


def test_free_space_carving_rejects_a_floater_every_other_view_sees_through():
    """Phase 3c (USER 2026-10-23: "no voladores ni puntos donde no van"). Seven keyframes of a wall at
    4 m; three of them also carry a blob 1.2 m in front of the wall (a floater three cameras agree
    on). Every other keyframe measures the wall through the blob's place: more views see free space
    through its points than confirm them — the blob leaves (reason contradicted), with its counts;
    the wall stays whole."""
    biases = [1.0] * 7
    frames, c2w, depth, valid = _scene(biases, xs=[0.3 * (i - 3) for i in range(7)], zs=[0.0] * 7)
    blob = np.zeros((H, W), bool); blob[18:30, 26:38] = True
    for f in (2, 3, 4):
        # the floater: a patch of ONE world plane 1.2 m in front of the wall (z = 2.8 + 0.15 x), seen
        # at the blob's pixels of the three keyframes
        depth[f] = np.where(blob, _wall(c2w[f], z0=2.8).astype(np.float32), depth[f])
    res = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=True, min_evidence=1, **CFG)
    rep = res.report
    assert rep["free_space"]["enabled"] and rep["free_space"]["rejected"] > 0
    for f in (2, 3, 4):
        assert res.free_space[f][blob].mean() > 0.9, f"keyframe {f}: {res.free_space[f][blob].mean():.2f} of the blob seen through"
        assert not res.owner[f][blob].any()
        assert res.contra[f][blob].min() >= 1
        # the wall around the blob is untouched
        ring = np.zeros((H, W), bool); ring[12:36, 20:44] = True; ring &= ~blob
        assert not res.free_space[f][ring].any()
    # the published cloud holds no point near the floater's place
    P = _points(frames, c2w, res.depth, res.owner)
    assert not np.any((P[:, 2] > 2.5) & (P[:, 2] < 3.1)), "floater points published"
    assert _plane_rms(P) < 0.006
    # without carving the blob is published (three agreeing views)
    res0 = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=False, min_evidence=1, **CFG)
    P0 = _points(frames, c2w, res0.depth, res0.owner)
    assert np.any((P0[:, 2] > 2.5) & (P0[:, 2] < 3.1))
    assert res0.report["free_space"]["rejected"] == 0


def test_a_learned_map_never_vetoes_a_photometric_point():
    """The instrument rule of the carving: a tier-0 (COLMAP) point is seen through only by tier-0
    maps. The same floater, but measured by tier 0 in its three keyframes and seen through by
    tier-1 maps only — it stays (a learned depth cannot veto a photometric one)."""
    frames, c2w, depth, valid = _scene([1.0] * 7, xs=[0.3 * (i - 3) for i in range(7)], zs=[0.0] * 7)
    blob = np.zeros((H, W), bool); blob[18:30, 26:38] = True
    tier = {f: np.ones((H, W), np.int8) for f in frames}
    for f in (2, 3, 4):
        depth[f] = np.where(blob, _wall(c2w[f], z0=2.8).astype(np.float32), depth[f])
        tier[f] = np.where(blob, 0, 1).astype(np.int8)
    res = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=True, min_evidence=1,
                  tier=tier, tau_by_tier=(0.005, 0.02), **CFG)
    for f in (2, 3, 4):
        assert not res.free_space[f][blob].any()
        assert res.contra[f][blob].max() == 0


def test_minimum_evidence_marks_what_one_camera_alone_saw():
    """A patch of the wall seen by one keyframe only (the others have no depth there): with
    min_evidence 2 its points are rejected as unverified (reason insufficient_witnesses), never
    deleted — they are in rejected_points.npz with n_agree 0; with min_evidence 1 they publish."""
    frames, c2w, depth, valid = _scene([1.0] * 5, xs=[0.3 * (i - 2) for i in range(5)], zs=[0.0] * 5)
    patch = np.zeros((H, W), bool); patch[4:12, 4:12] = True
    hole = np.zeros((H, W), bool); hole[0:20, 0:30] = True          # where keyframe 2's patch lands in the others
    for f in frames:
        if f != 2:
            valid[f] = valid[f] & ~hole
            depth[f] = np.where(hole, 0.0, depth[f]).astype(np.float32)
    # keyframe 2's patch pixels land where the others have no depth: nobody agrees, nobody sees through
    res2 = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=True, min_evidence=2, **CFG)
    single = res2.unverified[2] & patch
    assert single.sum() > 0.5 * patch.sum() and not res2.owner[2][single].any()
    assert (res2.members[2][single] == 0).all()
    assert res2.report["unverified"]["rejected"] == int(sum(res2.unverified[f].sum() for f in frames))
    res1 = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=True, min_evidence=1, **CFG)
    assert res1.owner[2][single].all() and res1.report["unverified"]["rejected"] == 0
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "rejected_points.npz"
        n = OF.write_rejected(p, frames, res2, valid)
        with np.load(p) as z:
            assert n == len(z["reason"]) and (z["reason"] == OF.REJECT_UNVERIFIED).sum() == res2.report["unverified"]["rejected"]
            unv = z["reason"] == OF.REJECT_UNVERIFIED
            assert (z["n_agree"][unv] == 0).all() and (z["winner_frame"][unv] == -1).all()
