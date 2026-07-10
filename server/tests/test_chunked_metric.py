# STAC-Builder — chunked-metric Omega: unit tests (synthetic, no GPU).
#
# Covers the three pure pieces the two-phase pipeline stands on:
#   1. chunk_plan — walk measurement, meter-sized chunk planning, anchor placement
#      (every chunk must get anchors, and ranges must match VGGT-Long's slicing).
#   2. metric_lock — per-chunk scale recovery from synthetic DA3 anchors and its
#      application to world_points / depth / extrinsics.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "vendor", "VGGT-Long")))

from reconstruction.chunk_plan import (  # noqa: E402
    walk_length_m, plan_chunks, chunk_ranges, plan_anchor_indices,
)
from loop_utils.metric_lock import anchor_ratio, chunk_scale, apply_scale  # noqa: E402

rng = np.random.default_rng(7)


# ── chunk_plan ───────────────────────────────────────────────────────

def test_walk_length():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "camera_poses.txt"
        rows = []
        for i in range(11):                      # 10 steps of 1 m along +z
            M = np.eye(4); M[2, 3] = float(i)
            rows.append(" ".join(f"{v:.6f}" for v in M.reshape(-1)))
        p.write_text("\n".join(rows))
        assert abs(walk_length_m(p) - 10.0) < 1e-6


def test_plan_chunks_by_meters():
    # 100 kf over 100 m → 1 m/kf → 12 m chunks = 12 kf... clamped to min 24
    size, ov = plan_chunks(100, walk_m=100.0, chunk_walk_m=12.0)
    assert size == 24 and ov == 12
    # 200 kf over 100 m → 0.5 m/kf → 12 m = 24 kf
    size, ov = plan_chunks(200, walk_m=100.0, chunk_walk_m=12.0)
    assert size == 24 and ov == 12
    # dense scan: 300 kf over 30 m → 0.1 m/kf → 12 m = 120 kf
    size, ov = plan_chunks(300, walk_m=30.0, chunk_walk_m=12.0)
    assert size == 120 and ov == 60
    # never exceeds the keyframe count or max_size
    size, _ = plan_chunks(40, walk_m=200.0, chunk_walk_m=50.0)
    assert size <= 40
    size, _ = plan_chunks(1000, walk_m=10.0, chunk_walk_m=12.0)
    assert size == 150                            # max clamp


def test_chunk_ranges_match_vendor_slicing():
    """chunk_ranges must reproduce VGGT-Long's process_long_sequence slicing exactly."""
    def vendor(n, chunk_size, overlap):
        if n <= chunk_size:
            return [(0, n)]
        step = chunk_size - overlap
        num = (n - overlap + step - 1) // step
        return [(i * step, min(i * step + chunk_size, n)) for i in range(num)]
    for n, cs, ov in [(100, 60, 30), (90, 60, 30), (95, 60, 30), (66, 40, 20),
                      (227, 80, 40), (57, 60, 30), (61, 60, 30)]:
        assert chunk_ranges(n, cs, ov) == vendor(n, cs, ov), (n, cs, ov)


def test_anchor_indices_cover_every_chunk():
    for n, cs, ov in [(100, 60, 30), (66, 24, 12), (227, 80, 40)]:
        picks = set(plan_anchor_indices(n, cs, ov, per_chunk=3))
        for start, end in chunk_ranges(n, cs, ov):
            inside = [i for i in picks if start <= i < end]
            assert len(inside) >= 2, f"chunk ({start},{end}) has {len(inside)} anchors"
        assert all(0 <= i < n for i in picks)


# ── metric_lock ──────────────────────────────────────────────────────

def _synthetic_chunk(S=4, H=24, W=32, true_scale=0.05):
    """Omega chunk whose depth is `true_scale`× the metric depth. The metric field is
    SMOOTH (like real depth) so cross-resolution resampling in anchor_ratio holds."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 4.0 + 6.0 * np.sin(xx / W * 3.0) * np.cos(yy / H * 2.0) + 0.08 * xx
    metric = np.stack([base + 0.5 * k for k in range(S)]).astype(np.float32)
    metric = np.clip(metric, 1.0, None)
    ext = np.tile(np.eye(4), (S, 1, 1)).astype(np.float32)
    ext[:, :3, 3] = rng.normal(0, 1, (S, 3)) * true_scale
    chunk = {
        "depth": metric * true_scale,
        "world_points": rng.normal(0, 1, (S, H, W, 3)).astype(np.float32) * true_scale,
        "world_points_conf": np.full((S, H, W), 10.0, np.float32),
        "extrinsic": ext,
    }
    return chunk, metric


def test_anchor_ratio_recovers_scale():
    chunk, metric = _synthetic_chunk(true_scale=0.05)
    r = anchor_ratio(chunk["depth"][0], metric[0], conf=chunk["world_points_conf"][0])
    assert r is not None and abs(r - 20.0) < 0.2          # 1/0.05


def test_anchor_ratio_handles_resolution_mismatch_and_sky():
    chunk, metric = _synthetic_chunk()
    import cv2
    da3_small = cv2.resize(metric[0], (16, 12))            # DA3 at another resolution
    conf = chunk["world_points_conf"][0].copy()
    conf[:8] = 0.0                                         # sky band masked to conf 0
    r = anchor_ratio(chunk["depth"][0], da3_small, conf=conf)
    assert r is not None and abs(r - 20.0) < 1.0


def test_chunk_scale_and_apply():
    with tempfile.TemporaryDirectory() as d:
        chunk, metric = _synthetic_chunk(S=6, true_scale=0.02)
        nums = [100, 108, 116, 124, 132, 140]
        for local in (1, 3, 5):                            # 3 anchors inside the chunk
            np.savez(Path(d) / f"frame_{nums[local]}.npz", depth=metric[local])
        s, n, ratios = chunk_scale(chunk, nums, d)
        assert n == 3 and abs(s - 50.0) < 0.5              # 1/0.02

        t_before = np.asarray(chunk["extrinsic"])[:, :3, 3].copy()
        apply_scale(chunk, s)
        assert abs(float(np.median(chunk["depth"] / np.maximum(metric, 1e-9))) - 1.0) < 0.02
        assert np.allclose(np.asarray(chunk["extrinsic"])[:, :3, 3], t_before * s)
        # rotations untouched
        assert np.allclose(np.asarray(chunk["extrinsic"])[:, :3, :3], np.eye(3), atol=1e-6)


def test_chunk_scale_no_anchor():
    with tempfile.TemporaryDirectory() as d:
        chunk, _ = _synthetic_chunk()
        s, n, _ = chunk_scale(chunk, [1, 2, 3, 4], d)
        assert s is None and n == 0


# ── motion keyframe selector (shared by frame selection and phase 2) ─

def test_motion_keyframes_selector():
    import json
    from workers.map_worker import _motion_keyframes
    with tempfile.TemporaryDirectory() as d:
        frames = []
        for i in range(100):
            moving = 10 <= i < 60                  # still → walk → still
            frames.append({"file": f"{i:06d}.jpg",
                           "fft_score": 5000.0 + (i % 7) * 100,
                           "inter_frame_diff": 12.0 if moving else 0.2,
                           "valid": i % 10 != 3})  # some blurry frames
        (Path(d) / "frame_quality.json").write_text(json.dumps({"frames": frames}))
        kf, n_total, soft = _motion_keyframes(Path(d), quantum=60.0)
        assert n_total == 100
        nums = [int(f[:6]) for f in kf]
        moving_kf = [x for x in nums if 10 <= x < 60]
        # one keyframe every ~5 moving frames (60/12), almost none while still
        assert 8 <= len(moving_kf) <= 12, moving_kf
        assert len(nums) - len(moving_kf) <= 2
        # a DENSER quantum yields more keyframes (phase-2 re-selection contract)
        kf2, _, _ = _motion_keyframes(Path(d), quantum=25.0)
        assert len(kf2) > len(kf)
        # picks avoid the invalid (blurry) frames when a sharp neighbour exists
        invalid = {f["file"] for f in frames if not f["valid"]}
        assert sum(1 for f in kf if f in invalid) == 0


# ── scale graph (seam-relative + DA3-absolute fusion) ──────────────

def test_seam_relative_scale():
    from loop_utils.metric_lock import seam_relative_scale
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float32)
    base = 3.0 + np.sin(xx / 9.0) + 0.02 * yy
    # chunk B's raw depth is 0.8x chunk A's for the same pixels → s_B/s_A = 1.25
    r = seam_relative_scale(base, base * 0.8)
    assert r is not None and abs(r - 1.25) < 1e-3


def test_scale_graph_fixes_noisy_anchors():
    from loop_utils.metric_lock import solve_scale_graph
    rng2 = np.random.default_rng(3)
    true = np.array([10.0 * (1.02 ** k) for k in range(13)])      # smooth true scales
    # DA3: each chunk's absolute estimate off by an independent ±8%
    s_da3 = {k: float(true[k] * np.exp(rng2.normal(0, 0.08))) for k in range(13)}
    n_anch = {k: 7 for k in range(13)}
    # seams: relative scale measured to 0.3%
    seam = {k: float(true[k + 1] / true[k] * np.exp(rng2.normal(0, 0.003)))
            for k in range(12)}
    s_opt = solve_scale_graph(s_da3, n_anch, seam, 13)

    rel_err_da3 = np.array([abs(s_da3[k] / s_da3[k + 1] * true[k + 1] / true[k] - 1)
                            for k in range(12)])
    rel_err_opt = np.abs(s_opt[:-1] / s_opt[1:] * true[1:] / true[:-1] - 1)
    # neighbour CONSISTENCY (what kills the seams) must collapse to ~seam noise
    assert rel_err_opt.max() < 0.01, rel_err_opt.max()
    assert rel_err_opt.max() < rel_err_da3.max() / 5
    # and the global metre stays anchored (mean of DA3, ±3%)
    g = float(np.exp(np.mean(np.log(s_opt / true))))
    assert abs(g - 1) < 0.03, g


def test_scale_graph_chunk_without_anchor():
    from loop_utils.metric_lock import solve_scale_graph
    true = [10.0, 11.0, 12.1]
    s_da3 = {0: 10.0, 2: 12.1}            # chunk 1 has NO anchors
    seam = {0: 1.1, 1: 1.1}
    s = solve_scale_graph(s_da3, {0: 5, 2: 5}, seam, 3)
    assert abs(s[1] - 11.0) < 0.15        # recovered purely from the seams


# ── exact seam fit + frame ownership ─────────────────────────────────

def test_robust_rigid_recovers_pose_with_outliers():
    from loop_utils.metric_lock import robust_rigid
    rng3 = np.random.default_rng(5)
    src = rng3.uniform(-5, 5, (20000, 3))
    ang = np.radians(3.0)
    R_true = np.array([[np.cos(ang), -np.sin(ang), 0],
                       [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    t_true = np.array([0.35, -0.10, 0.50])
    dst = src @ R_true.T + t_true + rng3.normal(0, 0.01, src.shape)   # 1 cm noise
    out = rng3.random(len(src)) < 0.2                                  # 20% junk
    dst[out] += rng3.uniform(-3, 3, (int(out.sum()), 3))
    R, t, resid, n = robust_rigid(src, dst, sample=20000)
    assert np.abs(t - t_true).max() < 0.01, t
    assert np.degrees(np.arccos(np.clip((np.trace(R @ R_true.T) - 1) / 2, -1, 1))) < 0.1
    assert resid < 0.05


def test_frame_owner_partitions_every_frame_once():
    from loop_utils.metric_lock import frame_owner
    from reconstruction.chunk_plan import chunk_ranges
    for n, cs_, ov_ in [(192, 28, 14), (100, 60, 30), (66, 40, 20)]:
        ranges = chunk_ranges(n, cs_, ov_)
        owner = frame_owner(ranges, n)
        assert (owner >= 0).all()                       # every frame owned
        for g in range(n):                              # owner really contains g
            s0, e0 = ranges[owner[g]]
            assert s0 <= g < e0
        # each chunk keeps its centre, gives away its far overlap edges
        for k, (s0, e0) in enumerate(ranges):
            mid = (s0 + e0) // 2
            assert owner[mid] == k


# ── elastic per-frame seam consensus ─────────────────────────────────

def _small_rigid(rng_, ang_deg, t_m):
    import cv2
    v = rng_.normal(0, 1, 3)
    v /= np.linalg.norm(v)
    R, _ = cv2.Rodrigues(v * np.radians(ang_deg))
    return R, rng_.normal(0, t_m, 3)


def test_rigid_fraction_endpoints_and_midpoint():
    from loop_utils.metric_lock import rigid_fraction
    ang = np.radians(2.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0],
                  [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    t = np.array([0.03, -0.01, 0.02])
    assert np.allclose(rigid_fraction(R, t, 0.0), np.eye(4), atol=1e-12)
    M1 = rigid_fraction(R, t, 1.0)
    assert np.allclose(M1[:3, :3], R, atol=1e-12) and np.allclose(M1[:3, 3], t)
    Mh = rigid_fraction(R, t, 0.5)                 # half the angle, half the shift
    half = np.degrees(np.arccos(np.clip((np.trace(Mh[:3, :3]) - 1) / 2, -1, 1)))
    assert abs(half - 1.0) < 1e-9 and np.allclose(Mh[:3, 3], t / 2)


def test_elastic_corrections_consensus_exact():
    """The anchoring directive: corrected copies of every shared frame COINCIDE —
    A_g @ T_g == B_g for the dst (chunk j) and src (chunk j+1) sides."""
    from loop_utils.metric_lock import elastic_corrections, rigid_mat
    rng4 = np.random.default_rng(11)
    ci = [(0, 28), (14, 42), (28, 56)]
    fits = {j: {g: _small_rigid(rng4, 0.3, 0.03)
                for g in range(ci[j + 1][0], ci[j][1])} for j in range(2)}
    corr = [elastic_corrections(ci, k, fits) for k in range(3)]
    for j in range(2):
        for g in range(ci[j + 1][0], ci[j][1]):
            T = rigid_mat(*fits[j][g])
            A = corr[j][g - ci[j][0]]
            B = corr[j + 1][g - ci[j + 1][0]]
            assert np.allclose(A @ T, B, atol=1e-10), (j, g)
    # exclusive head of chunk 0 untouched; consensus starts AT chunk 0's own copy
    assert np.allclose(corr[0][:14], np.tile(np.eye(4), (14, 1, 1)), atol=1e-12)
    assert np.allclose(corr[0][14], np.eye(4), atol=1e-10)
    # ... and ends at chunk 1's own copy (identity at ITS centre)
    assert np.allclose(corr[1][27 - 14], np.eye(4), atol=1e-10)
    # a starved frame inherits its nearest fitted neighbour (both sides agree on it)
    del fits[0][20]
    cA = elastic_corrections(ci, 0, fits)
    cB = elastic_corrections(ci, 1, fits)
    T19 = rigid_mat(*fits[0][19])
    assert np.allclose(cA[20] @ T19, cB[20 - 14], atol=1e-10)


def test_elastic_corrections_smooth_fields():
    """One rigid residual per seam (the smooth real-world case): each chunk's
    correction field must step by ~|t|/L between frames — edges bend, interiors
    never tear."""
    from loop_utils.metric_lock import elastic_corrections
    rng5 = np.random.default_rng(12)
    ci = [(0, 28), (14, 42), (28, 56)]
    fits = {}
    for j in range(2):
        R, t = _small_rigid(rng5, 0.2, 0.02)
        fits[j] = {g: (R, t) for g in range(ci[j + 1][0], ci[j][1])}
    for k in range(3):
        corr = elastic_corrections(ci, k, fits)
        steps = np.linalg.norm(np.diff(corr[:, :3, 3], axis=0), axis=1)
        t_max = max(np.linalg.norm(fits[j][g][1]) for j in fits for g in fits[j])
        assert steps.max() <= t_max / 13.0 * 1.5 + 1e-12, (k, steps.max())


def test_elastic_end_to_end_copies_coincide():
    """Full loop on synthetic points: two chunks disagree by a small rigid offset
    per shared frame → robust_rigid fits + elastic corrections put BOTH copies at
    the same 3D position (exact correspondences, so to numerical precision)."""
    from loop_utils.metric_lock import robust_rigid, elastic_corrections
    rng6 = np.random.default_rng(13)
    ci = [(0, 8), (4, 12)]
    fits = {0: {}}
    copies = {}
    for g in range(4, 8):
        p0 = rng6.uniform(-4, 4, (5000, 3))                  # chunk 0's copy
        R, t = _small_rigid(rng6, 0.4, 0.04)
        p1 = (p0 - t) @ R                                    # chunk 1's copy: p0 = R p1 + t
        fit = robust_rigid(p1, p0, sample=5000)
        assert fit is not None
        fits[0][g] = (fit[0], fit[1])
        copies[g] = (p0, p1)
    c0 = elastic_corrections(ci, 0, fits)
    c1 = elastic_corrections(ci, 1, fits)
    for g in range(4, 8):
        p0, p1 = copies[g]
        A, B = c0[g], c1[g - 4]
        q0 = p0 @ A[:3, :3].T + A[:3, 3]
        q1 = p1 @ B[:3, :3].T + B[:3, 3]
        d = np.linalg.norm(q0 - q1, axis=1)
        assert np.median(d) < 1e-6, (g, float(np.median(d)))


# ── chunk health gate + coverage trim ────────────────────────────────

def test_flag_sick_chunks_on_test4_numbers():
    """The gate must reproduce the test4 verdict from the chunks' OWN numbers:
    10 sick by anchor incoherence, 11-12 sick by missing parallax, body clean."""
    from loop_utils.metric_lock import flag_sick_chunks
    tri = {0: 0.098, 1: 0.102, 2: 0.112, 3: 0.085, 4: 0.095, 5: 0.081, 6: 0.109,
           7: 0.111, 8: 0.083, 9: 0.090, 10: 0.046, 11: 0.0051, 12: 0.00084}
    aiq = {0: 0.095, 1: 0.058, 2: 0.100, 3: 0.099, 4: 0.168, 5: 0.012, 6: 0.057,
           7: 0.073, 8: 0.107, 9: 0.120, 10: 0.374, 11: 0.170, 12: 0.168}
    sick = flag_sick_chunks(tri, aiq)
    assert set(sick) == {10, 11, 12}, sick
    assert any("anchors" in r.lower() or "anchor" in r for r in sick[10])
    assert any("triangulation" in r for r in sick[11])
    assert any("triangulation" in r for r in sick[12])


def test_flag_sick_chunks_healthy_session():
    from loop_utils.metric_lock import flag_sick_chunks
    tri = {k: 0.09 + 0.01 * (k % 3) for k in range(8)}
    aiq = {k: 0.06 + 0.02 * (k % 4) for k in range(8)}
    assert flag_sick_chunks(tri, aiq) == {}
    # starved inputs never crash the gate
    assert flag_sick_chunks({0: None, 1: 0.1}, {0: None, 1: None}) == {}


def test_chunk_tri_angle_scale_invariant():
    from loop_utils.metric_lock import chunk_tri_angle
    S, H, W = 6, 16, 16
    ext = np.tile(np.eye(4), (S, 1, 1))
    ext[:, 2, 3] = np.arange(S) * 0.4                       # 40 cm steps
    depth = np.full((S, H, W), 4.0, np.float32)             # 4 m scene
    conf = np.full((S, H, W), 5.0, np.float32)
    a1 = chunk_tri_angle(depth, conf, ext)
    ext2 = ext.copy(); ext2[:, :3, 3] *= 9.7                # metric lock scaling
    a2 = chunk_tri_angle(depth * 9.7, conf, ext2)
    assert abs(a1 - 0.1) < 1e-9 and abs(a1 - a2) / a1 < 1e-6   # float32 depth rounding


def test_elastic_corrections_sick_side_pinned():
    """A healthy chunk never bends toward a sick neighbour: its side of the seam
    stays identity, the sick side adopts the healthy copy fully."""
    from loop_utils.metric_lock import elastic_corrections, rigid_mat
    rng7 = np.random.default_rng(21)
    ci = [(0, 28), (14, 42), (28, 56)]
    fits = {j: {g: _small_rigid(rng7, 0.3, 0.03)
                for g in range(ci[j + 1][0], ci[j][1])} for j in range(2)}
    # chunk 2 sick: seam 1 (chunks 1-2) pins to chunk 1's copy
    c1 = elastic_corrections(ci, 1, fits, sick={2})
    c2 = elastic_corrections(ci, 2, fits, sick={2})
    for g in range(28, 42):
        T = rigid_mat(*fits[1][g])
        assert np.allclose(c1[g - 14], np.eye(4), atol=1e-12), g   # healthy: untouched
        assert np.allclose(c2[g - 28], T, atol=1e-10)              # sick: adopts healthy
    # seam 0 (both healthy) keeps the normal blend + exact consensus
    c0 = elastic_corrections(ci, 0, fits, sick={2})
    for g in range(14, 28):
        T = rigid_mat(*fits[0][g])
        assert np.allclose(c0[g] @ T, c1[g - 14], atol=1e-10)


def test_trim_static_ends():
    from reconstruction.chunk_plan import trim_static_ends
    rng8 = np.random.default_rng(30)
    walk = np.cumsum(rng8.uniform(0.3, 0.5, (40, 1)) * np.array([[1, 0, 0]]), axis=0)
    still_tail = walk[-1] + rng8.normal(0, 0.003, (10, 3))   # turning in place: ~mm steps
    still_head = walk[0] + rng8.normal(0, 0.003, (5, 3))
    centers = np.vstack([still_head, walk, still_tail])
    lo, hi = trim_static_ends(centers)
    assert 4 <= lo <= 6 and len(centers) - 11 <= hi <= len(centers) - 9, (lo, hi)
    # healthy walk untrimmed; degenerate all-static untouched
    assert trim_static_ends(walk) == (0, len(walk))
    static = np.zeros((20, 3)) + rng8.normal(0, 0.001, (20, 3))
    assert trim_static_ends(static) == (0, 20)


# ── per-frame depth graph ────────────────────────────────────────────

def test_pair_depth_relation_robust():
    from loop_utils.metric_lock import pair_depth_relation
    rng9 = np.random.default_rng(40)
    z = rng9.uniform(2.0, 12.0, 6000)
    zd = 1.03 * z + 0.05 + rng9.normal(0, 0.02, z.shape)     # 3% scale + 5 cm offset
    out = rng9.random(len(z)) < 0.15                          # 15% occlusion junk
    zd[out] += rng9.uniform(0.5, 4.0, int(out.sum()))
    al, be, before, n = pair_depth_relation(z, zd)
    assert abs(al - 1.03) < 0.005 and abs(be - 0.05) < 0.03, (al, be)
    # a broken pair (relation far from identity) is rejected, not fitted
    assert pair_depth_relation(z, z * 1.6) is None


def test_solve_depth_graph_recovers_and_collapses():
    """Frames with a smooth true depth-error field: the graph must recover the
    corrections (up to the mean gauge) and collapse pairwise disagreement."""
    from loop_utils.metric_lock import solve_depth_graph
    rng10 = np.random.default_rng(41)
    N = 60
    a_true = 1.0 + 0.02 * np.sin(np.arange(N) / 7.0)          # ±2% depth scale drift
    b_true = 0.05 * np.cos(np.arange(N) / 11.0)               # ±5 cm offset drift
    meas = []
    for f in range(N):
        for d in (1, 3, 7):
            g = f + d
            if g >= N:
                continue
            # measured relation z_g = alpha z_f + beta given the true errors:
            # a_f z + b_f == a_g (alpha z + beta) + b_g  =>
            alpha = a_true[f] / a_true[g]
            beta = (b_true[f] - b_true[g]) / a_true[g]
            meas.append((f, g, alpha * np.exp(rng10.normal(0, 5e-4)),
                         beta + rng10.normal(0, 2e-3)))
    a, b = solve_depth_graph(meas, N)
    # gauge: compare shape, not absolute level
    ra = a / np.exp(np.mean(np.log(a))) - a_true / np.exp(np.mean(np.log(a_true)))
    assert np.abs(ra).max() < 0.004, np.abs(ra).max()
    # pairwise disagreement at z=5 m collapses by >10x after the corrections
    d_before = np.median([abs((a_true[f] * 5 + b_true[f]) - (a_true[g] * 5 + b_true[g]))
                          for f, g, _, _ in meas])
    resid = [abs((a[f] * 5 + b[f]) - (a[g] * (al * 5 + be) + b[g]))
             for f, g, al, be in meas]
    assert np.median(resid) < d_before / 10, (np.median(resid), d_before)


def test_apply_depth_correction_moves_along_rays():
    from loop_utils.metric_lock import apply_depth_correction
    H, W = 8, 10
    cam = np.array([1.0, 2.0, 3.0])
    dirs = np.stack(np.meshgrid(np.linspace(-0.2, 0.2, W),
                                np.linspace(-0.15, 0.15, H), indexing="xy") + [np.ones((H, W))],
                    axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    depth = np.full((H, W), 5.0, np.float32)
    wp = cam + dirs * (depth[..., None] / dirs[..., 2:])      # z-depth 5 m
    wp2, d2 = apply_depth_correction(wp.astype(np.float32), depth, cam, 1.02, 0.04)
    assert np.allclose(d2, 5.0 * 1.02 + 0.04, atol=1e-5)
    # every point stays on its original ray from the camera
    r0 = wp - cam; r1 = wp2 - cam
    cosang = (r0 * r1).sum(-1) / (np.linalg.norm(r0, axis=-1) * np.linalg.norm(r1, axis=-1))
    assert cosang.min() > 1 - 1e-9
    # invalid depth pixels untouched
    depth_bad = depth.copy(); depth_bad[0, 0] = 0.0
    wp3, d3 = apply_depth_correction(wp.astype(np.float32), depth_bad, cam, 1.02, 0.04)
    assert np.allclose(wp3[0, 0], wp[0, 0]) and d3[0, 0] == 0.0


def test_depth_pair_samples_end_to_end():
    """Two frames observing the same wall with a 2% depth-scale disagreement:
    depth_pair_samples + pair_depth_relation must measure alpha ~= 1/1.02."""
    from loop_utils.metric_lock import depth_pair_samples, pair_depth_relation
    H, W = 60, 80
    fx = fy = 70.0; cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones((H, W))], -1)
    # frame g at origin, TILTED wall (depth varies 4-8 m across the image so the
    # affine fit has slope information); frame f 0.5 m to the right sees the same
    # wall but with its depth field SCALED 1.02
    z_g = 6.0 + 2.0 * (uu - cx) / (W / 2) * rays[..., 2]      # planar, tilted in x
    wp_g = rays * z_g[..., None]
    cam_f = np.array([0.5, 0.0, 0.0])
    wp_f = cam_f + (wp_g - cam_f) * 1.02
    conf = np.full((H, W), 9.0, np.float32)
    w2c_g = np.eye(4)
    out = depth_pair_samples(wp_f, conf, wp_g, conf, w2c_g, K)
    assert out is not None
    al, be, before, n = pair_depth_relation(out[0], out[1])
    assert abs(al - 1 / 1.02) < 0.01 or abs(al * 1.02 - 1) < 0.02, al
    assert before > 0.01                                     # the 2% is visible


def test_blend_two_copies():
    from loop_utils.metric_lock import blend_two_copies
    rng12 = np.random.default_rng(60)
    H, W = 6, 8
    w1 = rng12.normal(0, 1, (H, W, 3)).astype(np.float32)
    w2 = w1 + rng12.normal(0, 0.02, (H, W, 3)).astype(np.float32)
    c1 = np.full((H, W), 5.0, np.float32); c2 = np.full((H, W), 8.0, np.float32)
    c1[0, 0] = 0.0                                    # only copy 2 valid there
    c2[1, 1] = 0.0                                    # only copy 1 valid there
    d1 = np.full((H, W), 4.0, np.float32); d2 = np.full((H, W), 4.2, np.float32)
    wp, cf, dd = blend_two_copies(w1, c1, w2, c2, d1, d2)
    assert np.allclose(wp[2, 2], 0.5 * (w1[2, 2] + w2[2, 2]), atol=1e-6)
    assert np.allclose(wp[0, 0], w2[0, 0]) and np.allclose(wp[1, 1], w1[1, 1])
    assert cf[2, 2] == 8.0 and cf[0, 0] == 8.0 and cf[1, 1] == 5.0
    assert np.allclose([dd[2, 2], dd[0, 0], dd[1, 1]], [4.1, 4.2, 4.0], atol=1e-6)
    # idempotent once both copies hold the blend
    wp2, cf2, dd2 = blend_two_copies(wp, cf, wp, cf, dd, dd)
    assert np.array_equal(wp2, wp) and np.array_equal(cf2, cf) and np.array_equal(dd2, dd)


def test_classify_far_points_contradiction():
    """The v2 policy core: a near frame g sees a tilted wall; far points that
    (a) match it -> AGREE (kept), (b) sit 1 m displaced -> CONTRADICTED (dropped),
    (c) fall outside g's FOV or beyond g's reach -> untested (kept)."""
    from loop_utils.metric_lock import classify_far_points
    H, W = 60, 80
    fx = fy = 70.0; cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones((H, W))], -1)
    depth_g = (4.0 + 1.5 * (uu - cx) / (W / 2)).astype(np.float32)   # tilted wall
    conf_g = np.full((H, W), 9.0, np.float32)
    wp_g = rays * depth_g[..., None]
    cam_g = np.zeros(3); w2c_g = np.eye(4)
    # far points: on-surface / displaced 1 m deeper / far behind the camera's FOV
    pix = [(30, 20), (30, 40), (30, 60)]
    on = np.array([wp_g[v, u] for v, u in pix])
    off = on + np.array([0, 0, 1.0])
    out = np.array([[50.0, 0.0, -5.0]])                 # behind g -> untested
    pts = np.vstack([on, off, out])
    ok = np.ones(len(pts), bool)
    agree, contra = classify_far_points(pts, ok, cam_g, depth_g, conf_g, w2c_g, K,
                                        cap=8.0, floor_m=0.045, rate=0.0067)
    assert agree[:3].all() and not contra[:3].any()      # corroborated
    assert contra[3:6].all() and not agree[3:6].any()    # displaced duplicate
    assert not agree[6] and not contra[6]                # unseen -> keep
    # a camera too far from the points tests nothing
    a2, c2 = classify_far_points(pts, ok, cam_g + 50.0, depth_g, conf_g, w2c_g, K,
                                 cap=8.0, floor_m=0.045, rate=0.0067)
    assert not a2.any() and not c2.any()


def test_write_depth_cap_math():
    """cap = seam floor / pairwise error rate — with test4's real numbers the cone
    gallery's far observations (15-25 m) must fall OUTSIDE the cap while the
    galleries themselves (2-8 m) stay inside."""
    floor_m = 0.045          # median elastic per-frame residual, test4
    rate = 0.0067            # median pairwise depth disagreement, test4
    cap = floor_m / rate
    assert 5.0 < cap < 9.0, cap          # ~6.7 m for this session
    assert cap < 15.0                     # chunk 4's far view of the cones: dropped
    assert cap > 4.0                      # the gallery structures themselves: kept


# ── zoom detection ───────────────────────────────────────────────────

def test_flag_sick_chunks_zoom():
    from loop_utils.metric_lock import flag_sick_chunks
    tri = {k: 0.09 for k in range(13)}                        # parallax fine everywhere
    fx = {k: 550.0 + 3.0 * (k % 5) for k in range(10)}
    fx.update({10: 904.0, 11: 1317.0, 12: 1199.0})            # test4's zoom tail
    sick = flag_sick_chunks(tri, {}, fx_median=fx)
    assert set(sick) == {10, 11, 12}, sick
    assert all(any("ZOOM" in r for r in sick[k]) for k in sick)


def test_trim_zoom_tail_with_jumpy_poses():
    """test4's actual failure mode: the zoomed tail's garbage poses JUMP metres
    (not static), so the step criterion alone misses it — fx must catch it."""
    from reconstruction.chunk_plan import trim_static_ends
    rng11 = np.random.default_rng(50)
    walk = np.cumsum(rng11.uniform(0.3, 0.5, (50, 1)) * np.array([[1, 0, 0]]), axis=0)
    jumpy = walk[-1] + rng11.uniform(-2, 2, (12, 3))          # metre-scale garbage jumps
    centers = np.vstack([walk, jumpy])
    fx = np.concatenate([550 + rng11.normal(0, 5, 50), rng11.uniform(700, 1330, 12)])
    lo, hi = trim_static_ends(centers)                        # steps alone: misses it
    assert hi == len(centers)
    lo, hi = trim_static_ends(centers, fx=fx)                 # fx: catches it
    assert lo == 0 and 49 <= hi <= 51, (lo, hi)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all chunked-metric tests passed")
