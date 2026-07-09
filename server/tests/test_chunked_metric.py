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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all chunked-metric tests passed")
