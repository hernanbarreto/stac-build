"""Every reader of the chunks reads THE list (USER 2026-10-06: "todo lo que luego use los
chunks también debe tener esa lista") — the co-visibility plan's explicit ranges, variable
lengths, each seam its own overlap — never a layout rebuilt from a size, an overlap or a fixed
divisor. Synthetic, no GPU: an uneven layout whose uniform reading would be wrong everywhere."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.chunk_plan import (ChunkLayoutError, latest_chunk_of,  # noqa: E402
                                       omega_chunk_ranges, chunk_ranges)

# lengths 8 / 24 / 14, seams 4 / 6 — no (size, overlap) reproduces it
UNEVEN = [(0, 8), (4, 28), (22, 36)]
N = 36
H, W = 3, 4


def _write_config(out: Path, ranges):
    out.mkdir(parents=True, exist_ok=True)
    (out / "vggt_omega_config.yaml").write_text(yaml.dump(
        {"Model": {"chunk_ranges": [[a, b] for a, b in ranges], "omega_resolution": 832,
                   "omega_mode": "max_size"}}))


def test_the_run_layout_comes_from_what_the_fork_ran_then_the_config_then_the_plan(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    assert omega_chunk_ranges(out) is None
    (out / "chunk_plan.json").write_text(json.dumps({"chunk_ranges": [[0, 8], [4, 28], [22, 36]],
                                                     "n_keyframes": N}))
    assert omega_chunk_ranges(out)[0] == UNEVEN
    _write_config(out, UNEVEN)
    assert "Model.chunk_ranges" in omega_chunk_ranges(out)[1]
    (out / "maplong_run").mkdir()
    (out / "maplong_run" / "chunk_sim3.json").write_text(json.dumps(
        {"chunk_indices": [list(r) for r in UNEVEN], "sim3": []}))
    r, src = omega_chunk_ranges(out)
    assert r == UNEVEN and src.endswith("chunk_sim3.json")
    # what the fork ran disagreeing with what the run was told: products of another plan
    _write_config(out, [(0, 8), (4, 29), (22, 36)])
    with pytest.raises(ChunkLayoutError, match="another chunk plan"):
        omega_chunk_ranges(out)


def test_the_later_chunk_rule_is_the_uniform_one_on_uniform_layouts():
    for n, cs, ov in [(100, 60, 30), (95, 60, 30), (227, 80, 40), (66, 24, 12)]:
        r = chunk_ranges(n, cs, ov)
        for pos in range(n):
            assert latest_chunk_of(r, pos) == min(pos // (cs - ov), len(r) - 1), (n, cs, ov, pos)
    assert [latest_chunk_of(UNEVEN, p) for p in (0, 3, 4, 7, 8, 21, 22, 27, 28, 35)] == \
        [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def _aligned_chunks(out: Path, ranges, n=N):
    """maplong_run/_tmp_results_aligned/chunk_K.npy whose depth NAMES the keyframe (depth =
    100·K + position) and frame_list.json with real frame numbers 10·position."""
    run = out / "maplong_run"
    al = run / "_tmp_results_aligned"
    al.mkdir(parents=True)
    for k, (a, b) in enumerate(ranges):
        S = b - a
        dep = np.stack([np.full((H, W), 100.0 * k + g, np.float32) for g in range(a, b)])
        np.save(al / f"chunk_{k}.npy", {"depth": dep, "intrinsic": np.tile(np.eye(3), (S, 1, 1)),
                                         "world_points": np.zeros((S, H, W, 3), np.float32),
                                         "world_points_conf": np.ones((S, H, W), np.float32)},
                allow_pickle=True)
    (run / "frame_list.json").write_text(json.dumps([f"{10 * g:06d}.jpg" for g in range(n)]))
    return al


def test_the_tsdf_depth_loader_reads_the_explicit_ranges(tmp_path):
    from segmentation.tsdf_export import _resolve_mapanything_depth
    out = tmp_path / "output"
    _aligned_chunks(out, UNEVEN)
    _write_config(out, UNEVEN)
    res = _resolve_mapanything_depth(out)
    assert res is not None
    load, hw = res
    assert hw == (H, W)
    for g in range(N):
        rec = load(10 * g)
        k = latest_chunk_of(UNEVEN, g)
        assert rec is not None, g
        assert float(np.median(rec["depth"])) == pytest.approx(100.0 * k + g), g
    # a uniform reading of the same files would have been wrong: chunk_step 4 (8 − 4) sends
    # keyframe 30 to chunk 2 at local 22, beyond its 14 frames
    assert (30 - 2 * 4) >= UNEVEN[2][1] - UNEVEN[2][0]


def test_the_tsdf_depth_loader_refuses_a_layout_that_is_not_on_disk(tmp_path):
    from segmentation.tsdf_export import _resolve_mapanything_depth
    out = tmp_path / "output"
    _aligned_chunks(out, UNEVEN)
    _write_config(out, [(0, 8), (4, 36)])          # two chunks configured, three on disk
    assert _resolve_mapanything_depth(out) is None
    out2 = tmp_path / "other" / "output"
    _aligned_chunks(out2, UNEVEN)                   # no record of the layout at all
    assert _resolve_mapanything_depth(out2) is None


def test_the_trace_normals_index_reads_the_explicit_ranges(tmp_path):
    from reconstruction.trace_normals import _load_frame_depth_index
    out = tmp_path / "output"
    al = _aligned_chunks(out, UNEVEN)
    _write_config(out, UNEVEN)
    idx = _load_frame_depth_index(out)
    centres = [(a + b) / 2.0 for a, b in UNEVEN]
    for g in range(N):
        own = min((k for k, (a, b) in enumerate(UNEVEN) if a <= g < b),
                  key=lambda k: (abs(g - centres[k]), k))
        path, local = idx[10 * g]
        assert path == al / f"chunk_{own}.npy" and local == g - UNEVEN[own][0], g


def test_the_origins_and_chunk_meta_read_the_explicit_ranges(tmp_path):
    """The origins fallback (no inline origins): position = local + ranges[K][0], and every
    chunk meta carries its real first / last position — no chunk_step on an explicit layout."""
    from workers.map_worker import _generate_origins

    class _Pipe:
        def send_log(self, *a, **k):
            pass

    out = tmp_path / "output"
    save = out / "maplong_run"
    _aligned_chunks(out, UNEVEN)
    (save / "pcd").mkdir(parents=True)
    for k in range(len(UNEVEN)):
        (save / "pcd" / f"{k}_pcd.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    cfg = {"Model": {"chunk_ranges": [list(r) for r in UNEVEN],
                     "Pointcloud_Save": {"conf_threshold_coef": 0.75, "sample_ratio": 1.0}}}
    _generate_origins(save, out, cfg, _Pipe())
    for k, (a, b) in enumerate(UNEVEN):
        with np.load(out / f"chunk_{k:03d}_origins.npz") as z:
            fg = z["frame_global"]
        assert sorted(set(fg.tolist())) == [10 * g for g in range(a, b)], k
        meta = json.loads((out / f"chunk_{k:03d}_meta.json").read_text())
        assert meta["frame_global_start"] == a and meta["frame_global_end"] == b - 1, k
        assert "chunk_step" not in meta
