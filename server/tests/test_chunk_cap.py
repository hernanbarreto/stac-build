"""The chunk limit (USER 2026-11-04: "¿los chunks no eran de 32?" … "ese es el límite, con 50 % de
solape con corte por covisibilidad"): at most MAX_CHUNK_VIEWS (64) views per Omega pass — keyframes
and the turn frames inferred with them — every block at most half, the 50 % overlap, the cuts still
the co-visibility DP's; a minimum block already over it is admitted and flagged; the turn frames of
such a chunk are all kept (USER 2026-11-04: Omega sees the turn whole). And the map worker: the turn frames before the plan, the
chunk graph mode on a chunked run. CPU only."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.chunk_covis import (MAX_CHUNK_VIEWS, PlanError, inputs_sha256,  # noqa: E402
                                        plan_detail, planning_views)


def _walk(n, d=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return np.full(n, d), rng.uniform(1, 5, n), rng.uniform(0, 3, n - 1)


def test_the_limit_is_vggt_slams_32_new_plus_32_shared():
    assert MAX_CHUNK_VIEWS == 64


def test_a_cheap_long_walk_is_cut_into_chunks_of_at_most_64_with_half_overlap():
    d, z, t = _walk(200, d=0.01)                         # D_total 2 ≤ H: once a single pass
    assert plan_detail(d, z, t)["single_pass"]
    det = plan_detail(d, z, t, max_chunk_views=64)
    assert not det["single_pass"] and "views > 64" in det["why"]
    assert all(b["views"] <= 32 for b in det["blocks"])
    assert all(c["views"] <= 64 for c in det["chunks"])
    r = det["ranges"]
    assert r[0][0] == 0 and r[-1][1] == 200
    for (a0, b0), (a1, b1), blk in zip(r, r[1:], det["blocks"][1:]):
        assert a1 < b0 and [a1, b0] == blk["range"]   # consecutive chunks share exactly one block
    assert det["flags"] == []


def test_a_walk_within_the_limit_stays_one_pass():
    d, z, t = _walk(60, d=0.01)
    det = plan_detail(d, z, t, max_chunk_views=64)
    assert det["single_pass"] and det["ranges"] == [(0, 60)]
    det = plan_detail(d, z, t, max_chunk_views=64, views=np.full(60, 2))     # 120 views
    assert not det["single_pass"]


def test_heavy_keyframes_make_shorter_blocks_and_a_minimum_block_over_the_limit_is_flagged():
    d, z, t = _walk(160, d=0.01)
    v = np.ones(160, dtype=int)
    v[60:100] = 2                                        # a turn: rotation keyframes + turn frames
    det = plan_detail(d, z, t, max_chunk_views=64, views=v)
    assert all(c["views"] <= 64 for c in det["chunks"]) and det["flags"] == []
    in_turn = [b["frames"] for b in det["blocks"] if b["range"][0] >= 60 and b["range"][1] <= 100]
    assert in_turn and max(in_turn) <= 16                # 32 views at 2 per keyframe
    v[100:112] = 4                                       # even the minimum block (12) holds 48 > 32
    det = plan_detail(d, z, t, max_chunk_views=64, views=v)
    assert "over_view_cap_at_minimum" in det["flags"]
    with pytest.raises(PlanError):
        plan_detail(d, z, t, max_chunk_views=64, views=np.zeros(160))


def test_the_views_of_each_planning_keyframe():
    frames = [0, 10, 12, 20, 30]                         # 12 is a rotation keyframe (not planned)
    plan_index = [0, 1, 3, 4]
    turns = ["000005.jpg", "000011.jpg", "000015.jpg", "000016.jpg", "000025.jpg", "000040.jpg"]
    v = planning_views(frames, plan_index, turns)
    # 0: itself + 5 → 2; 10: itself + 11 + rotation kf 12 + 15, 16 → 5; 20: itself + 25 → 2; 30: 1
    assert v.tolist() == [2, 5, 2, 1]
    assert planning_views(frames, plan_index, None).tolist() == [1, 2, 1, 1]
    d, z, t = _walk(4)
    assert inputs_sha256(d, z, t, 24, 13.67) != inputs_sha256(d, z, t, 24, 13.67, 64, v)


def test_the_map_worker_plans_with_the_turn_frames_and_joins_the_chunks_by_sim3():
    src = (Path(__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()
    i_turn = src.index("_turn = session_turn_frames(")
    i_plan = src.index('_cplan = plan_session(output_dir.parent, log=pipe.send_log, turn_files=_turn["files"],')
    i_ctx = src.index('_ctx = chunk_context(_ranges, _sel_files, _turn["files"])')
    assert i_turn < i_plan < i_ctx
    # a chunk over the limit keeps EVERY turn frame (USER 2026-11-04: option 2)
    assert "cap_context" not in src and "kept whole" in src
    assert src.count('turn_files=_turn["files"]') == 2        # the regenerated-windows retry too
    assert 'cfg_v["Model"]["chunk_graph"] = {"mode": "sim3"}' in src
    import yaml
    live = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert live["reconstruction"]["vggtomega"]["chunk_graph"] == "sim3"
