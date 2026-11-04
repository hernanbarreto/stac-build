"""The fork's CONTEXT views (USER 2026-11-04, vendor/VGGT-Long/loop_utils/context_views.py and
vggt_long.VGGT_Long.process_single_chunk run from its source): a chain chunk is inferred with its
keyframes and its turn frames in frame order, and the chunk file keeps the keyframes' slots only —
the same keyframes, poses, depth and images every reader of the chunks saw before; a chunk on
disk inferred with other context views is never resumed; the views' bytes are in the run stamp.
CPU only, a stand-in model."""
import ast
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FORK = ROOT / "vendor" / "VGGT-Long"
sys.path.insert(0, str(FORK))

from loop_utils.context_views import (ContextViewsError, check_chunk_context,  # noqa: E402
                                      keep_keyframe_slots, merge_views)

SRC = (FORK / "vggt_long.py").read_text()


def _fn(f):
    return f"{f:06d}.jpg"


def _frames(tmp_path, numbers):
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    for f in numbers:
        (d / _fn(f)).write_bytes(b"jpg")
    return d


def test_the_views_are_merged_in_frame_order_and_the_keyframe_slots_kept():
    kf = ["/f/000000.jpg", "/f/000010.jpg", "/f/000020.jpg"]
    ctx = ["/f/000015.jpg", "/f/000003.jpg"]
    views, keep = merge_views(kf, ctx)
    assert views == ["/f/000000.jpg", "/f/000003.jpg", "/f/000010.jpg", "/f/000015.jpg",
                     "/f/000020.jpg"]
    assert keep == [0, 2, 4]
    with pytest.raises(ContextViewsError, match="precedes"):
        merge_views(kf[1:], ["/f/000005.jpg"])
    with pytest.raises(ContextViewsError, match="repeats"):
        merge_views(kf, ["/f/000010.jpg"])


def test_only_per_view_arrays_are_sliced_and_a_stranger_fails():
    pred = {"depth": np.arange(5 * 2 * 3).reshape(5, 2, 3), "extrinsic": np.arange(5 * 16)
            .reshape(5, 4, 4), "mask": None, "_stac_range": [0, 3]}
    keep_keyframe_slots(pred, [0, 2, 4], 5)
    assert pred["depth"].shape == (3, 2, 3) and pred["extrinsic"].shape == (3, 4, 4)
    assert np.array_equal(pred["depth"][1], np.arange(30).reshape(5, 2, 3)[2])
    assert pred["mask"] is None and pred["_stac_range"] == [0, 3]
    with pytest.raises(ContextViewsError, match="first axis"):
        keep_keyframe_slots({"depth": np.zeros((4, 2))}, [0, 2], 5)


def test_the_configured_context_is_checked_against_the_layout(tmp_path):
    d = _frames(tmp_path, range(0, 41))
    img = [str(d / _fn(f)) for f in (0, 10, 20, 30, 40)]
    chunks = [(0, 3), (1, 5)]
    assert check_chunk_context(None, chunks, img, str(d)) == [[], []]
    got = check_chunk_context([[_fn(15), _fn(5)], [_fn(25)]], chunks, img, str(d))
    assert got == [[str(d / _fn(5)), str(d / _fn(15))], [str(d / _fn(25))]]
    for bad, msg in (([[_fn(5)]], "one list per chunk"),
                     ([[_fn(10)], []], "is a keyframe"),
                     ([[_fn(25)], []], "strictly between"),
                     ([[], [_fn(5)]], "strictly between"),
                     ([[_fn(5), _fn(5)], []], "listed twice"),
                     ([["sub/000005.jpg"], []], "file name")):
        with pytest.raises(ContextViewsError, match=msg):
            check_chunk_context(bad, chunks, img, str(d))
    (d / _fn(7)).unlink()
    with pytest.raises(ContextViewsError, match="does not exist"):
        check_chunk_context([[_fn(7)], []], chunks, img, str(d))


def _method(name):
    tree = ast.parse(SRC)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VGGT_Long")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(SRC, fn))


class _Model:
    def __init__(self):
        self.calls = []

    def infer_chunk(self, paths):
        self.calls.append([os.path.basename(p) for p in paths])
        S = len(paths)
        num = torch.tensor([float(int(Path(p).stem)) for p in paths])
        depth = num.view(1, S, 1, 1).expand(1, S, 2, 3).clone()
        ext = torch.eye(4).repeat(1, S, 1, 1)
        ext[0, :, 0, 3] = num
        return {"world_points": depth[..., None].expand(1, S, 2, 3, 3).clone(),
                "world_points_conf": torch.ones(1, S, 2, 3), "extrinsic": ext,
                "intrinsic": torch.eye(3).repeat(1, S, 1, 1), "depth": depth,
                "depth_conf": torch.ones(1, S, 2, 3),
                "images": torch.zeros(1, S, 3, 2, 3), "mask": None,
                "camera_and_register_tokens": num.view(1, S, 1, 1).expand(1, S, 2, 4).clone()}


class _Mismatch(Exception):
    pass


def _runner(tmp_path, img, chunks, context):
    ns = {"np": np, "os": os, "torch": torch, "_StacPlanMismatch": _Mismatch,
          "_stac_images_to_u8": lambda p: None}
    exec(_method("process_single_chunk"), ns)
    exec(_method("_stac_save_context_views"), ns)

    class _Self:
        pass
    s = _Self()
    s.img_list = img
    s.chunk_indices = chunks
    s.chunk_context = context
    s.result_unaligned_dir = str(tmp_path / "unaligned")
    s.result_loop_dir = str(tmp_path / "loop")
    os.makedirs(s.result_unaligned_dir, exist_ok=True)
    s.all_camera_poses, s.all_camera_intrinsics = [], []
    s.model = _Model()
    s._stac_check_fork_stamp = lambda pred, path: None
    def _sky(pred, paths):
        # the stand-in segmenter: frame 15's corner pixel is sky
        s.__dict__.setdefault("sky", []).append([os.path.basename(p) for p in paths])
        for i, p in enumerate(paths):
            if os.path.basename(p) == _fn(15):
                pred["world_points_conf"][i, 0, 0] = 0.0
    s._stac_mask_sky = _sky
    s._stac_layout_stamp = lambda: [list(c) for c in chunks]
    s._stac_stamp_digest = lambda: None
    s.output_dir = str(tmp_path / "ml")
    s._stac_save_context_views = lambda *a: ns["_stac_save_context_views"](s, *a)
    return s, ns["process_single_chunk"]


def test_a_chunk_is_inferred_with_its_context_and_saved_with_its_keyframes(tmp_path):
    d = _frames(tmp_path, range(0, 41))
    img = [str(d / _fn(f)) for f in (0, 10, 20, 30, 40)]
    chunks = [(0, 3), (1, 5)]
    ctx = check_chunk_context([[_fn(5), _fn(15)], [_fn(25)]], chunks, img, str(d))
    s, run = _runner(tmp_path, img, chunks, ctx)
    run(s, chunks[0], chunk_idx=0)
    assert s.model.calls == [[_fn(0), _fn(5), _fn(10), _fn(15), _fn(20)]]
    pred = np.load(tmp_path / "unaligned" / "chunk_0.npy", allow_pickle=True).item()
    assert pred["depth"].shape == (3, 2, 3)
    assert [float(x) for x in pred["depth"][:, 0, 0]] == [0.0, 10.0, 20.0]
    assert [float(x) for x in pred["extrinsic"][:, 0, 3]] == [0.0, 10.0, 20.0]
    assert pred["camera_and_register_tokens"].shape[0] == 3 and pred["images"].shape[0] == 3
    assert pred["_stac_context"] == [_fn(5), _fn(15)] and pred["_stac_range"] == [0, 3]
    # the sky mask sees the WHOLE pass, once, before the context views are kept (USER 2026-11-04)
    assert s.sky == [[_fn(0), _fn(5), _fn(10), _fn(15), _fn(20)]]
    # the context views stay as F6 witnesses (USER 2026-11-04): anchored on the keyframe before
    # (the stand-in's depth is the frame number: keyframe 0 has no valid depth to anchor frame 5)
    saved = sorted((tmp_path / "ml" / "context_views").glob("*.npz"))
    assert [x.name for x in saved] == ["c000_000015.npz"]
    with np.load(saved[0]) as z:
        assert int(z["anchor_frame"]) == 10 and int(z["anchor_index"]) == 1 and int(z["chunk"]) == 0
        assert float(z["depth"][0, 0]) == 15.0 and str(z["frame_file"]) == _fn(15)
        assert float(z["conf"][0, 0]) == 0.0 and float(z["conf"][1, 2]) == 1.0   # its sky masked
    assert [float(x) for x in s.all_camera_poses[0][1][:, 0, 3]] == [0.0, 10.0, 20.0]
    # resumed by the same context: no inference
    s2, run2 = _runner(tmp_path, img, chunks, ctx)
    run2(s2, chunks[0], chunk_idx=0)
    assert s2.model.calls == [] and len(s2.all_camera_poses) == 1
    # the same keyframes with OTHER (or no) context views are another chunk
    s3, run3 = _runner(tmp_path, img, chunks, [[_fn(5)], [_fn(25)]])
    with pytest.raises(_Mismatch, match="context views"):
        run3(s3, chunks[0], chunk_idx=0)
    s4, run4 = _runner(tmp_path, img, chunks, [[], []])
    with pytest.raises(_Mismatch, match="context views"):
        run4(s4, chunks[0], chunk_idx=0)


def test_without_context_the_chunk_is_the_one_it_always_was(tmp_path):
    d = _frames(tmp_path, range(0, 41))
    img = [str(d / _fn(f)) for f in (0, 10, 20, 30, 40)]
    chunks = [(0, 3), (1, 5)]
    s, run = _runner(tmp_path, img, chunks, check_chunk_context(None, chunks, img, str(d)))
    run(s, chunks[1], chunk_idx=1)
    assert s.model.calls == [[_fn(10), _fn(20), _fn(30), _fn(40)]]
    assert s.sky == [[_fn(10), _fn(20), _fn(30), _fn(40)]]
    pred = np.load(tmp_path / "unaligned" / "chunk_1.npy", allow_pickle=True).item()
    assert "_stac_context" not in pred and pred["depth"].shape == (4, 2, 3)


def test_a_loop_bridge_never_takes_the_chunks_context(tmp_path):
    body = _method("process_single_chunk")
    seg = body[body.index("_ctx = ("):body.index("_ctx_names =")]
    assert "not is_loop and range_2 is None" in seg


def test_the_layout_resolves_the_context_and_the_stamp_reads_its_bytes(tmp_path):
    body = _method("_stac_build_layout")
    assert "check_chunk_context(self.chunk_context_cfg, self.chunk_indices," in body
    from loop_utils import fork_stamp as FS
    d = _frames(tmp_path, range(0, 41))
    cfg = {"Model": {"chunk_context": [[_fn(5)], [_fn(25)]], "loop_enable": False},
           "Weights": {}}
    seen = {}

    class _R:
        @staticmethod
        def stamp(inputs, code, config):
            seen.update(inputs)
            return {"sha256": "x", "inputs": inputs, "code": {}}
    real = FS.repro
    FS.repro = lambda: _R
    try:
        FS.run_stamp(img_list=[str(d / _fn(0)), str(d / _fn(10))], img_dir=str(d), config=cfg,
                     code_files=[])
    finally:
        FS.repro = real
    assert seen["frames/000005.jpg"] == str(d / _fn(5))
    assert seen["frames/000025.jpg"] == str(d / _fn(25))
