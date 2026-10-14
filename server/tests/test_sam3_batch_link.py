"""Two SAM3 batches of one category (docs/plan_determinismo.md point 93): an object the
overlap frames never show gets a FRESH id. It used to keep its raw per-session SAM3 id,
which collides with the ids the first batch was given — the object fused with whatever
same-category object held that number (a cross-batch chimera, no IoU test, no record)."""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _BatchedFakeSAM3:
    """``process_batch`` per batch: ``script[category][first frame of the batch]`` =
    {frame: {local obj id: lit pixel}}. Every batch is its own SAM3 session, so its
    local ids restart at 0 like the vendor's."""

    def __init__(self, script):
        self.script = script

    def process_batch(self, batch_dir, category, index_mapping, boxes_by_local=None):
        frames = sorted(index_mapping.values())
        per_frame = self.script.get(category, {}).get(frames[0], {})
        out = {}
        for fr in frames:
            objs = per_frame.get(fr)
            if not objs:
                continue
            ids = sorted(objs)
            masks = np.zeros((len(ids), 4, 4), np.uint8)
            for k, oid in enumerate(ids):
                masks[k].flat[objs[oid]] = 1
            out[fr] = {"out_binary_masks": masks, "out_obj_ids": np.array(ids)}
        return out

    def release_batch_session(self):
        pass

    def unload_model(self):
        pass

    def load_model(self):
        pass


def test_an_object_the_overlap_never_shows_gets_its_own_id(tmp_path, monkeypatch):
    import torch
    import segmentation.sam3_wrapper as w
    from segmentation import pipeline as P
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # 14 frames in batches of 10 with 6 of overlap: [0, 10) and [4, 14), shared 4..9
    first = {f: {0: 1} for f in range(10)}              # door A (pixel 1) in every frame
    for f in range(3):
        first[f][1] = 2                                 # door B (pixel 2), frames 0-2 only
    second = {f: {0: 1} for f in range(4, 14)}          # door A again, linked through 4..9
    for f in range(11, 14):
        second[f][1] = 3                                # door C (pixel 3), frames 11-13 only
    monkeypatch.setattr(w, "get_sam3_wrapper",
                        lambda: _BatchedFakeSAM3({"door": {0: first, 4: second}}))
    frames = tmp_path / "frames_valid"
    frames.mkdir()
    files = [f"{i:06d}.jpg" for i in range(14)]
    for f in files:
        (frames / f).write_bytes(b"")
    out = tmp_path / "output"
    out.mkdir()
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    status = {}
    P._run_sam3_batched(frames, files, ["door"], 10, 6, 0.3, 0.9, output_dir=out, cfg=cfg,
                        prompt_status=status)
    z = np.load(out / "seg_masks.npz")
    got = {}
    for k in z.files:
        if k.startswith("f") and "_o" in k:
            f, o = k[1:].split("_o")
            got.setdefault(int(o), set()).update((int(f), int(p)) for p in np.flatnonzero(z[k]))
    assert sorted(tuple(sorted(v)) for v in got.values()) == sorted([
        tuple((f, 1) for f in range(14)),               # A, one object across both batches
        tuple((f, 2) for f in range(3)),                # B
        tuple((f, 3) for f in range(11, 14)),           # C, its own object
    ])
    assert status["door"]["n_objects"] == 3
