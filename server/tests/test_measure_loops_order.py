"""The fork's loop measurement in a planned order (vggt_long.VGGT_Long._stac_measure_loops, run
from its source on a stand-in object — the class needs the GPU stack to import): every bridge is
measured on ITS two chunks and returned in its own place, never more than two chunks held at
once, and on station's bridge sequence (2026-11-01) 17 chunk loads per pass instead of 29."""
import ast
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FORK = ROOT / "vendor" / "VGGT-Long"
SRC = (FORK / "vggt_long.py").read_text()

STATION = [(0, 0), (0, 0), (6, 6), (7, 1), (0, 0), (4, 2), (2, 2), (7, 3), (6, 1), (7, 6), (4, 3),
           (7, 2), (1, 0), (3, 2), (5, 4), (7, 7), (7, 5), (7, 2), (6, 3), (3, 1)]


def _method(name):
    tree = ast.parse(SRC)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VGGT_Long")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(SRC, fn))


class _Stub:
    pass


class _Chunk(dict):
    __hash__ = object.__hash__          # identity: the WeakSet tracks each loaded chunk


def _run(tmp_path, monkeypatch, seq):
    """Run the method on ``seq`` (one chunk pair per bridge); returns (results, the chunk files
    loaded in order, the most chunks alive at any measurement)."""
    import gc
    import types
    import weakref
    live = weakref.WeakSet()
    loads = []

    def counting_load(path, *a, **kw):
        loads.append(Path(path).name)
        c = _Chunk(chunk=int(Path(path).stem.split("_")[1]))
        live.add(c)
        return types.SimpleNamespace(item=lambda c=c: c)

    peak = [0]

    def fake_measure(pred, layout, da, db, lcfg, rigid):
        gc.collect()
        peak[0] = max(peak[0], len(live))
        return {"bridge": pred, "a": da["chunk"], "b": db["chunk"], "rigid": rigid}

    mod = types.ModuleType("loop_utils.loop_bridges")
    mod.measure_bridge = fake_measure
    monkeypatch.setitem(sys.modules, "loop_utils.loop_bridges", mod)
    monkeypatch.setitem(sys.modules, "loop_utils", types.ModuleType("loop_utils"))
    ns = {"np": types.SimpleNamespace(load=counting_load), "os": os}
    exec(_method("_stac_measure_loops"), ns)
    stub = _Stub()
    stub.result_unaligned_dir = str(tmp_path)
    stub.loop_predict_list = [((a, (0, 1), b, (0, 1)), li, {"layout": {}}) for li, (a, b) in enumerate(seq)]
    stub._stac_loops_cfg = lambda: {}
    stub._stac_drop_chunk_cache = lambda: None
    # a bridge held in memory is read as it is (the fork's reader, 2026-11-04: a bridge on disk
    # is read back from its file instead)
    exec(_method("_stac_bridge_pred"), ns)
    stub._stac_bridge_pred = lambda pred, meta: ns["_stac_bridge_pred"](stub, pred, meta)
    out = ns["_stac_measure_loops"](stub, True)
    return out, loads, peak[0]


def test_every_bridge_gets_its_chunks_in_its_place(tmp_path, monkeypatch):
    out, loads, peak = _run(tmp_path, monkeypatch, STATION)
    assert [o["bridge"] for o in out] == list(range(len(STATION)))
    assert [(o["a"], o["b"]) for o in out] == STATION
    assert all(o["rigid"] for o in out)
    assert len(loads) == 17 and peak <= 2         # list order with the FIFO cache loaded 29


@pytest.mark.parametrize("seed", range(6))
def test_random_sequences_never_hold_more_than_two(tmp_path, monkeypatch, seed):
    rng = np.random.default_rng(seed)
    seq = [tuple(int(x) for x in rng.integers(0, 6, 2)) for _ in range(25)]
    out, loads, peak = _run(tmp_path, monkeypatch, seq)
    assert [(o["a"], o["b"]) for o in out] == seq and [o["bridge"] for o in out] == list(range(25))
    assert peak <= 2
