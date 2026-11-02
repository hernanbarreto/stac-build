"""The fork's one-ahead chunk reader (vggt_long.VGGT_Long._stac_iter_chunks, run from its source):
the chunks in the order asked, each the dict np.load gives, never more than two alive — and the
passes that use it (uncertainty, the depth graph's cache, the deferred outputs) read through it."""
import ast
import gc
import os
import textwrap
import weakref
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "vendor" / "VGGT-Long" / "vggt_long.py").read_text()


def _method(name):
    tree = ast.parse(SRC)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VGGT_Long")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(SRC, fn)).replace("@staticmethod\n", "")


class _D(dict):
    __hash__ = object.__hash__


def test_the_chunks_in_order_and_at_most_two_alive(tmp_path):
    for k in range(7):
        np.save(tmp_path / f"chunk_{k}.npy", {"k": k, "a": np.arange(k + 3)}, allow_pickle=True)
    live = weakref.WeakSet()
    real = np.load

    class _np:
        @staticmethod
        def load(path, *a, **kw):
            d = _D(real(path, *a, **kw).item())
            live.add(d)

            class _W:
                def item(self_inner):
                    return d
            return _W()
    ns = {"np": _np, "os": os}
    exec(_method("_stac_iter_chunks"), ns)
    order = [3, 0, 6, 1, 1, 5]
    got = []
    for k, data in ns["_stac_iter_chunks"](str(tmp_path), order):
        gc.collect()
        assert len(live) <= 2
        assert data["k"] == k and np.array_equal(data["a"], np.arange(k + 3))
        got.append(k)
        del data
    assert got == order
    assert list(ns["_stac_iter_chunks"](str(tmp_path), [])) == []


def test_the_per_chunk_passes_read_through_it():
    for name in ("_stac_uncertainty", "_stac_depth_graph", "_stac_write_deferred_outputs",
                 "_stac_holdout_pairs"):
        body = _method(name)
        assert "self._stac_iter_chunks(self.result_aligned_dir" in body, name


def test_the_seams_read_each_chunk_once_through_it():
    body = _method("process_long_sequence")
    seg = body[body.index('print("Aligning all the chunks...")'):]
    seg = seg[:seg.index("self.sim3_list.append((s, R, t))")]
    assert "self._stac_iter_chunks(" in seg and "self.result_unaligned_dir" in seg
    assert "np.load(" not in seg            # no chunk read twice per seam
