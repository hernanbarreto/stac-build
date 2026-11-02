"""reconstruction/fork_scratch.py — Omega's chunk directories on the local disk through symlinks:
placed when they fit (the paths unchanged, the files written and read through the link), kept on the
network when they do not or when a real directory already holds a run's chunks, a lost target
re-created empty, removal through the link (target and link), orphan targets collected; and the map
worker's wipes and cleanups go through it (no target left behind)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from reconstruction import fork_scratch as FS


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    root = tmp_path / "local_scratch"
    monkeypatch.setattr(FS, "SCRATCH_ROOT", root)
    return root


def test_estimate_is_every_chunk_once_plus_the_largest(scratch):
    est = FS.estimate_bytes([127, 67, 79], (1024, 576))
    assert est == (127 + 67 + 79 + 127) * 1024 * 576 * FS.BYTES_PER_PIXEL_BOUND


def test_place_links_and_the_files_go_through(tmp_path, scratch):
    save = tmp_path / "out" / "maplong_run"
    rec = FS.place(save, ["_tmp_results_unaligned", "_tmp_results_aligned"], 1000, log=lambda m: None)
    assert rec["placed"] == ["_tmp_results_unaligned", "_tmp_results_aligned"] and not rec["kept"]
    for sub in rec["placed"]:
        link = save / sub
        assert link.is_symlink() and Path(os.readlink(link)) == FS.scratch_dir_for(save) / sub
    np.save(save / "_tmp_results_unaligned" / "chunk_0.npy", {"a": np.arange(5)}, allow_pickle=True)
    assert (FS.scratch_dir_for(save) / "_tmp_results_unaligned" / "chunk_0.npy").is_file()
    d = np.load(save / "_tmp_results_unaligned" / "chunk_0.npy", allow_pickle=True).item()
    assert np.array_equal(d["a"], np.arange(5))
    owner = json.loads((FS.scratch_dir_for(save) / FS.OWNER_FILE).read_text())
    assert owner["save_dir"] == str(save.resolve())
    # placing again reuses the links; a lost target comes back empty
    os.system(f"rm -rf '{FS.scratch_dir_for(save) / '_tmp_results_aligned'}'")
    rec2 = FS.place(save, ["_tmp_results_unaligned", "_tmp_results_aligned"], 1000, log=lambda m: None)
    assert rec2["placed"] == rec["placed"]
    assert (save / "_tmp_results_aligned").is_dir() and not any((save / "_tmp_results_aligned").iterdir())
    assert (save / "_tmp_results_unaligned" / "chunk_0.npy").is_file()        # kept on reuse
    os.makedirs(save / "_tmp_results_aligned", exist_ok=True)                 # the fork's own makedirs


def test_too_big_or_already_on_the_network_stays(tmp_path, scratch, monkeypatch):
    save = tmp_path / "out" / "maplong_run"
    monkeypatch.setattr(FS, "_free_bytes", lambda root: 10_000)
    rec = FS.place(save, ["_tmp_results_unaligned"], 9_000, log=lambda m: None)
    assert rec["placed"] == [] and rec["kept"] == ["_tmp_results_unaligned"] and "network" in rec["reason"]
    assert not (save / "_tmp_results_unaligned").exists()
    monkeypatch.setattr(FS, "_free_bytes", lambda root: 10 ** 12)
    (save / "_tmp_results_unaligned").mkdir(parents=True)
    (save / "_tmp_results_unaligned" / "chunk_0.npy").write_bytes(b"x")        # a run started on the network
    rec = FS.place(save, ["_tmp_results_unaligned", "_tmp_results_aligned"], 9_000, log=lambda m: None)
    assert rec["kept"] == ["_tmp_results_unaligned"] and rec["placed"] == ["_tmp_results_aligned"]
    assert not (save / "_tmp_results_unaligned").is_symlink()


def test_remove_through_the_link_and_gc_of_orphans(tmp_path, scratch):
    save = tmp_path / "out" / "maplong_run"
    FS.place(save, ["_tmp_results_unaligned"], 10, log=lambda m: None)
    (save / "_tmp_results_unaligned" / "chunk_0.npy").write_bytes(b"0123456789")
    target = FS.scratch_dir_for(save) / "_tmp_results_unaligned"
    assert FS.remove(save / "_tmp_results_unaligned") == 10
    assert not (save / "_tmp_results_unaligned").exists() and not target.exists()
    # a wipe that unlinked the link (rmtree of the output does not follow links) leaves an orphan
    FS.place(save, ["_tmp_results_aligned"], 10, log=lambda m: None)
    (save / "_tmp_results_aligned" / "chunk_1.npy").write_bytes(b"abc")
    other = tmp_path / "other" / "maplong_run"
    FS.place(other, ["_tmp_results_unaligned"], 10, log=lambda m: None)
    (save / "_tmp_results_aligned").unlink()
    assert FS.gc(log=lambda m: None) >= 3                 # the chunk and the owner record
    assert not FS.scratch_dir_for(save).exists() and FS.scratch_dir_for(other).exists()
    # a real directory is removed as before
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "f").write_bytes(b"12")
    assert FS.remove(tmp_path / "plain") == 2 and not (tmp_path / "plain").exists()
    assert FS.remove(tmp_path / "missing") == 0


def test_the_map_worker_wipes_and_cleans_through_it(tmp_path, scratch, monkeypatch):
    from workers import map_worker as MW
    out = tmp_path / "output"
    ml = out / "maplong_run"
    FS.place(ml, ["_tmp_results_unaligned", "_tmp_results_aligned"], 10, log=lambda m: None)
    (ml / "_tmp_results_unaligned" / "chunk_0.npy").write_bytes(b"x" * 7)
    (ml / "_tmp_results_aligned" / "chunk_0.npy").write_bytes(b"y" * 5)
    (ml / "sky_masks").mkdir()
    MW._wipe_plan_products(out, "test", log=lambda m: None)
    assert not (ml / "_tmp_results_unaligned").exists() and not (ml / "_tmp_results_aligned").exists()
    assert not FS.scratch_dir_for(ml).joinpath("_tmp_results_unaligned").exists()
    assert not FS.scratch_dir_for(ml).joinpath("_tmp_results_aligned").exists()
    assert (ml / "sky_masks").is_dir()                                    # plan-independent: kept
    src = Path(MW.__file__).read_text()
    assert "_FS.remove(_aligned)" in src and "_FS.remove(tmp_dir)" in src
    assert "_FS.place(output_dir / \"maplong_run\"" in src and "_FS.gc(log=pipe.send_log)" in src
