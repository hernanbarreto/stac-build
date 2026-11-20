"""The server's side of the cube run's single closing pass (USER 2026-11-19: "Todo debe hacerse en placa … es tiempo
perdido" / "Dije que si al diseño, dale"): the four keys typed and written by the builder, the cube run's own
per-keyframe records taken by a rename, the point files and origins given their output names without a copy."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers import map_worker as MW  # noqa: E402

BASE = {"enabled": True, "chunk_view_cap": 64, "probe": "amb3r", "amb3r_model": "omega", "cube_m": 5.0}


class _Pipe:
    def __init__(self):
        self.lines = []

    def send_log(self, m, level=None):
        self.lines.append(str(m))


def test_the_keys_are_typed_and_the_record_is_what_an_absent_key_means():
    st = MW._structure_config({"structure": dict(BASE)})
    assert st["cube_closing"] == "steps"
    assert [st[k] for k in MW.CUBE_DEVICE_KEYS] == ["numpy", "numpy", "numpy"]
    st = MW._structure_config({"structure": dict(BASE, cube_closing="single_pass", cube_closing_device="cuda",
                                                 cube_seams_device="cpu", cube_graph_device="cuda")})
    assert (st["cube_closing"], st["cube_closing_device"], st["cube_seams_device"], st["cube_graph_device"]) == (
        "single_pass", "cuda", "cpu", "cuda")
    for key, bad in (("cube_closing", "one_pass"), ("cube_closing", True), ("cube_closing_device", "gpu"),
                     ("cube_seams_device", 0), ("cube_graph_device", "CUDA")):
        with pytest.raises(RuntimeError, match=f"structure.{key}"):
            MW._structure_config({"structure": dict(BASE, **{key: bad})})


def test_config_yaml_declares_the_four_keys():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    raw = cfg["reconstruction"]["vggtomega"]["structure"]
    assert all(k in raw for k in ("cube_closing",) + MW.CUBE_DEVICE_KEYS), "declared, not left to a default"
    st = MW._structure_config(cfg["reconstruction"]["vggtomega"])
    assert st["cube_closing"] in MW.CUBE_CLOSINGS and all(st[k] in MW.CUBE_DEVICES for k in MW.CUBE_DEVICE_KEYS)


def test_the_builder_writes_the_keys_and_refuses_a_single_pass_that_must_keep_the_placed_cubes():
    st = MW._structure_config({"structure": dict(BASE, cube_closing="single_pass", cube_closing_device="cuda")})
    cfg = MW._apply_cube_closing({"Model": {"x": 1}}, st, keep_aligned=False)
    assert cfg["Model"] == {"x": 1, "cube_closing": "single_pass", "cube_closing_device": "cuda",
                            "cube_seams_device": "numpy", "cube_graph_device": "numpy"}
    with pytest.raises(RuntimeError, match="cube_closing.*keep_aligned_chunks"):
        MW._apply_cube_closing({"Model": {}}, st, keep_aligned=True)
    rec = MW._structure_config({"structure": dict(BASE)})
    assert MW._apply_cube_closing({}, rec, keep_aligned=True)["Model"]["cube_closing"] == "steps"
    src = Path(MW.__file__).read_text()
    assert src.count("_apply_cube_closing(vggt_config, _st_cfg,") == 1, "written where the cube's other keys are"


def _records(tmp_path, n=5, digest="d" * 64):
    save, out = tmp_path / "maplong_run", tmp_path / "output"
    (save / MW.OMEGA_RECORDS_DIR).mkdir(parents=True)
    stems = [g * 7 for g in range(n)]
    rng = np.random.default_rng(3)
    recs = {}
    for g, num in enumerate(stems):
        recs[num] = {"depth": rng.random((6, 8)).astype(np.float32), "pose_c2w": rng.normal(size=(4, 4)),
                     "chunk": np.int64(g // 3), "frame_global": np.int64(g), "conf": rng.random((6, 8)).astype(np.float32)}
        np.savez_compressed(save / MW.OMEGA_RECORDS_DIR / f"frame_{num}.npz", **recs[num])
    (save / MW.OMEGA_RECORDS_DOC).write_text(json.dumps({"layout": "cubes", "layout_sha256": digest, "n_frames": n,
                                                         "frames": stems, "dir": MW.OMEGA_RECORDS_DIR}))
    (save / "chunk_sim3.json").write_text(json.dumps({"layout": "cubes", "layout_sha256": digest}))
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"selected_files": [f"{num:06d}.jpg" for num in stems]}))
    cubes = ([list(range(0, 3)), list(range(2, n))], [0, 0, 0] + [1] * (n - 3))
    return save, out, sel, cubes, stems, recs


def test_the_cube_runs_own_records_are_taken_by_a_rename(tmp_path):
    save, out, sel, cubes, stems, recs = _records(tmp_path)
    inodes = {num: os.stat(save / MW.OMEGA_RECORDS_DIR / f"frame_{num}.npz").st_ino for num in stems}
    pipe = _Pipe()
    MW._emit_omega_depth(save, out, None, str(sel), pipe, cubes=cubes)       # (no placed cube, no pose file: not read)
    dst = out / "omega_run" / "results_output"
    for num in stems:
        assert os.stat(dst / f"frame_{num}.npz").st_ino == inodes[num], "the same file, not a copy"
        with np.load(dst / f"frame_{num}.npz") as z:
            assert sorted(z.files) == sorted(recs[num]) and all(np.array_equal(z[k], recs[num][k]) for k in z.files)
    assert not (save / MW.OMEGA_RECORDS_DIR).exists() and any("taken from the cube run" in ln for ln in pipe.lines)
    MW._emit_omega_depth(save, out, None, str(sel), _Pipe(), cubes=cubes)    # again (a resume): nothing to take, nothing lost
    assert sorted(p.name for p in dst.glob("frame_*.npz")) == sorted(f"frame_{num}.npz" for num in stems)


def test_records_of_another_layout_other_keyframes_or_with_a_hole_are_refused(tmp_path):
    save, out, sel, cubes, stems, _ = _records(tmp_path / "a")
    (save / "chunk_sim3.json").write_text(json.dumps({"layout_sha256": "e" * 64}))
    with pytest.raises(RuntimeError, match="another run"):
        MW._emit_omega_depth(save, out, None, str(sel), _Pipe(), cubes=cubes)
    save, out, sel, cubes, stems, _ = _records(tmp_path / "b")
    doc = json.loads((save / MW.OMEGA_RECORDS_DOC).read_text())
    (save / MW.OMEGA_RECORDS_DOC).write_text(json.dumps(dict(doc, frames=doc["frames"][::-1])))
    with pytest.raises(RuntimeError, match="not this run's"):
        MW._emit_omega_depth(save, out, None, str(sel), _Pipe(), cubes=cubes)
    save, out, sel, cubes, stems, _ = _records(tmp_path / "c")
    (save / MW.OMEGA_RECORDS_DIR / f"frame_{stems[2]}.npz").unlink()
    with pytest.raises(RuntimeError, match=f"frame {stems[2]} is neither"):
        MW._emit_omega_depth(save, out, None, str(sel), _Pipe(), cubes=cubes)


def test_without_the_cube_runs_records_the_placed_cubes_are_read_as_before(tmp_path):
    """No omega_records.json (Model.cube_closing: steps): the function is the one it was — it asks for the aligned
    poses and reads _tmp_results_aligned."""
    save, out, sel, cubes, stems, _ = _records(tmp_path)
    (save / MW.OMEGA_RECORDS_DOC).unlink()
    with pytest.raises(RuntimeError, match="no aligned camera_poses.txt"):
        MW._emit_omega_depth(save, out, None, str(sel), _Pipe(), cubes=cubes)


def test_a_second_name_instead_of_a_copy_and_a_copy_where_linking_fails(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.ply", tmp_path / "out" / "chunk_000.ply"
    dst.parent.mkdir()
    src.write_bytes(b"points" * 1000)
    dst.write_bytes(b"an older file")
    assert MW._link_or_copy(src, dst) == "link"
    assert os.stat(dst).st_ino == os.stat(src).st_ino and os.stat(dst).st_nlink == 2
    src.unlink()                                             # (the caller's cleanup of the fork's folder)
    assert dst.read_bytes() == b"points" * 1000 and os.stat(dst).st_nlink == 1
    src.write_bytes(b"other" * 10)

    def no_link(a, b):
        raise OSError(18, "Invalid cross-device link")
    monkeypatch.setattr(os, "link", no_link)
    assert MW._link_or_copy(src, dst) == "copy"
    assert dst.read_bytes() == b"other" * 10 and os.stat(dst).st_ino != os.stat(src).st_ino


def test_the_post_process_links_and_no_longer_recompresses_the_origins():
    src = Path(MW.__file__).read_text()
    body = src[src.index("def _postprocess_reconstruction"):src.index("def _build_da3_config")]
    assert "_link_or_copy(Path(ply_src), ply_dst)" in body and "shutil.copyfile(ply_src, ply_dst)" not in body
    org = src[src.index("def _generate_origins"):]
    inline = org[org.index("inline = pcd_dir"):org.index("npy_path = chunks_dir")]
    assert "_link_or_copy(inline," in inline and "np.savez_compressed" not in inline
