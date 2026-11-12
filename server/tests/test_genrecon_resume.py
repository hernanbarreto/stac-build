"""tools/genrecon/resume.py — a GenRecon run whose mesh step failed resumes AT the mesh step: the reconstruction's
products are reused only under the stamp of what they are made from, and whole (pccr 2026-08-24, 2026-11-12)."""
import importlib.util
from pathlib import Path

import pytest

_SERVER = Path(__file__).resolve().parent.parent
_TOOLS = _SERVER / "tools" / "genrecon"
_spec = importlib.util.spec_from_file_location("genrecon_resume", _TOOLS / "resume.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

ADAPT = "SES W 600000 3.7 0.0 10"
RECON = "reconstruct_scene.py --mode Iphone --coords_mode cloud"


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """A session, its work folder and the reconstruction's products; the step's code as two files of the test's."""
    ses = tmp_path / "projects" / "s" / "scans" / "d" / "src_default"
    out = ses / "output"
    out.mkdir(parents=True)
    (ses / "frames").mkdir()
    (out / "cleaned_cloud.ply").write_bytes(b"ply cloud")
    (out / "camera_poses.txt").write_text("1 0 0 0\n")
    (out / "camera.json").write_text("{}")
    (out / "camera_frames.txt").write_text("0\n6\n")
    (out / "segmentation_result.json").write_text('{"instances": []}')
    for fr in (0, 6):
        (ses / "frames" / f"{fr:06d}.jpg").write_bytes(b"j" * (10 + fr))
    work = tmp_path / "work" / "s"
    (work / "rgb").mkdir(parents=True)
    (work / "cloud_genrecon.npy").write_bytes(b"c" * 30)
    (work / "rgb" / "000000.jpg").symlink_to(ses / "frames" / "000000.jpg")
    o = tmp_path / "work" / "s_cloud_out"
    o.mkdir()
    for n in S.PRODUCTS:
        (o / n).write_bytes(b"p" * 20)
    code = tmp_path / "code"
    code.mkdir()
    (code / "adapter.py").write_text("a = 1\n")
    (code / "reconstruct_scene.py").write_text("r = 1\n")
    monkeypatch.setattr(S.repro, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(S, "code_files", lambda: [code / "adapter.py", code / "reconstruct_scene.py"])
    monkeypatch.setattr(S, "adapter_config", lambda: {"presence_vote": False})
    monkeypatch.setattr(S, "checkpoints", lambda: {"checkpoints/a.pt": 5})
    return ses, work, o, code


def _check(scene, adapt=ADAPT, recon=RECON, conf=0.0):
    ses, work, o, _ = scene
    return S.check(ses, work, o, conf, adapt, recon)


def _sealed(scene):
    _, work, o, _ = scene
    assert len(_check(scene)) == 1
    S.seal(work, o)
    assert _check(scene) == []


def test_no_stamp_then_sealed_then_reusable(scene):
    _, work, o, _ = scene
    d = _check(scene)
    assert len(d) == 1 and "no reconstruct.stamp.json" in d[0]
    assert S.pending_path(o).is_file() and S.pending_path(o).parent == o.parent      # survives the folder's wipe
    assert S.seal(work, o) == o / S.STAMP_NAME and not S.pending_path(o).exists()
    assert _check(scene) == []


def test_a_changed_input_code_or_order_is_named(scene):
    ses, _, _, code = scene
    _sealed(scene)
    cloud = ses / "output" / "cleaned_cloud.ply"
    cloud.write_bytes(b"ply cloud, cleaned again")
    assert any("input 'output/cleaned_cloud.ply' changed" in d for d in _check(scene))
    cloud.write_bytes(b"ply cloud")
    assert _check(scene) == []
    (ses / "output" / "floor_transform.npz").write_bytes(b"floor")                    # the adapter would read it now
    assert any("input 'output/floor_transform.npz' is new" in d for d in _check(scene))
    (ses / "output" / "floor_transform.npz").unlink()
    (code / "adapter.py").write_text("a = 2\n")
    assert any("code 'code/adapter.py' changed" in d for d in _check(scene))
    (code / "adapter.py").write_text("a = 1\n")
    assert any("config 'order' changed" in d for d in _check(scene, adapt=ADAPT.replace("3.7", "2.85")))
    assert any("config 'order' changed" in d for d in _check(scene, recon=RECON + " --coords_min_points 1"))
    (ses / "frames" / "000006.jpg").write_bytes(b"j")
    assert any("config 'keyframes' changed" in d for d in _check(scene))


def test_the_potree_metadata_counts_only_under_a_confidence(scene):
    ses, work, o, _ = scene
    (ses / "output" / "potree").mkdir()
    (ses / "output" / "potree" / "metadata.json").write_text("{}")
    assert "output/potree/metadata.json" not in S.session_inputs(ses, 0.0)
    assert "output/potree/metadata.json" in S.session_inputs(ses, 0.3)


def test_products_that_are_not_whole_are_named(scene):
    _, work, o, _ = scene
    _sealed(scene)
    (o / "to_glb_inputs.pt").write_bytes(b"cut")
    assert any("product 'out/to_glb_inputs.pt' was 20, is 3" in d for d in _check(scene))
    (o / "to_glb_inputs.pt").unlink()
    assert any("product 'out/to_glb_inputs.pt' was 20, is missing" in d for d in _check(scene))
    (o / "to_glb_inputs.pt").write_bytes(b"p" * 20)
    assert _check(scene) == []
    (work / "cloud_genrecon.npy").unlink()
    assert any("product 'work/cloud_genrecon.npy' was 30, is absent" in d for d in _check(scene))
    (work / "cloud_genrecon.npy").write_bytes(b"c" * 30)
    (work / "rgb" / "000000.jpg").unlink()
    (work / "rgb" / "000000.jpg").symlink_to(work / "cloud_genrecon.npy")
    assert any("product 'work/rgb/000000.jpg'" in d for d in _check(scene))


def test_the_stamp_is_the_one_computed_before_the_run(scene):
    """An input that changes while the reconstruction runs: the products carry the stamp of what they were made
    from, so the next launch does not reuse them for the changed input."""
    ses, work, o, _ = scene
    _check(scene)                                                    # before the adapter
    (ses / "output" / "camera_poses.txt").write_text("0 1 0 0\n")    # during the run
    S.seal(work, o)
    assert any("input 'output/camera_poses.txt' changed" in d for d in _check(scene))


def test_seal_refuses_without_a_pending_stamp_or_a_product(scene):
    _, work, o, _ = scene
    with pytest.raises(RuntimeError, match="after the fact"):
        S.seal(work, o)
    _check(scene)
    (o / "chunk_inputs.pt").unlink()
    with pytest.raises(RuntimeError, match="out/chunk_inputs.pt"):
        S.seal(work, o)
    assert not (o / S.STAMP_NAME).exists()


def test_the_command_line(scene, capsys):
    ses, work, o, _ = scene
    args = [str(ses), str(work), str(o)]
    order = ["--conf", "0.0", "--adapter", ADAPT, "--reconstruct", RECON]
    assert S.main(["check", *args, *order]) == S.NOT_REUSABLE == 3
    assert "not reusable" in capsys.readouterr().out
    assert S.main(["seal", *args]) == 0
    assert S.main(["check", *args, *order]) == 0
    assert "continues at the mesh step" in capsys.readouterr().out and not S.pending_path(o).exists()
    with pytest.raises(SystemExit):
        S.main(["check", *args])                                      # the order is part of the stamp


def test_vendor_sources_leave_the_mesh_step_out(tmp_path):
    v = tmp_path / "GenRecon"
    for rel in ("reconstruct_scene.py", "chunked_to_glb.py", "inference/chunked_glb.py", "inference/get_chunks.py",
                "configs/a.json", "work/s/args.json", "checkpoints/c.json", "genrecon/__pycache__/x.py", "README.md",
                "genrecon/work/kept.py"):
        (v / rel).parent.mkdir(parents=True, exist_ok=True)
        (v / rel).write_text("x")
    got = sorted(p.relative_to(v).as_posix() for p in S.vendor_sources(v))
    assert got == ["configs/a.json", "genrecon/work/kept.py", "inference/get_chunks.py", "reconstruct_scene.py"]


def test_the_stamp_holds_the_device_genrecons_process_undistorts_on(monkeypatch):
    monkeypatch.delenv(S.UNDISTORT_ENV, raising=False)
    assert S.adapter_config()["undistort_device"] == "numpy"          # the runner by hand: the vendor's own lines
    monkeypatch.setenv(S.UNDISTORT_ENV, "cuda")
    assert S.adapter_config()["undistort_device"] == "cuda"


def test_the_real_lists():
    """What the stamp covers in this repo: the adapter's files, and — when the vendor clone is here — its
    reconstruction sources without the mesh step's two files."""
    files = S.code_files()
    assert all(p.is_file() for p in files)
    rel = {p.relative_to(_SERVER.parent).as_posix() for p in files}
    assert {"server/tools/genrecon/adapter.py", "server/tools/genrecon/presence.py",
            "server/tools/genrecon/genrecon_levels.py", "server/tools/genrecon/layout_gpu.py",
            "server/tools/genrecon/remap_gpu.py", "server/jpeg_gpu.py", "server/tools/genrecon/cloud_shape.py"} <= rel
    if S.VENDOR.is_dir():
        assert "vendor/GenRecon/reconstruct_scene.py" in rel
        assert all((S.VENDOR / m).is_file() and f"vendor/GenRecon/{m}" not in rel for m in S.MESH_STEP_FILES)
        assert not any("/work/" in r for r in rel)
    assert set(S.adapter_config()) == {"presence_vote", "presence_min_views", "presence_device", "adapter_device",
                                       "undistort_device", "support_surfaces"}
    assert S.adapter_config()["adapter_device"] in ("numpy", "cpu", "cuda")


def test_the_runner_checks_before_the_adapter_and_seals_after_the_reconstruction():
    sh = (_TOOLS / "run_cloud.sh").read_text()
    check, adapter = sh.index("resume.py check"), sh.index("adapter.py $ADAPT")
    recon, seal, dump = sh.index("$GR $RECON"), sh.index("resume.py seal"), sh.index("--dump_geometry_plys")
    assert check < adapter < recon < seal < dump
    assert sh.count("rm -rf $O $OL") == 1 and adapter < sh.index("rm -rf $O $OL") < recon   # never on the resume path
