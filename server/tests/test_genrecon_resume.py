"""tools/genrecon/resume.py — a GenRecon run whose mesh step failed resumes AT the mesh step: the reconstruction's
products are reused only under the stamp of what they are made from, and whole (pccr 2026-08-24, 2026-11-12); a run
whose DECODE failed, or whose decode changed, resumes at the decode on its sampled latents under their own stamp
(2026-11-13)."""
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
            "server/tools/genrecon/remap_gpu.py", "server/jpeg_gpu.py", "server/tools/genrecon/cloud_shape.py",
            "server/tools/genrecon/latents.py", "server/tools/genrecon/decode_regions.py"} <= rel
    # the sampled latents' stamp: the same code without the decode's own (the latents' file format stays in it)
    gen = {p.relative_to(_SERVER.parent).as_posix() for p in S.generation_code_files()}
    late = {p.relative_to(_SERVER.parent).as_posix() for p in S.decode_files()}
    assert gen < rel and rel - gen == late & rel and "server/tools/genrecon/decode_regions.py" in rel - gen
    assert "server/tools/genrecon/latents.py" in gen and "server/tools/genrecon/adapter.py" in gen
    if S.VENDOR.is_dir():
        assert "vendor/GenRecon/reconstruct_scene.py" in rel and "vendor/GenRecon/reconstruct_scene.py" in gen
        assert all((S.VENDOR / m).is_file() and f"vendor/GenRecon/{m}" not in rel for m in S.MESH_STEP_FILES)
        assert all((S.VENDOR / m).is_file() and f"vendor/GenRecon/{m}" in rel - gen for m in S.DECODE_STEP_FILES)
        assert not any("/work/" in r for r in rel)
    assert set(S.adapter_config()) == {"presence_vote", "presence_min_views", "presence_device", "adapter_device",
                                       "undistort_device", "support_surfaces"}
    assert S.adapter_config()["adapter_device"] in ("numpy", "cpu", "cuda")


def test_the_runner_checks_before_the_adapter_and_seals_after_the_reconstruction():
    sh = (_TOOLS / "run_cloud.sh").read_text()
    check, adapter = sh.index("resume.py check"), sh.index("adapter.py $ADAPT")
    recon, dump = sh.index("$GR $RECON"), sh.index("--dump_geometry_plys")
    seal_latents, seal = sh.index("resume.py seal-latents "), sh.index("resume.py seal ")
    assert check < adapter < recon < seal_latents < seal < dump
    assert sh.count("rm -rf $O $OL") == 1 and adapter < sh.index("rm -rf $O $OL") < recon   # never on a resume path
    # ONE reconstruct line for the three paths (from the adapter, at the decode, and none at the mesh step); where the
    # latents are written or read is appended to it, outside both stamped command lines
    assert sh.count("$GR $RECON") == 1 and "$GR $RECON $LAT" in sh
    assert "--save_latents $O/latents.pt" in sh and "--resume_latents $O/latents.pt" in sh
    gen = sh[sh.index('GEN="reconstruct_scene.py'):sh.index('RECON="$GEN')]
    assert "latents" not in gen and "joint_decode_max_voxels" not in gen     # the cap is the decode's alone
    assert 'RECON="$GEN --joint_decode_max_voxels $CAP"' in sh
    assert '--reconstruct "$RECON" --generation "$GEN"' in sh
    # the latents are sealed as soon as reconstruct_scene.py has exited — BEFORE its exit code stops the run
    assert seal_latents < sh.index('echo "reconstruct failed $sid"') < seal
    # the resume at the decode removes what a decode left and keeps the latents and their stamp
    wipe = next(ln for ln in sh.splitlines() if ln.strip().startswith("rm -f $O/mesh.ply"))
    assert "latents" not in wipe and "$O/reconstruct.stamp.json" in wipe and all(f"$O/{n}" in wipe or n.startswith("to_glb")
                                                                                 for n in S.PRODUCTS)
    assert "$O/to_glb_inputs*.pt" in wipe


# ── AT THE DECODE (USER 2026-11-13: "Debemos hacer entrar pccr2408 hay que ver la forma la técnica para que sea
# escalable a nubes masivas") — the sampled latents under their own stamp ──────────────────────────────────────────
GEN = RECON                                                          # the generation's command: no decode parameter
CAPPED = RECON + " --joint_decode_max_voxels 100000"                 # the reconstruction's: the generation's + the cap


@pytest.fixture
def decode_scene(scene, monkeypatch):
    """The scene with the decode's own code as a third file and the sampled latents beside the products."""
    _, _, o, code = scene
    (code / "joint_decode.py").write_text("d = 1\n")
    monkeypatch.setattr(S, "code_files", lambda: [code / "adapter.py", code / "reconstruct_scene.py",
                                                  code / "joint_decode.py"])
    monkeypatch.setattr(S, "decode_files", lambda: [code / "joint_decode.py"])
    (o / S.LATENTS_NAME).write_bytes(b"l" * 40)
    return scene


def _cli(scene, recon=CAPPED, gen=GEN, adapt=ADAPT, dry=False):
    ses, work, o, _ = scene
    return S.main(["check", str(ses), str(work), str(o), "--conf", "0.0", "--adapter", adapt, "--reconstruct", recon,
                   "--generation", gen, *(["--dry"] if dry else [])])


def _whole_run(scene):
    """A launch from the adapter that ended whole: both stamps sealed."""
    _, work, o, _ = scene
    assert _cli(scene) == S.NOT_REUSABLE
    S.seal_latents(work, o)
    S.seal(work, o)
    assert _cli(scene) == 0


def test_a_failed_decode_continues_at_the_decode(decode_scene, capsys):
    """reconstruct_scene.py wrote its latents and died in the decode: the next launch reuses them (exit 4); once the
    decode has succeeded and the products are sealed, the launch after that continues at the mesh step."""
    ses, work, o, _ = decode_scene
    args = [str(ses), str(work), str(o)]
    for n in S.PRODUCTS:                                             # the decode never wrote them
        (o / n).unlink()
    assert _cli(decode_scene) == S.NOT_REUSABLE
    said = capsys.readouterr().out
    assert "starts at the adapter" in said and "nor are the sampled latents" in said
    assert "no latents.stamp.json beside the sampled latents" in said
    pend, lat_pend = S.pending_path(o), S.pending_path(o, S.LATENTS_STAMP_NAME)
    assert pend.is_file() and lat_pend.is_file() and pend != lat_pend and lat_pend.parent == o.parent
    assert S.main(["seal-latents", *args]) == 0                      # reconstruct_scene.py has exited, its decode failed
    assert (o / S.LATENTS_STAMP_NAME).is_file() and not lat_pend.exists() and pend.is_file()
    with pytest.raises(RuntimeError, match="out/to_glb_inputs.pt"):  # nothing decoded: no products' stamp
        S.seal(work, o)
    assert _cli(decode_scene) == S.AT_THE_DECODE == 4
    said = capsys.readouterr().out
    assert "continues at the decode" in said and "no reconstruct.stamp.json" in said
    assert pend.is_file() and not lat_pend.exists()                  # the products' stamp waits for the decode
    for n in S.PRODUCTS:                                             # the decode, whole this time
        (o / n).write_bytes(b"p" * 20)
    assert S.main(["seal", *args]) == 0
    assert _cli(decode_scene) == 0
    assert "continues at the mesh step" in capsys.readouterr().out
    assert not pend.exists() and not lat_pend.exists()


def test_only_the_decodes_own_changes_keep_the_latents(decode_scene, capsys):
    """After a whole run: a change of the decode's code or of its cap re-runs the decode alone; a change of anything
    the latents are made from is named for both and the run starts at the adapter."""
    ses, _, _, code = decode_scene
    _whole_run(decode_scene)
    capsys.readouterr()
    (code / "joint_decode.py").write_text("d = 2\n")                 # a fix in the decode
    assert _cli(decode_scene) == S.AT_THE_DECODE
    said = capsys.readouterr().out
    assert "code 'code/joint_decode.py' changed" in said and "continues at the decode" in said
    (code / "joint_decode.py").write_text("d = 1\n")
    assert _cli(decode_scene, recon=RECON + " --joint_decode_max_voxels 60000") == S.AT_THE_DECODE     # another cap
    assert "config 'order' changed" in capsys.readouterr().out
    assert _cli(decode_scene) == 0
    capsys.readouterr()
    (code / "adapter.py").write_text("a = 2\n")
    assert _cli(decode_scene) == S.NOT_REUSABLE
    said = capsys.readouterr().out
    assert "nor are the sampled latents" in said and said.count("code 'code/adapter.py' changed") == 2
    (code / "adapter.py").write_text("a = 1\n")
    more = " --coords_min_points 1"                                  # a parameter of the generation
    assert _cli(decode_scene, recon=CAPPED + more, gen=GEN + more) == S.NOT_REUSABLE
    assert capsys.readouterr().out.count("config 'order' changed") == 2
    assert _cli(decode_scene, adapt=ADAPT.replace("3.7", "2.5")) == S.NOT_REUSABLE
    assert capsys.readouterr().out.count("config 'order' changed") == 2
    (ses / "output" / "cleaned_cloud.ply").write_bytes(b"ply cloud, cleaned again")
    assert _cli(decode_scene) == S.NOT_REUSABLE
    assert capsys.readouterr().out.count("input 'output/cleaned_cloud.ply' changed") == 2


def test_latents_that_are_not_whole_are_named(decode_scene, capsys):
    _, work, o, code = decode_scene
    _whole_run(decode_scene)
    (code / "joint_decode.py").write_text("d = 2\n")                 # the products are out: the latents decide
    (o / S.LATENTS_NAME).write_bytes(b"cut")
    capsys.readouterr()
    assert _cli(decode_scene) == S.NOT_REUSABLE
    assert "product 'out/latents.pt' was 40, is 3" in capsys.readouterr().out
    (o / S.LATENTS_NAME).unlink()
    assert _cli(decode_scene) == S.NOT_REUSABLE
    assert "product 'out/latents.pt' was 40, is missing" in capsys.readouterr().out
    (o / S.LATENTS_NAME).write_bytes(b"l" * 40)
    assert _cli(decode_scene) == S.AT_THE_DECODE
    (work / "cloud_genrecon.npy").write_bytes(b"c" * 31)             # the work folder the latents were sampled from
    capsys.readouterr()
    assert _cli(decode_scene) == S.NOT_REUSABLE
    assert capsys.readouterr().out.count("product 'work/cloud_genrecon.npy' was 30, is 31") == 2


def test_the_latents_stamp_is_the_one_computed_before_the_run(decode_scene, capsys):
    ses, work, o, _ = decode_scene
    assert _cli(decode_scene) == S.NOT_REUSABLE                      # before the adapter
    (ses / "output" / "camera_poses.txt").write_text("0 1 0 0\n")    # during the run
    S.seal_latents(work, o)
    capsys.readouterr()
    assert _cli(decode_scene) == S.NOT_REUSABLE
    said = capsys.readouterr().out
    assert "nor are the sampled latents" in said and "input 'output/camera_poses.txt' changed" in said


def test_seal_latents_refuses_without_a_pending_stamp_or_the_latents(decode_scene):
    _, work, o, _ = decode_scene
    with pytest.raises(RuntimeError, match="after the fact"):
        S.seal_latents(work, o)
    assert _cli(decode_scene) == S.NOT_REUSABLE
    (o / S.LATENTS_NAME).unlink()
    with pytest.raises(RuntimeError, match="out/latents.pt"):
        S.seal_latents(work, o)
    assert not (o / S.LATENTS_STAMP_NAME).exists() and S.pending_path(o, S.LATENTS_STAMP_NAME).is_file()


def test_a_check_clears_the_generation_stamp_an_earlier_launch_left_pending(decode_scene):
    """A launch that died before reconstruct_scene.py wrote its latents leaves the generation's stamp pending; the
    next launch's check removes it before computing its own — `seal-latents` never stamps with another order's."""
    ses, work, o, _ = decode_scene
    assert _cli(decode_scene) == S.NOT_REUSABLE
    lat_pend = S.pending_path(o, S.LATENTS_STAMP_NAME)
    assert lat_pend.is_file()
    S.check(ses, work, o, 0.0, ADAPT, CAPPED)                        # the products' check alone
    assert not lat_pend.exists()
    with pytest.raises(RuntimeError, match="after the fact"):
        S.seal_latents(work, o)


def test_a_dry_check_writes_and_removes_nothing(decode_scene):
    """Asking beside a live run of the same session: no pending stamp is created, and the live run's stay byte for
    byte whatever the answer (from the adapter, at the decode, at the mesh step)."""
    _, work, o, code = decode_scene
    pend, lat_pend = S.pending_path(o), S.pending_path(o, S.LATENTS_STAMP_NAME)
    assert _cli(decode_scene, dry=True) == S.NOT_REUSABLE and not pend.exists() and not lat_pend.exists()
    _whole_run(decode_scene)
    pend.write_text("the live run's"), lat_pend.write_text("the live run's too")
    assert _cli(decode_scene, dry=True) == 0
    (code / "joint_decode.py").write_text("d = 2\n")
    assert _cli(decode_scene, dry=True) == S.AT_THE_DECODE
    (code / "adapter.py").write_text("a = 2\n")
    assert _cli(decode_scene, dry=True) == S.NOT_REUSABLE
    assert pend.read_text() == "the live run's" and lat_pend.read_text() == "the live run's too"
