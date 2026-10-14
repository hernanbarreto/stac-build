"""intake.walk's I3 contract on CPU (docs/plan_determinismo.md points 4, 13, 21, 26, 41, 42, 44):
the plan is sized from the committed card table under the DA3 identity of the extracting
interpreter, the card is checked free before DA3, an OOM FAILS (never halves), the walk records
every window's content and DA3's reference views, and a REGENERATION must be of the walk's exact
plan and reproduce its windows bit for bit — or it fails naming the difference. The extractor is
a fake subprocess: it writes deterministic window files from the plan it is handed."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro                                                    # noqa: E402
from intake import vram as V                                    # noqa: E402
from intake import walk as W                                    # noqa: E402
from tests.test_intake_walk import _trajectory, _w2c            # noqa: E402

A100 = "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"
MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
IDENT = {"card": A100, "weights": {"model_id": MODEL, "revision": "b2359bdf"},
         "code": {"server/extract_da3_depth.py": "c0de"}, "torch": {"version": "2.5"},
         "autocast_dtype": "bfloat16"}


class G:
    window_frames = 32
    window_overlap_frac = 0.5
    process_res = 840
    model_id = MODEL


class FakeExtractor:
    """``subprocess.Popen`` of extract_da3_depth.py: writes window_<i>.npz for the plan in
    --windows_json (poses from a fixed trajectory, depth from the window index), or exits with
    the code ``exit_code`` without writing. ``salt`` perturbs the content (another stack)."""
    def __init__(self):
        self.exit_code = 0
        self.salt = 0.0
        self.calls = 0
        self.truth = _trajectory(150)

    def __call__(self, cmd, **kw):
        self.calls += 1
        self.kw = kw
        self.cmd = list(cmd)
        args = dict(zip(cmd, cmd[1:]))
        spec = json.loads(Path(args["--windows_json"]).read_text())
        out = Path(args["--output_dir"])
        proc = self
        if self.exit_code == 0:
            for i, win in enumerate(spec["windows"]):
                # the plan's paths are session-relative (point 70): the real extractor resolves
                # them against --windows_root; every frame named must exist there
                assert all(not Path(p).is_absolute() for p in win), win
                assert all((Path(args["--windows_root"]) / p).is_file() for p in win), win
                idx = np.array([int(Path(p).stem) for p in win])
                c2w = self.truth[idx]
                h, w = 4, 6
                np.savez(out / f"window_{i:04d}.npz", frames=idx.astype(np.int64),
                         depth=np.full((len(idx), h, w), float(i) + 1.0 + self.salt, np.float32),
                         conf=np.ones((len(idx), h, w), np.float32), extrinsics=_w2c(c2w),
                         intrinsics=np.tile(np.eye(3), (len(idx), 1, 1)),
                         scale_factor=np.float64(1.0), is_metric=np.int64(1),
                         ref_views=np.asarray([int(idx[0] % 3)], np.int64), stamp=np.asarray("fake"))

        class P:
            stdout = iter([f"[DA3 windows] fake run {proc.calls}\n"])
            returncode = proc.exit_code

            def wait(self):
                return self.returncode

            def terminate(self):
                pass
        return P()


@pytest.fixture
def session(tmp_path, monkeypatch):
    import cv2
    sess = tmp_path / "s"
    frames = sess / "frames"
    frames.mkdir(parents=True)
    files = [f"{k:06d}.jpg" for k in range(150)]
    img = np.zeros((464, 832, 3), np.uint8)
    for f in files:
        cv2.imwrite(str(frames / f), img)
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": files}))
    fake = FakeExtractor()
    monkeypatch.setattr(W.subprocess, "Popen", fake)
    monkeypatch.setattr(W, "da3_identity", lambda python, model_id, log=print: dict(IDENT))
    gpu_checks = []
    monkeypatch.setattr(repro, "require_exclusive_gpu", lambda log=print: gpu_checks.append(1) or {})
    return sess, frames, files, fake, gpu_checks


def test_the_plan_is_sized_from_the_table_under_the_da3_identity(session):
    sess, frames, files, fake, _g = session
    spec, got, fd = W.planned_spec(sess, G, "python", log=lambda m: None)
    assert got == files and fd == frames
    assert spec["window_sizing"]["window_frames"] == 26 and spec["window_sizing"]["source"] == "card_table"
    assert spec["da3_environment"] == IDENT and spec["process_res"] == 840 and spec["model_id"] == MODEL
    assert [len(w) for w in spec["windows"]] == [26] * len(spec["windows"])
    assert spec == W.planned_spec(sess, G, "python", log=lambda m: None)[0], "deterministic"
    # point 70 (USER 2026-10-08): the plan names its frames RELATIVE to the session — no
    # absolute path in windows.json, so its sha256 (the walk's key) survives a copied session
    assert spec["windows"][0][:2] == ["frames/000000.jpg", "frames/000001.jpg"]
    assert str(sess) not in json.dumps(spec)
    assert W.spec_sha256(spec) == W.spec_sha256(json.loads(json.dumps(spec)))
    # the identity (the card) decides the size: another card has no entry → FAIL naming the CLI
    import card_table
    W.da3_identity = lambda python, model_id, log=print: dict(IDENT, card="NVIDIA RTX A6000 | 49140 MiB | sm_8.6")
    with pytest.raises(card_table.CardTableError, match="--calibrate"):
        W.planned_spec(sess, G, "python", log=lambda m: None)


def test_i3_runs_on_a_free_card_measures_the_walk_and_is_reused_on_the_same_identity(session):
    sess, frames, files, fake, gpu_checks = session
    logs = []
    (sess / "output" / "intake").mkdir(parents=True)
    (sess / "output" / "intake" / "da3_vram.json").write_text("{}")        # the old cache: goes, said
    wdir, windows = W.run_da3_windows(sess, G, "python", log=logs.append, for_new_walk=True)
    assert gpu_checks == [1] and fake.calls == 1 and len(windows) == len(list(wdir.glob("window_*.npz")))
    assert not (sess / "output" / "intake" / "da3_vram.json").exists()
    assert any("da3_vram.json deleted" in m for m in logs)
    # the launcher hands the extractor the session as the root of the plan's relative paths
    assert fake.cmd[fake.cmd.index("--windows_root") + 1] == str(sess)
    assert fake.cmd[fake.cmd.index("--image_dir") + 1] == str(frames)
    spec_on_disk = json.loads((wdir / "windows.json").read_text())
    assert str(sess) not in json.dumps(spec_on_disk) and spec_on_disk["windows"][0][0] == "frames/000000.jpg"
    env = fake.kw["env"]
    assert env["HF_HOME"] == "/workspace/hf_cache" and env["HF_HUB_OFFLINE"] == "1"
    assert env[repro.CUBLAS_WORKSPACE_ENV] == repro.CUBLAS_WORKSPACE_VALUE and env["PYTHONHASHSEED"] == "0"
    walk = W.measure_walk(sess, G, log=logs.append)
    assert walk["version"] == W.WALK_VERSION and walk["da3_environment"] == IDENT
    assert walk["window_sizing"]["window_frames"] == 26 and len(walk["windows_spec_sha256"]) == 64
    assert all(len(w["content_sha256"]) == 64 and w["reference_views"] is not None and w["stamp"] == "fake"
               for w in walk["windows"])
    assert abs(walk["walk_length_m"] - 149 * 0.08) < 1e-6              # 149 steps of 0.08 m
    # the walk is reused on the same identity — BEFORE any revisit reference exists: the
    # reference is no part of the reuse (point 66, USER 2026-10-08)
    assert not (sess / "output" / W.REVISIT_REFERENCE_NAME).exists()
    assert W.walk_is_current(sess, files, frames, G, "python", log=logs.append)
    assert any("I3 reused" in m for m in logs)
    ref = W.revisit_reference(sess, log=logs.append)
    assert not any("deleted" in m and W.REVISIT_REFERENCE_NAME in m for m in logs), "nothing to delete yet"
    spec = json.loads((wdir / "windows.json").read_text())
    # for the record (never compared): what it was computed from — the plan, the windows'
    # content as the walk recorded it, the chaining code, the bootstrap and the user's factor
    assert ref["version"] == W.REVISIT_REFERENCE_VERSION and ref["windows_spec_sha256"] == W.spec_sha256(spec)
    st = ref["stamp"]
    assert st["windows_spec_sha256"] == W.spec_sha256(spec) and "server/intake/walk.py" in st["code"]
    assert st["inputs"] == {f"output/{W.WINDOWS_DIRNAME}/window_{w['index']:04d}.npz": w["content_sha256"]
                            for w in walk["windows"]}
    assert st["config"]["bar_bootstrap"]["seed"] == W.BAR_BOOTSTRAP_SEED and len(st["sha256"]) == 64
    assert not hasattr(W, "revisit_reference_is_of"), "the reuse-by-version path is gone (point 66)"
    # NEVER reused: a file on disk — whatever it says — is deleted and the reference recomputed
    # whole from the walk's windows, bit for bit the same
    tampered = dict(ref, dist_bar_m=99.0, windows_spec_sha256="0" * 64)
    (sess / "output" / W.REVISIT_REFERENCE_NAME).write_text(json.dumps(tampered))
    again = W.revisit_reference(sess, log=logs.append)
    assert again == ref and json.loads((sess / "output" / W.REVISIT_REFERENCE_NAME).read_text()) == ref
    assert any("deleted" in m and "never" in m and W.REVISIT_REFERENCE_NAME in m for m in logs)
    # another identity (other weights): the walk is not reusable, said
    W.da3_identity = lambda python, model_id, log=print: dict(IDENT, weights={"revision": "other"})
    assert not W.walk_is_current(sess, files, frames, G, "python", log=logs.append)
    assert any("I3 not reusable" in m and "weights" in m for m in logs)
    # no reuse path of the reference is left in the launcher either
    W.da3_identity = lambda python, model_id, log=print: dict(IDENT)
    mw = (Path(W.__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()
    i0 = mw.index("from intake.walk import revisit_reference")
    blk = mw[i0:mw.index("_persist_chunk_plan(_ranges", i0)]
    assert "revisit_reference(output_dir.parent" in blk and "_ref_p.exists()" not in blk
    assert "read_text()" not in blk, "the launcher never reads a revisit reference back"


def test_a_new_walk_declares_a_card_change_but_not_an_old_format_key(session):
    """A walk recorded under the pre-2026-10-07 key (torch's usable bytes) is not 'another card':
    the same A100 read 3 MiB less after a pod restart. A different MODEL key is declared as such."""
    sess, frames, files, fake, _g = session
    W.run_da3_windows(sess, G, "python", log=lambda m: None, for_new_walk=True)
    W.measure_walk(sess, G, log=lambda m: None)
    wp = sess / "intake" / W.WALK_NAME
    for old, says, never in (("NVIDIA A100 80GB PCIe | 85097971712 B | sm_8.0", "old key", "card changed"),
                             ("NVIDIA RTX A6000 | 49140 MiB | sm_8.6", "card changed", "old key")):
        walk = json.loads(wp.read_text())
        walk["da3_environment"]["card"] = old
        wp.write_text(json.dumps(walk))
        logs = []
        W.run_da3_windows(sess, G, "python", log=logs.append, for_new_walk=True)
        line = [m for m in logs if old in m]
        assert len(line) == 1 and says in line[0] and never not in line[0] and A100 in line[0], logs


def test_an_oom_fails_and_never_halves_the_window(session):
    sess, frames, files, fake, _g = session
    fake.exit_code = V.OOM_EXIT
    with pytest.raises(W.WalkError, match="nothing is halved") as e:
        W.run_da3_windows(sess, G, "python", log=lambda m: None, for_new_walk=True)
    assert "26 keyframes" in str(e.value) and A100 in str(e.value)
    spec = json.loads((sess / "output" / W.WINDOWS_DIRNAME / "windows.json").read_text())
    assert spec["window_sizing"]["window_frames"] == 26, "the plan on disk is still the table's"
    assert fake.calls == 1
    src = Path(W.__file__).read_text()
    assert "w_frames // 2" not in src and "max(2, w_frames" not in src
    fake.exit_code = V.REF_VIEW_EXIT
    with pytest.raises(W.WalkError, match="another reference view"):
        W.run_da3_windows(sess, G, "python", log=lambda m: None, for_new_walk=True)
    fake.exit_code = V.IDENTITY_EXIT
    with pytest.raises(W.WalkError, match="not the DA3 environment"):
        W.run_da3_windows(sess, G, "python", log=lambda m: None, for_new_walk=True)


def test_a_regeneration_must_be_of_the_walks_plan_and_reproduce_its_windows(session):
    sess, frames, files, fake, gpu_checks = session
    logs = []
    # no walk yet: a regeneration call IS the first I3 (python -m precision.gauge runs I3 + the
    # walk when missing) — declared, and the walk is measured on exactly those windows
    W.run_da3_windows(sess, G, "python", log=logs.append)
    assert any("a first I3, not a regeneration" in m for m in logs) and fake.calls == 1
    walk = W.measure_walk(sess, G, log=logs.append)
    assert W.run_da3_windows(sess, G, "python", log=logs.append, for_new_walk=True)[1] and fake.calls == 2
    assert json.loads((sess / "intake" / W.WALK_NAME).read_text()) == walk, "the same plan: walk.json stands"
    W.delete_windows(sess, log=logs.append)                                 # after the chain
    wdir = sess / "output" / W.WINDOWS_DIRNAME
    assert not list(wdir.glob("window_*.npz")) and (wdir / "windows.json").exists()
    # the same stack reproduces the windows: regenerated, verified bit for bit
    W.run_da3_windows(sess, G, "python", log=logs.append)
    assert gpu_checks == [1, 1, 1] and fake.calls == 3
    assert any("bit-identical" in m for m in logs)
    for w in walk["windows"]:
        assert W.window_content_sha256(wdir / f"window_{w['index']:04d}.npz") == w["content_sha256"]
    # another stack: other bytes → FAIL naming the window (no silent mix of two plans)
    W.delete_windows(sess, log=logs.append)
    fake.salt = 1e-3
    with pytest.raises(W.WalkError, match="does not reproduce") as e:
        W.run_da3_windows(sess, G, "python", log=logs.append)
    assert "window_0000.npz" in str(e.value)
    # another plan than the walk's (another identity): refused before DA3 runs
    fake.salt = 0.0
    calls = fake.calls
    W.da3_identity = lambda python, model_id, log=print: dict(IDENT, torch={"version": "2.6"})
    with pytest.raises(W.WalkError, match="cannot be regenerated") as e:
        W.run_da3_windows(sess, G, "python", log=logs.append)
    assert "torch" in str(e.value) and fake.calls == calls


def test_the_revisit_reference_needs_no_window_file(session):
    """Point 66 (USER 2026-10-08) on a RESUME: F2 deletes the window files after the chain
    (gauge.delete_windows_after_chain) and a reused walk never regenerates them, so a reference
    recomputed from the files killed every replace-OFF 'Reconstruir' of a finished session on
    window_0000.npz (the first cut of point 66; USER 2026-10-06: no 20-min DA3 regeneration on a
    resume). walk.json now RECORDS what the reference needs — every keyframe's chained pose, each
    window's median depth and FOV — and the reference reads only that: the same document, bit for
    bit, with the window files gone; a walk without the record is refused naming re-run I3, never
    served from the files."""
    sess, frames, files, fake, _g = session
    W.run_da3_windows(sess, G, "python", log=lambda m: None, for_new_walk=True)
    walk = W.measure_walk(sess, G, log=lambda m: None)
    assert walk["version"] == W.WALK_VERSION == 3
    assert all(len(e["c2w"]) == 3 and len(e["c2w"][0]) == 4 for e in walk["chainage"])
    # the fake extractor's window i has depth i + 1 everywhere and K = I on 6-px-wide maps
    assert all(w["median_depth_m"] == float(w["index"]) + 1.0 for w in walk["windows"])
    assert all(abs(w["hfov_rad"] - 2 * np.arctan(3.0)) < 1e-12 for w in walk["windows"])
    ref = W.revisit_reference(sess, log=lambda m: None)
    e0 = walk["chainage"][0]
    assert ref["frames"][0] == e0["frame"]
    assert ref["centres"][0] == [e0["c2w"][r][3] for r in range(3)]
    assert ref["forward"][0] == [e0["c2w"][r][2] for r in range(3)]
    assert ref["dist_bar_m"] == float(np.median([float(w["index"]) + 1.0 for w in walk["windows"]]))
    assert ref["stamp"]["walk_version"] == 3 and None not in ref["stamp"]["inputs"].values()
    # the window files deleted (after F2's chain): the same reference, bit for bit, no DA3
    W.delete_windows(sess, log=lambda m: None)
    wdir = sess / "output" / W.WINDOWS_DIRNAME
    assert not list(wdir.glob("window_*.npz")) and fake.calls == 1
    logs = []
    again = W.revisit_reference(sess, log=logs.append)
    assert again == ref and fake.calls == 1
    assert json.loads((sess / "output" / W.REVISIT_REFERENCE_NAME).read_text()) == ref
    assert any("deleted" in m and "never" in m for m in logs)
    src = inspect.getsource(W.revisit_reference)
    assert "load_window(" not in src and "np.load(" not in src and "windows.json" not in src
    # a walk without the record (the previous version) is refused naming re-run I3 — and the
    # stale file is already gone (deleted first), nothing on disk lies
    wp = sess / "intake" / W.WALK_NAME
    old = json.loads(wp.read_text())
    old["version"] = 2
    for e in old["chainage"]:
        e.pop("c2w")
    wp.write_text(json.dumps(old))
    with pytest.raises(W.WalkError, match="re-run I3"):
        W.revisit_reference(sess, log=lambda m: None)
    assert not (sess / "output" / W.REVISIT_REFERENCE_NAME).exists()
    assert "predates" in (W._walk_spec_check(sess, json.loads((wdir / "windows.json").read_text()))
                          or ""), "an old walk is re-measured by I3, never reused"
    wp.unlink()
    with pytest.raises(W.WalkError, match="run I3"):
        W.revisit_reference(sess, log=lambda m: None)
    # the launcher: without SALAD the previous reconstruction's reference is deleted too
    mw = (Path(W.__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()
    i0 = mw.index("from intake.walk import revisit_reference")
    blk = mw[i0:mw.index("_persist_chunk_plan(_ranges", i0)]
    assert "no SALAD in this launch" in blk and "_stale_ref.unlink()" in blk


def test_i3_reads_the_jobs_configuration_never_config_yaml_at_spawn(session):
    """Point 69: a command-line job freezes the configuration it starts with; the worker hands
    its own, which must agree with the frozen copy — a difference FAILS naming the field; the walk
    records the frozen configuration's sha256; no `from config import cfg` is left in I3's path."""
    import json as _json
    from intake.run_config import RunConfigError, load_run_config, run_config_path
    sess, frames, files, fake, _g = session
    logs = []
    assert not run_config_path(sess).exists()
    spec, _f, _d = W.planned_spec(sess, G, "python", log=logs.append)
    assert run_config_path(sess).exists() and any("[run_config] frozen" in m for m in logs)
    frozen, sha = load_run_config(sess)
    assert spec["window_sizing"]["margin_frac"] == frozen["intake"]["parallax"]["vram_margin_frac"]
    # the worker's handed configuration: the same → fine; another margin → refused, named
    assert W.planned_spec(sess, G, "python", log=logs.append, run_cfg=frozen)[0] == spec
    other = _json.loads(_json.dumps(frozen))
    other["intake"]["parallax"]["vram_margin_frac"] = 0.2
    with pytest.raises(RunConfigError, match="parallax.vram_margin_frac"):
        W.planned_spec(sess, G, "python", log=logs.append, run_cfg=other)
    with pytest.raises(RunConfigError, match="parallax.vram_margin_frac"):
        W.run_da3_windows(sess, G, "python", log=logs.append, for_new_walk=True, run_cfg=other)
    W.run_da3_windows(sess, G, "python", log=logs.append, for_new_walk=True, run_cfg=frozen)
    walk = W.measure_walk(sess, G, log=logs.append)
    W.revisit_reference(sess)
    assert walk["run_config_sha256"] == sha
    assert W.walk_is_current(sess, files, frames, G, "python", log=logs.append, run_cfg=frozen)
    with pytest.raises(RunConfigError, match="parallax.vram_margin_frac"):
        W.walk_is_current(sess, files, frames, G, "python", log=logs.append, run_cfg=other)
    src = Path(W.__file__).read_text()
    assert "from config import cfg" not in src, "I3 never re-reads config.yaml at spawn"
    # the worker hands its configuration on every I3 call
    mw = (Path(W.__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()
    blk = mw[mw.index("if walk_is_current("):mw.index("_ranges = [(int(a), int(b)) for a, b in _cplan")]
    assert blk.count("run_cfg=config") == 3
