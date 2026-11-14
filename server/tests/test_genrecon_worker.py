"""workers/genrecon_worker.py — the stage runs BOTH variants (USER 2026-11-02): the mesh from the cloud with
the real textures, then the generative one with its own texture; a failure names its variant."""
import json
from pathlib import Path

import pytest

import workers.genrecon_worker as G


class _Pipe:
    def __init__(self):
        self.progress, self.logs = [], []

    def send_progress(self, pct, msg, stage=""):
        self.progress.append((pct, msg))

    def send_log(self, msg, level="info"):
        self.logs.append(msg)

    def check_cancel(self):
        return False


def _session(tmp_path):
    ses = tmp_path / "projects" / "station" / "scans" / "2026-10-17" / "src_default"
    (ses / "output").mkdir(parents=True)
    (ses / "output" / "cleaned_cloud.ply").write_bytes(b"ply")
    return ses


envs: list = []


def _fake_popen(calls, fail_mode=None):
    class _P:
        def __init__(self, argv, **kw):
            calls.append(argv)
            envs.append(kw["env"])
            _, _, spec, name, minpts, mode = argv
            self._code = 1 if mode == fail_mode else 0
            out = []
            if self._code == 0:
                d = Path(kw["env"]["_TEST_OUTPUT"]) / "tsdf" / name
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{name}.glb").write_bytes(b"glb")
                (d / f"{name}.meta.json").write_text(json.dumps({"n_faces": 10}))
                out.append(f"published {d / name}.glb (1 MB, 10 faces)\n")
            else:
                out.append("reconstruct failed station\n")
            self.stdout = iter(out)

        def wait(self):
            return self._code
    return _P


def _no_vllm_stop(monkeypatch, calls):
    """The stage frees the card before generating — recorded here, never the real service."""
    monkeypatch.setattr(G, "stop_semantic_service_verified", lambda pipe, stage="": calls.append(["stop", stage]))


def _run(tmp_path, monkeypatch, gcfg, fail_mode=None):
    ses = _session(tmp_path)
    calls = []
    _no_vllm_stop(monkeypatch, calls)
    monkeypatch.setattr(G.subprocess, "Popen", _fake_popen(calls, fail_mode))
    monkeypatch.setenv("_TEST_OUTPUT", str(ses / "output"))
    pipe = _Pipe()
    G._genrecon_work(pipe, str(ses), {"genrecon": gcfg})
    return calls, pipe


_CFG = {"mesh_name": "genrecon_photo", "mode": "cloud", "generative_mesh_name": "genrecon_generative",
        "min_points_per_voxel": 10, "simplify_faces_per_chunk": 150000, "texture_size": 2048, "bake_views": 3,
        "cube_size_m": 3.7, "undistort_device": "cuda", "solid_ground": False, "shape_from_cloud": True,
        "shape_refine_t": 0.0, "smooth_iters": 0, "edge_filter": True, "denoise_normal_iters": 20,
        "denoise_vertex_iters": 10, "denoise_sigma_r": 0.35, "denoise_block_faces": 2000000,
        "joint_decode_max_voxels": 100000, "keep_work": False}


def test_both_variants_in_order(tmp_path, monkeypatch):
    calls, pipe = _run(tmp_path, monkeypatch, _CFG)
    assert calls[0] == ["stop", "GenRecon mesh"]                  # the card freed first
    calls = calls[1:]
    assert [c[3:] for c in calls] == [["genrecon_photo", "10", "cloud"], ["genrecon_generative", "10", "generative"]]
    assert all(c[2] == "station:2026-10-17" for c in calls)
    pcts = [p for p, _ in pipe.progress]
    assert pcts == sorted(pcts) and pcts[-1] == 100
    assert "genrecon_photo (10 faces), genrecon_generative (10 faces)" in pipe.progress[-1][1]


def test_an_empty_generative_name_runs_only_the_cloud_mesh(tmp_path, monkeypatch):
    calls, _ = _run(tmp_path, monkeypatch, {**_CFG, "generative_mesh_name": ""})
    assert [c[3] for c in calls[1:]] == ["genrecon_photo"]


def test_the_same_name_twice_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="overwrite"):
        _run(tmp_path, monkeypatch, {**_CFG, "generative_mesh_name": "genrecon_photo"})


def test_a_failed_first_variant_stops_and_names_itself(tmp_path, monkeypatch):
    ses = _session(tmp_path)
    calls = []
    _no_vllm_stop(monkeypatch, calls)
    monkeypatch.setattr(G.subprocess, "Popen", _fake_popen(calls, fail_mode="cloud"))
    monkeypatch.setenv("_TEST_OUTPUT", str(ses / "output"))
    with pytest.raises(RuntimeError, match="1/2 from the cloud"):
        G._genrecon_work(_Pipe(), str(ses), {"genrecon": _CFG})
    assert len(calls) == 2 and calls[0][0] == "stop"


def test_the_cube_side_reaches_the_runner(tmp_path, monkeypatch):
    envs.clear()
    _run(tmp_path, monkeypatch, {**_CFG, "generative_mesh_name": ""})
    assert envs and envs[0]["GENRECON_CUBE_M"] == "3.7"


def test_the_undistortion_device_reaches_the_runner_and_is_typed(tmp_path, monkeypatch):
    for n, dev in enumerate(("numpy", "cpu", "cuda")):
        envs.clear()
        _run(tmp_path / str(n), monkeypatch, {**_CFG, "generative_mesh_name": "", "undistort_device": dev})
        assert envs and envs[0][G.UNDISTORT_ENV] == dev
    envs.clear()
    missing = {k: v for k, v in _CFG.items() if k != "undistort_device"}
    with pytest.raises(RuntimeError, match="genrecon.undistort_device is missing"):
        _run(tmp_path / "missing", monkeypatch, missing)
    for n, bad in enumerate(("gpu", "cuda:0", "", None, True)):
        with pytest.raises(RuntimeError, match="genrecon.undistort_device must be one of"):
            _run(tmp_path / f"bad{n}", monkeypatch, {**_CFG, "undistort_device": bad})
    assert not envs                                               # nothing was launched


def test_the_tiles_switch_reaches_the_runner(tmp_path, monkeypatch):
    envs.clear()
    _run(tmp_path / "on", monkeypatch, {**_CFG, "generative_mesh_name": "", "tiles": True})
    assert envs and envs[0]["GENRECON_TILES"] == "1"
    envs.clear()
    _run(tmp_path / "off", monkeypatch, {**_CFG, "generative_mesh_name": "", "tiles": False})
    assert envs and envs[0]["GENRECON_TILES"] == "0"


def test_every_failure_line_of_the_runner_is_named():
    """run_cloud.sh ends a failed step with `echo "<step> failed <sid>"`: the worker knows every one of them (pccr
    2026-08-24, 2026-11-12: the mesh step's "dump failed" was not in the list and the stage said only "exit 1")."""
    import re
    script = (Path(G.__file__).resolve().parent.parent / "tools" / "genrecon" / "run_cloud.sh").read_text()
    said = set(re.findall(r'echo "([a-z ]+ failed) \$sid"', script))
    assert said and said == set(G._FAILS)


def test_the_shape_from_the_cloud_keys_reach_the_runner_typed(tmp_path, monkeypatch):
    """USER 2026-11-12: the shape from the cloud, its refinement, the bake's smoothing and no solid ground reach
    run_cloud.sh as GENRECON_*; a missing key or a wrong type fails naming it, before anything is launched."""
    envs.clear()
    _run(tmp_path / "ok", monkeypatch, {**_CFG, "generative_mesh_name": "", "shape_refine_t": 0.3})
    e = envs[0]
    assert (e["GENRECON_SHAPE_FROM_CLOUD"], e["GENRECON_SHAPE_REFINE_T"], e["GENRECON_SMOOTH_ITERS"],
            e["GENRECON_SOLID_GROUND"]) == ("1", "0.3", "0", "0")
    # the edge filter (USER 2026-11-12 night: "Implementamos el filtro y hacemos las mallas de todas las sesiones con el
    # filtro"): its switch, its three values and its memory block
    assert (e["GENRECON_EDGE_FILTER"], e["GENRECON_DENOISE_NORMAL_ITERS"], e["GENRECON_DENOISE_VERTEX_ITERS"],
            e["GENRECON_DENOISE_SIGMA_R"], e["GENRECON_DENOISE_BLOCK_FACES"]) == ("1", "20", "10", "0.35", "2000000")
    envs.clear()
    _run(tmp_path / "off", monkeypatch, {**_CFG, "generative_mesh_name": "", "edge_filter": False})
    assert envs[0]["GENRECON_EDGE_FILTER"] == "0"
    envs.clear()
    for n, (key, bad, msg) in enumerate((("shape_from_cloud", None, "genrecon.shape_from_cloud is missing"),
                                         ("shape_from_cloud", 1, "genrecon.shape_from_cloud must be true or false"),
                                         ("solid_ground", "no", "genrecon.solid_ground must be true or false"),
                                         ("shape_refine_t", 1.0, r"genrecon.shape_refine_t must be a float in \[0.0, 1.0\)"),
                                         ("smooth_iters", 2.5, "genrecon.smooth_iters must be a int"),
                                         ("denoise_sigma_r", 0.0, r"genrecon.denoise_sigma_r must be a float in"),
                                         ("denoise_normal_iters", None, "genrecon.denoise_normal_iters is missing"),
                                         ("edge_filter", None, "genrecon.edge_filter is missing"),
                                         ("edge_filter", "yes", "genrecon.edge_filter must be true or false"),
                                         ("denoise_block_faces", 0, "genrecon.denoise_block_faces must be a int in"),
                                         ("denoise_block_faces", None, "genrecon.denoise_block_faces is missing"))):
        cfg = {k: v for k, v in _CFG.items() if not (k == key and bad is None)}
        if bad is not None:
            cfg[key] = bad
        with pytest.raises(RuntimeError, match=msg):
            _run(tmp_path / f"bad{n}", monkeypatch, cfg)
    assert not envs                                               # nothing was launched


def test_the_decode_cap_reaches_the_runner_typed(tmp_path, monkeypatch):
    """USER 2026-11-13 ("Debemos hacer entrar pccr2408 hay que ver la forma la técnica para que sea escalable a nubes
    masivas"): genrecon.joint_decode_max_voxels reaches run_cloud.sh as GENRECON_DECODE_MAX_VOXELS — 0 (always the
    single pass) included; a missing key or anything but a non-negative integer fails naming it, nothing launched."""
    for n, (value, said) in enumerate(((100000, "100000"), (0, "0"))):
        envs.clear()
        _run(tmp_path / f"ok{n}", monkeypatch, {**_CFG, "generative_mesh_name": "", "joint_decode_max_voxels": value})
        assert envs[0]["GENRECON_DECODE_MAX_VOXELS"] == said
    envs.clear()
    missing = {k: v for k, v in _CFG.items() if k != "joint_decode_max_voxels"}
    with pytest.raises(RuntimeError, match="genrecon.joint_decode_max_voxels is missing"):
        _run(tmp_path / "missing", monkeypatch, missing)
    for n, bad in enumerate((-1, 2.5, True, "100000", None)):
        with pytest.raises(RuntimeError, match=r"genrecon.joint_decode_max_voxels must be a int in \[0, "):
            _run(tmp_path / f"bad{n}", monkeypatch, {**_CFG, "joint_decode_max_voxels": bad})
    assert not envs                                               # nothing was launched


def test_the_two_resume_lines_of_the_runner_have_their_own_marks():
    """run_cloud.sh says where a resumed run continues; the worker's loop takes the FIRST mark a line holds, and the
    decode's line holds the mesh step's "=== RESUME" too — each line reaches its own message."""
    script = (Path(G.__file__).resolve().parent.parent / "tools" / "genrecon" / "run_cloud.sh").read_text()
    assert 'echo "=== RESUME AT THE DECODE $sid ' in script and 'echo "=== RESUME $sid at the mesh step ' in script

    def first(line):
        return next((p, msg) for mark, p, msg in G._MARKS if mark in line)

    assert first("=== RESUME AT THE DECODE pccr 07:10:02") == (60, "the sampled latents reused — continuing at the decode")
    assert first("=== RESUME pccr at the mesh step 07:10:02") == (70, "the reconstruction reused — continuing at the mesh step")
    decode = [p for mark, p, _ in G._MARKS if mark.startswith("joint_decode")]
    assert decode and min(decode) > 60                            # the decode's own marks still advance after it


def test_the_work_directories_leave_after_a_good_run(tmp_path, monkeypatch):
    """USER 2026-11-14: "borra los intermedios y procura que se eliminen cuando termina genrecon para que no quede
    basura" — genrecon.keep_work reaches the runner typed; the adapter's inputs stay only for a variant that follows."""
    envs.clear()
    _run(tmp_path / "two", monkeypatch, _CFG)
    assert [(e["GENRECON_KEEP_WORK"], e["GENRECON_KEEP_INPUTS"]) for e in envs] == [("0", "1"), ("0", "0")]
    envs.clear()
    _run(tmp_path / "keep", monkeypatch, {**_CFG, "generative_mesh_name": "", "keep_work": True})
    assert [(e["GENRECON_KEEP_WORK"], e["GENRECON_KEEP_INPUTS"]) for e in envs] == [("1", "0")]
    envs.clear()
    with pytest.raises(RuntimeError, match="genrecon.keep_work is missing"):
        _run(tmp_path / "bad", monkeypatch, {k: v for k, v in _CFG.items() if k != "keep_work"})
    assert not envs


def test_drop_work_removes_only_the_runs_own_directories(tmp_path):
    """drop_work.sh: $W, $O and $OL go; kept on GENRECON_KEEP_WORK; $W kept on GENRECON_KEEP_INPUTS; a path that is
    not strictly inside $G/work/ is never removed; a failure's way out of run_cloud.sh never calls it."""
    import subprocess
    root = Path(G.__file__).resolve().parent.parent / "tools" / "genrecon"

    def go(env, w="s"):
        g = tmp_path / env.replace("=", "_") / "G"
        for d in ("s", "s_cloud_out", "s_cloud_out_light", "other"):
            (g / "work" / d).mkdir(parents=True)
            (g / "work" / d / "f").write_text("x")
        sh = (f'G={g}; W=$G/work/{w}; O=$G/work/s_cloud_out; OL=$G/work/s_cloud_out_light; '
              f'source {root / "drop_work.sh"}; {env} drop_work')
        subprocess.run(["bash", "-c", sh], check=True, capture_output=True)
        return sorted(p.name for p in (g / "work").iterdir())

    assert go("X=1") == ["other"]
    assert go("GENRECON_KEEP_WORK=1") == ["other", "s", "s_cloud_out", "s_cloud_out_light"]
    assert go("GENRECON_KEEP_INPUTS=1") == ["other", "s"]
    assert go("Y=1", w="") == ["other", "s"]                      # an empty session id: $G/work/ itself is never removed
    sh = (root / "run_cloud.sh").read_text()
    assert sh.count("\ndrop_work\n") + sh.count("\n  drop_work\n") == 2
    for line in sh.splitlines():                                  # no failure exit removes the work
        assert not ("exit 1" in line and "drop_work" in line)
