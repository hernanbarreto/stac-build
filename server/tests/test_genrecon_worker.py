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
        "cube_size_m": 3.7}


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
