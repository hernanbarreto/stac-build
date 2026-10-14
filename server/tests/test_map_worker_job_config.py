"""workers.map_worker reads the JOB's configuration — the dict the worker received — never
``config.yaml`` re-read from disk inside the worker (docs/plan_determinismo.md point 69,
USER 2026-10-08): the post-reconstruction cleanup's ``certify.keep_aligned_chunks`` comes from
the job config handed down by every backend's post-process, and no ``from config import cfg``
is left in the module. CPU only, no pipeline."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers import map_worker as MW                            # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
SRC = Path(MW.__file__).read_text()


class _Pipe:
    def __init__(self):
        self.logs = []

    def send_log(self, m, level="info"):
        self.logs.append(str(m))

    def send_progress(self, *a, **k):
        pass


def _job_config(keep: bool) -> dict:
    with open(SERVER / "config.yaml") as f:
        raw = yaml.safe_load(f)
    raw["certify"]["keep_aligned_chunks"] = keep
    return raw


def _temps(tmp_path):
    save, out = tmp_path / "maplong_run", tmp_path / "output"
    for d in ("_tmp_results_unaligned", "_tmp_results_loop", "pcd", "uncert", "_tmp_results_aligned"):
        (save / d).mkdir(parents=True)
        (save / d / "x.bin").write_bytes(b"x" * 8)
    out.mkdir()
    return save, out


@pytest.mark.parametrize("keep", [True, False])
def test_the_cleanup_reads_keep_aligned_chunks_from_the_job_config(tmp_path, keep):
    save, out = _temps(tmp_path)
    pipe = _Pipe()
    MW._cleanup_recon_temps(save, out, "mapanything", pipe, _job_config(keep))
    assert not (save / "_tmp_results_unaligned").exists() and not (save / "pcd").exists()
    assert (save / "_tmp_results_aligned").exists()                 # the TSDF depth source
    assert (save / "_tmp_results_loop").exists() is keep and (save / "uncert").exists() is keep
    assert not any("certify config unavailable" in m for m in pipe.logs)


def test_an_unloadable_job_config_is_declared_and_the_bridges_go(tmp_path):
    save, out = _temps(tmp_path)
    pipe = _Pipe()
    MW._cleanup_recon_temps(save, out, "mapanything", pipe, {"certify": {}})
    assert not (save / "_tmp_results_loop").exists()
    assert any("certify config unavailable" in m for m in pipe.logs)


def test_no_config_yaml_is_re_read_inside_the_worker():
    assert "from config import cfg" not in SRC, "point 69: the worker runs on the job's dict"
    # every backend's post-process hands the job's config down to the cleanup
    import inspect
    assert "job_config" in inspect.signature(MW._postprocess_reconstruction).parameters
    assert "job_config" in inspect.signature(MW._cleanup_recon_temps).parameters
    calls = [ln for ln in SRC.splitlines() if "_postprocess_reconstruction(pipe," in ln]
    assert len(calls) == 4 and all(", config," in ln for ln in calls), calls
    body = SRC[SRC.index("def _postprocess_reconstruction"):SRC.index("def _build_da3_config")]
    assert body.count("_cleanup_recon_temps(save_dir, output_dir, backend, pipe, job_config)") == 2
