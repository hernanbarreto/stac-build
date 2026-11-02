"""tools/genrecon/genrecon_levels.py — the chunk cubes' base is the LOWEST level, of the support surfaces
or of the whole cloud (station 2026-11-02: the tracks 1.15 m under the platform sat under every cube)."""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "genrecon"
_spec = importlib.util.spec_from_file_location("genrecon_levels", _TOOLS / "genrecon_levels.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)


def _old_support_levels(ys):
    """The adapter's support levels before 2026-11-02, verbatim."""
    h, e = np.histogram(ys, bins=np.arange(np.floor(ys.min() * 20) / 20, ys.max() + 0.05, 0.05))
    return [float(e[i]) for i in range(len(h)) if h[i] / h.sum() >= 0.01]


def _scene(rng, trench=True):
    platform = rng.normal(0.0, 0.02, 200_000)                     # the support the cameras walk on
    walls = rng.uniform(0.0, 5.0, 100_000)
    ceiling = rng.normal(5.3, 0.03, 30_000)
    parts = [platform, walls, ceiling]
    if trench:
        parts.append(rng.normal(-1.08, 0.03, 40_000))            # the tracks: no support noun
    return np.concatenate(parts), platform, ceiling


def test_the_tracks_under_the_platform_set_the_base():
    rng = np.random.default_rng(0)
    y, sup, ceil = _scene(rng)
    lv = L.chunk_levels(y, sup, [ceil])
    assert lv["source"].startswith("the whole cloud's lowest level")
    assert -1.20 <= lv["z_floor"] <= -1.05                        # the trench, not the platform
    assert lv["support_levels"] == _old_support_levels(sup)       # the support's levels as before
    assert min(lv["support_levels"]) > -0.2
    assert lv["z_ceiling"] == float(np.percentile(ceil, 95))


def test_a_scene_on_one_floor_keeps_the_support_level():
    rng = np.random.default_rng(1)
    y, sup, ceil = _scene(rng, trench=False)
    lv = L.chunk_levels(y, sup, [ceil])
    assert lv["source"] == "segmentation support surfaces"
    assert lv["z_floor"] == min(_old_support_levels(sup))         # the base the adapter used before


def test_no_support_and_no_level_falls_back_to_the_percentile():
    y = np.random.default_rng(2).uniform(-1.0, 9.0, 50_000)       # spread over 200 bins: none holds 1 %
    lv = L.chunk_levels(y, np.zeros(0), [])
    assert lv["cloud_levels"] == [] and "support_levels" not in lv
    assert lv["z_floor"] == float(np.percentile(y, 0.2))
    assert lv["z_ceiling"] == float(np.percentile(y, 99.97))
    assert lv["source"].startswith("percentile 0.2")


def test_the_runner_parses():
    assert subprocess.run(["bash", "-n", str(_TOOLS / "run_cloud.sh")]).returncode == 0
    text = (_TOOLS / "run_cloud.sh").read_text()
    assert '[ "$mode" = "generative" ] && tag=gen' in text      # its own output folders
    assert 'COORDS=""' in text                                  # no voxel from the cloud
