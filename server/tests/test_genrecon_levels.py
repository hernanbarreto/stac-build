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


def test_stacked_cubes_stand_on_the_lowest_level_and_reach_the_top():
    cube = 3.7
    zs = L.stacked_centers(-1.15, 5.661, cube)              # station: the tracks' level to the cloud's top
    assert zs[0] - cube / 2 == -1.15                        # the first layer stands on the lowest level
    assert zs[-1] + cube / 2 >= 5.661                       # the last one reaches the top
    strides = np.diff(zs)
    assert len(zs) == 3 and np.allclose(strides, strides[0])
    assert strides[0] <= cube * (1 - L.MIN_OVERLAP) + 1e-9  # at least a quarter of overlap between layers
    k = strides[0] / (cube * L.ATOMIC)
    assert abs(k - round(k)) < 1e-9                         # on GenRecon's 1/16 lattice
    # the fewest layers: one less would not reach the top with that overlap
    n = len(zs) - 1
    assert -1.15 + cube + (n - 1) * cube * (1 - L.MIN_OVERLAP) < 5.661


def test_a_span_shorter_than_a_cube_is_one_layer_and_zero_keeps_genrecons_own():
    assert L.stacked_centers(0.0, 3.0, 3.7) == [1.85]
    lv = {"z_floor": -0.1, "z_top": 5.7}
    assert "chunk_z_centers" not in L.stack_layers(dict(lv), 0.0)
    out = L.stack_layers(dict(lv), 3.7)
    assert out["chunk_size_m"] == 3.7 and out["chunk_z_centers"] == L.stacked_centers(-0.1, 5.7, 3.7)


def test_the_top_is_the_clouds_highest_surface_above_the_ceiling():
    rng = np.random.default_rng(3)
    y, sup, ceil = _scene(rng)
    y = np.concatenate([y, rng.normal(6.0, 0.02, 2_000)])  # ducts above the ceiling instance
    lv = L.chunk_levels(y, sup, [ceil])
    assert lv["z_top"] == max(lv["z_ceiling"], float(np.percentile(y, 99.97)))
    assert lv["z_top"] > lv["z_ceiling"]
