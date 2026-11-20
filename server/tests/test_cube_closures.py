"""Every revisit bridge with two fits is a loop edge: no closure is set aside by its relative scale (USER 2026-11-19: "tampoco está bien que
rechacemos costuras y correcciones que si son correctas pero las rechazamos con criterio nuestro" / "dejar el
criterio en el pipeline programado para que no se rechacen ni en esta ni en las otras escenas")."""
import os
import sys
from pathlib import Path

import numpy as np

FORK = Path(os.environ.get("STAC_FORK_DIR") or Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
sys.path.insert(0, str(FORK))
from loop_utils import cube_run as CR  # noqa: E402


def _run(tmp_path, bridges):
    """A CubeRun with three cubes that share no frame (0–9, 10–19, 20–29) and the given bridge fits on disk."""
    r = CR.CubeRun.__new__(CR.CubeRun)
    r.sets = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)]
    r.K, r.n = 3, 30
    r.owner = np.repeat(np.arange(3), 10)
    r.local = [{int(g): i for i, g in enumerate(F)} for F in r.sets]
    r.pairs = CR.overlap_pairs(r.sets)
    r.bridge_half = 2
    r.un = str(tmp_path)
    r.revisits = [(i, j) for (i, j), _ in bridges]
    for (i, j), fits in bridges:
        a, b = int(r.owner[i]), int(r.owner[j])
        rec = {"_stac_frames": [], "_stac_split": 0, "fits": fits,
               "side_pts": {"centroid": np.array([1.0, 0.5, 2.0]), "lever_m": 1.5}}
        np.save(r._bridge_path(a, b, i, j), rec)
    return r


def _fit(s, t=(0.0, 0.0, 0.0), res=0.02, n=5000):
    return (float(s), np.eye(3), np.asarray(t, float), float(res), int(n))


def test_a_closure_is_an_edge_whatever_its_relative_scale(tmp_path, capsys):
    r = _run(tmp_path, [((3, 14), {"a": _fit(1.02), "b": _fit(1.00)}),          # 1.02: what two good locks give
                        ((5, 25), {"a": _fit(1.60), "b": _fit(1.00)}),          # 1.60: a cube locked far off
                        ((7, 27), {"a": _fit(0.55), "b": _fit(1.00)})])         # 0.55: the other way
    out = r.closures()
    assert [e["revisit"] for e in out] == [[3, 14], [5, 25], [7, 27]], "every bridge with two fits is a loop edge"
    assert np.allclose([e["s"] for e in out], [1.02, 1.60, 0.55])
    # the seams' scale bound is gone (USER 2026-11-20: "no debe quedar nada que no agregue valor en la cadena"): once
    # it set no edge aside it was a flag nothing read
    assert all("scale_outside_seams_bound" not in e for e in out) and not hasattr(CR, "scale_bound")
    said = capsys.readouterr().out
    assert "3 bridge(s) are loop edges" in said and "bound" not in said
    assert "not an edge" not in said


def test_the_edge_is_the_one_the_two_fits_compose_to(tmp_path):
    r = _run(tmp_path, [((5, 25), {"a": _fit(1.5, t=(1.0, 0.0, 0.0), res=0.03, n=900),
                                    "b": _fit(1.2, t=(0.0, 2.0, 0.0), res=0.04, n=700)})])
    (e,) = r.closures()
    assert (e["a"], e["b"]) == (2, 0) and e["fit"] == "bridge" and e["exact"] is False
    assert np.isclose(e["s"], 1.5 / 1.2) and np.allclose(e["R"], np.eye(3))
    assert np.allclose(e["t"], np.array([1.0, 0.0, 0.0]) - (1.5 / 1.2) * np.array([0.0, 2.0, 0.0]))
    assert np.isclose(e["sigma_m"], np.hypot(0.03, 0.04)) and e["correspondences"] == 700 and e["lever_m"] == 1.5


def test_a_bridge_without_fits_is_still_no_edge_and_it_is_said(tmp_path, capsys):
    r = _run(tmp_path, [((3, 14), None), ((5, 25), {"a": _fit(1.3), "b": _fit(1.0)})])
    out = r.closures()
    assert [e["revisit"] for e in out] == [[5, 25]]
    assert "a side starved or degenerate — not an edge" in capsys.readouterr().out
