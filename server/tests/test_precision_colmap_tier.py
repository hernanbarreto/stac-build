"""COLMAP as TIER 2 (USER 2026-09-29): where the sweep measured nothing, COLMAP's
geometric depth — consistent by the same rule as tier 0 — replaces the prior, with
its own provenance (source 20), switchable by precision.depth.colmap.as_tier."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from precision import depth_sweep as DS                       # noqa: E402
from precision import fuse as FU                               # noqa: E402
from precision import provenance as PV                         # noqa: E402

import test_precision_depth as T                               # noqa: E402


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    from precision.config import load_precision_config
    p = load_precision_config()
    pc = replace(p, depth=replace(p.depth, n_views=3, best_k=2, n_hyp=16, null_frames=2,
                                  min_scale_samples=50, calib_min_bin_samples=50,
                                  calib_samples_per_frame=2000, view_samples=128,
                                  propagation_iters=1, min_consistent_views=2,
                                  colmap=replace(p.depth.colmap, as_tier=True)),
                 fuse=replace(p.fuse, cleaning=False))
    s, _ = T._session(tmp_path_factory.mktemp("colmap_tier"), pc)
    DS.run_sweep(s, pc, log=lambda *a: None, device="cpu")
    # a synthetic COLMAP result: the prior's surface wherever the sweep did not measure,
    # confirmed by three views, contradicted by none — and a contradicted patch
    out = s / "output"
    cdir = out / "depth_colmap"
    cdir.mkdir()
    swept_frames = sorted(int(p.stem.split("_")[1]) for p in (out / DS.DEPTH_DIRNAME).glob("frame_*.npz"))
    for f in swept_frames:
        with np.load(out / DS.DEPTH_DIRNAME / f"frame_{f}.npz") as z:
            depth, src = z["depth"], z["source"]
        cd = np.where(src == DS.SOURCE_PRIOR_FILL, depth, 0.0).astype(np.float32)
        n_c = np.where(cd > 0, 3, 0).astype(np.uint8)
        n_b = np.zeros(cd.shape, np.uint8)
        n_b[:4, :] = 5                                         # the top rows: seen through
        np.savez_compressed(cdir / f"frame_{f}.npz", depth=cd, n_consistent=n_c, n_contradict=n_b,
                            residual_rel=np.where(cd > 0, 0.001, np.nan).astype(np.float32))
    return s, pc


def test_colmap_replaces_the_prior_where_the_sweep_measured_nothing(swept):
    s, pc = swept
    on = FU.fuse(s, pc, log=lambda *a: None)
    off = FU.fuse(s, replace(pc, depth=replace(pc.depth, colmap=replace(pc.depth.colmap, as_tier=False))),
                  log=lambda *a: None)
    rep_on, rep_off = on["report"], off["report"]
    assert rep_on["colmap_as_tier"]["enabled"] and not rep_off["colmap_as_tier"]["enabled"]
    assert rep_on["points_by_tier"]["tier2_colmap"] > 0
    assert rep_off["points_by_tier"]["tier2_colmap"] == 0
    # every tier-2 pixel came out of the tier-1 set; the contradicted rows never entered
    assert rep_on["points_by_tier"]["tier1_prior_fill"] < rep_off["points_by_tier"]["tier1_prior_fill"]
    o = on["origins"]
    t2 = o["source"] == PV.SOURCE_COLMAP
    assert np.all(o["pixel_v_und"][t2] >= 4)
    assert np.all(o["n_consistent"][t2] >= pc.fuse.min_witness_views)
    # tier 0 is untouched by the switch
    assert rep_on["points_by_tier"]["tier0"] == rep_off["points_by_tier"]["tier0"]
    # the accounting closes and the PLY carries the tier with its own confidence channel
    assert rep_on["n_points"] + rep_on["n_rejected"] == rep_on["n_candidates"]
    data = on["data"]
    m = data["source"] == PV.SOURCE_COLMAP
    assert m.any() and np.allclose(data["confidence"][m], np.clip(data["n_consistent"][m] / pc.depth.n_views, 0, 1))


def test_evidence_ranks_measured_over_colmap_over_prior():
    n = np.array([3, 8, 8], np.uint8)
    ncc = np.array([0.9, np.nan, np.nan], np.float32)
    src = np.array([DS.SOURCE_SWEEP, PV.SOURCE_COLMAP, DS.SOURCE_PRIOR_FILL], np.uint8)
    e = FU.evidence(n, ncc, src)
    assert e[0] > e[1] > e[2]
