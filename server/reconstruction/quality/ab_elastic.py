"""Single-variable A/B of ``Model.elastic_seam`` (claude_stac.txt §7).

A2 (2026-07-11) saw no visible change when the elastic seam consensus was
switched off; the stage stays in the flow with no measured evidence either
way. This harness runs the CPU post-alignment stages of the fork TWICE on
COPIES of a session's aligned chunks — identical inputs, identical config,
only ``elastic_seam`` differs — and measures both outcomes with the same
instruments the fork already carries:

  * two-copy disagreement of every shared frame (§4.8 uncertainty, median m);
  * held-out surface pairs (exact-surface correspondences between frames at
    ``graph.holdout_offsets``, median m);
  * cross-frame depth disagreement of the depth graph's pair sensor (%).

The report (``output/quality/ab_elastic.json``) states the numbers and the
deltas; it does NOT change the default (``reconstruction.vggtomega.
elastic_seam``) — that decision is the user's, with his viewer.

The session must still hold ``maplong_run/_tmp_results_aligned`` (the
certification runs keep it: ``certify.keep_aligned_chunks``) and
``maplong_run/chunk_sim3.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

AB_JSON = "ab_elastic.json"


def _vendor_on_path():
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def make_runner(save_dir: Path, model_cfg: dict, img_list: List[str],
                chunk_indices: List[List[int]], sim3_list: List[tuple], img_dir: str):
    """A VGGT_Long instance without __init__ (no model), bound to ``save_dir``
    (which must hold _tmp_results_aligned/). Same construction the fork's own
    tests use."""
    _vendor_on_path()
    import vggt_long as vl
    r = object.__new__(vl.VGGT_Long)
    r.config = {"Model": copy.deepcopy(model_cfg)}
    r.chunk_size = int(model_cfg["chunk_size"])
    r.overlap = int(model_cfg["overlap"])
    r.chunk_indices = [tuple(int(x) for x in ci) for ci in chunk_indices]
    r.img_dir = str(img_dir)
    r.img_list = list(img_list)
    r.output_dir = str(save_dir)
    r.result_unaligned_dir = str(save_dir / "_tmp_results_unaligned")
    r.result_aligned_dir = str(save_dir / "_tmp_results_aligned")
    r.result_loop_dir = str(save_dir / "_tmp_results_loop")
    r.pcd_dir = str(save_dir / "pcd")
    for d in (r.result_unaligned_dir, r.result_loop_dir, r.pcd_dir):
        os.makedirs(d, exist_ok=True)
    r.sim3_list = [(float(s), np.asarray(R, np.float64), np.asarray(t, np.float64))
                   for s, R, t in sim3_list]
    r.loop_enable = False
    r.loop_predict_list = []
    r.loop_sim3_list = []
    r.loop_list = []
    r.loop_cands = []
    return r


def measure(runner, graph_cfg: dict, depth_offsets=(1, 2, 3, 5, 8, 12)) -> dict:
    """The three instruments on the runner's CURRENT aligned chunks."""
    _vendor_on_path()
    from loop_utils.metric_lock import depth_pair_samples, pair_depth_relation, frame_owner
    runner._stac_uncertainty()
    unc = float(getattr(runner, "_stac_uncert_median", float("nan")))
    held = runner._stac_holdout_pairs(graph_cfg)
    held_med = float(np.median([float(np.median(np.linalg.norm(p - q, axis=1)))
                                for _f, _g, p, q in held])) if held else float("nan")
    # depth-graph pair sensor on owner frames (same association as _stac_depth_graph)
    N = len(runner.img_list)
    owner = frame_owner(runner.chunk_indices, N)
    cache = {}
    for k, (start, end) in enumerate(runner.chunk_indices):
        data = runner._stac_load_chunk_aligned(k)
        wp = np.asarray(data['world_points']); wp = wp[0] if wp.ndim == 5 else wp
        cf = np.asarray(data['world_points_conf']).reshape(wp.shape[:3])
        ext = np.asarray(data['extrinsic']); K = np.asarray(data['intrinsic'])
        for local, g in enumerate(range(start, end)):
            if owner[g] == k:
                c2w = runner._stac_aligned_pose(k, local, ext[local])
                cache[g] = (wp[local].astype(np.float32), cf[local].astype(np.float32),
                            np.linalg.inv(c2w), K[local])
    runner._stac_drop_aligned_cache()
    before = []
    for f in sorted(cache):
        for d in depth_offsets:
            g = f + d
            if g not in cache:
                continue
            zs = depth_pair_samples(cache[f][0], cache[f][1], cache[g][0], cache[g][1],
                                    cache[g][2], cache[g][3])
            if zs is None:
                continue
            rel = pair_depth_relation(zs[0], zs[1])
            if rel is not None:
                before.append(rel[2])
    return {"two_copy_disagreement_median_m": unc,
            "holdout_surface_pairs_median_m": held_med,
            "holdout_n_pairs": len(held),
            "depth_pair_disagreement_median_pct": (float(np.median(before)) * 100.0
                                                   if before else float("nan")),
            "depth_pairs_n": len(before)}


def run_variant(src_aligned: Path, model_cfg: dict, img_list, chunk_indices, sim3_list,
                img_dir: str, graph_cfg: dict, elastic: bool, work: Path,
                log: Callable[[str], None] = print) -> dict:
    vdir = work / ("elastic_on" if elastic else "elastic_off")
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    shutil.copytree(src_aligned, vdir / "_tmp_results_aligned")
    cfg = copy.deepcopy(model_cfg)
    cfg["elastic_seam"] = bool(elastic)
    r = make_runner(vdir, cfg, img_list, chunk_indices, sim3_list, img_dir)
    t0 = time.time()
    r._stac_elastic_seams()
    r._stac_intra_chunk()
    r._stac_depth_graph()
    r._stac_blend_copies()
    m = measure(r, graph_cfg)
    m["elastic_seam"] = bool(elastic)
    m["elapsed_s"] = round(time.time() - t0, 1)
    m["stages"] = {k: bool(cfg.get(k)) for k in ("elastic_seam", "intra_chunk", "depth_graph",
                                                  "blend_copies")}
    return m


def run_ab(src_aligned: Path, model_cfg: dict, img_list, chunk_indices, sim3_list, img_dir: str,
           graph_cfg: dict, out_dir: Path, work: Optional[Path] = None,
           log: Callable[[str], None] = print) -> dict:
    """Both variants on copies; report with deltas (on − off)."""
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="ab_elastic_"))
    off = run_variant(src_aligned, model_cfg, img_list, chunk_indices, sim3_list, img_dir,
                      graph_cfg, False, work, log)
    on = run_variant(src_aligned, model_cfg, img_list, chunk_indices, sim3_list, img_dir,
                     graph_cfg, True, work, log)
    keys = ("two_copy_disagreement_median_m", "holdout_surface_pairs_median_m",
            "depth_pair_disagreement_median_pct")
    deltas = {k: (float(on[k]) - float(off[k])) if np.isfinite(on[k]) and np.isfinite(off[k])
              else None for k in keys}
    rep = {"version": 1, "variable": "Model.elastic_seam", "off": off, "on": on,
           "delta_on_minus_off": deltas,
           "note": "single-variable A/B on copies of the aligned chunks; the default is "
                   "NOT changed by this harness — the user decides with the numbers and "
                   "his viewer (claude_stac.txt §7)",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "provenance": "tool_measured"}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / AB_JSON).write_text(json.dumps(rep, indent=1, default=float))
    log(f"[ab-elastic] off: 2-copy {off['two_copy_disagreement_median_m']*100:.2f} cm, held-out "
        f"{off['holdout_surface_pairs_median_m']*100:.2f} cm, depth {off['depth_pair_disagreement_median_pct']:.2f}% "
        f"| on: {on['two_copy_disagreement_median_m']*100:.2f} cm, {on['holdout_surface_pairs_median_m']*100:.2f} cm, "
        f"{on['depth_pair_disagreement_median_pct']:.2f}% → {out_dir / AB_JSON}")
    shutil.rmtree(work, ignore_errors=True)
    return rep


def run_ab_session(session_dir, log: Callable[[str], None] = print) -> dict:
    """CLI entry: resolve everything from a session directory."""
    import yaml
    session_dir = Path(session_dir)
    output_dir = session_dir / "output"
    run_dir = output_dir / "maplong_run"
    aligned = run_dir / "_tmp_results_aligned"
    if not aligned.exists() or not any(aligned.glob("chunk_*.npy")):
        raise RuntimeError(f"{aligned} is missing — the A/B needs the aligned chunks "
                           f"(certify.keep_aligned_chunks keeps them after the scale)")
    sim3_path = run_dir / "chunk_sim3.json"
    if not sim3_path.exists():
        raise RuntimeError(f"{sim3_path} is missing — re-run the reconstruction with the "
                           f"F2 fork (it persists the accumulated chunk transforms)")
    sim3 = json.loads(sim3_path.read_text())
    cfg_path = output_dir / "vggt_omega_config.yaml"
    if not cfg_path.exists():
        raise RuntimeError(f"{cfg_path} is missing")
    model_cfg = yaml.full_load(cfg_path.read_text())["Model"]   # our own artifact (older files carry tuple tags)
    names = json.loads((run_dir / "frame_list.json").read_text())
    frames_dir = session_dir / "frames"
    img_list = [str(frames_dir / n) for n in names]
    graph_cfg = model_cfg.get("graph")
    if not graph_cfg:
        from reconstruction.loops.config import load_loops_config, fork_model_graph
        graph_cfg = fork_model_graph(load_loops_config())
    return run_ab(aligned, model_cfg, img_list, sim3["chunk_indices"],
                  [(e["s"], e["R"], e["t"]) for e in sim3["sim3"]], str(frames_dir),
                  graph_cfg, output_dir / "quality", log=log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A/B of Model.elastic_seam on a session (§7)")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    run_ab_session(args.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
