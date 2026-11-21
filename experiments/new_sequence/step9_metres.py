"""THE NEW ORDER, step 9 — "Escala a metros (scale_align) y orientación."

What the pipeline's map worker does after the cube stage, with its own functions and the job's own settings
(workers/map_worker.py: _postprocess_reconstruction, then the body of _metricize_and_orient):

  1. the cube stage's products collected into the session (_postprocess_reconstruction);
  2. each keyframe's record written from the placed cubes (_emit_omega_depth) — here the corrected, tied cubes;
  3. the metre: reconstruction.scale_align.run — ONE factor for the whole session against DA3's depth of the keyframes;
  4. the upright orientation from the camera poses: reconstruction.orient.run.

The DA3 depth of the keyframes (da3_run/results_output — the anchors the worker extracts before Omega, the same files
step 2's lock read) is copied into the session first.

    python step9_metres.py <session scan dir> <session the DA3 anchors are copied from>"""
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cube_sessions as CS  # noqa: E402


def main() -> int:
    scan, anchors_from = Path(sys.argv[1]), Path(sys.argv[2])
    out = scan / "output"
    save_dir = out / "maplong_run"
    from precision.config import load_precision_config, raw_of
    from reconstruction.chunk_plan import omega_cube_layout
    from reconstruction.loops.config import load_loops_config
    from reconstruction.orient import run as orient_run
    from reconstruction.scale_align import run as scale_run
    from workers import map_worker as MW
    config = raw_of(load_precision_config(session_dir=scan))
    recon = config["reconstruction"]
    va, simple = recon.get("vggtomega", {}) or {}, recon.get("simple") or {}
    cfg_v = yaml.safe_load((out / "vggt_omega_config.yaml").read_text())
    pipe = CS._Pipe()
    log = lambda m: print(m, flush=True)   # noqa: E731

    dst = out / "da3_run" / "results_output"
    if not dst.is_dir():
        shutil.copytree(anchors_from / "output" / "da3_run" / "results_output", dst)
    log(f"[step 9] DA3 depth of the keyframes in the session: {len(list(dst.glob('frame_*.npz')))} file(s)")

    MW._postprocess_reconstruction(pipe, save_dir, out, cfg_v, config, backend="mapanything")
    cub = omega_cube_layout(out)
    if cub is None:
        raise SystemExit("[step 9] no record of the cube layout the tie ran (maplong_run/chunk_sim3.json)")
    if (cub["chunk_sets"] != [[int(g) for g in F] for F in cfg_v["Model"]["chunk_sets"]]
            or cub["owner"] != [int(x) for x in cfg_v["Model"]["chunk_owner"]]):
        raise SystemExit(f"[step 9] the tie ran another cube layout ({cub['source']}) than the plan")
    MW._emit_omega_depth(save_dir, out, None, str(scan / "frames" / "selected_frames.json"), pipe,
                         cubes=(cub["chunk_sets"], cub["owner"]))
    (out / "zones.json").unlink(missing_ok=True)
    keep = bool(load_loops_config(config).certify.keep_aligned_chunks)
    rng = MW._view_range_m(config)
    before = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    s = scale_run(out, dry_run=False, aligned_chunks=keep, fresh_products=True, range_m=rng,
                  log=lambda m: log(f"[scale-align] {m}"),
                  near_frac=float(va.get("scale_near_frac", 0.25)),
                  conf_top_frac=float(va.get("scale_conf_top_frac", 0.10)),
                  cfg={"mode": str(va.get("scale_mode", "global_median")),
                       "vio": bool(va.get("scale_vio", True)),
                       "vio_segment_s": float(va.get("vio_segment_s", 5.0)),
                       "vio_min_segments": int(va.get("vio_min_segments", 8)),
                       "vio_min_coverage": float(va.get("vio_min_coverage", 0.5))},
                  session_dir=scan)
    if s is None:
        raise SystemExit("[step 9] scale_align could not estimate the metre — nothing is published non-metric")
    T = None
    if bool(simple.get("enabled", False)) and bool(simple.get("orient_from_poses", True)):
        T = orient_run(out, log=lambda m: log(f"[orient] {m}"), aligned_chunks=keep, fresh_products=True, range_m=rng)
    after = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    path = lambda P: float(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1).sum())   # noqa: E731
    log(f"[step 9] the metre: ONE factor x{float(s):.5f} for the whole session; the walk measures {path(before):.3f} "
        f"before and {path(after):.3f} m after; orientation {'applied' if T is not None else 'NOT applied'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
