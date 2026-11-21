"""THE NEW ORDER, step 8 — "Atada del pipeline: costuras, cierres con puentes de Omega y grafo Sim(3) exactamente igual
que en el pipeline principal".

The pipeline's own cube stage from the point where step 2 stopped it — loop_utils.cube_run.process_cubes after the
metric lock, call for call and in its order: the bridges between passes (Omega, one joint pass per revisit), the seams
between cubes that share keyframes, the closures and the Sim(3) graph, each cube taken to its place, then the rest of
the stage as the pipeline runs it (uncertainty, depth graph, the cubes' outputs, one pose per keyframe). The cubes it
reads are the corrected ones (step8_cubes.py). No file of the fork is edited.

The metric lock is NOT run again (step 2 locked the cubes; a second run would measure it on corrected depth): the two
values that step leaves in the running object — recorded in the graph's report, not used by its solve — are read back
from its own record, metric_lock.json.

    python step8_tie.py --image_dir <frames> --config <vggt_omega_config.yaml> --save_dir <tie dir> --selected_frames <json>
    (the fork's own arguments; run from vendor/VGGT-Long with the mapanything interpreter)"""
import json
import os
import runpy
import sys

FORK = "/workspace/stac-build/vendor/VGGT-Long"


def cubes_tie(vl, *, images_to_u8, images_u8, save_ply, apply_sim3_direct):
    """loop_utils.cube_run.process_cubes from the bridges on."""
    import torch
    from loop_utils import cube_run as CR
    run = CR.CubeRun(vl, images_to_u8, images_u8, save_ply, apply_sim3_direct)
    missing = [k for k in range(run.K) if not os.path.exists(CR._path(run.un, k))]
    if missing:
        raise RuntimeError(f"{CR.LOG} corrected cube(s) {missing} are not on disk — nothing is tied")
    rep = json.load(open(os.path.join(vl.output_dir, "metric_lock.json")))
    ml = run.cfg.get('metric_lock') or {}
    chunks = {int(k): c for k, c in rep["chunks"].items()}
    run.scale_inputs = {"s_da3": {k: float(c["s"]) for k, c in chunks.items() if c.get("s")},
                        "n_anchors": {k: int(c["n_anchors"]) for k, c in chunks.items() if c.get("s")},
                        "sigma_anchor": float(ml.get('sigma_anchor', 0.08))}
    run.scales_applied = {k: float(c["s_applied"]) for k, c in chunks.items() if c.get("s_applied")}
    print(f"{CR.LOG} [new order] the tie starts on {run.K} CORRECTED cube(s); the lock of step 2 read back: "
          + ", ".join(f"{k}: {v:.4f}" for k, v in sorted(run.scales_applied.items())), flush=True)
    if run.revisits and run.K > 1:
        run.bridges()
    del vl.model
    torch.cuda.empty_cache()
    run.graph(run.seams() if run.K > 1 else [])
    run.apply()
    if vl._stac_graph_cfg() is not None:
        run.uncertainty()
    run.depth_graph()
    run.outputs()
    run.camera_poses()
    print(f"{CR.LOG} done", flush=True)


def main() -> None:
    sys.path.insert(0, FORK)
    import loop_utils.cube_run as CR
    CR.process_cubes = cubes_tie                 # vggt_long.run() imports it at call time: it gets this one
    sys.argv = [FORK + "/vggt_long.py", *sys.argv[1:]]
    runpy.run_path(FORK + "/vggt_long.py", run_name="__main__")


if __name__ == "__main__":
    main()
