"""THE NEW ORDER, step 2 — "Omega por cubo y candado métrico DA3 por cubo: como hoy. No se ata nada."

The pipeline's own cube stage (vendor/VGGT-Long: vggt_long.py -> loop_utils.cube_run), launched exactly as the
pipeline launches it, and STOPPED after the metric lock: for every cube CubeRun.infer and CubeRun.witness_pass (Omega's
two passes), then CubeRun.metric_lock (each cube's DA3 scale). The steps of the tie — bridges, seams, closures, graph,
apply — are NOT called here: they are step 8.

No file of the fork is edited: the fork's entry point runs as it is (its own deterministic set-up, model load, cube
layout) and reaches this function where it would reach process_cubes; every call below is the fork's own method, in the
fork's own order.

    python step2_omega.py --image_dir <frames> --config <vggt_omega_config.yaml> --save_dir <dir> --selected_frames <json>
    (the fork's own arguments; run from vendor/VGGT-Long with the mapanything interpreter)"""
import runpy
import sys

FORK = "/workspace/stac-build/vendor/VGGT-Long"


def cubes_until_lock(vl, *, images_to_u8, images_u8, save_ply, apply_sim3_direct):
    """loop_utils.cube_run.process_cubes up to and including the metric lock."""
    import torch
    from loop_utils import cube_run as CR
    run = CR.CubeRun(vl, images_to_u8, images_u8, save_ply, apply_sim3_direct)
    sizes = [len(F) + len(c) for F, c in zip(run.sets, vl.chunk_context)]
    print(f"{CR.LOG} {run.K} cube(s) over {run.n} keyframes: {min(sizes)}–{max(sizes)} Omega views each, "
          f"{len(run.pairs)} pair(s) sharing frames; layout {run.digest[:12]}")
    for k in range(run.K):
        print(f"[Progress]: {k}/{run.K - 1}")
        run.infer(k)
        torch.cuda.empty_cache()
        run.witness_pass(k)
        torch.cuda.empty_cache()
    run.metric_lock()
    print("[new order] step 2 ends here: Omega per cube and the DA3 lock per cube are done; NOTHING IS TIED "
          "(no bridges, no seams, no closures, no graph, no placement)", flush=True)


def main() -> None:
    sys.path.insert(0, FORK)
    import loop_utils.cube_run as CR
    CR.process_cubes = cubes_until_lock          # vggt_long.run() imports it at call time: it gets this one
    sys.argv = [FORK + "/vggt_long.py", *sys.argv[1:]]
    runpy.run_path(FORK + "/vggt_long.py", run_name="__main__")


if __name__ == "__main__":
    main()
