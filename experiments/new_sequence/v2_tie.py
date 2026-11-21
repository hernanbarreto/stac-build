"""VERSION 2, step 11 — "Segunda atada, solo con los puentes de Omega entre pasadas."

The fork's own cube stage (loop_utils.cube_run) on the cubes of v2_tie_cubes.py — the session after F5 and F6's depth:

  * CubeRun.bridges: one joint Omega pass per revisit between two cubes that share no keyframe, fitted onto those
    two cubes as they stand NOW (the fits of the first tie were made on the cubes as Omega left them);
  * CubeRun.seams: every two cubes that share keyframes hold the same points for them, so the fit is the identity —
    a seam measures nothing here; its tolerance is the one it measured in the FIRST tie (its residual there, read
    from the session's chunk_graph.json) — without one the graph could not weigh the bridges against anything;
  * CubeRun.closures + CubeRun.graph: the Sim(3) graph, cube 0 fixed, loop edges under Cauchy — as in the pipeline.
    The closures are judged by the scale bound of the first tie's seams (the session's own; none when it had fewer
    than three seams), not by the identity fits' (zero spread).

Only the graph is solved: its per-cube Sim(3) (chunk_graph.json in the tie's directory) is what step 12 applies. No
file of the fork is edited.

    V2_FIRST_GRAPH=<session>/output/maplong_run/chunk_graph.json
    python v2_tie.py --image_dir <frames> --config <vggt_omega_config.yaml> --save_dir <tie dir> --selected_frames <json>
    (the fork's own arguments; run from vendor/VGGT-Long with the mapanything interpreter)"""
import json
import os
import runpy
import sys

FORK = "/workspace/stac-build/vendor/VGGT-Long"


def second_tie(vl, *, images_to_u8, images_u8, save_ply, apply_sim3_direct):
    import numpy as np
    import torch
    from loop_utils import cube_run as CR
    run = CR.CubeRun(vl, images_to_u8, images_u8, save_ply, apply_sim3_direct)
    missing = [k for k in range(run.K) if not os.path.exists(CR._path(run.un, k))]
    if missing:
        raise RuntimeError(f"{CR.LOG} cube(s) {missing} are not on disk — nothing is tied")
    first = json.load(open(os.environ["V2_FIRST_GRAPH"]))
    if first.get("layout_sha256") != run.digest:
        raise RuntimeError(f"{CR.LOG} the first tie's graph is of another cube layout")
    rep = json.load(open(os.path.join(vl.output_dir, "metric_lock.json")))
    ml = run.cfg.get('metric_lock') or {}
    chunks = {int(k): c for k, c in rep["chunks"].items()}
    run.scale_inputs = {"s_da3": {k: float(c["s"]) for k, c in chunks.items() if c.get("s")},
                        "n_anchors": {k: int(c["n_anchors"]) for k, c in chunks.items() if c.get("s")},
                        "sigma_anchor": float(ml.get('sigma_anchor', 0.08))}
    run.scales_applied = {k: float(c["s_applied"]) for k, c in chunks.items() if c.get("s_applied")}
    print(f"{CR.LOG} [version 2] the SECOND tie: {run.K} cube(s) in the session's state after F5 and F6's depth, "
          f"{len(run.revisits)} revisit(s), {len(run._bridge_items())} of them between two cubes that share no keyframe",
          flush=True)
    if run.revisits and run.K > 1:
        run.bridges()
    del vl.model
    torch.cuda.empty_cache()
    seams = run.seams() if run.K > 1 else []
    sig1 = {(int(e["a"]), int(e["b"])): float(e["sigma_m"]) for e in first["seams"]}
    for e in seams:
        key = (int(e["a"]), int(e["b"]))
        if key not in sig1:
            raise RuntimeError(f"{CR.LOG} seam {key} has no record in the first tie")
        e["identity_fit"] = {"s": float(e["s"]), "residual_m": float(e["sigma_m"]),
                             "t_norm_m": float(np.linalg.norm(np.asarray(e["t"], np.float64)))}
        e["sigma_m"], e["sigma_source"] = sig1[key], "the first tie's residual of this seam"
        # THE IDENTITY, WRITTEN EXACTLY. The two cubes hold the same points, so the fit is the identity — returned
        # with rounding dust (|t| ~ 5e-16 m, s - 1 ~ 2e-16). The graph starts at the seams' tree, and scipy's solver
        # takes the NORM OF ITS START as its first trust radius (1.0 only when that norm is exactly 0): from a start
        # of 1e-16 its first step is 1e-16 long, the cost does not change and it stops there as "converged" — the
        # first run of this step returned the session unmoved with the bridges' cost untouched (6.14; the optimum,
        # reached from the exact identity and from three other starts, is 4.94).
        dust = max(abs(float(e["s"]) - 1.0), float(np.abs(np.asarray(e["R"], np.float64) - np.eye(3)).max()),
                   float(np.abs(np.asarray(e["t"], np.float64)).max()))
        if dust > 1e-9:
            raise RuntimeError(f"{CR.LOG} seam {key} of the second tie is not the identity (off by {dust:.3e})")
        e["s"], e["R"], e["t"] = 1.0, np.eye(3), np.zeros(3)
    run.scale_bound = CR.scale_bound([float(e["s"]) for e in first["seams"] if e.get("fit") == "Sim3"])
    sb = run.scale_bound
    print(f"{CR.LOG} [version 2] seams: {len(seams)} identity fit(s) (largest |t| "
          f"{max([e['identity_fit']['t_norm_m'] for e in seams] or [0.0]) * 1000:.3f} mm, largest |s - 1| "
          f"{max([abs(e['identity_fit']['s'] - 1.0) for e in seams] or [0.0]):.2e}); their tolerance the first tie's: "
          + ", ".join(f"{e['b']}<->{e['a']} {e['sigma_m'] * 100:.1f} cm" for e in seams)
          + "; closures' scale bound: "
          + (f"|log s - {sb['median']:+.4f}| <= {sb['bound']:.4f} (the first tie's seams)" if sb["bound"] is not None
             else f"none ({sb['n']} seam scale(s) in the first tie — no spread to measure)"), flush=True)
    run.graph(seams)
    doc = json.load(open(os.path.join(vl.output_dir, "chunk_graph.json")))
    P = doc["poses"]
    print(f"{CR.LOG} [version 2] second tie solved ({doc['applied']}): {len(doc['closures'])} closure(s) in the solve; "
          f"per cube scale " + ", ".join(f"{float(p['s']):.5f}" for p in P) + "; moved (m) "
          + ", ".join(f"{float(np.linalg.norm(np.asarray(p['t'], np.float64))):.4f}" for p in P), flush=True)
    print(f"{CR.LOG} done", flush=True)


def main() -> None:
    sys.path.insert(0, FORK)
    import loop_utils.cube_run as CR
    CR.process_cubes = second_tie                # vggt_long.run() imports it at call time: it gets this one
    sys.argv = [FORK + "/vggt_long.py", *sys.argv[1:]]
    runpy.run_path(FORK + "/vggt_long.py", run_name="__main__")


if __name__ == "__main__":
    main()
