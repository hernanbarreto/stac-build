#!/usr/bin/env python3
"""
CGAL WLOP satellite — runs in the CloudComPy310 conda env (the only one with
CGAL Python bindings). Launched by run_wlop.sh / surface_fit.consolidate.

WLOP (Weighted Locally Optimal Projection, Huang 2009) PROJECTS a noisy,
multi-layered point set onto a thin, locally uniform surface. It does not
invent geometry: every output point is an average of measured points, so the
output is still measurement — exactly the stage-1 consolidation the pipeline
charter requires ("colapsa capas múltiples a una superficie delgada").

Protocol (stdout):
    [WLOP]{"n_in": ..., "n_out": ..., "elapsed": ...}
    [WLOP-RESULT]<output_path>      (or [WLOP-RESULT]NONE on failure)
"""
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="CGAL WLOP consolidation")
    ap.add_argument("--input", required=True, help="input point cloud (.ply/.xyz/.off)")
    ap.add_argument("--output", required=True, help="output path (.ply/.xyz)")
    ap.add_argument("--select-percentage", type=float, default=25.0,
                    help="output size as %% of input (WLOP also downsamples)")
    ap.add_argument("--neighbor-radius", type=float, default=0.06,
                    help="WLOP neighborhood radius (m); must exceed the layer separation")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--uniform", action="store_true",
                    help="require uniform sampling (slower, denser gaps)")
    args = ap.parse_args()

    try:
        from CGAL.CGAL_Point_set_3 import Point_set_3
        from CGAL import CGAL_Point_set_processing_3 as psp
    except Exception as e:
        print(f"[WLOP] CGAL bindings unavailable: {e}", flush=True)
        print("[WLOP-RESULT]NONE", flush=True)
        sys.exit(2)

    t0 = time.time()
    ps = Point_set_3(args.input)
    n_in = ps.size()
    if n_in == 0:
        print(f"[WLOP] empty/unreadable input {args.input}", flush=True)
        print("[WLOP-RESULT]NONE", flush=True)
        sys.exit(1)
    out = Point_set_3()
    psp.wlop_simplify_and_regularize_point_set(
        ps, out, float(args.select_percentage), float(args.neighbor_radius),
        int(args.iterations), bool(args.uniform))
    out.write(args.output)
    print("[WLOP]" + json.dumps({"n_in": int(n_in), "n_out": int(out.size()),
                                 "elapsed": round(time.time() - t0, 2)}), flush=True)
    print("[WLOP-RESULT]" + args.output, flush=True)


if __name__ == "__main__":
    main()
