"""AMB3R-SLAM's run.py under the pipeline's deterministic numerics (repro.enable_deterministic_torch: STRICT — an op with
no deterministic kernel raises — cuDNN deterministic, no TF32, seeds), run with AMB3R's own interpreter (env amb3r) from
vendor/AMB3R-SLAM by reconstruction/amb3r_probe.py:

  python amb3r_launch.py <run.py arguments>

The environment comes from repro.deterministic_env() (the cuBLAS workspace pinned before CUDA starts)."""
import os
import runpy
import sys

SERVER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMB3R = os.path.join(os.path.dirname(SERVER), "vendor", "AMB3R-SLAM")

if __name__ == "__main__":
    if not os.path.isfile(os.path.join(AMB3R, "run.py")):
        raise SystemExit(f"{AMB3R}/run.py not found — AMB3R-SLAM is not installed where the pipeline expects it")
    sys.path.insert(0, AMB3R)
    sys.path.append(SERVER)
    import repro
    rec = repro.enable_deterministic_torch(0)
    print(f"[amb3r-launch] deterministic torch: {rec}", flush=True)
    sys.argv = [os.path.join(AMB3R, "run.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(AMB3R, "run.py"), run_name="__main__")
