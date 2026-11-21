"""Launches a step that runs inside the fork (vendor/VGGT-Long) the way the pipeline's map worker launches Omega
(workers/map_worker.py _omega_pass + server/run_mapanything.sh): the mapanything interpreter with its conda
activation, the fork's directory as the working directory, repro.deterministic_env(), 8 threads for every CPU library,
MKL in its reproducible mode, the one Hugging Face cache — with the card's lease taken (card.py).

    python run_omega_step.py <log file> <label> <driver.py> -- <the fork's own arguments ...>"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from card import SERVER, card  # noqa: E402

FORK = "/workspace/stac-build/vendor/VGGT-Long"
CONDA_ROOT = "/workspace/miniforge3"


def main() -> int:
    log_path, label, driver = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    assert sys.argv[4] == "--", "usage: run_omega_step.py <log> <label> <driver.py> -- <args ...>"
    args = sys.argv[5:]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as fh:
        def log(m):
            fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {m}\n")
        import da3_weights
        import repro
        from precision.config import load_precision_config
        py = load_precision_config().runner.python_mapanything
        env = repro.deterministic_env()
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = "8"
        env["MKL_CBWR"] = "COMPATIBLE"
        env["HF_HOME"] = da3_weights.HF_HOME
        env["PYTHONUNBUFFERED"] = "1"
        import os as _os
        env.update({k: v for k, v in _os.environ.items() if k.startswith("V2_")})   # a driver's own inputs
        shell = (f'source "{CONDA_ROOT}/etc/profile.d/conda.sh" && conda activate mapanything && cd "{FORK}" && '
                 f'exec "{py}" -u "$0" "$@"')
        import os
        with card(f"the new order: {label}", session=os.environ.get("NEW_SEQUENCE_SESSION", "bufferStop_cubos"), log=log):
            log("run: " + Path(driver).name + " " + " ".join(Path(a).name if a.startswith("/") else a for a in args))
            t0 = time.time()
            r = subprocess.run(["bash", "-c", shell, driver, *args], cwd=FORK, env=env, stdout=fh,
                               stderr=subprocess.STDOUT)
            log(f"{label}: exit {r.returncode} after {time.time() - t0:.0f} s")
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
