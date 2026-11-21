"""Runs ONE step of the experiment on the card, the way the pipeline's runner runs a step: the card's lease taken (the
chat's engine and the voice stopped, given back at the end), the step's interpreter and the runner's own environment
(deterministic numerics, the BLAS pin).

    python run_on_card.py <log file> <label> <da3|mapanything> <pipeline step whose environment is used> -- <args ...>

e.g.  run_on_card.py f5.log "F5 per cube" mapanything f5_refine -- /path/f5_per_cube.py <scan> ..."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from card import SERVER, card  # noqa: E402


def main() -> int:
    log_path, label, env_name, step_key = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    assert sys.argv[5] == "--", "usage: run_on_card.py <log> <label> <env> <step> -- <args ...>"
    args = sys.argv[6:]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as fh:
        def log(m):
            fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {m}\n")
        from precision import runner as RN
        from precision.config import load_precision_config
        pcfg = load_precision_config()
        step = next(s for s in RN.STEPS if s.key == step_key)
        env = RN.step_env(RN.step_threads(step, pcfg.runner))
        py = {"da3": pcfg.runner.python_da3, "mapanything": pcfg.runner.python_mapanything}[env_name]
        cmd = [py, "-u", *args]
        import os
        with card(f"the new order: {label}", session=os.environ.get("NEW_SEQUENCE_SESSION", "bufferStop_cubos"), log=log):
            log("run: " + " ".join(Path(a).name if a.startswith("/") else a for a in args))
            t0 = time.time()
            r = subprocess.run(cmd, cwd=SERVER, env=env, stdout=fh, stderr=subprocess.STDOUT)
            log(f"{label}: exit {r.returncode} after {time.time() - t0:.0f} s")
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
