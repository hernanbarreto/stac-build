"""THE NEW ORDER, step 3: F4 (the correspondences between images) on the test session, BEFORE anything is tied to
them — the pipeline's own module, unchanged, launched exactly as the pipeline's runner launches it (its interpreter,
its environment, its arguments).

    python run_f4.py <scan dir> <log file>"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from card import SERVER, card  # noqa: E402


def main() -> int:
    scan, log_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as fh:
        def log(m):
            fh.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {m}\n")
        from precision import runner as RN
        from precision.config import load_precision_config
        pcfg = load_precision_config()
        step = next(s for s in RN.STEPS if s.key == "f4_tracks")
        env = RN.step_env(RN.step_threads(step, pcfg.runner))
        py = {"da3": pcfg.runner.python_da3, "mapanything": pcfg.runner.python_mapanything}[step.env]
        cmd = [py, "-u", "-m", step.module, "--session", str(scan), *step.args]
        with card("the new order, F4 on bufferStop_cubos", log=log):
            log("run: " + " ".join(cmd[2:]))
            t0 = time.time()
            r = subprocess.run(cmd, cwd=SERVER, env=env, stdout=fh, stderr=subprocess.STDOUT)
            log(f"F4 exit {r.returncode} after {time.time() - t0:.0f} s")
        return r.returncode


if __name__ == "__main__":
    sys.exit(main())
