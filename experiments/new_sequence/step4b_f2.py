"""THE NEW ORDER, step 4 — "F2: dentro de cada cubo."

Each of the five cubes is laid out as a session of its own (cube_sessions.build: its own cameras, its own Omega
records, its own stretch of the walk) and the pipeline's F2 is run on it unchanged, launched as the pipeline's runner
launches the step (python -m precision.gauge --session <cube>). F2 measures the scale along the cube's walk against the
DA3 windows, applies it to the cube's cameras as that cube's next epoch, and leaves the tolerance F5 uses for its
anchors (gauge.json).

    python step4b_f2.py <session scan dir> <step-2 run dir (maplong_run)> <root for the cube sessions>"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cube_sessions as CS  # noqa: E402


def main() -> int:
    scan, run, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    from precision import runner as RN
    from precision.config import load_precision_config
    pcfg = load_precision_config(session_dir=scan)
    step = next(s for s in RN.STEPS if s.key == "f2_gauge")
    env = RN.step_env(RN.step_threads(step, pcfg.runner))
    sets, _owner, _names = CS.layout(run)
    for k in range(len(sets)):
        dst = root / f"cube_{k}"
        if not (dst / "output" / "gauge.json").exists():
            CS.build(run, scan, k, dst)
            cmd = [pcfg.runner.python_da3, "-u", "-m", step.module, "--session", str(dst), *step.args]
            print(f"[step 4] cube {k}: " + " ".join(cmd[2:]), flush=True)
            r = subprocess.run(cmd, cwd=CS.SERVER, env=env)
            if r.returncode:
                print(f"[step 4] cube {k}: F2 ended with code {r.returncode}", flush=True)
                return r.returncode
        g = json.loads((dst / "output" / "gauge.json").read_text())
        inst = g.get("applied_instrument")
        sig = (g.get("sigma_by_instrument") or {}).get(inst)
        print(f"[step 4] cube {k}: F2 applied {g.get('applied')} (epoch {g.get('epoch_to')}), instrument {inst}, "
              f"scale along the cube's walk x{g['s_keyframes']['min']:.4f} .. x{g['s_keyframes']['max']:.4f}, "
              f"cameras moved at most {g['camera_shift_max_m'] * 100:.2f} cm, tolerance (relative sigma) "
              f"{sig if sig is None else round(float(sig), 5)}, walk {g.get('walk_length_m'):.3f} m", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
