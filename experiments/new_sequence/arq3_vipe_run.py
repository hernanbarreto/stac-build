"""ViPE (NVIDIA's Video Pose Engine, installed on the pod's local disk: /root/vipe, env /root/envs/vipe-venv) on one
scan's stored frames, under the card's lease — the test of 2026-11-22 (USER: "ok, me parece bien, buffer stop sobre
vipe, y luego, amber + vipe"). Its own CLI and preset, nothing changed: `vipe infer --image-dir <frames> --pipeline
<preset> --output <out>`. Read only on the session.

    python arq3_vipe_run.py <scan dir> <out dir> [preset]"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
VIPE = Path("/root/vipe")
VENV = Path("/root/envs/vipe-venv")
TOOLS = Path("/root/envs/vipe-cu128")


def main() -> int:
    scan, out = Path(sys.argv[1]), Path(sys.argv[2])
    preset = sys.argv[3] if len(sys.argv) > 3 else "dav3"
    from card import card
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME="/workspace/hf_cache", HF_HUB_ENABLE_HF_TRANSFER="0", CUDA_HOME=str(TOOLS),
               PATH=f"{VENV}/bin:{TOOLS}/bin:" + os.environ.get("PATH", ""), OMP_NUM_THREADS="8")
    cmd = [str(VENV / "bin" / "vipe"), "infer", "--image-dir", str(scan / "frames"), "--pipeline", preset,
           "--output", str(out)]
    with card(f"ViPE ({preset}) on {scan.parent.parent.parent.name} (test)", session="pccr"):
        t0 = time.time()
        print(f"[vipe] {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(cmd, cwd=str(VIPE), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
        for line in p.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)
        rc = p.wait()
        print(f"[vipe] exit {rc} after {time.time() - t0:.0f} s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
