"""ViPE with MetricTrail integrated, the whole battery (USER 2026-11-22: "lo quiero correr con toda la bateria
funcionando" / "A"): `run.py pipeline=metrictrail` with the global BA preset `metrictrail_gba_refine` and the focal
refinement (RoMa) on; ViPE's dynamic-object masks (`pipeline.init.instance={}`), its depth fusion
(`pipeline.post.depth_align_model=…`) and any other override only when given after the two paths. Then MetricTrail's
own point cloud (`vipe.metrictrail.pointcloud --filter strict`, every pixel, 5 mm), outside the card's lease. Read only
on the session. ViPE: /root/vipe, env /root/envs/vipe-venv (the pod's local disk).

    python arq3_vipe_mt_run.py <frames dir (only the frames; its name = the sequence)> <out dir> [override …]"""
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
DPVO = "/workspace/hf_cache/hub/models--nvidia--MetricTrail/snapshots/f5a3ce1cebd2a6a35c30ffb138caa07122731fff/dino_dpvo.pth"
DINOV3 = "/workspace/tmp/new_sequence/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def run(cmd, env, cwd):
    print(f"[vipe-mt] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1)
    for line in p.stdout:
        line = line.rstrip()
        if line:
            print(line, flush=True)
    rc = p.wait()
    print(f"[vipe-mt] exit {rc} after {time.time() - t0:.0f} s", flush=True)
    return rc


def main() -> int:
    frames, out = Path(sys.argv[1]), Path(sys.argv[2])
    extra = sys.argv[3:]                                    # more of run.py's overrides (e.g. the input raster)
    from card import card
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME="/workspace/hf_cache", HF_HUB_ENABLE_HF_TRANSFER="0", CUDA_HOME=str(TOOLS),
               PATH=f"{VENV}/bin:{TOOLS}/bin:" + os.environ.get("PATH", ""), OMP_NUM_THREADS="8",
               VIPE_METRICTRAIL_DPVO_CHECKPOINT=DPVO, VIPE_DINOV3_VITS16_CHECKPOINT=DINOV3,
               VIPE_METRICTRAIL_ROMA_PYTHON=str(VENV / "bin" / "python"),
               VIPE_METRICTRAIL_ROMA_CHECKPOINTS="/workspace/tmp/new_sequence/weights/roma")
    if os.environ.get("STAC_VIPE_PYTHONPATH"):      # e.g. MoGe-3's code and its two dependencies, for ViPE's processes
        env["PYTHONPATH"] = os.environ["STAC_VIPE_PYTHONPATH"]
    py = str(VENV / "bin" / "python")
    # ViPE's dynamic masks and its depth fusion only when given among the extra overrides (USER 2026-11-22: "okok,
    # sin mascaras dale" for the MoGe-3 run with the static-scene fusion)
    track = [py, "run.py", "pipeline=metrictrail", "streams=frame_dir_stream", f"streams.base_path={frames}",
             "slam@pipeline.slam=metrictrail_gba_refine", "pipeline.focal_refine.enabled=true",
             f"pipeline.output.path={out}"] + extra
    # its own export, the strict flying-point filter, every pixel, 5 mm voxels, no practical point budget
    # (USER 2026-11-22: "si" to "con todos los píxeles, vóxel de 5 mm como el nuestro y sin tope práctico de puntos")
    cloud = [py, "-m", "vipe.metrictrail.pointcloud", str(out), "--name", frames.name, "--frames", str(frames),
             "--filter", "strict", "--pixel-stride", "1", "--voxel-size", "0.005", "--max-input-points", "1000000000"]
    with card(f"ViPE + MetricTrail (gba_refine, whole battery) on {frames.name} (test)", session="pccr"):
        rc = run(track, env, VIPE)
    if rc == 0:                     # the export is numpy on the CPU: outside the card's lease
        rc = run(cloud, env, VIPE)
    return rc


if __name__ == "__main__":
    sys.exit(main())
