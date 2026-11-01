# STAC-Builder: Gaussian-splat worker (subprocess) — the REALISM LAYER at the end of the chain (USER
# 2026-10-31: "el realismo de los GS con la capacidad de medir sobre malla matemáticamente buena como la
# nuestra … entrenar los GS y superponerlos").
#
# In this (da3) process: the training set is built from the session with the chain's own readers
# (gaussians.dataset — the certified poses and camera, the published cloud as seeds with their cloud
# rows, the chain's per-keyframe depth) and written as plain arrays. Then env `gs` (torch + gsplat,
# gaussians.train) fits the splats with the cameras FIXED and writes the record into
# output/gaussians/<name>/ — gaussians.ply, gaussians.splat, seed_index.bin, report.json — and this
# process stamps meta.json (what the splats are of: the reconstruction id, the epochs, the cloud's
# point count and sha). The instance of every gaussian is NOT written here: the server resolves it
# live from the cloud's instance_ids.npy through seed_index.bin (gaussians.label).
#
# vLLM is not stopped: the fit needs 10–20 GB beside it. The training set is deleted when the fit is
# through (a session keeps what the next reader needs).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
import json
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Optional

from workers.base import WorkerPipe, run_worker_safe

STAGE = "gaussians"
META_NAME = "meta.json"
TRAINSET_DIRNAME = "_trainset"
_PROGRESS = re.compile(r"\[gs\] it (\d+)/(\d+) \(([\d.]+)%\)")
_EVAL = re.compile(r"\[gs\] eval it (\d+): psnr ([\d.]+)")


def parse_progress(line: str) -> Optional[float]:
    """The fit's share of the stage (the fit is 10 → 95 % of it) from a ``[gs] it i/n (p%)`` line."""
    m = _PROGRESS.search(line)
    if not m:
        return None
    return 10.0 + 85.0 * float(m.group(3)) / 100.0


def _gaussians_work(pipe: WorkerPipe, session_dir: str, config: dict):
    from gaussians.config import load_gaussians_config
    from gaussians.dataset import TRAINSET_NAME, build_training_set, write_trainset
    import repro

    session_path = Path(session_dir).resolve()
    output_dir = session_path / "output"
    if not (output_dir / "cleaned_cloud.ply").exists():
        raise FileNotFoundError("cleaned_cloud.ply not found — no published cloud; the gaussians need it")
    cfg = load_gaussians_config(config)
    py = Path(cfg.python)
    if not py.exists():
        raise FileNotFoundError(f"gaussians.python = {py} does not exist (env gs: torch + gsplat)")
    out_dir = output_dir / "gaussians" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    trainset_dir = out_dir / TRAINSET_DIRNAME
    shutil.rmtree(trainset_dir, ignore_errors=True)
    t0 = time.time()
    pipe.send_progress(1, "building the training set from the certified session", stage=STAGE)
    ts, loader = build_training_set(session_path, cfg, log=lambda m: pipe.send_log(str(m)))
    write_trainset(ts, trainset_dir, loader, log=lambda m: pipe.send_log(str(m)))
    tcfg_path = trainset_dir / "train_config.json"
    d = cfg.to_dict()
    tcfg_path.write_text(json.dumps({k: d[k] for k in ("init", "train", "export", "label")}, indent=1))
    pipe.send_progress(10, f"fitting the splats ({ts.n_seeds:,} seeds, {ts.n_frames} keyframes) in env gs", stage=STAGE)
    server_dir = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(server_dir)}
    cmd = [str(py), "-m", "gaussians.train", "--trainset", str(trainset_dir), "--out", str(out_dir), "--config", str(tcfg_path)]
    pipe.send_log(f"[gaussians] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(server_dir), env=env)
    assert proc.stdout is not None
    last_pct = 10.0
    tail = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        tail.append(line[:400]); tail = tail[-40:]
        p = parse_progress(line)
        if p is not None and p > last_pct:
            last_pct = p
            pipe.send_progress(p, line.split("[gs] ", 1)[-1][:120], stage=STAGE)
        if _EVAL.search(line) or line.startswith("[gs] done") or "Traceback" in line or "Error" in line:
            pipe.send_log(f"[gaussians] {line[:300]}")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"gaussians.train failed (exit {code}) — last lines: " + " | ".join(tail[-6:]))
    report_p = out_dir / "report.json"
    if not report_p.exists():
        raise RuntimeError(f"gaussians.train ended without {report_p}")
    report = json.loads(report_p.read_text())
    from correction.epoch import RECONSTRUCTION_ID_KEY
    meta = {"version": 1, "stage": STAGE, "name": cfg.name, "provenance": "tool_measured",
            "files": report.get("files", {}), "n_gaussians": report.get("n_gaussians"), "n_seeds": report.get("n_seeds"),
            "heldout": report.get("heldout"),
            "cloud": {"n_points": ts.meta.get("n_cloud_points"),
                      "sha256": repro.sha256_file(output_dir / "cleaned_cloud.ply")},
            RECONSTRUCTION_ID_KEY: ts.meta.get(RECONSTRUCTION_ID_KEY),
            "geometry_epoch": ts.meta.get("geometry_epoch"), "camera_epoch": ts.meta.get("camera_epoch"),
            "seconds": round(time.time() - t0, 1),
            "reading": "the splats are what is seen; the cloud and the mesh measure. The instance of a gaussian is the "
                       "cloud's current instance_ids.npy at its seed's row (seed_index.bin) — the Segmentation Manager's "
                       "edits apply live; a cloud with another point count makes the mapping stale"}
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=1, default=str))
    shutil.rmtree(trainset_dir, ignore_errors=True)
    hv = report.get("heldout") or {}
    pipe.send_progress(100, f"gaussians published: {cfg.name} ({report.get('n_gaussians', '?'):,} gaussians, held-out PSNR "
                            f"{hv.get('psnr_mean', float('nan')):.2f})", stage=STAGE)
    pipe.send_log(f"[gaussians] {out_dir / META_NAME}")


def run(conn: Connection, session_dir: str, config: dict):
    run_worker_safe(_gaussians_work, conn, session_dir, config)


if __name__ == "__main__":
    print("gaussians_worker is spawned by the pipeline manager", file=sys.stderr)
    sys.exit(2)
