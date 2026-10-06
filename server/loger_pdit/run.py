"""Orchestrator of the separate LoGeR + PointDiT + DA3 path (see __init__)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
ROOT = SERVER.parent
PY = "/workspace/miniforge3/envs/da3/bin/python"
CKPT = ROOT / "vendor" / "LoGeR" / "ckpts" / "LoGeR_star" / "latest.pt"
CFG = ROOT / "vendor" / "LoGeR" / "ckpts" / "LoGeR_star" / "original_config.yaml"


def keyframes(session: Path, stride: int):
    sel = session / "frames" / "selected_frames.json"
    imgs = sorted(p for p in (session / "frames").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if sel.exists() and stride <= 0:
        files = json.loads(sel.read_text())["selected_files"]
        return [session / "frames" / f for f in files], "intake keyframes"
    st = stride if stride > 0 else 10
    return imgs[::st], f"every {st}th frame"


def sh(cmd, log):
    log("$ " + " ".join(str(c) for c in cmd[:4]) + " ...")
    p = subprocess.Popen([str(c) for c in cmd], cwd=str(SERVER), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1", HF_HOME="/workspace/hf_cache",
                                  HF_HUB_ENABLE_HF_TRANSFER="0"))   # the backend's own HF setup (scripts/start.sh)
    for line in p.stdout:
        line = line.rstrip()
        if line and ("[" in line[:12] or "Error" in line or "Traceback" in line):
            log(line)
    if p.wait() != 0:
        raise RuntimeError(f"step failed: {' '.join(str(c) for c in cmd[:4])} (exit {p.returncode})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loger_pdit.run")
    ap.add_argument("--session", required=True, help="scan dir (…/scans/<date>/src_<source>)")
    ap.add_argument("--stride", type=int, default=0, help="0 = intake keyframes when present, else every Nth frame")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=3)
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    session = Path(a.session).resolve()
    out = session / "output" / "loger_pdit"; out.mkdir(parents=True, exist_ok=True)
    kf, how = keyframes(session, a.stride)
    (out / "images.txt").write_text("\n".join(str(p) for p in kf) + "\n")
    log(f"[run] {len(kf)} keyframe(s) ({how}) → {out}")
    reset = 5 if len(kf) > 1000 else 0          # LoGeR's own note: state resets past ~1k frames
    if not (out / "loger.npz").exists():
        sh([PY, "-m", "loger_pdit.loger_infer", "--images", out / "images.txt", "--out", out / "loger.npz",
            "--ckpt", CKPT, "--config", CFG, "--window", a.window, "--overlap", a.overlap,
            "--reset_every", reset], log)
    kfdir = out / "kf_images"; kfdir.mkdir(exist_ok=True)
    for p in kf:
        l = kfdir / p.name
        if l.is_symlink() and not l.exists():
            l.unlink()                       # a broken link from an earlier run
        if not l.exists():
            l.symlink_to(p.resolve())
    sh([PY, str(SERVER / "extract_da3_depth.py"), "--image_dir", kfdir, "--output_dir", out / "da3", "--per_frame"], log)
    sh([PY, "-m", "loger_pdit.pdit_maps", "--images", out / "images.txt", "--out_dir", out / "pointdit"], log)
    from loger_pdit.fuse import fuse
    fuse(out, session / "frames", log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
