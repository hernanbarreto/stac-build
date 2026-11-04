"""The fork's sky masks computed BEFORE Omega, in parallel, while the card runs I3 (USER 2026-11-01:
"si lancé reconstruir, la GPU no debe quedar parada nunca hasta terminar").

MEASURED on station 2026-11-01 (analysis/2026-11-01_station_gpu_run): Omega masks every chunk's
frames with the pinned skyseg.onnx on onnxruntime's CPU provider with ONE thread
(vendor/VGGT-Long/loop_utils/sky_mask.py: one kernel order on every machine), a few seconds per
keyframe, and the card waits for it before every chunk — most of each minute of Omega with the
card at 0 %.

A frame's mask depends only on what the fork's cache is keyed on — the model, the segmenting code,
the image's bytes (``<cache root>/<model sha>-<code sha>/<frame>_<image sha>.png``) — so it can be
computed before Omega by the SAME code: :func:`start` runs the fork's own functions (skyseg_path,
cpu_session, segment_sky, mask_cache_dir, mask_path) on the SAME interpreter Omega runs on
(``reconstruction.precision.runner.python_mapanything``: its onnxruntime is the one whose kernels
make the mask), from the fork's directory (Omega's own import context), in ``workers`` processes of
one onnxruntime thread each, and every mask is written under its cache name ATOMICALLY (a temporary
PNG renamed into place: a reader sees no mask or the whole mask). Omega then finds every keyframe's
mask and reuses it — ``apply_sky_masks`` reads every mask back from its PNG, fresh or cached, one
path. The frames a loop bridge adds between keyframes are not known here; Omega computes those.

The masks are the bytes Omega writes (tests/test_sky_precompute.py; on station's real keyframes,
byte for byte against the masks Omega computed itself). A worker that fails FAILS the stage, as the
fork's own masking does: a frame left unmasked is another cloud.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# processes of ONE onnxruntime thread each (the fork's cpu_session): the pod's cgroup gives ~26
# CPUs and I3 (DA3 on the card) needs few of them; the mask of a frame does not depend on how many
# run (each is the fork's single-threaded session on one image)
WORKERS = 12


def keyframe_paths(frames_dir, selected_files: Sequence[str], stride: int = 1) -> List[str]:
    """The images Omega reads, in its order (vggt_long.VGGT_Long.run): the frames directory's
    *.jpg + *.png sorted, restricted to ``selected_files`` when that list is not empty, then one in
    ``stride``."""
    d = str(frames_dir)
    paths = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
    sel = set(selected_files)
    if sel:
        paths = [p for p in paths if os.path.basename(p) in sel]
    stride = int(stride or 1)
    return paths[::stride] if stride > 1 else paths


def omega_python() -> str:
    """The interpreter server/run_mapanything.sh runs Omega on, by the launcher's own rule:
    STAC_PYTHON_MAPANYTHING (the pipeline manager exports it from the job's frozen
    reconstruction.precision.runner.python_mapanything), else the conda env's python under
    CONDA_ROOT. It must exist (the launcher fails the same way)."""
    py = os.environ.get("STAC_PYTHON_MAPANYTHING") or os.path.join(
        os.environ.get("CONDA_ROOT", "/workspace/miniforge3"), "envs",
        os.environ.get("MAPANYTHING_CONDA_ENV", "mapanything"), "bin", "python")
    if not os.access(py, os.X_OK):
        raise RuntimeError(f"[sky-precompute] Omega's interpreter {py} does not exist or is not "
                           f"executable (STAC_PYTHON_MAPANYTHING / reconstruction.precision.runner."
                           f"python_mapanything) — the masks must come from Omega's own onnxruntime")
    return py


class SkyPrecompute:
    """The running workers: :meth:`wait` joins them (a failure raises with the worker's log),
    :meth:`kill` stops them (cancel, an error elsewhere)."""

    def __init__(self, procs, logs: List[Path], n_frames: int, t0: float, log: Callable):
        self.procs, self.logs, self.n_frames, self.t0, self.log = procs, logs, n_frames, t0, log

    def wait(self, check_cancel: Optional[Callable[[], object]] = None, poll_s: float = 0.5) -> Dict:
        try:
            while any(p.poll() is None for p in self.procs):
                if check_cancel is not None:
                    check_cancel()
                time.sleep(poll_s)
        except BaseException:
            self.kill()
            raise
        bad = [(k, p.returncode) for k, p in enumerate(self.procs) if p.returncode != 0]
        if bad:
            k, rc = bad[0]
            tail = self.logs[k].read_text(errors="replace")[-2000:] if self.logs[k].exists() else ""
            raise RuntimeError(f"[sky-precompute] worker {k} exited {rc} — the sky masks of this "
                               f"run could not be computed:\n{tail}")
        done = {"computed": 0, "reused": 0}
        for lp in self.logs:
            for line in lp.read_text(errors="replace").splitlines():
                if line.startswith("{") and '"computed"' in line:
                    rec = json.loads(line)
                    done["computed"] += int(rec["computed"])
                    done["reused"] += int(rec["reused"])
        sec = time.time() - self.t0
        self.log(f"[sky-precompute] {self.n_frames} keyframe sky masks ready in {sec:.0f} s "
                 f"({done['computed']} computed, {done['reused']} already cached; "
                 f"{len(self.procs)} processes of one onnxruntime thread)")
        return {**done, "seconds": round(sec, 1), "workers": len(self.procs)}

    def kill(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def start(image_paths: Sequence[str], cache_root, fork_dir, python: str, env: Optional[dict] = None,
          workers: int = WORKERS, log: Callable = print) -> Optional[SkyPrecompute]:
    """Launch the workers over ``image_paths`` (worker k takes every ``workers``-th image from k, so
    each covers the whole walk) and return at once. ``cache_root``: Omega's own
    ``<save_dir>/sky_masks``; ``fork_dir``: vendor/VGGT-Long; ``python``: Omega's interpreter."""
    paths = [os.path.abspath(p) for p in image_paths]
    if not paths:
        return None
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    work = root / ".precompute"
    work.mkdir(exist_ok=True)
    lst = work / "frames.json"
    lst.write_text(json.dumps(paths))
    n = max(1, min(int(workers), len(paths)))
    penv = dict(os.environ if env is None else env)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        penv[k] = "1"
    penv["PYTHONUNBUFFERED"] = "1"
    procs, logs = [], []
    for k in range(n):
        lp = work / f"worker_{k}.log"
        fh = open(lp, "w")
        cmd = [str(python), "-u", os.path.abspath(__file__), "--fork", str(fork_dir), "--cache-root",
               str(root), "--list", str(lst), "--k", str(k), "--n", str(n)]
        # in the stage's process group (workers/base.py setsid()s the stage): a cancel or the
        # stage's death kills these with it
        procs.append(subprocess.Popen(cmd, cwd=str(fork_dir), env=penv, stdout=fh,
                                      stderr=subprocess.STDOUT))
        fh.close()
        logs.append(lp)
    log(f"[sky-precompute] {len(paths)} frames' sky masks (keyframes + context views) started on {n} processes "
        f"({python}) while the card runs — Omega reuses them from {root}")
    return SkyPrecompute(procs, logs, len(paths), time.time(), log)


def _worker(fork_dir: str, cache_root: str, list_file: str, k: int, n: int) -> None:
    # Omega's import context: vggt_long.py runs from the fork's directory, first on sys.path
    sys.path[0] = os.path.abspath(fork_dir)
    import cv2
    cv2.setNumThreads(1)
    from loop_utils.sky_mask import (cpu_session, mask_cache_dir, mask_path, segmenter_code_file,
                                     skyseg_path, _sha256_cached)
    from loop_utils.stac_repro import repro
    from loop_utils.visual_util import segment_sky
    model = skyseg_path()                                 # the pinned model, verified
    cdir = mask_cache_dir(cache_root, _sha256_cached(model), repro().sha256_file(segmenter_code_file()))
    os.makedirs(cdir, exist_ok=True)
    paths = json.load(open(list_file))[k::n]
    session = None
    computed = reused = 0
    t0 = time.time()
    for p in paths:
        mp = mask_path(cdir, p)
        if os.path.exists(mp):
            reused += 1
            continue
        if session is None:
            session = cpu_session(model)
        tmp = f"{mp[:-4]}.tmp{os.getpid()}.png"           # cv2 picks the encoder by the extension
        segment_sky(p, session, tmp)
        os.replace(tmp, mp)
        computed += 1
    print(json.dumps({"worker": k, "computed": computed, "reused": reused,
                      "seconds": round(time.time() - t0, 1), "cache_dir": cdir}), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    _worker(a.fork, a.cache_root, a.list, a.k, a.n)
