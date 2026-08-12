"""
PGSR precision-mode stage (precision task, Phase D) — runs AFTER CloudCompy
(it seeds the Gaussians from cleaned_cloud.ply) and BEFORE the TSDF (which then
integrates the rendered depths via depth_source "pgsr_render").

Active ONLY when reconstruction.backend == "vggtomega_pgsr"; for every other
backend the stage logs and no-ops (the stage list stays static, the backend
decides). The heavy training runs in the `pgsr` conda env via run_pgsr.sh
(compiled diff-plane-rasterization); this worker orchestrates:

    1. dynamic-class masks from the SAM3 stage artifacts (reconstruction/
       dynamic_masks.py) — people / mobile equipment excluded from the loss
    2. COLMAP-layout scene export (reconstruction/pgsr_export.py)
    3. the trainer subprocess (reconstruction/pgsr_train.py), logs streamed
    4. fail-fast verification of output/pgsr_render/ (depths + report)

Runs like every other stage via run(conn, session_dir, config).
"""
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

from workers.base import WorkerPipe, run_worker_safe, stop_semantic_service


def _pgsr_work(pipe: WorkerPipe, session_dir: str, config: dict):
    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()
    output_dir = (session_path / "output").resolve()
    recon_cfg = config.get("reconstruction", {}) or {}
    backend = str(recon_cfg.get("backend", "")).lower()
    if backend != "vggtomega_pgsr":
        pipe.send_log(f"PGSR stage: backend is '{backend}' (not vggtomega_pgsr) — "
                      f"nothing to do")
        pipe.send_progress(100, "PGSR skipped (backend)", stage="pgsr")
        return

    pcfg = recon_cfg.get("pgsr", {}) or {}
    cloud = output_dir / "cleaned_cloud.ply"
    if not cloud.exists():
        raise FileNotFoundError("cleaned_cloud.ply not found — PGSR seeds from the "
                                "cleaned cloud; run CloudCompy first")

    # PGSR needs the GPU to itself (native-res training) — same policy as Omega.
    if bool(pcfg.get("exclusive_gpu", True)):
        stop_semantic_service(pipe, stage="PGSR photometric optimization")

    # 1) dynamic masks from SAM3 artifacts (explicit no-op when absent)
    from reconstruction.dynamic_masks import generate as _gen_masks
    from reconstruction.pgsr_export import export_scene, SCENE_DIRNAME
    scene_dir = output_dir / SCENE_DIRNAME
    masks_dir = scene_dir / "masks"
    pipe.send_progress(5, "PGSR: dynamic masks from SAM3...", stage="pgsr")
    _gen_masks(output_dir, masks_dir, log=lambda m: pipe.send_log(f"[pgsr] {m}"))

    # 2) scene export (poses + intrinsics + seed cloud at native res)
    pipe.send_progress(10, "PGSR: exporting COLMAP scene...", stage="pgsr")
    export_scene(output_dir, frames_dir,
                 max_seed_pts=int(pcfg.get("max_seed_pts", 1_500_000)),
                 log=lambda m: pipe.send_log(f"[pgsr] {m}"))

    # 3) trainer subprocess (pgsr env)
    server_dir = Path(__file__).resolve().parent.parent
    cmd = ["bash", str(server_dir / "run_pgsr.sh"),
           "--scene", str(scene_dir),
           "--model_dir", str(output_dir / "pgsr_model"),
           "--render_dir", str(output_dir / "pgsr_render"),
           "--iterations", str(int(pcfg.get("iterations", 30000))),
           # vendor's published max-quality regime (see config.yaml pgsr:) —
           # validated on test2 2026-08-11 (30k in 1h57, PSNR 24.8, 11.3 GB peak)
           "--resolution", str(int(pcfg.get("resolution", 2))),
           "--ncc_scale", str(float(pcfg.get("ncc_scale", 0.5))),
           "--densify_abs_grad_threshold",
           str(float(pcfg.get("densify_abs_grad_threshold", 0.00015))),
           "--opacity_cull_threshold",
           str(float(pcfg.get("opacity_cull_threshold", 0.05)))]
    if bool(pcfg.get("exposure_compensation", True)):
        cmd.append("--exposure_compensation")
    if bool(pcfg.get("pose_refine", False)):
        cmd.append("--pose_refine")
    if bool(pcfg.get("quick", False)):
        cmd.append("--quick")
    pipe.send_progress(15, "PGSR: photometric optimization (native res)...",
                       stage="pgsr")
    pipe.send_log(f"[pgsr] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if pipe.check_cancel():
            proc.terminate()
            pipe.send_log("PGSR cancelled by user", level="warning")
            return
        pipe.send_log(f"[pgsr] {line}")
        if "] iter " in line:
            try:  # "[stac-pgsr] iter 12000/30000 ..."
                done, total = line.split("iter ")[1].split()[0].split("/")
                pipe.send_progress(15 + 75 * int(done) / max(int(total), 1),
                                   f"PGSR iter {done}/{total}", stage="pgsr")
            except Exception:
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"PGSR trainer exited with code {proc.returncode}")

    # 4) verify outputs — the TSDF depends on them (fail fast, never silent)
    render_dir = output_dir / "pgsr_render"
    n_depths = len(list(render_dir.glob("frame_*.npz")))
    if n_depths == 0 or not (render_dir / "report.json").exists():
        raise RuntimeError(f"PGSR produced no rendered depths in {render_dir} — "
                           f"the precision-mode TSDF cannot run")
    pipe.send_log(f"[pgsr] rendered depths: {n_depths} frames → {render_dir}")

    # 5) CONSISTENT CLOUD ("ir para atrás con la nube"): rebuild the user-facing
    # point cloud + Potree FROM the same verified depths the mesh integrates, so
    # cloud and mesh always agree. cleaned_cloud.ply stays untouched (it remains
    # the source for segmentation mapping / BIM).
    if bool(pcfg.get("consistent_cloud", True)):
        pipe.send_progress(92, "PGSR: rebuilding consistent cloud + Potree...",
                           stage="pgsr")
        from reconstruction.pgsr_cloud import build_cloud
        ply = build_cloud(output_dir, frames_dir,
                          stride=int(pcfg.get("cloud_stride", 2)),
                          voxel_m=float(pcfg.get("cloud_voxel_m", 0.006)),
                          log=lambda m: pipe.send_log(f"[pgsr] {m}"))
        from potree_converter import convert_ply_to_potree
        ok = convert_ply_to_potree(session_path, force=True, ply_override=ply)
        if not ok:
            raise RuntimeError("pgsr consistent cloud: Potree rebuild failed — "
                               "the viewer would show the stale ghosted cloud")
        pipe.send_log("[pgsr] Potree rebuilt from the consistent cloud ✓")
    pipe.send_progress(100, "PGSR complete", stage="pgsr")


def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_pgsr_work, conn, session_dir, config)
