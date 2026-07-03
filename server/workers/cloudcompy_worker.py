# STAC-Builder: CloudCompy Worker (Subprocess)
# Runs CloudCompPy post-processing as a shell subprocess.
# Reads chunk PLYs, writes cleaned_cloud.ply.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import subprocess
import re
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _cloudcompy_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """CloudCompy cleaning — runs inside a dedicated subprocess."""

    session_path = Path(session_dir)
    output_dir = (session_path / "output").resolve()
    output_ply = output_dir / "cleaned_cloud.ply"

    server_dir = Path(__file__).resolve().parent.parent
    script_path = server_dir / "run_cloudcompy.sh"

    postproc = config.get("postprocessing", {})
    voxel_size = postproc.get("voxel_size", 0.001)

    # NO FALLBACK: CloudCompy produces cleaned_cloud.ply, mandatory for the TSDF/viewers.
    if not script_path.exists():
        raise FileNotFoundError(f"run_cloudcompy.sh not found at {script_path}")

    chunks = sorted(output_dir.glob("chunk_*.ply"))
    if not chunks:
        raise RuntimeError(f"No chunk_*.ply in {output_dir} — reconstruction produced no cloud")

    # For legacy sessions, check old name as well (new sessions use chunk_999_lidar.ply)
    for ext_cloud in ["lidar_complement.ply", "chunk_999_lidar.ply"]:
        ext_path = output_dir / ext_cloud
        if ext_path.exists() and ext_path not in chunks:
            chunks.append(ext_path)
            pipe.send_log(f"Including {ext_cloud} in merge ({ext_path.stat().st_size / 1048576:.0f} MB)")

    pipe.send_progress(0, f"Cleaning {len(chunks)} clouds (voxel={voxel_size*1000:.1f}mm)",
                       stage="cloudcompy")

    cmd = [
        "bash", str(script_path),
        "--input-dir", str(output_dir),
        "--output", str(output_ply),
        "--voxel-size", str(voxel_size),
        "--sor-knn", str(postproc.get("sor_knn", 6)),
        "--sor-sigma", str(postproc.get("sor_sigma", 1.0)),
        "--noise-radius", str(postproc.get("noise_radius", 0.01)),
        "--noise-sigma", str(postproc.get("noise_sigma", 1.0)),
        "--conf-min-norm", str(postproc.get("conf_min_norm", 0.0)),
    ]
    max_points = postproc.get("max_points", 0)
    if max_points > 0:
        cmd.extend(["--max-points", str(max_points)])
    for flag in ("skip_duplicates", "skip_sor", "skip_noise", "skip_normals"):
        if postproc.get(flag, False):
            cmd.append(f"--{flag.replace('_', '-')}")

    pipe.send_progress(5, "Running CloudCompPy...", stage="cloudcompy")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Parse progress from CloudCompy stdout
    # Look for lines like "[Step 2/5]" or percentage patterns
    step_pattern = re.compile(r"\[Step\s+(\d+)/(\d+)\]")
    pct_pattern = re.compile(r"(\d+(?:\.\d+)?)%")
    line_count = 0

    for line in iter(proc.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue

        line_count += 1
        pipe.send_log(line)

        # Try to extract progress
        m = step_pattern.search(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            pct = 5 + (step / total) * 90
            pipe.send_progress(pct, line, stage="cloudcompy")
        else:
            m2 = pct_pattern.search(line)
            if m2:
                pipe.send_progress(float(m2.group(1)), line, stage="cloudcompy")
            elif line_count % 5 == 0:
                # Report progress every 5 lines even without patterns
                estimated_pct = min(5 + line_count * 2, 90)
                pipe.send_progress(estimated_pct, line, stage="cloudcompy")

        if pipe.check_cancel():
            proc.terminate()
            proc.wait()
            pipe.send_log("Cancelled by user", level="warning")
            return

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"CloudCompy script exited with code {rc}")

    if output_ply.exists():
        size_mb = output_ply.stat().st_size / (1024 * 1024)
        pipe.send_log(f"Cleaned cloud: {size_mb:.1f} MB")

        # Chunks are now redundant — their points + per-point traceability are
        # baked into cleaned_cloud.ply, and nothing downstream (Potree, TSDF,
        # segmentation) reads the chunks. Delete them (gate to keep if ever
        # re-running ONLY cloudcompy without re-running reconstruction).
        if config.get("postprocessing", {}).get("delete_chunks_after_merge", True):
            removed = 0
            for pat in ("chunk_*.ply", "chunk_*_origins.npz", "chunk_*_meta.json"):
                for f in output_dir.glob(pat):
                    try:
                        f.unlink(); removed += 1
                    except Exception:
                        pass
            if removed:
                pipe.send_log(f"[cleanup] removed {removed} redundant chunk files (baked into cleaned_cloud)")

        # ── Link to project merged/ dir (new-style projects) ──
        # SourceContext.merged_cloud expects merged/merged_cloud.ply
        # but the pipeline writes to src_xxx/output/cleaned_cloud.ply.
        # Create a symlink so the sidebar and viewers find the cloud.
        try:
            import os
            # Walk up from session_dir to find project root with project.json
            project_root = session_path
            for _ in range(5):  # max depth guard
                project_root = project_root.parent
                if (project_root / "project.json").exists():
                    break
            else:
                project_root = None

            if project_root and (project_root / "project.json").exists():
                merged_dir = project_root / "merged"
                merged_dir.mkdir(parents=True, exist_ok=True)
                merged_cloud = merged_dir / "merged_cloud.ply"
                # Remove existing symlink/file
                if merged_cloud.exists() or merged_cloud.is_symlink():
                    merged_cloud.unlink()
                # Create relative symlink
                rel_path = os.path.relpath(str(output_ply), str(merged_dir))
                os.symlink(rel_path, str(merged_cloud))
                pipe.send_log(f"Linked merged_cloud.ply → {rel_path}")

                # Also link floor_transform if present
                floor_src = output_dir / "floor_transform.npz"
                floor_dst = merged_dir / "floor_transform.npz"
                if floor_src.exists():
                    if floor_dst.exists() or floor_dst.is_symlink():
                        floor_dst.unlink()
                    rel_ft = os.path.relpath(str(floor_src), str(merged_dir))
                    os.symlink(rel_ft, str(floor_dst))
        except Exception as e:
            pipe.send_log(f"Merged dir symlink failed (non-critical): {e}", level="warning")

        # ── Scene-level consolidation (surface_fit stage 1 at scene scale) ──
        # Normal-aware robust MLS over cleaned_cloud IN PLACE: collapses the
        # residual onion layers fine_register left (<δ) into thin surfaces so
        # TSDF masking, Potree and segmentation all see clean geometry. Point
        # count/order (→ colors, globalIndices) are preserved; the untouched
        # measurement is kept as cleaned_cloud_raw.ply (metric reference for
        # surface_fit residuals). Radius adapts to fine_register_report.json.
        sc_cfg = postproc.get("scene_consolidate", {}) or {}
        if sc_cfg.get("enabled", True):
            pipe.send_progress(93, "Consolidating cloud (normal-aware MLS)...",
                               stage="cloudcompy")
            try:
                import sys
                server_dir_str = str(Path(__file__).resolve().parent.parent)
                if server_dir_str not in sys.path:
                    sys.path.insert(0, server_dir_str)
                from reconstruction.surface_fit.consolidate import scene_consolidate
                stats = scene_consolidate(
                    output_dir,
                    radius_m=sc_cfg.get("radius_m"),
                    min_radius_m=float(sc_cfg.get("min_radius_m", 0.02)),
                    max_radius_m=float(sc_cfg.get("max_radius_m", 0.06)),
                    iterations=int(sc_cfg.get("iterations", 2)),
                    normal_gate=float(sc_cfg.get("normal_gate", 0.25)),
                )
                if stats:
                    pipe.send_log(
                        f"[consolidate] {stats['n_points']:,} pts, r={stats['radius_m']:.3f}m, "
                        f"mean move {stats['mean_move_mm']:.2f}mm (p95 {stats['p95_move_mm']:.2f}mm)")
            except Exception as e:
                pipe.send_log(f"[consolidate] scene consolidation failed "
                              f"(non-fatal, cloud kept raw): {e}", level="warning")

        # Compute and save floor alignment transform
        pipe.send_progress(95, "Computing floor alignment...", stage="cloudcompy")
        try:
            import sys
            server_dir_str = str(Path(__file__).resolve().parent.parent)
            if server_dir_str not in sys.path:
                sys.path.insert(0, server_dir_str)

            import numpy as np
            from plyfile import PlyData

            plydata = PlyData.read(str(output_ply))
            vx = plydata['vertex']
            xyz = np.column_stack([
                np.array(vx['x'], dtype=np.float64),
                np.array(vx['y'], dtype=np.float64),
                np.array(vx['z'], dtype=np.float64),
            ])

            from alignment_manager import get_alignment_manager
            am = get_alignment_manager()
            s, R, t = am.compute_leveling_from_points(xyz)

            if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
                transform_path = output_dir / "floor_transform.npz"
                np.savez(transform_path, s=np.array(s), R=R, t=t)
                pipe.send_log(f"Floor transform saved to {transform_path.name}")
            else:
                pipe.send_log("No floor plane detected, skipping alignment", level="warning")
        except Exception as e:
            pipe.send_log(f"Floor alignment computation failed: {e}", level="warning")

        # ── Build Potree LOD octree (so the first viewer load is instant) ──
        # Runs as the final reconstruction step. Carries the per-point
        # confidence + origin (frame_global/pixel_row/pixel_col) into the octree
        # (see potree_converter._ply_to_las LAS extra dims). Non-fatal.
        if postproc.get("build_potree", True):
            pipe.send_progress(96, "Building Potree LOD octree...", stage="cloudcompy")
            try:
                import sys, os
                server_dir_str = str(Path(__file__).resolve().parent.parent)
                if server_dir_str not in sys.path:
                    sys.path.insert(0, server_dir_str)
                from potree_converter import convert_ply_to_potree

                ok = convert_ply_to_potree(session_path, force=True)
                if ok:
                    potree_dir = output_dir / "potree"
                    pipe.send_log(f"Potree octree built → {potree_dir}")
                    # Mirror merged/potree symlink for new-style projects (serving
                    # checks merged_potree first, then output/potree).
                    try:
                        project_root = session_path
                        for _ in range(5):
                            project_root = project_root.parent
                            if (project_root / "project.json").exists():
                                break
                        else:
                            project_root = None
                        if project_root and (project_root / "project.json").exists():
                            merged_dir = project_root / "merged"
                            merged_dir.mkdir(parents=True, exist_ok=True)
                            merged_potree = merged_dir / "potree"
                            if merged_potree.exists() or merged_potree.is_symlink():
                                if merged_potree.is_symlink() or merged_potree.is_file():
                                    merged_potree.unlink()
                                else:
                                    import shutil as _sh
                                    _sh.rmtree(merged_potree, ignore_errors=True)
                            rel_potree = os.path.relpath(str(potree_dir), str(merged_dir))
                            os.symlink(rel_potree, str(merged_potree))
                            pipe.send_log(f"Linked merged/potree → {rel_potree}")
                    except Exception as e:
                        pipe.send_log(f"merged/potree symlink failed (non-critical): {e}", level="warning")
                else:
                    pipe.send_log("Potree conversion returned False (non-critical)", level="warning")
            except Exception as e:
                pipe.send_log(f"Potree build failed (non-critical): {e}", level="warning")
    else:
        pipe.send_log("Warning: cleaned_cloud.ply not created", level="warning")

    pipe.send_progress(100, "Cloud cleaning + Potree complete", stage="cloudcompy")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_cloudcompy_work, conn, session_dir, config)
