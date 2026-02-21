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

    if not script_path.exists():
        pipe.send_log("run_cloudcompy.sh not found, skipping", level="warning")
        return

    chunks = sorted(output_dir.glob("chunk_*.ply"))
    if not chunks:
        pipe.send_log("No chunk PLYs found, skipping", level="warning")
        return

    pipe.send_progress(0, f"Cleaning {len(chunks)} chunks (voxel={voxel_size*1000:.1f}mm)",
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

    for line in iter(proc.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue

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
    else:
        pipe.send_log("Warning: cleaned_cloud.ply not created", level="warning")

    pipe.send_progress(100, "Cloud cleaning complete", stage="cloudcompy")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_cloudcompy_work, conn, session_dir, config)
