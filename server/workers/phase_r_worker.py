# STAC-Builder: Phase R Worker (Subprocess)
# Runs the semantic-anchoring refinement (server/phase_r) as a pipeline stage
# between cloud cleaning and TSDF, so refined poses/depth land in
# camera_poses.txt / results_output BEFORE the fusion reads them.
#
# The stage is an ENHANCEMENT layer with its own R.9 fail-safe: it never fails
# the reconstruction. No segmentation yet → skipped with a note; any internal
# error → logged, baseline artifacts untouched (writeback is backup-gated).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import json
import sys
from multiprocessing.connection import Connection
from pathlib import Path

from workers.base import WorkerPipe, run_worker_safe


def _window_map(session_dir: Path, output_dir: Path, config: dict) -> dict[int, str]:
    """frame number -> window id, mirroring the reconstruction chunking
    (chunk_size/chunk_overlap). Overlap frames are OWNED by the chunk whose
    centre is nearest — re-identification across windows itself happens purely
    by tracking continuity (R.1), this map only feeds the Sim(3) pose graph."""
    frames_txt = None
    for base in (output_dir / "omega_run", output_dir / "maplong_run",
                 output_dir / "da3_run", output_dir):
        p = base / "camera_frames.txt"
        if p.exists():
            frames_txt = p
            break
    if frames_txt is None:
        return {}
    nums = [int(float(x)) for x in frames_txt.read_text().split()]
    recon = config.get("reconstruction", {})
    backend = recon.get("backend", "vggtomega")
    bcfg = recon.get(backend, {})
    chunk = int(bcfg.get("chunk_size", 120))
    overlap = int(bcfg.get("chunk_overlap", bcfg.get("overlap", 60)))
    stride = max(1, chunk - overlap)
    n_chunks = max(1, (max(len(nums) - overlap, 1) + stride - 1) // stride)
    out: dict[int, str] = {}
    for i, fn in enumerate(nums):
        k = int(round((i - chunk / 2) / stride))
        k = min(max(k, 0), n_chunks - 1)
        out[fn] = f"w{k:03d}"
    return out


def _phase_r_work(pipe: WorkerPipe, session_dir: str, config: dict):
    session_path = Path(session_dir)
    output_dir = (session_path / "output").resolve()

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    if not (output_dir / "segmentation.json").exists():
        pipe.send_log("Phase R skipped: no segmentation.json yet (run the "
                      "auto-prompter / Segmentation Manager, then re-run TSDF "
                      "to fuse with anchoring)", level="warning")
        pipe.send_progress(100, "Phase R skipped (no segmentation)", stage="phase_r")
        return

    if pipe.check_cancel():
        return

    try:
        from phase_r.pipeline import PhaseRPipeline

        pipe.send_progress(5, "Building instance store (R.1/R.8)...", stage="phase_r")
        wmap = _window_map(session_path, output_dir, config)
        n_windows = len(set(wmap.values())) if wmap else 1
        pipe.send_log(f"Phase R: {len(wmap)} frames across {n_windows} window(s)")

        pr = PhaseRPipeline(session_path, output_dir,
                            output_dir / "scene_r.db", config=config,
                            window_map=wmap)
        pipe.send_progress(20, "Vote + onion + Sim(3) refinement (R.2–R.6)...",
                           stage="phase_r")
        report = pr.run()

        (output_dir / "phase_r_report.json").write_text(json.dumps({
            "summary": report.summary(),
            "n_instances": report.n_instances,
            "n_windows": report.n_windows,
            "iterations": report.iterations,
            "converged": report.converged,
            "onion_bimodal": report.onion_bimodal,
            "onion_separation_median_m": report.onion_separation_median_m,
            "seams": report.seams,
            "scale_report": report.scale_report,
            "scale_conflicts": report.scale_conflicts,
            "failsafe": report.failsafe,
            "used_anchoring": report.used_anchoring,
            "writeback": report.writeback,
        }, indent=2, default=str))
        pipe.send_log(f"Phase R: {report.summary()}")
        pipe.send_progress(100, f"Phase R done — anchoring "
                                f"{'kept' if report.used_anchoring else 'fallback'}",
                           stage="phase_r")
    except Exception as e:  # noqa: BLE001 — enhancement layer, never fatal
        pipe.send_log(f"Phase R failed non-fatally: {e}; reconstruction keeps "
                      f"the no-anchor baseline", level="warning")
        pipe.send_progress(100, "Phase R skipped (error — baseline kept)",
                           stage="phase_r")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_phase_r_work, conn, session_dir, config)
