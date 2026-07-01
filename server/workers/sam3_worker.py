# STAC-Builder: SAM3 Worker (Subprocess)
# Runs SAM3 segmentation in its own process.
# Reads VLM analysis + frames, writes segmentation.json + masks.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import json
import sys
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _sam3_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """SAM3 segmentation — runs inside a dedicated subprocess."""

    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()
    output_dir = (session_path / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    # Read VLM analysis if available (written by vlm_worker)
    vlm_path = output_dir / "vlm_analysis.json"
    if vlm_path.exists():
        vlm_data = json.loads(vlm_path.read_text())
        prompt = vlm_data.get("prompt", "")
        frame_map = vlm_data.get("frame_map", {})
        pipe.send_log(f"Using VLM prompt: '{prompt}'")
    else:
        prompt = config.get("segmentation_prompt", "floor;wall;ceiling;door;window;furniture;object")
        frame_map = {}
        pipe.send_log(f"No VLM analysis found, using config prompt: '{prompt}'")

    if pipe.check_cancel():
        return

    pipe.send_progress(0, "Loading SAM3 model...", stage="sam3")

    from segmentation_pipeline import run_segmentation

    pipe.send_progress(10, "Running segmentation...", stage="sam3")

    def _seg_progress(pct, msg):
        # Map internal 0-100% to pipeline range 10-80%
        mapped_pct = 10 + (pct / 100) * 70
        pipe.send_progress(mapped_pct, msg, stage="sam3")

    result = run_segmentation(
        frames_dir=str(frames_dir),
        output_dir=str(output_dir),
        prompt=prompt,
        frame_map=frame_map,
        on_progress=_seg_progress,
    )

    if pipe.check_cancel():
        return

    if "error" in result:
        raise RuntimeError(f"Segmentation failed: {result['error']}")

    n_instances = len(result.get("instances", []))
    pipe.send_log(f"Segmentation complete: {n_instances} instances")

    # Apply to cleaned cloud
    pipe.send_progress(80, "Applying to point cloud...", stage="sam3")

    try:
        from segmentation_pipeline import apply_segmentation_to_cloud
        seg_data = apply_segmentation_to_cloud(output_dir)
        n_applied = len(seg_data.get("instances", []))
        pipe.send_log(f"Applied {n_applied} instances to cloud")

        # Save seg_data for broadcast
        seg_broadcast_path = output_dir / "seg_broadcast.json"
        seg_broadcast_path.write_text(json.dumps(seg_data))
    except Exception as e:
        pipe.send_log(f"Apply to cloud failed (non-fatal): {e}", level="warning")

    # ── Per-object textured TSDF mesh ──
    # Just as the cloud is split per instance above, carve each instance's
    # textured surface mesh out of the scene TSDF (output/tsdf/scene/scene.glb).
    # This is the faithful, scanned-surface deliverable per object (textured) —
    # part of segmentation, not an optional manual step. Best-effort: never fail
    # segmentation if the crop has an issue. Requires the scene TSDF to have run
    # (forced on in the pipeline whenever reconstruction runs).
    if not pipe.check_cancel():
        pipe.send_progress(85, "Carving per-object TSDF meshes...", stage="sam3")
        try:
            from segmentation.tsdf_export import crop_scene_mesh_to_instances
            result_path = output_dir / "segmentation_result.json"
            scene_dir = output_dir / "tsdf" / "scene"
            has_scene = (scene_dir / "scene.glb.orig").exists() or \
                        (scene_dir / "scene.glb").exists()
            if not has_scene:
                pipe.send_log("No scene TSDF mesh — skipping per-object TSDF crop "
                              "(run reconstruction/TSDF first)", level="warning")
            elif not result_path.exists():
                pipe.send_log("No segmentation_result.json — skipping per-object TSDF crop",
                              level="warning")
            else:
                with open(result_path) as f:
                    segments_result = json.load(f)
                n_tot = len(segments_result.get("instances", []))

                def _crop_progress(inst_id, phase, elapsed, mesh_path):
                    if phase == "done":
                        _crop_progress.done += 1
                        pct = 85 + (_crop_progress.done / max(n_tot, 1)) * 13
                        pipe.send_progress(min(pct, 98),
                                           f"TSDF mesh {_crop_progress.done}/{n_tot}",
                                           stage="sam3")
                _crop_progress.done = 0

                written = crop_scene_mesh_to_instances(
                    output_dir=output_dir,
                    segments_result=segments_result,
                    progress_cb=_crop_progress,
                )
                pipe.send_log(f"Per-object TSDF: wrote {len(written)} textured mesh(es)")
        except Exception as e:
            pipe.send_log(f"Per-object TSDF crop failed (non-fatal): {e}", level="warning")

    pipe.send_progress(100, f"Segmentation complete: {n_instances} objects", stage="sam3")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_sam3_work, conn, session_dir, config)
