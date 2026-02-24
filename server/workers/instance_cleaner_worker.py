# STAC-Builder: Instance Cleaner Worker
# Processes segmented instances by isolating clusters with DBSCAN and running CloudComPy smoothing.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import json
import logging
import subprocess
import traceback
from pathlib import Path
from multiprocessing.connection import Connection

import numpy as np

from workers.base import WorkerPipe, run_worker_safe


def _run_dbscan(point_cloud: np.ndarray, eps: float = 0.05, min_samples: int = 10):
    """Run Open3D DBSCAN to keep only the largest cluster of a point cloud."""
    try:
        import open3d as o3d
    except ImportError:
        logging.warning("Open3D not installed. Skipping DBSCAN.")
        return point_cloud

    if len(point_cloud) < min_samples:
        return point_cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    
    # DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
    
    if len(labels) == 0 or labels.max() < 0:
        return point_cloud # All noise or empty
        
    # Find largest cluster (ignoring noise label -1)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return point_cloud
        
    largest_cluster_label = unique_labels[np.argmax(counts)]
    
    # Filter points
    mask = labels == largest_cluster_label
    return point_cloud[mask]


def _save_ply(points: np.ndarray, colors: np.ndarray, filepath: Path):
    """Save a simple PLY file."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.io.write_point_cloud(str(filepath), pcd)


def _load_ply(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLY file and return points and colors."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(filepath))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


def _instance_cleaner_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """Runs DBSCAN and CloudComPy on each SAM3 segmented instance."""

    session_path = Path(session_dir)
    output_dir = (session_path / "output").resolve()
    seg_broadcast = output_dir / "seg_broadcast.json"
    cleaned_cloud_ply = output_dir / "cleaned_cloud.ply"
    final_output_ply = output_dir / "inst_cleaned_cloud.ply"

    if not seg_broadcast.exists() or not cleaned_cloud_ply.exists():
        pipe.send_log("Missing segmentation or cleaned cloud data. Skipping.", level="warning")
        return

    try:
        with open(seg_broadcast, "r") as f:
            seg_data = json.load(f)
    except Exception as e:
        pipe.send_log(f"Failed to load segmentation data: {e}", level="error")
        return

    instances = seg_data.get("instances", [])
    if not instances:
        pipe.send_log("No instances found to clean.", level="info")
        # Just copy the array
        import shutil
        shutil.copy2(cleaned_cloud_ply, final_output_ply)
        return

    pipe.send_progress(5, f"Cleaning {len(instances)} instances", stage="instance_cleaner")

    # Load main cloud
    pipe.send_log(f"Loading main cloud from {cleaned_cloud_ply.name}...")
    try:
        cloud_pts, cloud_colors = _load_ply(cleaned_cloud_ply)
    except Exception as e:
        pipe.send_log(f"Failed to load main cloud: {e}", level="error")
        return
        
    if cloud_pts is None or len(cloud_pts) == 0:
         pipe.send_log("Cloud is empty.", level="error")
         return

    # Configuration
    inst_cfg = config.get("instance_cleaning", {})
    dbscan_eps = inst_cfg.get("dbscan_eps", 0.05)
    dbscan_min_samples = inst_cfg.get("dbscan_min_samples", 10)
    
    server_dir = Path(__file__).resolve().parent.parent
    cc_script = server_dir / "run_cloudcompy.sh"
    
    has_cc = cc_script.exists()
    if not has_cc:
        pipe.send_log(f"CloudComPy script not found at {cc_script}. Proceeding with only DBSCAN.", level="warning")

    # To rebuild the cloud, we will start with all Unassigned points
    # and then add back the cleaned instance points.
    
    # 1. Identify all point indices belonging to ANY instance
    all_assigned_indices = set()
    for inst in instances:
        indices = inst.get("point_indices", [])
        all_assigned_indices.update(indices)
        
    all_indices_set = set(range(len(cloud_pts)))
    unassigned_indices = list(all_indices_set - all_assigned_indices)
    
    # Start the new cloud with unassigned points
    new_points_list = [cloud_pts[unassigned_indices]]
    new_colors_list = [cloud_colors[unassigned_indices]] if cloud_colors is not None else []
    
    # 2. Process each instance
    for i, inst in enumerate(instances):
        inst_id = inst.get("id", f"unknown_{i}")
        label = inst.get("label", "Object")
        indices = inst.get("point_indices", [])
        
        if not indices:
            continue
            
        pipe.send_log(f"Processing instance {inst_id} ({label}) with {len(indices)} points")
        pct = 10 + (i / len(instances)) * 80
        pipe.send_progress(pct, f"Cleaning instance {label}...", stage="instance_cleaner")
        
        # Extract instance cloud containing 3D pos + colors (or 0s)
        inst_pts = cloud_pts[indices]
        inst_cols = cloud_colors[indices] if cloud_colors is not None else np.zeros_like(inst_pts)
        
        # Merge for DBSCAN
        inst_cloud = np.hstack((inst_pts, inst_cols))
        
        # -- Step A: DBSCAN --
        pre_pts = len(inst_cloud)
        inst_cloud = _run_dbscan(inst_cloud, eps=dbscan_eps, min_samples=dbscan_min_samples)
        post_pts = len(inst_cloud)
        pipe.send_log(f"  DBSCAN: {pre_pts} -> {post_pts} points")
        
        if post_pts == 0:
            continue
            
        # -- Step B: CloudComPy (SOR/Erosion) --
        if has_cc:
            # Save temp PLY
            temp_ply = output_dir / f"temp_inst_{inst_id}.ply"
            cleaned_temp_ply = output_dir / f"clean_temp_inst_{inst_id}.ply"
            
            _save_ply(inst_cloud[:, :3], inst_cloud[:, 3:6], temp_ply)
            
            sor_knn = inst_cfg.get("sor_knn", 10)
            sor_sigma = inst_cfg.get("sor_sigma", 0.5)
            
            cmd = [
                "bash", str(cc_script),
                "--input", str(temp_ply),
                "--output", str(cleaned_temp_ply),
                "--sor-knn", str(sor_knn),
                "--sor-sigma", str(sor_sigma),
                # No subsampling here to preserve density
                "--skip-noise", "--skip-normals" 
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                if cleaned_temp_ply.exists():
                     cc_pts, cc_cols = _load_ply(cleaned_temp_ply)
                     if cc_pts is not None and len(cc_pts) > 0:
                         inst_cloud = np.hstack((cc_pts, cc_cols if cc_cols is not None else np.zeros_like(cc_pts)))
                         pipe.send_log(f"  CloudComPy: eroded to {len(inst_cloud)} points")
                     cleaned_temp_ply.unlink(missing_ok=True)
            except Exception as e:
                pipe.send_log(f"  CloudComPy failed for {inst_id}: {e}", level="warning")
            finally:
                temp_ply.unlink(missing_ok=True)
                
        # Append cleaned instance to final cloud
        new_points_list.append(inst_cloud[:, :3])
        if len(new_colors_list) > 0:
             new_colors_list.append(inst_cloud[:, 3:6])
             
        # Optional: Save individual cleaned instance for the frontend
        if inst_cfg.get("save_individual_plys", True):
             indiv_ply = output_dir / f"instance_{inst_id}.ply"
             _save_ply(inst_cloud[:, :3], inst_cloud[:, 3:6], indiv_ply)
    
    # 3. Assemble final cloud
    pipe.send_progress(90, "Assembling final cloud...", stage="instance_cleaner")
    
    try:
        final_pts = np.vstack(new_points_list)
        final_cols = np.vstack(new_colors_list) if len(new_colors_list) > 0 else None
        
        _save_ply(final_pts, final_cols, final_output_ply)
        
        pipe.send_log(f"Final assembled cloud: {len(final_pts)} points saved to {final_output_ply.name}")
        pipe.send_progress(100, "Instance cleaning complete", stage="instance_cleaner")
    except Exception as e:
        pipe.send_log(f"Failed to assemble final cloud: {e}", level="error")
        pipe.send_progress(100, "Instance cleaning failed", stage="instance_cleaner")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_instance_cleaner_work, conn, session_dir, config)
