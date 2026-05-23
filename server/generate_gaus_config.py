#!/usr/bin/env python3
"""
Generate GauS-SLAM config.py from TUM bridge camera intrinsics.

Produces a complete config matching the reference TUM config structure
from vendor/gaus-slam/configs/tum/config.py, with paths and resolution
adapted to the DA3→TUM bridge output.

Hernán Barreto — Ingerop IN3 Session IV — STAC
"""
import argparse
import glob
import os
import yaml


def main():
    parser = argparse.ArgumentParser("Generate GauS-SLAM config")
    parser.add_argument("--tum_dir", required=True, help="TUM bridge output dir")
    parser.add_argument("--output", required=True, help="Output config.py path")
    parser.add_argument("--device", default="cuda", help="Primary device")
    args = parser.parse_args()

    # Read intrinsics from TUM bridge
    intrinsics_path = os.path.join(args.tum_dir, "camera_intrinsics.yaml")
    with open(intrinsics_path) as f:
        cam = yaml.safe_load(f)["camera_params"]

    h = cam["image_height"]
    w = cam["image_width"]
    n_frames = len(glob.glob(os.path.join(args.tum_dir, "rgb", "*.png")))

    basedir = os.path.dirname(args.tum_dir)
    sequence = os.path.basename(args.tum_dir)

    # Tuning parameters (same as reference TUM config)
    # Camera pose learning rates: very low to keep poses anchored to ARKit odometry.
    # ARKit poses are confirmed correct (pure ARKit reconstruction was accurate).
    # GauS-SLAM should optimize Gaussians, not deviate from the ARKit trajectory.
    trans_lr_base = 0.0002   # was 0.004 — 20x lower to minimize camera drift
    rot_lr_base   = 0.00005  # was 0.001 — 20x lower
    num_tracking_iter = 120
    num_ba_iters = 120
    localmap_max_frames = 40

    config_content = f'''import os

# Auto-generated GauS-SLAM config for STAC-Builder
# {n_frames} frames, {w}x{h} resolution

seed = 0
h = {h}
w = {w}
trans_lr_base = {trans_lr_base}
rot_lr_base = {rot_lr_base}
num_tracking_iter = {num_tracking_iter}
num_ba_iters = {num_ba_iters}
localmap_max_frames = {localmap_max_frames}

config = dict(
    vis_base_dir="{basedir}/gaus_slam_output",
    seed=seed,
    primary_device="{args.device}",
    use_wandb=False,
    wandb=dict(
        name="STAC_GauS",
        project_name="GauS_SLAM_stac",
    ),

    render=dict(
        method="2dgs",
        use_sa=True,
        use_weight_norm=True,
        enable_exposure=False,
        eps=1e-6,
        depth_far=1e2,
        depth_near=1e-2,
    ),

    frontend=dict(
        num_tracking_iters=num_tracking_iter,
        num_mapping_iters=localmap_max_frames,
        converged_th=-1,
        tau_k=0.01,
        tau_l=h * w * 1.5,
        max_frames=localmap_max_frames,
        vel_pose_init=True,
        enable_retracking=False,
        additional_densify=False,
    ),

    backend=dict(
        num_ba_iters=num_ba_iters,
        num_frame_saved=localmap_max_frames // 4,
        num_covis_submaps=30,
        sleep_time=0.1,
        mesh_vis=False,
        render_vis=False,
        common_vis=True,
        gs_densify=False,
        random_process=True,
        final_refinement=0,
    ),

    densify=dict(
        use_edge_growth=False,
        densify_interval=20,
        method="splatam",
        sil_thres=0.6,
        edge_thres=0.4,
        opacity_cuil=0.05,
        scale_cuil=5e-4,
        scale_max=0.1,
        num_addpts=h * w,
        percent_dense=0.01,
        densify_grad_threshold=0.0002,
        extent=2,
    ),

    loss=dict(
        ignore_outliners=False,
        use_normal_loss=False,
        silmask_th=0.90,
        tracking=dict(
            color=0.5,
            depth=1.0,
            normal=0,
            dist=0,
        ),
        mapping=dict(
            color=0.5,
            depth=1.0,
            normal=0,
            dist=0.1,
        ),
    ),

    data=dict(
        dataset_name="tum",
        basedir="{basedir}",
        gradslam_data_cfg="{intrinsics_path}",
        sequence="{sequence}",
        desired_image_height=h,
        desired_image_width=w,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),

    gaussians=dict(
        gaussian_distribution="anisotropic",
        training_args=dict(
            xyz_lr=0.0001,
            feature_lr=0.0025,
            rgb_lr=0.0025,
            rotation_lr=0.001,
            opacity_lr=0.05,
            scaling_lr=0.001,
        ),
    ),

    cameras=dict(
        height=h,
        width=w,
        intrinsics=[
            [{cam["fx"]}, 0.0, {cam["cx"]}],
            [0.0, {cam["fy"]}, {cam["cy"]}],
            [0.0, 0.0, 1.0],
        ],
        adam_betas=(0.7, 0.99),
        frontend_lr=dict(
            cam_rot_lr_init=rot_lr_base,
            cam_rot_lr_final=rot_lr_base / 10,
            cam_rot_lr_max_step=num_tracking_iter,
            cam_trans_lr_init=trans_lr_base,
            cam_trans_lr_final=trans_lr_base / 10,
            cam_trans_lr_max_step=num_tracking_iter,
            exposure_lr_init=0.0001,
            exposure_lr_final=0.00001,
            exposure_lr_max_step=100,
        ),
        backend_lr=dict(
            cam_rot_lr_init=rot_lr_base / 2,
            cam_rot_lr_final=rot_lr_base / 10,
            cam_rot_lr_max_step=2 * num_ba_iters,
            cam_trans_lr_init=trans_lr_base / 2,
            cam_trans_lr_final=trans_lr_base / 10,
            cam_trans_lr_max_step=2 * num_ba_iters,
            exposure_lr_init=0.0001,
            exposure_lr_final=0.00001,
            exposure_lr_max_step=100,
        ),
    ),

    viz=dict(
        viz_w=600,
        viz_h=340,
        view_scale=2,
        mesh_every=5,
        gen_animation=False,
        video_freq=30,
    ),

    eval=dict(
        save_renders=True,
        eval_mesh=True,
        save_mesh=True,
        mesh_interval=5,
        voxel_size=0.01,
    ),
)
'''

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(config_content)

    print(f"[GauS-SLAM] Generated config: {args.output}")
    print(f"  Resolution: {w}x{h}")
    print(f"  Frames: {n_frames}")
    print(f"  Device: {args.device}")


if __name__ == "__main__":
    main()
