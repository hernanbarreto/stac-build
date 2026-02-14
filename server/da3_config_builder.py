"""
DA3 Configuration Builder

Shared utility to build DA3_Streaming config dicts from config.yaml.
Resolves model weights and config.json from HuggingFace cache.
ALL values come from config.yaml — zero hardcoding.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

import os
from pathlib import Path


def resolve_hf_model_paths(model_name: str):
    """
    Resolve model.safetensors and config.json from HuggingFace cache.
    
    Args:
        model_name: HuggingFace model name, e.g. "depth-anything/DA3-LARGE-1.1"
    
    Returns:
        (weights_path, config_path) absolute paths to the cached files
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_name.replace("/", "--")
    model_dir = os.path.join(cache_dir, model_dir_name)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model '{model_name}' not found in HF cache at {model_dir}. "
            f"Run: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_name}')\""
        )
    
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"No snapshots dir found for {model_name}")
    
    snapshots = sorted(os.listdir(snapshots_dir))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found for {model_name}")
    
    snapshot_dir = os.path.join(snapshots_dir, snapshots[-1])
    
    weights_path = os.path.join(snapshot_dir, "model.safetensors")
    config_path = os.path.join(snapshot_dir, "config.json")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"model.safetensors not found in {snapshot_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {snapshot_dir}")
    
    return weights_path, config_path


def build_da3_config(cfg: dict) -> dict:
    """
    Build a DA3_Streaming-compatible config dict from config.yaml.
    
    ALL values come from cfg (loaded config.yaml). Model files are
    resolved from HuggingFace cache based on models.depth.name.
    
    Args:
        cfg: The full config.yaml dict
    
    Returns:
        Config dict compatible with DA3_Streaming constructor
    """
    depth_cfg = cfg["models"]["depth"]
    da3_cfg = cfg["models"]["da3"]
    alignment_cfg = cfg["alignment"]
    server_cfg = cfg["server"]
    
    # Resolve model from HF cache
    model_name = depth_cfg["name"]
    weights_path, config_path = resolve_hf_model_paths(model_name)
    
    print(f"[DA3 Config] Model: {model_name}")
    print(f"[DA3 Config] Weights: {weights_path}")
    print(f"[DA3 Config] Config: {config_path}")
    print(f"[DA3 Config] Device: {depth_cfg['device']}")
    
    # SALAD weights for loop closure
    salad_path = os.path.join(cfg["paths"]["da3_weights_dir"], "dino_salad.ckpt")
    
    return {
        "frame_stride": server_cfg.get("frame_stride", 1),
        "Weights": {
            "DA3_CONFIG": config_path,
            "DA3": weights_path,
            "SALAD": salad_path,
        },
        "Model": {
            "chunk_size": server_cfg["chunk_size"],
            "overlap": server_cfg["chunk_overlap"],
            "loop_chunk_size": da3_cfg["loop_chunk_size"],
            "loop_enable": da3_cfg.get("loop_enable", False),
            "useDBoW": da3_cfg.get("useDBoW", False),
            "delete_temp_files": da3_cfg.get("delete_temp_files", False),
            "align_method": alignment_cfg["method"],
            "align_lib": alignment_cfg["align_lib"],
            "scale_compute_method": alignment_cfg["scale_compute_method"],
            "align_type": alignment_cfg["align_type"],
            "device": depth_cfg["device"],
            "use_ray_pose": depth_cfg.get("use_ray_pose", True),
            "ref_view_strategy": depth_cfg["ref_view_strategy"],
            "ref_view_strategy_loop": depth_cfg.get("ref_view_strategy_loop", depth_cfg["ref_view_strategy"]),
            "depth_threshold": da3_cfg["depth_threshold"],
            "save_depth_conf_result": da3_cfg.get("save_depth_conf_result", False),
            "save_debug_info": da3_cfg.get("save_debug_info", False),
            "Sparse_Align": {
                "keypoint_select": alignment_cfg["sparse_align"]["keypoint_select"],
                "keypoint_num": alignment_cfg["sparse_align"]["keypoint_num"],
            },
            "IRLS": {
                "delta": alignment_cfg["ransac"]["delta"],
                "max_iters": alignment_cfg["ransac"]["max_iters"],
                "tol": alignment_cfg["ransac"]["tolerance"],
            },
            "Pointcloud_Save": {
                "conf_threshold_coef": da3_cfg["pointcloud_save"]["conf_threshold_coef"],
                "sample_ratio": da3_cfg["pointcloud_save"]["sample_ratio"],
            },
        },
        "Loop": {
            "SALAD": da3_cfg["loop_salad"],
            "SIM3_Optimizer": da3_cfg["loop_sim3_optimizer"],
        },
    }
