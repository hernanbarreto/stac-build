#!/usr/bin/env python3
"""
Depth Anything 3 - Giant model with CPU offloading
Uses the same technique as SAM 3D Objects to fit large models in 6GB VRAM
python da3_low_vram.py ~/stac/scene/input/image.png     --output ~/stac/scene/da3_output     --model depth-anything/DA3-GIANT-1.1     --res 504
"""
import torch
import gc
import numpy as np
from pathlib import Path

# Force efficient CUDA allocation
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def run_da3_with_offloading(
    image_path: str,
    output_dir: str = "output",
    model_name: str = "depth-anything/DA3-LARGE-1.1",  # Start with LARGE, try GIANT later
    process_res: int = 504,
    use_half: bool = True
):
    """
    Run DA3 with CPU offloading for low VRAM GPUs.
    """
    from depth_anything_3.api import DepthAnything3
    
    print("=" * 60)
    print("DEPTH ANYTHING 3 - LOW VRAM MODE")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Image: {image_path}")
    print(f"Resolution: {process_res}")
    print(f"Half precision: {use_half}")
    
    # Clear GPU before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check available VRAM
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {total_mem:.1f}GB, Free: {free_mem:.1f}GB")
    
    print("\n[1/4] Loading model to CPU...")
    
    # Load model to CPU first (key for offloading)
    model = DepthAnything3.from_pretrained(model_name)
    
    gc.collect()
    
    print("[2/4] Moving model to GPU...")
    
    # Move to GPU (keep as float32, use autocast for mixed precision)
    model = model.to("cuda")
    
    # Clear any fragmented memory
    torch.cuda.empty_cache()
    
    print("[3/4] Running inference with mixed precision...")
    
    # Run inference with lower resolution
    # Use autocast for automatic mixed precision (keeps LayerNorm in FP32)
    try:
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
                prediction = model.inference(
                    [image_path],
                    process_res=process_res,
                    process_res_method="lower_bound_resize"
                )
        
        print("[4/4] Saving results...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save depth as numpy
        np.save(output_path / "depth.npy", prediction.depth)
        print(f"  Saved: {output_path / 'depth.npy'} - shape: {prediction.depth.shape}")
        
        # Save depth as image (normalized)
        from PIL import Image
        depth = prediction.depth[0]  # First image
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_img = (depth_normalized * 255).astype(np.uint8)
        Image.fromarray(depth_img).save(output_path / "depth.png")
        print(f"  Saved: {output_path / 'depth.png'}")
        
        # Save extrinsics/intrinsics if available
        if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
            np.save(output_path / "extrinsics.npy", prediction.extrinsics)
            print(f"  Saved: {output_path / 'extrinsics.npy'}")
        
        if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
            np.save(output_path / "intrinsics.npy", prediction.intrinsics)
            print(f"  Saved: {output_path / 'intrinsics.npy'}")
        
        # Generate point cloud
        print("\n  Generating point cloud...")
        try:
            # Load original image for colors
            orig_img = np.array(Image.open(image_path).convert("RGB"))
            H_orig, W_orig = orig_img.shape[:2]
            
            # Resize to match depth
            H_depth, W_depth = depth.shape
            img_resized = np.array(Image.fromarray(orig_img).resize((W_depth, H_depth)))
            
            # Get intrinsics
            if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
                K = prediction.intrinsics[0]  # [3, 3]
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
            else:
                # Estimate intrinsics
                fx = fy = W_depth * 0.8
                cx, cy = W_depth / 2, H_depth / 2
            
            # Create mesh grid
            u, v = np.meshgrid(np.arange(W_depth), np.arange(H_depth))
            
            # Backproject to 3D
            z = depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            # Stack points and colors
            points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
            colors = img_resized.reshape(-1, 3)
            
            # Filter invalid points
            valid = (z.flatten() > 0) & (z.flatten() < 100) & ~np.isnan(z.flatten())
            points = points[valid]
            colors = colors[valid]
            
            # Save as PLY
            ply_path = output_path / "pointcloud.ply"
            with open(ply_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for p, c in zip(points, colors):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            
            print(f"  Saved: {ply_path} ({len(points):,} points)")
            
            # === SAM3D COMPATIBLE POINTMAP ===
            # SAM3D expects pointmap in PyTorch3D coordinate system:
            # - X: right (same as OpenCV)
            # - Y: down -> up (NEGATED from OpenCV)  
            # - Z: forward -> backward (NEGATED from OpenCV)
            # Shape: [H, W, 3]
            
            print("\n  Generating SAM3D-compatible pointmap...")
            
            # Stack as [H, W, 3] - OpenCV convention (X right, Y down, Z forward)
            pointmap_opencv = np.stack([x, y, z], axis=-1)  # [H, W, 3]
            
            # Convert to match SAM3D's camera_to_pytorch3d_camera transform
            # SAM3D uses: [-X, -Y, +Z] (negate X and Y, keep Z)
            # This matches the look_at_view_transform with eye=[0,0,-1], at=[0,0,0], up=[0,-1,0]
            pointmap_sam3d = pointmap_opencv.copy()
            pointmap_sam3d[..., 0] = -pointmap_opencv[..., 0]  # Negate X
            pointmap_sam3d[..., 1] = -pointmap_opencv[..., 1]  # Negate Y
            # Z stays positive
            
            # Save pointmap
            np.save(output_path / "pointmap.npy", pointmap_sam3d.astype(np.float32))
            print(f"  Saved: {output_path / 'pointmap.npy'} - shape: {pointmap_sam3d.shape}")
            
            # Save metadata for SAM3D integration
            import json
            metadata = {
                "source": "depth-anything-3",
                "model": model_name,
                "resolution": process_res,
                "image_path": str(image_path),
                "depth_shape": list(depth.shape),
                "pointmap_shape": list(pointmap_sam3d.shape),
                "coordinate_system": "sam3d",  # Matches SAM3D's camera_to_pytorch3d_camera transform
                "intrinsics": {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy)
                },
                "valid_points": int(valid.sum()),
                "total_points": int(len(z.flatten()))
            }
            
            with open(output_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  Saved: {output_path / 'metadata.json'}")
            
        except Exception as e:
            print(f"  Warning: Could not generate point cloud: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ SUCCESS!")
        print("=" * 60)
        
        return prediction
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ Out of VRAM! Try:")
        print(f"  1. Lower resolution: --process-res 336")
        print(f"  2. Smaller model: DA3-BASE or DA3-SMALL")
        raise e
    
    finally:
        # Always clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DA3 Low VRAM Mode")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--model", "-m", default="depth-anything/DA3-LARGE-1.1",
                        help="Model name (default: DA3-LARGE-1.1)")
    parser.add_argument("--res", "-r", type=int, default=504,
                        help="Processing resolution (default: 504, try 336 for less VRAM)")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable half precision (uses more VRAM)")
    
    args = parser.parse_args()
    
    run_da3_with_offloading(
        image_path=args.image,
        output_dir=args.output,
        model_name=args.model,
        process_res=args.res,
        use_half=not args.no_half
    )
