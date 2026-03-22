#!/usr/bin/env python3
"""
Phase 2: BIM Render from Camera Poses
Renders the IFC BIM model from the same camera poses used in reconstruction,
producing synthetic views for comparison with real construction frames.

Usage:
  conda activate stac-build
  cd /home/hernan/stac-builder

  python tests/test_bim_render.py \
    --project server/data/projects/live_1770423147 \
    --scan 2026-02-06 \
    --output /tmp/bim_render_test \
    --max-frames 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from pathlib import Path
from PIL import Image


def load_selected_frames(frames_dir: str) -> list:
    """Load keyframe filenames from selected_frames.json."""
    sf_path = os.path.join(frames_dir, "selected_frames.json")
    with open(sf_path) as f:
        data = json.load(f)
    return data["selected_files"]


def load_camera_poses(poses_path: str) -> np.ndarray:
    """Load camera poses from camera_poses.txt.
    Format: one 4x4 c2w (camera-to-world) matrix per line, row-major.
    Returns array of 4x4 c2w matrices.
    """
    poses = []
    with open(poses_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                mat = np.array(vals).reshape(4, 4)
                poses.append(mat)
    return np.array(poses)


def load_intrinsics(intrinsics_path: str) -> list:
    """Load per-frame intrinsics from intrinsic.txt (fx, fy, cx, cy per line)."""
    intrinsics = []
    with open(intrinsics_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 4:
                intrinsics.append({
                    "fx": vals[0], "fy": vals[1],
                    "cx": vals[2], "cy": vals[3],
                })
    return intrinsics


def load_floor_transform(transform_path: str) -> tuple:
    """Load Sim3 transform from floor_transform.npz → (s, R, t)."""
    data = np.load(transform_path)
    s = float(data["s"])
    R = data["R"]  # (3, 3)
    t = data["t"]  # (3,)
    return s, R, t


def load_ifc_mesh(ifc_path: str):
    """Extract all geometry from IFC file as a single trimesh Scene."""
    import ifcopenshell
    import ifcopenshell.geom
    import trimesh
    
    print(f"[BIM Render] Loading IFC: {ifc_path}")
    ifc = ifcopenshell.open(ifc_path)
    
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)
    
    meshes = []
    element_names = []
    face_counts = []  # number of faces per element
    
    # Process all products that have geometry
    products = ifc.by_type("IfcProduct")
    print(f"[BIM Render] Processing {len(products)} IFC products...")
    
    for product in products:
        if product.Representation is None:
            continue
        # Skip non-physical elements that block the view
        if product.is_a() in ("IfcSpace", "IfcOpeningElement"):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            faces = np.array(shape.geometry.faces).reshape(-1, 3)
            
            if len(verts) == 0 or len(faces) == 0:
                continue
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            
            # Assign a color based on IFC type
            ifc_type = product.is_a()
            color = _ifc_type_color(ifc_type)
            mesh.visual.face_colors = np.tile(color, (len(faces), 1))
            
            meshes.append(mesh)
            element_names.append(f"{ifc_type}:{product.Name or product.GlobalId}")
            face_counts.append(len(faces))
        except Exception as e:
            continue
    
    print(f"[BIM Render] Extracted {len(meshes)} meshes from IFC")
    
    if not meshes:
        raise ValueError("No geometry extracted from IFC")
    
    # Build per-face → element index mapping
    face_to_element = np.concatenate([
        np.full(fc, i, dtype=np.int32) for i, fc in enumerate(face_counts)
    ])
    
    # Combine into a single mesh for rendering
    combined = trimesh.util.concatenate(meshes)
    return combined, element_names, face_to_element


def _ifc_type_color(ifc_type: str) -> list:
    """Assign distinct, saturated colors per IFC element type."""
    colors = {
        "IfcWall": [230, 220, 200, 255],
        "IfcWallStandardCase": [230, 220, 200, 255],
        "IfcSlab": [180, 170, 140, 255],
        "IfcColumn": [100, 140, 180, 255],
        "IfcBeam": [80, 120, 160, 255],
        "IfcDoor": [160, 100, 50, 255],
        "IfcWindow": [120, 200, 240, 200],
        "IfcStair": [200, 180, 140, 255],
        "IfcStairFlight": [200, 180, 140, 255],
        "IfcRailing": [80, 80, 80, 255],
        "IfcRoof": [200, 80, 80, 255],
        "IfcPlate": [160, 160, 170, 255],
        "IfcCovering": [210, 200, 170, 255],
        "IfcFurnishingElement": [180, 130, 90, 255],
        "IfcBuildingElementProxy": [170, 170, 170, 255],
        "IfcSpace": [80, 100, 180, 80],
    }
    return colors.get(ifc_type, [160, 160, 160, 255])


def render_scene_from_pose(mesh, c2w: np.ndarray, intrinsic: dict, 
                           width: int, height: int) -> np.ndarray:
    """
    Render a trimesh mesh from a camera pose using trimesh's built-in renderer.
    
    Args:
        mesh: trimesh.Trimesh — combined BIM mesh
        c2w: (4, 4) camera-to-world matrix
        intrinsic: dict with fx, fy, cx, cy
        width, height: render resolution
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    import trimesh
    
    # Create a scene with the mesh
    scene = trimesh.Scene(mesh)
    
    # Set camera from pose
    # trimesh uses camera_transform as w2c, so invert c2w
    w2c = np.linalg.inv(c2w)
    
    # trimesh camera: looking down -Z in camera space
    # We need to flip Y and Z to convert from OpenCV convention to OpenGL
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float64)
    camera_transform = c2w @ cv_to_gl  # trimesh wants c2w in OpenGL convention
    
    # Create camera with intrinsics
    fov_y = 2 * np.arctan(height / (2 * intrinsic["fy"]))
    
    camera = trimesh.scene.Camera(
        resolution=(width, height),
        fov=(np.degrees(fov_y), np.degrees(fov_y)),
    )
    
    scene.camera = camera
    scene.camera_transform = camera_transform
    
    # Try offscreen render
    try:
        # This uses pyglet/osmesa if available
        data = scene.save_image(resolution=(width, height))
        img = Image.open(trimesh.util.wrap_as_stream(data))
        return np.array(img)[..., :3]  # Remove alpha
    except Exception as e:
        print(f"[BIM Render] Offscreen render failed: {e}")
        print(f"[BIM Render] Trying ray-based render...")
        
        # Fallback: ray-based rendering (slower but doesn't need OpenGL)
        return render_raytrace(mesh, c2w, intrinsic, width, height)


def render_raytrace(mesh, c2w: np.ndarray, intrinsic: dict,
                    width: int, height: int) -> tuple:
    """
    Render using trimesh ray casting (no OpenGL needed).
    Returns (rgb_image, depth_map).
    """
    import trimesh
    
    fx, fy = intrinsic["fx"], intrinsic["fy"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]
    
    # Generate ray directions in camera space
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)
    
    # Camera-space ray directions (OpenCV convention: Z forward)
    dirs_cam = np.stack([
        (uu - cx) / fx,
        (vv - cy) / fy,
        np.ones_like(uu),
    ], axis=-1)  # (H, W, 3)
    
    dirs_cam_flat = dirs_cam.reshape(-1, 3)
    # Normalize
    dirs_cam_flat = dirs_cam_flat / np.linalg.norm(dirs_cam_flat, axis=1, keepdims=True)
    
    # Transform to world space
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    
    dirs_world = (R @ dirs_cam_flat.T).T  # (N, 3)
    origins = np.tile(t, (len(dirs_world), 1))  # (N, 3)
    
    # Cast rays
    print(f"[BIM Render]   Ray casting {width}x{height} = {len(dirs_world)} rays...")
    t0 = time.time()
    
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=dirs_world,
        multiple_hits=False,
    )
    
    t_ray = time.time() - t0
    print(f"[BIM Render]   {len(locations)} hits in {t_ray:.1f}s")
    
    # Build images
    rgb = np.full((height * width, 3), 255, dtype=np.uint8)  # White background
    depth = np.full(height * width, np.inf, dtype=np.float32)
    face_id_buf = np.full(height * width, -1, dtype=np.int32)
    normal_buf = np.zeros((height * width, 3), dtype=np.float32)
    
    if len(locations) > 0:
        # Get base face colors
        face_colors = mesh.visual.face_colors[index_tri][:, :3].astype(np.float32)
        
        # Compute face normals for shading
        face_normals = mesh.face_normals[index_tri]  # (N_hits, 3)
        
        # Light direction: from above-right-behind camera
        cam_forward = R[:, 2]  # camera Z axis in world space
        cam_up = -R[:, 1]      # camera -Y = up in OpenCV
        light_dir = cam_forward * 0.3 + cam_up * 0.6 + R[:, 0] * 0.3
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Diffuse shading: N·L (use absolute value for double-sided)
        ndotl = np.abs(np.sum(face_normals * light_dir, axis=1))
        
        # Ambient + diffuse
        ambient = 0.35
        diffuse = 0.65
        shade = np.clip(ambient + diffuse * ndotl, 0, 1)[:, None]
        
        # Apply shading to face colors
        shaded_colors = np.clip(face_colors * shade, 0, 255).astype(np.uint8)
        rgb[index_ray] = shaded_colors
        
        # Compute depths
        dists = np.linalg.norm(locations - origins[index_ray], axis=1)
        depth[index_ray] = dists
        
        # Store face IDs and normals for edge detection
        face_id_buf[index_ray] = index_tri
        normal_buf[index_ray] = face_normals
    
    rgb = rgb.reshape(height, width, 3)
    depth = depth.reshape(height, width)
    face_id_buf = face_id_buf.reshape(height, width)
    normal_buf = normal_buf.reshape(height, width, 3)
    
    # Edge detection: draw dark lines at geometry boundaries
    edge_mask = np.zeros((height, width), dtype=bool)
    
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        # Shifted face IDs
        fid_shifted = np.roll(face_id_buf, (dy, dx), axis=(0, 1))
        depth_shifted = np.roll(depth, (dy, dx), axis=(0, 1))
        
        # Edge where adjacent pixels have different face IDs (both valid)
        valid = (face_id_buf >= 0) & (fid_shifted >= 0)
        face_edge = valid & (face_id_buf != fid_shifted)
        
        # Edge where depth changes abruptly (silhouette)
        d_valid = (depth < np.inf) & (depth_shifted < np.inf)
        depth_ratio = np.where(d_valid, np.abs(depth - depth_shifted) / (depth + 1e-6), 0)
        depth_edge = d_valid & (depth_ratio > 0.05)
        
        # Edge at boundary of geometry (hit vs miss)
        boundary_edge = (face_id_buf >= 0) != (fid_shifted >= 0)
        
        edge_mask |= face_edge | depth_edge | boundary_edge
    
    # Draw dark lines at edges
    rgb[edge_mask] = np.clip(rgb[edge_mask].astype(np.float32) * 0.25, 0, 60).astype(np.uint8)
    
    return rgb, depth, face_id_buf


def create_comparison_image(real_frame: np.ndarray, bim_render: np.ndarray,
                           frame_name: str) -> np.ndarray:
    """Create a side-by-side comparison image."""
    # Resize to same height
    h = min(real_frame.shape[0], bim_render.shape[0])
    
    from PIL import Image as PILImage
    real_pil = PILImage.fromarray(real_frame).resize(
        (int(real_frame.shape[1] * h / real_frame.shape[0]), h))
    bim_pil = PILImage.fromarray(bim_render).resize(
        (int(bim_render.shape[1] * h / bim_render.shape[0]), h))
    
    real_np = np.array(real_pil)
    bim_np = np.array(bim_pil)
    
    # Side by side with divider
    divider = np.full((h, 4, 3), 255, dtype=np.uint8)
    comparison = np.concatenate([real_np, divider, bim_np], axis=1)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Render BIM from camera poses")
    parser.add_argument("--project", type=str, required=True,
                       help="Project directory (e.g. server/data/projects/live_xxx)")
    parser.add_argument("--scan", type=str, required=True,
                       help="Scan date folder (e.g. 2026-02-06)")
    parser.add_argument("--output", type=str, default="/tmp/bim_render_test",
                       help="Output directory")
    parser.add_argument("--max-frames", type=int, default=5,
                       help="Max number of keyframes to render")
    parser.add_argument("--resolution-scale", type=float, default=0.5,
                       help="Scale factor for render resolution (0.5 = half res)")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Resolve paths
    project_dir = Path(args.project)
    scan_dir = project_dir / "scans" / args.scan / "src_legacy"
    frames_dir = scan_dir / "frames"
    output_dir = scan_dir / "output"
    maplong_dir = output_dir / "maplong_run"
    ifcs_dir = project_dir / "ifcs"
    
    # 1. Load keyframe selection
    selected = load_selected_frames(str(frames_dir))
    print(f"[BIM Render] {len(selected)} keyframes from selected_frames.json")
    
    # 2. Load camera poses (from maplong_run — matches keyframe order)
    poses = load_camera_poses(str(maplong_dir / "camera_poses.txt"))
    print(f"[BIM Render] {len(poses)} camera poses loaded")
    
    # 3. Load intrinsics
    intrinsics = load_intrinsics(str(maplong_dir / "intrinsic.txt"))
    print(f"[BIM Render] {len(intrinsics)} intrinsics loaded")
    
    # 4. Load floor transform (scan → BIM)
    ft_s, ft_R, ft_t = load_floor_transform(str(output_dir / "floor_transform.npz"))
    print(f"[BIM Render] Floor transform loaded (Sim3)")
    print(f"  s = {ft_s}")
    print(f"  R = \n{ft_R}")
    print(f"  t = {ft_t}")
    
    # 5. Load IFC mesh
    ifc_files = list(ifcs_dir.glob("*.ifc"))
    if not ifc_files:
        print("[BIM Render] ERROR: No IFC file found")
        return
    
    bim_mesh, element_names, face_to_element = load_ifc_mesh(str(ifc_files[0]))
    print(f"[BIM Render] BIM mesh (IFC Z-up): {len(bim_mesh.vertices)} verts, {len(bim_mesh.faces)} faces")
    print(f"[BIM Render] BIM bounds (Z-up): {bim_mesh.bounds}")
    
    # ifcopenshell outputs Z-up, but Three.js/viewer uses Y-up.
    # floor_transform was computed in Y-up viewer space.
    # Rotate BIM: x stays, y_new = z_old, z_new = -y_old
    verts = bim_mesh.vertices.copy()
    verts_yup = np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])
    bim_mesh.vertices = verts_yup
    print(f"[BIM Render] BIM bounds (Y-up): {bim_mesh.bounds}")
    
    # Verify data consistency
    assert len(poses) == len(selected), \
        f"Mismatch: {len(poses)} poses vs {len(selected)} keyframes"
    assert len(intrinsics) == len(selected), \
        f"Mismatch: {len(intrinsics)} intrinsics vs {len(selected)} keyframes"
    
    # 6. Render BIM from each camera pose
    n_render = min(args.max_frames, len(poses))
    print(f"\n[BIM Render] Rendering {n_render} BIM views...")
    
    for i in range(n_render):
        frame_name = selected[i]
        frame_path = frames_dir / frame_name
        
        # Load real frame
        real_frame = np.array(Image.open(str(frame_path)).convert("RGB"))
        h_real, w_real = real_frame.shape[:2]
        
        # Scale intrinsics from MapAnything processing resolution to original frame size
        # intrinsics were calibrated for ~(2*cx, 2*cy) resolution
        raw_intr = intrinsics[i]
        scale_w = w_real / (2 * raw_intr["cx"])
        scale_h = h_real / (2 * raw_intr["cy"])
        scale = (scale_w + scale_h) / 2  # average (should be nearly equal)
        intr = {
            "fx": raw_intr["fx"] * scale,
            "fy": raw_intr["fy"] * scale,
            "cx": w_real / 2.0,
            "cy": h_real / 2.0,
        }
        w_render = w_real
        h_render = h_real
        
        # Poses are already c2w (camera-to-world)
        c2w_scan = poses[i]
        
        # Transform camera pose to BIM space
        # Position: p_bim = s * R @ p_scan + t
        # Orientation: R_bim = R @ R_scan (pure rotation, no scale)
        c2w_bim = np.eye(4)
        c2w_bim[:3, :3] = ft_R @ c2w_scan[:3, :3]  # rotate orientation (no scale!)
        c2w_bim[:3, 3] = ft_s * ft_R @ c2w_scan[:3, 3] + ft_t  # transform position
        
        print(f"\n[BIM Render] Frame {i+1}/{n_render}: {frame_name}")
        print(f"  Real: {w_real}x{h_real}, Render: {w_render}x{h_render}")
        fov_y_deg = np.degrees(2 * np.arctan(intr['cy'] / intr['fy']))
        fov_x_deg = np.degrees(2 * np.arctan(intr['cx'] / intr['fx']))
        print(f"  FOV: {fov_x_deg:.1f}° x {fov_y_deg:.1f}° (fx={intr['fx']:.1f} fy={intr['fy']:.1f})")
        print(f"  Camera pos (scan):  {c2w_scan[:3, 3]}")
        print(f"  Camera pos (BIM):   {c2w_bim[:3, 3]}")
        
        # Render
        t0 = time.time()
        bim_rgb, depth_map, face_id_buf = render_raytrace(
            bim_mesh, c2w_bim, intr, w_render, h_render)
        
        t_render = time.time() - t0
        print(f"  Render time: {t_render:.1f}s")
        
        # Identify visible IFC elements
        from collections import Counter
        visible_faces = face_id_buf[face_id_buf >= 0]
        legend_entries = []
        if len(visible_faces) > 0:
            visible_elements = face_to_element[visible_faces]
            pixel_counts = Counter(visible_elements.tolist())
            total_pixels = w_render * h_render
            print(f"  Visible elements:")
            for elem_idx, px_count in pixel_counts.most_common(10):
                pct = px_count / total_pixels * 100
                name = element_names[elem_idx]
                print(f"    {pct:5.1f}% | {name}")
                if pct > 0.5:  # only show elements covering >0.5%
                    legend_entries.append((name, pct))
        else:
            print(f"  ⚠️ No BIM geometry visible from this pose!")
        
        # Draw legend on BIM image
        from PIL import ImageDraw, ImageFont
        bim_pil = Image.fromarray(bim_rgb)
        draw = ImageDraw.Draw(bim_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            font_small = font
        
        # Semi-transparent background for legend
        if legend_entries:
            y_pos = 10
            # Draw title
            draw.rectangle([5, y_pos - 2, 500, y_pos + 22], fill=(0, 0, 0, 180))
            draw.text((10, y_pos), f"BIM Elements — {frame_name}", fill=(255, 255, 255), font=font)
            y_pos += 28
            
            for name, pct in legend_entries:
                # Parse IFC type and element name
                parts = name.split(":", 1)
                ifc_type = parts[0]
                elem_name = parts[1] if len(parts) > 1 else ""
                # Get the color for this type
                color = tuple(_ifc_type_color(ifc_type)[:3])
                
                # Draw background
                draw.rectangle([5, y_pos - 2, 500, y_pos + 20], fill=(0, 0, 0, 180))
                # Color swatch
                draw.rectangle([10, y_pos + 1, 28, y_pos + 17], fill=color, outline=(255, 255, 255))
                # Text
                label = f"{pct:4.1f}% {ifc_type}"
                if elem_name:
                    label += f" — {elem_name[:40]}"
                draw.text((34, y_pos), label, fill=(255, 255, 255), font=font_small)
                y_pos += 24
        
        bim_rgb = np.array(bim_pil)
        
        # Save individual images
        Image.fromarray(bim_rgb).save(
            os.path.join(args.output, f"bim_{frame_name}"))
        
        if depth_map is not None:
            # Normalize depth for visualization
            valid = depth_map < np.inf
            if np.any(valid):
                d_min, d_max = depth_map[valid].min(), depth_map[valid].max()
                depth_vis = np.zeros_like(depth_map)
                depth_vis[valid] = (depth_map[valid] - d_min) / (d_max - d_min + 1e-8)
                depth_vis = (depth_vis * 255).astype(np.uint8)
                Image.fromarray(depth_vis).save(
                    os.path.join(args.output, f"depth_{frame_name}"))
        
        # Create comparison
        comparison = create_comparison_image(real_frame, bim_rgb, frame_name)
        Image.fromarray(comparison).save(
            os.path.join(args.output, f"compare_{frame_name}"))
        
        print(f"  Saved: bim_{frame_name}, depth_{frame_name}, compare_{frame_name}")
    
    print(f"\n[BIM Render] ✅ Done! {n_render} renders saved to {args.output}")


if __name__ == "__main__":
    main()
