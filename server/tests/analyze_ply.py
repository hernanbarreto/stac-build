import open3d as o3d
import numpy as np
import sys

def main(ply_path):
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Number of points: {len(points)}")
    
    if len(points) == 0:
        print("Empty point cloud!")
        return

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"Bounding box extent (X, Y, Z): {extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f} meters")
    print(f"Bounding box center:     {bbox.get_center()[0]:.3f}, {bbox.get_center()[1]:.3f}, {bbox.get_center()[2]:.3f}")
    
    # Calculate density metrics
    print("Computing nearest neighbor distances...")
    distances = pcd.compute_nearest_neighbor_distance()
    distances = np.asarray(distances)
    print(f"Average NN distance: {np.mean(distances):.6f} m")
    print(f"Std dev of NN distance: {np.std(distances):.6f} m")
    
    colors = np.asarray(pcd.colors)
    if len(colors) > 0:
        print("Color distribution check passed. Point cloud has RGB attributes.")
    else:
        print("Warning: point cloud lacks color attributes.")

    print("Done")

if __name__ == '__main__':
    main("/home/hernan/stac-builder/server/test2_hybrid_output/vggt_hybrid.ply")
