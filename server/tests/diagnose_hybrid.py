"""
Diagnóstico: ¿El pipeline híbrido está realmente usando DA3?
Compara los mapas de profundidad LiDAR vs DA3 calibrado para cada frame
y reporta cuántos pixeles vienen de cada fuente.
"""
import numpy as np
import cv2
from pathlib import Path
import json

# --- Paths (match the hybrid pipeline) ---
OUTPUT_DIR = Path("/home/hernan/stac-builder/server/test2_hybrid_output")
FRAMES_DIR = OUTPUT_DIR / "frames"
DA3_DIR = OUTPUT_DIR / "da3_depths"
STRAY_DIR = Path("/home/hernan/stac-builder/server/test2")

# --- Load depth intrinsics from stray data ---
depth_dir = STRAY_DIR / "depth"
depth_files = sorted(depth_dir.glob("*.png"))

# Load LiDAR depth frames used (stride=4, max_frames=40)
frame_files = sorted(FRAMES_DIR.glob("*.jpg"))
print(f"Found {len(frame_files)} RGB frames")
print(f"Found {len(list(DA3_DIR.glob('*.npy')))} DA3 depth maps")

H, W = 192, 256  # Native depth resolution

total_pixels = 0
total_lidar_valid = 0
total_da3_only = 0
total_both_valid = 0
total_neither = 0

for frame_file in frame_files:
    fname = frame_file.stem  # e.g. "000000"
    
    # Load LiDAR depth
    frame_idx = int(fname)
    depth_png = STRAY_DIR / "depth" / f"{frame_idx:06d}.png"
    
    if depth_png.exists():
        depth_raw = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED)
        if depth_raw is not None:
            depth_lidar = depth_raw.astype(np.float32) / 1000.0  # mm → meters
            # Resize to native resolution if needed
            if depth_lidar.shape != (H, W):
                depth_lidar = cv2.resize(depth_lidar, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            depth_lidar = np.zeros((H, W), dtype=np.float32)
    else:
        depth_lidar = np.zeros((H, W), dtype=np.float32)
    
    # Load DA3 depth
    da3_path = DA3_DIR / f"{fname}.npy"
    if da3_path.exists():
        depth_da3 = np.load(str(da3_path))
        depth_da3_small = cv2.resize(depth_da3, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        depth_da3_small = np.zeros((H, W), dtype=np.float32)
    
    # Analyze
    lidar_valid = depth_lidar > 0
    da3_valid = depth_da3_small > 0
    
    n_pixels = H * W
    n_lidar = lidar_valid.sum()
    n_da3 = da3_valid.sum()
    n_both = (lidar_valid & da3_valid).sum()
    n_da3_only = (da3_valid & ~lidar_valid).sum()
    n_lidar_only = (lidar_valid & ~da3_valid).sum()
    n_neither = (~lidar_valid & ~da3_valid).sum()
    
    total_pixels += n_pixels
    total_lidar_valid += n_lidar
    total_da3_only += n_da3_only
    total_both_valid += n_both
    total_neither += n_neither
    
    lidar_pct = 100.0 * n_lidar / n_pixels
    da3_only_pct = 100.0 * n_da3_only / n_pixels
    
    # Also compute depth ranges
    if n_lidar > 0:
        lidar_min = depth_lidar[lidar_valid].min()
        lidar_max = depth_lidar[lidar_valid].max()
        lidar_range = f"{lidar_min:.2f}-{lidar_max:.2f}m"
    else:
        lidar_range = "N/A"
    
    print(f"  {fname}: LiDAR={lidar_pct:5.1f}% ({n_lidar}/{n_pixels}), "
          f"DA3-only={da3_only_pct:5.1f}% ({n_da3_only}), "
          f"LiDAR range={lidar_range}")

print("\n" + "=" * 70)
print("RESUMEN GLOBAL")
print("=" * 70)
print(f"Total pixeles procesados: {total_pixels:,}")
print(f"Pixeles con LiDAR válido: {total_lidar_valid:,} ({100*total_lidar_valid/total_pixels:.1f}%)")
print(f"Pixeles SOLO DA3 (contribución real): {total_da3_only:,} ({100*total_da3_only/total_pixels:.1f}%)")
print(f"Pixeles con ambos (LiDAR gana): {total_both_valid:,} ({100*total_both_valid/total_pixels:.1f}%)")
print(f"Pixeles sin dato: {total_neither:,} ({100*total_neither/total_pixels:.1f}%)")

if total_da3_only == 0:
    print("\n⚠️  DA3 NO ESTÁ CONTRIBUYENDO NINGÚN PIXEL.")
    print("   El LiDAR cubre el 100% de la geometría visible.")
    print("   La nube de puntos es PURAMENTE LiDAR (DA3 fue redundante).")
elif total_da3_only < total_pixels * 0.01:
    print(f"\n⚠️  DA3 contribuye menos del 1% de los pixeles.")
    print("   Su contribución es marginal en esta escena.")
else:
    print(f"\n✅  DA3 contribuye {100*total_da3_only/total_pixels:.1f}% de los pixeles.")
    print("   La fusión híbrida está funcionando como se esperaba.")
