#!/usr/bin/env python3
"""
Dry-run: simula exactamente lo que hará HybridDepthAdapter con la corrección
polinomial, usando los 40 frames ya extraídos. Verifica que no haya errores
y muestra las estadísticas reales de fusión.
"""
import numpy as np
import cv2
from pathlib import Path

OUTPUT_DIR = Path("/home/hernan/stac-builder/server/test2_hybrid_output")
STRAY_DIR = Path("/home/hernan/stac-builder/server/test2")
FRAMES_DIR = OUTPUT_DIR / "frames"
DA3_DIR = OUTPUT_DIR / "da3_depths"

LIDAR_MAX_TRUST_M = 2.0
H, W = 192, 256

frame_files = sorted(FRAMES_DIR.glob("*.jpg"))
print(f"Frames: {len(frame_files)}")

total_lidar_px = 0
total_da3_px = 0
total_px = 0
all_poly_coeffs = []

for frame_file in frame_files:
    fname = frame_file.stem
    frame_idx = int(fname)
    
    depth_png = STRAY_DIR / "depth" / f"{frame_idx:06d}.png"
    depth_raw = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED)
    depth_lidar = depth_raw.astype(np.float32) / 1000.0
    if depth_lidar.shape != (H, W):
        depth_lidar = cv2.resize(depth_lidar, (W, H), interpolation=cv2.INTER_NEAREST)
    
    da3_path = DA3_DIR / f"{fname}.npy"
    depth_da3 = np.load(str(da3_path))
    depth_da3_small = cv2.resize(depth_da3, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Same logic as HybridDepthAdapter
    valid_lidar = (depth_lidar > 0) & (depth_lidar < LIDAR_MAX_TRUST_M)
    calib_mask = valid_lidar & (depth_lidar > 0.3)
    if calib_mask.sum() < 100:
        calib_mask = valid_lidar
    
    if calib_mask.sum() > 50:
        lidar_vals = depth_lidar[calib_mask]
        da3_vals = depth_da3_small[calib_mask]
        poly_coeffs = np.polyfit(da3_vals, lidar_vals, 2)
        depth_da3_corrected = np.polyval(poly_coeffs, depth_da3_small)
    else:
        depth_da3_corrected = depth_da3_small.copy()
        poly_coeffs = [0, 1, 0]
    
    depth_da3_corrected = np.clip(depth_da3_corrected, 0, 20.0)
    
    depth_hybrid = depth_da3_corrected.copy()
    depth_hybrid[valid_lidar] = depth_lidar[valid_lidar]
    
    valid_hybrid = (depth_hybrid > 0) & (depth_hybrid < 20.0)
    
    n_lidar = int(valid_lidar.sum())
    da3_pixels = ~valid_lidar & (depth_hybrid > 0)
    n_da3 = int(da3_pixels.sum())
    total_lidar_px += n_lidar
    total_da3_px += n_da3
    total_px += H * W
    all_poly_coeffs.append(poly_coeffs)
    
    # Per-frame stats
    da3_max = depth_da3_small.max()
    corrected_max = depth_da3_corrected[da3_pixels].max() if da3_pixels.sum() > 0 else 0
    hybrid_max = depth_hybrid[valid_hybrid].max() if valid_hybrid.sum() > 0 else 0
    
    print(f"  {fname}: LiDAR={n_lidar:>5} DA3={n_da3:>5} | "
          f"DA3 raw max={da3_max:.2f}m → corrected max={corrected_max:.2f}m | "
          f"poly=[{poly_coeffs[0]:.3f}, {poly_coeffs[1]:.3f}, {poly_coeffs[2]:.3f}]")

print(f"\n{'='*70}")
print(f"RESUMEN")
print(f"{'='*70}")
print(f"LiDAR (<{LIDAR_MAX_TRUST_M}m): {total_lidar_px:,} ({100*total_lidar_px/total_px:.1f}%)")
print(f"DA3 (>={LIDAR_MAX_TRUST_M}m):  {total_da3_px:,} ({100*total_da3_px/total_px:.1f}%)")
print(f"Total pixels:       {total_px:,}")

# Polynomial stability
coeffs_arr = np.array(all_poly_coeffs)
print(f"\nEstabilidad polinomial entre frames:")
print(f"  a (cuadrático): media={coeffs_arr[:,0].mean():.4f} std={coeffs_arr[:,0].std():.4f}")
print(f"  b (lineal):     media={coeffs_arr[:,1].mean():.4f} std={coeffs_arr[:,1].std():.4f}")
print(f"  c (offset):     media={coeffs_arr[:,2].mean():.4f} std={coeffs_arr[:,2].std():.4f}")

# Extrapolation check for 15-20m target
print(f"\nExtrapolación del polinomio promedio:")
avg_poly = np.poly1d(coeffs_arr.mean(axis=0))
for da3_val in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
    corrected = avg_poly(da3_val)
    print(f"  DA3={da3_val:.1f}m → Corregido={corrected:.2f}m")

print(f"\n✅ Dry-run completado sin errores. Pipeline listo para ejecutar.")
