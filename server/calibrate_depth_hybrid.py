#!/usr/bin/env python3
"""
calibrate_depth_hybrid.py — Calibra depth maps de DA3 usando profundidad LiDAR.

Para cada frame:
  1. Lee el depth map de DA3 (da3_full/{stem}_depth.npy)
  2. Lee el depth map LiDAR correspondiente (Stray Scanner depth/)
  3. Ajuste lineal robusto: DA3 vs LiDAR en zona 0-{lidar_trust_m}m
  4. Aplica corrección a TODO el rango del depth map DA3
  5. Fusión: LiDAR gana donde es válido, DA3 calibrado en el resto
  6. Sobreescribe da3_full/{stem}_depth.npy con el depth calibrado
  7. Sobreescribe da3_full/extrinsics.npy con poses ARKit (desde odometry.csv)

Reutiliza _robust_linear_fit de stray_da3_streaming.py (caso de éxito probado).

Uso:
  python calibrate_depth_hybrid.py \\
    --da3_dir  .../output/gaus_slam_run/da3_full \\
    --stray_dir .../stray_scanner_raw \\
    --lidar_trust_m 5.0 \\
    --calib_min_depth 0.3 \\
    --da3_max_range 20.0

Hernán Barreto — Ingerop IN3
"""

import argparse
import csv
import json
import sys
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


# ── Calibración lineal robusta (idéntica a stray_da3_streaming.py) ────────────

def _robust_linear_fit(da3_vals, lidar_vals, inlier_rounds=2, inlier_sigma=3.0):
    """Ajuste: depth_corrected = da3 * scale + offset con rechazo de outliers."""
    mask = np.ones(len(da3_vals), dtype=bool)
    for _ in range(inlier_rounds):
        if mask.sum() < 20:
            break
        coeffs = np.polyfit(da3_vals[mask], lidar_vals[mask], 1)
        residuals = lidar_vals - np.polyval(coeffs, da3_vals)
        mad = np.median(np.abs(residuals[mask]))
        mask = np.abs(residuals) < inlier_sigma * mad * 1.4826

    if mask.sum() > 20:
        scale, offset = np.polyfit(da3_vals[mask], lidar_vals[mask], 1)
    else:
        scale, offset = np.polyfit(da3_vals, lidar_vals, 1)

    if scale <= 0:
        scale, offset = 1.0, 0.0
    return float(scale), float(offset)


# ── Carga poses ARKit desde odometry.csv ──────────────────────────────────────

def load_arkit_poses(stray_dir: Path) -> dict:
    """
    Lee odometry.csv de Stray Scanner.
    Retorna dict: frame_idx -> w2c [3,4] float32
    """
    odo_path = stray_dir / "odometry.csv"
    if not odo_path.exists():
        raise FileNotFoundError(f"No se encontró odometry.csv en {stray_dir}")

    poses = {}
    with open(odo_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            frame_idx = int(row["frame"])
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            qx, qy, qz, qw = (float(row["qx"]), float(row["qy"]),
                               float(row["qz"]), float(row["qw"]))
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rot
            c2w[:3, 3] = [x, y, z]
            w2c = np.linalg.inv(c2w)[:3, :].astype(np.float32)
            poses[frame_idx] = w2c
    return poses


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibra depth maps DA3 con LiDAR e inyecta poses ARKit"
    )
    parser.add_argument("--da3_dir",       required=True,
                        help="Directorio da3_full/ generado por extract_da3_full.py")
    parser.add_argument("--stray_dir",     required=True,
                        help="Directorio raw de Stray Scanner (con depth/, odometry.csv)")
    parser.add_argument("--lidar_trust_m", type=float, default=5.0,
                        help="Rango máximo de LiDAR confiable en metros (default: 5.0)")
    parser.add_argument("--calib_min_depth", type=float, default=0.3,
                        help="Profundidad mínima para zona de calibración (default: 0.3m)")
    parser.add_argument("--da3_max_range", type=float, default=20.0,
                        help="Clip máximo para depth DA3 calibrado (default: 20.0m)")
    args = parser.parse_args()

    da3_dir   = Path(args.da3_dir)
    stray_dir = Path(args.stray_dir)

    # ── Cargar manifest ──────────────────────────────────────────────────────
    manifest_path = da3_dir / "da3_manifest.json"
    if not manifest_path.exists():
        print(f"[HybridCalib] ❌ No se encontró da3_manifest.json en {da3_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    frame_files = manifest["frame_files"]          # lista de "NNNNNN.jpg"
    n_frames    = len(frame_files)
    print(f"[HybridCalib] Frames a calibrar: {n_frames}")
    print(f"[HybridCalib] LiDAR trust range: 0–{args.lidar_trust_m}m")

    # ── Cargar extrinsics DA3 actuales (serán reemplazados por ARKit) ─────────
    ext_path = da3_dir / "extrinsics.npy"
    extrinsics_da3 = np.load(str(ext_path))        # [N, 3, 4]

    # ── Cargar poses ARKit ───────────────────────────────────────────────────
    print("[HybridCalib] Cargando poses ARKit desde odometry.csv...")
    arkit_poses = load_arkit_poses(stray_dir)
    print(f"[HybridCalib] {len(arkit_poses)} poses ARKit cargadas")

    # ── Stray Scanner depth dir ──────────────────────────────────────────────
    lidar_depth_dir = stray_dir / "depth"
    if not lidar_depth_dir.exists():
        print(f"[HybridCalib] ❌ No se encontró {lidar_depth_dir}")
        sys.exit(1)

    # ── Procesar frame por frame ─────────────────────────────────────────────
    extrinsics_out = extrinsics_da3.copy()
    n_calibrated   = 0
    n_arkit_injected = 0
    scales_log = []

    for i, fname in enumerate(frame_files):
        stem = Path(fname).stem                    # "NNNNNN"
        frame_idx = int(stem)

        # ── 1. Cargar depth DA3 ───────────────────────────────────────────
        da3_npy = da3_dir / f"{stem}_depth.npy"
        if not da3_npy.exists():
            continue
        depth_da3 = np.load(str(da3_npy)).astype(np.float32)  # (H, W) metros
        da3_H, da3_W = depth_da3.shape

        # ── 2. Cargar depth LiDAR (PNG uint16 mm) ────────────────────────
        lidar_png = lidar_depth_dir / f"{stem}.png"
        if not lidar_png.exists():
            # Sin LiDAR para este frame — skip calibración, mantener DA3
            continue

        depth_lidar_mm = cv2.imread(str(lidar_png), cv2.IMREAD_ANYDEPTH)
        if depth_lidar_mm is None:
            continue
        depth_lidar_m = depth_lidar_mm.astype(np.float32) / 1000.0  # mm → m

        lidar_H, lidar_W = depth_lidar_m.shape

        # ── 3. Escalar LiDAR a resolución DA3 (INTER_NEAREST para depth) ──
        # Mismo patrón que stray_da3_streaming.py (caso de éxito)
        depth_lidar_resized = cv2.resize(
            depth_lidar_m, (da3_W, da3_H),
            interpolation=cv2.INTER_NEAREST
        )

        # ── 4. Máscara de calibración: zona LiDAR válida ──────────────────
        valid_lidar = (
            (depth_lidar_resized > args.calib_min_depth) &
            (depth_lidar_resized < args.lidar_trust_m)
        )

        # ── 5. Ajuste lineal robusto DA3 → LiDAR ─────────────────────────
        if valid_lidar.sum() > 50 and depth_da3.max() > 0.01:
            scale, offset = _robust_linear_fit(
                depth_da3[valid_lidar],
                depth_lidar_resized[valid_lidar]
            )
            da3_calibrated = depth_da3 * scale + offset
            da3_calibrated = np.clip(da3_calibrated, 0.0, args.da3_max_range)
            n_calibrated += 1
            scales_log.append(scale)
        else:
            da3_calibrated = depth_da3.copy()
            scale, offset = 1.0, 0.0

        # ── 6. Fusión: LiDAR gana donde es válido ────────────────────────
        depth_fused = da3_calibrated.copy()
        depth_fused[valid_lidar] = depth_lidar_resized[valid_lidar]

        # ── 7. Sobreescribir depth en da3_full/ ───────────────────────────
        np.save(str(da3_npy), depth_fused.astype(np.float32))

        # ── 8. Inyectar pose ARKit ────────────────────────────────────────
        if frame_idx in arkit_poses:
            extrinsics_out[i] = arkit_poses[frame_idx]
            n_arkit_injected += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_frames:
            print(f"  [{i+1}/{n_frames}] frame {stem} | "
                  f"scale={scale:.4f} offset={offset:.4f} | "
                  f"LiDAR px={valid_lidar.sum()}")

    # ── Guardar extrinsics actualizados (poses ARKit) ────────────────────────
    np.save(str(ext_path), extrinsics_out)

    print(f"\n[HybridCalib] ✅ Calibración completa")
    print(f"  Frames calibrados con LiDAR: {n_calibrated}/{n_frames}")
    print(f"  Poses ARKit inyectadas:       {n_arkit_injected}/{n_frames}")
    if scales_log:
        print(f"  Scale promedio DA3→LiDAR:    {np.mean(scales_log):.4f} "
              f"(±{np.std(scales_log):.4f})")
    print(f"  Depth maps actualizados en:  {da3_dir}/")
    print(f"  Poses ARKit guardadas en:    {ext_path}")


if __name__ == "__main__":
    main()
