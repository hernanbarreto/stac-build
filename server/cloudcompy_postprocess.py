#!/usr/bin/env python3
"""
CloudCompPy Professional Point Cloud Post-Processing
=====================================================
Industrial-grade cleaning pipeline for reconstructed point clouds.
Runs in the CloudComPy310 conda environment as a subprocess.

Optimized Pipeline Order:
  1. Load & merge all chunk PLYs
  2. Remove exact duplicate points
  3. Voxel spatial subsampling (reduces density FIRST for performance)
  4. Statistical Outlier Removal (SOR) — on reduced cloud
  5. Noise filter (radius-based) — on reduced cloud
  6. Normal estimation + consistent orientation
  7. Save cleaned PLY with normals
"""
import sys
import os
import argparse
import time
import glob


def _clone_from_ref(source_cloud, ref_cloud, step_name="step"):
    """Convert ReferenceCloud to ccPointCloud via partialClone."""
    clone_result = source_cloud.partialClone(ref_cloud)
    if clone_result is None:
        print(f"  ⚠️  {step_name}: partialClone returned None, keeping previous cloud")
        return source_cloud
    cloud = clone_result[0] if isinstance(clone_result, tuple) else clone_result
    return cloud


def main():
    parser = argparse.ArgumentParser(description="CloudCompPy professional point cloud cleaning")
    parser.add_argument("--input-dir", required=True, help="Directory with chunk_XXX.ply files")
    parser.add_argument("--output", required=True, help="Output cleaned PLY path")
    
    # Voxel subsampling
    parser.add_argument("--voxel-size", type=float, default=0.001,
                        help="Voxel size in meters (default: 0.001 = 1mm)")
    
    # SOR parameters
    parser.add_argument("--sor-knn", type=int, default=6,
                        help="SOR: number of nearest neighbors (default: 6)")
    parser.add_argument("--sor-sigma", type=float, default=1.0,
                        help="SOR: sigma multiplier (default: 1.0)")
    
    # Noise filter parameters
    parser.add_argument("--noise-radius", type=float, default=0.05,
                        help="Noise filter kernel radius in meters (default: 0.05 = 50mm)")
    parser.add_argument("--noise-sigma", type=float, default=3.0,
                        help="Noise filter nSigma (default: 3.0, conservative)")
    
    # Max points
    parser.add_argument("--max-points", type=int, default=0,
                        help="Max points in output (0=unlimited)")
    
    # Skip flags
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip duplicate removal")
    parser.add_argument("--skip-sor", action="store_true", help="Skip SOR filter")
    parser.add_argument("--skip-noise", action="store_true", help="Skip noise filter")
    parser.add_argument("--skip-normals", action="store_true", help="Skip normal estimation")
    
    args = parser.parse_args()

    # ── Import CloudCompPy ──
    try:
        import cloudComPy as cc
        cc.initCC()
        print("[CloudCompPy] ✅ Initialized successfully")
    except ImportError as e:
        print(f"[CloudCompPy] ❌ Failed to import: {e}")
        sys.exit(1)

    # ── Find chunk PLYs ──
    chunk_files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*.ply")))
    if not chunk_files:
        chunk_files = sorted(glob.glob(os.path.join(args.input_dir, "*_pcd.ply")))
    if not chunk_files:
        print(f"[CloudCompPy] ❌ No chunk PLY files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"\n{'='*65}")
    print(f"  PROFESSIONAL POINT CLOUD CLEANING PIPELINE")
    print(f"  Chunks: {len(chunk_files)}  |  Voxel: {args.voxel_size*1000:.1f}mm")
    print(f"  SOR: knn={args.sor_knn}, σ={args.sor_sigma}")
    print(f"  Noise: r={args.noise_radius*1000:.0f}mm, σ={args.noise_sigma}")
    print(f"{'='*65}\n")

    t_pipeline = time.time()

    # ══════════════════════════════════════════════════════════════
    # STEP 1: LOAD & MERGE
    # ══════════════════════════════════════════════════════════════
    t0 = time.time()
    print(f"[Step 1/6] Loading and merging {len(chunk_files)} chunks...")
    
    clouds = []
    total_input = 0
    for i, ply_path in enumerate(chunk_files):
        cloud = cc.loadPointCloud(ply_path)
        if cloud is None:
            print(f"  ⚠️  Failed to load {os.path.basename(ply_path)}, skipping")
            continue
        n_pts = cloud.size()
        total_input += n_pts
        clouds.append(cloud)
        print(f"  [{i+1}/{len(chunk_files)}] {os.path.basename(ply_path)}: {n_pts:,} pts")

    if not clouds:
        print("[CloudCompPy] ❌ No clouds loaded")
        sys.exit(1)

    # Merge all at once
    if len(clouds) == 1:
        current = clouds[0]
    else:
        current = cc.MergeEntities(clouds, deleteOriginalClouds=True, createSFcloudIndex=False)
        if current is None:
            print("[CloudCompPy] ❌ MergeEntities failed")
            sys.exit(1)
    
    current.setName("merged")
    print(f"  ✅ Merged: {current.size():,} points ({time.time()-t0:.1f}s)\n")

    # ── Step 1b: Inject origin scalar fields from .npz ──
    origin_files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*_origins.npz")))
    if origin_files:
        import numpy as np
        t_orig = time.time()
        
        all_fg, all_pr, all_pc = [], [], []
        for of in origin_files:
            d = np.load(of)
            all_fg.append(d["frame_global"].astype(np.float32))
            all_pr.append(d["pixel_row"].astype(np.float32))
            all_pc.append(d["pixel_col"].astype(np.float32))
        
        fg = np.concatenate(all_fg)
        pr = np.concatenate(all_pr)
        pc = np.concatenate(all_pc)
        
        n_cloud = current.size()
        if len(fg) == n_cloud:
            for name, arr in [('frame_global', fg), ('pixel_row', pr), ('pixel_col', pc)]:
                idx = current.addScalarField(name)
                sf = current.getScalarField(idx)
                sf.fromNpArrayCopy(arr)
            print(f"  ✅ Injected origin scalar fields ({n_cloud:,} pts) ({time.time()-t_orig:.1f}s)\n")
        else:
            print(f"  ⚠️ Origin size mismatch: {len(fg)} vs cloud {n_cloud} — origins NOT injected\n")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: NEAR-DUPLICATE REMOVAL (micro-voxel 0.1mm)
    # Points from overlapping chunks are nearly identical but not
    # bit-exact.  A tiny voxel pass collapses them efficiently.
    # ══════════════════════════════════════════════════════════════
    if not args.skip_duplicates:
        t1 = time.time()
        n_before = current.size()
        MICRO_VOXEL = 0.0001  # 0.1mm — catches near-duplicates without losing detail
        print(f"[Step 2/6] Removing near-duplicate points (micro-voxel {MICRO_VOXEL*1000:.1f}mm)...")
        
        ref_micro = cc.CloudSamplingTools.resampleCloudSpatially(current, MICRO_VOXEL)
        if ref_micro is not None and ref_micro.size() > 0:
            deduped = _clone_from_ref(current, ref_micro, "NearDedup")
            n_removed = n_before - deduped.size()
            pct = (n_removed / n_before) * 100
            print(f"  ✅ {n_before:,} → {deduped.size():,} ({n_removed:,} near-duplicates, {pct:.1f}%)")
            current = deduped
        else:
            print(f"  ℹ️  Micro-voxel returned None, keeping original")
        
        print(f"  ({time.time()-t1:.1f}s)\n")
    else:
        print(f"[Step 2/6] Near-duplicate removal: SKIPPED\n")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: VOXEL SPATIAL SUBSAMPLING
    # (BEFORE SOR/noise for performance — reduce 13M→5M first)
    # ══════════════════════════════════════════════════════════════
    t3 = time.time()
    n_before = current.size()
    print(f"[Step 3/6] Voxel spatial subsampling ({args.voxel_size*1000:.1f}mm)...")
    
    ref_voxel = cc.CloudSamplingTools.resampleCloudSpatially(current, args.voxel_size)
    
    if ref_voxel is not None and ref_voxel.size() > 0:
        current = _clone_from_ref(current, ref_voxel, "Voxel")
        reduction = (1 - current.size() / n_before) * 100
        print(f"  ✅ {n_before:,} → {current.size():,} ({reduction:.1f}% reduction)")
    else:
        print(f"  ⚠️  Voxel returned None, keeping original")
    
    current.setName("subsampled")
    print(f"  ({time.time()-t3:.1f}s)\n")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: STATISTICAL OUTLIER REMOVAL (SOR)
    # (on reduced cloud — fast and effective)
    # ══════════════════════════════════════════════════════════════
    if not args.skip_sor:
        t4 = time.time()
        n_before = current.size()
        print(f"[Step 4/6] Statistical Outlier Removal (knn={args.sor_knn}, σ={args.sor_sigma})...")
        
        ref_sor = cc.CloudSamplingTools.sorFilter(current, knn=args.sor_knn, nSigma=args.sor_sigma)
        
        if ref_sor is not None and ref_sor.size() > 0:
            current = _clone_from_ref(current, ref_sor, "SOR")
            n_removed = n_before - current.size()
            pct = (n_removed / n_before) * 100
            print(f"  ✅ {n_before:,} → {current.size():,} ({n_removed:,} outliers, {pct:.1f}%)")
        else:
            print(f"  ⚠️  SOR returned None")
        
        current.setName("sor_cleaned")
        print(f"  ({time.time()-t4:.1f}s)\n")
    else:
        print(f"[Step 4/6] SOR: SKIPPED\n")

    # ══════════════════════════════════════════════════════════════
    # STEP 5: NOISE FILTER (radius-based)
    # (on reduced+SOR cloud — catches remaining isolated points)
    # ══════════════════════════════════════════════════════════════
    if not args.skip_noise:
        t5 = time.time()
        n_before = current.size()
        print(f"[Step 5/6] Noise filter (r={args.noise_radius*1000:.0f}mm, σ={args.noise_sigma})...")
        
        ref_noise = cc.CloudSamplingTools.noiseFilter(current, args.noise_radius, args.noise_sigma)
        
        if ref_noise is not None and ref_noise.size() > 0:
            # Safety check: if noise filter removes >50%, it's too aggressive
            removal_pct = (1 - ref_noise.size() / n_before) * 100
            if removal_pct > 50:
                print(f"  ⚠️  Noise filter would remove {removal_pct:.1f}% — too aggressive, SKIPPING")
                print(f"      (Increase noise_radius or noise_sigma in config.yaml)")
            else:
                current = _clone_from_ref(current, ref_noise, "Noise")
                n_removed = n_before - current.size()
                pct = (n_removed / n_before) * 100
                print(f"  ✅ {n_before:,} → {current.size():,} ({n_removed:,} noisy pts, {pct:.1f}%)")
        else:
            print(f"  ⚠️  Noise filter returned None")
        
        current.setName("denoised")
        print(f"  ({time.time()-t5:.1f}s)\n")
    else:
        print(f"[Step 5/6] Noise filter: SKIPPED\n")

    # ── Secondary subsample if exceeding max_points ──
    if args.max_points > 0 and current.size() > args.max_points:
        t_max = time.time()
        n_before = current.size()
        target_ratio = args.max_points / n_before
        larger_voxel = args.voxel_size / (target_ratio ** (1/3))
        print(f"  🔧 Capping to {args.max_points:,} pts (voxel={larger_voxel*1000:.1f}mm)...")
        
        ref2 = cc.CloudSamplingTools.resampleCloudSpatially(current, larger_voxel)
        if ref2 is not None:
            current = _clone_from_ref(current, ref2, "MaxPts")
            print(f"  ✅ {n_before:,} → {current.size():,}")
        print(f"  ({time.time()-t_max:.1f}s)\n")

    # ══════════════════════════════════════════════════════════════
    # STEP 6: NORMAL ESTIMATION
    # ══════════════════════════════════════════════════════════════
    if not args.skip_normals:
        t6 = time.time()
        print(f"[Step 6/6] Computing normals...")
        
        cc.computeNormals([current])
        
        if current.hasNormals():
            print(f"  Orienting normals (MST)...")
            current.orientNormalsWithMST()
            print(f"  ✅ Normals computed and oriented ({current.size():,} points)")
        else:
            print(f"  ⚠️  Normal computation failed")
        
        current.setName("with_normals")
        print(f"  ({time.time()-t6:.1f}s)\n")
    else:
        print(f"[Step 6/6] Normal estimation: SKIPPED\n")

    # ══════════════════════════════════════════════════════════════
    # SAVE (Binary PLY — with origin fields preserved if available)
    # ══════════════════════════════════════════════════════════════
    t_save = time.time()
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    import numpy as np
    
    print(f"[Save] Writing binary PLY...")
    
    # Extract data from CloudCompPy cloud
    xyz = current.toNpArray()  # [N, 3] float32
    n_pts = xyz.shape[0]
    
    # Get colors if available
    has_colors = current.hasColors()
    if has_colors:
        rgb = current.colorsToNpArray()  # [N, 3] or [N, 4] uint8
        rgb = rgb[:, :3]  # keep only RGB, drop alpha if present
    
    # Check for origin scalar fields (preserved through partialClone operations)
    origin_fields = {}
    sf_dic = current.getScalarFieldDic()  # {name: index}
    for field_name in ['frame_global', 'pixel_row', 'pixel_col']:
        if field_name in sf_dic:
            sf_idx = sf_dic[field_name]
            sf = current.getScalarField(sf_idx)
            origin_fields[field_name] = sf.toNpArray()
    
    has_origins = len(origin_fields) == 3
    if has_origins:
        print(f"  ✅ Origin fields preserved ({n_pts} points with frame_global, pixel_row, pixel_col)")
    
    # Check for confidence scalar field
    has_confidence = 'confidence' in sf_dic
    conf_array = None
    if has_confidence:
        sf_idx = sf_dic['confidence']
        sf = current.getScalarField(sf_idx)
        conf_array = sf.toNpArray()
        print(f"  ✅ Confidence field preserved ({n_pts} points)")
    
    # Write binary PLY
    with open(output_path, 'wb') as f:
        # Header
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {n_pts}\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        if has_colors:
            header += "property uchar red\n"
            header += "property uchar green\n"
            header += "property uchar blue\n"
        if has_confidence:
            header += "property float confidence\n"
        if has_origins:
            header += "property int frame_global\n"
            header += "property short pixel_row\n"
            header += "property short pixel_col\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))
        
        # Build structured dtype and pack
        fields = [('x','<f4'),('y','<f4'),('z','<f4')]
        if has_colors:
            fields += [('r','u1'),('g','u1'),('b','u1')]
        if has_confidence:
            fields += [('confidence','<f4')]
        if has_origins:
            fields += [('frame_global','<i4'),('pixel_row','<i2'),('pixel_col','<i2')]
        
        dtype = np.dtype(fields)
        packed = np.empty(n_pts, dtype=dtype)
        packed['x'] = xyz[:, 0]
        packed['y'] = xyz[:, 1]
        packed['z'] = xyz[:, 2]
        if has_colors:
            packed['r'] = rgb[:, 0]
            packed['g'] = rgb[:, 1]
            packed['b'] = rgb[:, 2]
        if has_confidence:
            packed['confidence'] = conf_array.astype(np.float32)
        if has_origins:
            packed['frame_global'] = origin_fields['frame_global'].astype(np.int32)
            packed['pixel_row'] = origin_fields['pixel_row'].astype(np.int16)
            packed['pixel_col'] = origin_fields['pixel_col'].astype(np.int16)
        
        packed.tofile(f)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    origin_tag = " +origins" if has_origins else ""
    print(f"  ✅ {output_path}")
    print(f"     {n_pts:,} points | {file_size:.1f} MB | Binary PLY{origin_tag}")
    
    print(f"  ({time.time()-t_save:.1f}s)\n")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    t_total = time.time() - t_pipeline
    print(f"{'='*65}")
    print(f"  ✅ PIPELINE COMPLETE — {t_total:.1f}s total")
    print(f"{'='*65}")
    print(f"  Input:     {total_input:,} points ({len(chunk_files)} chunks)")
    print(f"  Output:    {current.size():,} points")
    print(f"  Reduction: {(1 - current.size()/total_input)*100:.1f}%")
    print(f"  Normals:   {'YES' if current.hasNormals() else 'NO'}")
    print(f"  File:      {os.path.abspath(output_path)}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
