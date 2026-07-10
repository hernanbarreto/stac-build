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
        # NO FALLBACK: partialClone returns None only on real failure (a valid ref always
        # clones) → discarding the filter result and keeping the unfiltered cloud is a
        # silent degradation. Fail.
        print(f"[CloudCompPy] ❌ {step_name}: partialClone returned None")
        sys.exit(1)
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

    # Confidence gate (min-max normalised, same as the viewer slider)
    parser.add_argument("--conf-min-norm", type=float, default=0.0,
                        help="Keep points whose min-max-normalised 'confidence' >= this "
                             "[0,1] (0=off). 0.1 ≈ the UI slider at 0.1. Drops the noisy "
                             "low/mid-confidence tail so cleaned_cloud is denoised at source.")

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
            # NO FALLBACK: a chunk that won't load = a whole spatial region silently
            # missing from the cloud. Fail.
            print(f"[CloudCompPy] ❌ Failed to load {os.path.basename(ply_path)}")
            sys.exit(1)
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

    # ── Step 1b: Inject per-point scalar fields from .npz ──
    # frame_global / pixel_row / pixel_col + confidence, so every point stays
    # traceable to its source keyframe + pixel + confidence. These ride along
    # through dedup/voxel/SOR (partialClone preserves SFs) into the saved PLY.
    origin_files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*_origins.npz")))
    if origin_files:
        import numpy as np
        t_orig = time.time()

        cols = {"frame_global": [], "pixel_row": [], "pixel_col": [], "confidence": []}
        have_conf = True
        for of in origin_files:
            d = np.load(of)
            cols["frame_global"].append(d["frame_global"].astype(np.float32))
            cols["pixel_row"].append(d["pixel_row"].astype(np.float32))
            cols["pixel_col"].append(d["pixel_col"].astype(np.float32))
            if "confidence" in d:
                cols["confidence"].append(d["confidence"].astype(np.float32))
            else:
                have_conf = False

        fields = [("frame_global", np.concatenate(cols["frame_global"])),
                  ("pixel_row", np.concatenate(cols["pixel_row"])),
                  ("pixel_col", np.concatenate(cols["pixel_col"]))]
        if have_conf and cols["confidence"]:
            fields.append(("confidence", np.concatenate(cols["confidence"])))

        n_cloud = current.size()
        existing = current.getScalarFieldDic()  # avoid duplicating SFs already on the cloud
        if len(fields[0][1]) == n_cloud:
            injected = []
            for name, arr in fields:
                if name in existing:
                    continue  # e.g. a LiDAR chunk PLY already carried 'confidence'
                idx = current.addScalarField(name)
                sf = current.getScalarField(idx)
                sf.fromNpArrayCopy(arr)
                injected.append(name)
            print(f"  ✅ Injected scalar fields {injected} ({n_cloud:,} pts) ({time.time()-t_orig:.1f}s)\n")
        else:
            # NO FALLBACK: without per-point frame_global/pixel the cleaned cloud is NOT
            # traceable → the TSDF (mask_to_cleaned_cloud) cannot mask it. A size mismatch
            # means a chunk PLY/origins desync — a real bug. Fail.
            print(f"[CloudCompPy] ❌ Origin size mismatch: {len(fields[0][1])} vs cloud "
                  f"{n_cloud} — cannot inject traceability")
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════
    # STEP 1c: CONFIDENCE GATE (min-max normalised == the viewer slider)
    # Keep points where (conf-min)/(max-min) >= conf_min_norm. Run BEFORE
    # voxel/SOR so they process fewer points AND the saved cleaned_cloud is
    # already denoised at source — the downstream TSDF then needs no extra
    # confidence filtering.
    # ══════════════════════════════════════════════════════════════
    if args.conf_min_norm and args.conf_min_norm > 0:
        sfdic = current.getScalarFieldDic()
        ci = sfdic.get("confidence", -1)
        if ci is not None and ci >= 0:
            t1c = time.time()
            sf = current.getScalarField(ci)
            sf.computeMinAndMax()
            cmin, cmax = float(sf.getMin()), float(sf.getMax())
            thr = cmin + float(args.conf_min_norm) * max(cmax - cmin, 1e-6)
            n_before = current.size()
            current.setCurrentScalarField(ci)
            current.setCurrentOutScalarField(ci)
            filtered = cc.filterBySFValue(thr, cmax + 1.0, current)
            if filtered is not None and filtered.size() > 0:
                pct = 100.0 * filtered.size() / max(n_before, 1)
                print(f"[Step 1c] Confidence gate: norm>={args.conf_min_norm} → raw>={thr:.1f} "
                      f"(range {cmin:.1f}..{cmax:.1f})  {n_before:,} → {filtered.size():,} "
                      f"({pct:.1f}% kept)  ({time.time()-t1c:.1f}s)\n")
                current = filtered
            else:
                # NO FALLBACK: the gate was explicitly requested (conf_min_norm>0) → None
                # (failure) or 0 points (threshold drops everything) must not silently pass.
                print(f"[CloudCompPy] ❌ Confidence gate produced empty/None at norm="
                      f"{args.conf_min_norm} (threshold too aggressive or filter failed)")
                sys.exit(1)
        else:
            # NO FALLBACK: gate requested but there is no 'confidence' field to gate on.
            print("[CloudCompPy] ❌ Confidence gate requested but no 'confidence' scalar field")
            sys.exit(1)

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
            print(f"[CloudCompPy] ❌ Near-duplicate micro-voxel returned None")
            sys.exit(1)
        
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
        print(f"[CloudCompPy] ❌ Voxel subsampling returned None")
        sys.exit(1)
    
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
            print(f"[CloudCompPy] ❌ SOR filter returned None")
            sys.exit(1)
        
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
            print(f"[CloudCompPy] ❌ Noise filter returned None")
            sys.exit(1)
        
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
    # Step 6 (normals) DISABLED: the PLY writer below does not emit normal properties and no
    # downstream consumer reads cloud normals — computing + MST-orienting them on millions of
    # points was wasted work. Re-enable AND add nx/ny/nz to the writer if a consumer needs them.
    print(f"[Step 6/6] Normal estimation: DISABLED (not written to PLY / not used)\n")

    # ══════════════════════════════════════════════════════════════
    # NO NaN TOLERATED — final gate before the cloud becomes the pipeline's truth.
    # Every downstream consumer (Potree, TSDF, segmentation, Q&A) trusts this file;
    # a single non-finite coordinate or confidence here means some step above
    # corrupted data, and shipping it would just move the explosion downstream.
    import numpy as np
    _xyz_chk = current.toNpArray()
    if not np.isfinite(_xyz_chk).all():
        print(f"[CloudCompPy] ❌ {int((~np.isfinite(_xyz_chk)).any(1).sum()):,} points "
              f"with non-finite XYZ — refusing to save a corrupted cloud")
        sys.exit(1)
    del _xyz_chk
    _sfd = current.getScalarFieldDic()
    for _sfname in ("confidence", "frame_global", "pixel_row", "pixel_col"):
        _si = _sfd.get(_sfname, -1)
        if _si is not None and _si >= 0:
            _vals = current.getScalarField(_si).toNpArray()
            _nbad = int((~np.isfinite(_vals)).sum())
            if _nbad:
                print(f"[CloudCompPy] ❌ scalar field '{_sfname}' has {_nbad:,} "
                      f"non-finite values — refusing to save a corrupted cloud")
                sys.exit(1)

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
