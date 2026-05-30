"""
StrayDA3Streaming: DA3-streaming subclass with Stray Scanner data injection.

Inherits DA3_Streaming without modifying anything. After DA3 inference,
injects ARKit poses, LiDAR intrinsics, and hybrid depth (LiDAR + DA3 calibrated).

If no Stray Scanner data is provided, falls back to pure DA3-streaming.

Used by:
  - backend: "hybrid" — DA3 inference + LiDAR injection
  - backend: "lidar"  — DA3 SLAM only (skip neural inference), use LiDAR depth

Authors: Hernán Barreto — Ingerop IN3
"""
import os
import sys
import glob
import json
import numpy as np
import cv2
from pathlib import Path

# ── Path setup ──
server_dir = Path(__file__).resolve().parent
da3_streaming_dir = server_dir.parent / "vendor" / "depth-anything-3" / "da3_streaming"

if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))
if str(da3_streaming_dir) not in sys.path:
    sys.path.insert(0, str(da3_streaming_dir))

from da3_streaming import DA3_Streaming


def _robust_linear_fit(da3_vals, lidar_vals, inlier_rounds=2, inlier_sigma=3.0):
    """Fit depth_corrected = da3 * scale + offset with outlier rejection."""
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
    return scale, offset


class StrayDA3Streaming(DA3_Streaming):
    """DA3-streaming with automatic Stray Scanner data injection.

    If stray_data is set, replaces:
      - predictions.extrinsics → ARKit poses (c2w → w2c)
      - predictions.intrinsics → LiDAR K (rescaled to DA3 output resolution)
      - predictions.depth → LiDAR (<trust_range) + DA3 calibrated (>trust_range)
      - predictions.conf → boosted in LiDAR zone

    If stray_data is None, behaves exactly like DA3_Streaming.
    """

    # ── Default configuration (overridable via config.yaml reconstruction.lidar) ──
    LIDAR_TRUST_M = 5.0       # Use LiDAR depth up to this range (iPad full range)
    DA3_MAX_RANGE_M = 20.0    # Clamp DA3 extrapolation
    CALIB_MIN_DEPTH = 0.3     # Min depth for calibration zone
    INLIER_ROUNDS = 2         # Outlier rejection iterations
    INLIER_SIGMA = 3.0        # MAD sigma threshold

    def __init__(self, image_dir, save_dir, config, stray_data=None, lidar_cfg=None):
        """
        Args:
            image_dir: Directory with RGB frames
            save_dir: Output directory
            config: DA3-streaming config dict
            stray_data: Dict from prepare_stray_data() or None for pure DA3
            lidar_cfg: Dict from config.yaml reconstruction.lidar (optional overrides)
        """
        super().__init__(image_dir, save_dir, config)
        self.stray_data = stray_data

        # Per-frame export directory for the fused (LiDAR + calibrated-DA3)
        # depth. Layout mirrors what ``extract_da3_full.py`` produces so
        # ``tsdf_export._resolve_da3_depth`` discovers them automatically.
        # ``save_dir`` is usually ``<output>/da3_run/`` so its parent is the
        # session output dir.
        self.da3_full_dir = Path(save_dir).parent / "da3_full"
        self.da3_full_dir.mkdir(parents=True, exist_ok=True)

        # Apply config overrides
        if lidar_cfg:
            self.LIDAR_TRUST_M = lidar_cfg.get("trust_range", self.LIDAR_TRUST_M)

        if stray_data is not None:
            self._stray_index = {}  # filename → index in stray_data
            for i, fp in enumerate(stray_data['frame_paths']):
                self._stray_index[Path(fp).name] = i
            print(f"[StrayDA3] Injecting Stray Scanner data for "
                  f"{len(self._stray_index)} frames")
            print(f"[StrayDA3] LiDAR trust: 0–{self.LIDAR_TRUST_M}m, "
                  f"DA3 extends to {self.DA3_MAX_RANGE_M}m")
        else:
            self._stray_index = {}
            print("[StrayDA3] No Stray Scanner data — running pure DA3-streaming")

    def run(self, selected_frames_path=None):
        """Override base DA3_Streaming.run() to support keyframe filtering.

        The vendored DA3_Streaming.run() globs the whole image_dir and has no
        keyframe filter. Rather than patch the (gitignored) vendor, we reproduce
        its tiny body here and additionally restrict self.img_list to the
        keyframes listed in selected_frames.json ("selected_files" = basenames,
        produced by server/frames/selector.py). Used by both pure-DA3
        (run_da3_main.py, stray_data=None) and hybrid/lidar paths.
        """
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")

        if selected_frames_path and os.path.exists(selected_frames_path):
            with open(selected_frames_path) as f:
                sf_data = json.load(f)
            keyframe_names = set(sf_data.get("selected_files", []))
            if keyframe_names:
                filtered = [p for p in self.img_list
                            if os.path.basename(p) in keyframe_names]
                if filtered:
                    print(f"Keyframe filter: {len(filtered)}/{len(self.img_list)} "
                          f"images kept (from {selected_frames_path})")
                    self.img_list = filtered
                else:
                    print(f"[WARN] selected_frames.json matched 0 images — "
                          f"using all {len(self.img_list)} images")

        print(f"Found {len(self.img_list)} images")
        self.process_long_sequence()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None,
                             is_loop=False):
        """Override: run DA3 inference, then inject Stray Scanner data."""
        # 1. Let DA3 do its full inference
        predictions = super().process_single_chunk(
            range_1, chunk_idx=chunk_idx, range_2=range_2, is_loop=is_loop
        )

        # 2. If no Stray data, return as-is (pure DA3)
        if self.stray_data is None:
            return predictions

        # 3. Inject Stray Scanner data
        start_idx, end_idx = range_1
        chunk_images = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            s2, e2 = range_2
            chunk_images += self.img_list[s2:e2]

        N = len(chunk_images)
        da3_H, da3_W = predictions.depth.shape[-2], predictions.depth.shape[-1]
        lidar_K = self.stray_data['intrinsics']  # (3,3) at 256×192
        lidar_H, lidar_W = 192, 256  # Stray Scanner depth resolution

        # Scale factors to go from LiDAR resolution to DA3 output resolution
        sx = da3_W / lidar_W
        sy = da3_H / lidar_H

        injected_count = 0
        scale, offset = 1.0, 0.0

        for local_idx in range(N):
            fname = Path(chunk_images[local_idx]).name
            if fname not in self._stray_index:
                continue

            stray_idx = self._stray_index[fname]
            injected_count += 1

            # ── A. Inject extrinsics (ARKit c2w → w2c 3×4) ──
            c2w = self.stray_data['poses'][stray_idx]  # (4,4)
            w2c = np.linalg.inv(c2w)[:3, :]  # (3,4)
            predictions.extrinsics[local_idx] = w2c.astype(np.float32)

            # ── B. Inject intrinsics (rescaled to DA3 output resolution) ──
            K_scaled = lidar_K.copy().astype(np.float32)
            K_scaled[0, 0] *= sx  # fx
            K_scaled[1, 1] *= sy  # fy
            K_scaled[0, 2] *= sx  # cx
            K_scaled[1, 2] *= sy  # cy
            predictions.intrinsics[local_idx] = K_scaled

            # ── C. Fuse depth: LiDAR (<trust_range) + DA3 calibrated ──
            depth_lidar = self.stray_data['depths'][stray_idx]  # (192,256)
            # Resize LiDAR to DA3 output resolution
            depth_lidar_resized = cv2.resize(
                depth_lidar, (da3_W, da3_H),
                interpolation=cv2.INTER_NEAREST  # No interpolation for depth
            )

            depth_da3 = predictions.depth[local_idx]  # (H, W) DA3's inferred depth

            # Calibrate DA3 against LiDAR in overlap zone
            valid_lidar = ((depth_lidar_resized > self.CALIB_MIN_DEPTH) &
                          (depth_lidar_resized < self.LIDAR_TRUST_M))

            if valid_lidar.sum() > 50 and depth_da3.max() > 0.01:
                scale, offset = _robust_linear_fit(
                    depth_da3[valid_lidar], depth_lidar_resized[valid_lidar],
                    self.INLIER_ROUNDS, self.INLIER_SIGMA)
                da3_calibrated = depth_da3 * scale + offset
            else:
                da3_calibrated = depth_da3.copy()
                scale, offset = 1.0, 0.0

            da3_calibrated = np.clip(da3_calibrated, 0, self.DA3_MAX_RANGE_M)

            # Fusion: LiDAR wins where valid, DA3 fills the rest
            depth_fused = da3_calibrated.copy()
            depth_fused[valid_lidar] = depth_lidar_resized[valid_lidar]

            predictions.depth[local_idx] = depth_fused.astype(np.float32)

            # ── D. Boost confidence in LiDAR zone ──
            conf_max = predictions.conf[local_idx].max()
            if conf_max > 0:
                predictions.conf[local_idx][valid_lidar] = conf_max

            # ── E. Persist per-frame fused depth for downstream pipelines
            #       (TSDF, reconstruction_v2, etc). One .npy per frame keyed
            #       by the original Stray frame index — same naming as
            #       ``extract_da3_full.py``. Camera intrinsics are NOT saved:
            #       downstream readers compute K at depth resolution from the
            #       full-res Stray K + the .npy shape, so we don't need a
            #       separate file.
            stem = Path(chunk_images[local_idx]).stem
            np.save(str(self.da3_full_dir / f"{stem}_depth.npy"),
                    depth_fused.astype(np.float32))

        # 4. Re-save the modified chunk to disk
        if injected_count > 0:
            if is_loop:
                save_dir = self.result_loop_dir
                filename = (f"loop_{range_1[0]}_{range_1[1]}_"
                           f"{range_2[0]}_{range_2[1]}.npy")
            else:
                save_dir = self.result_unaligned_dir
                filename = f"chunk_{chunk_idx}.npy"

            save_path = os.path.join(save_dir, filename)
            np.save(save_path, predictions)

            # Fix self.all_camera_poses with injected extrinsics
            if not is_loop and range_2 is None and self.all_camera_poses:
                self.all_camera_poses[-1] = (
                    self.all_camera_poses[-1][0],
                    predictions.extrinsics
                )
                self.all_camera_intrinsics[-1] = (
                    self.all_camera_intrinsics[-1][0],
                    predictions.intrinsics
                )

            print(f"[StrayDA3] Injected {injected_count}/{N} frames | "
                  f"last calib: S={scale:.4f} off={offset:.4f}")

        return predictions


class StrayLiDAROnly(StrayDA3Streaming):
    """DA3-streaming SLAM with LiDAR-only depth (no neural inference).

    Overrides process_single_chunk to create Prediction objects directly
    from LiDAR depth + ARKit poses, bypassing the DA3 neural network.
    Still benefits from DA3-streaming's SIM3 alignment and loop closure.
    """

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None,
                             is_loop=False):
        """Create predictions from LiDAR data — skips neural inference entirely."""
        from depth_anything_3.specs import Prediction

        if self.stray_data is None:
            # No LiDAR data — fall back to DA3 inference
            return super().process_single_chunk(
                range_1, chunk_idx=chunk_idx, range_2=range_2, is_loop=is_loop
            )

        start_idx, end_idx = range_1
        chunk_images = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            s2, e2 = range_2
            chunk_images += self.img_list[s2:e2]

        N = len(chunk_images)
        lidar_K = self.stray_data['intrinsics']  # (3,3) at 256×192

        # DA3-streaming works at a specific resolution — determine it from first image
        sample_img = cv2.imread(chunk_images[0])
        # DA3 internally resizes to 378×504 (or similar), but we need to match
        # For LiDAR-only mode, we use LiDAR's native 256×192 upsampled
        # to DA3 output resolution to maintain compatibility with alignment code
        da3_H, da3_W = 378, 504  # DA3's default output resolution
        lidar_H, lidar_W = 192, 256

        sx = da3_W / lidar_W
        sy = da3_H / lidar_H

        # Build arrays for Prediction
        depths = np.zeros((N, da3_H, da3_W), dtype=np.float32)
        confs = np.ones((N, da3_H, da3_W), dtype=np.float32)  # Max confidence for LiDAR
        extrinsics = np.zeros((N, 3, 4), dtype=np.float32)
        intrinsics = np.zeros((N, 3, 3), dtype=np.float32)
        images = np.zeros((N, da3_H, da3_W, 3), dtype=np.uint8)

        for local_idx in range(N):
            fname = Path(chunk_images[local_idx]).name

            # Load and resize image to DA3 resolution
            img = cv2.imread(chunk_images[local_idx])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[local_idx] = cv2.resize(img_rgb, (da3_W, da3_H))

            if fname in self._stray_index:
                stray_idx = self._stray_index[fname]

                # Depth from LiDAR
                depth_lidar = self.stray_data['depths'][stray_idx]
                depths[local_idx] = cv2.resize(
                    depth_lidar, (da3_W, da3_H),
                    interpolation=cv2.INTER_NEAREST
                )

                # Confidence: max where LiDAR is valid, 0 elsewhere
                valid = depths[local_idx] > 0
                confs[local_idx][valid] = 1.0
                confs[local_idx][~valid] = 0.0

                # Extrinsics from ARKit (C2W → W2C 3×4)
                c2w = self.stray_data['poses'][stray_idx]
                extrinsics[local_idx] = np.linalg.inv(c2w)[:3, :].astype(np.float32)

            # Intrinsics scaled to DA3 resolution
            K_scaled = lidar_K.copy().astype(np.float32)
            K_scaled[0, 0] *= sx
            K_scaled[1, 1] *= sy
            K_scaled[0, 2] *= sx
            K_scaled[1, 2] *= sy
            intrinsics[local_idx] = K_scaled

        predictions = Prediction(
            depth=depths,
            is_metric=1,
            conf=confs,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            processed_images=images,
        )

        print(f"[StrayLiDAR] Created LiDAR-only predictions: "
              f"{N} frames, {da3_H}×{da3_W}")

        # Save to disk (same as base class)
        if is_loop:
            save_dir = self.result_loop_dir
            filename = (f"loop_{range_1[0]}_{range_1[1]}_"
                       f"{range_2[0]}_{range_2[1]}.npy")
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        np.save(save_path, predictions)

        return predictions
