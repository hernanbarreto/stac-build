# VIO trajectory ingestion format

STAC-BUILD can use a **visual-inertial odometry (VIO) trajectory** recorded
alongside the session video as its **metric scale source**. Any modern phone —
with or without LiDAR — produces metric poses via ARCore (Android) or ARKit
(iOS); most capture apps can export them. When a VIO file is present in the
session, **VIO sets the scale** (DA3 monocular depth becomes a cross-check and
the VIO↔DA3 agreement is reported per session).

## Why per-segment, not end-to-end

VIO position drifts over minutes, but its **short-horizon scale is excellent**
(it is anchored by the IMU's metric accelerometer). The pipeline therefore
computes the scale as the **median of trajectory-length ratios over ~5 s
segments** (`vio_segment_s` in `config.yaml`), never as a single global length
quotient that drift would corrupt. Loops, pauses and shaky sections are handled
robustly: static segments are skipped, outlier segments are absorbed by the
median.

## File placement (auto-detected)

Put ONE of these in the session directory (`src_*/`, same level as
`source_video.*`):

```
vio_trajectory.csv
vio_trajectory.json
vio/trajectory.csv
vio/trajectory.json
```

Detection is automatic (like Stray sessions). **A present but unusable file
aborts the reconstruction with the exact reason** — fix it or remove it; there
is no silent fallback to DA3.

## Timestamps

Timestamps are **seconds from the start of the video**. Frame `N` of the video
is matched to time `N / fps` (fps is read from `source_video.*`). Start the VIO
recording and the video together (same app, same tap, ideally the same
capture session). Sub-second sync error is tolerated by the segment averaging;
multi-second offsets are not — the segment ratios will disperse and the
run aborts on the `vio_min_segments` / `vio_min_coverage` gates.

## CSV format

Header optional. Columns, in order (or named in the header):

```
timestamp, x, y, z [, qx, qy, qz, qw]
```

- `timestamp` — seconds (float) from video start.
- `x y z` — position in **meters**, any fixed world frame (ARCore/ARKit world
  frame is fine; only trajectory *lengths* are used, so the frame and origin do
  not matter).
- `qx qy qz qw` — optional orientation quaternion; accepted and currently
  ignored (scale needs positions only).

Example:

```csv
timestamp,x,y,z,qx,qy,qz,qw
0.000,0.0000,0.0000,0.0000,0.0,0.0,0.0,1.0
0.033,0.0021,-0.0003,0.0110,0.0,0.0,0.0,1.0
0.066,0.0044,-0.0005,0.0221,0.0,0.0,0.0,1.0
```

Any sample rate ≥ ~5 Hz works (ARCore/ARKit native 30–60 Hz is ideal).

## JSON format

Either a plain list of samples, or an object with metadata:

```json
{
  "video_fps": 30.0,
  "samples": [
    {"t": 0.000, "p": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]},
    {"t": 0.033, "p": [0.0021, -0.0003, 0.0110]}
  ]
}
```

- `t` — seconds from video start; `p` — position [x, y, z] in meters;
  `q` — optional quaternion [qx, qy, qz, qw], ignored.
- `video_fps` — optional hint; the pipeline still reads fps from the video
  when present.

## Quality gates (config.yaml → reconstruction.vggtomega)

| Key | Default | Meaning |
|---|---|---|
| `scale_vio` | `true` | enable VIO auto-detection |
| `vio_segment_s` | `5.0` | segment length (s) for the ratio median |
| `vio_min_segments` | `8` | fewer usable segments → abort |
| `vio_min_coverage` | `0.5` | segments must span ≥ 50% of the keyframe time range → abort |

The per-session result (s_vio, number of segments, MAD, coverage, and the
VIO↔DA3 agreement in %) is persisted in `output/scale_diagnostics.json` and
echoed in the run log.
