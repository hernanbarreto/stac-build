"""Per-point provenance, v1 and v2 (claude_stac.txt §4-F7) — the ONE loader.

v1 (every cloud before F7): ``frame_global``, ``pixel_row``, ``pixel_col`` as
vertex properties of ``cleaned_cloud.ply`` (frame number, native pixel).

v2 (the F7 fused cloud) is a superset with the SAME v1 names — every v1 consumer
(session_io, correction/, witness/, certify/, mask_filter) keeps reading the PLY
as before — plus, row-aligned in ``output/origins.npz``:

- ``pixel_u_und``, ``pixel_v_und``: the EXACT undistorted native pixel where the
  depth was measured (``pixel_row/col`` are that pixel carried back through the
  lens to the original frame and rounded, which is what the masks index);
- ``n_consistent`` (views that confirm the depth), ``ncc`` (the ZNCC reached),
  ``source`` (tier: 0 plane sweep, 1 prior fill), ``residual_rel``;
- ``content_flags`` (bit 1 reflective, bit 2 low_info — I2's vlm_proposed tags
  of the keyframe; they weigh, never exclude);
- ``geometry_epoch``, ``camera_epoch`` of the measurement.

``rejected_points.npz``: the same columns for every candidate that did not enter
the cloud, with ``reason`` (u8, :data:`REJECT_REASONS`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ORIGINS_NAME = "origins.npz"
REJECTED_NAME = "rejected_points.npz"
ORIGINS_VERSION = 2
V1_FIELDS = ("frame_global", "pixel_row", "pixel_col")
V2_FIELDS = V1_FIELDS + ("pixel_u_und", "pixel_v_und", "n_consistent", "ncc", "source",
                         "residual_rel", "content_flags", "geometry_epoch", "camera_epoch")
V2_DTYPES = {"frame_global": np.int32, "pixel_row": np.int32, "pixel_col": np.int32,
             "pixel_u_und": np.int32, "pixel_v_und": np.int32, "n_consistent": np.uint8,
             "ncc": np.float32, "source": np.uint8, "residual_rel": np.float32,
             "content_flags": np.uint8, "geometry_epoch": np.int16, "camera_epoch": np.int16}
REJECT_REASONS = {"insufficient_witnesses": 1, "inconsistent": 2, "prior_fill_dropped": 3,
                  "excluded_mask": 4, "dedup": 5, "unlocalized_frame": 6, "sor": 7}
REJECT_NAMES = {v: k for k, v in REJECT_REASONS.items()}
CONTENT_FLAG_BITS = {"reflective": 1, "low_info": 2}


class ProvenanceError(RuntimeError):
    """Provenance missing or inconsistent — with the exact reason."""


def content_flags_of(tags: Optional[Dict[str, Any]]) -> int:
    """The keyframe's weight flags from its I2 tags (vlm_proposed)."""
    if not tags:
        return 0
    return int(sum(bit for name, bit in CONTENT_FLAG_BITS.items() if bool(tags.get(name))))


def write_origins(path: Path, cols: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Path:
    missing = [k for k in V2_FIELDS if k not in cols]
    if missing:
        raise ProvenanceError(f"origins v2 needs {missing}")
    n = {len(cols[k]) for k in V2_FIELDS}
    if len(n) != 1:
        raise ProvenanceError(f"origins v2 columns have different lengths {sorted(n)}")
    np.savez(path, version=np.int64(ORIGINS_VERSION), meta=np.array(json.dumps(meta)),
             **{k: np.asarray(cols[k]).astype(V2_DTYPES[k]) for k in V2_FIELDS})
    return Path(path)


def load_origins(output_dir: Path) -> Dict[str, Any]:
    """{'version': 1|2, <columns>} row-aligned with ``cleaned_cloud.ply``. v2 when
    ``origins.npz`` exists and describes the live cloud (same rows, same geometry
    epoch); v1 from the PLY's own fields otherwise."""
    from correction.session import read_ply
    output_dir = Path(output_dir)
    ply = output_dir / "cleaned_cloud.ply"
    if not ply.exists():
        raise ProvenanceError(f"{ply} does not exist")
    _h, data = read_ply(ply)
    names = data.dtype.names or ()
    missing = [k for k in V1_FIELDS if k not in names]
    if missing:
        raise ProvenanceError(f"{ply} lacks the provenance field(s) {missing}")
    org = output_dir / ORIGINS_NAME
    if org.exists():
        from precision.camera import read_geometry_epoch
        with np.load(org) as z:
            if int(z["version"]) == ORIGINS_VERSION and len(z["frame_global"]) == len(data):
                ep = int(z["geometry_epoch"][0]) if len(data) else read_geometry_epoch(output_dir)
                if ep == read_geometry_epoch(output_dir) and \
                        np.array_equal(z["frame_global"], np.asarray(data["frame_global"], np.int32)):
                    return {"version": ORIGINS_VERSION, **{k: z[k] for k in V2_FIELDS}}
    return {"version": 1, **{k: np.asarray(data[k]) for k in V1_FIELDS}}
