"""Per-point provenance, v1 and v2 (claude_stac.txt §4-F7) — the ONE loader.

v1 (every cloud before F7): ``frame_global``, ``pixel_row``, ``pixel_col`` as
vertex properties of ``cleaned_cloud.ply`` (frame number, native pixel).

v2 (the F7 fused cloud) is a superset with the SAME v1 names — every v1 consumer
(session_io, correction/, witness/, certify/, mask_filter) keeps reading the PLY
as before — plus, row-aligned in ``output/origins.npz``:

- ``pixel_u_und``, ``pixel_v_und``: the EXACT undistorted native pixel where the
  depth was measured (``pixel_row/col`` are that pixel carried back through the
  lens to the original frame and rounded, which is what the masks index);
- ``n_consistent`` (views that confirm the depth), ``n_contradict`` (views that see free
  space through it — the vote's count, audit 2026-10-23; 0 in files written before it),
  ``ncc`` (the ZNCC reached), ``source`` (tier: 0 plane sweep, 1 prior fill), ``residual_rel``;
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
                         "residual_rel", "content_flags", "geometry_epoch", "camera_epoch",
                         "n_contradict")
V2_DTYPES = {"frame_global": np.int32, "pixel_row": np.int32, "pixel_col": np.int32,
             "pixel_u_und": np.int32, "pixel_v_und": np.int32, "n_consistent": np.uint8,
             "ncc": np.float32, "source": np.uint8, "residual_rel": np.float32,
             "content_flags": np.uint8, "geometry_epoch": np.int16, "camera_epoch": np.int16,
             "n_contradict": np.uint8}
# fields added to v2 after files of it were written: a file without one reads it as zeros (the
# value every writer before the field existed would have put there)
V2_OPTIONAL = {"n_contradict": 0}
# ``source`` of a fused point: depth_sweep's SOURCE_SWEEP (0, tier 0: photometric
# measurement) and SOURCE_PRIOR_FILL (1, tier 1: Omega's prior confirmed by other
# views), plus TIER 2 — COLMAP PatchMatch's geometric depth (USER 2026-09-29: a second
# measuring instrument, switchable: precision.depth.colmap.as_tier), judged by the
# same consistency rule as tier 0 and ranked between the two
SOURCE_COLMAP = 20
SOURCE_NAMES_FUSED = {0: "tier0", 1: "tier1_prior_fill", SOURCE_COLMAP: "tier2_colmap"}

REJECT_REASONS = {"insufficient_witnesses": 1, "inconsistent": 2, "prior_fill_dropped": 3,
                  "excluded_mask": 4, "dedup": 5, "unlocalized_frame": 6, "sor": 7,
                  "prior_low_conf": 8,      # tier-1 prior under the session's confidence floor
                  "contradicted": 9,        # a view sees free space through the point
                  "fused": 10,              # consumed by a winner within the consistency tolerance
                  "prior_not_independent": 11,  # RETIRED 2026-09-29 (F6's independence rule): never
                                                # written any more; kept so older files still read
                  "voxel": 12}              # the cleaning recipe's voxel pick (postprocessing.voxel_size)
REJECT_NAMES = {v: k for k, v in REJECT_REASONS.items()}
# DECLARED (point 75, USER 2026-10-08, I2 eliminated): the origins v2 column ``content_flags``
# is ALWAYS 0 since 2026-10-08 — its only writer, the I2 content pass (content_tags.json), no
# longer exists and nothing under intake/ is read by any precision stage. The column stays in
# the published format so every reader of older and newer origins files reads the same layout;
# retiring it is a FORMAT change (the readers go with it), for a later wave, not tonight.
CONTENT_FLAG_BITS = {"reflective": 1, "low_info": 2}


class ProvenanceError(RuntimeError):
    """Provenance missing or inconsistent — with the exact reason."""


def content_flags_of(tags: Optional[Dict[str, Any]]) -> int:
    """The keyframe's weight flags from its I2 tags (vlm_proposed)."""
    if not tags:
        return 0
    return int(sum(bit for name, bit in CONTENT_FLAG_BITS.items() if bool(tags.get(name))))


def write_origins(path: Path, cols: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Path:
    cols = dict(cols)
    n_rows = {len(v) for k, v in cols.items() if k in V2_FIELDS}
    for k, fill in V2_OPTIONAL.items():
        if k not in cols and len(n_rows) == 1:
            cols[k] = np.full(next(iter(n_rows)), fill, V2_DTYPES[k])
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
                    n = len(data)
                    return {"version": ORIGINS_VERSION,
                            **{k: (z[k] if k in z.files else np.full(n, V2_OPTIONAL[k], V2_DTYPES[k]))
                               for k in V2_FIELDS}}
    return {"version": 1, **{k: np.asarray(data[k]) for k in V1_FIELDS}}
