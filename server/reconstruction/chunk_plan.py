# STAC-Builder — chunked-metric Omega: the chunk layout helpers every stage shares.
#
# THE LAYOUT IS AN EXPLICIT LIST (USER 2026-10-06): the co-visibility planner
# (reconstruction/chunk_covis.py) decides the [start, end) keyframe ranges — variable
# lengths, each seam its own overlap — and the map worker hands them to the fork as
# Model.chunk_ranges (a single pass is the one range [0, n)). Every consumer reads
# THAT list (`omega_chunk_ranges`: what the fork ran, chunk_sim3.json; else what the
# run was configured with) — none rebuilds chunks from a size, an overlap or a fixed
# divisor. The card never shapes the plan: Omega's processing resolution adapts so
# that the largest planned chunk fits it (`omega_resolution_for`).
#
# `walk_length_m` stays because the walk is still worth REPORTING. `chunk_ranges` is
# the vendor's UNIFORM layout (step = size − overlap) — the layout of the legacy
# backends (DA3 streaming, MapAnything), never the Omega plan. The walk-based sizing
# (`plan_chunks`, chunk_walk_m / max_walk_single_pass_m / chunk_frames_over_walk /
# chunk_frames) was DELETED 2026-10-06 with the co-visibility plan.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CHUNK_PLAN_NAME = "chunk_plan.json"
OMEGA_CONFIG_NAME = "vggt_omega_config.yaml"
CHUNK_SIM3_NAME = "chunk_sim3.json"

# ── Omega's memory model on the card (the capacity formula, unchanged since 2026-10-04) ──
# 4.0 GB of base (weights + workspace) and 0.086 GB per frame MEASURED at pccr's grid 832x464
# (the vendor's own footprint: ~500 frames on a 48 GB card, the paper's number); a frame's
# footprint grows with its pixels (tokens), so at another grid it scales by the pixel count.
# The card's TOTAL memory enters (a property of the card, deterministic) — never the free
# memory of the moment, which made the geometry a function of transient GPU state.
OMEGA_BASE_GB = 4.0
OMEGA_GB_PER_FRAME_REF = 0.086
OMEGA_REF_GRID_WH = (832, 464)
# USER 2026-10-06 ("1 sí"): the capacity is computed on the card's total memory LESS the same
# margin the I3 DA3 windows leave (intake.parallax.vram_margin_frac, 0.15) — memory.total includes
# what CUDA reserves for itself and the 0.086 GB/frame of 832x464 is extrapolated linearly in
# pixels to grids never run (without it: zaragoza 183 keyframes at 1808, predicted 79.48 of 80.00
# GB). And an Omega pass that still runs out of memory is retried ONE patch step lower, declared
# in the log and in the report (workers/map_worker.py, the Omega pass).


class ChunkLayoutError(RuntimeError):
    """The chunk layout of a run cannot be established, or two records of it disagree."""


def walk_length_m(poses_txt: Path) -> float:
    """Trajectory length from camera_poses.txt (one flattened 4x4 c2w per line, or
    frame_idx + 16 values). Metric ONLY after scale_align has been applied."""
    rows = [line.split() for line in open(poses_txt) if line.strip()]
    arr = np.array([[float(x) for x in r] for r in rows])
    if arr.shape[1] == 17:
        arr = arr[:, 1:]
    if arr.shape[1] != 16:
        raise ValueError(f"unexpected camera_poses.txt layout: {arr.shape[1]} cols")
    centers = arr.reshape(-1, 4, 4)[:, :3, 3]
    if len(centers) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())


def chunk_ranges(n_keyframes: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """[(start, end)) keyframe-index ranges of the vendor's UNIFORM layout, EXACTLY as
    VGGT-Long slices it without Model.chunk_ranges (step = chunk_size - overlap; last chunk
    clipped to the end) — the layout of the legacy backends, never the Omega plan."""
    if n_keyframes <= chunk_size:
        return [(0, n_keyframes)]
    step = chunk_size - overlap
    ranges = []
    start = 0
    while start < n_keyframes:
        end = min(start + chunk_size, n_keyframes)
        ranges.append((start, end))
        if end >= n_keyframes:
            break
        start += step
    return ranges


def as_ranges(ranges: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """A layout as (start, end) int tuples (JSON lists, YAML tuples and numpy ints alike)."""
    return [(int(a), int(b)) for a, b in ranges]


def chunk_lengths(ranges: Sequence[Sequence[int]]) -> List[int]:
    return [int(b) - int(a) for a, b in ranges]


def seam_overlaps(ranges: Sequence[Sequence[int]]) -> List[int]:
    """Frames shared by chunks k and k+1 (end of k − start of k+1) — each seam its own."""
    r = as_ranges(ranges)
    return [r[k][1] - r[k + 1][0] for k in range(len(r) - 1)]


def latest_chunk_of(ranges: Sequence[Sequence[int]], pos: int) -> int:
    """The LAST chunk whose range holds keyframe position ``pos`` (the later copy of a seam
    frame): on a uniform layout this is min(pos // step, n_chunks − 1), the rule the depth
    loaders always used; on an explicit one the same "later chunk" rule over the real ranges.
    −1 when no chunk holds it."""
    best = -1
    for k, (a, b) in enumerate(as_ranges(ranges)):
        if a <= pos < b:
            best = k
    return best


def plan_anchor_indices(ranges: Sequence[Sequence[int]], per_chunk: int = 3) -> List[int]:
    """Keyframe indices (sorted, deduped) for the DA3 metric anchors: `per_chunk`
    spread inside EVERY planned chunk's range, so no chunk is left without a metric lock."""
    per_chunk = max(1, int(per_chunk))
    picks = set()
    for start, end in as_ranges(ranges):
        span = end - start
        if span <= 0:
            continue
        if per_chunk == 1:
            fracs = [0.5]
        else:
            fracs = [0.15 + 0.7 * i / (per_chunk - 1) for i in range(per_chunk)]
        for fr in fracs:
            picks.add(start + min(span - 1, int(round(fr * (span - 1)))))
    return sorted(picks)


# ── Omega's resolution: the largest planned chunk must fit the card ──

def omega_capacity(total_gb: float, grid_w: int, grid_h: int, margin_frac: float = 0.0) -> int:
    """Frames one Omega pass holds on a card of ``total_gb`` (less ``margin_frac`` of it) at a
    ``grid_w`` x ``grid_h`` grid."""
    per_frame = OMEGA_GB_PER_FRAME_REF * (float(grid_w) * float(grid_h)) / float(
        OMEGA_REF_GRID_WH[0] * OMEGA_REF_GRID_WH[1])
    return int((float(total_gb) * (1.0 - float(margin_frac)) - OMEGA_BASE_GB) / per_frame)


def omega_resolution_for(n_frames: int, total_gb: float, native_wh: Tuple[int, int], mode: str,
                         ceiling: int, margin_frac: float = 0.0) -> Dict[str, Any]:
    """USER 2026-10-06 (*"el plan de chunk no cambia, el tamaño de la tarjeta no lo podemos
    cambiar, para que encaje adaptamos la resolución"*): Omega's processing resolution for a run
    whose LARGEST chunk holds ``n_frames``. ``ceiling`` is the configured resolution (``native`` =
    the frames' long side rounded up to Omega's patch); it stands when n_frames fit the card at
    it, otherwise the LARGEST patch-aligned resolution below it at which they do (aspect kept —
    the grid is Omega's own preprocessing, ``precision.camera.omega_grid_for``). Deterministic per
    card: the card's TOTAL memory and the per-frame footprint ∝ pixels (``omega_capacity``).
    Returns the report (resolution, mode, grid, capacity, why); raises ChunkLayoutError when not
    even one patch fits."""
    from precision.camera import OMEGA_PATCH_SIZE, omega_grid_for
    n_frames, ceiling = int(n_frames), int(ceiling)
    nw, nh = int(native_wh[0]), int(native_wh[1])
    if n_frames < 1:
        raise ChunkLayoutError("no frame to place on the card")
    if total_gb is None or not np.isfinite(float(total_gb)) or float(total_gb) <= OMEGA_BASE_GB:
        raise ChunkLayoutError(f"the card's total memory ({total_gb} GB) does not exceed Omega's "
                               f"{OMEGA_BASE_GB:g} GB base — no frame fits")
    if ceiling < OMEGA_PATCH_SIZE or ceiling % OMEGA_PATCH_SIZE:
        raise ChunkLayoutError(f"Omega's resolution {ceiling} is not a positive multiple of its "
                               f"patch ({OMEGA_PATCH_SIZE})")
    head = {"mode": str(mode), "native_wh": [nw, nh], "ceiling": ceiling,
            "card_total_gb": round(float(total_gb), 3), "base_gb": OMEGA_BASE_GB,
            "gb_per_frame_ref": OMEGA_GB_PER_FRAME_REF, "ref_grid_wh": list(OMEGA_REF_GRID_WH),
            "largest_chunk_frames": n_frames, "patch": OMEGA_PATCH_SIZE,
            "margin_frac": float(margin_frac)}
    for res in range(ceiling, 0, -OMEGA_PATCH_SIZE):
        g = omega_grid_for(nw, nh, str(mode), res)
        cap = omega_capacity(total_gb, g.w, g.h, margin_frac)
        if n_frames <= cap:
            gb = OMEGA_GB_PER_FRAME_REF * g.w * g.h / float(OMEGA_REF_GRID_WH[0] * OMEGA_REF_GRID_WH[1])
            if res == ceiling:
                why = (f"the largest chunk ({n_frames} frames) fits the card at the configured "
                       f"resolution {ceiling} (grid {g.w}x{g.h}): capacity {cap} frames")
            else:
                g_up = omega_grid_for(nw, nh, str(mode), res + OMEGA_PATCH_SIZE)
                cap_up = omega_capacity(total_gb, g_up.w, g_up.h, margin_frac)
                why = (f"the largest chunk ({n_frames} frames) does not fit at {ceiling}; "
                       f"{res} (grid {g.w}x{g.h}, capacity {cap}) is the largest patch-aligned "
                       f"resolution that holds it ({res + OMEGA_PATCH_SIZE}: grid "
                       f"{g_up.w}x{g_up.h}, capacity {cap_up})")
            peak = OMEGA_BASE_GB + n_frames * float(gb)
            return {**head, "resolution": int(res), "grid_wh": [int(g.w), int(g.h)],
                    "gb_per_frame": round(float(gb), 6), "capacity_frames": int(cap),
                    # the prediction and what it leaves of the card (the margin is applied)
                    "predicted_peak_gb": round(peak, 3),
                    "headroom_gb": round(float(total_gb) - peak, 3),
                    "margin_applied": float(margin_frac) > 0.0,
                    "reduced": res != ceiling, "why": why, "provenance": "tool_measured"}
    raise ChunkLayoutError(f"{n_frames} frames do not fit a {float(total_gb):.1f} GB card (margin "
                           f"{float(margin_frac):.0%}) even at one Omega patch ({OMEGA_PATCH_SIZE} px)")


# ── the layout a run used, for every reader after Omega ──

def _ranges_of(doc: Any, key: str) -> Optional[List[Tuple[int, int]]]:
    r = doc.get(key) if isinstance(doc, dict) else None
    if r is None:
        return None
    try:
        out = as_ranges(r)
    except (TypeError, ValueError) as e:
        raise ChunkLayoutError(f"{key} is not a list of [start, end] pairs: {r!r}") from e
    if not out:
        raise ChunkLayoutError(f"{key} is empty")
    return out


def omega_chunk_ranges(output_dir: Path) -> Optional[Tuple[List[Tuple[int, int]], str]]:
    """(ranges, source) — the chunk layout Omega RAN for this session, in keyframe positions of
    its frame list: ``maplong_run/chunk_sim3.json`` ``chunk_indices`` (the fork writes what it
    ran), else the run config's ``Model.chunk_ranges`` (``output/vggt_omega_config.yaml``, what
    it was told to run), else ``output/chunk_plan.json`` ``chunk_ranges``. When the fork's
    record and the config both exist they must agree — the products on disk would otherwise
    belong to another plan (ChunkLayoutError). None when no record exists."""
    out = Path(output_dir)
    sim3_ranges = cfg_ranges = None
    p = out / "maplong_run" / CHUNK_SIM3_NAME
    if p.exists():
        try:
            sim3_ranges = _ranges_of(json.loads(p.read_text()), "chunk_indices")
        except (OSError, ValueError) as e:
            raise ChunkLayoutError(f"{p} is unreadable ({e})") from e
    c = out / OMEGA_CONFIG_NAME
    if c.exists():
        import yaml
        try:
            model = (yaml.full_load(c.read_text()) or {}).get("Model") or {}
        except (OSError, yaml.YAMLError) as e:
            raise ChunkLayoutError(f"{c} is unreadable ({e})") from e
        cfg_ranges = _ranges_of(model, "chunk_ranges")
    if sim3_ranges is not None and cfg_ranges is not None and sim3_ranges != cfg_ranges:
        raise ChunkLayoutError(f"{p} records the layout {sim3_ranges}, {c} configured "
                               f"{cfg_ranges} — the products on disk are of another chunk plan")
    if sim3_ranges is not None:
        return sim3_ranges, str(p)
    if cfg_ranges is not None:
        return cfg_ranges, f"{c} Model.chunk_ranges"
    pp = out / CHUNK_PLAN_NAME
    if pp.exists():
        try:
            r = _ranges_of(json.loads(pp.read_text()), "chunk_ranges")
        except (OSError, ValueError) as e:
            raise ChunkLayoutError(f"{pp} is unreadable ({e})") from e
        if r is not None:
            return r, str(pp)
    return None
