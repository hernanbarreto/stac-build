# STAC-Builder — which keyframes, and which crops of them, the VLM looks at to
# NAME the scene (the SAM3 vocabulary).
#
# USER 2026-09-29: *"la segmentación VLM no es lo suficientemente detallada …
# debe segmentar todo, absolutamente preciso y completo"*. In the SIMPLE
# pipeline the phrases the VLM writes ARE the SAM3 prompts, so what it never
# sees is never named, never segmented, never measured.
#
# What it saw on pccr 2026-09-29, from the run's own log: the VLM runs at the
# intake (I2), before any geometry exists, so the coverage cover
# (`coverage_sample.cover_keyframes`) found no cloud and fell back to
# `understand_sample: 8` keyframes by linspace — 8 images for a 17.5 m walk of
# 289 keyframes, one of which did not parse. A desk seen for two metres of the
# walk between two samples was simply never looked at.
#
# THE RULE NOW: the frames are spread UNIFORMLY ALONG THE WALK at a declared
# density — by the walked chainage of `intake/walk.json` when it covers every
# keyframe (metres), else by keyframe index (keyframes are parallax-uniform, so
# the index is the walk as the camera experienced it). Optionally every sampled
# frame is also shown as a grid of overlapping crops, each enlarged to the
# frame's own size: a crop shown at its own pixel count gives the VLM exactly
# the pixels per object the full frame did — cutting only narrows the
# attention; enlarging it is what lets a small object (a doorbell panel, a
# conduit) fill enough of the image to be named.
#
# The number of VLM calls this implies (frames × (1 + crops)) is logged and
# BOUNDED by `max_calls`: past it the frames are thinned uniformly along the
# same axis (never the crops) and the plan says so. Nothing here measures
# anything — it only decides which pixels the VLM is shown.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SECTION = "autoprompt.vlm_sampling"
AXIS_CHAINAGE = "chainage_m"
AXIS_INDEX = "keyframe_index"
AXIS_COVER = "coverage"
AXIS_ALL = "all_keyframes"
# keys that decided the sampling before this module and must not linger in a
# config that no longer reads them (a leftover would look like a decision)
REMOVED_KEYS = {"understand_sample": "autoprompt.vlm_sampling.spacing_kf"}


class VLMSamplingConfigError(RuntimeError):
    """A missing / invalid ``autoprompt.vlm_sampling`` key — always named."""


@dataclass(frozen=True)
class VLMSampling:
    all_keyframes: bool         # USER 2026-09-29: the VLM sees EVERY keyframe (no density, no
                                # thinning — max_calls is not applied in this mode, declared)
    spacing_m: float            # one VLM frame per this many metres of walked chainage
    spacing_kf: int             # without chainage: one VLM frame per this many keyframes
    tile_rows: int              # crop grid per sampled frame; 1 x 1 = no crops
    tile_cols: int
    tile_overlap_frac: float    # share of a crop's width/height its neighbour repeats
    max_calls: int              # BOUND on VLM calls (frames x (1 + crops))

    @property
    def n_tiles(self) -> int:
        n = int(self.tile_rows) * int(self.tile_cols)
        return 0 if n == 1 else n

    @property
    def calls_per_frame(self) -> int:
        return 1 + self.n_tiles


def _req(sec: dict, key: str):
    if not isinstance(sec, dict) or key not in sec:
        raise VLMSamplingConfigError(
            f"config.yaml is missing mandatory key '{SECTION}.{key}' — there is no "
            f"hidden default in code")
    return sec[key]


def _num(sec, key, *, integer=False, lo=None, lo_excl=False, hi=None, hi_excl=False):
    v = _req(sec, key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise VLMSamplingConfigError(f"'{SECTION}.{key}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise VLMSamplingConfigError(f"'{SECTION}.{key}' must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise VLMSamplingConfigError(
            f"'{SECTION}.{key}' = {v} is below its range (min {lo}"
            f"{' exclusive' if lo_excl else ''})")
    if hi is not None and (v >= hi if hi_excl else v > hi):
        raise VLMSamplingConfigError(
            f"'{SECTION}.{key}' = {v} is above its range (max {hi}"
            f"{' exclusive' if hi_excl else ''})")
    return int(v) if integer else float(v)


def load_vlm_sampling(config: dict) -> VLMSampling:
    """Typed, strict read of ``autoprompt.vlm_sampling`` (a missing key fails
    naming it; a removed key still present fails naming its replacement)."""
    ap = (config or {}).get("autoprompt")
    if not isinstance(ap, dict):
        raise VLMSamplingConfigError("config.yaml has no 'autoprompt' section")
    for old, new in REMOVED_KEYS.items():
        if old in ap:
            raise VLMSamplingConfigError(
                f"'autoprompt.{old}' was REMOVED (the VLM frames are spread along the "
                f"walk now) — delete it; the density is '{new}'")
    sec = _req(ap, "vlm_sampling")
    all_kf = _req(sec, "all_keyframes")
    if not isinstance(all_kf, bool):
        raise VLMSamplingConfigError(f"'{SECTION}.all_keyframes' must be true or false, got {all_kf!r}")
    cfg = VLMSampling(
        all_keyframes=all_kf,
        spacing_m=_num(sec, "spacing_m", lo=0.0, lo_excl=True),
        spacing_kf=_num(sec, "spacing_kf", integer=True, lo=1),
        tile_rows=_num(sec, "tile_rows", integer=True, lo=1),
        tile_cols=_num(sec, "tile_cols", integer=True, lo=1),
        tile_overlap_frac=_num(sec, "tile_overlap_frac", lo=0.0, hi=1.0, hi_excl=True),
        max_calls=_num(sec, "max_calls", integer=True, lo=1),
    )
    if cfg.max_calls < cfg.calls_per_frame:
        raise VLMSamplingConfigError(
            f"'{SECTION}.max_calls' = {cfg.max_calls} cannot hold even ONE frame with its "
            f"{cfg.n_tiles} crop(s) ({cfg.calls_per_frame} calls) — raise it or shrink "
            f"the crop grid")
    return cfg


def load_max_sam3_prompts(config: dict) -> int:
    """``autoprompt.max_sam3_prompts`` — the BOUND on the SAM3 prompts the
    understanding may hand over (strict: a missing key fails naming it)."""
    ap = (config or {}).get("autoprompt")
    if not isinstance(ap, dict) or "max_sam3_prompts" not in ap:
        raise VLMSamplingConfigError(
            "config.yaml is missing mandatory key 'autoprompt.max_sam3_prompts' — the "
            "BOUND on SAM3 prompts; there is no hidden default in code")
    v = ap["max_sam3_prompts"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < 1:
        raise VLMSamplingConfigError(
            f"'autoprompt.max_sam3_prompts' must be an integer >= 1, got {v!r}")
    return int(v)


# ── the walk axis ─────────────────────────────────────────────────────────

def _frame_num(filename: str) -> int:
    return int(os.path.splitext(os.path.basename(filename))[0])


def walk_chainage(session_dir, files: Sequence[str]) -> Tuple[Optional[Dict[int, float]], str]:
    """``{video frame: chainage_m}`` from ``<session>/intake/walk.json`` when it
    covers EVERY keyframe in ``files``; otherwise ``(None, why)``. At the intake
    of a fresh session the walk is not measured yet (it is measured after I2),
    so the keyframe index is the axis there — declared, not an error."""
    p = Path(session_dir) / "intake" / "walk.json"
    if not p.is_file():
        return None, "intake/walk.json not on disk yet (the walk is measured after I2)"
    try:
        doc = json.loads(p.read_text())
        ch = {int(e["frame"]): float(e["chainage_m"]) for e in doc.get("chainage") or []}
    except Exception as e:  # noqa: BLE001 — an unreadable walk is no walk
        return None, f"intake/walk.json unreadable ({e})"
    missing = [f for f in files if _frame_num(f) not in ch]
    if missing:
        return None, (f"intake/walk.json does not cover {len(missing)} of the "
                      f"{len(files)} keyframes (e.g. {missing[:3]}) — it belongs to "
                      f"another keyframe set")
    return {_frame_num(f): ch[_frame_num(f)] for f in files}, "intake/walk.json"


# ── the plan ──────────────────────────────────────────────────────────────

@dataclass
class SamplingPlan:
    axis: str                                  # chainage_m | keyframe_index | coverage
    axis_reason: str
    configured_spacing: Optional[float]        # along the axis (None for coverage)
    effective_spacing: Optional[float]         # after the bound (None for coverage)
    n_keyframes: int
    frames: List[dict] = field(default_factory=list)   # file, frame, keyframe_index, position
    n_tiles: int = 0
    tile_grid: Tuple[int, int] = (1, 1)
    tile_overlap_frac: float = 0.0
    n_calls: int = 0
    max_calls: int = 0
    bound_reached: bool = False
    n_before_bound: int = 0

    def files(self) -> List[str]:
        return [f["file"] for f in self.frames]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tile_grid"] = list(self.tile_grid)
        return d

    def summary(self) -> str:
        unit = {"chainage_m": "m", "keyframe_index": "keyframe(s)"}.get(self.axis, "")
        how = ("EVERY keyframe (all_keyframes: the density and the call bound are not applied)"
               if self.axis == AXIS_ALL else
               f"uniform along the walk by {self.axis}, one every "
               f"{self.effective_spacing:.3g} {unit}" if self.effective_spacing is not None
               else "the coverage cover")
        s = (f"VLM sampling: {len(self.frames)}/{self.n_keyframes} keyframe(s), {how} "
             f"({self.axis_reason}); {self.n_tiles} crop(s) each "
             f"(grid {self.tile_grid[0]}x{self.tile_grid[1]}, overlap "
             f"{self.tile_overlap_frac:g}) → {self.n_calls} VLM call(s), bound {self.max_calls}")
        if self.bound_reached:
            s += (f" — BOUND REACHED: {self.n_before_bound} frame(s) at the configured "
                  f"density would need {self.n_before_bound * (1 + self.n_tiles)} calls; "
                  f"thinned uniformly to {len(self.frames)}")
        return s


def _nearest_unique(positions: np.ndarray, targets: np.ndarray) -> List[int]:
    """For every target the index of the keyframe whose position is nearest
    (ties → the earlier one), duplicates dropped, walk order kept."""
    out = set()
    for t in targets:
        j = int(np.searchsorted(positions, t, side="left"))
        cand = [c for c in (j - 1, j) if 0 <= c < len(positions)]
        out.add(min(cand, key=lambda c: (abs(float(positions[c]) - float(t)), c)))
    return sorted(out)


def _uniform(positions: np.ndarray, spacing: float, n_max: int) -> Tuple[List[int], int, float]:
    """Indices spread uniformly over [first, last] position at ``spacing`` or
    denser (both ends always in), capped at ``n_max`` frames. Returns
    (indices, n requested before the cap, effective spacing)."""
    L = float(positions[-1] - positions[0]) if len(positions) else 0.0
    n_want = 1 if L <= 0 else int(math.ceil(L / spacing - 1e-9)) + 1
    n_want = min(n_want, len(positions))
    n = min(n_want, n_max)
    targets = np.linspace(float(positions[0]), float(positions[-1]), n) if n > 1 \
        else np.array([float(positions[0])])
    eff = (L / (n - 1)) if n > 1 else L
    return _nearest_unique(positions, targets), n_want, eff


def plan_vlm_frames(files: Sequence[str], cfg: VLMSampling, *,
                    chainage: Optional[Dict[int, float]] = None,
                    axis_reason: str = "",
                    preselected: Optional[Sequence[str]] = None) -> SamplingPlan:
    """Which of ``files`` (the keyframes, walk order) the VLM is shown.

    ``chainage`` (``{video frame: metres}`` covering every file) makes the axis
    the walked distance; without it the axis is the keyframe index.
    ``preselected`` (the optional coverage cover) replaces the density rule —
    only the bound thins it. ``max_calls`` holds in every mode but ``all_keyframes``
    (USER 2026-09-29: "el VLM debe ver todos los KF"), where every keyframe is shown
    and the plan says the bound was not applied."""
    files = list(files)
    if not files:
        raise ValueError("no keyframes to sample for the VLM")
    n_max = max(1, cfg.max_calls // cfg.calls_per_frame)
    pos_of = {f: i for i, f in enumerate(files)}

    if cfg.all_keyframes:
        sel = list(range(len(files)))
        n_before = len(sel)
        n_max = len(sel)                          # the bound is not applied: every keyframe
        axis, conf_sp, eff_sp = AXIS_ALL, 1.0, 1.0
        positions = np.arange(len(files), dtype=float)
    elif preselected:
        sel = [pos_of[f] for f in preselected if f in pos_of]
        n_before = len(sel)
        if len(sel) > n_max:
            keep = np.linspace(0, len(sel) - 1, n_max).round().astype(int)
            sel = [sel[k] for k in sorted(set(keep.tolist()))]
        axis, conf_sp, eff_sp = AXIS_COVER, None, None
        positions = np.arange(len(files), dtype=float)
    else:
        if chainage is not None:
            axis = AXIS_CHAINAGE
            positions = np.array([chainage[_frame_num(f)] for f in files], float)
            # chainage is cumulative; a non-monotone document is not a walk
            positions = np.maximum.accumulate(positions)
            conf_sp = float(cfg.spacing_m)
        else:
            axis = AXIS_INDEX
            positions = np.arange(len(files), dtype=float)
            conf_sp = float(cfg.spacing_kf)
        sel, n_before, eff_sp = _uniform(positions, conf_sp, n_max)

    frames = [{"file": files[i], "frame": _frame_num(files[i]), "keyframe_index": int(i),
               "position": float(positions[i])} for i in sel]
    return SamplingPlan(
        axis=axis, axis_reason=axis_reason or axis,
        configured_spacing=conf_sp, effective_spacing=eff_sp,
        n_keyframes=len(files), frames=frames,
        n_tiles=cfg.n_tiles, tile_grid=(cfg.tile_rows, cfg.tile_cols),
        tile_overlap_frac=cfg.tile_overlap_frac,
        n_calls=len(frames) * cfg.calls_per_frame, max_calls=cfg.max_calls,
        bound_reached=n_before > n_max,
        n_before_bound=n_before)


# ── the crops ─────────────────────────────────────────────────────────────

def tile_boxes(width: int, height: int, rows: int, cols: int,
               overlap: float) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """``[(tile_id, (x0, y0, x1, y1))]`` — a ``rows x cols`` grid of crops that
    covers the whole image, neighbours sharing ``overlap`` of a crop's size.
    A 1 x 1 grid IS the full frame: no crops."""
    rows, cols = int(rows), int(cols)
    if rows * cols == 1:
        return []
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"tile overlap must be in [0, 1), got {overlap}")
    tw = width / (cols - (cols - 1) * overlap)
    th = height / (rows - (rows - 1) * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * tw * (1.0 - overlap)))
            y0 = int(round(r * th * (1.0 - overlap)))
            x1 = width if c == cols - 1 else int(round(x0 + tw))
            y1 = height if r == rows - 1 else int(round(y0 + th))
            out.append((f"r{r}c{c}", (x0, y0, min(x1, width), min(y1, height))))
    return out


def crop_for_vlm(image, box: Tuple[int, int, int, int]):
    """The crop enlarged (aspect kept) until it fills the frame's own size —
    the frame is the reference, no pixel count is invented here."""
    from PIL import Image
    W, H = image.size
    crop = image.crop(box)
    cw, ch = crop.size
    s = min(W / max(1, cw), H / max(1, ch))
    if s <= 1.0:
        return crop
    return crop.resize((max(1, int(round(cw * s))), max(1, int(round(ch * s)))),
                       Image.BICUBIC)
