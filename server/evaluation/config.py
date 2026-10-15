"""Typed configuration of the evaluation battery — ``config.yaml`` section ``evaluation:``.

Rule 7 of claude_stac.txt: no price, tolerance or path lives in the code. Every key is REQUIRED
(a missing one fails the load naming it, the repo's rule), so a value that decides something is
always visible in the YAML next to the reason it has.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

SECTION = "evaluation"


class EvaluationConfigError(ValueError):
    """config.yaml ``evaluation:`` is missing, incomplete or holds a value outside its bounds."""


def _require(section: Mapping[str, Any], key: str, path: str) -> Any:
    if not isinstance(section, Mapping):
        raise EvaluationConfigError(f"'{SECTION}.{path}' must be a mapping, got "
                                    f"{type(section).__name__}")
    if key not in section:
        raise EvaluationConfigError(f"'{SECTION}.{path}.{key}' is missing from config.yaml"
                                    if path else f"'{SECTION}.{key}' is missing from config.yaml")
    return section[key]


def _name(path: str, key: str) -> str:
    return f"{SECTION}.{path}.{key}" if path else f"{SECTION}.{key}"


def _num(section, key, path, lo=None, hi=None, integer=False, lo_excl=False):
    v = _require(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EvaluationConfigError(f"'{_name(path, key)}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise EvaluationConfigError(f"'{_name(path, key)}' must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise EvaluationConfigError(f"'{_name(path, key)}' must be {'>' if lo_excl else '>='} "
                                    f"{lo}, got {v!r}")
    if hi is not None and v > hi:
        raise EvaluationConfigError(f"'{_name(path, key)}' must be <= {hi}, got {v!r}")
    return int(v) if integer else float(v)


def _bool(section, key, path) -> bool:
    v = _require(section, key, path)
    if not isinstance(v, bool):
        raise EvaluationConfigError(f"'{_name(path, key)}' must be true or false, got {v!r}")
    return v


def _str(section, key, path, allowed: Optional[Sequence[str]] = None) -> str:
    v = _require(section, key, path)
    if not isinstance(v, str) or not v:
        raise EvaluationConfigError(f"'{_name(path, key)}' must be a non-empty string, got {v!r}")
    if allowed is not None and v not in allowed:
        raise EvaluationConfigError(f"'{_name(path, key)}' must be one of {list(allowed)}, got {v!r}")
    return v


def _num_list(section, key, path, lo=None, min_len=1, increasing=False) -> Tuple[float, ...]:
    v = _require(section, key, path)
    if not isinstance(v, (list, tuple)) or len(v) < min_len:
        raise EvaluationConfigError(f"'{_name(path, key)}' must be a list of >= {min_len} numbers")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise EvaluationConfigError(f"'{_name(path, key)}' holds a non-number {x!r}")
        if lo is not None and x < lo:
            raise EvaluationConfigError(f"'{_name(path, key)}' holds {x!r} < {lo}")
        out.append(float(x))
    if increasing and any(b <= a for a, b in zip(out, out[1:])):
        raise EvaluationConfigError(f"'{_name(path, key)}' must be strictly increasing")
    return tuple(out)


def _sub(section, key, path) -> Dict[str, Any]:
    v = _require(section, key, path)
    if not isinstance(v, Mapping):
        raise EvaluationConfigError(f"'{_name(path, key)}' must be a mapping, got {type(v).__name__}")
    return dict(v)


# ── phase 0 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReferencesConfig:
    file: str           # file name under the scan's output/ holding the reference measures


# ── phases 1–2 ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EndpointsConfig:
    depth_window_px: int        # BOUND: half-window around the anchor pixel whose valid depths
                                # are pooled (median); 0 = the anchor pixel alone
    zbuffer_window_px: int      # BOUND: half-window of the dense-cloud z-buffer read for a
                                # variant without depth maps (the nearest splat wins)
    min_triangulation_deg: float    # BOUND: two rays crossing under this parallax do not
                                    # triangulate — the end is declared "not reconstructed"


@dataclass(frozen=True)
class PlaneFitConfig:
    max_points: int     # BOUND: stable subsample of a reference region before the plane fit
    seed: int           # seed of that subsample (recorded)
    neighbour_k: int    # BOUND: k of the k-NN spacing (resolution)


@dataclass(frozen=True)
class CurvesConfig:
    distance_bins_m: Tuple[float, ...]  # edges of the camera-distance bins
    walk_bins_m: Tuple[float, ...]      # edges of the walk-from-the-nearest-gauge bins
    distance_statistic: str             # a distance's camera distance = mean | max of its two ends'
    tolerances_m: Tuple[float, ...]     # tolerances of the "confidence distance"
    min_bin_n: int                      # BOUND: a bin with fewer samples reports "no data"
    plane_fit: PlaneFitConfig


# ── phase 4 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostConfig:
    sample_interval_s: float            # nvidia-smi / RSS sampling period of the stage monitor
    gpu_price_per_hour: Dict[str, float]    # USD per GPU hour by card key; "default" = unlisted card


# ── phase 5 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfidenceConfig:
    bins: int           # BOUND: quantile bins of the confidence column
    min_group_n: int    # BOUND: a group with fewer points reports "no data"


# ── phase 7 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProfilesConfig:
    min_test_coverage: float    # share of the test ends a variant must reconstruct to qualify


@dataclass(frozen=True)
class EvaluationConfig:
    references: ReferencesConfig
    runs_dir: str                   # where runs land (relative to the repo root, or absolute)
    scenes: Dict[str, str]          # scene name → session (scan) dir; the CLI also takes paths
    endpoints: EndpointsConfig
    curves: CurvesConfig
    cost: CostConfig
    confidence: ConfidenceConfig
    profiles: ProfilesConfig
    deterministic: bool             # phase 6 mode (seeds, strict torch, fixed order) — recorded
    raw: Dict[str, Any]             # the section as read (variants / colmap are read by their
                                    # own loaders in evaluation.variants)


def load_evaluation_config(raw: Optional[Mapping[str, Any]] = None) -> EvaluationConfig:
    """``raw`` = the whole config mapping (``config.cfg`` when None)."""
    if raw is None:
        from config import cfg as raw  # type: ignore[no-redef]
    if SECTION not in raw:
        raise EvaluationConfigError(f"config.yaml has no '{SECTION}:' section")
    sec = raw[SECTION]
    if not isinstance(sec, Mapping):
        raise EvaluationConfigError(f"'{SECTION}' must be a mapping")
    refs = _sub(sec, "references", "")
    ends = _sub(sec, "endpoints", "")
    curves = _sub(sec, "curves", "")
    pf = _sub(curves, "plane_fit", "curves")
    cost = _sub(sec, "cost", "")
    prices = _sub(cost, "gpu_price_per_hour", "cost")
    if "default" not in prices:
        raise EvaluationConfigError(f"'{SECTION}.cost.gpu_price_per_hour' needs a 'default' entry")
    for k, v in prices.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise EvaluationConfigError(f"'{SECTION}.cost.gpu_price_per_hour.{k}' must be a "
                                        f"non-negative number, got {v!r}")
    conf = _sub(sec, "confidence", "")
    prof = _sub(sec, "profiles", "")
    scenes = _sub(sec, "scenes", "")
    for k, v in scenes.items():
        if not isinstance(v, str) or not v:
            raise EvaluationConfigError(f"'{SECTION}.scenes.{k}' must be a path string")
    return EvaluationConfig(
        references=ReferencesConfig(file=_str(refs, "file", "references")),
        runs_dir=_str(sec, "runs_dir", ""),
        scenes={str(k): str(v) for k, v in scenes.items()},
        endpoints=EndpointsConfig(
            depth_window_px=_num(ends, "depth_window_px", "endpoints", lo=0, integer=True),
            zbuffer_window_px=_num(ends, "zbuffer_window_px", "endpoints", lo=0, integer=True),
            min_triangulation_deg=_num(ends, "min_triangulation_deg", "endpoints", lo=0.0)),
        curves=CurvesConfig(
            distance_bins_m=_num_list(curves, "distance_bins_m", "curves", lo=0.0, min_len=2,
                                      increasing=True),
            walk_bins_m=_num_list(curves, "walk_bins_m", "curves", lo=0.0, min_len=2,
                                  increasing=True),
            distance_statistic=_str(curves, "distance_statistic", "curves", allowed=("mean", "max")),
            tolerances_m=_num_list(curves, "tolerances_m", "curves", lo=0.0, min_len=1),
            min_bin_n=_num(curves, "min_bin_n", "curves", lo=1, integer=True),
            plane_fit=PlaneFitConfig(
                max_points=_num(pf, "max_points", "curves.plane_fit", lo=10, integer=True),
                seed=_num(pf, "seed", "curves.plane_fit", lo=0, integer=True),
                neighbour_k=_num(pf, "neighbour_k", "curves.plane_fit", lo=1, integer=True))),
        cost=CostConfig(
            sample_interval_s=_num(cost, "sample_interval_s", "cost", lo=0.0, lo_excl=True),
            gpu_price_per_hour={str(k): float(v) for k, v in prices.items()}),
        confidence=ConfidenceConfig(
            bins=_num(conf, "bins", "confidence", lo=1, integer=True),
            min_group_n=_num(conf, "min_group_n", "confidence", lo=1, integer=True)),
        profiles=ProfilesConfig(
            min_test_coverage=_num(prof, "min_test_coverage", "profiles", lo=0.0, hi=1.0)),
        deterministic=_bool(sec, "deterministic", ""),
        raw=dict(sec),
    )


def repo_root() -> Path:
    """The repository root (server/ is its child)."""
    return Path(__file__).resolve().parents[2]


def runs_root(cfg: EvaluationConfig) -> Path:
    p = Path(cfg.runs_dir)
    return p if p.is_absolute() else repo_root() / p
