"""The ``gaussians:`` section of config.yaml as typed, frozen dataclasses. Every key is mandatory
and named when missing — no hidden default in code (the repo's rule for every stage that decides)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Sequence, Tuple

SECTION = "gaussians"
STRATEGIES = ("default", "mcmc")
EXPORT_FORMATS = ("ply", "splat")


class GaussiansConfigError(ValueError):
    pass


def _req(sec: Mapping[str, Any], key: str, path: str) -> Any:
    if not isinstance(sec, Mapping) or key not in sec:
        raise GaussiansConfigError(f"config.yaml is missing mandatory key '{SECTION}.{path}{key}' — add it; "
                                   f"there is no hidden default in code")
    return sec[key]


def _num(sec, key, path, *, lo=None, hi=None, integer=False, lo_excl=False) -> float:
    v = _req(sec, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be {'>' if lo_excl else '>='} {lo}, got {v!r}")
    if hi is not None and v > hi:
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be <= {hi}, got {v!r}")
    return int(v) if integer else float(v)


def _bool(sec, key, path) -> bool:
    v = _req(sec, key, path)
    if not isinstance(v, bool):
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be true or false, got {v!r}")
    return v


def _str(sec, key, path, allowed: Sequence[str] = ()) -> str:
    v = _req(sec, key, path)
    if not isinstance(v, str) or not v:
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be a non-empty string, got {v!r}")
    if allowed and v not in allowed:
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be one of {tuple(allowed)}, got {v!r}")
    return v


def _strs(sec, key, path, allowed: Sequence[str], min_len: int = 1) -> Tuple[str, ...]:
    v = _req(sec, key, path)
    if not isinstance(v, (list, tuple)) or len(v) < min_len or not all(isinstance(x, str) for x in v):
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be a list of at least {min_len} string(s), got {v!r}")
    bad = [x for x in v if x not in allowed]
    if bad:
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}': {bad} not in {tuple(allowed)}")
    return tuple(v)


def _sub(sec, key, path) -> Mapping[str, Any]:
    v = _req(sec, key, path)
    if not isinstance(v, Mapping):
        raise GaussiansConfigError(f"'{SECTION}.{path}{key}' must be a mapping")
    return v


@dataclass(frozen=True)
class InitConfig:
    max_points: int             # BOUND: the stable subsample of the published cloud the gaussians start from
    min_confidence: float       # a cloud point under this confidence does not seed a gaussian
    scale_knn: int              # the initial scale = the mean distance to this many nearest seeds
    seed: int                   # the subsample's permutation seed (recorded)
    opacity: float              # gsplat simple_trainer `init_opa` (the default preset; train.mcmc overrides)
    scale: float                # gsplat simple_trainer `init_scale`: × the knn distance (train.mcmc overrides)


@dataclass(frozen=True)
class LearningRates:
    means: float                # × the scene extent (gsplat's own convention)
    scales: float
    quats: float
    opacities: float
    sh0: float
    shN: float
    exposure: float


@dataclass(frozen=True)
class MCMCConfig:
    """gsplat 1.5.3 examples/simple_trainer.py `mcmc` preset + MCMCStrategy's own defaults, declared here."""
    init_opacity: float
    init_scale: float
    opacity_reg: float
    scale_reg: float
    noise_lr: float
    refine_start_iter: int
    refine_stop_iter: int
    refine_every: int
    min_opacity: float


@dataclass(frozen=True)
class ConfineConfig:
    """The splats stay on the measured surface (USER 2026-10-31): every `every` iterations and at the end, a
    gaussian with no published-cloud point within tolerance_m (per axis) is REMOVED, optimizer state included."""
    tolerance_m: float
    every: int


@dataclass(frozen=True)
class TrainConfig:
    iterations: int
    sh_degree: int              # 0..3
    sh_degree_interval: int     # a degree more every this many iterations (gsplat's warm-up)
    strategy: str               # default | mcmc
    max_gaussians: int          # BOUND: the mcmc cap; the default strategy stops refining at refine_stop_frac
    refine_start_frac: float    # densification window, as fractions of the iterations
    refine_stop_frac: float
    depth_lambda: float         # gsplat's depth loss: L1 of the disparity (1/z) against the measured depth × the
                                #   scene extent × this, where the session measured a depth (0 = off)
    means_lr_decay: float       # the means' lr decays exponentially to this fraction by the last iteration
    ssim_weight: float          # the photometric loss = (1 - w)·L1 + w·(1 - SSIM)
    exposure: bool              # a per-keyframe affine colour correction, learned
    exposure_reg: float         # its pull toward identity
    holdout_every: int          # every k-th keyframe is held out: the report's PSNR / SSIM are measured there
    eval_every: int             # held-out evaluation every this many iterations (and at the end)
    seed: int
    lr: LearningRates
    mcmc: MCMCConfig
    confine: ConfineConfig


@dataclass(frozen=True)
class ExportConfig:
    formats: Tuple[str, ...]    # ply (the 3DGS PLY, always the first) and/or splat (the viewer's compact form)


@dataclass(frozen=True)
class LabelConfig:
    nearest_radius_m: float     # a gaussian takes the instance of the nearest SEED gaussian within this radius


@dataclass(frozen=True)
class GaussiansConfig:
    enabled: bool
    name: str                   # output/gaussians/<name>/
    python: str                 # the interpreter of env `gs` (torch + gsplat)
    init: InitConfig
    train: TrainConfig
    export: ExportConfig
    label: LabelConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_gaussians_config(raw: Mapping[str, Any]) -> GaussiansConfig:
    """``raw`` is the whole configuration mapping (the frozen run config or config.yaml's)."""
    if not isinstance(raw, Mapping) or SECTION not in raw or not isinstance(raw[SECTION], Mapping):
        raise GaussiansConfigError(f"config.yaml is missing the '{SECTION}:' section")
    g = raw[SECTION]
    init = _sub(g, "init", "")
    tr = _sub(g, "train", "")
    lr = _sub(tr, "lr", "train.")
    mc = _sub(tr, "mcmc", "train.")
    cf = _sub(tr, "confine", "train.")
    ex = _sub(g, "export", "")
    lb = _sub(g, "label", "")
    sh = _num(tr, "sh_degree", "train.", lo=0, hi=3, integer=True)
    cfg = GaussiansConfig(
        enabled=_bool(g, "enabled", ""),
        name=_str(g, "name", ""),
        python=_str(g, "python", ""),
        init=InitConfig(max_points=_num(init, "max_points", "init.", lo=1000, integer=True),
                        min_confidence=_num(init, "min_confidence", "init."),
                        scale_knn=_num(init, "scale_knn", "init.", lo=1, integer=True),
                        seed=_num(init, "seed", "init.", lo=0, integer=True),
                        opacity=_num(init, "opacity", "init.", lo=0.0, lo_excl=True, hi=1.0),
                        scale=_num(init, "scale", "init.", lo=0.0, lo_excl=True)),
        train=TrainConfig(iterations=_num(tr, "iterations", "train.", lo=1, integer=True),
                          sh_degree=sh,
                          sh_degree_interval=_num(tr, "sh_degree_interval", "train.", lo=1, integer=True),
                          strategy=_str(tr, "strategy", "train.", STRATEGIES),
                          max_gaussians=_num(tr, "max_gaussians", "train.", lo=1000, integer=True),
                          refine_start_frac=_num(tr, "refine_start_frac", "train.", lo=0.0, hi=1.0),
                          refine_stop_frac=_num(tr, "refine_stop_frac", "train.", lo=0.0, hi=1.0),
                          depth_lambda=_num(tr, "depth_lambda", "train.", lo=0.0),
                          means_lr_decay=_num(tr, "means_lr_decay", "train.", lo=0.0, lo_excl=True, hi=1.0),
                          ssim_weight=_num(tr, "ssim_weight", "train.", lo=0.0, hi=1.0),
                          exposure=_bool(tr, "exposure", "train."),
                          exposure_reg=_num(tr, "exposure_reg", "train.", lo=0.0),
                          holdout_every=_num(tr, "holdout_every", "train.", lo=2, integer=True),
                          eval_every=_num(tr, "eval_every", "train.", lo=1, integer=True),
                          seed=_num(tr, "seed", "train.", lo=0, integer=True),
                          lr=LearningRates(**{k: _num(lr, k, "train.lr.", lo=0.0, lo_excl=True)
                                              for k in ("means", "scales", "quats", "opacities", "sh0", "shN", "exposure")}),
                          mcmc=MCMCConfig(init_opacity=_num(mc, "init_opacity", "train.mcmc.", lo=0.0, lo_excl=True, hi=1.0),
                                          init_scale=_num(mc, "init_scale", "train.mcmc.", lo=0.0, lo_excl=True),
                                          opacity_reg=_num(mc, "opacity_reg", "train.mcmc.", lo=0.0),
                                          scale_reg=_num(mc, "scale_reg", "train.mcmc.", lo=0.0),
                                          noise_lr=_num(mc, "noise_lr", "train.mcmc.", lo=0.0),
                                          refine_start_iter=_num(mc, "refine_start_iter", "train.mcmc.", lo=0, integer=True),
                                          refine_stop_iter=_num(mc, "refine_stop_iter", "train.mcmc.", lo=0, integer=True),
                                          refine_every=_num(mc, "refine_every", "train.mcmc.", lo=1, integer=True),
                                          min_opacity=_num(mc, "min_opacity", "train.mcmc.", lo=0.0, hi=1.0)),
                          confine=ConfineConfig(tolerance_m=_num(cf, "tolerance_m", "train.confine.", lo=0.0, lo_excl=True),
                                                every=_num(cf, "every", "train.confine.", lo=1, integer=True))),
        export=ExportConfig(formats=_strs(ex, "formats", "export.", EXPORT_FORMATS)),
        label=LabelConfig(nearest_radius_m=_num(lb, "nearest_radius_m", "label.", lo=0.0, lo_excl=True)),
    )
    if cfg.train.refine_stop_frac < cfg.train.refine_start_frac:
        raise GaussiansConfigError(f"'{SECTION}.train.refine_stop_frac' ({cfg.train.refine_stop_frac}) is under "
                                   f"refine_start_frac ({cfg.train.refine_start_frac})")
    if cfg.export.formats[0] != "ply":
        raise GaussiansConfigError(f"'{SECTION}.export.formats' must start with 'ply' (the 3DGS PLY is the record)")
    return cfg
