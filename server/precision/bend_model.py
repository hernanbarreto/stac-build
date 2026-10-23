"""The per-keyframe depth model of the bend (depth_on_f5 step 2) — WHICH family of corrections a
keyframe's Omega depth may take, fitted on F5's landmarks.

AUDIT 2026-10-23. Epoch 7's family, ``affine_pixel``, multiplies Omega's depth by a ratio that is
affine in the pixel: z' = z · (c0 + c1·u' + c2·v'). It is not a geometric transformation: for a plane
1/z is affine in (u', v'), so 1/z' = affine / affine — every plane seen obliquely (a floor, a wall
along a corridor) leaves the bend CURVED whenever c1 or c2 is not zero. Measured on pccr's camera
(464x832, f 388 px), a floor 1.5 m under the camera: c1 = 0.03 → 2.5 cm from its best plane,
c1 = 0.10 → 8.8 cm (tests/test_bend_model.py). The tilt terms entered because they lowered the
held-out error on the landmarks — points on textured things — and were applied to every pixel,
textureless walls included.

``inverse_depth`` is affine in INVERSE depth: 1/z' = a/z + b0 + b1·u' + b2·v'. It is the
classic scale-and-shift ambiguity of monocular depth plus a plane-at-infinity tilt — a projective
map along the ray that takes planes to planes EXACTLY (affine in 1/z stays affine in 1/z), with
the pure scale as its a-only member. Same unknown count as the pixel family plus the shift, fitted
with the same Huber IRLS, gated by the same significance rule (point 48), judged by the same
held-out rows (point 49).

``judged`` (the default since 2026-10-23): both families are fitted on every window and the
held-out half A chooses between them by the window rule's own construction — the best (family,
window) by pooled median sets the bar (best + error_factor × its bootstrap standard error) and,
among the candidates within the bar, the plane-preserving family wins the tie, then the widest
window. A session whose landmarks really prefer the pixel family keeps it, declared.

Every function takes ``model`` by name; the coefficient vector is the family's own (3 for
affine_pixel, 4 for inverse_depth) and the report writes ``model`` next to it.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

AFFINE_PIXEL = "affine_pixel"
INVERSE_DEPTH = "inverse_depth"
MODELS = (AFFINE_PIXEL, INVERSE_DEPTH)
JUDGED = "judged"
MODEL_CHOICES = MODELS + (JUDGED,)
# the family that cannot bend a plane — the one that wins a held-out tie under `judged`
PLANE_PRESERVING = INVERSE_DEPTH


class BendModelError(RuntimeError):
    pass


def check_model(model: str) -> str:
    if model not in MODELS:
        raise BendModelError(f"bend model must be one of {MODELS}, got {model!r}")
    return model


def n_coef(model: str) -> int:
    return 3 if check_model(model) == AFFINE_PIXEL else 4


def identity(model: str) -> np.ndarray:
    """The coefficients that leave Omega's depth as it is (k = 1)."""
    return np.array([1.0, 0.0, 0.0]) if check_model(model) == AFFINE_PIXEL else np.array([1.0, 0.0, 0.0, 0.0])


def gated_columns(model: str) -> Tuple[int, ...]:
    """The columns point 48 gates by significance (the first column — the scale — always enters)."""
    return (1, 2) if check_model(model) == AFFINE_PIXEL else (1, 2, 3)


def pixel_columns(u: np.ndarray, v: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """u', v': the pixel centred and normalised by the image (depth_on_f5.design's convention)."""
    return (np.asarray(u, np.float64) - W / 2) / W, (np.asarray(v, np.float64) - H / 2) / H


def design(model: str, u: np.ndarray, v: np.ndarray, z_omega: np.ndarray, W: int, H: int) -> np.ndarray:
    """The fit's design matrix for rows at pixels (u, v) with Omega depth ``z_omega``:
    affine_pixel [1, u', v'] (target z_lm / z_om); inverse_depth [1/z_om, 1, u', v'] (target 1/z_lm)."""
    up, vp = pixel_columns(u, v, W, H)
    n = len(up)
    if check_model(model) == AFFINE_PIXEL:
        return np.c_[np.ones(n), up, vp]
    zo = np.asarray(z_omega, np.float64)
    return np.c_[1.0 / zo, np.ones(n), up, vp]


def target(model: str, z_landmark: np.ndarray, z_omega: np.ndarray) -> np.ndarray:
    """What the design matrix times the coefficients should reproduce, per row."""
    zl = np.asarray(z_landmark, np.float64); zo = np.asarray(z_omega, np.float64)
    return zl / zo if check_model(model) == AFFINE_PIXEL else 1.0 / zl


def fit_rows(model: str, u: np.ndarray, v: np.ndarray, z_landmark: np.ndarray, z_omega: np.ndarray,
             W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """(A, r) of a keyframe's landmark rows for ``model`` — what the IRLS fits."""
    return design(model, u, v, z_omega, W, H), target(model, z_landmark, z_omega)


def apply(model: str, coef: np.ndarray, z_omega: np.ndarray, u: np.ndarray, v: np.ndarray,
          W: int, H: int) -> np.ndarray:
    """Omega's depth at pixels (u, v) under the model: z' = z·k(u', v') (affine_pixel) or
    z' = 1 / (a/z + b0 + b1·u' + b2·v') (inverse_depth). A pixel the inverse model sends to a
    non-positive inverse depth (behind the camera / at infinity) gets 0 — no depth, declared by the
    value, never a negative one. Shapes broadcast: ``u, v`` may be meshgrids of a map."""
    c = np.asarray(coef, np.float64)
    if len(c) != n_coef(model):
        raise BendModelError(f"{model} takes {n_coef(model)} coefficients, got {len(c)}")
    zo = np.asarray(z_omega, np.float64)
    up, vp = pixel_columns(u, v, W, H)
    if model == AFFINE_PIXEL:
        return zo * (c[0] + c[1] * up + c[2] * vp)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = c[0] / zo + c[1] + c[2] * up + c[3] * vp
        out = np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1.0), 0.0)
    return out


def apply_map(model: str, coef: np.ndarray, z_omega: np.ndarray, W: int, H: int) -> np.ndarray:
    """``apply`` over a whole H x W depth map."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return apply(model, coef, z_omega, uu, vv, W, H)


def scale_equivalent(model: str, coef: np.ndarray) -> float:
    """The per-keyframe SCALE the model amounts to at the image centre, ignoring the shift — what
    the reports call s_k: c0 (affine_pixel) or 1/a (inverse_depth)."""
    c = np.asarray(coef, np.float64)
    if check_model(model) == AFFINE_PIXEL:
        return float(c[0])
    return float(1.0 / c[0]) if c[0] != 0 else float("inf")


def tilt_terms(model: str, coef: np.ndarray) -> Tuple[float, ...]:
    """The coefficients beyond the scale, as the report lists them under ``bend``."""
    c = np.asarray(coef, np.float64)
    return tuple(float(x) for x in c[1:])


def plane_rms_m(z: np.ndarray, K: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    """(rms, max) distance in metres of the unprojected VALID pixels of a depth map from their
    best-fit plane (total least squares) — the instrument of tests/test_bend_model.py and of the
    planarity report: a family that bends planes reads > 0 on a plane."""
    H, W = z.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    m = np.asarray(valid, bool) & np.isfinite(z) & (z > 0)
    zz = z[m]
    X = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * zz, (vv[m] - K[1, 2]) / K[1, 1] * zz, zz], 1)
    if len(X) < 3:
        return float("nan"), float("nan")
    c = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    r = (X - c) @ Vt[-1]
    return float(np.sqrt(np.mean(r * r))), float(np.abs(r).max())
