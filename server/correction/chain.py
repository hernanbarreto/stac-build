"""The epoch CHAIN as the ONE source of what a session has applied (docs/plan_determinismo.md
points 135, 136, 147 — 2026-10-08).

Every transform epoch persists its exact per-keyframe warp in ``corrections/epoch_<N>.npz``
(correction.ledger.save_epoch_npz). Anything that needs to know what the live geometry holds
over the reconstruction's PRODUCT epoch — the depth sidecar ``depth_correction.json`` the TSDF,
the hole audit and the VLM/SAM3 depth provider read, the ``scale_diagnostics.json`` history, the
anchor rows' "has this chunk moved" question, the certification's own naming — composes those
files from the product epoch to the live one. Nothing reads a sidecar left on disk by a previous
run and nothing appends to a history: the chain is re-derived every time, so two runs of the
same session write the same bytes whatever ran before.

The PRODUCT epoch (:func:`base_epoch`) is the most recent new-cloud epoch of the live lineage —
the cloud precision/corrected_cloud.publish wrote (``corrected_cloud.json``); epoch 0 when the
session holds none (a legacy or synthetic session).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from correction.epoch import (EPOCH_DIR_PREFIX, EPOCH_KIND_NEW_CLOUD, current_epoch,
                              epoch_kind, epoch_lineage)
from correction.ledger import EPOCH_NPZ_DIR, load_epoch_npz

QUALITY_DIR = "quality"                    # reconstruction.quality.report.QUALITY_DIR (no import: cycles)


class ChainError(RuntimeError):
    """The epoch chain cannot be composed — an epoch without its persisted warp, a product epoch
    published after the base. Always with the exact reason; never an identity in its place."""


def base_epoch(output_dir, live: Optional[int] = None) -> int:
    """The reconstruction's PRODUCT epoch under ``live``: the most recent new-cloud epoch of the
    live lineage (0 when the lineage holds none). A certification starts from it and names its
    epoch relative to it (points 126, 136, 150)."""
    out = Path(output_dir)
    live = int(current_epoch(out) if live is None else live)
    for e in reversed(epoch_lineage(out, live)):
        if int(e) == 0 or epoch_kind(out, int(e)) == EPOCH_KIND_NEW_CLOUD:
            return int(e)
    return 0


def transform_epochs(output_dir, base: int, live: Optional[int] = None) -> List[int]:
    """The transform epochs of the live lineage AFTER ``base``, in order. A new-cloud epoch after
    ``base`` is another product: composing a warp across it is meaningless and RAISES."""
    out = Path(output_dir)
    live = int(current_epoch(out) if live is None else live)
    lineage = epoch_lineage(out, live)
    if int(base) not in lineage:
        raise ChainError(f"epoch {base} is not in the lineage {lineage} of the live epoch {live}")
    after = [int(e) for e in lineage[lineage.index(int(base)) + 1:]]
    for e in after:
        if epoch_kind(out, e) == EPOCH_KIND_NEW_CLOUD:
            raise ChainError(f"epoch {e} is a NEW CLOUD published after epoch {base} — the chain "
                             f"from {base} to {live} crosses another product")
    return after


def compose_moves(moves: Sequence[Tuple[dict, bool]], frames: Sequence[int],
                  log: Callable = print) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ONE per-keyframe transform equal to applying ``moves`` in order.

    Each move is ``(npz, inverse)``: the persisted warp of one epoch —
    ``p' = R·(c + (p − c)·(k z + b)/z) + t`` about the keyframe's own camera ``c`` and optical
    axis, ``z`` the depth along it — applied inverted while climbing to a common ancestor. Two
    such warps compose into one of the same form: ``R = R₂R₁``, ``t = R₂t₁ + t₂``, ``k = k₁k₂``,
    ``b = k₂b₁ + b₂`` (the camera moves rigidly with the pose, so the second depth op acts along
    the same ray at ``z₁ = k₁z + b₁``). With no move it is the identity — what a ``new_cloud``
    edge is worth. Keyed by real frame number; a frame a transform names that the session no
    longer has is declared and skipped."""
    n = len(frames)
    R = np.tile(np.eye(3), (n, 1, 1))
    t = np.zeros((n, 3))
    k = np.ones(n)
    b = np.zeros(n)
    idx = {int(f): i for i, f in enumerate(frames)}
    for mv, inverse in moves:
        Rm, tm, km = mv["R_kf"], mv["t_kf"], mv["k_kf"]
        bm = np.asarray(mv["b_kf"])
        if inverse:
            Rm = np.transpose(Rm, (0, 2, 1))
            tm = -np.einsum('nij,nj->ni', Rm, mv["t_kf"])
            km = 1.0 / mv["k_kf"]
            bm = -bm / mv["k_kf"]      # inverse of z' = k z + b is z = z'/k − b/k
        missing = 0
        for j, f in enumerate(mv["frames"]):
            i = idx.get(int(f))
            if i is None:
                missing += 1
                continue
            R[i] = Rm[j] @ R[i]
            t[i] = Rm[j] @ t[i] + tm[j]
            b[i] = km[j] * b[i] + bm[j]
            k[i] = k[i] * km[j]
        if missing:
            log(f"  {missing} keyframe(s) of a stored transform are not in the "
                f"session's frame list — their findings cannot follow")
    return R, t, k, b


def chain_stamp(output_dir, epochs: Sequence[int]) -> str:
    """The identity of a chain: sha256 over the (epoch, sha256 of its npz) rows — what a sidecar
    or a history derived from it carries, so a reader can tell which chain it came from."""
    from repro import sha256_file, sha256_json
    out = Path(output_dir)
    rows = []
    for e in epochs:
        p = out / EPOCH_NPZ_DIR / f"epoch_{int(e)}.npz"
        if not p.exists():
            raise ChainError(f"{p} does not exist — epoch {e} has no persisted warp")
        rows.append([int(e), sha256_file(p)])
    return sha256_json(rows)


def chain_transform(output_dir, frames: Sequence[int], base: Optional[int] = None,
                    live: Optional[int] = None, log: Callable = lambda m: None) -> Dict[str, Any]:
    """What the live epoch holds OVER the base: the composition of every transform epoch between
    them (``{R, t, k, b, base, live, epochs, stamp}``). An epoch without its npz RAISES."""
    out = Path(output_dir)
    live = int(current_epoch(out) if live is None else live)
    base = int(base_epoch(out, live) if base is None else base)
    epochs = transform_epochs(out, base, live)
    moves = [(load_epoch_npz(out, e), False) for e in epochs]
    R, t, k, b = compose_moves(moves, list(frames), log=log)
    return {"R": R, "t": t, "k": k, "b": b, "base": base, "live": live, "epochs": epochs,
            "stamp": chain_stamp(out, epochs)}


def depth_sidecar(output_dir, frames: Sequence[int], epoch_to: int,
                  k_new: Optional[np.ndarray] = None, b_new: Optional[np.ndarray] = None,
                  base: Optional[int] = None, live: Optional[int] = None) -> Dict[str, Any]:
    """``depth_correction.json`` of epoch ``epoch_to`` DERIVED from the chain (point 135): the
    per-keyframe affine ``z' = k·z + b`` the live geometry holds over the product epoch, composed
    with this epoch's own ``(k_new, b_new)`` — never read-modify-written from a sidecar on disk.
    Only the frames whose affine is not the identity are listed (the reader's convention,
    segmentation.session_io.load_depth_affine); the file is written even when none is, and it
    carries the base epoch, the chain and the chain's stamp."""
    tr = chain_transform(output_dir, frames, base=base, live=live)
    k, b = tr["k"].copy(), tr["b"].copy()
    if k_new is not None:
        kn = np.asarray(k_new, np.float64)
        bn = (np.asarray(b_new, np.float64) if b_new is not None else np.zeros(len(kn)))
        if kn.shape != k.shape:
            raise ChainError(f"depth_sidecar: {kn.shape[0]} new factors for {k.shape[0]} keyframes")
        b = kn * b + bn
        k = k * kn
    kb = {int(f): (float(k[i]), float(b[i])) for i, f in enumerate(frames)
          if k[i] != 1.0 or b[i] != 0.0}
    # version 2: the format every reader parses ({k, b} per frame) is unchanged — what is new
    # is where the numbers come from (the chain) and the keys that say so
    return {"version": 2, "epoch": int(epoch_to), "base_epoch": int(tr["base"]),
            "chain": [int(e) for e in tr["epochs"]] + [int(epoch_to)],
            "chain_stamp": tr["stamp"],
            # 6 decimals: the precision every reader of this file has always been handed
            "k": {str(f): round(v[0], 6) for f, v in sorted(kb.items())},
            "b": {str(f): round(v[1], 6) for f, v in sorted(kb.items())}}


def applied_depth_factor(output_dir, frames: Sequence[int], base: Optional[int] = None,
                         live: Optional[int] = None) -> np.ndarray:
    """The cumulative depth factor K per keyframe the live epoch holds over the base — the
    chain's ``k`` (point 147 reads |log K| against the anchors' own error)."""
    return np.asarray(chain_transform(output_dir, frames, base=base, live=live)["k"], np.float64)


def lineage_cleanup(output_dir, keep: Sequence[int], log: Callable = print) -> Dict[str, List[int]]:
    """Delete every stored epoch directory, every persisted warp and every per-epoch quality
    report of an epoch NOT in ``keep`` (the live lineage of the product epoch, point 136): the
    leftovers of a previous certification of this product or of a previous reconstruction. They
    are what made ``next_epoch`` count from the session's history instead of from the product.
    The ledger is history and stays. Returns what was deleted, per kind."""
    out = Path(output_dir)
    keep_set = {int(e) for e in keep}
    gone: Dict[str, List[int]] = {"dirs": [], "npz": [], "quality": []}
    for d in sorted(out.glob(f"{EPOCH_DIR_PREFIX}*")):
        if not d.is_dir() or d.is_symlink():
            continue
        try:
            e = int(d.name[len(EPOCH_DIR_PREFIX):])
        except ValueError:
            continue
        if e not in keep_set:
            shutil.rmtree(d)
            gone["dirs"].append(e)
    npz_dir = out / EPOCH_NPZ_DIR
    for q in sorted(npz_dir.glob("epoch_*.npz")) if npz_dir.is_dir() else []:
        try:
            e = int(q.stem.split("_")[-1])
        except ValueError:
            continue
        if e not in keep_set:
            q.unlink()
            gone["npz"].append(e)
    qdir = out / QUALITY_DIR
    for q in sorted(qdir.glob("report_epoch_*.json")) if qdir.is_dir() else []:
        try:
            e = int(q.stem.split("_")[-1])
        except ValueError:
            continue
        if e not in keep_set:
            q.unlink()
            gone["quality"].append(e)
    if any(gone.values()):
        log(f"  epochs outside the lineage {sorted(keep_set)} removed — stored dirs {gone['dirs']}, "
            f"warps {gone['npz']}, quality reports {gone['quality']} (the ledger keeps their history)")
    return gone


def product_epoch(output_dir) -> Tuple[int, Optional[dict]]:
    """(the reconstruction's PRODUCT epoch, its report): the epoch ``corrected_cloud.json`` (or
    ``fuse_report.json``) names — precision.product.product_report —, epoch 0 and None for a
    session the core published no cloud for (a legacy or synthetic session)."""
    from precision.product import product_report
    rep = product_report(Path(output_dir))
    if rep is None or rep.get("epoch_to") is None:
        return 0, None
    return int(rep["epoch_to"]), rep


def select_product_epoch(output_dir, log: Callable = print) -> Dict[str, Any]:
    """Make the reconstruction's PRODUCT epoch the live one (points 126 / 150) — what every
    stage that works ON the product reads first: the certification, and the mask projection
    an Autosegment runs before it (so the masks are projected on the cloud the certification
    starts from, never on a certified epoch). A no-op when it is already live. The product
    epoch no longer on disk RAISES :class:`ChainError` naming it. Returns {base_epoch,
    live_before, selected, product_file, product_correction_id}."""
    from correction.run import run_select
    out = Path(output_dir)
    base, rep = product_epoch(out)
    live = int(current_epoch(out))
    info: Dict[str, Any] = {"base_epoch": base, "live_before": live, "selected": False,
                            "product_file": (rep or {}).get("product_file"),
                            "product_correction_id": (rep or {}).get("correction_id")}
    if live != base:
        try:
            run_select(out, base, "product", log=log)
        except Exception as e:  # noqa: BLE001 — the reason travels
            raise ChainError(f"the reconstruction's product epoch {base} cannot be made live (the "
                             f"session shows epoch {live}): {e} — it is no longer on disk") from e
        info["selected"] = True
        log(f"  the session showed epoch {live} — the reconstruction's product epoch {base} is "
            f"live again (points 126 / 150: a stage on the product never reads a transformed epoch)")
    return info


def epoch_record_of(output_dir, epoch: int) -> Optional[dict]:
    """The geometry_epoch.json of ``epoch`` (live or stored), None when it has none."""
    out = Path(output_dir)
    from correction.epoch import EPOCH_FILE
    p = (out / EPOCH_FILE) if int(epoch) == current_epoch(out) else \
        (out / f"{EPOCH_DIR_PREFIX}{int(epoch)}" / EPOCH_FILE)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None
