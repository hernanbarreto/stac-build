"""Orchestration of the correction runs that survive the 2026-09-24 cut.

The MANUAL object correction (mark duplicated instances → evidence →
observability → diagnose → solve → gates → distribute → apply) and the manual
revisit closure were removed with the UI "Corrections" button (USER
2026-09-24): the session's correction is the visit-drift loop
(``correction.visit_drift_run``), which runs inside the certification stage of
the pipeline and calls ``run_floor`` after every epoch it publishes.

What stays here:
  * ``run_floor``  — floor alignment (kind=floor), transactional apply, ledger;
  * ``run_select`` — show the session in one of its epochs (nothing is approved
    or undone; every epoch stays on disk).

Every stage's structured output lands in the report, ALSO when the run is
rejected: the user always receives the numbers and the exact reason. A gate
failure returns a ``rejected`` report (and a ledger record) without touching
one byte of the session; only a fully-gated solution reaches the
transactional apply.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from correction import diagnose, floor as floor_mod, gates, ledger, solve
from correction.apply import (assert_no_interrupted_swap, available_epochs,
                              stage_transaction, swap_transaction)
from correction.config import CorrectionConfig, load_correction_config
from correction.epoch import current_epoch, epoch_path
from correction.invalidate import update_instance_store
from correction.report import build_report, save_report
from correction.session import load_session


def _noop_progress(pct: float, msg: str) -> None:
    pass


def _check_ready(output_dir: Path) -> None:
    """A half-finished swap is the only thing that blocks a new correction.

    It used to refuse while a previous epoch was "pending approval". There is
    no approval any more (USER 2026-09-16: *"todas viven, solo se seleccionan y
    la que se selecciona se muestra"*): every epoch stays on disk and a new
    correction simply runs on top of whichever one is being shown.
    """
    assert_no_interrupted_swap(output_dir)


def run_floor(output_dir, model: Optional[str], keyframes: Optional[List[int]],
              operator: str, log: Callable = print,
              progress: Callable = _noop_progress,
              cfg: Optional[CorrectionConfig] = None,
              pre: Optional[dict] = None) -> dict:
    """Floor alignment (kind=floor): same gates, same transactional apply,
    same ledger.

    ``pre`` is a per-keyframe transform ({R_kf, t_kf, k_kf}) to apply BEFORE
    the floor and publish COMPOSED with it, in ONE epoch (USER 2026-09-19:
    *"podría generarse una sola época que tenga la profundidad y el piso, es
    decir, la cero y la corregida, nada más"*).

    It is not a convenience: the floor has to be MEASURED on the geometry the
    depth correction already produced — measuring both on the raw cloud gives
    the wrong floor, the same way it gave the wrong translation. So ``pre`` is
    applied to an IN-MEMORY session, the floor is solved against that, and the
    two are composed exactly:

        warp(p; R, t, k) = R·(C + (p−C)k) + t        C → R·C + t

    so stage 1 then stage 2 is stage (R₂R₁, R₂·t₁ + t₂, k₁k₂). The floor's own
    k is 1 and the depth's R is the identity, but the composition is written in
    full because nothing here should depend on that staying true.

    The GATES still judge the floor's own solution, not the composed one: each
    stage is answerable for its own motion (the depth stage has its own
    `max_correction_log`), and a floor whose steps are fine should not be
    vetoed for a depth correction that already passed.
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    if cfg is None:
        cfg = load_correction_config()
    _check_ready(output_dir)
    correction_id = ledger.new_correction_id()
    epoch_from = current_epoch(output_dir)
    model = model or cfg.floor.model_default
    rng = np.random.default_rng(cfg.solve.seed)

    def _p(pct, msg):
        log(msg)
        progress(pct, msg)

    _p(2, "loading cloud + provenance...")
    session = load_session(output_dir)

    def _reject(reason, gates_list, sol=None):
        report = build_report(
            correction_id=correction_id, kind="floor", operator=operator,
            status="rejected", instance_ids=None, visits=None,
            observability=None, diagnosis=None,
            solutions=([{k: v for k, v in sol.items()
                         if k not in ("R_kf", "t_kf", "k_kf", "floor_npz")}]
                       if sol else None),
            distribution=None, gates=gates_list, overrides=None,
            epoch_from=epoch_from, epoch_to=None, rejection_reason=reason,
            elapsed_s=time.time() - t0)
        path = save_report(output_dir, report)
        ledger.record_run(
            output_dir, correction_id=correction_id, epoch_from=epoch_from,
            epoch_to=epoch_from, kind="floor", operator=operator,
            instance_ids=[], visits=[], observability=[],
            diagnosis=[{"model": model}], anchors=[], gates=gates_list,
            overrides=None,
            report_path=str(path.relative_to(output_dir)),
            verdict="rejected")
        _p(100, f"❌ floor alignment REJECTED: {reason}")
        return report

    g_int = gates.gate_integrity(session)
    if not g_int["passed"]:
        return _reject(g_int["detail"], [g_int])

    _p(15, f"solving floor alignment (model: {model})...")
    session_for_floor = session
    if pre is not None:
        from reconstruction.certify.run import transformed_session
        log("  a previous stage is composed into this epoch — solving the "
            "floor on the geometry it produces")
        session_for_floor = transformed_session(
            session, pre["R_kf"], pre["t_kf"], pre["k_kf"])
    sol = floor_mod.solve_floor(session_for_floor, cfg, model, keyframes, rng,
                                log=log)
    R_kf, t_kf, k_kf = sol["R_kf"], sol["t_kf"], sol["k_kf"]
    if pre is not None:
        R_pre = np.asarray(pre["R_kf"], np.float64)
        t_pre = np.asarray(pre["t_kf"], np.float64)
        k_pre = np.asarray(pre["k_kf"], np.float64)
        R_kf = np.einsum("nij,njk->nik", R_kf, R_pre)
        t_kf = np.einsum("nij,nj->ni", sol["R_kf"], t_pre) + t_kf
        k_kf = k_pre * k_kf
        log(f"  composed: depth {k_kf.min():.4f}-{k_kf.max():.4f}, "
            f"translation up to {np.linalg.norm(t_kf, axis=1).max() * 100:.1f} cm")

    anchors_as_solutions = [
        {"R": R_kf[a["kf"]], "t": t_kf[a["kf"]]} for a in sol["anchors"]]
    g_plaus = gates.gate_plausibility(anchors_as_solutions, cfg)
    steps_t = np.linalg.norm(np.diff(t_kf, axis=0), axis=1)
    steps_r = [solve.rot_deg(R_kf[i + 1] @ R_kf[i].T)
               for i in range(session.n_kf - 1)]
    dist_report = {
        "identity_until_kf": -1,
        "anchors": sol["anchors"],
        "keyframes_warped": session.n_kf,
        "max_step_between_keyframes_mm":
            round(float(steps_t.max()) * 1000, 2) if len(steps_t) else 0.0,
        "max_step_between_keyframes_deg":
            round(float(max(steps_r)), 4) if steps_r else 0.0,
        "depth_keyframes": 0,
    }
    g_cont = gates.gate_continuity(dist_report, cfg)
    worst_mm = sol["exam"]["worst_residual_mm"]
    g_exam = {"name": "floor_model_exam",
              "passed": worst_mm <= cfg.gates.heldout_floor_abs_m * 1000,
              "worst_residual_mm": worst_mm,
              "detail": f"worst anchor-floor residual vs the {model} model: "
                        f"{worst_mm:.1f} mm (limit "
                        f"{cfg.gates.heldout_floor_abs_m*1000:.0f} mm)"}
    gate_results = [g_int, g_plaus, g_cont, g_exam]
    failed = [g for g in gate_results if not g["passed"]]
    warnings: List[str] = []
    if failed:
        if cfg.gates.mode != "advisory":
            return _reject(f"gate(s) failed: {[g['name'] for g in failed]} — "
                           f"{failed[0]['detail']}", gate_results, sol)
        for g in failed:
            g["advisory"] = True
            warnings.append(f"{g['name']}: {g['detail']}")
        log(f"  ⚠ advisory gate(s) failed (applied anyway, USER 2026-09-09): "
            f"{[g['name'] for g in failed]}")

    # never "all gates passed" when one did not (see the floor path above)
    _p(50, ("staging the transaction — "
            + (f"{len(warnings)} advisory gate(s) failed and are declared "
               f"(gates.mode: advisory)" if warnings else "all gates passed")))
    tx_info = stage_transaction(
        session, cfg, R_kf, t_kf, k_kf, correction_id=correction_id,
        scale_diag_new=diagnose.regenerate_scale_diagnostics(
            output_dir, {}, epoch_from + 1, correction_id),
        floor_npz=sol["floor_npz"], log=log, progress=progress)
    _p(90, "atomic swap...")
    swap_transaction(output_dir, tx_info, log=log)
    _p(93, "updating the instance store in place...")
    store_summary = update_instance_store(
        output_dir, R_kf, t_kf, k_kf, session.frames, log=log)

    report = build_report(
        correction_id=correction_id, kind="floor", operator=operator,
        status="applied", instance_ids=None, visits=None,
        observability=None,
        diagnosis=[{"model": model, "model_params": sol["model_params"]}],
        solutions=[{"anchors": sol["anchors"],
                    "per_kf_report": sol["per_kf_report"],
                    "exam": sol["exam"]}],
        distribution=dist_report, gates=gate_results, overrides=None,
        epoch_from=epoch_from, epoch_to=tx_info["epoch_to"],
        extra={"points_moved": tx_info["points_moved"],
               "pose_copies_skipped": tx_info["pose_copies_skipped"],
               "instance_store": store_summary,
               "warnings": warnings},
        elapsed_s=time.time() - t0)
    path = save_report(output_dir, report)
    ledger.record_run(
        output_dir, correction_id=correction_id, epoch_from=epoch_from,
        epoch_to=tx_info["epoch_to"], kind="floor", operator=operator,
        instance_ids=[], visits=[], observability=[],
        diagnosis=[{"model": model, "model_params": sol["model_params"]}],
        anchors=sol["anchors"], gates=gate_results, overrides=None,
        report_path=str(path.relative_to(output_dir)))
    _p(100, f"✅ floor alignment applied (epoch {tx_info['epoch_to']}, "
            f"model {model}) — select any epoch to compare")
    return report


def run_select(output_dir, epoch: int, operator: str = "user",
               log: Callable = print) -> dict:
    """Show the session in one of its epochs. Nothing is approved or undone.

    USER 2026-09-16: *"todas viven, solo se seleccionan y la que se selecciona
    se muestra"*. Approve used to delete every previous epoch and Undo the
    current one, so a session could only hold two states and choosing wrong
    destroyed the other. Every epoch now stays on disk and this only decides
    which one is on screen.

    The geometry is swapped by `select_epoch`; the instance store has to follow
    it, which means composing the transforms of the epochs BETWEEN the two —
    inverted while walking UP to their common ancestor, forward while walking
    DOWN to the chosen one. Ancestry, not arithmetic: a correction run on top
    of an older epoch branches the history, so cur and epoch are not always on
    the same line (`epoch_path`).
    """
    from correction.apply import select_epoch
    output_dir = Path(output_dir)
    epoch = int(epoch)
    cur = current_epoch(output_dir)
    if epoch == cur:
        return {"ok": True, "epoch": cur, "changed": False,
                "available": [e["epoch"] for e in available_epochs(output_dir)]}

    # (epoch, inverse) for every edge to travel — each transform is stored and
    # exact, so the store lands on the geometry, never near it
    moves = []
    for e, inverse in epoch_path(output_dir, cur, epoch):
        try:
            moves.append((ledger.load_epoch_npz(output_dir, e), inverse))
        except RuntimeError as err:
            raise RuntimeError(
                f"epoch {e} has no persisted transform, so the instance store "
                f"cannot follow the geometry to epoch {epoch}: {err}")

    res = select_epoch(output_dir, epoch, log=log)

    # The record has to name the epoch actually on screen. It travels with the
    # geometry whenever the epoch that wrote it listed it as an artifact; an
    # epoch published before that was the rule leaves the previous record live,
    # and then `current_epoch` lies — the session showed epoch 0 while the file
    # still said 3, and the next correction numbered itself from the lie
    # (pccr 2026-09-18). Repairing it here costs the correction_id of those old
    # epochs and nothing else.
    if current_epoch(output_dir) != epoch:
        from correction.epoch import make_epoch_record, EPOCH_FILE
        (output_dir / EPOCH_FILE).write_text(json.dumps(make_epoch_record(
            epoch, f"select/epoch_{epoch}", max(epoch - 1, 0)), indent=2))
        log(f"  epoch record did not travel with the geometry — rewritten to "
            f"epoch {epoch}")

    for mv, inverse in moves:
        R, t, k, b, frames = (mv["R_kf"], mv["t_kf"], mv["k_kf"],
                              np.asarray(mv["b_kf"]), mv["frames"])
        if inverse:
            R = np.transpose(R, (0, 2, 1))
            t = -np.einsum('nij,nj->ni', R, mv["t_kf"])
            k = 1.0 / mv["k_kf"]
            b = -b / mv["k_kf"]        # inverse of z' = k z + b is z = z'/k − b/k
        try:
            update_instance_store(output_dir, R, t, k, frames, log=log, b_kf=b)
        except RuntimeError as e:
            log(f"  instance-store refresh failed (the geometry IS at epoch "
                f"{epoch}; the store stays stale until the next rebuild): {e}")
            break
    res["ok"] = True
    res["available"] = [e["epoch"] for e in available_epochs(output_dir)]
    log(f"[correction] session shown at epoch {epoch} "
        f"(available: {res['available']})")
    return res
