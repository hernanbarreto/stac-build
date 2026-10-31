"""The VGGT-SLAM 2.0 variants of the battery — claude_stac_slam2.txt, phase 4.

Four chain variants beside ``reference`` (``evaluation.variants``): slam2_focal_common (the common
K, session median), slam2_attention_gate (the attention gate, the SALAD bar as configured),
slam2_attention_gate_salad (the gate with the SALAD bar at the configured number) and slam2_all.
Every one is a full reconstruction from Ω on its own copy of the session (the fork's stamp changes
with the code and the config: nothing is reused).

What this module adds around the executor (evaluation.variants):
- the RESOLUTION of a slam2 variant before it runs: the gate variants take their layer and
  threshold from the scene's attention_calibration.json (the reference run's, this run's or the
  scene's own) — absent or without a choice, the variant is recorded as not run with the reason;
  slam2_focal_common is recorded as not run when ``evaluation.slam2.skip_if_offline_identity`` is on
  and the phase-1 instrument (seam_radial.json of the same run) says the common K does not pass the
  rule — it would be the reference;
- the OFFLINE instruments after every finished chain variant that kept its unaligned chunks:
  loop_utils.seam_radial (CPU) and, after the reference, loop_utils.attention_calibrate (GPU, one
  two-frame pass per labelled pair) — their records feed the resolution and the table;
- the BIT-FOR-BIT comparator of this branch's reference against an existing run of the same scene
  made with main (the published cloud by provenance, the chunk npys when both sides kept them, the
  camera files as text, the fork's json records without their stamps) — equality, or the first
  difference by name;
- the TABLE per scene (slam2_comparison.csv / .json / .png): seam residuals per seam and radius
  bin, the focal model's verdict, SALAD candidates / gate passed / bridges / cycle-consistent
  closures / judges and the keyframe graph's verdict, the fork's held-out pairs and F5's held-out,
  the battery's phases 1–2 where references exist, coverage and cost — each with its difference
  against the reference, and no verdict;
- the variant's cloud opens in the viewer under the variant's name: its session is published as
  a scan of the project ``<scene><suffix>`` (a symlink; the viewer reads the kept products).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

SLAM2_GATE_VARIANTS = ("slam2_attention_gate", "slam2_attention_gate_salad", "slam2_all")
SLAM2_FOCAL_VARIANTS = ("slam2_focal_common", "slam2_all")
GATE_KEY = "loops.attention_gate"
SEAMS_KEY = "correction_graph.seams"
TABLE_NAME = "slam2_comparison"
STAMP_KEYS = ("fork_stamp", "_stac_fork_stamp", "stamp", "stac-stamp")


class Slam2Error(RuntimeError):
    pass


def slam2_config(raw_eval: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``evaluation.slam2`` validated (every key mandatory when the section exists; absent → None)."""
    sec = raw_eval.get("slam2")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        raise Slam2Error("'evaluation.slam2' must be a mapping")
    out = {}
    for key, typ in (("skip_if_offline_identity", bool), ("references_optional", bool), ("publish_to_viewer", bool),
                     ("viewer_project_suffix", str), ("calibration_file", str), ("seam_radial_file", str),
                     ("existing_runs", dict)):
        if key not in sec:
            raise Slam2Error(f"'evaluation.slam2.{key}' is missing")
        if not isinstance(sec[key], typ) or (typ is bool and not isinstance(sec[key], bool)):
            raise Slam2Error(f"'evaluation.slam2.{key}' must be a {typ.__name__}")
        out[key] = sec[key]
    if "mirror_instruments_to_scene" not in sec or not isinstance(sec["mirror_instruments_to_scene"], bool):
        raise Slam2Error("'evaluation.slam2.mirror_instruments_to_scene' is missing or not true/false")
    out["mirror_instruments_to_scene"] = sec["mirror_instruments_to_scene"]
    if "projects_dir" not in sec:
        raise Slam2Error("'evaluation.slam2.projects_dir' is missing (null = the server's projects directory)")
    pd = sec["projects_dir"]
    if pd is not None and not isinstance(pd, str):
        raise Slam2Error("'evaluation.slam2.projects_dir' must be null or a path")
    out["projects_dir"] = pd
    return out


# ── records of a run directory ────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(Path(path).read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def run_dir_of(session_dir: Path) -> Path:
    return Path(session_dir) / "output" / "maplong_run"


def calibration_choice(run_dir: Path, file_rel: str) -> Dict[str, Any]:
    """{'available': bool, 'layer', 'threshold', 'reason', 'path', 'exists'} from attention_calibration.json."""
    p = Path(run_dir) / file_rel
    doc = _read_json(p) if p.is_file() else None
    if doc is None:
        return {"available": False, "exists": False, "path": str(p),
                "reason": f"{p} does not exist (python -m loop_utils.attention_calibrate did not run on this scene)"}
    ch = doc.get("choice") or {}
    if not ch.get("chosen"):
        return {"available": False, "exists": True, "path": str(p),
                "reason": f"{p}: no layer / threshold chosen — {ch.get('reason')}"}
    return {"available": True, "layer": int(ch["layer"]), "threshold": float(ch["threshold"]), "path": str(p),
            "precision": ch.get("precision"), "recall": ch.get("recall"), "n_pos": ch.get("n_pos"), "n_neg": ch.get("n_neg")}


def offline_common_k_verdict(run_dir: Path, file_rel: str) -> Dict[str, Any]:
    """The phase-1 instrument's pooled session rule for the common K (session median)."""
    p = Path(run_dir) / file_rel
    doc = _read_json(p) if p.is_file() else None
    if doc is None:
        return {"available": False, "exists": False, "path": str(p),
                "reason": f"{p} does not exist (python -m loop_utils.seam_radial did not run)"}
    rule = (doc.get("session_rule") or {}).get("common_K_session_median") or {}
    if "improves_by_rule" not in rule:
        return {"available": False, "exists": True, "path": str(p),
                "reason": f"{p}: the common_K counterfactual was not judged — {rule.get('reason')}"}
    return {"available": True, "improves_by_rule": bool(rule["improves_by_rule"]), "reason": rule.get("reason"),
            "median_before_m": rule.get("median_before_m"), "median_after_m": rule.get("median_after_m"), "path": str(p)}


def _most_telling(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Among the instrument records none of which is usable, the one whose file EXISTS (it says why
    nothing was chosen) before one whose file is missing."""
    return next((r for r in records if r.get("exists")), records[-1])


def resolve_variant(spec, *, scene_session: Path, results: Dict[str, Dict[str, Any]], slam2: Dict[str, Any],
                    reference: str = "reference", log: Callable = print):
    """The spec a slam2 variant runs with: the gate's layer / min_score filled from the scene's
    calibration, or the variant turned into ``not_executed`` with the reason (the executor records
    it). Other variants pass through untouched."""
    from evaluation.variants import VariantSpec
    ov = dict(spec.overrides or {})
    is_gate = GATE_KEY in ov and isinstance(ov[GATE_KEY], dict) and ov[GATE_KEY].get("mode") == "gate"
    is_focal = SEAMS_KEY in ov and isinstance(ov[SEAMS_KEY], dict) and ov[SEAMS_KEY].get("focal_model") == "common_K"
    if spec.kind != "chain" or not (is_gate or is_focal):
        return spec
    # where the scene's instruments were run: the reference variant of this run first (its own
    # Ω pass of this code), then the scene's own session
    candidates = []
    ref = results.get(reference) or {}
    if ref.get("status") == "finished" and ref.get("session_dir"):
        candidates.append(run_dir_of(Path(ref["session_dir"])))
    candidates.append(run_dir_of(Path(scene_session)))
    detail: Dict[str, Any] = {"instrument_dirs": [str(c) for c in candidates]}
    if is_focal and slam2["skip_if_offline_identity"]:
        verdicts = [offline_common_k_verdict(c, slam2["seam_radial_file"]) for c in candidates]
        found = next((v for v in verdicts if v["available"]), None)
        detail["offline_common_K"] = found or _most_telling(verdicts)
        if found is not None and not found["improves_by_rule"]:
            return VariantSpec(spec.name, "not_executed", spec.description,
                               reason=f"evaluation.slam2.skip_if_offline_identity: the phase-1 instrument says the common K "
                                      f"does not pass the rule on this scene ({found['path']}: {found['reason']}) — the "
                                      f"variant would be the reference", enabled=spec.enabled)
        if found is None:
            log(f"[slam2] {spec.name}: no offline common-K verdict — {_most_telling(verdicts)['reason']}; the variant runs")
    if is_gate:
        gate = dict(ov[GATE_KEY])
        if gate.get("layer") is None or gate.get("min_score") is None:
            choices = [calibration_choice(c, slam2["calibration_file"]) for c in candidates]
            found = next((c for c in choices if c["available"]), None)
            detail["calibration"] = found or _most_telling(choices)
            if found is None:
                return VariantSpec(spec.name, "not_executed", spec.description,
                                   reason=f"the attention gate has no layer / threshold for this scene: "
                                          f"{_most_telling(choices)['reason']}", enabled=spec.enabled)
            gate["layer"] = int(found["layer"])
            gate["min_score"] = float(found["threshold"])
            ov[GATE_KEY] = gate
    out = VariantSpec(spec.name, spec.kind, spec.description, overrides=ov, stages_off=list(spec.stages_off),
                      base=spec.base, colmap=dict(spec.colmap), reason=spec.reason, enabled=spec.enabled)
    out.resolution = detail                                   # type: ignore[attr-defined]
    return out


# ── the offline instruments after a variant ───────────────────────────────────────────────

def _fork_python(config: Dict[str, Any]) -> str:
    r = ((config.get("reconstruction") or {}).get("precision") or {}).get("runner") or {}
    return str(r.get("python_mapanything") or sys.executable)


def _fork_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"


def run_seam_radial(session_dir: Path, config: Dict[str, Any], log: Callable, timeout_s: int = 7200) -> Dict[str, Any]:
    """python -m loop_utils.seam_radial on the variant's run (CPU); the record of what happened."""
    run = run_dir_of(session_dir)
    if not (run / "_tmp_results_unaligned").is_dir() or not any((run / "_tmp_results_unaligned").glob("chunk_*.npy")):
        return {"ran": False, "reason": "the run kept no unaligned chunk npys (certify.keep_unaligned_chunks off, or the "
                                        "fork removed them)"}
    cmd = [_fork_python(config), "-m", "loop_utils.seam_radial", "--run", str(run)]
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONHASHSEED="0")
    env.pop("PYTHONPATH", None)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(_fork_dir()), env=env, capture_output=True, text=True, timeout=timeout_s)
    rec = {"ran": True, "returncode": r.returncode, "seconds": round(time.time() - t0, 1), "cmd": cmd,
           "output": str(run / "seam_radial" / "seam_radial.json")}
    if r.returncode != 0:
        rec["error"] = (r.stderr or r.stdout)[-2000:]
        log(f"[slam2] seam_radial failed ({r.returncode}): {rec['error'][-300:]}")
    return rec


def run_attention_calibrate(session_dir: Path, config: Dict[str, Any], params: Dict[str, Any], log: Callable,
                            timeout_s: int = 14400) -> Dict[str, Any]:
    """python -m loop_utils.attention_calibrate on the variant's run (GPU; the card is free between
    variants); ``params`` = the calibration parameters when the run carries no gate section."""
    run = run_dir_of(session_dir)
    out = run / "attention_calibration"
    pfile = out / "params.json"
    out.mkdir(parents=True, exist_ok=True)
    pfile.write_text(json.dumps(params, indent=1))
    cmd = [_fork_python(config), "-m", "loop_utils.attention_calibrate", "--run", str(run), "--params", str(pfile),
           "--out", str(out)]
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONHASHSEED="0",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env.pop("PYTHONPATH", None)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(_fork_dir()), env=env, capture_output=True, text=True, timeout=timeout_s)
    rec = {"ran": True, "returncode": r.returncode, "seconds": round(time.time() - t0, 1), "cmd": cmd,
           "output": str(out / "attention_calibration.json")}
    if r.returncode != 0:
        rec["error"] = (r.stderr or r.stdout)[-2000:]
        log(f"[slam2] attention_calibrate failed ({r.returncode}): {rec['error'][-300:]}")
    return rec


def gate_params_from_overrides(specs: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """The calibration parameters the gate variants declare (their loops.attention_gate override):
    the calibration block + top_fraction — what attention_calibrate needs on a run without the section."""
    for s in specs:
        ov = getattr(s, "overrides", None) or {}
        g = ov.get(GATE_KEY)
        if isinstance(g, dict) and isinstance(g.get("calibration"), dict):
            return dict(g["calibration"], top_fraction=g.get("top_fraction"))
    return None


# ── the bit-for-bit comparator ────────────────────────────────────────────────────────────

def _strip(doc):
    if isinstance(doc, dict):
        return {k: _strip(v) for k, v in doc.items() if k not in STAMP_KEYS}
    if isinstance(doc, list):
        return [_strip(v) for v in doc]
    return doc


def _first_diff(path: str, a, b) -> Optional[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            return f"{path}: keys differ {sorted(ka ^ kb)[:6]}"
        for k in sorted(ka):
            d = _first_diff(f"{path}.{k}", a[k], b[k])
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: lengths {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(f"{path}[{i}]", x, y)
            if d:
                return d
        return None
    if isinstance(a, float) and isinstance(b, float) and (a != a) and (b != b):
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def _compare_npy_dir(da: Path, db: Path) -> Dict[str, Any]:
    fa = sorted(p.name for p in da.glob("chunk_*.npy")) if da.is_dir() else []
    fb = sorted(p.name for p in db.glob("chunk_*.npy")) if db.is_dir() else []
    if not fa or not fb:
        return {"compared": False, "reason": f"{'both sides' if not fa and not fb else ('side A' if not fa else 'side B')} "
                                             f"kept no chunk npys ({da.name})"}
    if fa != fb:
        return {"compared": True, "equal": False, "first_difference": f"{da.name}: files {fa} vs {fb}"}
    for name in fa:
        a = np.load(da / name, allow_pickle=True)
        b = np.load(db / name, allow_pickle=True)
        a = a.item() if a.dtype == object and a.shape == () else a
        b = b.item() if b.dtype == object and b.shape == () else b
        if isinstance(a, dict) and isinstance(b, dict):
            ka = set(a) - set(STAMP_KEYS)
            kb = set(b) - set(STAMP_KEYS)
            if ka != kb:
                return {"compared": True, "equal": False, "first_difference": f"{da.name}/{name}: keys {sorted(ka ^ kb)}"}
            for k in sorted(ka):
                x, y = a[k], b[k]
                if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                    x, y = np.asarray(x), np.asarray(y)
                    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                        return {"compared": True, "equal": False,
                                "first_difference": f"{da.name}/{name}[{k}]: arrays differ ({x.dtype}{x.shape} vs {y.dtype}{y.shape})"}
                elif x != y:
                    return {"compared": True, "equal": False, "first_difference": f"{da.name}/{name}[{k}]: {x!r} != {y!r}"}
        else:
            if np.asarray(a).tobytes() != np.asarray(b).tobytes():
                return {"compared": True, "equal": False, "first_difference": f"{da.name}/{name}: arrays differ"}
    return {"compared": True, "equal": True, "n_files": len(fa)}


def bitwise_compare(session_a: Path, session_b: Path, log: Callable = print) -> Dict[str, Any]:
    """This branch's reference (A) against an existing run of the same scene (B): the published
    cloud by provenance, the chunk npys both sides kept, the camera files as text, the fork's json
    records without their stamps. Returns equality per product and the first difference by name."""
    from evaluation.determinism import compare_products
    from evaluation.product import StacProduct
    A, B = Path(session_a), Path(session_b)
    out: Dict[str, Any] = {"session_a": str(A), "session_b": str(B), "products": {}}
    try:
        pa = StacProduct(A, "A", log=lambda *a: None)
        pb = StacProduct(B, "B", log=lambda *a: None)
        cp = compare_products(pa, pb)
        out["products"]["published_cloud"] = {"compared": True, "equal": bool(cp.get("bit_identical_cloud")),
                                              "detail": {k: v for k, v in cp.items() if k != "bit_identical_cloud"}}
        if not cp.get("bit_identical_cloud"):
            d = cp.get("position_diff_m") or {}
            out["products"]["published_cloud"]["first_difference"] = (
                f"cleaned_cloud.ply: {cp.get('n_paired')} points paired by provenance, max position difference "
                f"{d.get('max')} m, {cp.get('n_only_a')} only in A, {cp.get('n_only_b')} only in B")
    except Exception as e:  # noqa: BLE001 — declared
        out["products"]["published_cloud"] = {"compared": False, "reason": f"{type(e).__name__}: {e}"}
    ra, rb = run_dir_of(A), run_dir_of(B)
    for sub in ("_tmp_results_unaligned", "_tmp_results_aligned"):
        out["products"][sub] = _compare_npy_dir(ra / sub, rb / sub)
    for name in ("camera_poses.txt", "intrinsic.txt"):
        pa_, pb_ = ra / name, rb / name
        if pa_.is_file() and pb_.is_file():
            eq = pa_.read_bytes() == pb_.read_bytes()
            rec = {"compared": True, "equal": eq}
            if not eq:
                la, lb = pa_.read_text().splitlines(), pb_.read_text().splitlines()
                k = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
                rec["first_difference"] = f"{name}: line {k + 1} differs"
            out["products"][name] = rec
        else:
            out["products"][name] = {"compared": False, "reason": f"{name} missing on " + ("A" if not pa_.is_file() else "B")}
    for name in ("metric_lock.json", "scale_graph.json", "chunk_sim3.json", "loop_edges.json", "pose_graph.json",
                 "chunk_health.json"):
        pa_, pb_ = ra / name, rb / name
        if pa_.is_file() and pb_.is_file():
            da, db = _strip(_read_json(pa_) or {}), _strip(_read_json(pb_) or {})
            d = _first_diff(name, da, db)
            out["products"][name] = {"compared": True, "equal": d is None, **({"first_difference": d} if d else {})}
        else:
            out["products"][name] = {"compared": False, "reason": f"{name} missing on " + ("A" if not pa_.is_file() else "B")}
    compared = [v for v in out["products"].values() if v.get("compared")]
    out["equal_everywhere_compared"] = bool(compared) and all(v.get("equal") for v in compared)
    out["first_difference"] = next((v["first_difference"] for v in out["products"].values()
                                    if v.get("compared") and not v.get("equal")), None)
    out["not_compared"] = {k: v.get("reason") for k, v in out["products"].items() if not v.get("compared")}
    return out


# ── the table ─────────────────────────────────────────────────────────────────────────────

def _count(path: Path, pred) -> Optional[int]:
    doc = _read_json(path) if path.is_file() else None
    if doc is None:
        return None
    try:
        return int(pred(doc))
    except Exception:  # noqa: BLE001
        return None


def _salad_candidates(run: Path) -> Optional[int]:
    p = run / "loop_closures.txt"
    if not p.is_file():
        return None
    n = 0
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if s and not s.startswith("#") and "," in s:
            n += 1
    return n


def variant_metrics(session_dir: Optional[Path]) -> Dict[str, Any]:
    """The slam2 numbers of one finished variant, read from its records (None where a record is
    absent — never a default)."""
    m: Dict[str, Any] = {}
    if not session_dir:
        return m
    sd = Path(session_dir)
    run = run_dir_of(sd)
    out = sd / "output"
    m["salad_candidates"] = _salad_candidates(run)
    ag = _read_json(run / "attention_gate.json") if (run / "attention_gate.json").is_file() else None
    m["gate_candidates"] = None if ag is None else ag.get("n_candidates")
    m["gate_passed"] = None if ag is None else ag.get("n_passed")
    m["gate_layer"] = None if ag is None else ag.get("layer")
    m["gate_min_score"] = None if ag is None else ag.get("min_score")
    le = _read_json(run / "loop_edges.json") if (run / "loop_edges.json").is_file() else None
    if le is not None:
        ver = [e for e in le.get("edges") or [] if e.get("stage") == "verification"]
        m["bridges"] = len(ver)
        m["bridges_accepted"] = sum(1 for e in ver if e.get("status") == "accepted")
    pg = _read_json(run / "pose_graph.json") if (run / "pose_graph.json").is_file() else None
    if pg is not None:
        cyc = pg.get("cycle_consistency") or {}
        m["closures_cycle_consistent"] = cyc.get("n_kept")
        loo = pg.get("leave_one_out_closures") or {}
        m["graph_judges"] = loo.get("n_judges")
        m["graph_verdict"] = pg.get("verdict")
        m["graph_reason"] = pg.get("reason") or loo.get("reason")
        # the fork's held-out: the leave-one-out closures (every record), the surface pairs when the
        # graph was solved (an APPLY record's gates)
        m["fork_heldout_pairs"] = loo.get("n_judges")
        m["fork_heldout_before_m"] = loo.get("median_before_m")
        m["fork_heldout_after_m"] = loo.get("median_after_m")
        m["fork_heldout_error_m"] = loo.get("error_m")
        hp = (pg.get("gates") or {}).get("holdout_surface_pairs") or {}
        if hp:
            m["fork_surface_pairs"] = hp.get("n_pairs")
            m["fork_surface_pairs_before_m"] = hp.get("before_m", hp.get("median_before_m"))
            m["fork_surface_pairs_after_m"] = hp.get("after_m", hp.get("median_after_m"))
    fm = _read_json(run / "focal_model.json") if (run / "focal_model.json").is_file() else None
    if fm is not None:
        m["focal_model_applied"] = fm.get("applied")
        m["focal_model_reason"] = fm.get("reason")
        sr = fm.get("seams_rule") or {}
        m["focal_seams_before_m"] = sr.get("median_before_m")
        m["focal_seams_after_m"] = sr.get("median_after_m")
    srd = _read_json(run / "seam_radial" / "seam_radial.json") if (run / "seam_radial" / "seam_radial.json").is_file() else None
    if srd is not None:
        seams = []
        for s in srd.get("seams") or []:
            if s.get("starved"):
                seams.append({"seam": s.get("seam"), "starved": True})
                continue
            b = s["states"]["before"]
            seams.append({"seam": s["seam"], "held_norm_median_m": b["held_norm_median_m"],
                          "lateral_median_m": b["held_lateral_median_m"], "axial_abs_median_m": b["held_axial_abs_median_m"],
                          "radius_bins": [{"lo": r["lo"], "hi": r["hi"], "n": r["n"], "lateral_median_m": r["lateral_median_m"],
                                           "axial_abs_median_m": r["axial_abs_median_m"]} for r in b["radius_bins"]],
                          "intrinsics_offset_px": s["intrinsics"]["offset_median_px"]})
        m["seams"] = seams
        vals = [s["held_norm_median_m"] for s in seams if not s.get("starved")]
        m["seam_held_median_m_mean"] = float(np.mean(vals)) if vals else None
        m["offline_common_K_improves"] = ((srd.get("session_rule") or {}).get("common_K_session_median") or {}).get("improves_by_rule")
    rf = _read_json(out / "precision" / "refine.json") if (out / "precision" / "refine.json").is_file() else None
    if rf is not None:
        rungs = rf.get("rungs") or {}
        rung = rf.get("applied_rung")
        rec_r = rungs.get(rung) if isinstance(rung, str) else None
        if not isinstance(rec_r, dict):
            taken = [k for k, v in rungs.items() if isinstance(v, dict) and v.get("taken")]
            rung = taken[-1] if taken else None
            rec_r = rungs.get(rung) if rung else None
        m["f5_rung"] = rung
        m["f5_heldout_px"] = (rec_r or {}).get("heldout_median_px")
        m["f5_heldout_tracks"] = (rec_r or {}).get("n_heldout_tracks")
        m["f5_fit_rms_px"] = (rec_r or {}).get("fit_rms_px")
    return m


TABLE_COLUMNS = ["scene", "variant", "status", "reason", "salad_candidates", "gate_candidates", "gate_passed", "gate_layer",
                 "gate_min_score", "bridges", "bridges_accepted", "closures_cycle_consistent", "graph_judges", "graph_verdict",
                 "fork_heldout_pairs", "fork_heldout_before_m", "fork_heldout_after_m", "fork_heldout_error_m",
                 "fork_surface_pairs", "fork_surface_pairs_before_m", "fork_surface_pairs_after_m", "focal_model_applied",
                 "focal_seams_before_m", "focal_seams_after_m", "seam_held_median_m_mean", "offline_common_K_improves",
                 "f5_heldout_px", "f5_heldout_tracks", "f5_fit_rms_px", "f5_rung", "abs_median_m", "abs_p95_m",
                 "keyframes_registered_pct", "seconds", "gpu_hours",
                 "cost_usd", "d_seam_held_median_m_mean_vs_reference", "d_fork_heldout_after_m_vs_reference",
                 "d_f5_heldout_px_vs_reference", "d_seconds_vs_reference"]


def slam2_rows(scene: str, results: Dict[str, Dict[str, Any]], reference: str) -> List[Dict[str, Any]]:
    metrics = {name: (r.get("slam2") or {}).get("metrics") or {} for name, r in results.items()}
    ref = metrics.get(reference) or {}
    ref_r = results.get(reference) or {}
    rows = []
    for name, r in results.items():
        m = metrics[name]
        s = r.get("summary") or {}
        a = s.get("abs_error_m") or {}
        cov = r.get("coverage") or {}
        cost = r.get("cost") or {}

        def _d(key, mine=m, theirs=ref):
            x, y = mine.get(key), theirs.get(key)
            return (x - y) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
        row = {"scene": scene, "variant": name, "status": r.get("status"), "reason": r.get("reason", ""),
               **{k: m.get(k) for k in TABLE_COLUMNS if k in m},
               "abs_median_m": a.get("median"), "abs_p95_m": a.get("p95"),
               "keyframes_registered_pct": cov.get("keyframes_registered_pct"),
               "seconds": r.get("seconds"), "gpu_hours": cost.get("gpu_hours"), "cost_usd": cost.get("cost_usd"),
               "d_seam_held_median_m_mean_vs_reference": _d("seam_held_median_m_mean"),
               "d_fork_heldout_after_m_vs_reference": _d("fork_heldout_after_m"),
               "d_f5_heldout_px_vs_reference": _d("f5_heldout_px"),
               "d_seconds_vs_reference": ((r.get("seconds") - ref_r.get("seconds"))
                                          if isinstance(r.get("seconds"), (int, float)) and isinstance(ref_r.get("seconds"), (int, float))
                                          else None)}
        for k in TABLE_COLUMNS:
            row.setdefault(k, None)
        rows.append(row)
    return rows


def write_table(scene_dir: Path, scene: str, results: Dict[str, Dict[str, Any]], reference: str, *,
                bitwise: Optional[Dict[str, Any]] = None, png: bool = True) -> Dict[str, str]:
    from evaluation.dimension_error import write_csv, write_json
    rows = slam2_rows(scene, results, reference)
    outs = {"csv": str(write_csv(scene_dir / f"{TABLE_NAME}.csv", rows, TABLE_COLUMNS)),
            "json": str(write_json(scene_dir / f"{TABLE_NAME}.json",
                                   {"scene": scene, "reference": reference, "rows": rows, "bitwise_comparator": bitwise,
                                    "seams_by_variant": {n: ((r.get("slam2") or {}).get("metrics") or {}).get("seams")
                                                         for n, r in results.items()},
                                    "note": "measurements and their differences against the reference — no verdict"}))}
    if png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            for n, r in results.items():
                seams = ((r.get("slam2") or {}).get("metrics") or {}).get("seams") or []
                for s in seams:
                    if s.get("starved"):
                        continue
                    rb = [b for b in s["radius_bins"] if b["n"]]
                    axes[0].plot([0.5 * (b["lo"] + b["hi"]) for b in rb], [100 * b["lateral_median_m"] for b in rb],
                                 marker=".", label=f"{n} seam {s['seam']}")
            axes[0].set_title(f"{scene}: seam held-out residual, lateral (median, cm) by radius")
            axes[0].set_xlabel("normalised radius"); axes[0].set_ylabel("cm"); axes[0].grid(True, alpha=0.3)
            if axes[0].lines:
                axes[0].legend(fontsize=6)
            names = [n for n in results if ((results[n].get("slam2") or {}).get("metrics") or {})]
            keys = ["fork_heldout_after_m", "f5_heldout_px", "seam_held_median_m_mean"]
            x = np.arange(len(names))
            for i, key in enumerate(keys):
                vals = [((results[n].get("slam2") or {}).get("metrics") or {}).get(key) for n in names]
                axes[1].bar(x + 0.25 * i, [v if isinstance(v, (int, float)) else 0 for v in vals], width=0.25, label=key)
            axes[1].set_xticks(x + 0.25); axes[1].set_xticklabels(names, rotation=20, fontsize=7)
            axes[1].set_title("held-out evidence per variant (units differ: m, px, m)"); axes[1].legend(fontsize=7)
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            pp = scene_dir / f"{TABLE_NAME}.png"
            fig.savefig(pp, dpi=120)
            plt.close(fig)
            outs["png"] = str(pp)
        except Exception as e:  # noqa: BLE001
            outs["png_error"] = str(e)
    return outs


# ── the viewer ────────────────────────────────────────────────────────────────────────────

def publish_variant_scan(projects_dir: Path, scene: str, variant: str, session_dir: Path, suffix: str,
                         log: Callable = print) -> Dict[str, Any]:
    """The variant's session as a scan ``<variant>/default`` of the project ``<scene><suffix>``: a
    symlink, so the viewer opens the kept products (cleaned_cloud.ply, potree/) under the variant's
    name. Idempotent: an existing link is replaced."""
    slug = f"{scene}{suffix}"
    proj = Path(projects_dir) / slug
    (proj / "scans" / variant).mkdir(parents=True, exist_ok=True)
    pj = proj / "project.json"
    if not pj.exists():
        pj.write_text(json.dumps({"name": slug, "slug": slug, "location": "", "ifcs": [], "scan_days": [], "scans": [],
                                  "evaluation": {"scene": scene, "purpose": "slam2 variants (claude_stac_slam2.txt phase 4)"}},
                                 indent=2))
    link = proj / "scans" / variant / "src_default"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            shutil.rmtree(link)
    os.symlink(Path(session_dir).resolve(), link, target_is_directory=True)
    rec = {"project": slug, "scan_key": f"{variant}/default", "path": str(link), "target": str(Path(session_dir).resolve())}
    log(f"[slam2] viewer: {slug} / {variant} → {session_dir}")
    return rec


def mirror_instruments(variant_session: Path, scene_session: Path, log: Callable = print) -> Dict[str, Any]:
    """Copy the reference's seam_radial/ and attention_calibration/ directories (records of the
    instruments, json / csv / png) under the scene's own maplong_run/, replacing older copies."""
    src = run_dir_of(variant_session)
    dst = run_dir_of(scene_session)
    out: Dict[str, Any] = {}
    for name in ("seam_radial", "attention_calibration"):
        s_, d_ = src / name, dst / name
        if not s_.is_dir():
            out[name] = {"copied": False, "reason": f"{s_} does not exist"}
            continue
        dst.mkdir(parents=True, exist_ok=True)
        if d_.exists():
            shutil.rmtree(d_)
        shutil.copytree(s_, d_)
        out[name] = {"copied": True, "to": str(d_)}
        log(f"[slam2] {name}/ mirrored to the scene: {d_}")
    return out


# ── the hooks the executor calls ──────────────────────────────────────────────────────────

class Slam2Hooks:
    """Wired by the CLI when ``evaluation.slam2`` exists: ``resolve`` before a variant runs,
    ``after`` once it finished (instruments, viewer publication, bit-for-bit of the reference)."""

    def __init__(self, slam2: Dict[str, Any], base_config: Dict[str, Any], specs: Sequence[Any], *,
                 projects_dir: Optional[Path], reference: str = "reference", gpu: bool = True, log: Callable = print):
        self.slam2 = slam2
        self.base_config = base_config
        self.specs = list(specs)
        self.projects_dir = projects_dir
        self.reference = reference
        self.gpu = gpu
        self.log = log

    def resolve(self, spec, scene, results):
        return resolve_variant(spec, scene_session=scene.session_dir, results=results, slam2=self.slam2,
                               reference=self.reference, log=self.log)

    def after(self, spec, scene, rec: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> None:
        if rec.get("status") != "finished" or not rec.get("session_dir"):
            return
        sd = Path(rec["session_dir"])
        info: Dict[str, Any] = {}
        try:
            info["seam_radial"] = run_seam_radial(sd, self.base_config, self.log)
            if spec.name == self.reference:
                params = gate_params_from_overrides(self.specs)
                if not self.gpu:
                    info["attention_calibrate"] = {"ran": False, "reason": "no card for this run (--no-gpu-check): the "
                                                                           "calibration needs one Ω pass per labelled pair"}
                elif params is None:
                    info["attention_calibrate"] = {"ran": False, "reason": "no gate variant declares calibration parameters"}
                else:
                    info["attention_calibrate"] = run_attention_calibrate(sd, self.base_config, params, self.log)
            if spec.name == self.reference and self.slam2.get("mirror_instruments_to_scene"):
                # the reference's instruments are copied under the SCENE's own run directory, so a later
                # invocation on the same scene (one variant per run, the disk being what it is) resolves
                # its gate variants from them — a record, never a product of the scene
                info["mirrored_to_scene"] = mirror_instruments(sd, scene.session_dir, self.log)
            if spec.name == self.reference:
                existing = (self.slam2.get("existing_runs") or {}).get(scene.name)
                if existing:
                    ex = Path(str(existing))
                    if not ex.is_absolute():
                        from config import PROJECTS_DIR
                        parts = str(existing).split("/")
                        ex = Path(PROJECTS_DIR) / parts[0] / "scans" / parts[1] / f"src_{parts[2] if len(parts) > 2 else 'default'}"
                    info["bitwise_vs_existing"] = bitwise_compare(sd, ex, self.log)
                else:
                    info["bitwise_vs_existing"] = {"compared": False, "reason": f"no existing run declared for {scene.name} "
                                                                                 f"(evaluation.slam2.existing_runs)"}
            if self.slam2.get("publish_to_viewer") and self.projects_dir is not None:
                info["viewer"] = publish_variant_scan(self.projects_dir, scene.name, spec.name, sd,
                                                      self.slam2["viewer_project_suffix"], self.log)
        except Exception as e:  # noqa: BLE001 — the hook never fails a finished variant
            info["error"] = f"{type(e).__name__}: {e}"
            info["traceback"] = traceback.format_exc()
        info["metrics"] = variant_metrics(sd)
        rec["slam2"] = info

    def finish_scene(self, scene, scene_dir: Path, results: Dict[str, Dict[str, Any]], png: bool = True) -> Dict[str, str]:
        bitwise = ((results.get(self.reference) or {}).get("slam2") or {}).get("bitwise_vs_existing")
        return write_table(scene_dir, scene.name, results, self.reference, bitwise=bitwise, png=png)
