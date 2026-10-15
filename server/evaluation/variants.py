"""Phase 3 — the contribution of each stage: a variant executor that runs the chain on a COPY
of a scene with configuration OVERRIDES (the server's defaults never change), stages switched
off, the COLMAP baselines and external imports, then evaluates every variant with phases 1 and
2 and its coverage. A variant that fails is recorded with its error and the others run.

Variants are configured (``evaluation.variants``); each is one of
  chain               the chain with ``overrides`` (dotted config keys) and ``stages_off``
  chain_continuation  the chain re-run on a finished variant's copy (``base``) with overrides —
                      the gauge-only scale: F2's instrument becomes the gauge distances written
                      as output/known_dimensions.json (measured on Omega's own record)
  colmap              classical photogrammetry (evaluation.colmap_baseline)
  import              an external COLMAP-format model + dense cloud (the CLI's import-colmap)
  not_executed        recorded with its reason, never run
"""
from __future__ import annotations

import copy
import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from evaluation import curves as CURVES
from evaluation import dimension_error as D
from evaluation.config import EvaluationConfig, EvaluationConfigError
from evaluation.coverage import coverage as compute_coverage
from evaluation.product import Product, StacProduct

KINDS = ("chain", "chain_continuation", "colmap", "import", "not_executed")
STATUSES = ("finished", "failed", "not_executed")


class VariantError(RuntimeError):
    pass


@dataclass
class VariantSpec:
    name: str
    kind: str
    description: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)
    stages_off: List[str] = field(default_factory=list)
    base: Optional[str] = None
    colmap: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "description": self.description,
                "overrides": self.overrides, "stages_off": self.stages_off, "base": self.base,
                "colmap": self.colmap, "reason": self.reason, "enabled": self.enabled}


@dataclass
class SceneSpec:
    name: str
    session_dir: Path

    @property
    def frames_dir(self) -> Path:
        return self.session_dir / "frames"


def load_variants(raw_eval: Dict[str, Any]) -> Dict[str, VariantSpec]:
    sec = raw_eval.get("variants")
    if not isinstance(sec, dict) or not sec:
        raise EvaluationConfigError("'evaluation.variants' must be a non-empty mapping")
    out: Dict[str, VariantSpec] = {}
    for name, v in sec.items():
        if not isinstance(v, dict) or "kind" not in v:
            raise EvaluationConfigError(f"'evaluation.variants.{name}' must be a mapping with 'kind'")
        kind = v["kind"]
        if kind not in KINDS:
            raise EvaluationConfigError(f"'evaluation.variants.{name}.kind' must be one of {KINDS}, got {kind!r}")
        spec = VariantSpec(name=str(name), kind=kind, description=str(v.get("description", "")),
                           overrides=dict(v.get("overrides") or {}), stages_off=list(v.get("stages_off") or []),
                           base=v.get("base"), colmap=dict(v.get("colmap") or {}), reason=str(v.get("reason", "")),
                           enabled=bool(v.get("enabled", True)))
        if kind == "chain_continuation" and not spec.base:
            raise EvaluationConfigError(f"'evaluation.variants.{name}' (chain_continuation) needs 'base'")
        if kind == "not_executed" and not spec.reason:
            raise EvaluationConfigError(f"'evaluation.variants.{name}' (not_executed) needs 'reason'")
        if kind == "colmap":
            for k in ("features", "matcher", "mapper"):
                if k not in spec.colmap:
                    raise EvaluationConfigError(f"'evaluation.variants.{name}.colmap.{k}' is missing")
        out[spec.name] = spec
    return out


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """A deep copy of ``config`` with every dotted key set. The PARENT of a key must exist (a typo
    never creates a section the chain does not read); the leaf may be new (a flag added for the
    battery)."""
    out = copy.deepcopy(config)
    for dotted, value in overrides.items():
        parts = str(dotted).split(".")
        node = out
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise VariantError(f"override {dotted!r}: section {'.'.join(parts[:parts.index(p) + 1])!r} "
                                   f"does not exist in the configuration")
            node = node[p]
        if not isinstance(node, dict):
            raise VariantError(f"override {dotted!r}: {'.'.join(parts[:-1])!r} is not a mapping")
        node[parts[-1]] = copy.deepcopy(value)
    return out


# ── the chain ─────────────────────────────────────────────────────────────

class ChainRunner(Protocol):
    def run(self, session_dir: Path, config: Dict[str, Any], stages_off: Sequence[str], *, replace: bool,
            force: bool, log: Callable, on_stage: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        """Run the chain on ``session_dir`` with ``config`` (the whole configuration mapping);
        raises on failure; returns a record (stages run, seconds per stage)."""


class PipelineChainRunner:
    """The production chain through the pipeline manager, headless (what the server does on
    "Reconstruir"): the configuration is frozen into the session copy and every worker verifies
    it. Needs the da3 interpreter and a free card (the chain's own rules)."""

    def run(self, session_dir: Path, config: Dict[str, Any], stages_off: Sequence[str], *, replace: bool,
            force: bool, log: Callable, on_stage: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        import asyncio
        from pipeline_manager import (PipelineJob, PipelineManager, PipelineStage, StageState, StageId,
                                      build_pipeline_stages, select_stages)
        off = {StageId(s) for s in stages_off}
        stages = []
        for st in select_stages(build_pipeline_stages(config=config), segment=True):
            if st.id in off:
                st = PipelineStage(id=st.id, enabled=False, config=dict(st.config))
            stages.append(StageState(stage=st))
        job = PipelineJob(session_id=Path(session_dir).name, stages=stages, scan_key=None, kind="reconstruct")
        job.session_dir = str(session_dir)
        pm = PipelineManager()
        done: Dict[str, Any] = {}
        timings: Dict[str, float] = {}
        last = {"idx": -1, "t": time.time()}

        async def _progress(sid, state):
            idx = state.get("current_stage_idx", -1)
            if idx != last["idx"]:
                now = time.time()
                if last["idx"] >= 0:
                    timings[str(stages[last["idx"]].stage.id.value)] = round(now - last["t"], 2)
                last["idx"], last["t"] = idx, now
                if on_stage is not None and 0 <= idx < len(stages):
                    on_stage(str(stages[idx].stage.id.value), "start")

        async def _complete(sid, ok):
            done["ok"] = ok

        async def _body():
            pm._jobs[pm.job_key(job.session_id, job.scan_key, job.kind)] = job
            await pm._run_pipeline(job, str(session_dir), config, _progress, _complete, replace, force)

        asyncio.run(_body())
        if last["idx"] >= 0:
            timings[str(stages[last["idx"]].stage.id.value)] = round(time.time() - last["t"], 2)
        rec = {"ok": bool(done.get("ok")), "stages": [{"id": s.stage.id.value, "enabled": s.stage.enabled,
                                                       "status": str(s.status.value), "message": s.message}
                                                      for s in stages], "seconds_by_stage": timings}
        if not done.get("ok"):
            failed = [s for s in stages if str(s.status.value) == "failed"]
            raise VariantError("the chain failed" + (f" at {failed[0].stage.id.value}: {failed[0].message}"
                                                     if failed else ""))
        return rec


def copy_inputs(src: Path, dst: Path, log: Callable) -> None:
    """A session copy with its INPUTS only (frames/ with its manifest, the source video, inputs/):
    what a replace run keeps. The chain rebuilds the rest."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("frames", "inputs"):
        s = src / name
        if s.exists():
            if (dst / name).exists():
                shutil.rmtree(dst / name)
            shutil.copytree(s, dst / name, symlinks=True)
    for p in src.glob("source_video.*"):
        shutil.copy2(p, dst / p.name)
    log(f"[variants] inputs copied {src} → {dst}")


def copy_session(src: Path, dst: Path, log: Callable) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    log(f"[variants] session copied {src} → {dst}")


def keyframes_of_scene(scene: SceneSpec) -> List[str]:
    """The chain's keyframe files of the scene (the intake's selection; what every variant is
    given), from frames/selected_frames.json, else output/camera_frames.txt."""
    sel = scene.frames_dir / "selected_frames.json"
    if sel.exists():
        files = json.loads(sel.read_text()).get("selected_files") or []
        if files:
            return [str(f) for f in files]
    cf = scene.session_dir / "output" / "camera_frames.txt"
    if cf.exists():
        from intake.quality import frame_file
        return [frame_file(scene.frames_dir, int(x)).name for x in cf.read_text().split()]
    raise VariantError(f"{scene.session_dir}: no keyframe list (frames/selected_frames.json or "
                       f"output/camera_frames.txt) — run the intake first")


def frame_numbers(files: Sequence[str]) -> List[int]:
    return [int(Path(f).stem) for f in files]


# ── gauge-only scale: known_dimensions.json from Omega's record ───────────

def write_known_dimensions(session_dir: Path, doc: Dict[str, Any], log: Callable) -> Path:
    """The gauge distances measured on OMEGA'S OWN RECORD (omega_run/results_output/frame_<f>.npz:
    depth + pose in Omega's world, the geometry F2 gauges), written as F8's known_dimensions.json
    for ``gauge.instruments: [known_dims]``, stamped with the reconstruction id."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    from evaluation import geometry as G
    from precision.camera import load_camera_json, native_to_grid
    out = Path(session_dir) / "output"
    cam = load_camera_json(out / "camera.json")
    rec_dir = out / "omega_run" / "results_output"
    dims = []
    for d in doc.get("distances", []):
        if d["role"] != "gauge":
            continue
        pts, frames = [], []
        for end in d["ends"]:
            cand = []
            for o in end["observations"]:
                f = int(o["frame"])
                p = rec_dir / f"frame_{f}.npz"
                if not p.exists():
                    continue
                with np.load(p) as z:
                    depth, c2w, K = z["depth"], z["pose_c2w"], z["K_omega"]
                g = cam.omega_grid
                uv = native_to_grid(np.array([[float(o["u"]), float(o["v"])]]), g)[0]
                u, v = int(round(uv[0])), int(round(uv[1]))
                if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]) or not depth[v, u] > 0:
                    continue
                cand.append(G.unproject(uv[None], np.array([float(depth[v, u])]), K, c2w)[0])
                frames.append(f)
            if cand:
                pts.append(np.median(np.asarray(cand), axis=0))
        if len(pts) == 2:
            measured = float(np.linalg.norm(pts[0] - pts[1]))
            dims.append({"id": d["id"], "role": "gauge", "frames": sorted(set(frames)), "measured_m": measured,
                         "true_m": float(d["value_m"]), "sigma_m": float(d["sigma_m"])})
        else:
            log(f"[variants] gauge {d['id']}: not measurable on Omega's record — left out of known_dimensions")
    if not dims:
        raise VariantError("no gauge distance could be measured on Omega's record")
    path = out / "known_dimensions.json"
    path.write_text(json.dumps({RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(out), "dimensions": dims,
                                "provenance": "evaluation battery: gauge distances on Omega's record"},
                               indent=1, sort_keys=True))
    log(f"[variants] known_dimensions.json: {len(dims)} gauge dimension(s) → {path}")
    return path


# ── evaluation of one product ─────────────────────────────────────────────

def evaluate_product(product: Product, doc: Dict[str, Any], out_dir: Path, cfg: EvaluationConfig,
                     keyframes: Sequence[int], *, n_models: int = 1, models: Optional[List[dict]] = None,
                     plots: bool = True) -> Dict[str, Any]:
    from evaluation import confidence as CF
    rows, summary, recs = D.evaluate(doc, product, cfg)
    D.write_phase1(out_dir, rows, summary, recs)
    p2 = CURVES.run_phase2(doc, rows, summary, product, cfg, out_dir, plots=plots)
    cov = compute_coverage(product, keyframes, doc, recs, int(cfg.raw["coverage"]["cell_px"]), n_models, models)
    D.write_json(out_dir / "coverage.json", cov)
    p5 = CF.run_phase5(doc, rows, recs, summary, product, cfg.curves, cfg.confidence, cfg.endpoints.depth_window_px,
                       out_dir)
    return {"rows": rows, "summary": summary, "phase2": {k: v for k, v in p2.items() if k != "paths"},
            "coverage": cov, "recs": recs, "confidence": {k: v for k, v in p5.items() if k != "groups"}}


# ── one variant ───────────────────────────────────────────────────────────

def _record(out_dir: Path, rec: Dict[str, Any]) -> Dict[str, Any]:
    D.write_json(out_dir / "variant.json", rec)
    return rec


def run_variant(scene: SceneSpec, spec: VariantSpec, scene_dir: Path, cfg: EvaluationConfig, runner: ChainRunner,
                base_config: Dict[str, Any], doc: Dict[str, Any], keyframe_files: Sequence[str], log: Callable,
                results: Dict[str, Dict[str, Any]], monitor_factory: Optional[Callable] = None,
                plots: bool = True) -> Dict[str, Any]:
    """Run + evaluate one variant into ``scene_dir/<variant>/``. Never raises: the record carries
    status finished | failed | not_executed with the reason / traceback."""
    out_dir = scene_dir / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    kfs = frame_numbers(keyframe_files)
    rec: Dict[str, Any] = {"variant": spec.to_dict(), "scene": scene.name, "status": None, "reason": "",
                           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": None, "product": None}
    t0 = time.time()
    logf = open(out_dir / "log.txt", "a")

    def _log(msg: str) -> None:
        logf.write(str(msg) + "\n"); logf.flush()
        log(f"[{scene.name}/{spec.name}] {msg}")

    monitor = monitor_factory(out_dir, spec.name) if monitor_factory is not None else None
    try:
        if not spec.enabled:
            rec.update(status="not_executed", reason="disabled in evaluation.variants")
            return _record(out_dir, rec)
        if spec.kind == "not_executed":
            rec.update(status="not_executed", reason=spec.reason)
            return _record(out_dir, rec)
        if monitor is not None:
            monitor.start()
        if spec.kind in ("chain", "chain_continuation"):
            session = out_dir / "session"
            config = apply_overrides(base_config, spec.overrides)
            if spec.kind == "chain":
                copy_inputs(scene.session_dir, session, _log)
                replace, force = True, False
            else:
                base = results.get(spec.base or "")
                if not base or base.get("status") != "finished" or not base.get("session_dir"):
                    rec.update(status="not_executed",
                               reason=f"base variant {spec.base!r} did not finish — nothing to continue from")
                    return _record(out_dir, rec)
                copy_session(Path(base["session_dir"]), session, _log)
                rec["known_dimensions"] = str(write_known_dimensions(session, doc, _log))
                replace, force = False, True
            rec["effective_overrides"] = spec.overrides
            rec["chain"] = runner.run(session, config, spec.stages_off, replace=replace, force=force, log=_log,
                                      on_stage=(monitor.stage if monitor is not None else None))
            product = StacProduct(session, spec.name, log=_log)
            rec["session_dir"] = str(session)
            n_models, models = 1, [{"id": 0, "n_images": len(product.frames)}]
        elif spec.kind == "colmap":
            from evaluation import colmap_baseline as CB
            from evaluation.colmap_io import list_models, load_dense_xyz, model_product, read_model
            ccfg = cfg.raw["colmap"]
            man = CB.run_colmap_variant(scene.frames_dir, keyframe_files, scene_dir / "colmap", spec.colmap["features"],
                                        spec.colmap["matcher"], spec.colmap["mapper"], ccfg, _log)
            D.write_json(out_dir / "colmap_manifest.json", man)
            rec["colmap"] = {"workspace": man["workspace"], "models": man["mapping"]["models"]}
            ws = Path(man["workspace"])
            model_dirs = list_models(ws / man["sparse_dir"])
            if not model_dirs:
                raise VariantError("COLMAP registered no model")
            loaded = [(d, read_model(d)) for d in model_dirs]
            loaded.sort(key=lambda t: -t[1].n_images)
            models = [{"id": i, "dir": str(d.relative_to(ws)), "n_images": m.n_images, "n_points3d": m.n_points3d}
                      for i, (d, m) in enumerate(loaded)]
            largest_dir, largest = loaded[0]
            fused = ws / "dense" / largest_dir.relative_to(ws) / "fused.ply"
            if not fused.exists():
                raise VariantError(f"no dense cloud for the largest model ({largest.n_images} images): {fused}")
            product = model_product(largest, load_dense_xyz(fused), spec.name,
                                    meta={"sparse": str(largest_dir), "dense": str(fused)})
            n_models = len(loaded)
            rec["product"] = {"sparse": str(largest_dir), "dense": str(fused), "n_models": n_models}
        elif spec.kind == "import":
            rec.update(status="not_executed", reason="imports run through the CLI (import-colmap), not the variant list")
            return _record(out_dir, rec)
        else:
            raise VariantError(f"unknown kind {spec.kind}")
        if monitor is not None:
            monitor.stage("evaluation", "start")
        ev = evaluate_product(product, doc, out_dir, cfg, kfs, n_models=n_models, models=models, plots=plots)
        rec.update(status="finished", summary=ev["summary"], coverage=ev["coverage"],
                   confidence_distance=ev["phase2"]["confidence_distance"], growth_models=ev["phase2"]["growth_models"],
                   confidence=ev["confidence"])
        rec["product"] = (rec.get("product") or {}) | ev["summary"]["product"]
        return _record(out_dir, rec)
    except Exception as e:  # noqa: BLE001 — isolated: the record carries it, the other variants run
        rec.update(status="failed", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        _log(f"FAILED: {type(e).__name__}: {e}")
        return _record(out_dir, rec)
    finally:
        rec["seconds"] = round(time.time() - t0, 2)
        if monitor is not None:
            try:
                rec["cost"] = monitor.stop()
                from evaluation.cost import attach_torch_peaks
                attach_torch_peaks(rec)
            except Exception as e:  # noqa: BLE001
                rec["cost"] = {"error": f"{type(e).__name__}: {e}"}
            _record(out_dir, rec)
        logf.close()


# ── a scene ───────────────────────────────────────────────────────────────

COMPARISON_COLUMNS = ["scene", "variant", "status", "reason", "n_test_evaluated", "n_test_not_reconstructed",
                      "abs_median_m", "abs_p95_m", "abs_max_m", "rel_median", "rel_p95", "scale_factor",
                      "d_abs_median_vs_reference_m", "d_abs_p95_vs_reference_m", "keyframes_registered_pct",
                      "n_models", "test_ends_reconstructed_pct", "planes_surface_with_points_pct_mean",
                      "seconds", "gpu_hours", "cost_usd"]


def comparison_rows(scene: str, results: Dict[str, Dict[str, Any]], reference: str) -> List[Dict[str, Any]]:
    ref = results.get(reference) or {}
    ref_s = (ref.get("summary") or {}).get("abs_error_m") or {}
    rows = []
    for name, r in results.items():
        s = r.get("summary") or {}
        a = s.get("abs_error_m") or {}
        rl = s.get("rel_error") or {}
        cov = r.get("coverage") or {}
        cost = r.get("cost") or {}
        row = {"scene": scene, "variant": name, "status": r.get("status"), "reason": r.get("reason", ""),
               "n_test_evaluated": s.get("n_test_evaluated"), "n_test_not_reconstructed": s.get("n_test_not_reconstructed"),
               "abs_median_m": a.get("median"), "abs_p95_m": a.get("p95"), "abs_max_m": a.get("max"),
               "rel_median": rl.get("median"), "rel_p95": rl.get("p95"),
               "scale_factor": (s.get("scale") or {}).get("scale_factor"),
               "d_abs_median_vs_reference_m": (a["median"] - ref_s["median"]) if a.get("median") is not None
               and ref_s.get("median") is not None else None,
               "d_abs_p95_vs_reference_m": (a["p95"] - ref_s["p95"]) if a.get("p95") is not None
               and ref_s.get("p95") is not None else None,
               "keyframes_registered_pct": cov.get("keyframes_registered_pct"), "n_models": cov.get("n_models"),
               "test_ends_reconstructed_pct": cov.get("test_ends_reconstructed_pct"),
               "planes_surface_with_points_pct_mean": cov.get("planes_surface_with_points_pct_mean"),
               "seconds": r.get("seconds"), "gpu_hours": cost.get("gpu_hours"), "cost_usd": cost.get("cost_usd")}
        rows.append(row)
    return rows


def keep_policy(results: Dict[str, Dict[str, Any]], policy: str, log: Callable) -> None:
    """After a scene: ``all`` keeps every variant session, ``cloud`` keeps the published products
    the viewer and a re-evaluation need, ``none`` deletes the copies."""
    if policy == "all":
        return
    keep_names = {"camera.json", "camera_frames.txt", "camera_poses.txt", "cleaned_cloud.ply", "instance_ids.npy",
                  "run_config.yaml", "run_config.sha256", "geometry_epoch.json", "corrected_cloud.json",
                  "known_dimensions.json", "certify_acta.json", "chunk_plan.json", "gauge.json", "scale_diagnostics.json"}
    for name, r in results.items():
        sd = r.get("session_dir")
        if not sd or not Path(sd).exists():
            continue
        sd = Path(sd)
        if policy == "none":
            shutil.rmtree(sd, ignore_errors=True)
            log(f"[variants] {name}: session copy deleted (evaluation.variants_keep: none)")
            continue
        out = sd / "output"
        for p in list(sd.iterdir()):
            if p.name != "output":
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        if out.exists():
            for p in list(out.iterdir()):
                if p.name in keep_names or (p.is_dir() and p.name in ("potree", "precision")):
                    continue
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        log(f"[variants] {name}: session copy trimmed to its products (evaluation.variants_keep: cloud)")


REPEAT_SUFFIX = "__rep"


def repeated_specs(specs: Sequence[VariantSpec], repeat: int) -> List[VariantSpec]:
    """``--repeat N``: every executable variant N times (``<name>__rep2`` …), the same spec."""
    out = list(specs)
    for k in range(2, int(repeat) + 1):
        for s in specs:
            if s.kind in ("chain", "colmap", "chain_continuation") and s.enabled:
                out.append(VariantSpec(f"{s.name}{REPEAT_SUFFIX}{k}", s.kind, s.description, dict(s.overrides),
                                       list(s.stages_off), s.base, dict(s.colmap), s.reason, s.enabled))
    return out


def run_scene(scene: SceneSpec, specs: Sequence[VariantSpec], run_dir: Path, cfg: EvaluationConfig, runner: ChainRunner,
              base_config: Dict[str, Any], log: Callable, monitor_factory: Optional[Callable] = None,
              reference: str = "reference", plots: bool = True, repeat: int = 1) -> Dict[str, Any]:
    from evaluation.references import load_references
    scene_dir = run_dir / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)
    doc = load_references(scene.session_dir / "output", cfg.references.file, require_roles=True)
    kf_files = keyframes_of_scene(scene)
    results: Dict[str, Dict[str, Any]] = {}
    all_specs = repeated_specs(specs, repeat)
    order = sorted(all_specs, key=lambda s: 1 if s.kind == "chain_continuation" else 0)   # bases first
    for spec in order:
        results[spec.name] = run_variant(scene, spec, scene_dir, cfg, runner, base_config, doc, kf_files, log,
                                         results, monitor_factory, plots=plots)
    if repeat > 1:
        from evaluation.determinism import repeatability
        for s in specs:
            for k in range(2, int(repeat) + 1):
                rep = f"{s.name}{REPEAT_SUFFIX}{k}"
                if rep in results:
                    repeatability(scene_dir, s.name, rep, results[s.name], results[rep], log)
    rows = comparison_rows(scene.name, results, reference)
    D.write_csv(scene_dir / "comparison.csv", rows, COMPARISON_COLUMNS)
    D.write_json(scene_dir / "comparison.json", {"scene": scene.name, "reference": reference, "rows": rows,
                                                 "keyframes": len(kf_files)})
    keep_policy(results, str(cfg.raw["variants_keep"]), log)
    return {"scene": scene.name, "results": results, "comparison": rows}


# ── external import ───────────────────────────────────────────────────────

def import_colmap_variant(scene: SceneSpec, name: str, sparse_dir: Path, dense_ply: Path, run_dir: Path,
                          cfg: EvaluationConfig, log: Callable, plots: bool = True) -> Dict[str, Any]:
    """An external reconstruction (RealityScan, Metashape, …) exported in COLMAP format (cameras /
    images / points3D, binary or text; one model directory or a parent holding 0/, 1/, …) plus its
    dense cloud, evaluated as a variant with phases 1 and 2 and the coverage. Image names must be
    the scene's frame files ("000051.jpg"). The largest model is the product; every model counts."""
    from evaluation.colmap_io import list_models, load_dense_xyz, model_product, read_model
    from evaluation.references import load_references
    scene_dir = run_dir / scene.name
    out_dir = scene_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rec: Dict[str, Any] = {"variant": VariantSpec(name, "import", description="external import").to_dict(),
                           "scene": scene.name, "status": None, "reason": "", "import": {"sparse": str(sparse_dir),
                                                                                         "dense": str(dense_ply)}}
    t0 = time.time()
    try:
        doc = load_references(scene.session_dir / "output", cfg.references.file, require_roles=True)
        kfs = frame_numbers(keyframes_of_scene(scene))
        dirs = list_models(Path(sparse_dir))
        if not dirs:
            raise VariantError(f"{sparse_dir}: no COLMAP model found")
        loaded = sorted([(d, read_model(d)) for d in dirs], key=lambda t: -t[1].n_images)
        models = [{"id": i, "dir": str(d), "n_images": m.n_images, "n_points3d": m.n_points3d}
                  for i, (d, m) in enumerate(loaded)]
        product = model_product(loaded[0][1], load_dense_xyz(Path(dense_ply)), name,
                                meta={"sparse": str(loaded[0][0]), "dense": str(dense_ply)})
        ev = evaluate_product(product, doc, out_dir, cfg, kfs, n_models=len(loaded), models=models, plots=plots)
        rec.update(status="finished", summary=ev["summary"], coverage=ev["coverage"],
                   confidence_distance=ev["phase2"]["confidence_distance"], growth_models=ev["phase2"]["growth_models"],
                   product={"n_models": len(loaded)} | ev["summary"]["product"])
    except Exception as e:  # noqa: BLE001
        rec.update(status="failed", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    rec["seconds"] = round(time.time() - t0, 2)
    _record(out_dir, rec)
    # the scene's comparison table, refreshed with every variant recorded so far
    results = {}
    for d in sorted(p for p in scene_dir.iterdir() if p.is_dir() and (p / "variant.json").exists()):
        results[d.name] = json.loads((d / "variant.json").read_text())
    rows = comparison_rows(scene.name, results, "reference")
    D.write_csv(scene_dir / "comparison.csv", rows, COMPARISON_COLUMNS)
    D.write_json(scene_dir / "comparison.json", {"scene": scene.name, "reference": "reference", "rows": rows})
    return rec
