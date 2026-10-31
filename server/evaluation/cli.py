"""Phase 8 — the command line of the battery.

    python -m server.evaluation run --scenes <ids> --variants <list|all> [--repeat 2] [--no-plots]
                                   [--out <dir>] [--no-gpu-check]
    python -m server.evaluation import-colmap --scene <id> --name <variant> --sparse <dir> --dense <ply> [--run <id>]
    python -m server.evaluation list

A scene id is a path to a scan directory (frames/ + output/), a name of ``evaluation.scenes``, or a
project key ``<project>/<date>/<source>`` under the server's projects dir. A run lands in
``runs/eval/<YYYYmmdd_HHMMSS>/`` (``evaluation.runs_dir``) with manifest.json, every CSV / JSON /
PNG of phases 1–7 and a log per variant (``<scene>/<variant>/log.txt``) plus run.log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evaluation import variants as V
from evaluation.config import EvaluationConfig, load_evaluation_config, repo_root, runs_root


class CliError(RuntimeError):
    pass


def resolve_scene(spec: str, cfg: EvaluationConfig) -> V.SceneSpec:
    p = Path(spec)
    if p.is_dir() and (p / "frames").is_dir():
        return V.SceneSpec(p.name if p.name != "src_default" else p.parent.parent.name + "_" + p.parent.name, p.resolve())
    if spec in cfg.scenes:
        d = Path(cfg.scenes[spec])
        d = d if d.is_absolute() else repo_root() / d
        if not (d / "frames").is_dir():
            raise CliError(f"scene {spec!r}: {d} has no frames/")
        return V.SceneSpec(spec, d.resolve())
    parts = spec.split("/")
    if len(parts) == 3:
        from config import PROJECTS_DIR
        d = Path(PROJECTS_DIR) / parts[0] / "scans" / parts[1] / f"src_{parts[2]}"
        if (d / "frames").is_dir():
            return V.SceneSpec(f"{parts[0]}_{parts[1]}_{parts[2]}", d.resolve())
        raise CliError(f"scene {spec!r}: {d} does not exist")
    raise CliError(f"scene {spec!r}: not a scan directory, not in evaluation.scenes, not a project/date/source key")


def select_variants(spec: str, all_specs: Dict[str, V.VariantSpec]) -> List[V.VariantSpec]:
    if spec.strip() == "all":
        return list(all_specs.values())
    out = []
    for name in [x.strip() for x in spec.split(",") if x.strip()]:
        if name not in all_specs:
            raise CliError(f"variant {name!r} is not in evaluation.variants ({sorted(all_specs)})")
        out.append(all_specs[name])
    if not out:
        raise CliError("no variant selected")
    return out


def gpu_gate(specs: Sequence[V.VariantSpec], log: Callable) -> Optional[str]:
    """The card must be free for the chain and COLMAP (rule 6): the reason when it is not."""
    if not any(s.kind in ("chain", "chain_continuation", "colmap") and s.enabled for s in specs):
        return None
    try:
        from repro import require_exclusive_gpu
        require_exclusive_gpu(log=log)
        return None
    except ImportError as e:
        return f"cannot check the card: {e}"
    except Exception as e:  # noqa: BLE001 — ReproError and friends: the reason, recorded
        return f"the card is not free: {e}"


def card_key_or_none(log: Callable) -> Optional[str]:
    try:
        from repro import card_identity, card_key
        return card_key(card_identity())
    except Exception as e:  # noqa: BLE001
        log(f"[evaluation] card identity unavailable ({e}); the default GPU price applies")
        return None


def repo_git_state() -> Dict[str, Any]:
    """Branch and commit of stac-build and of vendor/VGGT-Long (the fork), as repro.git_state reads
    them — recorded in every run's manifest (claude_stac_slam2.txt phase 5)."""
    import repro
    root = Path(__file__).resolve().parents[2]
    out: Dict[str, Any] = {}
    import subprocess
    for name, path in (("stac_build", root), ("fork_vggt_long", root / "vendor" / "VGGT-Long")):
        try:
            rec = dict(repro.git_state(path))
            rec["branch"] = subprocess.run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
                                           capture_output=True, text=True, check=True).stdout.strip()
            rec["path"] = str(path)
            out[name] = rec
        except Exception as e:  # noqa: BLE001 — declared
            out[name] = {"error": f"{type(e).__name__}: {e}", "path": str(path)}
    return out


def _write_manifest(run_dir: Path, man: Dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str))


def cmd_run(args, cfg: EvaluationConfig, runner: Optional[V.ChainRunner], log: Callable) -> int:
    from evaluation import cost as COST
    from evaluation import determinism as DET
    from evaluation import pareto as PAR
    from pipeline_manager import read_config_once
    base_config = read_config_once()
    all_specs = V.load_variants(cfg.raw)
    specs = select_variants(args.variants, all_specs)
    scenes = [resolve_scene(s, cfg) for s in [x.strip() for x in args.scenes.split(",") if x.strip()]]
    if not scenes:
        raise CliError("no scene given")
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(args.out) if args.out else runs_root(cfg)) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = open(run_dir / "run.log", "a")

    def _log(msg: str) -> None:
        run_log.write(str(msg) + "\n"); run_log.flush()
        log(msg)

    blocked = None if args.no_gpu_check else gpu_gate(specs, _log)
    if blocked:
        _log(f"[evaluation] GPU variants will NOT run: {blocked}")
        specs = [V.VariantSpec(s.name, "not_executed", s.description, reason=blocked) if s.kind != "not_executed"
                 else s for s in specs]
    eff = {s.name: V.apply_overrides(base_config, s.overrides) if s.kind in ("chain", "chain_continuation") else {}
           for s in specs}
    man = DET.run_manifest(run_id, scenes, {s.name: s for s in specs}, cfg, base_config, eff, gpu=not args.no_gpu_check,
                           argv=list(sys.argv[1:]), status="running", repeat=args.repeat)
    man["git"] = repo_git_state()                 # branch + commit of stac-build and of the fork (phase 5)
    _write_manifest(run_dir, man)
    runner = runner or V.PipelineChainRunner()
    # claude_stac_slam2.txt phase 4: the slam2 hooks when evaluation.slam2 exists
    from evaluation import slam2 as S2
    slam2 = S2.slam2_config(cfg.raw)
    hooks = None
    if slam2 is not None:
        if slam2["projects_dir"]:
            _projects = Path(slam2["projects_dir"])
        else:
            from config import PROJECTS_DIR
            _projects = Path(PROJECTS_DIR)
        hooks = S2.Slam2Hooks(slam2, base_config, specs, projects_dir=_projects, gpu=not args.no_gpu_check, log=_log)
        _log(f"[evaluation] slam2 hooks on (skip_if_offline_identity {slam2['skip_if_offline_identity']}, references "
             f"{'optional' if slam2['references_optional'] else 'required'})")
    card = None if args.no_gpu_check else card_key_or_none(_log)
    scene_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    status = "finished"
    for scene in scenes:
        vmin = COST.video_minutes(scene.session_dir)

        def factory(out_dir, name, _vmin=vmin):
            return COST.StageMonitor(out_dir, name, cfg.cost, card_key=card, prices=cfg.cost.gpu_price_per_hour,
                                     video_min=_vmin)

        try:
            res = V.run_scene(scene, specs, run_dir, cfg, runner, base_config, _log, monitor_factory=factory,
                              plots=not args.no_plots, repeat=args.repeat,
                              references_optional=bool(slam2 and slam2["references_optional"]), hooks=hooks)
            scene_results[scene.name] = res["results"]
        except Exception as e:  # noqa: BLE001 — a scene that cannot even start (no references) is recorded
            status = "failed"
            scene_results[scene.name] = {}
            (run_dir / scene.name).mkdir(exist_ok=True)
            (run_dir / scene.name / "scene_error.json").write_text(json.dumps(
                {"scene": scene.name, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, indent=1))
            _log(f"[evaluation] scene {scene.name} FAILED: {type(e).__name__}: {e}")
    try:
        PAR.run_phase7(run_dir, scene_results, cfg.curves.tolerances_m, cfg.profiles.min_test_coverage,
                       plots=not args.no_plots)
    except Exception as e:  # noqa: BLE001
        status = "failed"
        _log(f"[evaluation] phase 7 FAILED: {type(e).__name__}: {e}")
    man["status"] = status
    man["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    man["results"] = {s: {v: {"status": r.get("status"), "reason": r.get("reason")} for v, r in res.items()}
                      for s, res in scene_results.items()}
    _write_manifest(run_dir, man)
    _log(f"[evaluation] run {run_id} {status}: {run_dir}")
    run_log.close()
    print(run_dir)
    return 0 if status == "finished" else 1


def cmd_import(args, cfg: EvaluationConfig, log: Callable) -> int:
    scene = resolve_scene(args.scene, cfg)
    run_dir = (Path(args.out) if args.out else runs_root(cfg)) / (args.run or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "manifest.json").exists():
        from evaluation import determinism as DET
        from pipeline_manager import read_config_once
        man = DET.run_manifest(run_dir.name, [scene], {args.name: V.VariantSpec(args.name, "import")}, cfg,
                               read_config_once(), {}, gpu=False, argv=list(sys.argv[1:]), status="running")
        _write_manifest(run_dir, man)
    rec = V.import_colmap_variant(scene, args.name, Path(args.sparse), Path(args.dense), run_dir, cfg, log,
                                  plots=not args.no_plots)
    man = json.loads((run_dir / "manifest.json").read_text())
    man.setdefault("results", {}).setdefault(scene.name, {})[args.name] = {"status": rec["status"], "reason": rec.get("reason")}
    if scene.name not in man.get("scenes", []):
        man.setdefault("scenes", []).append(scene.name)
    if args.name not in man.get("variants", []):
        man.setdefault("variants", []).append(args.name)
    man["status"] = "finished" if rec["status"] == "finished" else "failed"
    _write_manifest(run_dir, man)
    print(run_dir / scene.name / args.name)
    return 0 if rec["status"] == "finished" else 1


def cmd_list(args, cfg: EvaluationConfig) -> int:
    from evaluation.api import list_runs
    root = Path(args.out) if args.out else runs_root(cfg)
    for r in list_runs(root):
        print(f"{r['run_id']}  {r['status']:9s}  scenes={','.join(r['scenes'])}  variants={len(r['variants'])}  files={r['n_files']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m server.evaluation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the battery on scenes × variants")
    r.add_argument("--scenes", required=True, help="comma-separated scene ids (path | evaluation.scenes name | project/date/source)")
    r.add_argument("--variants", default="all", help="comma-separated variant names of evaluation.variants, or 'all'")
    r.add_argument("--repeat", type=int, default=1, help="run every executable variant N times (repeatability)")
    r.add_argument("--out", default=None, help="runs root (default evaluation.runs_dir)")
    r.add_argument("--run-id", default=None, help="run directory name (default: the date_time)")
    r.add_argument("--no-plots", action="store_true", help="skip the PNG charts")
    r.add_argument("--no-gpu-check", action="store_true", help="do not require a free card (tests / CPU-only variants)")
    i = sub.add_parser("import-colmap", help="evaluate an external COLMAP-format reconstruction as a variant")
    i.add_argument("--scene", required=True)
    i.add_argument("--name", required=True, help="variant name of the import")
    i.add_argument("--sparse", required=True, help="COLMAP model dir (cameras/images/points3D .bin|.txt) or its parent")
    i.add_argument("--dense", required=True, help="the dense cloud (PLY)")
    i.add_argument("--run", default=None, help="an existing run id to add to (default: a new run)")
    i.add_argument("--out", default=None)
    i.add_argument("--no-plots", action="store_true")
    ls = sub.add_parser("list", help="list the runs")
    ls.add_argument("--out", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None, *, runner: Optional[V.ChainRunner] = None,
         cfg: Optional[EvaluationConfig] = None, log: Callable = print) -> int:
    args = build_parser().parse_args(argv)
    cfg = cfg or load_evaluation_config()
    try:
        if args.cmd == "run":
            return cmd_run(args, cfg, runner, log)
        if args.cmd == "import-colmap":
            return cmd_import(args, cfg, log)
        if args.cmd == "list":
            return cmd_list(args, cfg)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 2
