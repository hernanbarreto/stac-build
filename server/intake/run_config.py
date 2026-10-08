"""The run's configuration, FROZEN at job start (docs/plan_determinismo.md point 69, 2026-10-07).

A pipeline job used to run with two configurations: the dict the backend loaded at startup
(what main.py handed the workers) and ``config.yaml`` as it stood on disk whenever a module did
``from config import cfg`` inside a spawned worker (intake.walk sized the I3 windows that way).
Edit config.yaml while a backend is up — which happens whenever anyone works on the repo — and
the focal probe, I1 and Omega ran on one configuration while I3 ran on another, with no record
of either.

Now the pipeline manager writes ``<session>/output/run_config.yaml`` (the job's configuration
as one canonical YAML text) and ``run_config.sha256`` (the digest of those bytes) once, after
the replace wipe and before the first stage, and every step reads THAT copy
(:func:`load_run_config` verifies the digest — an edited file is refused) and stamps its
products with its sha256. A command-line step (``python -m intake.run`` and the stage CLIs) is
its own job: it freezes the configuration it starts with (:func:`cli_intake_config`) when the
session holds none, and reads the frozen one when it does.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import repro
from intake.config import IntakeConfig, load_intake_config

OUTPUT_DIRNAME = "output"
RUN_CONFIG_NAME = "run_config.yaml"
RUN_CONFIG_SHA_NAME = "run_config.sha256"
STAGES_KEY = "_stages"                 # per-stage overrides of the pipeline's stage list
LOG_TAG = "[run_config]"


class RunConfigError(RuntimeError):
    """The frozen configuration is missing, unreadable, edited after it was frozen, or not the
    configuration a step was handed — always naming what differs."""


def run_config_path(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / OUTPUT_DIRNAME / RUN_CONFIG_NAME


def run_config_sha_path(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / OUTPUT_DIRNAME / RUN_CONFIG_SHA_NAME


def canonical_yaml(config: Mapping[str, Any]) -> str:
    """ONE text per configuration: JSON-able values only (a value yaml.safe_load cannot have
    produced is refused — it would not round-trip), keys sorted, block style, UTF-8."""
    import yaml
    try:
        plain = json.loads(json.dumps(dict(config)))
    except (TypeError, ValueError) as e:
        raise RunConfigError(f"the run configuration is not a plain YAML document: {e}") from e
    return yaml.safe_dump(plain, sort_keys=True, default_flow_style=False, allow_unicode=True,
                          width=100000)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def freeze_run_config(session_dir: os.PathLike, config: Mapping[str, Any], *,
                      stages: Optional[Mapping[str, Mapping[str, Any]]] = None,
                      log=print) -> Dict[str, Any]:
    """Write ``output/run_config.yaml`` (+ ``run_config.sha256``) from ``config`` — the job's
    configuration — with the pipeline's per-stage overrides under :data:`STAGES_KEY` (only the
    non-empty ones). Returns {path (session-relative), sha256}. Called ONCE per job by the
    pipeline manager after the replace wipe; a CLI calls it when it starts a job of its own."""
    doc = dict(config)
    doc.pop(STAGES_KEY, None)
    overrides = {str(k): dict(v) for k, v in (stages or {}).items() if v}
    if overrides:
        doc[STAGES_KEY] = overrides
    text = canonical_yaml(doc)
    sha = repro.sha256_bytes(text.encode("utf-8"))
    p = run_config_path(session_dir)
    _write_text_atomic(p, text)
    _write_text_atomic(run_config_sha_path(session_dir), sha + "\n")
    log(f"{LOG_TAG} frozen → {OUTPUT_DIRNAME}/{RUN_CONFIG_NAME} (sha256 {sha[:12]}; every step "
        f"of this job reads this copy, never config.yaml)")
    return {"path": f"{OUTPUT_DIRNAME}/{RUN_CONFIG_NAME}", "sha256": sha}


def has_run_config(session_dir: os.PathLike) -> bool:
    return run_config_path(session_dir).exists()


def load_run_config(session_dir: os.PathLike) -> Tuple[Dict[str, Any], str]:
    """(the frozen configuration, its sha256). The digest file must match the YAML's bytes: a
    file edited after it was frozen is refused, and so is a missing digest."""
    import yaml
    p, ps = run_config_path(session_dir), run_config_sha_path(session_dir)
    if not p.exists():
        raise RunConfigError(f"{p} does not exist — the job's configuration was not frozen (the "
                             f"pipeline manager writes it at job start; a CLI freezes its own)")
    data = p.read_bytes()
    sha = repro.sha256_bytes(data)
    if not ps.exists():
        raise RunConfigError(f"{ps} does not exist beside {p.name} — the frozen configuration "
                             f"cannot be verified")
    want = ps.read_text().strip()
    if want != sha:
        raise RunConfigError(f"{p} was edited after it was frozen (sha256 {sha[:12]} ≠ recorded "
                             f"{want[:12]}) — a step never runs on a configuration nobody froze")
    try:
        doc = yaml.safe_load(data.decode("utf-8"))
    except (ValueError, yaml.YAMLError) as e:
        raise RunConfigError(f"{p} is not readable YAML ({e})") from e
    if not isinstance(doc, dict):
        raise RunConfigError(f"{p} holds a {type(doc).__name__}, not a configuration mapping")
    return doc, sha


def effective_config(frozen: Mapping[str, Any], stage: Optional[str] = None) -> Dict[str, Any]:
    """The configuration a stage ran with: the job's, plus that stage's overrides."""
    doc = {k: v for k, v in frozen.items() if k != STAGES_KEY}
    if stage is not None:
        over = (frozen.get(STAGES_KEY) or {}).get(stage) or {}
        doc.update(dict(over))
    return doc


def intake_config_differences(a: IntakeConfig, b: IntakeConfig) -> Dict[str, Any]:
    """{'section.field': (a, b)} of every field that differs between two intake configurations."""
    out: Dict[str, Any] = {}
    da, db = asdict(a), asdict(b)
    for sec in sorted(set(da) | set(db)):
        fa, fb = da.get(sec) or {}, db.get(sec) or {}
        for k in sorted(set(fa) | set(fb)):
            if json.dumps(fa.get(k), sort_keys=True, default=list) != \
                    json.dumps(fb.get(k), sort_keys=True, default=list):
                out[f"{sec}.{k}"] = (fa.get(k), fb.get(k))
    return out


def verify_intake_config(session_dir: os.PathLike, icfg: IntakeConfig,
                         stage: str = "reconstruction") -> str:
    """The intake configuration a step was HANDED must be the frozen job's (its ``intake:`` and
    the keys it cross-reads, as :func:`load_intake_config` reads them). Returns the frozen
    configuration's sha256 — what the step stamps its products with. A difference FAILS naming
    the fields: a step never runs on a configuration nobody froze."""
    frozen, sha = load_run_config(session_dir)
    want = load_intake_config(effective_config(frozen, stage))
    diff = intake_config_differences(want, icfg)
    if diff:
        shown = "; ".join(f"{k}: frozen {a!r} ≠ handed {b!r}" for k, (a, b) in list(diff.items())[:6])
        raise RunConfigError(f"the intake was handed a configuration that is not the frozen "
                             f"{OUTPUT_DIRNAME}/{RUN_CONFIG_NAME} of this job — {shown}")
    return sha


def raw_with_intake(base_raw: Mapping[str, Any], icfg: IntakeConfig) -> Dict[str, Any]:
    """``base_raw`` (a full configuration dict) with its ``intake:`` section and the SAM3 batch
    key replaced by the values of ``icfg`` — so that ``load_intake_config`` of the result equals
    ``icfg``. For callers that hold a typed configuration and must freeze the raw one (tests, a
    CLI that overrides a parameter)."""
    raw = json.loads(json.dumps(dict(base_raw)))
    d = asdict(icfg)
    content = dict(d["content"])
    sam3_batch = content.pop("sam3_batch")
    content["prompts"] = {k: list(v) for k, v in content["prompts"].items()}
    content["exclusion_classes"] = list(content["exclusion_classes"])
    content["weight_classes"] = list(content["weight_classes"])
    raw["intake"] = {"runtime": dict(d["runtime"]), "quality": dict(d["quality"]),
                     "parallax": dict(d["parallax"]), "content": content}
    raw.setdefault("models", {}).setdefault("segmentation", {})["batch_size"] = int(sam3_batch)
    return raw


def cli_intake_config(session_dir: os.PathLike, log=print) -> Tuple[IntakeConfig, str]:
    """The intake configuration of a command-line step: the session's frozen run configuration
    when there is one (verified), else the server configuration frozen NOW — this CLI is the
    job's start. Returns (IntakeConfig, the frozen configuration's sha256)."""
    if has_run_config(session_dir):
        frozen, sha = load_run_config(session_dir)
        log(f"{LOG_TAG} reading the frozen {OUTPUT_DIRNAME}/{RUN_CONFIG_NAME} (sha256 {sha[:12]})")
        return load_intake_config(effective_config(frozen, "reconstruction")), sha
    from config import cfg as raw_cfg                     # server/config.py, read once here
    rec = freeze_run_config(session_dir, raw_cfg, log=log)
    return load_intake_config(raw_cfg), rec["sha256"]
