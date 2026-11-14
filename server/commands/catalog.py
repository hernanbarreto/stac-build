"""The declarations of the commands (``catalog/*.yaml``), read and VALIDATED: a missing or
mistyped field fails the load naming the file, the command and the field — a command is never
half declared."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

AREAS_DIR = Path(__file__).resolve().parent / "catalog"
LANGS = ("es", "en", "fr")                   # the system's three languages
KINDS = ("action", "query", "talk")          # STAC operates / STAC measures or reads / conversation
RUNS_ON = ("server", "ui")                   # the server's tool layer / the interface
STATUS = ("available", "planned")
# what a command needs to be there before it can run (the interface greys it otherwise)
NEEDS = ("session", "scan", "cloud", "segmentation", "selection", "mesh", "bim", "volume",
         "measurement", "admin", "engine")
ARG_TYPES = ("string", "integer", "number", "boolean", "array", "object")
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_TOOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class Catalog:
    areas: List[Dict[str, Any]]              # in position order
    commands: List[Dict[str, Any]]           # in (area position, order in the file)
    sha256: str                              # of the validated content: the database's stamp

    def command(self, cid: str) -> Dict[str, Any]:
        for c in self.commands:
            if c["id"] == cid:
                return c
        raise KeyError(cid)


def _fail(where: str, why: str) -> None:
    raise CatalogError(f"commands catalog — {where}: {why}")


def _texts(where: str, node: Any, fields: Mapping[str, bool]) -> Dict[str, Dict[str, Any]]:
    """``{lang: {field: value}}`` with every language and every required field present."""
    if not isinstance(node, dict):
        _fail(where, "the texts are missing (one block per language: es, en, fr)")
    out: Dict[str, Dict[str, Any]] = {}
    for lang in LANGS:
        t = node.get(lang)
        if not isinstance(t, dict):
            _fail(where, f"the '{lang}' text is missing")
        row: Dict[str, Any] = {}
        for f, required in fields.items():
            v = t.get(f)
            if f == "say":
                if v is None and not required:
                    v = []
                if not isinstance(v, list) or any(not isinstance(s, str) or not s.strip() for s in v) \
                        or (required and not v):
                    _fail(where, f"'{lang}.say' must be a list of phrases" + (" (at least one)" if required else ""))
                row[f] = [s.strip() for s in v]
                continue
            if v is None and not required:
                row[f] = ""
                continue
            if not isinstance(v, str) or not v.strip():
                _fail(where, f"'{lang}.{f}' is missing")
            row[f] = v.strip()
        extra = set(t) - set(fields)
        if extra:
            _fail(where, f"'{lang}' has unknown field(s) {sorted(extra)}")
        out[lang] = row
    extra = set(node) - set(LANGS)
    if extra:
        _fail(where, f"unknown language(s) {sorted(extra)}")
    return out


def _args(where: str, node: Any) -> Dict[str, Dict[str, Any]]:
    if node is None:
        return {}
    if not isinstance(node, dict):
        _fail(where, "'args' must be a mapping name → {type, ...}")
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in node.items():
        if isinstance(name, bool):
            # YAML 1.1 reads a bare on / off / yes / no as a boolean — the commonest argument name
            _fail(where, f"argument name {name!r}: YAML read a bare on / off / yes / no as a boolean — "
                         f"quote it ('on': …)")
        if not isinstance(name, str) or not _ID.match(name):
            _fail(where, f"argument name {name!r} is not snake_case")
        if not isinstance(spec, dict):
            _fail(where, f"argument '{name}' must be a mapping")
        t = spec.get("type")
        types = t if isinstance(t, list) else [t]
        if not types or any(x not in ARG_TYPES for x in types):
            _fail(where, f"argument '{name}': type must be one (or a list) of {list(ARG_TYPES)}")
        known = {"type", "description", "enum", "items", "minimum", "maximum", "required"}
        extra = set(spec) - known
        if extra:
            _fail(where, f"argument '{name}' has unknown field(s) {sorted(extra)}")
        if "required" in spec and not isinstance(spec["required"], bool):
            _fail(where, f"argument '{name}': required must be true / false")
        if "enum" in spec and (not isinstance(spec["enum"], list) or not spec["enum"]):
            _fail(where, f"argument '{name}': enum must be a non-empty list")
        out[name] = dict(spec)
    return out


def tool_schema(cmd: Mapping[str, Any]) -> Dict[str, Any]:
    """The command as the model is offered it (the OpenAI function format)."""
    props, required = {}, []
    for name, spec in cmd["args"].items():
        props[name] = {k: v for k, v in spec.items() if k != "required"}
        if spec.get("required"):
            required.append(name)
    return {"type": "function", "function": {
        "name": cmd["tool"], "description": cmd["model"],
        "parameters": {"type": "object", "properties": props, "required": required}}}


def load_catalog(directory: Path | str | None = None) -> Catalog:
    directory = Path(directory) if directory is not None else AREAS_DIR
    files = sorted(directory.glob("*.yaml"))
    if not files:
        _fail(str(directory), "no area file (*.yaml)")
    areas: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []
    seen_ids: Dict[str, str] = {}
    seen_tools: Dict[str, str] = {}
    for f in files:
        doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            _fail(f.name, "not a mapping")
        extra = set(doc) - {"area", "position", "title", "summary", "commands"}
        if extra:
            _fail(f.name, f"unknown field(s) {sorted(extra)}")
        area = doc.get("area")
        if not isinstance(area, str) or not _ID.match(area):
            _fail(f.name, "'area' is missing or not snake_case")
        if any(a["id"] == area for a in areas):
            _fail(f.name, f"area '{area}' is declared twice")
        pos = doc.get("position")
        if isinstance(pos, bool) or not isinstance(pos, int):
            _fail(f.name, "'position' (an integer: the area's order) is missing")
        head = _texts(f"{f.name} (area {area})",
                      {lang: {"title": (doc.get("title") or {}).get(lang),
                              "summary": (doc.get("summary") or {}).get(lang)} for lang in LANGS},
                      {"title": True, "summary": True})
        areas.append({"id": area, "position": pos, "text": head, "file": f.name})
        rows = doc.get("commands")
        if not isinstance(rows, list) or not rows:
            _fail(f.name, "'commands' is missing or empty")
        for k, row in enumerate(rows):
            if not isinstance(row, dict):
                _fail(f.name, f"command {k + 1} is not a mapping")
            cid = row.get("id")
            where = f"{f.name}, command {cid or k + 1}"
            if not isinstance(cid, str) or not re.match(rf"^{area}\.[a-z][a-z0-9_]*$", cid):
                _fail(where, f"'id' must be '{area}.<name>' in snake_case")
            if cid in seen_ids:
                _fail(where, f"id declared twice (also in {seen_ids[cid]})")
            seen_ids[cid] = f.name
            known = {"id", "tool", "kind", "runs_on", "handler", "model", "args", "needs", "confirm",
                     "status", "since", "core", "text"}
            extra = set(row) - known
            if extra:
                _fail(where, f"unknown field(s) {sorted(extra)}")
            tool = row.get("tool", cid.replace(".", "__"))
            if not isinstance(tool, str) or not _TOOL.match(tool):
                _fail(where, "'tool' (the function name the model calls) is not [A-Za-z0-9_]{1,64}")
            if tool in seen_tools:
                _fail(where, f"tool name '{tool}' is already {seen_tools[tool]}'s")
            seen_tools[tool] = cid
            kind, runs_on = row.get("kind"), row.get("runs_on")
            if kind not in KINDS:
                _fail(where, f"'kind' is missing or not one of {list(KINDS)}")
            if runs_on not in RUNS_ON:
                _fail(where, f"'runs_on' is missing or not one of {list(RUNS_ON)}")
            model = row.get("model")
            if not isinstance(model, str) or not model.strip():
                _fail(where, "'model' (what the model reads to choose this command) is missing")
            needs = row.get("needs", [])
            if not isinstance(needs, list) or any(n not in NEEDS for n in needs):
                _fail(where, f"'needs' must be a list among {list(NEEDS)}")
            confirm = row.get("confirm", False)
            if not isinstance(confirm, bool):
                _fail(where, "'confirm' must be true / false")
            core = row.get("core", False)
            if not isinstance(core, bool):
                _fail(where, "'core' must be true / false")
            status = row.get("status", "available")
            if status not in STATUS:
                _fail(where, f"'status' must be one of {list(STATUS)}")
            # an action of the interface says what was done; a server tool's answer is written
            # from its result (phase5_qa/answers.py)
            text = _texts(where, row.get("text"),
                          {"title": True, "about": False, "say": True,
                           "done": runs_on == "ui"})
            commands.append({
                "id": cid, "area": area, "tool": tool, "kind": kind, "runs_on": runs_on,
                "handler": str(row.get("handler") or (tool if runs_on == "server" else cid)),
                "model": " ".join(model.split()), "args": _args(where, row.get("args")),
                "needs": list(needs), "confirm": confirm, "core": core, "status": status,
                "since": str(row.get("since") or ""), "text": text})
    areas.sort(key=lambda a: (a["position"], a["id"]))
    order = {a["id"]: k for k, a in enumerate(areas)}
    commands.sort(key=lambda c: order[c["area"]])                    # stable: the file's order
    digest = hashlib.sha256(json.dumps(
        {"areas": [{k: v for k, v in a.items() if k != "file"} for a in areas], "commands": commands},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return Catalog(areas=areas, commands=commands, sha256=digest)
