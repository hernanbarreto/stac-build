"""The interface's commands as TypeScript: ``ui/src/commands/catalog.generated.ts`` — the ids of
every command the interface must run (``runs_on: ui``, ``status: available``) and the type of
each one's arguments. The interface's executors are typed against it: a command declared in the
catalogue with no executor does not compile, and neither does an executor of a command that is
not declared.

    python -m commands.export_ui            write the file
    python -m commands.export_ui --check    exit 1 when the file on disk is not the catalogue's
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from .catalog import Catalog, load_catalog

OUT = Path(__file__).resolve().parent.parent.parent / "ui" / "src" / "commands" / "catalog.generated.ts"
_TS = {"string": "string", "integer": "number", "number": "number", "boolean": "boolean",
       "array": "unknown[]", "object": "Record<string, unknown>"}


def _arg_type(spec: Dict[str, Any]) -> str:
    if "enum" in spec:
        return " | ".join(repr(v) if isinstance(v, str) else str(v).lower() for v in spec["enum"])
    types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
    return " | ".join(dict.fromkeys(_TS[t] for t in types))


def render(catalog: Catalog) -> str:
    rows = [c for c in catalog.commands if c["runs_on"] == "ui" and c["status"] == "available"]
    lines = [
        "// GENERATED from server/commands/catalog/*.yaml by `python -m commands.export_ui` — do not edit.",
        "// The commands of the interface STAC can run: their ids and the type of each one's arguments.",
        "// A new command is declared in the catalogue; this file is then regenerated and the compiler",
        "// asks for its executor (ui/src/commands/registry.ts).",
        f"export const CATALOG_SHA256 = '{catalog.sha256}'",
        "",
        "export const UI_COMMAND_IDS = [",
        *[f"  '{c['id']}'," for c in rows],
        "] as const",
        "",
        "export type UiCommandId = typeof UI_COMMAND_IDS[number]",
        "",
        "export interface UiCommandArgs {",
    ]
    for c in rows:
        fields = [f"{name}{'' if spec.get('required') else '?'}: {_arg_type(spec)}" for name, spec in c["args"].items()]
        body = "{ " + "; ".join(fields) + " }" if fields else "Record<string, never>"
        lines.append(f"  '{c['id']}': {body}")
    lines += ["}", ""]
    return "\n".join(lines)


def check(path: Path = OUT) -> bool:
    """Is the file on disk exactly what the catalogue gives?"""
    try:
        return path.read_text(encoding="utf-8") == render(load_catalog())
    except OSError:
        return False


def main(argv: list[str]) -> int:
    if "--check" in argv:
        if check():
            return 0
        print(f"{OUT} is not the catalogue's: run `python -m commands.export_ui`")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(load_catalog()), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
