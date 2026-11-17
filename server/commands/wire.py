"""Mark a command of the catalogue as RUN BY STAC: ``status: manual`` goes and its ``done`` (and
``ask`` / ``done_off``) texts are written, in the three languages, where the declaration keeps
them — a text edit that leaves the rest of the file as it is.

    python -m commands.wire texts.yaml     # {command id: {es: {done, ask?, done_off?}, en: …, fr: …}}

Then: the executor in ui/src/App.tsx and ``python -m commands.export_ui``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .catalog import AREAS_DIR, LANGS, load_catalog

FIELDS = ("done", "done_off", "ask")


def _scalar(text: str) -> str:
    out = yaml.safe_dump(text, allow_unicode=True, width=10 ** 6).strip()
    return out[:-3].strip() if out.endswith("...") else out


def wire(cid: str, texts: dict, directory: Path = AREAS_DIR) -> None:
    for path in sorted(directory.glob("*.yaml")):
        lines = path.read_text(encoding="utf-8").split("\n")
        try:
            start = lines.index(f"- id: {cid}")
        except ValueError:
            continue
        end = next((k for k in range(start + 1, len(lines)) if lines[k].startswith("- id: ")), len(lines))
        block = lines[start:end]
        if "  status: manual" not in block:
            raise SystemExit(f"{cid}: not a manual command")
        block.remove("  status: manual")
        for lang in LANGS:
            at = block.index(f"    {lang}:")
            stop = next((k for k in range(at + 1, len(block)) if not block[k].startswith("      ")), len(block))
            while stop > at + 1 and not block[stop - 1].strip():
                stop -= 1
            extra = set(texts[lang]) - set(FIELDS)
            if extra or "done" not in texts[lang]:
                raise SystemExit(f"{cid}, {lang}: 'done' is missing or unknown field(s) {sorted(extra)}")
            block[stop:stop] = [f"      {f}: {_scalar(texts[lang][f])}" for f in FIELDS if f in texts[lang]]
        path.write_text("\n".join(lines[:start] + block + lines[end:]), encoding="utf-8")
        return
    raise SystemExit(f"{cid}: not in the catalogue")


def main(argv: list[str]) -> int:
    doc = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    for cid, texts in doc.items():
        wire(cid, texts)
    load_catalog()                       # the declarations still load (every required text is there)
    print(f"wired {len(doc)} command(s): {', '.join(doc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
