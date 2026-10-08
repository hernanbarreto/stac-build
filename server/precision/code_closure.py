"""The code a precision step runs — plan point 31 (the chain's per-step stamps).

``code_closure(module)`` lists the repo files a step's module reaches through its imports,
followed recursively only inside the precision core (:data:`CODE_FOLLOWED_DIRS` /
:data:`CODE_FOLLOWED_FILES`). It lives in its own module, with no import of a step, so a step
(F2's ``gauge_stamp``) can stamp its own closure without pulling the runner — and through it
every other step — into that closure (an F6 edit must never re-run F2)."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import repro

SERVER_DIR = Path(__file__).resolve().parent.parent


class CodeClosureError(RuntimeError):
    pass



FORK_DIR = repro.FORK_ROOT
_CODE_ROOTS: Tuple[Path, ...] = (SERVER_DIR, FORK_DIR)
# The code a step's stamp FOLLOWS (recursively) — the precision core and what it computes with:
# the precision, correction and intake packages, the loop-closure / geometry helpers, the DA3
# extractor and its card table, the fork's metric lock and tracker. A module outside these that
# one of them imports is stamped too (its own content) but its imports are not followed: through
# correction.apply → segmentation.pipeline → … the transitive closure reached 162 files (measured
# 2026-10-07), so an edit anywhere in the server re-ran the chain from F2 and `--from f6_bend`
# after an F6 edit would have re-run F2–F5. Under this rule F2's stamp holds 57 files, none of
# F4–F6's modules; F6's holds every module up to it.
CODE_FOLLOWED_DIRS: Tuple[Path, ...] = (
    SERVER_DIR / "precision", SERVER_DIR / "correction", SERVER_DIR / "intake",
    SERVER_DIR / "reconstruction" / "loops", SERVER_DIR / "reconstruction" / "geometry",
    FORK_DIR / "loop_utils", FORK_DIR / "dependency")
CODE_FOLLOWED_FILES: Tuple[Path, ...] = (
    SERVER_DIR / "repro.py", SERVER_DIR / "config.py", SERVER_DIR / "card_table.py",
    SERVER_DIR / "da3_weights.py", SERVER_DIR / "extract_da3_depth.py",
    SERVER_DIR / "reconstruction" / "scale_align.py", SERVER_DIR / "reconstruction" / "vio_scale.py",
    SERVER_DIR / "reconstruction" / "colmap_ba.py", SERVER_DIR / "segmentation" / "session_io.py",
    SERVER_DIR / "base_models" / "vggtomega_adapter.py")


def _module_file(name: str, roots: Sequence[Path] = _CODE_ROOTS) -> Optional[Path]:
    rel = name.replace(".", "/")
    for root in roots:
        for cand in (root / f"{rel}.py", root / rel / "__init__.py"):
            if cand.is_file():
                return cand.resolve()
    return None


def _followed(path: Path) -> bool:
    return path in CODE_FOLLOWED_FILES or any(d in path.parents for d in CODE_FOLLOWED_DIRS)


def _imports_of(path: Path) -> List[str]:
    """Dotted module names a file imports anywhere (top level or inside functions), relative
    imports resolved against its package."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        raise CodeClosureError(f"{path}: cannot be parsed to stamp its imports ({e})")
    pkg_parts: List[str] = []
    d = path.parent
    while (d / "__init__.py").exists():
        pkg_parts.insert(0, d.name)
        d = d.parent
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[:len(pkg_parts) - (node.level - 1)] if node.level - 1 else list(pkg_parts)
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod:
                out.append(mod)
                out += [f"{mod}.{a.name}" for a in node.names if a.name != "*"]
            elif node.level:
                out += [".".join(pkg_parts + [a.name]) for a in node.names if a.name != "*"]
    return out


def code_closure(module: str) -> Tuple[List[Path], Dict[str, Path]]:
    """(repo files, external files) a step's module runs: the module and — following its imports
    recursively through :data:`CODE_FOLLOWED_DIRS` / :data:`CODE_FOLLOWED_FILES` (ast, so imports
    inside functions count) — each repo module it reaches; a repo module outside the followed set
    is included when a followed one imports it, its own imports are not. Library code is
    versioned by ``repro.environment_record``. A step module that is not in this repo (a test's
    stand-in on PYTHONPATH) comes back as an external file, stamped by content as an input."""
    start = _module_file(module)
    if start is None:
        import importlib.util
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin or not Path(spec.origin).is_file():
            raise CodeClosureError(f"step module {module!r} is neither a file under {_CODE_ROOTS} nor "
                             f"importable")
        return [], {f"module:{module}": Path(spec.origin).resolve()}
    seen: Dict[Path, None] = {}
    todo = [start]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen[p] = None
        if p is not start and not _followed(p):
            continue
        for name in _imports_of(p):
            f = _module_file(name)
            if f is not None and f not in seen:
                todo.append(f)
    return sorted(seen), {}


