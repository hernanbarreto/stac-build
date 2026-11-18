"""IS THE REBUILD KIT STILL THE POD? Every part of the live pod against what the kit would rebuild.

    python3 scripts/rebuild/drift.py            the whole comparison (≈ 4 min, read-only), exit 1 on any difference
    python3 scripts/rebuild/drift.py --fast     the package lists, the vendor commits and the weights' presence
                                                only (seconds: what tests/test_rebuild_kit.py runs)

USER 2026-11-18: "cada cosa que avancemos debemos actualizarlo para que siempre quede actualizado a
la perfección". The kit is a description of the pod; the pod changes with every install, clone,
download and compilation. This is what tells one from the other, part by part:

  envs       conda's exact packages and pip's distributions of every env against locks/<env>.*
             (the full run also compares every distribution's files and looks for files edited in
             place or owned by no package)
  vendors    every repository's commit and uncommitted diff against steps_vendors.sh and the inventory
  weights    models.tsv against the files (present, their size) and the hub caches' own content
  volume     inventory.py check: every file of the volume in its class, the archived ones by sha256
  backup     the backup set against the pod (members changed, new files it lacks, git work it lacks)
  git        what is not committed and not pushed in the repo itself — a kit that is not on the
             remote does not exist for a new volume

When it reports a difference: `bash scripts/rebuild/capture.sh`, commit scripts/rebuild/, copy the
backup set off the pod (and push, on the user's word).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LOCKS = HERE / "locks"
REPO = HERE.parent.parent
WORKSPACE = Path(os.environ.get("WORKSPACE") or "/workspace")
CONDA = Path(os.environ.get("CONDA_ROOT") or WORKSPACE / "miniforge3")


def sh(*cmd, cwd=None) -> str:
    return subprocess.run(list(cmd), capture_output=True, text=True, cwd=cwd).stdout


def envs():
    text = (HERE / "lib.sh").read_text()
    return re.search(r"^STAC_ENVS=\(([^)]*)\)", text, re.M).group(1).split()


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lines(path: Path):
    return sorted(l for l in path.read_text().splitlines() if l.strip()) if path.exists() else []


def diff(label: str, want, got, problems: list, limit: int = 8) -> None:
    want, got = set(want), set(got)
    for item in sorted(want - got)[:limit]:
        problems.append(f"{label}: in the lock, NOT on the pod — {item[:150]}")
    for item in sorted(got - want)[:limit]:
        problems.append(f"{label}: on the pod, NOT in the lock — {item[:150]}")
    extra = len(want ^ got) - min(len(want - got), limit) - min(len(got - want), limit)
    if extra > 0:
        problems.append(f"{label}: … and {extra} more")


# ── envs ───────────────────────────────────────────────────────────────────────────────────────
def conda_urls(prefix: Path):
    return [l.split("#")[0] for l in sh(str(CONDA / "bin" / "conda"), "list", "-p", str(prefix), "--explicit").splitlines() if l.startswith("https")]


def env_fast(env: str) -> list:
    """Names and versions only, read from the env's own metadata directories (no interpreter run)."""
    problems = []
    prefix = CONDA / "envs" / env
    if not prefix.is_dir():
        return [f"{env}: the env does not exist on this pod"]
    meta = {p.name[:-5] for p in (prefix / "conda-meta").glob("*.json")}
    lock = {re.sub(r"\.(conda|tar\.bz2)$", "", l.split("#")[0].rsplit("/", 1)[-1])
            for l in lines(LOCKS / f"{env}.conda-explicit.txt") if l.startswith("https")}
    diff(f"{env} conda", lock, meta, problems)
    records = LOCKS / f"{env}.records.tsv"
    if records.exists():
        want = {f"{norm(l.split(chr(9))[0])}=={l.split(chr(9))[1]}" for l in lines(records)}
        got = set()
        for site in prefix.glob("lib/python3*/site-packages"):
            for info in site.glob("*.dist-info"):
                if (info / "INSTALLER").exists() and (info / "INSTALLER").read_text().strip().lower() == "conda":
                    continue
                m = re.match(r"^(.+)-([^-]+)\.dist-info$", info.name)
                if m:
                    got.add(f"{norm(m.group(1))}=={m.group(2)}")
        diff(f"{env} pip", want, got, problems)
    return problems


def env_full(env: str, tmp: Path) -> list:
    problems = []
    prefix = CONDA / "envs" / env
    if not prefix.is_dir():
        return [f"{env}: the env does not exist on this pod"]
    want = [l.split("#")[0] for l in lines(LOCKS / f"{env}.conda-explicit.txt") if l.startswith("https")]
    diff(f"{env} conda", want, conda_urls(prefix), problems)
    out = tmp / env
    out.mkdir()
    subprocess.run(["python3", str(HERE / "audit_env_files.py"), str(prefix), env, str(out)], capture_output=True, text=True)
    for suffix in ("orphans.tsv", "conda-modified.tsv"):
        diff(f"{env} {suffix}", lines(LOCKS / f"{env}.{suffix}"), lines(out / f"{env}.{suffix}"), problems)
    python = prefix / "bin" / "python"
    if python.exists():
        run = subprocess.run([str(python), str(HERE / "capture_env.py"), env, str(out)], capture_output=True, text=True)
        if run.returncode != 0:
            return problems + [f"{env}: capture_env.py failed: {run.stderr.strip()[-200:]}"]
        for suffix in ("pip.txt", "pip-url.txt", "pip-local.tsv", "records.tsv", "overlaps.tsv", "modified.tsv"):
            diff(f"{env} {suffix}", lines(LOCKS / f"{env}.{suffix}"), lines(out / f"{env}.{suffix}"), problems)
        saved = {p.relative_to(HERE / "overlays" / env).as_posix() for p in (HERE / "overlays" / env).rglob("*") if p.is_file()} \
            if (HERE / "overlays" / env).is_dir() else set()
        for line in lines(out / f"{env}.modified.tsv"):
            path = line.split("\t")[3]
            if path not in saved:
                problems.append(f"{env}: {path} is edited in place and NOT saved under overlays/{env}/")
    return problems


# ── vendors ────────────────────────────────────────────────────────────────────────────────────
def vendors() -> list:
    problems = []
    text = (HERE / "steps_vendors.sh").read_text()
    for name, final, patch in re.findall(r'^\s*"([A-Za-z0-9._-]+)\|https://[^|]+\|[0-9a-f]{40}\|[^|]*\|([0-9a-f]{40})\|([^|]*)\|', text, re.M):
        repo = REPO / "vendor" / name
        head = sh("git", "-C", str(repo), "rev-parse", "HEAD").strip()
        if head != final:
            problems.append(f"vendor/{name}: at {head[:10]}, steps_vendors.sh rebuilds {final[:10]}")
        dirty = sh("git", "-C", str(repo), "diff", "HEAD")
        if patch != "-":
            if dirty != (REPO / patch).read_text():
                problems.append(f"vendor/{name}: its uncommitted diff is no longer {patch}")
        elif dirty.strip():
            problems.append(f"vendor/{name}: uncommitted changes that no patch records")
    # the submodules: the commit the repo pins against the clone's own HEAD (a clone that was never
    # registered with `git submodule init` is still the right tree when its HEAD is the pin)
    for line in sh("git", "-C", str(REPO), "ls-files", "-s", "vendor").splitlines():
        if line.startswith("160000"):
            pinned, path = line.split()[1], line.split("\t")[1]
            head = sh("git", "-C", str(REPO / path), "rev-parse", "HEAD").strip()
            if head != pinned:
                problems.append(f"submodule {path}: at {head[:10]}, the repo pins {pinned[:10]}")
            if sh("git", "-C", str(REPO / path), "status", "--porcelain", "--untracked-files=no").strip():
                problems.append(f"submodule {path}: uncommitted changes")
    return problems


# ── weights ────────────────────────────────────────────────────────────────────────────────────
def weights() -> list:
    import capture_models
    import models
    problems = []
    for row in models.rows():
        p = models.target(row)
        if row["kind"] == "link":
            if not p.is_symlink():
                problems.append(f"weight link missing: {row['dest']}")
        elif not p.exists():
            problems.append(f"weight missing on the pod: {row['dest'] if row['kind'] != 'hfcache' else row['source']}")
        elif row["size"] and row["kind"] not in ("copy",) and p.stat().st_size != row["size"]:
            problems.append(f"weight of another size: {row['dest'] if row['kind'] != 'hfcache' else row['source']}")
    in_file = {(r["dest"], r["source"], r["sha"], str(r["size"])) for r in models.rows() if r["kind"] == "hfcache"}
    live = {(r[1], r[2], r[3], r[4]) for r in capture_models.cache_rows({})}
    diff("hub caches", {" ".join(r[:2]) for r in in_file}, {" ".join(r[:2]) for r in live}, problems)
    return problems


# ── the repo itself ────────────────────────────────────────────────────────────────────────────
def repo_git() -> list:
    notes = []
    dirty = sh("git", "-C", str(REPO), "status", "--porcelain", "--", "scripts/rebuild", ".gitignore", ".gitmodules").strip()
    if dirty:
        notes.append(f"scripts/rebuild has {len(dirty.splitlines())} uncommitted change(s)")
    branch = sh("git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD").strip()
    ahead = sh("git", "-C", str(REPO), "rev-list", "--count", f"origin/{branch}..{branch}").strip()
    if ahead not in ("", "0"):
        notes.append(f"{ahead} commit(s) of {branch} are NOT on origin — the kit a new volume clones is the remote's")
    return notes


def main() -> int:
    fast = "--fast" in sys.argv
    sections = []
    names = envs()
    if fast:
        env_problems = [p for env in names for p in env_fast(env)]
    else:
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=6) as pool:
            env_problems = [p for group in pool.map(lambda e: env_full(e, Path(tmp)), names) for p in group]
    sections.append(("envs", env_problems))
    sections.append(("vendors", vendors()))
    sections.append(("weights", weights()))
    rc = 0
    for title, problems in sections:
        print(f"── {title}: {'as the kit describes' if not problems else str(len(problems)) + ' DIFFERENCE(S)'}")
        for p in problems:
            print(f"   {p}")
        rc |= bool(problems)
    if not fast:
        import backup
        import inventory
        print("── volume (every file, in its class):")
        rc |= inventory.check()
        print("── backup set:")
        rc |= backup.stale(Path(os.environ.get("STAC_BACKUP") or backup.DEFAULT_SET))
    notes = repo_git()
    print(f"── git: {'committed and pushed' if not notes else ''}")
    for n in notes:
        print(f"   {n}")
    rc |= bool(notes)
    print("\nTHE KIT IS THE POD." if not rc else
          "\nTHE KIT IS NOT THE POD: bash scripts/rebuild/capture.sh, commit scripts/rebuild/, copy the backup set off "
          "the pod (and push, on the user's word).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
