"""THE BACKUP SET: everything of the pod that no public source can give back, as it is.

    python3 backup.py make    [--out DIR]     write the set (default /workspace/stac_backup)
    python3 backup.py verify  [--set DIR]     are the files on this volume the ones of the set? (after a restore)
    python3 backup.py stale   [--set DIR]     has the pod changed since the set was written? (before trusting it)
    python3 backup.py restore-git [--set DIR] the local branches, stashes and worktrees of every repository

USER 2026-11-18: "debe quedar exactamente como está ahora". The locks rebuild the envs from the
exact packages, the manifest fetches the exact weights, git gives the exact code — and what is
left exists only on this volume: what was COMPILED here (a new compilation is the same source and
other bytes: PotreeConverter's sha256 is in every octree's stamp), what a program generated, what
was copied by hand, the git work that was never pushed, the secrets. The set holds them:

  asis.tar.gz      the `asis` class of inventory.py (compiled trees, the built interface, sidecars …)
  envs.tar.gz      inside the conda envs: the distributions compiled on the pod or installed from a
                   directory (their files and metadata), and the files no package owns
  private.tar.gz   the `private` class: .env, the tokens the hub client stored, the sessions' signing
                   key, the users' database, Tailscale's identity, the assistant's local settings
  git/             <repo>.bundle — the commits that are on no remote (unpushed branches, stashes);
                   refs.tsv — every local branch, stash and worktree of every repository
  mirror/<sha256>  the weights their source may refuse: Meta's DINOv3 (a link that expires)
                   — hard links, no second copy; `--all-weights` links every weight of models.tsv
  MANIFEST.tsv     every archive's sha256; <archive>.files.tsv — every member's sha256

Every path is relative to the workspace: restoring an archive is `tar -xzf … -C /workspace`.
THE SET IS NOT A BACKUP WHILE IT SITS ON THE VOLUME IT DESCRIBES: it must be copied off the pod.
private.tar.gz holds secrets: it goes to no public place.

Not in the set: the sessions and the other `data` (inventory.py lists them with their size) — they
are copied apart, they are 150+ GB. Standard library + git.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inventory  # noqa: E402
import models  # noqa: E402

WORKSPACE = inventory.WORKSPACE
REPO = inventory.REPO
LOCKS = HERE / "locks"
CONDA = Path(os.environ.get("CONDA_ROOT") or WORKSPACE / "miniforge3")
DEFAULT_SET = WORKSPACE / "stac_backup"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str, check: bool = False) -> str:
    run = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and run.returncode != 0:
        raise RuntimeError(f"git -C {repo} {' '.join(args)}: {run.stderr.strip()}")
    return run.stdout


# ── archives ───────────────────────────────────────────────────────────────────────────────────
def member_digest(full: Path) -> str:
    if full.is_symlink():
        return "link:" + os.readlink(full)
    if full.is_dir():
        return "dir"
    return sha256_of(full)


def expand(rel_paths):
    """Every file, link and empty directory under the given workspace-relative paths, sorted
    (bytecode left out: python writes it again)."""
    out = set()
    for rel in rel_paths:
        base = rel.rstrip("/")
        full = WORKSPACE / base
        if full.is_symlink() or full.is_file():
            out.add(base)
        elif full.is_dir():
            found = False
            for root, dirs, files in os.walk(full):
                dirs[:] = sorted(d for d in dirs if d != "__pycache__")
                for d in list(dirs):
                    sub = os.path.join(root, d)
                    if os.path.islink(sub):
                        out.add(os.path.relpath(sub, WORKSPACE))
                        dirs.remove(d)
                        found = True
                for name in files:
                    if not name.endswith(".pyc"):
                        out.add(os.path.relpath(os.path.join(root, name), WORKSPACE))
                        found = True
                if not dirs and not files and root != str(full):
                    out.add(os.path.relpath(root, WORKSPACE) + "/")
                    found = True
            if not found:
                out.add(base + "/")
    return sorted(out)


def write_archive(out_dir: Path, name: str, rel_paths) -> tuple:
    """A deterministic .tar.gz of the given paths (sorted, the gzip header without a clock) and its
    .files.tsv (path, sha256 | link | dir, bytes). Returns (members, bytes of the archive)."""
    members = expand(rel_paths)
    archive = out_dir / f"{name}.tar.gz"
    listing = []
    with open(archive, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in members:
            full = WORKSPACE / rel.rstrip("/")
            tar.add(str(full), arcname=rel.rstrip("/"), recursive=False)
            size = 0 if (full.is_symlink() or full.is_dir()) else full.stat().st_size
            listing.append(f"{rel}\t{member_digest(full)}\t{size}")
    (out_dir / f"{name}.files.tsv").write_text("".join(f"{line}\n" for line in listing))
    return len(members), archive.stat().st_size


# ── what lives inside the envs and is not a downloadable package ───────────────────────────────
def env_members():
    """Workspace-relative paths, per env, of the distributions compiled on the pod / installed from
    a directory, and of the files no package owns (capture.sh writes both lists in locks/)."""
    out = []
    for listing in sorted(LOCKS.glob("*.asis-files.tsv")):
        env = listing.name.split(".")[0]
        prefix = CONDA / "envs" / env
        for line in listing.read_text().splitlines():
            rel = line.split("\t")[0].strip()
            if rel and ((prefix / rel).exists() or (prefix / rel).is_symlink()):
                out.append(os.path.relpath(prefix / rel, WORKSPACE))
    return out


# ── git: what is on no remote, and every local name ────────────────────────────────────────────
def on_remotes(repo: Path) -> list:
    """The refs and commits that stand for "what the remotes hold": this clone's remote-tracking
    branches — but only those whose branch STILL EXISTS on the remote, asked of it now (a tracking
    ref may name a branch deleted there: its commits would be lost with the volume) — plus the
    remotes' present tips when this clone has them. A remote that does not answer: its tracking
    refs as they are."""
    out = set()
    for remote in git(repo, "remote").split():
        tracking = {}
        for line in git(repo, "for-each-ref", "--format=%(refname)\t%(objectname)", f"refs/remotes/{remote}").splitlines():
            ref, sha = line.split("\t")
            tracking[ref[len(f"refs/remotes/{remote}/"):]] = sha
        run = subprocess.run(["git", "-C", str(repo), "ls-remote", "--heads", "--tags", remote],
                             capture_output=True, text=True, timeout=180)
        if run.returncode != 0:
            out |= set(tracking.values())
            continue
        live = {}
        for line in run.stdout.splitlines():
            sha, ref = line.split()[:2]
            live[ref] = sha
            if git(repo, "cat-file", "-t", sha).strip() in ("commit", "tag"):
                out.add(sha)
        for branch, sha in tracking.items():
            if branch == "HEAD" or f"refs/heads/{branch}" in live:
                out.add(sha)
    return sorted(out)


def repo_state(repo: Path, out_dir: Path) -> list:
    """refs.tsv rows of one repository; writes <name>.bundle when it holds commits that are on no
    remote (branches ahead of every remote branch, stashes)."""
    name = os.path.relpath(repo, WORKSPACE).replace("/", "__")
    rows, refs = [], []
    remote_tips = on_remotes(repo)
    head = git(repo, "rev-parse", "HEAD").strip()
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    rows.append((name, "head", branch, head, ""))
    for remote in git(repo, "remote").split():
        url = git(repo, "remote", "get-url", remote).strip()
        url = __import__("re").sub(r"(https?://)[^@/]+@", r"\1", url)       # (a token written into a URL stays here)
        rows.append((name, "remote", remote, "-", url))
    for line in git(repo, "for-each-ref", "--format=%(refname:short)\t%(objectname)\t%(upstream:short)", "refs/heads").splitlines():
        b, sha, upstream = (line.split("\t") + ["", ""])[:3]
        rows.append((name, "branch", b, sha, upstream))
        if git(repo, "rev-list", "--count", b, "--not", *remote_tips).strip() not in ("", "0"):
            refs.append(f"refs/heads/{b}")
    stashes = [l.split("\t") for l in git(repo, "stash", "list", "--format=%H\t%gs").splitlines() if l.strip()]
    temp = []
    for i, (sha, message) in enumerate(stashes):
        rows.append((name, "stash", str(i), sha, message))
        ref = f"refs/stac-backup/stash-{i}"
        git(repo, "update-ref", ref, sha, check=True)
        temp.append(ref)
        refs.append(ref)
    for block in git(repo, "worktree", "list", "--porcelain").split("\n\n"):
        fields = dict(l.split(" ", 1) for l in block.splitlines() if " " in l)
        path = fields.get("worktree", "")
        if path and Path(path) != repo and "/.git/" not in path:          # (a submodule reports its own storage)
            if path.startswith(str(WORKSPACE) + os.sep):
                path = os.path.relpath(path, WORKSPACE)                     # (relative to the workspace, as everything here)
            rows.append((name, "worktree", path, fields.get("HEAD", ""), fields.get("branch", "detached").replace("refs/heads/", "")))
    bundle = out_dir / f"{name}.bundle"
    bundle.unlink(missing_ok=True)
    try:
        if refs:
            run = subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), *refs, "--not", *remote_tips],
                                 capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"{repo}: git bundle: {run.stderr.strip()}")
    finally:
        for ref in temp:
            git(repo, "update-ref", "-d", ref)
    return rows


def write_git(out_dir: Path) -> int:
    git_dir = out_dir / "git"
    git_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo in inventory.known_repo_dirs():
        rows += repo_state(repo, git_dir)
    (git_dir / "refs.tsv").write_text("# repo\tkind\tname\tcommit\tupstream | message | branch\n"
                                      + "".join("\t".join(r) + "\n" for r in rows))
    return len(list(git_dir.glob("*.bundle")))


def restore_git(set_dir: Path) -> int:
    """Put back every local branch, stash and worktree named in git/refs.tsv (the repositories are
    already cloned at their commits by the rebuild)."""
    git_dir = set_dir / "git"
    problems = 0
    by_repo = {}
    for line in (git_dir / "refs.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, kind, ref, sha, extra = (line.split("\t") + [""] * 5)[:5]
        by_repo.setdefault(name, []).append((kind, ref, sha, extra))
    for name, rows in by_repo.items():
        repo = WORKSPACE / name.replace("__", "/")
        if not (repo / ".git").exists():
            print(f"  MISSING repository {repo}")
            problems += 1
            continue
        bundle = git_dir / f"{name}.bundle"
        if bundle.exists():
            subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", str(bundle), "+refs/*:refs/stac-backup/*"],
                           capture_output=True, text=True)
        current = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
        remotes = git(repo, "remote").split()
        for kind, ref, sha, extra in rows:
            if kind == "remote":                                            # (the same remotes under the same names)
                if ref not in remotes:
                    git(repo, "remote", "add", ref, extra)
                elif git(repo, "remote", "get-url", ref).strip() != extra:
                    git(repo, "remote", "set-url", ref, extra)
        for kind, ref, sha, extra in rows:
            if kind == "branch":
                if git(repo, "cat-file", "-t", sha).strip() != "commit":
                    print(f"  {name}: branch {ref} needs {sha[:10]}, which neither the remotes nor the bundle hold")
                    problems += 1
                elif ref == current:
                    if git(repo, "rev-parse", "HEAD").strip() != sha:
                        run = subprocess.run(["git", "-C", str(repo), "merge", "--quiet", "--ff-only", sha], capture_output=True, text=True)
                        if run.returncode != 0:
                            print(f"  {name}: {ref} is checked out and cannot be fast-forwarded to {sha[:10]}: {run.stderr.strip()[:120]}")
                            problems += 1
                else:
                    git(repo, "branch", "--quiet", "-f", ref, sha, check=True)
                if extra and git(repo, "rev-parse", "--verify", "--quiet", f"refs/remotes/{extra}").strip():
                    git(repo, "branch", "--quiet", f"--set-upstream-to={extra}", ref)
        have = [l.split("\t")[0] for l in git(repo, "stash", "list", "--format=%H").splitlines()]
        for kind, ref, sha, extra in sorted((r for r in rows if r[0] == "stash"), key=lambda r: -int(r[1])):
            if sha not in have:                                             # (the oldest first: the same order)
                git(repo, "stash", "store", "-m", extra, sha, check=True)
        for kind, path, sha, extra in rows:
            full = str(WORKSPACE / path) if not os.path.isabs(path) else path
            if kind == "worktree" and not Path(full).exists():
                args = ["worktree", "add", "--quiet", full] + ([extra] if extra != "detached" else ["--detach", sha])
                subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        head = [r for r in rows if r[0] == "head"]
        if head and head[0][1] not in ("HEAD", current) and git(repo, "rev-parse", "--verify", "--quiet", head[0][1]).strip() == git(repo, "rev-parse", "HEAD").strip():
            git(repo, "checkout", "--quiet", head[0][1])                    # (on its branch, as on the pod)
        for ref in git(repo, "for-each-ref", "--format=%(refname)", "refs/stac-backup").splitlines():
            git(repo, "update-ref", "-d", ref)
        print(f"  {name}: {sum(r[0] == 'branch' for r in rows)} branch(es), {sum(r[0] == 'stash' for r in rows)} stash(es), "
              f"{sum(r[0] == 'worktree' for r in rows)} worktree(s)")
    return problems


# ── the weights their source may refuse ────────────────────────────────────────────────────────
def write_mirror(out_dir: Path, everything: bool) -> tuple:
    mirror = out_dir / "mirror"
    mirror.mkdir(exist_ok=True)
    n = total = 0
    for row in models.rows():
        if row["sha"] in ("", "-") or row["kind"] in ("link", "copy", "convert"):
            continue
        if not everything and row["kind"] != "dinov3":
            continue
        src = Path(os.path.realpath(models.target(row)))
        dest = mirror / row["sha"]
        if not src.exists():
            continue
        if not dest.exists():
            try:
                os.link(src, dest)
            except OSError:
                import shutil
                shutil.copy2(src, dest)
        n += 1
        total += dest.stat().st_size
    return n, total


# ── make / verify / stale ──────────────────────────────────────────────────────────────────────
def make(out_dir: Path, all_weights: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = inventory.scan()
    big = inventory.large_asis(rows)
    if big:
        for path, size in big:
            print(f"  LARGE FILE WITH NO SOURCE  {size / 1e6:.0f} MB  {path} — give it a row in models.tsv (or ALLOWED_LARGE)")
        return 1
    inventory.write(rows, LOCKS / "inventory.tsv")
    manifest = []
    for name, paths in (("asis", [r[1] for r in rows if r[0] == "asis"]),
                        ("private", [r[1] for r in rows if r[0] == "private"]),
                        ("envs", env_members())):
        n, size = write_archive(out_dir, name, paths)
        manifest.append((f"{name}.tar.gz", sha256_of(out_dir / f"{name}.tar.gz"), str(size), f"{n} members"))
        print(f"  {name}.tar.gz  {n} members, {size / 1e6:.1f} MB")
    bundles = write_git(out_dir)
    for f in sorted((out_dir / "git").iterdir()):
        manifest.append((f"git/{f.name}", sha256_of(f), str(f.stat().st_size), ""))
    print(f"  git/  refs.tsv + {bundles} bundle(s) of commits that are on no remote")
    n, total = write_mirror(out_dir, all_weights)
    print(f"  mirror/  {n} weight(s), {total / 1e9:.2f} GB (hard links)")
    (out_dir / "MANIFEST.tsv").write_text("# file\tsha256\tbytes\tnote\n" + "".join("\t".join(m) + "\n" for m in manifest))
    (out_dir / "README.txt").write_text(
        "STAC Build - the backup set (scripts/rebuild/backup.py). Copy this whole directory OFF the pod.\n"
        "private.tar.gz holds secrets. On a new volume: put it at /workspace/stac_backup and run\n"
        "scripts/rebuild/rebuild.sh - it restores from here what no public source can give back.\n")
    print(f"written: {out_dir}")
    return 0


def members_state(set_dir: Path, name: str):
    """(recorded, now) digests of an archive's members: {path: digest}."""
    recorded, now = {}, {}
    for line in (set_dir / f"{name}.files.tsv").read_text().splitlines():
        rel, digest, _size = (line.split("\t") + ["", ""])[:3]
        recorded[rel] = digest
        full = WORKSPACE / rel.rstrip("/")
        now[rel] = member_digest(full) if (full.exists() or full.is_symlink()) else "MISSING"
    return recorded, now


def verify(set_dir: Path) -> int:
    bad = 0
    for name in ("asis", "envs", "private"):
        recorded, now = members_state(set_dir, name)
        wrong = sorted(p for p in recorded if recorded[p] != now[p])
        bad += len(wrong)
        print(f"  {name}: {len(recorded) - len(wrong)} of {len(recorded)} member(s) on the volume are the set's")
        for p in wrong[:20]:
            print(f"     {'MISSING' if now[p] == 'MISSING' else 'DIFFERS'}  {p}")
    return 1 if bad else 0


def stale(set_dir: Path) -> int:
    """Does the set still describe the pod? Its members against the files of now, and the classes of
    now against the set's (a NEW file of a class the set archives is not in it yet)."""
    if not (set_dir / "MANIFEST.tsv").exists():
        print(f"  no backup set at {set_dir}: run `python3 scripts/rebuild/backup.py make`")
        return 1
    changes = 0
    rows = inventory.scan()
    want = {"asis": set(expand([r[1] for r in rows if r[0] == "asis"])),
            "private": set(expand([r[1] for r in rows if r[0] == "private"])),
            "envs": set(expand(env_members()))}
    for name in ("asis", "envs", "private"):
        recorded, now = members_state(set_dir, name)
        changed = sorted(p for p in recorded if recorded[p] != now[p])
        new = sorted(want[name] - set(recorded))
        changes += len(changed) + len(new)
        for p in changed[:15]:
            print(f"  {name}: {'GONE' if now[p] == 'MISSING' else 'CHANGED'}  {p}")
        for p in new[:15]:
            print(f"  {name}: NEW  {p}")
    # the git work: every local branch, stash and worktree, and the commits that are on no remote
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        write_git(Path(tmp))
        fresh = (Path(tmp) / "git" / "refs.tsv").read_text().splitlines()
        kept = (set_dir / "git" / "refs.tsv").read_text().splitlines() if (set_dir / "git" / "refs.tsv").exists() else []
        for line in sorted(set(fresh) - set(kept))[:15]:
            print(f"  git: NOT IN THE SET  {line[:140]}")
        for line in sorted(set(kept) - set(fresh))[:15]:
            print(f"  git: NO LONGER SO    {line[:140]}")
        changes += len(set(fresh) ^ set(kept))
        heads = lambda d: {b.name: subprocess.run(["git", "bundle", "list-heads", str(b)], capture_output=True, text=True, cwd=str(REPO)).stdout
                           for b in sorted((d / "git").glob("*.bundle"))}
        if heads(Path(tmp)) != heads(set_dir):
            print("  git: the commits that are on no remote are not the ones bundled in the set")
            changes += 1
    print("backup set: it holds the pod as it is now" if not changes else
          f"backup set: {changes} file(s) changed since it was written — run backup.py make again AND copy it off the pod")
    return 1 if changes else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    where = Path(argv[argv.index("--out") + 1]) if "--out" in argv else \
        Path(argv[argv.index("--set") + 1]) if "--set" in argv else Path(os.environ.get("STAC_BACKUP") or DEFAULT_SET)
    if argv[:1] == ["make"]:
        sys.exit(make(where, "--all-weights" in argv))
    if argv[:1] == ["verify"]:
        sys.exit(verify(where))
    if argv[:1] == ["stale"]:
        sys.exit(stale(where))
    if argv[:1] == ["restore-git"]:
        sys.exit(1 if restore_git(where) else 0)
    sys.exit(__doc__)
