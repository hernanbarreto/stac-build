"""THE BACKUP SET: everything of the pod that no public source can give back, as it is — in the repo.

    python3 backup.py make                    write the set (scripts/rebuild/set/) from the pod
    python3 backup.py publish-mirror          upload the weights whose source may refuse (Meta's DINOv3) as the
                                              assets of a DRAFT release of the repo (private to its collaborators)
    python3 backup.py verify                  are the files on this volume the ones of the set? (after a restore)
    python3 backup.py stale                   has the pod changed since the set was written?
    python3 backup.py restore asis|envs|private [--prefix PATH/]     put an archive back (rebuild.sh calls these)
    python3 backup.py restore-git             the local branches, stashes, remotes and worktrees of every repository

USER 2026-11-18: "debe quedar exactamente como está ahora" — and, asked where the set should live:
"Que el kit restaure la base de usuarios y todo lo que se necesita". So the set is PART OF THE KIT: it
is committed in scripts/rebuild/set/ and a `git clone` brings it.

  asis/<group>.tar.gz   the `asis` class of inventory.py — what was COMPILED on the pod (a new compilation is
                        the same source and other bytes: PotreeConverter's sha256 is in every octree's
                        stamp), the built interface, sidecars, empty directories — one archive per vendor
                        tree / area, so that a change rewrites one archive and not all
  envs.tar.gz           inside the conda envs: the distributions compiled on the pod or installed from a
                        directory or a git commit (their files and metadata), and the files no package owns
  private.tar.gz.gpg    the `private` class — .env, the tokens the hub client stored, the sessions' signing
                        key, THE USERS' DATABASE, Tailscale's identity, the assistant's local settings —
                        ENCRYPTED (gpg, AES-256) with STAC_BACKUP_KEY: the repo is public today, and even in
                        a private one tokens are not committed in the clear. The key is in .env (make
                        writes one when there is none) AND MUST BE KEPT OFF THE POD: without it this archive
                        cannot be opened — everything else of the set still can.
  git/                  refs.tsv — every local branch, stash, remote and worktree of every repository;
                        <repo>.bundle — the commits that are on no remote
  mirror.tsv            the weights kept as the assets of the draft release (publish-mirror)
  MANIFEST.tsv          every file of the set with its sha256; *.files.tsv — every member's sha256

An archive over 45 MB is cut in `.part-NN` files (GitHub refuses a file over 100 MB). Every member's path
is relative to the workspace. Not in the set: the sessions (each travels as its own ZIP: the app's
download / upload), runs, logs, scratch. Standard library + git + gpg + curl.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inventory  # noqa: E402
import models  # noqa: E402

WORKSPACE = inventory.WORKSPACE
REPO = inventory.REPO
LOCKS = HERE / "locks"
CONDA = Path(os.environ.get("CONDA_ROOT") or WORKSPACE / "miniforge3")
DEFAULT_SET = HERE / "set"
PART = 45_000_000                       # GitHub warns over 50 MB a file and refuses over 100
ASSET_PART = 1_900_000_000              # a release asset holds at most 2 GiB
KEY_NAME = "STAC_BACKUP_KEY"
RELEASE = "rebuild-assets"              # the draft release that holds the mirrored weights


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


# ── the key and the encryption of the private archive ──────────────────────────────────────────
def env_value(name: str) -> str:
    """A variable from the environment or from the pod's .env (`export NAME=value`)."""
    if os.environ.get(name):
        return os.environ[name]
    for f in (REPO / ".env", WORKSPACE / ".env"):
        if f.is_file():
            for line in f.read_text().splitlines():
                m = re.match(rf"^\s*(?:export\s+)?{name}=(.*)$", line)
                if m:
                    return m.group(1).strip().strip("'\"")
    return ""


def ensure_key() -> str:
    """STAC_BACKUP_KEY — written into .env the first time (256 random bits)."""
    key = env_value(KEY_NAME)
    if key:
        return key
    key = secrets.token_hex(32)
    env = REPO / ".env"
    with open(env, "a") as f:
        f.write("\n# The rebuild kit's key (scripts/rebuild/backup.py): scripts/rebuild/set/private.tar.gz.gpg — the users'\n"
                "# database, this file, Tailscale's identity — is encrypted with it. KEEP A COPY OFF THE POD: without it that\n"
                "# archive cannot be opened on a new volume.\n"
                f"export {KEY_NAME}={key}\n")
    print(f"  A NEW KEY was written to {env} ({KEY_NAME}) — keep a copy of it off the pod")
    return key


def gpg(args, data: bytes, key: str) -> bytes:
    """gpg with the passphrase on its own descriptor and nothing of the caller's keyring touched."""
    with tempfile.TemporaryDirectory() as home:
        os.chmod(home, 0o700)
        r, w = os.pipe()
        os.write(w, key.encode() + b"\n")
        os.close(w)
        try:
            run = subprocess.run(["gpg", "--homedir", home, "--batch", "--yes", "--quiet", "--no-tty", "--pinentry-mode", "loopback",
                                  "--passphrase-fd", str(r), *args], input=data, capture_output=True, pass_fds=(r,))
        finally:
            os.close(r)
    if run.returncode != 0:
        raise RuntimeError("gpg: " + run.stderr.decode(errors="replace").strip()[-300:])
    return run.stdout


def encrypt(data: bytes, key: str) -> bytes:
    return gpg(["--symmetric", "--cipher-algo", "AES256", "--s2k-digest-algo", "SHA512", "--compress-algo", "none", "-o", "-"], data, key)


def decrypt(data: bytes, key: str) -> bytes:
    return gpg(["--decrypt", "-o", "-"], data, key)


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


def tar_bytes_to(fileobj, members) -> list:
    """A deterministic .tar.gz of the members (sorted by the caller; the gzip header without a clock)
    written to ``fileobj``; returns the listing rows (path, digest, bytes)."""
    listing = []
    with gzip.GzipFile(fileobj=fileobj, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in members:
            full = WORKSPACE / rel.rstrip("/")
            tar.add(str(full), arcname=rel.rstrip("/"), recursive=False)
            size = 0 if (full.is_symlink() or full.is_dir()) else full.stat().st_size
            listing.append((rel, member_digest(full), str(size)))
    return listing


def write_archive(path: Path, members) -> list:
    """``path`` (…/name.tar.gz) as one file, or as name.tar.gz.part-NN when it is over PART bytes.
    Earlier files of that name (whole or parts) are replaced. Returns the listing rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in list(path.parent.glob(path.name + ".part-*")) + [path]:
        old.unlink(missing_ok=True)
    with open(path, "wb") as raw:
        listing = tar_bytes_to(raw, members)
    if path.stat().st_size > PART:
        with open(path, "rb") as f:
            for i, block in enumerate(iter(lambda: f.read(PART), b"")):
                path.with_name(f"{path.name}.part-{i:02d}").write_bytes(block)
        path.unlink()
    return listing


def archive_stream(path: Path):
    """The bytes of an archive that is one file or parts, as a readable object."""
    parts = sorted(path.parent.glob(path.name + ".part-*"))
    if parts:
        return io.BytesIO(b"".join(p.read_bytes() for p in parts)) if sum(p.stat().st_size for p in parts) < (1 << 30) \
            else subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE).stdout
    return open(path, "rb")


def extract(stream, prefix: str = "") -> int:
    """Extract a .tar.gz stream into the workspace — only the members under ``prefix`` when given."""
    n = 0
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            if prefix and not (member.name + "/").startswith(prefix):
                continue
            try:
                tar.extract(member, path=str(WORKSPACE), filter="tar")
            except TypeError:                                   # (a python before the `filter` argument)
                tar.extract(member, path=str(WORKSPACE))
            n += 1
    return n


def group_of(rel: str) -> str:
    """The archive an `asis` path goes into: one per vendor tree, per area of the repo, one for the
    rest of the volume."""
    parts = rel.rstrip("/").split("/")
    if parts[0] == inventory.REPO_REL:
        if len(parts) > 2 and parts[1] in ("vendor", "ui"):
            name = f"{parts[1]}-{parts[2]}"
        else:
            name = "repo-" + (parts[1] if len(parts) > 1 else "root")
    else:
        name = "volume"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


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
        # a branch may sit on a commit of a remote this clone has just been given (depth-anything-3's
        # local main is upstream's, and the rebuilt clone's origin was our fork): ask the remotes for it
        if any(kind == "branch" and git(repo, "cat-file", "-t", sha).strip() != "commit" for kind, _r, sha, _e in rows):
            subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", "--all"], capture_output=True, text=True, timeout=900)
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
                elif git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{ref}").strip() != sha:
                    run = subprocess.run(["git", "-C", str(repo), "branch", "--quiet", "-f", ref, sha], capture_output=True, text=True)
                    if run.returncode != 0:                                 # (a branch a worktree holds elsewhere)
                        print(f"  {name}: branch {ref} could not be set to {sha[:10]}: {run.stderr.strip()[:140]}")
                        problems += 1
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


# ── the weights their source may refuse: the assets of a draft release ─────────────────────────
def repo_slug() -> str:
    url = git(REPO, "remote", "get-url", "origin").strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        raise RuntimeError(f"origin is not a GitHub repository: {url}")
    return m.group(1)


def api(method: str, url: str, token: str, data=None, upload: Path = None):
    cmd = ["curl", "-sS", "-X", method, "-H", f"Authorization: Bearer {token}", "-H", "Accept: application/vnd.github+json",
           "-H", "X-GitHub-Api-Version: 2022-11-28", "--retry", "5", "--retry-delay", "5"]
    if upload is not None:                                  # (-T streams the file; -d would read 1.9 GB into memory)
        cmd += ["-H", "Content-Type: application/octet-stream", "-T", str(upload)]
    elif data is not None:
        cmd += ["-d", json.dumps(data)]
    run = subprocess.run(cmd + [url], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"{method} {url.split('?')[0]}: {run.stderr.strip()[-200:]}")
    return json.loads(run.stdout) if run.stdout.strip() else {}


def release(token: str, create: bool):
    """The draft release that holds the mirror (a draft has no tag and is seen only by who may push)."""
    slug = repo_slug()
    for page in range(1, 6):
        found = api("GET", f"https://api.github.com/repos/{slug}/releases?per_page=100&page={page}", token)
        if not isinstance(found, list) or not found:
            break
        for r in found:
            if r.get("name") == RELEASE:
                return r
    if not create:
        return None
    return api("POST", f"https://api.github.com/repos/{slug}/releases", token,
               {"tag_name": RELEASE, "name": RELEASE, "draft": True, "prerelease": True,
                "body": "STAC Build - scripts/rebuild: the weights whose source may refuse (scripts/rebuild/set/mirror.tsv). "
                        "A DRAFT on purpose: its assets are visible only to the repository's collaborators."})


def mirror_rows(set_dir: Path):
    f = set_dir / "mirror.tsv"
    return [l.split("\t") for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")] if f.exists() else []


def publish_mirror(set_dir: Path) -> int:
    """Upload every `dinov3` weight of models.tsv that the draft release does not hold yet, in parts
    of at most ASSET_PART bytes named <sha256>.part-NN; write set/mirror.tsv."""
    token = env_value("GIT_TOKEN")
    if not token:
        print("  no GIT_TOKEN (.env): the mirror cannot be published")
        return 1
    rel = release(token, create=True)
    if "id" not in rel:
        print(f"  the release could not be read or made: {str(rel)[:200]}")
        return 1
    have = {a["name"]: a for a in rel.get("assets", [])}
    slug = repo_slug()
    rows = []
    for row in models.rows():
        if row["kind"] != "dinov3":
            continue
        src = Path(os.path.realpath(models.target(row)))
        if not src.exists():
            print(f"  not on the pod: {row['dest']}")
            return 1
        n = -(-row["size"] // ASSET_PART)
        for i in range(n):
            name = f"{row['sha']}.part-{i:02d}"
            length = min(ASSET_PART, row["size"] - i * ASSET_PART)
            if name in have and have[name].get("size") == length and have[name].get("state") == "uploaded":
                print(f"  already in the release: {name}")
            else:
                if name in have:                                   # (a broken earlier upload)
                    api("DELETE", f"https://api.github.com/repos/{slug}/releases/assets/{have[name]['id']}", token)
                with tempfile.TemporaryDirectory() as tmp:
                    part = Path(tmp) / name
                    with open(src, "rb") as f, open(part, "wb") as out:
                        f.seek(i * ASSET_PART)
                        left = length
                        while left:
                            block = f.read(min(1 << 24, left))
                            out.write(block)
                            left -= len(block)
                    done = api("POST", f"https://uploads.github.com/repos/{slug}/releases/{rel['id']}/assets?name={name}", token, upload=part)
                if done.get("state") != "uploaded" or done.get("size") != length:
                    print(f"  upload of {name} failed: {str(done)[:200]}")
                    return 1
                print(f"  uploaded {name} ({length / 1e9:.2f} GB)")
            rows.append((row["sha"], str(row["size"]), name, str(length), row["dest"]))
    (set_dir / "mirror.tsv").write_text(
        f"# the weights kept as the assets of the DRAFT release '{RELEASE}' of {slug} (backup.py publish-mirror);\n"
        "# a rebuild takes them from there with GIT_TOKEN when their own source refuses (models.py)\n"
        "# sha256\tbytes\tasset\tasset bytes\tthe file\n" + "".join("\t".join(r) + "\n" for r in rows))
    print(f"  mirror.tsv: {len(rows)} asset(s) of {len({r[0] for r in rows})} weight(s)")
    return 0


def fetch_from_release(sha: str, dest: Path, set_dir: Path = DEFAULT_SET) -> bool:
    """The weight of that sha256 from the draft release into ``dest``. False when it is not mirrored
    (the caller goes on to the weight's own source); a mirror that holds it and cannot give it —
    no token, the release gone, a download cut — RAISES with the reason: it must not be read as
    "not mirrored"."""
    rows = [r for r in mirror_rows(set_dir) if r[0] == sha]
    if not rows:
        return False
    token = env_value("GIT_TOKEN")
    if not token:
        raise RuntimeError("the weight is mirrored in the repo's draft release and GIT_TOKEN is not set (.env)")
    rel = release(token, create=False)
    if not rel:
        raise RuntimeError(f"the draft release '{RELEASE}' of {repo_slug()} was not found with this token")
    assets = {a["name"]: a for a in rel.get("assets", [])}
    with open(dest, "wb") as out:
        for row in sorted(rows, key=lambda r: r[2]):
            asset = assets.get(row[2])
            if asset is None:
                raise RuntimeError(f"the release lacks the asset {row[2]}")
            run = subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "5", "-H", f"Authorization: Bearer {token}",
                                  "-H", "Accept: application/octet-stream", asset["url"]], stdout=out, stderr=subprocess.PIPE)
            if run.returncode != 0:
                raise RuntimeError(f"download of {row[2]} from the release: {run.stderr.decode(errors='replace').strip()[-200:]}")
    return True


# ── make ───────────────────────────────────────────────────────────────────────────────────────
def manifest_of(set_dir: Path) -> dict:
    f = set_dir / "MANIFEST.tsv"
    rows = {}
    for line in f.read_text().splitlines() if f.exists() else []:
        if line.strip() and not line.startswith("#"):
            name, sha, size, note = (line.split("\t") + [""] * 4)[:4]
            rows[name] = (sha, size, note)
    return rows


def private_plain(members) -> tuple:
    """(the deterministic tar.gz of the private members, its listing as text)."""
    buf = io.BytesIO()
    listing = tar_bytes_to(buf, members)
    return buf.getvalue(), "".join("\t".join(r) + "\n" for r in listing)


def make(set_dir: Path) -> int:
    set_dir.mkdir(parents=True, exist_ok=True)
    key = ensure_key()                                  # (before the scan: .env is one of the private files)
    rows = inventory.scan()
    big = inventory.large_asis(rows)
    if big:
        for path, size in big:
            print(f"  LARGE FILE WITH NO SOURCE  {size / 1e6:.0f} MB  {path} — give it a row in models.tsv (or ALLOWED_LARGE)")
        return 1
    inventory.write(rows, LOCKS / "inventory.tsv")
    before = manifest_of(set_dir)

    # asis: one archive per group
    groups = {}
    for rel in expand([r[1] for r in rows if r[0] == "asis"]):
        groups.setdefault(group_of(rel), []).append(rel)
    asis_dir = set_dir / "asis"
    asis_dir.mkdir(exist_ok=True)
    listing = []
    for name in sorted(groups):
        for rel, digest, size in write_archive(asis_dir / f"{name}.tar.gz", groups[name]):
            listing.append(f"{rel}\t{digest}\t{size}\t{name}")
    for stale_file in asis_dir.iterdir():                # (a group that no longer exists)
        if stale_file.name.split(".tar.gz")[0] not in groups:
            stale_file.unlink()
    (set_dir / "asis.files.tsv").write_text("".join(f"{line}\n" for line in listing))
    total = sum(f.stat().st_size for f in asis_dir.iterdir())
    print(f"  asis/  {len(groups)} archive(s), {len(listing)} members, {total / 1e6:.1f} MB")

    # envs
    env_listing = write_archive(set_dir / "envs.tar.gz", expand(env_members()))
    (set_dir / "envs.files.tsv").write_text("".join("\t".join(r) + "\n" for r in env_listing))
    print(f"  envs.tar.gz  {len(env_listing)} members")

    # private: encrypted; written again only when its content changed (a new encryption of the same
    # bytes is another file: the repo would change at every capture)
    plain, plain_listing = private_plain(expand([r[1] for r in rows if r[0] == "private"]))
    plain_sha = hashlib.sha256(plain).hexdigest()
    kept = before.get("private.tar.gz.gpg", ("", "", ""))[2] == f"plain={plain_sha}" and (set_dir / "private.tar.gz.gpg").exists() \
        and (set_dir / "private.files.tsv.gpg").exists()
    if kept:
        try:                                            # (and the key is still the one that opens it)
            kept = hashlib.sha256(decrypt((set_dir / "private.tar.gz.gpg").read_bytes(), key)).hexdigest() == plain_sha
        except RuntimeError:
            kept = False
    if not kept:
        (set_dir / "private.tar.gz.gpg").write_bytes(encrypt(plain, key))
        (set_dir / "private.files.tsv.gpg").write_bytes(encrypt(plain_listing.encode(), key))
    print(f"  private.tar.gz.gpg  {len(plain_listing.splitlines())} members, encrypted with {KEY_NAME}"
          + (" (unchanged)" if kept else ""))

    bundles = write_git(set_dir)
    print(f"  git/  refs.tsv + {bundles} bundle(s) of commits that are on no remote")

    manifest = []
    for f in sorted(p for p in set_dir.rglob("*") if p.is_file() and p.name != "MANIFEST.tsv"):
        rel = f.relative_to(set_dir).as_posix()
        note = f"plain={plain_sha}" if rel == "private.tar.gz.gpg" else ""
        manifest.append((rel, sha256_of(f), str(f.stat().st_size), note))
    (set_dir / "MANIFEST.tsv").write_text("# file\tsha256\tbytes\tnote\n" + "".join("\t".join(m) + "\n" for m in manifest))
    mirrored = {r[0] for r in mirror_rows(set_dir)}
    missing = [r["dest"] for r in models.rows() if r["kind"] == "dinov3" and r["sha"] not in mirrored]
    for dest in missing:
        print(f"  NOT MIRRORED: {dest} — python3 scripts/rebuild/backup.py publish-mirror")
    print(f"written: {set_dir} ({sum(int(m[2]) for m in manifest) / 1e6:.1f} MB in {len(manifest)} files)")
    return 0


# ── restore / verify / stale ───────────────────────────────────────────────────────────────────
def restore(kind: str, set_dir: Path, prefix: str = "") -> int:
    if kind == "asis":
        n = 0
        for name in sorted({p.name.split(".tar.gz")[0] for p in (set_dir / "asis").iterdir()}):
            n += extract(archive_stream(set_dir / "asis" / f"{name}.tar.gz"), prefix)
        print(f"asis: {n} member(s) restored")
    elif kind == "envs":
        print(f"envs: {extract(archive_stream(set_dir / 'envs.tar.gz'), prefix)} member(s) restored" + (f" under {prefix}" if prefix else ""))
    elif kind == "private":
        key = env_value(KEY_NAME)
        if not key:
            print(f"private: {KEY_NAME} is not set (export it, or write it in .env) — the users' database, .env and the rest "
                  f"of the private archive are NOT restored")
            return 1
        n = extract(io.BytesIO(decrypt((set_dir / "private.tar.gz.gpg").read_bytes(), key)))
        print(f"private: {n} member(s) restored (names not printed)")
    else:
        raise SystemExit(f"unknown archive {kind!r}")
    return 0


def state_of(lines) -> tuple:
    """(recorded, now) digests of listed members: {path: digest}."""
    recorded, now = {}, {}
    for line in lines:
        rel, digest = line.split("\t")[:2]
        recorded[rel] = digest
        full = WORKSPACE / rel.rstrip("/")
        now[rel] = member_digest(full) if (full.exists() or full.is_symlink()) else "MISSING"
    return recorded, now


def listings(set_dir: Path) -> dict:
    out = {"asis": (set_dir / "asis.files.tsv").read_text().splitlines(),
           "envs": (set_dir / "envs.files.tsv").read_text().splitlines()}
    key = env_value(KEY_NAME)
    if key and (set_dir / "private.files.tsv.gpg").exists():
        try:
            out["private"] = decrypt((set_dir / "private.files.tsv.gpg").read_bytes(), key).decode().splitlines()
        except RuntimeError:
            pass
    return out


def verify(set_dir: Path) -> int:
    bad = 0
    lists = listings(set_dir)
    for name in ("asis", "envs", "private"):
        if name not in lists:
            print(f"  {name}: not checked — {KEY_NAME} is not set or does not open the set's list")
            bad += 1
            continue
        recorded, now = state_of(lists[name])
        wrong = sorted(p for p in recorded if recorded[p] != now[p])
        bad += len(wrong)
        print(f"  {name}: {len(recorded) - len(wrong)} of {len(recorded)} member(s) on the volume are the set's")
        for p in wrong[:20]:
            print(f"     {'MISSING' if now[p] == 'MISSING' else 'DIFFERS'}  {p if name != 'private' else '(a private file)'}")
    return 1 if bad else 0


def stale(set_dir: Path) -> int:
    """Does the set still describe the pod? Its members against the files of now, the classes of now
    against the set's (a NEW file of a class the set archives is not in it yet), the private archive's
    content, the git work, the mirror."""
    if not (set_dir / "MANIFEST.tsv").exists():
        print(f"  no backup set at {set_dir}: run `python3 scripts/rebuild/backup.py make`")
        return 1
    changes = 0
    rows = inventory.scan()
    want = {"asis": set(expand([r[1] for r in rows if r[0] == "asis"])), "envs": set(expand(env_members()))}
    lists = listings(set_dir)
    for name in ("asis", "envs"):
        recorded, now = state_of(lists[name])
        changed = sorted(p for p in recorded if recorded[p] != now[p])
        new = sorted(want[name] - set(recorded))
        changes += len(changed) + len(new)
        for p in changed[:15]:
            print(f"  {name}: {'GONE' if now[p] == 'MISSING' else 'CHANGED'}  {p}")
        for p in new[:15]:
            print(f"  {name}: NEW  {p}")
    plain, _listing = private_plain(expand([r[1] for r in rows if r[0] == "private"]))
    if manifest_of(set_dir).get("private.tar.gz.gpg", ("", "", ""))[2] != f"plain={hashlib.sha256(plain).hexdigest()}":
        print("  private: its files changed since the set was written (the users' database, .env …)")
        changes += 1
    with tempfile.TemporaryDirectory() as tmp:
        write_git(Path(tmp))
        fresh = (Path(tmp) / "git" / "refs.tsv").read_text().splitlines()
        kept = (set_dir / "git" / "refs.tsv").read_text().splitlines() if (set_dir / "git" / "refs.tsv").exists() else []
        main_repo = inventory.REPO_REL + "\t"
        # (the repo's own branch moves with every commit — of this very set: `git status` and the push speak for it)
        own = lambda l: l.startswith(main_repo) and l.split("\t")[1] in ("head", "branch") and l.split("\t")[2] in (
            "HEAD", git(REPO, "rev-parse", "--abbrev-ref", "HEAD").strip())
        fresh_c, kept_c = {l for l in fresh if not own(l)}, {l for l in kept if not own(l)}
        for line in sorted(fresh_c - kept_c)[:15]:
            print(f"  git: NOT IN THE SET  {line[:140]}")
        for line in sorted(kept_c - fresh_c)[:15]:
            print(f"  git: NO LONGER SO    {line[:140]}")
        changes += len(fresh_c ^ kept_c)
    mirrored = {r[0] for r in mirror_rows(set_dir)}
    for r in models.rows():
        if r["kind"] == "dinov3" and r["sha"] not in mirrored:
            print(f"  mirror: {r['dest']} is not in the release — python3 scripts/rebuild/backup.py publish-mirror")
            changes += 1
    print("backup set: it holds the pod as it is now" if not changes else
          f"backup set: {changes} difference(s) — bash scripts/rebuild/capture.sh, then commit scripts/rebuild/")
    return 1 if changes else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    where = Path(argv[argv.index("--set") + 1]) if "--set" in argv else Path(os.environ.get("STAC_BACKUP") or DEFAULT_SET)
    prefix = argv[argv.index("--prefix") + 1] if "--prefix" in argv else ""
    if argv[:1] == ["make"]:
        sys.exit(make(where))
    if argv[:1] == ["publish-mirror"]:
        sys.exit(publish_mirror(where))
    if argv[:1] == ["verify"]:
        sys.exit(verify(where))
    if argv[:1] == ["stale"]:
        sys.exit(stale(where))
    if argv[:1] == ["restore"] and len(argv) > 1:
        sys.exit(restore(argv[1], where, prefix))
    if argv[:1] == ["restore-git"]:
        sys.exit(1 if restore_git(where) else 0)
    sys.exit(__doc__)
