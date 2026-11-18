# Rebuilding the pod on an empty volume — exactly as it is

USER 2026-11-18 (RunPod announced a maintenance of the pod's server): *"Prepara todo en .sh para que se creen
automáticamente los entornos se descarguen todos los modelos y se vuelva a armar en un nuevo volumen en cuestión
de poco tiempo … de forma completa y sin errores"* — and, on the first version: *"debe quedar exactamente como
está ahora … No decidas por mí … cada cosa que avancemos debemos actualizarlo para que siempre quede actualizado
a la perfección … Revisa pointdit r3d etc etc son muchos"*.

So this is not a list of things to install. It starts from **what is on the volume** — every file, classified
(`inventory.py`) — and each class has the one thing that puts it back:

| class | what | how it comes back | exact? |
|---|---|---|---|
| `git` | the repo, 13 vendor repositories (submodules and nested ones) | the commit of each (`steps_vendors.sh`), our patches | the same commit and diff |
| `env` | the ten conda envs, conda's base, the USD venv | `locks/<env>.conda-explicit.txt` (URL + md5), `locks/<env>.wheels.tsv` (URL + sha256) | the same package files |
| `weight` | 99 model files, 82 GB, in the repo's trees and in the TWO hub caches | `models.tsv`: source, pinned revision, sha256 | sha256-checked |
| `download` | KTX, CloudComPy's binary, Tailscale, cloudflared, Miniforge's installer, MapAnything's DINOv2 code | their pinned sources | held to the recorded sha256 |
| `npm` | `node_modules` (interface, mesh tools), the global CLI | `npm ci` on the tracked lock | npm's integrity hashes |
| `asis` | what exists nowhere else: everything COMPILED on the pod, the built interface, sidecars, empty directories | **the backup set** — archived as it is, put back byte for byte | the same bytes |
| `private` | `.env`, stored tokens, the sessions' signing key, the users' database, Tailscale's identity | **the backup set**'s private archive | the same bytes |
| `data` | the sessions (147 GB), runs, logs, scratch | copied apart — it is not software | — |
| `cache` | bytecode, pip's cache, derived images | whatever uses it writes it again | — |

## The two halves

**The kit** — this directory, in git: the scripts, the locks, the manifest, the inventory. Public sources only.

**The backup set** — `python3 scripts/rebuild/backup.py make` writes it to `/workspace/stac_backup` (≈ 0.2 GB plus
hard links to Meta's two DINOv3 files, 4.6 GB): `asis.tar.gz`, `envs.tar.gz` (the distributions compiled on the
pod, inside the envs), `private.tar.gz`, `git/` (every local branch, stash and worktree of every repository; the
commits that are on no remote, as bundles), `mirror/` (the weights whose source may refuse). **It is not a
backup while it sits on the volume it describes: it must be copied off the pod, and again after every capture.**
`private.tar.gz` holds secrets.

With the set, a rebuild compiles nothing: the compiled parts are the old pod's own bytes (PotreeConverter's
sha256 is in every octree's stamp — a new compilation would be the same program and another stamp). Without it
the rebuild still completes from public sources, compiles those parts again, restores no secret and no local git
work, and says so.

## Run it

A pod of the same template (Ubuntu 24.04, CUDA 12.8.1 **devel**, pythons 3.10–3.13 in `/usr/bin`; by what the pod
shows it is RunPod's PyTorch 2.8 / CUDA 12.8.1 image — copy its exact name from the pod's page before deleting
anything), the volume at `/workspace`:

```bash
git clone https://github.com/hernanbarreto/stac-build.git /workspace/stac-build
# copy the backup set to /workspace/stac_backup      (without it: write /workspace/stac-build/.env — env.template)
bash /workspace/stac-build/scripts/rebuild/rebuild.sh
bash /workspace/stac-build/init_pod.sh               # the services, as on every boot
```

About an hour (measured step by step on the pod's local disk: an env's conda packages 4–50 s, its pip packages
80–160 s, the vendor clones 5 min, 82 GB of weights at the pod's 100+ MB/s; a network volume writes slower). A
failure stops only what depends on it; **run the same command again** and every finished step is skipped.
`rebuild.sh --list` shows the steps, `--only STEP` runs one, `--verify` checks an installed pod, `ENVS="da3 voice"`
rebuilds some envs. Logs: `/workspace/.stac_rebuild/logs/<step>.log`.

Running it on the pod that already works changes nothing: an env it did not build is left alone, a vendor tree at
its commit and a weight with the right sha256 are kept.

## Keeping it true — after EVERY change of the pod

```bash
python3 scripts/rebuild/drift.py          # is the kit still the pod? (≈ 4 min, read-only; --fast: seconds)
bash scripts/rebuild/capture.sh           # write the locks, the manifest's cache rows, the inventory, the set
git add scripts/rebuild && git commit     # … and copy /workspace/stac_backup off the pod; push on the user's word
```

A `pip install`, a `conda install`, a vendor clone or commit, a new weight, a compilation, a new secret: each
makes the kit false without any error. `drift.py` compares the live pod with the kit part by part — every env's
packages and files, every vendor's commit and diff, every weight, every file of the volume in its class, the
backup set, what is not committed or pushed — and names each difference. `server/tests/test_rebuild_kit.py` runs
the fast half in the suite. What a capture cannot guess, it names:
- a file edited in place inside an env (`locks/<env>.modified.tsv`) → save it under `overlays/<env>/`;
- a large file with no source (a weight downloaded by hand) → give it a row in `models.tsv`;
- a commit made in a vendor clone → it goes into the set's bundle; pushing it to a remote of ours is better.

## What a rebuild cannot give back, and what is different

- **The sessions and the other `data`** are not in the set. `python3 scripts/rebuild/inventory.py summary` lists
  them with their sizes.
- **Meta's two DINOv3 checkpoints** (PointDiT, GenRecon): Meta's link in `.env` expires (it had on 2026-11-18).
  The set's `mirror/` holds them; without the set a new link is needed (`env.template` says where).
- **The USD venv** (`/workspace/usdtools`, the AR export): on the pod of 2026-11-18 it does NOT work — it was made
  when `/usr/bin/python3` was 3.11 and the image moved it to 3.12 (`import pxr` fails). The rebuild makes it with
  python 3.11 by name, so the rebuilt one works: the one place where the rebuilt pod is not the old one.
- An installer's own records of an install (conda-meta's dates, a distribution's RECORD of an editable install)
  are written again; `compare_env.py` counts them apart from the files that matter.

## What was checked on 2026-11-18, and what was not

In a scratch volume on the pod's local disk, at a path as long as `/workspace` (conda and pip then write the same
launchers and the same padding as on the real volume), the live install read and never written — but for one
mistake, below:

- **the packages**: every pip package of the nine python envs and the USD venv (1,065) downloaded and compared
  with the installed one — the same files byte for byte, but for what is compiled on the pod (in the set); the
  1,080 conda packages still published; a fresh Miniforge 26.3.2-2 gives the locked base env (89 packages);
- **the ten envs, rebuilt one at a time by `rebuild.sh` with the set, compared FILE BY FILE with the live ones**
  (`compare_env.py` — identical / the env's own path written in the file / the installers' bookkeeping / different):

  | env | identical | path-only | bookkeeping | DIFFERENT |
  |---|---|---|---|---|
  | da3 | 65,872 | 474 | 120 | 0 |
  | mapanything | 43,329 | 262 | 248 | 0 |
  | semantic | 48,313 | 265 | 375 | 0 |
  | voice | 31,805 | 257 | 251 | 0 |
  | amb3r | 72,178 | 1,304 | 450 | 0 |
  | genrecon | 55,462 | 658 | 438 | 0 |
  | CloudComPy310 | 87,873 | 1,367 | 563 | 0 |
  | colmap | 33,560 | 1,093 | 275 | 0 |
  | gs | 26,281 | 218 | 105 | 0 |
  | nodejs | 2,522 | 14 | 14 | 0 |

  no file on one side only; the compiled extensions restored from the set in 1–4 s an env, nothing compiled;
- **the repositories**: the repo and its 13 vendor repositories (submodules and nested ones included) — the same
  commit, uncommitted diff, branches (29 in the repo), stashes, remotes, worktrees;
- **the set**: its 4,293 `asis` members and 20 `private` members restored and equal to its lists;
- **the volume**: `inventory.py check` on the rebuilt volume against the lock — no difference in any class of
  software (git heads, downloads by sha256, asis, private, npm); only the data, the caches and the weights not
  fetched in this test were absent;
- **the weights**: the 99 entries verified on the pod; 65 of the 67 sources answer with the manifest's size and
  sha256 (the two others are DINOv3's: fetched from the set's mirror in the test); one file of every kind of
  source downloaded and identical, a hub cache tree rebuilt identical;
- **without the set** (the public-sources path): da3, mapanything, voice, amb3r, genrecon, colmap and nodejs built
  with their compiled extensions and verified; the rebuilt PotreeConverter gives the same `octree.bin`,
  `hierarchy.bin` and `metadata.json` as the pod's on bufferStop's cloud; the backend imports from a fresh clone.

NOT done: one single run on a really empty volume (the pod has no room for a second copy of itself: the envs were
rebuilt one at a time, the 82 GB of weights not downloaded again in full); nothing was started on the card from a
rebuilt env.

THE MISTAKE: the conversion of GenRecon's DINOv3 (`server/tools/genrecon/convert_dinov3.py`) had its paths
written as `/workspace/stac-build/…`; run from the scratch tree it REWROTE THE LIVE POD'S FILE, twice — with the
same bytes (its sha256 is the manifest's). The script now takes its paths from its own repo, and `models.py`
relocates every absolute path of the manifest under `$WORKSPACE`: a rebuild into another root cannot write to
`/workspace`.

The older scripts (`scripts/setup_vendors.sh`, `scripts/setup_pod_envs.sh`, `setup_weights.sh`, `export_envs.sh`,
`scripts/setup.sh`, `docs/migration/`) predate this and name removed envs and old pins.
