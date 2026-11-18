#!/usr/bin/env bash
# STAC Build — REBUILD, the backup set: what no public source can give back, put back as it was.
# Sourced by rebuild.sh and step.sh.
#
# The set is PART OF THE KIT: scripts/rebuild/set/, committed (backup.py make). With it the rebuilt pod holds
# the same bytes as the old one in everything compiled on it (no compilation is run), its local git branches
# and stashes — and, WHEN STAC_BACKUP_KEY IS GIVEN, its secrets and the users' database (private.tar.gz.gpg).

BACKUP_SET="${STAC_BACKUP:-$REBUILD_DIR/set}"

have_backup() { [ -f "$BACKUP_SET/MANIFEST.tsv" ]; }

# are the compiled parts taken from the set? Yes with the set — unless COMPILE=1: the set's binaries were
# built for the pod's card (an A100, sm_80); on another architecture the CUDA extensions are compiled
# again (the secrets, the local git work and the rest of the set are still restored).
use_prebuilt() { have_backup && [ "${COMPILE:-0}" != 1 ]; }

# every file of the set against its recorded sha256 — a set damaged in its copy is refused whole
step_backup_check() {
    local file sha size note got
    while IFS=$'\t' read -r file sha size note; do
        case "$file" in \#*|"") continue ;; esac
        [ -f "$BACKUP_SET/$file" ] || { echo "the set lacks $file"; return 1; }
        got="$(sha256sum "$BACKUP_SET/$file" | cut -d' ' -f1)"
        [ "$got" = "$sha" ] || { echo "$file: sha256 $got, the set's manifest says $sha"; return 1; }
    done < "$BACKUP_SET/MANIFEST.tsv"
    echo "$(grep -vc '^#' "$BACKUP_SET/MANIFEST.tsv") file(s) of the set, each with its recorded sha256"
    command -v gpg >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gnupg; }
}

# the secrets and the personal state — .env first of all (the tokens every later step needs), the users'
# database, the sessions' signing key, Tailscale's identity. Encrypted: STAC_BACKUP_KEY opens it.
step_restore_private() { python3 "$REBUILD_DIR/backup.py" restore private --set "$BACKUP_SET"; }

# what was compiled, generated or copied by hand outside the envs — after the vendor trees are cloned
step_restore_asis() { python3 "$REBUILD_DIR/backup.py" restore asis --set "$BACKUP_SET"; }

# inside one env: the distributions compiled on the old pod and the files no package owns — after the env's
# pip step, INSTEAD of compiling. Only the env named is touched.
restore_env_archive() {
    local rel
    rel="$(realpath --relative-to="$WORKSPACE" "$CONDA_ROOT/envs/$1")"
    python3 "$REBUILD_DIR/backup.py" restore envs --set "$BACKUP_SET" --prefix "$rel/"
}

# every local branch, stash, remote and worktree of every repository
step_restore_git() { python3 "$REBUILD_DIR/backup.py" restore-git --set "$BACKUP_SET"; }

# the files on the volume against the set's lists, member by member
step_backup_verify() { python3 "$REBUILD_DIR/backup.py" verify --set "$BACKUP_SET"; }
