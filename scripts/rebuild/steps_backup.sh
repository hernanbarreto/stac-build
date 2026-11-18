#!/usr/bin/env bash
# STAC Build — REBUILD, the backup set: what no public source can give back, put back as it was.
# Sourced by rebuild.sh and step.sh.
#
# The set (scripts/rebuild/backup.py make) is looked for at $STAC_BACKUP, default /workspace/stac_backup.
# WITH it the rebuilt pod holds the same bytes as the old one in everything compiled on it (no
# compilation is run), its secrets, its local git branches and stashes. WITHOUT it the rebuild still
# completes from public sources: the compiled parts are compiled again (the same sources, other
# bytes), and it says so in its summary.

BACKUP_SET="${STAC_BACKUP:-$WORKSPACE/stac_backup}"

have_backup() { [ -f "$BACKUP_SET/MANIFEST.tsv" ]; }

# are the compiled parts taken from the set? Yes with the set — unless COMPILE=1: the set's binaries were
# built for the pod's card (an A100, sm_80); on another architecture the CUDA extensions are compiled
# again (the secrets, the local git work and the rest of the set are still restored).
use_prebuilt() { have_backup && [ "${COMPILE:-0}" != 1 ]; }

# every archive of the set against its recorded sha256 — a set damaged in its copy is refused whole
step_backup_check() {
    local file sha size note got
    while IFS=$'\t' read -r file sha size note; do
        case "$file" in \#*|"") continue ;; esac
        [ -f "$BACKUP_SET/$file" ] || { echo "the set lacks $file"; return 1; }
        got="$(sha256sum "$BACKUP_SET/$file" | cut -d' ' -f1)"
        [ "$got" = "$sha" ] || { echo "$file: sha256 $got, the set's manifest says $sha"; return 1; }
        echo "ok  $file"
    done < "$BACKUP_SET/MANIFEST.tsv"
}

# the secrets and the personal state: .env first of all (the tokens every later step needs)
step_restore_private() {
    tar -xzf "$BACKUP_SET/private.tar.gz" -C "$WORKSPACE"
    echo "$(wc -l < "$BACKUP_SET/private.files.tsv") member(s) restored (names not printed: private)"
}

# what was compiled, generated or copied by hand outside the envs — after the vendor trees are cloned
step_restore_asis() {
    tar -xzf "$BACKUP_SET/asis.tar.gz" -C "$WORKSPACE"
    echo "$(wc -l < "$BACKUP_SET/asis.files.tsv") member(s) restored"
}

# inside one env: the distributions compiled on the old pod and the files no package owns — after
# the env's pip step, INSTEAD of compiling. Only the envs that exist are touched.
restore_env_archive() {
    local env="$1" rel
    rel="$(realpath --relative-to="$WORKSPACE" "$CONDA_ROOT/envs/$env")"
    if ! grep -q "^$rel/" "$BACKUP_SET/envs.files.tsv"; then echo "$env: nothing of it in the set"; return 0; fi
    tar -xzf "$BACKUP_SET/envs.tar.gz" -C "$WORKSPACE" --wildcards "$rel/*"
    echo "$env: $(grep -c "^$rel/" "$BACKUP_SET/envs.files.tsv") member(s) restored from the set (nothing compiled)"
}

# every local branch, stash and worktree of every repository
step_restore_git() { python3 "$REBUILD_DIR/backup.py" restore-git --set "$BACKUP_SET"; }

# the files on the volume against the set's lists, member by member
step_backup_verify() { python3 "$REBUILD_DIR/backup.py" verify --set "$BACKUP_SET"; }
