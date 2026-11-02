#!/bin/bash
# The ktx tool tile_mesh.mjs encodes the textures with (KTX2: Basis UASTC + zstd): KhronosGroup/KTX-Software 4.4.2
# (Apache-2.0), the Linux x86_64 release, its sha1 checked against the release's own, unpacked into vendor/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
V=4.4.2; NAME=KTX-Software-$V-Linux-x86_64
U=https://github.com/KhronosGroup/KTX-Software/releases/download/v$V
[ -x "$REPO/vendor/$NAME/bin/ktx" ] && { echo "already there: $REPO/vendor/$NAME/bin/ktx"; exit 0; }
DL=$(mktemp -d)
curl -sSL -o "$DL/$NAME.tar.bz2" "$U/$NAME.tar.bz2"
curl -sSL -o "$DL/sum.sha1" "$U/$NAME.tar.bz2.sha1"
want=$(cut -d' ' -f1 "$DL/sum.sha1"); got=$(sha1sum "$DL/$NAME.tar.bz2" | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "sha1 mismatch: $got, the release says $want"; exit 1; }
tar -xjf "$DL/$NAME.tar.bz2" -C "$REPO/vendor" --no-same-owner
rm -rf "$DL"
LD_LIBRARY_PATH="$REPO/vendor/$NAME/lib" "$REPO/vendor/$NAME/bin/ktx" --version
