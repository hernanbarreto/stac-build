#!/usr/bin/env bash
# STAC Build — REBUILD: ONE step, in its own process, where `set -e` stops at the first failed command.
#   bash step.sh <step function> [args…]          (rebuild.sh's run_step calls it; its output is the step's log)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
source "$HERE/steps_vendors.sh"
source "$HERE/steps_envs.sh"
source "$HERE/steps_builds.sh"
source "$HERE/steps_main.sh"
load_env || true
mkdir -p "$CACHE"
"$@"
