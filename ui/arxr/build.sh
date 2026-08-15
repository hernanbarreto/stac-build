#!/bin/bash
# STAC-Builder — bundle the AR/XR phone app (ui/arxr/main.js → static/ar/app.js).
# Uses the ui/ node_modules (three + three-mesh-bvh + esbuild). Re-run after
# editing main.js; the bundle is committed so the server needs no build step.
#
#   bash ui/arxr/build.sh
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
set -e
cd "$(dirname "$0")/.."
export PATH="/workspace/miniforge3/envs/nodejs/bin:$PATH"
./node_modules/.bin/esbuild arxr/main.js \
    --bundle --format=esm --minify \
    --outfile=../static/ar/app.js
echo "built → static/ar/app.js ($(du -h ../static/ar/app.js | cut -f1))"
