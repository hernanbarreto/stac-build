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
./node_modules/.bin/esbuild arxr/ios.js \
    --bundle --format=esm --minify \
    --outfile=../static/ar/ios-app.js
# cache-bust: stamp the content hash into the HTML — a stale cached bundle
# against fresh HTML took the whole app down with a TypeError once
HASH=$(md5sum ../static/ar/app.js | cut -c1-10)
sed -i -E "s/app\.js\?v=[a-z0-9]+/app.js?v=${HASH}/" ../static/ar/index.html
IHASH=$(md5sum ../static/ar/ios-app.js | cut -c1-10)
sed -i -E "s/ios-app\.js\?v=[a-z0-9]+/ios-app.js?v=${IHASH}/" ../static/ar/ios.html
# provision the 8th Wall engine binary (git-ignored) if missing/stale
if [ ! -f ../static/ar/xr8/xr.js ] \
   || ! cmp -s node_modules/@8thwall/engine-binary/dist/xr.js ../static/ar/xr8/xr.js; then
    mkdir -p ../static/ar/xr8
    cp -r node_modules/@8thwall/engine-binary/dist/* ../static/ar/xr8/
    echo "provisioned 8th Wall engine → static/ar/xr8/"
fi
echo "built → app.js v=${HASH}, ios-app.js v=${IHASH}"
