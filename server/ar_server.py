# STAC-Builder — standalone AR/XR server (phone surface, OWN process).
#
# Serves ONLY the mobile XR viewer, fully decoupled from the main backend:
# restarting/deploying this never touches the reconstruction pipeline or the
# desktop viewer (which are children of main.py). Reached through the tailnet:
# `tailscale serve --https=8443 http://127.0.0.1:8766` gives it a valid TLS
# cert at https://<pod>.<tailnet>.ts.net:8443 — required by iOS for camera
# access. Plain HTTP internally (only tailscaled talks to it).
#
#   /app/xr.html      the built React XR viewer (ui/dist, `npm run build:xr`)
#   /static/xr8/*     the self-hosted 8th Wall engine binary
#   /api/ar/*         session catalog / mesh / decimated cloud / usdz / telemetry
#   /                 redirect → /app/xr.html
#
# Launch: scripts/serve_ar.sh (env da3, port 8766, tmux session `arserver`
# via init_pod.sh).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ar_api import router as ar_router

app = FastAPI(title="STAC AR server")
app.include_router(ar_router)

_ROOT = Path(__file__).resolve().parent.parent
_UI_DIST = _ROOT / "ui" / "dist"
_XR8 = _ROOT / "static" / "xr8"

if _UI_DIST.is_dir():
    app.mount("/app", StaticFiles(directory=str(_UI_DIST), html=True), name="app")
if _XR8.is_dir():
    app.mount("/static/xr8", StaticFiles(directory=str(_XR8)), name="xr8")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/app/xr.html")


@app.get("/health", include_in_schema=False)
async def health():
    return {"ok": True, "ui_dist": _UI_DIST.is_dir(), "engine": _XR8.is_dir()}
