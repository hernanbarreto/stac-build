import { app, BrowserWindow } from "electron";
import { fileURLToPath } from "node:url";
import path from "node:path";
const __dirname$1 = path.dirname(fileURLToPath(import.meta.url));
process.env.APP_ROOT = path.join(__dirname$1, "..");
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
const MAIN_DIST = path.join(process.env.APP_ROOT, "dist-electron");
const RENDERER_DIST = path.join(process.env.APP_ROOT, "dist");
process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL ? path.join(process.env.APP_ROOT, "public") : RENDERER_DIST;
let win;
function createWindow() {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "STAC Build — Ingerop IN3",
    backgroundColor: "#0d1117",
    titleBarStyle: "default",
    webPreferences: {
      preload: path.join(__dirname$1, "preload.mjs"),
      // Allow WebGL for Three.js
      webgl: true
    }
  });
  win.webContents.on("render-process-gone", (_event, details) => {
    console.error(`[electron] renderer gone: ${details.reason} (exitCode=${details.exitCode})`);
    if (details.reason !== "clean-exit") {
      setTimeout(() => {
        if (win && !win.isDestroyed()) win.webContents.reloadIgnoringCache();
      }, 1e3);
    }
  });
  win.webContents.on("unresponsive", () => {
    console.warn("[electron] renderer unresponsive (heavy load or hang)");
  });
  win.maximize();
  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
    if (process.env.STAC_DEVTOOLS) win.webContents.openDevTools({ mode: "right" });
  } else {
    win.loadFile(path.join(RENDERER_DIST, "index.html"));
  }
}
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
app.whenReady().then(createWindow);
export {
  MAIN_DIST,
  RENDERER_DIST,
  VITE_DEV_SERVER_URL
};
