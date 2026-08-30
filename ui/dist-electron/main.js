import { app as n, BrowserWindow as i } from "electron";
import { fileURLToPath as a } from "node:url";
import o from "node:path";
const s = o.dirname(a(import.meta.url));
process.env.APP_ROOT = o.join(s, "..");
const t = process.env.VITE_DEV_SERVER_URL, _ = o.join(process.env.APP_ROOT, "dist-electron"), l = o.join(process.env.APP_ROOT, "dist");
process.env.VITE_PUBLIC = t ? o.join(process.env.APP_ROOT, "public") : l;
let e;
function d() {
  e = new i({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "STAC Build — Ingerop IN3",
    backgroundColor: "#0d1117",
    titleBarStyle: "default",
    webPreferences: {
      preload: o.join(s, "preload.mjs"),
      // Allow WebGL for Three.js
      webgl: !0
    }
  }), e.webContents.on("render-process-gone", (c, r) => {
    console.error(`[electron] renderer gone: ${r.reason} (exitCode=${r.exitCode})`), r.reason !== "clean-exit" && setTimeout(() => {
      e && !e.isDestroyed() && e.webContents.reloadIgnoringCache();
    }, 1e3);
  }), e.webContents.on("unresponsive", () => {
    console.warn("[electron] renderer unresponsive (heavy load or hang)");
  }), e.maximize(), t ? (e.loadURL(t), process.env.STAC_DEVTOOLS && e.webContents.openDevTools({ mode: "right" })) : e.loadFile(o.join(l, "index.html"));
}
n.on("window-all-closed", () => {
  process.platform !== "darwin" && (n.quit(), e = null);
});
n.on("activate", () => {
  i.getAllWindows().length === 0 && d();
});
n.whenReady().then(d);
export {
  _ as MAIN_DIST,
  l as RENDERER_DIST,
  t as VITE_DEV_SERVER_URL
};
