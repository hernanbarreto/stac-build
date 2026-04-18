import { app as n, BrowserWindow as t } from "electron";
import { fileURLToPath as d } from "node:url";
import e from "node:path";
const r = e.dirname(d(import.meta.url));
process.env.APP_ROOT = e.join(r, "..");
const i = process.env.VITE_DEV_SERVER_URL, m = e.join(process.env.APP_ROOT, "dist-electron"), l = e.join(process.env.APP_ROOT, "dist");
process.env.VITE_PUBLIC = i ? e.join(process.env.APP_ROOT, "public") : l;
let o;
function s() {
  o = new t({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "STAC Build — Ingerop IN3",
    backgroundColor: "#0d1117",
    titleBarStyle: "default",
    webPreferences: {
      preload: e.join(r, "preload.mjs"),
      // Allow WebGL for Three.js
      webgl: !0
    }
  }), o.maximize(), i ? (o.loadURL(i), o.webContents.openDevTools({ mode: "right" })) : o.loadFile(e.join(l, "index.html"));
}
n.on("window-all-closed", () => {
  process.platform !== "darwin" && (n.quit(), o = null);
});
n.on("activate", () => {
  t.getAllWindows().length === 0 && s();
});
n.whenReady().then(s);
export {
  m as MAIN_DIST,
  l as RENDERER_DIST,
  i as VITE_DEV_SERVER_URL
};
