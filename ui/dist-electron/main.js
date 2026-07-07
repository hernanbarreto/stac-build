import { app as n, BrowserWindow as t } from "electron";
import { fileURLToPath as d } from "node:url";
import e from "node:path";
const r = e.dirname(d(import.meta.url));
process.env.APP_ROOT = e.join(r, "..");
const i = process.env.VITE_DEV_SERVER_URL, m = e.join(process.env.APP_ROOT, "dist-electron"), s = e.join(process.env.APP_ROOT, "dist");
process.env.VITE_PUBLIC = i ? e.join(process.env.APP_ROOT, "public") : s;
let o;
function l() {
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
  }), o.maximize(), i ? (o.loadURL(i), process.env.STAC_DEVTOOLS && o.webContents.openDevTools({ mode: "right" })) : o.loadFile(e.join(s, "index.html"));
}
n.on("window-all-closed", () => {
  process.platform !== "darwin" && (n.quit(), o = null);
});
n.on("activate", () => {
  t.getAllWindows().length === 0 && l();
});
n.whenReady().then(l);
export {
  m as MAIN_DIST,
  s as RENDERER_DIST,
  i as VITE_DEV_SERVER_URL
};
