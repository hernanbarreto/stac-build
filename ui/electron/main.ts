/**
 * STAC Build — Electron Main Process
 * Hernán Barreto — Ingerop IN3
 */
import { app, BrowserWindow } from 'electron'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

process.env.APP_ROOT = path.join(__dirname, '..')

export const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
export const MAIN_DIST = path.join(process.env.APP_ROOT, 'dist-electron')
export const RENDERER_DIST = path.join(process.env.APP_ROOT, 'dist')

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL
  ? path.join(process.env.APP_ROOT, 'public')
  : RENDERER_DIST

let win: BrowserWindow | null

function createWindow() {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'STAC Build — Ingerop IN3',
    backgroundColor: '#0d1117',
    titleBarStyle: 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
      // Allow WebGL for Three.js
      webgl: true,
    },
  })

  // Remove default menu for cleaner look
  // win.setMenu(null)

  // Renderer crash safety net: a dead renderer (e.g. OOM while loading a huge
  // point cloud) is otherwise just a silent white window. Log the reason and
  // reload so the user gets a fresh session instead of a frozen blank screen.
  win.webContents.on('render-process-gone', (_event, details) => {
    console.error(`[electron] renderer gone: ${details.reason} (exitCode=${details.exitCode})`)
    if (details.reason !== 'clean-exit') {
      setTimeout(() => {
        if (win && !win.isDestroyed()) win.webContents.reloadIgnoringCache()
      }, 1000)
    }
  })
  win.webContents.on('unresponsive', () => {
    console.warn('[electron] renderer unresponsive (heavy load or hang)')
  })

  // Maximize on start for full workspace
  win.maximize()

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL)
    // DevTools only when explicitly requested (STAC_DEVTOOLS=1) — keep the
    // default launch clean/professional.
    if (process.env.STAC_DEVTOOLS) win.webContents.openDevTools({ mode: 'right' })
  } else {
    win.loadFile(path.join(RENDERER_DIST, 'index.html'))
  }
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
    win = null
  }
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

app.whenReady().then(createWindow)
