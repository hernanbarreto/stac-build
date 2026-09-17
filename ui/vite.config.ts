import { defineConfig } from 'vite'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Check if we should include Electron plugin
const isWebOnly = process.env.WEB_ONLY === 'true'
// Backend target for the dev proxy; the screenshot harness points it at the mock server.
const apiTarget = process.env.STAC_API_TARGET ?? 'https://localhost:8765'

// ── the dev proxy says the backend is down ONCE, not once per request ───────
// The UI polls /health every 5 s and the login page does its own poll, so a
// backend that is merely restarting buried the Vite window under ECONNREFUSED
// stacks — 3308 of them in one morning, and the 3309th hides the line that
// actually matters. The proxy now answers 503 itself (the UI reads that
// exactly like the old connection failure and shows "server down") and logs
// only the transitions.
let apiDown = false
const quietProxy = (proxy: any, label: string) => {
  proxy.on('error', (err: any, _req: any, res: any) => {
    const refused = err?.code === 'ECONNREFUSED'
      || (Array.isArray(err?.errors) && err.errors.some((e: any) => e?.code === 'ECONNREFUSED'))
    if (!refused) {
      console.warn(`[proxy:${label}] ${err?.message ?? err}`)   // a REAL error is never hidden
      return
    }
    if (!apiDown) {
      apiDown = true
      console.warn(`[proxy] backend ${apiTarget} is down — further refusals silenced until it answers`)
    }
    if (res && typeof res.writeHead === 'function') {
      if (!res.headersSent) res.writeHead(503, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ detail: 'backend down' }))
    } else if (res && typeof res.destroy === 'function') {
      res.destroy()                                             // a websocket, not a response
    }
  })
  proxy.on('proxyRes', () => {
    if (apiDown) {
      apiDown = false
      console.log('[proxy] backend is back')
    }
  })
}

/** One proxy entry for the backend, quiet while it is down. */
const api = (label: string, extra: Record<string, unknown> = {}) => ({
  target: apiTarget,
  changeOrigin: true,
  secure: false,                 // accept the self-signed cert
  configure: (proxy: any) => quietProxy(proxy, label),
  ...extra,
})

// https://vitejs.dev/config/
export default defineConfig(async () => {
  const plugins = [react()]

  // Only include Electron plugin when not in web-only mode
  if (!isWebOnly) {
    const electronModule = await import('vite-plugin-electron/simple')
    const electron = electronModule.default as any
    const electronPlugins = electron({
      main: {
        entry: 'electron/main.ts',
      },
      preload: {
        input: path.join(__dirname, 'electron/preload.ts'),
      },
      renderer: process.env.NODE_ENV === 'test' ? undefined : {},
    })
    plugins.push(...(Array.isArray(electronPlugins) ? electronPlugins : [electronPlugins]))
  }

  return {
    plugins,
    build: {
      rollupOptions: {
        input: {
          index: path.join(__dirname, 'index.html'),
          xr: path.join(__dirname, 'xr.html'),      // mobile XR viewer entry
        },
      },
    },
    server: {
      // Proxy API requests to STAC server (avoids CORS)
      proxy: {
        '/sessions': api('sessions'),
        '/ws': api('ws', { ws: true }),
        '/slam': api('slam'),
        '/segments': api('segments'),
        '/mode': api('mode'),
        // 5 min: SAM3 model loading takes ~3 min and the proxy must not drop it
        '/api': api('api', { timeout: 300000, proxyTimeout: 300000 }),
        '/health': api('health'),
        '/status': api('status'),
        '/potree': api('potree'),
      },
    },
  }
})
