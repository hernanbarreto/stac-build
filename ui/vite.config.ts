import { defineConfig } from 'vite'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Check if we should include Electron plugin
const isWebOnly = process.env.WEB_ONLY === 'true'

// https://vitejs.dev/config/
export default defineConfig(async () => {
  const plugins = [react()]

  // Only include Electron plugin when not in web-only mode
  if (!isWebOnly) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
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
    server: {
      // Proxy API requests to STAC server (avoids CORS)
      proxy: {
        '/sessions': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false, // Accept self-signed certs
        },
        '/ws': {
          target: 'https://localhost:8765',
          ws: true,
          changeOrigin: true,
          secure: false,
        },
        '/slam': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false,
        },
        '/segments': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false,
        },
        '/mode': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false,
        },
        '/api': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false,
          timeout: 300000,        // 5 min — SAM3 model loading takes ~3 min
          proxyTimeout: 300000,   // 5 min — prevent proxy from dropping long requests
        },
        '/health': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false,
        },
        '/status': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false,
        },
        '/potree': {
          target: 'https://localhost:8765',
          changeOrigin: true,
          secure: false,
        },
      },
    },
  }
})
