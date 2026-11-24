/**
 * EACH DEVICE WORKS ON ITS OWN SCAN (USER 2026-11-24: "debe mantener el contexto y la conversación para cada sesión
 * de usuario que está trabajando, OJO con eso!!!") — the server's request_scan.py.
 *
 * Every request this page makes to its own server says which session and scan it is on (the header
 * X-STAC-Scan: "<session>|<scan>", each part URL-encoded), so that a route naming no scan resolves THIS device's
 * scan, never the project's active scan another device rewrites (the cloud's octree, fetched as the camera moves,
 * the segmentation list, the sheet …). What a browser loads by URL with no header (an <img>) says it in its
 * query: withScan(url). A route that names its scan keeps it; the server ignores the header for another session.
 */
let current: { session: string; scan: string } | null = null

/** the session and scan on screen (App, as they change) */
export function setScanContext(session: string | null, scan: string | null): void {
  current = session && scan ? { session, scan } : null
}

function value(): string | null {
  return current ? `${encodeURIComponent(current.session)}|${encodeURIComponent(current.scan)}` : null
}

function sameOrigin(url: string): boolean {
  if (url.startsWith('/')) return !url.startsWith('//')
  try { return new URL(url, window.location.href).origin === window.location.origin } catch { return false }
}

/** a URL the browser loads by itself (an <img>): the scan in its query */
export function withScan(url: string): string {
  const v = value()
  return v ? `${url}${url.includes('?') ? '&' : '?'}stac_scan=${encodeURIComponent(v)}` : url
}

let installed = false

/** once, before the app renders: every same-origin fetch carries the header (an explicit one is kept) */
export function installScanHeader(): void {
  if (installed || typeof window === 'undefined' || typeof window.fetch !== 'function') return
  installed = true
  const original = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const v = value()
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    if (!v || !sameOrigin(url)) return original(input, init)
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined))
    if (!headers.has('X-STAC-Scan')) headers.set('X-STAC-Scan', v)
    return original(input, { ...init, headers })
  }
}
