/**
 * THE GLOBE'S DATA — one georeference per project (server geo.py / geo_api.py; USER 2026-11-24: "vamos a agregarle
 * además la ubicación … gps de cada proyecto … cesium siempre sin cuenta").
 */
export type GeoSource = 'manual' | 'video' | 'points'

export interface Georef {
  lat: number
  lon: number
  height_m: number
  /** the azimuth, clockwise from true north, of the plan's +Y (the display frame's −z) */
  heading_deg: number
  scan: string | null
  source: GeoSource
  provenance: string
  accuracy_m: number | null
  set_by: string
  set_at: string
  note: string
}

export interface GeoProject {
  session_id: string
  name: string
  scans: string[]
  reference: string | null
  georef: Georef | null
  /** the project's IFC files (ifcs/) */
  ifcs: string[]
}

export interface GeoPlacement {
  lat: number
  lon: number
  height_m: number
  heading_deg?: number
  source: GeoSource
  accuracy_m?: number | null
  scan?: string | null
}

async function answer<T>(r: Response): Promise<T> {
  let d: Record<string, unknown> = {}
  try { d = await r.json() } catch { /* no body */ }
  if (!r.ok) throw new Error(String(d.detail ?? d.error ?? `HTTP ${r.status}`))
  return d as T
}

export async function fetchGeoProjects(): Promise<GeoProject[]> {
  const d = await answer<{ projects?: GeoProject[] }>(await fetch('/api/geo', { cache: 'no-store' }))
  return Array.isArray(d.projects) ? d.projects : []
}

function auth(token: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function saveGeoref(sid: string, place: GeoPlacement, token: string | null): Promise<Georef> {
  const d = await answer<{ georef: Georef }>(await fetch(`/api/geo/${encodeURIComponent(sid)}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json', ...auth(token) }, body: JSON.stringify(place),
  }))
  return d.georef
}

export async function removeGeoref(sid: string, token: string | null): Promise<void> {
  await answer(await fetch(`/api/geo/${encodeURIComponent(sid)}`, { method: 'DELETE', headers: auth(token) }))
}

/** "41.64880, -0.88910 · 199 m" */
export function formatPlace(g: Pick<Georef, 'lat' | 'lon' | 'height_m'>): string {
  return `${g.lat.toFixed(5)}, ${g.lon.toFixed(5)} · ${Math.round(g.height_m)} m`
}
