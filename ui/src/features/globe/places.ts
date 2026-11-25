/**
 * FINDING A PLACE (USER 2026-11-24: "para ubicar el punto GPS agrega varias herramientas para buscar ciudad, por
 * ejemplo, como google maps tiene, de hecho quise agregar este punto 19°23'53.8"N 99°11'58.4"W y me fue imposible,
 * debe ser mas amigable"):
 *  - coordinates as people paste them, read here: degrees-minutes-seconds (19°23'53.8"N 99°11'58.4"W, O for oeste,
 *    with or without symbols), degrees and decimal minutes, decimal degrees (19.398278, -99.199556 · 19.398278N
 *    99.199556W · with decimal commas: 19,398278; -99,199556), a Google Maps link (…/@19.398,-99.199,17z or
 *    …!3d19.398!4d-99.199);
 *  - names and addresses: Esri's public World Geocoding Service (no account, the imagery's own provider) — its
 *    suggestions as one types, then the place a suggestion names.
 */
export interface Place { lat: number; lon: number; label: string; extent?: { west: number; south: number; east: number; north: number } }

const NUM = String.raw`(\d+(?:[.,]\d+)?)`

/** one coordinate in degrees: "19°23'53.8"N", "99 11 58.4 W", "-99.1995", "19,398N" — null if it is not one */
function oneCoordinate(text: string): { value: number; hemi: string | null } | null {
  let t = text.trim().toUpperCase().replace(/[′’´`]/g, "'").replace(/[″“”]/g, '"').replace(/''/g, '"')
  let hemi: string | null = null
  const h = t.match(/^([NSEWO])\s*|\s*([NSEWO])$/)
  if (h) { hemi = (h[1] ?? h[2]) === 'O' ? 'W' : (h[1] ?? h[2]); t = t.replace(/^([NSEWO])\s*|\s*([NSEWO])$/, '').trim() }
  let sign = 1
  if (t.startsWith('-')) { sign = -1; t = t.slice(1).trim() } else if (t.startsWith('+')) t = t.slice(1).trim()
  const num = (s: string) => Number(s.replace(',', '.'))
  // degrees [minutes [seconds]], with symbols or spaces between them
  const m = t.match(new RegExp(`^${NUM}\\s*(?:°|º|D|DEG)?\\s*(?:${NUM}\\s*(?:'|M|MIN)?\\s*(?:${NUM}\\s*(?:"|S|SEC)?)?)?$`))
  if (!m) return null
  const d = num(m[1]), mi = m[2] != null ? num(m[2]) : 0, s = m[3] != null ? num(m[3]) : 0
  if (mi >= 60 || s >= 60 || (m[2] != null && !Number.isInteger(d)) || (m[3] != null && !Number.isInteger(mi))) return null
  return { value: sign * (d + mi / 60 + s / 3600), hemi }
}

/** a latitude and a longitude in what was typed or pasted, or null */
export function parseCoordinates(text: string): { lat: number; lon: number } | null {
  const raw = text.trim()
  if (!raw) return null
  // a Google Maps link
  const link = raw.match(/!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)/) ?? raw.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/)
    ?? raw.match(/[?&](?:q|query|ll)=(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/)
  if (link) return valid(Number(link[1]), Number(link[2]))
  // two parts: split where a hemisphere letter ends the first, or at ";" / "," between them, or at whitespace
  const u = raw.toUpperCase()
  const candidates: [string, string][] = []
  const byLetter = u.match(/^(.*?[NS])\s*[,;]?\s*(.+)$/)
  if (byLetter) candidates.push([byLetter[1], byLetter[2]])
  if (u.includes(';')) { const [a, ...b] = u.split(';'); candidates.push([a, b.join(';')]) }
  const commas = u.split(',')
  if (commas.length === 2) candidates.push([commas[0], commas[1]])
  if (commas.length === 4) candidates.push([`${commas[0]},${commas[1]}`, `${commas[2]},${commas[3]}`])   // decimal commas
  const ws = u.split(/\s+/)
  for (let k = 1; k < ws.length; k++) candidates.push([ws.slice(0, k).join(' '), ws.slice(k).join(' ')])
  for (const [a, b] of candidates) {
    const p = oneCoordinate(a), q = oneCoordinate(b)
    if (!p || !q) continue
    // which is the latitude: the hemisphere letters say it; else the first one is
    const latFirst = !(p.hemi === 'E' || p.hemi === 'W' || q.hemi === 'N' || q.hemi === 'S')
    const [la, lo] = latFirst ? [p, q] : [q, p]
    const lat = la.hemi === 'S' ? -Math.abs(la.value) : la.value
    const lon = lo.hemi === 'W' ? -Math.abs(lo.value) : lo.value
    const v = valid(lat, lon)
    if (v) return v
  }
  return null
}

function valid(lat: number, lon: number) {
  return Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180 ? { lat, lon } : null
}

/** 19°23'53.8"N 99°11'58.4"W */
export function formatDms(lat: number, lon: number): string {
  const one = (v: number, pos: string, neg: string) => {
    const a = Math.abs(v), d = Math.floor(a), mf = (a - d) * 60, m = Math.floor(mf), s = (mf - m) * 60
    const sr = Math.round(s * 10) / 10
    const [dd, mm, ss] = sr >= 60 ? (m + 1 >= 60 ? [d + 1, 0, 0] : [d, m + 1, 0]) : [d, m, sr]
    return `${dd}°${mm}'${ss.toFixed(1)}"${v < 0 ? neg : pos}`
  }
  return `${one(lat, 'N', 'S')} ${one(lon, 'E', 'W')}`
}

const GEOCODER = 'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer'

/** suggestions as the user types (Esri's public geocoder: no account) */
export async function suggestPlaces(text: string, signal?: AbortSignal): Promise<{ text: string; magicKey: string }[]> {
  const q = new URLSearchParams({ text, f: 'json', maxSuggestions: '6' })
  const r = await fetch(`${GEOCODER}/suggest?${q}`, { signal })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const d = await r.json() as { suggestions?: { text: string; magicKey: string; isCollection?: boolean }[] }
  return (d.suggestions ?? []).filter(s => !s.isCollection).map(s => ({ text: s.text, magicKey: s.magicKey }))
}

/** the place a suggestion (or a whole typed text) names */
export async function findPlace(text: string, magicKey?: string, signal?: AbortSignal): Promise<Place | null> {
  const q = new URLSearchParams({ SingleLine: text, f: 'json', maxLocations: '1', outFields: 'Match_addr' })
  if (magicKey) q.set('magicKey', magicKey)
  const r = await fetch(`${GEOCODER}/findAddressCandidates?${q}`, { signal })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const d = await r.json() as { candidates?: { address: string; location: { x: number; y: number }; extent?: { xmin: number; ymin: number; xmax: number; ymax: number } }[] }
  const c = d.candidates?.[0]
  if (!c) return null
  return { lat: c.location.y, lon: c.location.x, label: c.address,
           extent: c.extent ? { west: c.extent.xmin, south: c.extent.ymin, east: c.extent.xmax, north: c.extent.ymax } : undefined }
}
