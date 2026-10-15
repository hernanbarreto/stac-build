/**
 * Reference document — hand-measured distances and planes anchored on
 * keyframe IMAGES for the evaluation battery (`/api/evaluation/references`).
 *
 * Pixels are native image pixels (u = column, v = row, floats allowed); a
 * distance end may be observed on several keyframes; a plane is one or more
 * polygons, each on its own keyframe. Ids are assigned by the UI
 * (`d_0001`, `p_0001`: 1 + the highest existing number, zero-padded).
 */
export interface RefObservation { frame: number; file?: string; u: number; v: number }
export interface RefEnd { observations: RefObservation[] }
export type RefRole = 'gauge' | 'test'
export type RefTool = 'tape' | 'laser' | 'total_station'
export interface RefDistance { id: string; value_m: number; sigma_m: number; role: RefRole; tool: RefTool; note: string; ends: [RefEnd, RefEnd] }
export interface RefPolygon { frame: number; file?: string; points: [number, number][] }
export interface RefPlane { id: string; label: string; note: string; polygons: RefPolygon[] }
export interface ReferenceDocument { schema_version: 1; image_size: [number, number] | null; distances: RefDistance[]; planes: RefPlane[] }

export const REF_ROLES: RefRole[] = ['gauge', 'test']
export const REF_TOOLS: RefTool[] = ['tape', 'laser', 'total_station']

export function emptyReferenceDocument(): ReferenceDocument {
  return { schema_version: 1, image_size: null, distances: [], planes: [] }
}

/** Defensive read of whatever the server returns: every list present, every field typed. */
export function normalizeReferenceDocument(raw: unknown): ReferenceDocument {
  const doc = (raw && typeof raw === 'object' ? raw : {}) as Partial<ReferenceDocument>
  const num = (v: unknown, fallback = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback)
  const str = (v: unknown, fallback = '') => (typeof v === 'string' ? v : fallback)
  const obs = (list: unknown): RefObservation[] => (Array.isArray(list) ? list : [])
    .filter(o => o && typeof o === 'object')
    .map((o: any) => ({ frame: Math.round(num(o.frame)), ...(typeof o.file === 'string' ? { file: o.file } : {}), u: num(o.u), v: num(o.v) }))
  const distances: RefDistance[] = (Array.isArray(doc.distances) ? doc.distances : [])
    .filter(d => d && typeof d === 'object')
    .map((d: any) => {
      const ends = Array.isArray(d.ends) ? d.ends : []
      return {
        id: str(d.id),
        value_m: num(d.value_m),
        sigma_m: num(d.sigma_m),
        role: REF_ROLES.includes(d.role) ? d.role as RefRole : 'test',
        tool: REF_TOOLS.includes(d.tool) ? d.tool as RefTool : 'tape',
        note: str(d.note),
        ends: [{ observations: obs(ends[0]?.observations) }, { observations: obs(ends[1]?.observations) }],
      }
    })
  const planes: RefPlane[] = (Array.isArray(doc.planes) ? doc.planes : [])
    .filter(p => p && typeof p === 'object')
    .map((p: any) => ({
      id: str(p.id),
      label: str(p.label, 'floor'),
      note: str(p.note),
      polygons: (Array.isArray(p.polygons) ? p.polygons : [])
        .filter((g: any) => g && typeof g === 'object')
        .map((g: any) => ({
          frame: Math.round(num(g.frame)),
          ...(typeof g.file === 'string' ? { file: g.file } : {}),
          points: (Array.isArray(g.points) ? g.points : [])
            .filter((pt: any) => Array.isArray(pt) && pt.length >= 2)
            .map((pt: any) => [num(pt[0]), num(pt[1])] as [number, number]),
        })),
    }))
  const size = doc.image_size
  const image_size: [number, number] | null = Array.isArray(size) && size.length === 2 && size.every(n => typeof n === 'number' && n > 0)
    ? [size[0], size[1]] : null
  return { schema_version: 1, image_size, distances, planes }
}

/** `d_0007` for prefix `d` when the highest existing number is 6. */
export function nextReferenceId(prefix: 'd' | 'p', existing: Array<{ id: string }>): string {
  let max = 0
  for (const { id } of existing) {
    const m = /^[dp]_(\d+)$/.exec(id)
    if (m) max = Math.max(max, parseInt(m[1], 10))
  }
  return `${prefix}_${String(max + 1).padStart(4, '0')}`
}

export function referenceCounts(doc: ReferenceDocument): { distances: number; gauge: number; test: number; planes: number } {
  return {
    distances: doc.distances.length,
    gauge: doc.distances.filter(d => d.role === 'gauge').length,
    test: doc.distances.filter(d => d.role === 'test').length,
    planes: doc.planes.length,
  }
}
