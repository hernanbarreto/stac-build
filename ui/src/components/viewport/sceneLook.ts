/**
 * Scene look — the viewer's background and grid, chosen per session (USER 2026-11-09: "que el usuario pueda
 * cambiar entre tres tipos [de fondo] … y también que le pueda cambiar el color a la grilla … el tamaño de la
 * cuadrícula … que sea ajustable … la selección es por sesión, por defecto están todas así como está ahora,
 * y se guarda en el archivo de preferencias del usuario").
 *
 * Background kinds (names after Navisworks' background settings):
 *   default  the diffused charcoal gradient (ReCap Pro, USER 2026-11-09)
 *   horizon  green ground at y = 0 and a graduated sky above it (the look of 2026-11-08)
 *   plain    one flat colour the user picks
 * Grid: minor lines every `cellM` metres, major (brighter) lines every `majorEvery` cells — the major lines
 * let a distance be counted at a glance; colour free whatever the background.
 *
 * A colour of `null` is the design token (the default look): nothing here carries a literal colour.
 */
import * as THREE from 'three'
import { tokenHex, tokenThree, VP } from './palette'

export type BackgroundKind = 'default' | 'horizon' | 'plain'

export interface SceneLook {
  background: { kind: BackgroundKind; color: string | null }
  grid: { cellM: number; majorEvery: number; color: string | null }
}

/** The cell sizes offered (metres) and the major-line spacings (cells). */
export const GRID_CELLS_M = [0.1, 0.25, 0.5, 1, 2, 5] as const
export const GRID_MAJOR_EVERY = [2, 4, 5, 10] as const

/** Today's look: the charcoal gradient, a 1 m white grid, a major line every 10 cells. */
export const DEFAULT_LOOK: SceneLook = {
  background: { kind: 'default', color: null },
  grid: { cellM: 1, majorEvery: 10, color: null },
}

const HEX = /^#[0-9a-fA-F]{6}$/

/** A session's saved look, field by field: anything missing or malformed takes the default. */
export function parseLook(raw: unknown): SceneLook {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, any>
  const bg = (r.background && typeof r.background === 'object' ? r.background : {}) as Record<string, unknown>
  const gr = (r.grid && typeof r.grid === 'object' ? r.grid : {}) as Record<string, unknown>
  const kind: BackgroundKind = bg.kind === 'horizon' || bg.kind === 'plain' ? bg.kind : 'default'
  const colour = (v: unknown) => (typeof v === 'string' && HEX.test(v) ? v.toLowerCase() : null)
  const cell = typeof gr.cellM === 'number' && (GRID_CELLS_M as readonly number[]).includes(gr.cellM)
    ? gr.cellM : DEFAULT_LOOK.grid.cellM
  const major = typeof gr.majorEvery === 'number' && (GRID_MAJOR_EVERY as readonly number[]).includes(gr.majorEvery)
    ? gr.majorEvery : DEFAULT_LOOK.grid.majorEvery
  return { background: { kind, color: colour(bg.color) }, grid: { cellM: cell, majorEvery: major, color: colour(gr.color) } }
}

// ── the grid ────────────────────────────────────────────────────────────

/** A grid length as people write it: "1 m", "0.5 m", "0.25 m" — never "1.000 m". */
export function gridLength(m: number, number: (v: number, digits: number) => string): string {
  const digits = Number.isInteger(m) ? 0 : Number.isInteger(m * 10) ? 1 : 2
  return `${number(m, digits)} m`
}

/** Line positions at multiples of `cell` covering [lo, hi], anchored to the world origin (a line at every
 * whole cell, so the cell size reads as real coordinates), with the major ones flagged. */
export function gridLines(lo: number, hi: number, cell: number, majorEvery: number): { at: number; major: boolean }[] {
  const out: { at: number; major: boolean }[] = []
  const i0 = Math.round(lo / cell), i1 = Math.round(hi / cell)
  for (let i = i0; i <= i1; i++) out.push({ at: i * cell, major: i % majorEvery === 0 })
  return out
}

/** The most lines one axis draws: a grid finer than this over a large cloud is drawn coarser, never frozen. */
const MAX_LINES_PER_AXIS = 2000

/** The grid over a cloud's footprint (x, z in [min, max]) at y = 0: minor and major line sets. The footprint
 * is widened to whole cells so every line spans the same closed rectangle, and its border is drawn as a major
 * line (USER 2026-11-09: "la grilla tiene errores, fijate que no cierra bien" — the lines of one direction ended
 * at the cloud's footprint, the other direction's at whole cells beyond it). */
export function makeLookGrid(min: THREE.Vector2, max: THREE.Vector2, look: SceneLook): THREE.Group {
  let cell = look.grid.cellM
  while (Math.max(max.x - min.x, max.y - min.y) / cell > MAX_LINES_PER_AXIS) cell *= 2
  const x0 = Math.floor(min.x / cell) * cell, x1 = Math.ceil(max.x / cell) * cell
  const z0 = Math.floor(min.y / cell) * cell, z1 = Math.ceil(max.y / cell) * cell
  const colour = look.grid.color ? new THREE.Color(look.grid.color) : tokenThree(VP.gridMajor)
  const minor: number[] = [], major: number[] = []
  const xs = gridLines(x0, x1, cell, look.grid.majorEvery), zs = gridLines(z0, z1, cell, look.grid.majorEvery)
  xs.forEach((l, i) => ((l.major || i === 0 || i === xs.length - 1) ? major : minor).push(l.at, 0, z0, l.at, 0, z1))
  zs.forEach((l, i) => ((l.major || i === 0 || i === zs.length - 1) ? major : minor).push(x0, 0, l.at, x1, 0, l.at))
  const group = new THREE.Group()
  group.name = 'grid'
  for (const [pts, opacity] of [[minor, 0.28], [major, 0.7]] as [number[], number][]) {
    if (!pts.length) continue
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
    const mat = new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity, depthWrite: false, toneMapped: false })
    group.add(new THREE.LineSegments(geo, mat))
  }
  return group
}

export function disposeGroup(g: THREE.Object3D): void {
  g.traverse(o => {
    const m = o as THREE.Mesh
    m.geometry?.dispose()
    const mat = m.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(mat)) mat.forEach(x => x.dispose()); else mat?.dispose()
  })
}

// ── the backgrounds ─────────────────────────────────────────────────────

/** The horizon: a sky dome drawn first at the far plane (rotation-only view) — a gradient from the horizon to
 * the zenith above y = 0 and the dark ground colour below — and a translucent green ground plane at y = 0 (the
 * points under the floor still show through, tinted). Hidden unless the background is `horizon`. */
export function makeHorizon(): THREE.Group {
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, depthTest: false, toneMapped: false,
    uniforms: {
      uZenith: { value: tokenThree(VP.skyZenith) },
      uHorizon: { value: tokenThree(VP.skyHorizon) },
      uGround: { value: tokenThree(VP.skyGround) },
    },
    vertexShader: `
      varying vec3 vDir;
      void main() {
        vDir = position;
        vec4 p = projectionMatrix * mat4(mat3(viewMatrix)) * vec4(position, 1.0);
        gl_Position = p.xyww;
      }`,
    fragmentShader: `
      uniform vec3 uZenith; uniform vec3 uHorizon; uniform vec3 uGround;
      varying vec3 vDir;
      void main() {
        float h = normalize(vDir).y;
        vec3 c = h >= 0.0 ? mix(uHorizon, uZenith, pow(clamp(h, 0.0, 1.0), 0.55))
                          : mix(uHorizon * 0.35 + uGround * 0.65, uGround, clamp(-h * 8.0, 0.0, 1.0));
        gl_FragColor = vec4(c, 1.0);
      }`,
  })
  const sky = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), skyMat)
  sky.frustumCulled = false
  sky.renderOrder = -1000
  const groundMat = new THREE.MeshBasicMaterial({
    color: tokenHex(VP.groundPlane), transparent: true, opacity: 0.55, depthWrite: false,
    side: THREE.DoubleSide, toneMapped: false,
  })
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), groundMat)
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -0.002          // just under the grid: no z-fighting with its lines
  ground.renderOrder = -10
  const g = new THREE.Group()
  g.name = 'horizon'
  g.userData.isEnvironment = true
  sky.userData.isEnvironment = true
  ground.userData.isEnvironment = true
  g.add(sky, ground)
  g.visible = false
  return g
}

/** Apply a background: the gradient texture (default), the horizon group (horizon) or a flat colour (plain). */
export function applyBackground(scene: THREE.Scene, look: SceneLook, gradient: THREE.Texture, horizon: THREE.Object3D): void {
  const kind = look.background.kind
  horizon.visible = kind === 'horizon'
  if (kind === 'horizon') scene.background = null
  else if (kind === 'plain') scene.background = look.background.color ? new THREE.Color(look.background.color) : tokenThree(VP.clear)
  else scene.background = gradient
}
