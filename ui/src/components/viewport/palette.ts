/**
 * Viewport palette — Three.js colours resolved from the design tokens at
 * runtime (prompt_ui.txt §7: selection orange, measurement cyan, point
 * status --pt-*, trajectory edges --edge-*, categorical --cat-*).
 *
 * Nothing in the 3D code carries a literal colour: every material asks the
 * palette by semantic name and the palette reads `--token` from the
 * document root. Values are cached and re-read when `refresh()` is called
 * (theme switch).
 */
import * as THREE from 'three'

const cache = new Map<string, string>()

function readToken(name: string): string {
  const hit = cache.get(name)
  if (hit) return hit
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  if (!raw) throw new Error(`palette: token ${name} is not defined`)
  cache.set(name, raw)
  return raw
}

/** CSS colour string of a token (usable for canvas 2D and inline SVG). */
export function tokenColor(name: string): string {
  return readToken(name)
}

/** Token as a 24-bit integer (THREE material `color`). Alpha is dropped. */
export function tokenHex(name: string): number {
  return new THREE.Color(readToken(name)).getHex()
}

export function tokenThree(name: string): THREE.Color {
  return new THREE.Color(readToken(name))
}

export function refreshPalette(): void {
  cache.clear()
}

/** Semantic names used by the viewport code — one place to see them all. */
export const VP = {
  clear: '--vp-clear',
  gridMajor: '--vp-grid-major',
  gridMinor: '--vp-grid-minor',
  light: '--vp-light',
  axisX: '--vp-axis-x',
  axisY: '--vp-axis-y',
  axisZ: '--vp-axis-z',
  selection: '--vp-selection',
  hover: '--vp-hover',
  measure: '--measure',
  marker: '--vp-marker',
  angle: '--vp-angle',
  sectionBox: '--vp-section-box',
  eraseMark: '--vp-erase-mark',
  eraseBox: '--vp-erase-box',
  epochTint: '--vp-epoch-tint',
  bimHighlight: '--vp-bim-highlight',
  sabanaUnmatched: '--vp-sabana-unmatched',
  sabanaDefault: '--vp-sabana-default',
  volumeFree: '--vp-volume-free',
  volumeTouching: '--vp-volume-touching',
  volumeColliding: '--vp-volume-colliding',
  labelBg: '--vp-label-bg',
  text1: '--text-1',
  text2: '--text-2',
  ok: '--ok',
  warn: '--warn',
  err: '--err',
  brand: '--brand-orange',
  ptVerified: '--pt-verified',
  ptSingle: '--pt-single',
  ptConflict: '--pt-conflict',
  ptDynamic: '--pt-dynamic',
  ptUnobserved: '--pt-unobserved',
  edgeOdo: '--edge-odo',
  edgeLoop: '--edge-loop',
  edgeScaleBreak: '--edge-scale-break',
  edgeRejected: '--edge-rejected',
  edgeAmbiguous: '--edge-ambiguous',
} as const

/** Trajectory edge colour by kind (validation kit). */
export function edgeColorHex(kind: string): number {
  switch (kind) {
    case 'accepted': return tokenHex(VP.edgeLoop)
    case 'scale_break': return tokenHex(VP.edgeScaleBreak)
    case 'rejected':
    case 'vetoed': return tokenHex(VP.edgeRejected)
    case 'ambiguous': return tokenHex(VP.edgeAmbiguous)
    default: return tokenHex(VP.edgeOdo)
  }
}

/** Point witness status colour token (legend + shader uniforms). */
export const STATUS_TOKENS: Record<string, string> = {
  verified: VP.ptVerified,
  single_witness: VP.ptSingle,
  mask_conflict: VP.ptConflict,
  dynamic: VP.ptDynamic,
  unobserved: VP.ptUnobserved,
}

export const EDGE_TOKENS: Record<string, string> = {
  odometry: VP.edgeOdo,
  accepted: VP.edgeLoop,
  scale_break: VP.edgeScaleBreak,
  rejected: VP.edgeRejected,
  ambiguous: VP.edgeAmbiguous,
}

/** The 12 categorical colours as CSS strings (instance colouring, new segments). */
export function categoricalPalette(): string[] {
  return Array.from({ length: 12 }, (_, i) => tokenColor(`--cat-${String(i + 1).padStart(2, '0')}`))
}

/** Deviation heat ramp tokens, within tolerance -> far out. */
export const DEV_RAMP = ['--dev-0', '--dev-1', '--dev-2', '--dev-3', '--dev-4']

/** Interpolate the deviation ramp for a fraction 0..1 -> CSS rgb string. */
export function deviationColor(frac: number): string {
  const f = Math.max(0, Math.min(1, frac)) * (DEV_RAMP.length - 1)
  const i = Math.min(DEV_RAMP.length - 2, Math.floor(f))
  const a = new THREE.Color(readToken(DEV_RAMP[i])), b = new THREE.Color(readToken(DEV_RAMP[i + 1]))
  return `#${a.lerp(b, f - i).getHexString()}`
}
