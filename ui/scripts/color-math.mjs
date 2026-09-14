/**
 * Colour math shared by the design verification scripts.
 * sRGB parsing, WCAG relative luminance / contrast, CIE Lab + ΔE2000, and the
 * Machado et al. (2009) colour-vision-deficiency simulation matrices.
 * Pure functions, no dependencies.
 */

export function parseColor(str) {
  const s = str.trim().toLowerCase()
  let m
  if ((m = s.match(/^#([0-9a-f]{3,8})$/))) {
    let h = m[1]
    if (h.length === 3 || h.length === 4) h = h.split('').map(c => c + c).join('')
    const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16)
    const a = h.length === 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1
    return { r, g, b, a }
  }
  if ((m = s.match(/^rgba?\(([^)]+)\)$/))) {
    const parts = m[1].split(/[\s,\/]+/).filter(Boolean).map(Number)
    return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 }
  }
  if (s === 'transparent') return { r: 0, g: 0, b: 0, a: 0 }
  if (s === 'none') return null
  throw new Error(`unparseable colour: ${str}`)
}

/** Composite `fg` (with alpha) over an opaque `bg`. */
export function over(fg, bg) {
  const a = fg.a ?? 1
  return {
    r: fg.r * a + bg.r * (1 - a),
    g: fg.g * a + bg.g * (1 - a),
    b: fg.b * a + bg.b * (1 - a),
    a: 1,
  }
}

/** color-mix(in srgb, A p%, B) with B taking the remainder. */
export function mixSrgb(a, pa, b) {
  const pb = 1 - pa
  return { r: a.r * pa + b.r * pb, g: a.g * pa + b.g * pb, b: a.b * pa + b.b * pb, a: 1 }
}

export function toHex(c) {
  const h = v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0')
  return `#${h(c.r)}${h(c.g)}${h(c.b)}`
}

function linear(v) {
  const c = v / 255
  return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

export function luminance(c) {
  return 0.2126 * linear(c.r) + 0.7152 * linear(c.g) + 0.0722 * linear(c.b)
}

export function contrast(a, b) {
  const la = luminance(a), lb = luminance(b)
  const [hi, lo] = la > lb ? [la, lb] : [lb, la]
  return (hi + 0.05) / (lo + 0.05)
}

// ── CIE Lab / ΔE2000 ────────────────────────────────────────────────────
function toXyz(c) {
  const r = linear(c.r), g = linear(c.g), b = linear(c.b)
  return {
    x: r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
    y: r * 0.2126729 + g * 0.7151522 + b * 0.072175,
    z: r * 0.0193339 + g * 0.119192 + b * 0.9503041,
  }
}

export function toLab(c) {
  const { x, y, z } = toXyz(c)
  const xn = 0.95047, yn = 1.0, zn = 1.08883
  const f = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116)
  const fx = f(x / xn), fy = f(y / yn), fz = f(z / zn)
  return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) }
}

export function deltaE2000(c1, c2) {
  const lab1 = toLab(c1), lab2 = toLab(c2)
  const rad = Math.PI / 180, deg = 180 / Math.PI
  const C1 = Math.hypot(lab1.a, lab1.b), C2 = Math.hypot(lab2.a, lab2.b)
  const Cbar = (C1 + C2) / 2
  const G = 0.5 * (1 - Math.sqrt(Math.pow(Cbar, 7) / (Math.pow(Cbar, 7) + Math.pow(25, 7))))
  const a1p = (1 + G) * lab1.a, a2p = (1 + G) * lab2.a
  const C1p = Math.hypot(a1p, lab1.b), C2p = Math.hypot(a2p, lab2.b)
  const h = (a, b) => { let v = Math.atan2(b, a) * deg; if (v < 0) v += 360; return v }
  const h1p = C1p === 0 ? 0 : h(a1p, lab1.b), h2p = C2p === 0 ? 0 : h(a2p, lab2.b)
  const dLp = lab2.L - lab1.L, dCp = C2p - C1p
  let dhp
  if (C1p * C2p === 0) dhp = 0
  else if (Math.abs(h2p - h1p) <= 180) dhp = h2p - h1p
  else dhp = h2p - h1p > 180 ? h2p - h1p - 360 : h2p - h1p + 360
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * rad)
  const Lbp = (lab1.L + lab2.L) / 2, Cbp = (C1p + C2p) / 2
  let hbp
  if (C1p * C2p === 0) hbp = h1p + h2p
  else if (Math.abs(h1p - h2p) <= 180) hbp = (h1p + h2p) / 2
  else hbp = h1p + h2p < 360 ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2
  const T = 1 - 0.17 * Math.cos((hbp - 30) * rad) + 0.24 * Math.cos(2 * hbp * rad)
    + 0.32 * Math.cos((3 * hbp + 6) * rad) - 0.20 * Math.cos((4 * hbp - 63) * rad)
  const dTheta = 30 * Math.exp(-Math.pow((hbp - 275) / 25, 2))
  const Rc = 2 * Math.sqrt(Math.pow(Cbp, 7) / (Math.pow(Cbp, 7) + Math.pow(25, 7)))
  const Sl = 1 + (0.015 * Math.pow(Lbp - 50, 2)) / Math.sqrt(20 + Math.pow(Lbp - 50, 2))
  const Sc = 1 + 0.045 * Cbp, Sh = 1 + 0.015 * Cbp * T
  const Rt = -Math.sin(2 * dTheta * rad) * Rc
  return Math.sqrt(Math.pow(dLp / Sl, 2) + Math.pow(dCp / Sc, 2) + Math.pow(dHp / Sh, 2) + Rt * (dCp / Sc) * (dHp / Sh))
}

// ── Colour vision deficiency (Machado, Oliveira & Fernandes 2009, severity 1.0) ──
const CVD = {
  protanopia: [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
  ],
  deuteranopia: [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881],
  ],
  tritanopia: [
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900],
  ],
}

function toLinearRgb(c) { return [linear(c.r), linear(c.g), linear(c.b)] }
function fromLinear(v) {
  const c = Math.max(0, Math.min(1, v))
  return 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055)
}

export function simulateCvd(c, kind) {
  const M = CVD[kind]
  const [r, g, b] = toLinearRgb(c)
  const out = M.map(row => row[0] * r + row[1] * g + row[2] * b)
  return { r: fromLinear(out[0]), g: fromLinear(out[1]), b: fromLinear(out[2]), a: 1 }
}

export const CVD_KINDS = Object.keys(CVD)
