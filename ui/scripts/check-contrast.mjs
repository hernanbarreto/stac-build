#!/usr/bin/env node
/**
 * check-contrast.mjs — WCAG AA + palette separability over the design tokens.
 *
 * For every declared (foreground, background) pair, in BOTH densities:
 *   - text pairs must reach ≥ 4.5:1 (≥ 3:1 when the text token is ≥ --fs-4)
 *   - icon / control-border / fill pairs must reach ≥ 3:1
 * Translucent tokens (`--*-soft`, `--glass-bg`) are composited over the
 * surface they sit on before measuring.
 *
 * Categorical palette (--cat-*): pairwise ΔE2000 ≥ 20 between entries and
 * against every --pt-*, state and brand colour; under simulated protanopia /
 * deuteranopia / tritanopia the minimum pairwise ΔE2000 must stay ≥ 10.
 *
 * Exit 1 on any failure and print every offending pair with its ratio.
 */
import { readTokens } from './tokens-read.mjs'
import { parseColor, over, contrast, deltaE2000, simulateCvd, CVD_KINDS, toHex } from './color-math.mjs'

const TEXT_AA = 4.5
const LARGE_AA = 3.0
const NON_TEXT = 3.0
const CAT_MIN_DE = 20
const CAT_MIN_DE_CVD = 10

/** Text-on-surface pairs. `over` = the opaque surface a translucent bg sits on. */
const TEXT_PAIRS = [
  // body text on every anchored surface
  ...['--surface-0', '--surface-1', '--surface-2', '--surface-3', '--surface-4', '--surface-brand'].flatMap(bg => [
    { fg: '--text-1', bg },
    { fg: '--text-2', bg },
  ]),
  // muted metadata: panels, cards, hover rows (not on selected rows — declared)
  ...['--surface-0', '--surface-1', '--surface-2', '--surface-3', '--surface-brand'].map(bg => ({ fg: '--text-3', bg })),
  // glass overlays over the viewport
  { fg: '--text-1', bg: '--glass-bg', over: '--surface-0' },
  { fg: '--text-2', bg: '--glass-bg', over: '--surface-0' },
  { fg: '--text-3', bg: '--glass-bg', over: '--surface-0' },
  // inverse text on filled buttons
  { fg: '--text-inverse', bg: '--brand-orange' },
  { fg: '--text-inverse', bg: '--brand-orange-deep', large: true },
  { fg: '--text-inverse', bg: '--err' },
  { fg: '--text-inverse', bg: '--ok' },
  { fg: '--text-inverse', bg: '--measure' },
  // state text (fg variants) on panels and on their soft fills
  ...['ok', 'warn', 'err', 'info'].flatMap(s => [
    { fg: `--${s}-fg`, bg: '--surface-1' },
    { fg: `--${s}-fg`, bg: '--surface-2' },
    { fg: `--${s}-fg`, bg: `--${s}-soft`, over: '--surface-1' },
    { fg: `--${s}-fg`, bg: `--${s}-soft`, over: '--surface-2' },
    { fg: `--${s}-fg`, bg: '--glass-bg', over: '--surface-0' },
  ]),
  // measurement readouts and brand text (tab labels, selected items)
  { fg: '--measure', bg: '--surface-0' },
  { fg: '--measure', bg: '--surface-1' },
  { fg: '--measure', bg: '--glass-bg', over: '--surface-0' },
  { fg: '--measure', bg: '--measure-soft', over: '--surface-1' },
  { fg: '--brand-orange', bg: '--surface-1' },
  { fg: '--brand-orange', bg: '--surface-2' },
  { fg: '--brand-orange', bg: '--brand-orange-soft', over: '--surface-1' },
  { fg: '--brand-orange', bg: '--glass-bg', over: '--surface-0' },
  // readout value (large) in mono over glass
  { fg: '--text-1', bg: '--vp-label-bg', over: '--surface-0' },
]

/** Icons, control outlines, fills: ≥ 3:1 against their surface. */
const NON_TEXT_PAIRS = [
  // control boundaries (inputs, selects, checkboxes, secondary buttons);
  // --border-1/2/3 are decorative separators and card outlines: not gated
  { fg: '--border-ctl', bg: '--surface-1' },
  { fg: '--border-ctl', bg: '--surface-2' },
  { fg: '--border-ctl', bg: '--surface-0' },
  { fg: '--text-3', bg: '--surface-4' },        // muted icon on selected row
  { fg: '--brand-orange', bg: '--surface-0' },
  { fg: '--brand-orange', bg: '--surface-brand' },
  { fg: '--ok', bg: '--surface-1' },
  { fg: '--warn', bg: '--surface-1' },
  { fg: '--err', bg: '--surface-1' },
  { fg: '--info', bg: '--surface-1' },
  { fg: '--measure', bg: '--surface-brand' },
  ...['--pt-verified', '--pt-single', '--pt-conflict', '--pt-dynamic'].map(fg => ({ fg, bg: '--surface-0' })),
  ...['--edge-odo', '--edge-loop', '--edge-scale-break', '--edge-rejected', '--edge-ambiguous'].map(fg => ({ fg, bg: '--surface-0' })),
  ...['--dev-0', '--dev-1', '--dev-2', '--dev-3', '--dev-4'].map(fg => ({ fg, bg: '--surface-0' })),
]

const CAT_CONTRAST_BGS = ['--surface-0', '--surface-1', '--surface-2']
const CAT_VS = ['--brand-orange', '--ok', '--warn', '--err', '--info',
  '--pt-verified', '--pt-single', '--pt-conflict', '--pt-dynamic']

function resolve(map, name) {
  const v = map.get(name)
  if (v == null) throw new Error(`token ${name} is not defined in tokens.css`)
  return parseColor(v)
}

function composite(map, pair) {
  let bg = resolve(map, pair.bg)
  if (bg.a < 1) {
    if (!pair.over) throw new Error(`${pair.bg} is translucent: declare "over" for pair ${pair.fg}/${pair.bg}`)
    bg = over(bg, resolve(map, pair.over))
  }
  const fg = resolve(map, pair.fg)
  return { fg: fg.a < 1 ? over(fg, bg) : fg, bg }
}

const failures = []
const report = []
const tokens = readTokens()

for (const density of ['compact', 'comfortable']) {
  const map = tokens[density]
  const fsLarge = parseFloat(map.get('--fs-4'))
  const fs0 = parseFloat(map.get('--fs-0'))
  if (!(fs0 >= 11)) failures.push(`[${density}] --fs-0 is ${fs0}px, minimum is 11px`)
  for (const pair of TEXT_PAIRS) {
    const { fg, bg } = composite(map, pair)
    const ratio = contrast(fg, bg)
    const need = pair.large ? LARGE_AA : TEXT_AA
    const label = `[${density}] text ${pair.fg} on ${pair.bg}${pair.over ? ` over ${pair.over}` : ''}`
    report.push(`${label}: ${ratio.toFixed(2)}:1 (need ${need})`)
    if (ratio < need) failures.push(`${label} = ${ratio.toFixed(2)}:1 < ${need}:1  (${toHex(fg)} / ${toHex(bg)})`)
  }
  for (const pair of NON_TEXT_PAIRS) {
    const { fg, bg } = composite(map, pair)
    const ratio = contrast(fg, bg)
    const label = `[${density}] non-text ${pair.fg} on ${pair.bg}`
    report.push(`${label}: ${ratio.toFixed(2)}:1 (need ${NON_TEXT})`)
    if (ratio < NON_TEXT) failures.push(`${label} = ${ratio.toFixed(2)}:1 < ${NON_TEXT}:1  (${toHex(fg)} / ${toHex(bg)})`)
  }
  void fsLarge
}

// ── Categorical palette ──────────────────────────────────────────────────
const map = tokens.compact
const cats = [...map.keys()].filter(k => /^--cat-\d+$/.test(k)).sort()
if (cats.length !== 12) failures.push(`expected 12 --cat-* tokens, found ${cats.length}`)
const catColors = cats.map(k => ({ name: k, c: resolve(map, k) }))

for (let i = 0; i < catColors.length; i++) {
  for (let j = i + 1; j < catColors.length; j++) {
    const de = deltaE2000(catColors[i].c, catColors[j].c)
    if (de < CAT_MIN_DE) failures.push(`ΔE00 ${catColors[i].name} vs ${catColors[j].name} = ${de.toFixed(1)} < ${CAT_MIN_DE}`)
  }
  for (const other of CAT_VS) {
    const de = deltaE2000(catColors[i].c, resolve(map, other))
    if (de < CAT_MIN_DE) failures.push(`ΔE00 ${catColors[i].name} vs ${other} = ${de.toFixed(1)} < ${CAT_MIN_DE}`)
  }
  for (const bg of CAT_CONTRAST_BGS) {
    const ratio = contrast(catColors[i].c, resolve(map, bg))
    if (ratio < NON_TEXT) failures.push(`contrast ${catColors[i].name} on ${bg} = ${ratio.toFixed(2)} < ${NON_TEXT}`)
  }
}
for (const kind of CVD_KINDS) {
  const sim = catColors.map(({ name, c }) => ({ name, c: simulateCvd(c, kind) }))
  let min = Infinity, worst = ''
  for (let i = 0; i < sim.length; i++) for (let j = i + 1; j < sim.length; j++) {
    const de = deltaE2000(sim[i].c, sim[j].c)
    if (de < min) { min = de; worst = `${sim[i].name} vs ${sim[j].name}` }
  }
  report.push(`[cvd ${kind}] min pairwise ΔE00 among --cat-* = ${min.toFixed(1)} (${worst})`)
  if (min < CAT_MIN_DE_CVD) failures.push(`[cvd ${kind}] ${worst} ΔE00 = ${min.toFixed(1)} < ${CAT_MIN_DE_CVD}`)
}

if (process.argv.includes('--verbose')) console.log(report.join('\n'))
if (failures.length) {
  console.error(`check-contrast: ${failures.length} failure(s)\n  ` + failures.join('\n  '))
  process.exit(1)
}
console.log(`check-contrast: ok (${TEXT_PAIRS.length} text pairs + ${NON_TEXT_PAIRS.length} non-text pairs × 2 densities, 12 categorical colours, 3 CVD simulations)`)
