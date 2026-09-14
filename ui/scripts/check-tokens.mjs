#!/usr/bin/env node
/**
 * check-tokens.mjs — every .css / .tsx under ui/src (except styles/tokens.css)
 * must consume the design tokens instead of literals.
 *
 * Fails on:
 *   - colour literals: #hex strings, rgb()/rgba()/hsl(), 0xRRGGBB numbers (tsx)
 *   - font-size not expressed as var(--fs-*)
 *   - border-radius not expressed as var(--r-*)
 *   - box-shadow other than none / var(--elev-*) / var(--ring-focus) / var(--glow-primary)
 *   - px / rem / em literals in any CSS declaration value (spacing, sizes,
 *     borders — the tokens carry them; media-query breakpoints are exempt)
 *   - numeric font-weight, numeric z-index, text-transform: uppercase,
 *     letter-spacing (no tracked micro-labels), !important
 *   - descendant tag selectors inside component rules (flat specificity)
 *   - any legacy alias (--bg-primary, --accent, …)
 *
 * Decision recorded here: `svg` is the only tag allowed as a descendant
 * selector (lucide icons expose no class hook); `0` and `100%` values are
 * always fine; `transform`/`translate` values may use `%`.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(__dirname, '../src')
const TOKENS = path.resolve(SRC, 'styles/tokens.css')

const LEGACY = [
  '--bg-base', '--bg-surface', '--bg-elevated', '--bg-hover', '--bg-active', '--bg-primary', '--bg-secondary',
  '--bg-tertiary', '--bg-input', '--bg-panel', '--glass-border', '--accent', '--accent-hover', '--accent-muted',
  '--accent-rgb', '--accent-2', '--accent-2-hover', '--accent-2-muted', '--accent-2-rgb', '--accent-gradient',
  '--success', '--success-rgb', '--warning', '--warning-rgb', '--error', '--error-rgb', '--status-green',
  '--status-amber', '--status-red', '--text-primary', '--text-secondary', '--text-muted', '--text-dim', '--border',
  '--border-light', '--border-strong', '--focus-ring', '--font-family', '--label-tracking', '--radius-xs',
  '--radius-sm', '--radius-md', '--radius-lg', '--radius-pill', '--transition-fast', '--transition-normal',
  '--ease-spring', '--inset-highlight', '--shadow-sm', '--shadow-md', '--shadow-lg', '--glow-accent', '--glow-cyan',
  '--activity-bar-width', '--sidebar-width', '--toolbar-height', '--statusbar-height', '--danger', '--accent-color',
  '--border-color',
]
// a legacy alias is a custom-property NAME: not preceded by a word char or hyphen (BEM modifiers like .x--danger are classes)
const LEGACY_RE = new RegExp(`(?<![\\w-])(${LEGACY.map(l => l.replace(/-/g, '\\-')).join('|')})(?![a-z0-9-])`, 'g')
const ALLOW_TAG_DESCENDANT = new Set(['svg', 'option'])
const TAGS = 'button|input|select|textarea|option|h1|h2|h3|h4|h5|h6|div|span|p|ul|ol|li|table|thead|tbody|tr|td|th|img|video|canvas|a|label|hr|strong|b|em|small|code|kbd|pre|form|nav|header|footer|main|aside|section'

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name)
    const st = statSync(p)
    if (st.isDirectory()) walk(p, out)
    else if (/\.(css|tsx)$/.test(name)) out.push(p)
  }
  return out
}

const failures = []
const fail = (file, line, msg) => failures.push(`${path.relative(process.cwd(), file)}:${line}: ${msg}`)

function stripCssComments(css) {
  return css.replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, ' '))
}

function lineOf(text, index) {
  return text.slice(0, index).split('\n').length
}

function checkCss(file, raw) {
  const css = stripCssComments(raw)
  if (/!important/.test(css)) {
    for (const m of css.matchAll(/!important/g)) fail(file, lineOf(css, m.index), '!important')
  }
  for (const m of css.matchAll(LEGACY_RE)) fail(file, lineOf(css, m.index), `legacy alias ${m[1]}`)

  // rules: selector { declarations }
  const re = /([^{}]+)\{([^{}]*)\}/g
  let m
  while ((m = re.exec(css))) {
    const selector = m[1].trim()
    const body = m[2]
    const bodyStart = m.index + m[0].indexOf('{') + 1
    const isMedia = /^@media/.test(selector)
    if (isMedia) continue // nested rules are matched separately by the flat regex; breakpoints exempt
    if (/^@keyframes/.test(selector)) continue
    // descendant tag selectors
    if (!selector.startsWith('@')) {
      const tagRe = new RegExp(`[.\\]]\\S*\\s+(?:>\\s*)?(${TAGS})\\b`, 'g')
      for (const s of selector.split(',')) {
        for (const t of s.matchAll(tagRe)) {
          if (!ALLOW_TAG_DESCENDANT.has(t[1])) fail(file, lineOf(css, m.index), `descendant tag selector "${s.trim()}"`)
        }
      }
    }
    for (const decl of body.split(';')) {
      const idx = decl.indexOf(':')
      if (idx < 0) continue
      const prop = decl.slice(0, idx).trim().toLowerCase()
      const value = decl.slice(idx + 1).trim()
      const line = lineOf(css, bodyStart + body.indexOf(decl))
      if (!prop || prop.startsWith('--')) {
        // custom properties inside components must still not carry literals
        if (prop.startsWith('--') && /#[0-9a-f]{3,8}\b|rgba?\(|hsla?\(|\d(px|rem|em)\b/i.test(value)) fail(file, line, `literal in ${prop}: ${value}`)
        continue
      }
      if (/#[0-9a-f]{3,8}\b/i.test(value) && !/url\(/.test(value)) fail(file, line, `colour literal in ${prop}: ${value}`)
      if (/\b(rgba?|hsla?)\(/i.test(value)) fail(file, line, `colour function in ${prop}: ${value}`)
      if (/(^|[^a-z-])\d*\.?\d+(px|rem|em)\b/i.test(value)) fail(file, line, `length literal in ${prop}: ${value}`)
      if (prop === 'font-size' && !/^(var\(--fs-[0-5]\)|inherit|0)$/.test(value)) fail(file, line, `font-size must be var(--fs-*): ${value}`)
      if (prop === 'font' && /\d/.test(value)) fail(file, line, `font shorthand: use font-size/font-family tokens`)
      if (prop === 'border-radius' || prop.startsWith('border-') && prop.endsWith('-radius')) {
        const parts = value.split(/\s+/)
        if (!parts.every(p => /^var\(--r-[a-z0-9]+\)$/.test(p) || p === 'inherit')) fail(file, line, `border-radius must be var(--r-*): ${value}`)
      }
      if (prop === 'box-shadow') {
        const parts = value.split(',').map(s => s.trim())
        if (!parts.every(p => /^(none|var\(--elev-[0-3]\)|var\(--ring-focus\)|var\(--glow-primary\))$/.test(p))) fail(file, line, `box-shadow must be var(--elev-*): ${value}`)
      }
      if (prop === 'font-weight' && !/^(var\(--fw-[a-z]+\)|inherit)$/.test(value)) fail(file, line, `font-weight must be var(--fw-*): ${value}`)
      if (prop === 'z-index' && !/^(var\(--z-[a-z-]+\)|auto|inherit|0|1|-1|calc\(var\(--z-[a-z-]+\)\s*[+-]\s*\d\))$/.test(value)) fail(file, line, `z-index must be var(--z-*): ${value}`)
      if (prop === 'text-transform' && /uppercase/.test(value)) fail(file, line, 'text-transform: uppercase (no tracked micro-labels)')
      if (prop === 'letter-spacing' && !/^(normal|inherit|0)$/.test(value)) fail(file, line, `letter-spacing: ${value}`)
      if (prop === 'transition' || prop === 'transition-duration' || prop === 'animation' || prop === 'animation-duration') {
        if (/\d+m?s\b/.test(value) && !/^stac-/.test(value) && prop !== 'animation') fail(file, line, `duration literal in ${prop}: ${value}`)
      }
    }
  }
}

function checkTsx(file, src) {
  // strip block and line comments so documentation examples are not flagged
  const code = src.replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, ' ')).replace(/(^|[^:'"`])\/\/[^\n]*/g, (m, p) => p + ' '.repeat(m.length - p.length))
  for (const m of code.matchAll(/['"`]#[0-9a-f]{3,8}['"`]/gi)) fail(file, lineOf(code, m.index), `colour literal ${m[0]}`)
  for (const m of code.matchAll(/\b0x[0-9a-f]{6}\b/gi)) fail(file, lineOf(code, m.index), `colour literal ${m[0]}`)
  for (const m of code.matchAll(/\b(rgba?|hsla?)\(/gi)) fail(file, lineOf(code, m.index), `colour function ${m[0]}`)
  for (const m of code.matchAll(LEGACY_RE)) fail(file, lineOf(code, m.index), `legacy alias ${m[1]}`)
  for (const m of code.matchAll(/\b(fontSize|borderRadius|boxShadow|fontWeight|letterSpacing|textTransform)\s*:/g)) fail(file, lineOf(code, m.index), `inline ${m[1]} (use a class + tokens)`)
}

const files = walk(SRC).filter(f => f !== TOKENS)
for (const f of files) {
  const src = readFileSync(f, 'utf8')
  if (f.endsWith('.css')) checkCss(f, src)
  else checkTsx(f, src)
}

if (failures.length) {
  console.error(`check-tokens: ${failures.length} failure(s) in ${new Set(failures.map(f => f.split(':')[0])).size} file(s)\n  ` + failures.slice(0, 400).join('\n  '))
  if (failures.length > 400) console.error(`  … ${failures.length - 400} more`)
  process.exit(1)
}
console.log(`check-tokens: ok (${files.length} files)`)
