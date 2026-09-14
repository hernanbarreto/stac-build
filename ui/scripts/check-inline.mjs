#!/usr/bin/env node
/**
 * check-inline.mjs — no inline styles except CSS custom properties.
 *
 * Every `style={…}` in a .tsx under ui/src must be an object literal whose
 * keys are all CSS variables (`'--w': …`), so the value is dynamic but the
 * rule lives in a stylesheet. Anything else (a plain property, a spread, a
 * variable reference, a conditional) fails.
 *
 * Also fails on imperative `el.style.<prop> =` writes for presentation
 * properties (background, color, fontSize, …); geometry writes that mirror a
 * canvas buffer (width/height/cursor/transform/display) are allowed because
 * they are computed from the DOM at runtime, not design decisions.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(__dirname, '../src')
const ALLOWED_IMPERATIVE = new Set(['width', 'height', 'cursor', 'transform', 'display', 'userSelect', 'left', 'top'])

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (name.endsWith('.tsx')) out.push(p)
  }
  return out
}

const failures = []
const lineOf = (text, i) => text.slice(0, i).split('\n').length

/** Extract the balanced `{…}` starting at index `start` (which points at '{'). */
function balanced(src, start) {
  let depth = 0
  for (let i = start; i < src.length; i++) {
    const ch = src[i]
    if (ch === '{') depth++
    else if (ch === '}') { depth--; if (depth === 0) return src.slice(start, i + 1) }
    else if (ch === '\'' || ch === '"' || ch === '`') {
      const q = ch
      i++
      while (i < src.length && src[i] !== q) { if (src[i] === '\\') i++; i++ }
    }
  }
  return src.slice(start)
}

/** Top-level keys of an object literal text `{ a: 1, 'b': 2 }`. */
function topLevelKeys(objText) {
  const inner = objText.slice(1, -1)
  const keys = []
  let depth = 0, token = '', i = 0
  const flush = () => {
    const t = token.trim()
    if (!t) return
    if (t.startsWith('...')) { keys.push('...spread'); return }
    const m = t.match(/^(?:'([^']*)'|"([^"]*)"|\[([^\]]*)\]|([A-Za-z_$][\w$]*))\s*:/)
    keys.push(m ? (m[1] ?? m[2] ?? (m[3] != null ? `[${m[3]}]` : m[4])) : '?')
  }
  for (; i < inner.length; i++) {
    const ch = inner[i]
    if (ch === '\'' || ch === '"' || ch === '`') {
      const q = ch
      token += ch; i++
      while (i < inner.length && inner[i] !== q) { if (inner[i] === '\\') { token += inner[i]; i++ } token += inner[i]; i++ }
      token += inner[i] ?? ''
      continue
    }
    if (ch === '{' || ch === '(' || ch === '[') depth++
    if (ch === '}' || ch === ')' || ch === ']') depth--
    if (ch === ',' && depth === 0) { flush(); token = ''; continue }
    token += ch
  }
  flush()
  return keys
}

for (const file of walk(SRC)) {
  const src = readFileSync(file, 'utf8')
  const rel = path.relative(process.cwd(), file)
  for (const m of src.matchAll(/\sstyle=\{/g)) {
    const braceIdx = m.index + m[0].length - 1
    const outer = balanced(src, braceIdx)          // {{ … }} or {expr}
    const inner = outer.slice(1, -1).trim()
    const line = lineOf(src, m.index)
    if (!inner.startsWith('{')) { failures.push(`${rel}:${line}: style={${inner.slice(0, 40)}…} is not an object literal`); continue }
    const objText = balanced(inner, 0)
    const keys = topLevelKeys(objText)
    const bad = keys.filter(k => !/^--[a-z0-9-]+$/i.test(k))
    if (bad.length) failures.push(`${rel}:${line}: inline style property ${bad.map(b => JSON.stringify(b)).join(', ')} (only CSS variables allowed)`)
  }
  for (const m of src.matchAll(/\.style\.([A-Za-z]+)\s*=/g)) {
    if (!ALLOWED_IMPERATIVE.has(m[1])) failures.push(`${rel}:${lineOf(src, m.index)}: imperative style.${m[1]} write`)
  }
}

if (failures.length) {
  console.error(`check-inline: ${failures.length} failure(s)\n  ` + failures.slice(0, 300).join('\n  '))
  if (failures.length > 300) console.error(`  … ${failures.length - 300} more`)
  process.exit(1)
}
console.log('check-inline: ok')
