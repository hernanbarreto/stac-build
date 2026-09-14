/**
 * Reads ui/src/styles/tokens.css and resolves every custom property to a
 * concrete value (var() references and the single color-mix() form used by
 * the tokens are evaluated). Returns one map per density.
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { parseColor, mixSrgb, toHex } from './color-math.mjs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
export const TOKENS_PATH = path.resolve(__dirname, '../src/styles/tokens.css')

function collectBlocks(css) {
  // returns [{ selector, declarations: Map }]
  const blocks = []
  const re = /([^{}]+)\{([^{}]*)\}/g
  let m
  while ((m = re.exec(css))) {
    const selector = m[1].trim()
    const decls = new Map()
    for (const line of m[2].split(';')) {
      const idx = line.indexOf(':')
      if (idx < 0) continue
      const name = line.slice(0, idx).trim()
      const value = line.slice(idx + 1).trim()
      if (name.startsWith('--')) decls.set(name, value)
    }
    blocks.push({ selector, declarations: decls })
  }
  return blocks
}

function resolveValue(value, map, depth = 0) {
  if (depth > 20) throw new Error(`token recursion too deep: ${value}`)
  let out = value
  // var(--x) / var(--x, fallback)
  out = out.replace(/var\((--[a-z0-9-]+)(?:\s*,\s*([^)]+))?\)/gi, (_, name, fallback) => {
    const v = map.get(name)
    if (v == null) {
      if (fallback != null) return fallback
      throw new Error(`unknown token ${name} referenced by "${value}"`)
    }
    return resolveValue(v, map, depth + 1)
  })
  // color-mix(in srgb, A p%, B)
  const cm = out.match(/^color-mix\(in srgb,\s*(.+?)\s+(\d+(?:\.\d+)?)%\s*,\s*(.+)\)$/)
  if (cm) {
    const a = parseColor(cm[1]), b = parseColor(cm[3])
    out = toHex(mixSrgb(a, Number(cm[2]) / 100, b))
  }
  return out
}

/** @returns {{ compact: Map<string,string>, comfortable: Map<string,string>, raw: string }} */
export function readTokens() {
  const css = readFileSync(TOKENS_PATH, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')
  const blocks = collectBlocks(css)
  const base = new Map()
  const perDensity = { compact: new Map(), comfortable: new Map() }
  for (const b of blocks) {
    if (b.selector === ':root') for (const [k, v] of b.declarations) base.set(k, v)
    for (const d of Object.keys(perDensity)) {
      if (b.selector.includes(`data-density='${d}'`)) for (const [k, v] of b.declarations) perDensity[d].set(k, v)
    }
  }
  const result = { raw: css }
  for (const d of Object.keys(perDensity)) {
    const merged = new Map([...base, ...perDensity[d]])
    const resolved = new Map()
    for (const [k, v] of merged) resolved.set(k, resolveValue(v, merged))
    result[d] = resolved
  }
  return result
}

export function tokenNames(map, prefix) {
  return [...map.keys()].filter(k => k.startsWith(prefix))
}
