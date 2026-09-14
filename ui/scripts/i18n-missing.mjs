#!/usr/bin/env node
/**
 * i18n-missing.mjs — development helper: lists every `t('key')` /
 * `t.plural('key', …)` / `t(\`prefix.${x}\`)` usage under ui/src whose key
 * (or prefix, for template literals) is absent from es.json, and every
 * es.json key that no source file references (dead strings).
 * Not part of verify:design (check-strings covers the two languages);
 * run it while migrating: `node scripts/i18n-missing.mjs`.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(__dirname, '../src')
const es = JSON.parse(readFileSync(path.join(SRC, 'i18n/es.json'), 'utf8'))

function flatten(obj, prefix = '', out = new Set()) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object') flatten(v, key, out)
    else out.add(key)
  }
  return out
}
const keys = flatten(es)

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (/\.(tsx|ts)$/.test(name) && !p.includes('/i18n/')) out.push(p)
  }
  return out
}

const used = new Set()
const missing = []
for (const f of walk(SRC)) {
  const src = readFileSync(f, 'utf8')
  for (const m of src.matchAll(/\bt(?:\.plural)?\(\s*'([^']+)'/g)) {
    used.add(m[1])
    const plural = m[0].includes('.plural')
    const ok = plural ? (keys.has(`${m[1]}_other`)) : keys.has(m[1])
    if (!ok) missing.push(`${path.relative(process.cwd(), f)}: ${m[1]}${plural ? ' (plural)' : ''}`)
  }
  for (const m of src.matchAll(/\bt\(\s*`([^`$]+)\$\{/g)) {
    const prefix = m[1]
    const any = [...keys].some(k => k.startsWith(prefix))
    for (const k of keys) if (k.startsWith(prefix)) used.add(k)
    if (!any) missing.push(`${path.relative(process.cwd(), f)}: ${prefix}* (template)`)
  }
}
const unused = [...keys].filter(k => !used.has(k) && !/_(one|other)$/.test(k) && !/^(provenance|witness|edgeKind|pipelineStage|role|toolLabel)\./.test(k))
if (missing.length) console.log(`missing (${missing.length}):\n  ${missing.join('\n  ')}`)
if (unused.length) console.log(`unreferenced es.json keys (${unused.length}):\n  ${unused.join('\n  ')}`)
if (!missing.length && !unused.length) console.log('i18n-missing: every used key exists and every key is used')
process.exit(missing.length ? 1 : 0)
