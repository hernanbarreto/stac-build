#!/usr/bin/env node
/**
 * purgecss-report.mjs — REPORT mode only (prompt_ui.txt §10): lists the CSS
 * selectors under ui/src that PurgeCSS cannot find referenced from any .tsx
 * / .ts / .html source. Nothing is deleted; the list is for a human to act
 * on. Dynamic class names (BEM modifiers built with template strings) are
 * declared as safelist patterns so they are not reported as dead.
 */
import { PurgeCSS } from 'purgecss'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(__dirname, '../src')

function walk(dir, ext, out = []) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name)
    if (statSync(p).isDirectory()) walk(p, ext, out)
    else if (ext.test(name)) out.push(p)
  }
  return out
}

const cssFiles = walk(SRC, /\.css$/)
const content = [...walk(SRC, /\.(tsx|ts)$/), path.resolve(__dirname, '../index.html'), path.resolve(__dirname, '../xr.html')]
const results = await new PurgeCSS().purge({
  content,
  css: cssFiles,
  rejected: true,
  safelist: {
    standard: [/^stac-/, /^xr-/, /^viewport-/, /^btn/, /^chip/, /^session-card/, /^primary-cta/, /^spin$/],
    greedy: [/--/, /__/],
  },
  variables: false,
})
let total = 0
for (const r of results) {
  const rejected = (r.rejected || []).filter(sel => !sel.startsWith('::-webkit') && !sel.startsWith(':'))
  if (!rejected.length) continue
  total += rejected.length
  console.log(`${path.relative(process.cwd(), r.file)} (${rejected.length} unreferenced selectors)`)
  for (const sel of rejected) console.log(`  ${sel}`)
}
console.log(total ? `purgecss-report: ${total} selector(s) not found in the sources (report only, nothing removed)` : 'purgecss-report: every selector is referenced')
