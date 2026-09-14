#!/usr/bin/env node
/**
 * check-strings.mjs — every visible string lives in ui/src/i18n.
 *
 * Uses the TypeScript compiler API (already a devDependency) so JSX is parsed,
 * not guessed:
 *   1. JsxText nodes with letters outside src/i18n fail (`<span>Load</span>`);
 *      `{t('key')}` expressions are the way in.
 *   2. User-facing attributes (title, placeholder, alt, aria-label,
 *      aria-description, label, tooltip, confirmLabel, cancelLabel, emptyText,
 *      description, hint) with a string literal, or a template literal that
 *      does not call `t(`, fail. Identifier-like values (unit codes, keys)
 *      without spaces are allowed except for title/placeholder.
 *   3. Emoji (Extended_Pictographic) or icon-like glyphs anywhere under ui/src
 *      (tsx, ts, css, json) — icons come from lucide-react; the middle dot is
 *      not a separator.
 *   4. es.json / en.json key sets and placeholders must match.
 *   5. Glossary terms (i18n/glossary.ts) must be written as `{@term}` in both
 *      JSON files, never literally (one concept, one word — §8).
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(__dirname, '../src')
const I18N = path.resolve(SRC, 'i18n')

const GLYPHS = '✕✓✔✗★☆▶◀▼▲△▽⟲↩↪⌃●○◼◻⬚⬛⬜↔↕⤢⊘⇩⇧⇦⇨⛶⚠⚙•·▪▸▹◂◃➤➕➖⏳⏸⏹⏵⏴☁☑☒✖✚✦✧⧉⊿∠⌀⌂⏏→←'
const GLYPH_RE = new RegExp(`[${GLYPHS}]|\\p{Extended_Pictographic}`, 'u')
const ATTRS = new Set(['title', 'placeholder', 'alt', 'aria-label', 'aria-description', 'label', 'tooltip', 'confirmLabel', 'cancelLabel', 'emptyText', 'description', 'hint'])
const IDENT_OK = new Set(['label', 'alt', 'aria-label', 'hint', 'description', 'tooltip', 'confirmLabel', 'cancelLabel', 'emptyText', 'aria-description'])

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (/\.(tsx|ts|css|json)$/.test(name)) out.push(p)
  }
  return out
}

const failures = []
const rel = f => path.relative(process.cwd(), f)

// ── 3. glyphs everywhere ────────────────────────────────────────────────
for (const file of walk(SRC)) {
  const src = readFileSync(file, 'utf8')
  src.split('\n').forEach((line, i) => {
    const m = line.match(GLYPH_RE)
    if (m) failures.push(`${rel(file)}:${i + 1}: icon glyph / emoji "${m[0]}" (use lucide-react)`)
  })
}

// ── 1 + 2. JSX text and attributes outside i18n (TypeScript AST) ────────
for (const file of walk(SRC).filter(f => f.endsWith('.tsx') && !f.startsWith(I18N))) {
  const src = readFileSync(file, 'utf8')
  const sf = ts.createSourceFile(file, src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  const lineOf = node => sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1
  const visit = node => {
    if (ts.isJsxText(node)) {
      const text = node.getText(sf).trim()
      if (/[A-Za-zÀ-ÿ]/.test(text)) failures.push(`${rel(file)}:${lineOf(node)}: JSX text "${text.slice(0, 50)}" outside i18n`)
    }
    if (ts.isJsxAttribute(node) && node.initializer) {
      const name = node.name.getText(sf)
      if (ATTRS.has(name)) {
        const init = node.initializer
        let literal = null
        if (ts.isStringLiteral(init)) literal = init.text
        else if (ts.isJsxExpression(init) && init.expression) {
          const e = init.expression
          if (ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) literal = e.text
          else if (ts.isTemplateExpression(e) && !/\bt\(|\btt\(|\bT\(|getT\(\)/.test(e.getText(sf))) literal = e.getText(sf)
        }
        if (literal != null && /[A-Za-zÀ-ÿ]/.test(literal)) {
          const identLike = /^[A-Za-z0-9_./-]+$/.test(literal) && IDENT_OK.has(name)
          if (!identLike) failures.push(`${rel(file)}:${lineOf(node)}: literal ${name}="${literal.slice(0, 50)}" outside i18n`)
        }
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(sf)
}

// ── 4 + 5. es.json vs en.json ───────────────────────────────────────────
function flatten(obj, prefix = '', out = {}) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object') flatten(v, key, out)
    else out[key] = String(v)
  }
  return out
}
// glossary refs compare by base term: {@Scans} and {@scan} are the same concept in another grammar
const placeholders = s => [...s.matchAll(/\{(@?[A-Za-z_][\w]*)(?:,[^}]*)?\}/g)].map(m => (m[1].startsWith('@') ? '@' + m[1].slice(1).toLowerCase().replace(/s$/, '') : m[1])).sort().join(',')

let es, en
try {
  es = flatten(JSON.parse(readFileSync(path.join(I18N, 'es.json'), 'utf8')))
  en = flatten(JSON.parse(readFileSync(path.join(I18N, 'en.json'), 'utf8')))
} catch (e) {
  failures.push(`i18n: cannot read es.json / en.json (${e.message})`)
}
if (es && en) {
  for (const k of Object.keys(es)) if (!(k in en)) failures.push(`i18n: key "${k}" is in es.json but not in en.json`)
  for (const k of Object.keys(en)) if (!(k in es)) failures.push(`i18n: key "${k}" is in en.json but not in es.json`)
  for (const k of Object.keys(es)) {
    if (k in en && placeholders(es[k]) !== placeholders(en[k])) failures.push(`i18n: placeholders differ for "${k}": es {${placeholders(es[k])}} vs en {${placeholders(en[k])}}`)
  }
  const gl = readFileSync(path.join(I18N, 'glossary.ts'), 'utf8')
  const terms = [...gl.matchAll(/^\s{2}([a-z_]+):\s*\{\s*es:\s*'([^']+)',\s*en:\s*'([^']+)'/gm)].map(m => ({ key: m[1], es: m[2], en: m[3] }))
  const check = (lang, dict) => {
    for (const [k, v] of Object.entries(dict)) {
      for (const term of terms) {
        const word = term[lang]
        const re = new RegExp(`(^|[^\\p{L}@])${word}(s|es)?(?![\\p{L}])`, 'iu')
        if (re.test(v.replace(/\{[^}]*\}/g, ' '))) failures.push(`i18n ${lang}.json "${k}": glossary term "${word}" spelled literally — use {@${term.key}}`)
      }
    }
  }
  check('es', es); check('en', en)
}

if (failures.length) {
  console.error(`check-strings: ${failures.length} failure(s)\n  ` + failures.slice(0, 400).join('\n  '))
  if (failures.length > 400) console.error(`  … ${failures.length - 400} more`)
  process.exit(1)
}
console.log('check-strings: ok')
