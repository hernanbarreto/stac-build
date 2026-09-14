#!/usr/bin/env node
/**
 * check-strings.mjs — every visible string lives in ui/src/i18n.
 *
 * Fails on:
 *   1. JSX text nodes with letters outside src/i18n (e.g. `<span>Load</span>`);
 *      `{t('key')}` expressions are the way in. Punctuation-only nodes and
 *      units passed through `fmt*` helpers are fine.
 *   2. Literal strings in user-facing attributes: title, placeholder, alt,
 *      aria-label, aria-description, label= (component prop), tooltip=,
 *      confirmLabel=, cancelLabel=, emptyText=, description=.
 *   3. Emoji (Extended_Pictographic) or icon-like glyphs (✕ ✓ ★ ☆ ▶ ◀ ▼ ▲ ⟲ ↩
 *      ⌃ ● ○ ◼ ⬚ ↔ ⤢ ⊘ ⇩ ⛶ ⚠ · etc.) anywhere under ui/src (tsx, ts, css, json),
 *      including the i18n JSON files: icons come from lucide-react.
 *   4. Keys present in es.json and missing in en.json or vice versa, and
 *      placeholders `{name}` that differ between the two languages.
 *   5. Glossary terms (i18n/glossary.ts) written literally instead of via
 *      `{@term}` in either JSON (one concept, one word — §8).
 *
 * Non-visible strings (fetch URLs, JSON keys, class names, CSS values,
 * console messages, `key=`) are not user-facing and are not checked.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SRC = path.resolve(__dirname, '../src')
const I18N = path.resolve(SRC, 'i18n')

const GLYPHS = '✕✓✔✗★☆▶◀▼▲△▽⟲↩↪⌃●○◼◻⬚⬛⬜↔↕⤢⊘⇩⇧⇦⇨⛶⚠⚙•·▪▸▹◂◃➤➕➖⏳⏸⏹⏵⏴☁☑☒✖✚✦✧⧉⊿∠⌀⌂⏏';
const GLYPH_RE = new RegExp(`[${GLYPHS}]|\\p{Extended_Pictographic}`, 'u')
const ATTRS = ['title', 'placeholder', 'alt', 'aria-label', 'aria-description', 'label', 'tooltip', 'confirmLabel', 'cancelLabel', 'emptyText', 'description', 'hint', 'unit']

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (/\.(tsx|ts|css|json)$/.test(name)) out.push(p)
  }
  return out
}

const failures = []
const lineOf = (text, i) => text.slice(0, i).split('\n').length
const rel = f => path.relative(process.cwd(), f)

// ── 3. glyphs everywhere ────────────────────────────────────────────────
for (const file of walk(SRC)) {
  const src = readFileSync(file, 'utf8')
  src.split('\n').forEach((line, i) => {
    const m = line.match(GLYPH_RE)
    if (m) failures.push(`${rel(file)}:${i + 1}: icon glyph / emoji "${m[0]}" (use lucide-react)`)
  })
}

// ── 1 + 2. JSX text and attributes outside i18n ─────────────────────────
function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, ' '))
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, m => m.replace(/[^\n]/g, ' '))
    .replace(/(^|[^:'"`\\])\/\/[^\n]*/g, (m, p) => p + ' '.repeat(m.length - p.length))
}

for (const file of walk(SRC).filter(f => f.endsWith('.tsx') && !f.startsWith(I18N))) {
  const src = stripComments(readFileSync(file, 'utf8'))
  // JSX text nodes: `>text<` where text has letters and is not an expression
  for (const m of src.matchAll(/>([^<>{}]*[A-Za-zÀ-ÿ][^<>{}]*)</g)) {
    const text = m[1].trim()
    if (!text) continue
    // ignore generics / arrow returns that the regex mistakes for tags
    if (/^[=(]/.test(text) || /=>/.test(m[0])) continue
    if (/^[A-Za-z]+\.[A-Za-z]/.test(text) && !/\s/.test(text)) continue
    failures.push(`${rel(file)}:${lineOf(src, m.index)}: JSX text "${text.slice(0, 50)}" outside i18n`)
  }
  for (const attr of ATTRS) {
    const re = new RegExp(`\\s${attr}=(?:"([^"]*[A-Za-z][^"]*)"|'([^']*[A-Za-z][^']*)'|\\{\\s*(?:'([^']*[A-Za-z][^']*)'|"([^"]*[A-Za-z][^"]*)"|\`([^\`]*[A-Za-z][^\`]*)\`)\\s*\\})`, 'g')
    for (const m of src.matchAll(re)) {
      const val = m[1] ?? m[2] ?? m[3] ?? m[4] ?? m[5]
      if (/^[a-z0-9_./-]+$/i.test(val) && !/\s/.test(val) && attr !== 'placeholder' && attr !== 'title') continue // identifiers (unit codes, keys)
      failures.push(`${rel(file)}:${lineOf(src, m.index)}: literal ${attr}="${val.slice(0, 50)}" outside i18n`)
    }
  }
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
const placeholders = s => [...s.matchAll(/\{(@?[A-Za-z_][\w]*)(?:,[^}]*)?\}/g)].map(m => m[1]).sort().join(',')

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
  // glossary: terms must be referenced as {@term}, never spelled literally
  const gl = readFileSync(path.join(I18N, 'glossary.ts'), 'utf8')
  const terms = [...gl.matchAll(/^\s{2}([a-z_]+):\s*\{\s*es:\s*'([^']+)',\s*en:\s*'([^']+)'/gm)].map(m => ({ key: m[1], es: m[2], en: m[3] }))
  const check = (lang, dict) => {
    for (const [k, v] of Object.entries(dict)) {
      for (const t of terms) {
        const word = t[lang]
        const re = new RegExp(`(^|[^\\p{L}@])${word}(s|es)?(?![\\p{L}])`, 'iu')
        if (re.test(v)) failures.push(`i18n ${lang}.json "${k}": glossary term "${word}" spelled literally — use {@${t.key}}`)
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
