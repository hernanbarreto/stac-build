/**
 * i18n — Spanish (Mexico) by default, English as the second language.
 *
 * `useT()` returns a stable translator: `t('path.to.key', { name })`.
 * Placeholders: `{name}` from vars, `{@term}` / `{@Term}` / `{@terms}` from
 * the glossary (singular, capitalised, plural). Plurals use two keys per
 * message (`key_one`, `key_other`) selected with Intl.PluralRules.
 *
 * Numbers, dates and lengths go through Intl with the active locale, so a
 * measurement is never rendered without its unit (§9): `fmt.length(1.234)`
 * gives `{ value: '1.234', unit: 'm' }`, `fmt.length(0.087)` → `8.7 cm`.
 *
 * Language and density are the only preferences kept in localStorage (§15).
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import es from './es.json'
import en from './en.json'
import { GLOSSARY, GLOSSARY_PLURAL, type GlossaryTerm } from './glossary'

export type Lang = 'es' | 'en'
export const LOCALES: Record<Lang, string> = { es: 'es-MX', en: 'en-US' }
const STORAGE_KEY = 'stac.lang'
const DICTS: Record<Lang, Record<string, unknown>> = { es, en }

type Vars = Record<string, string | number | null | undefined>

function lookup(dict: Record<string, unknown>, key: string): string | undefined {
  let cur: unknown = dict
  for (const part of key.split('.')) {
    if (cur == null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[part]
  }
  return typeof cur === 'string' ? cur : undefined
}

function glossaryWord(term: string, lang: Lang): string | undefined {
  const plural = term.endsWith('s') && !(term in GLOSSARY)
  const base = (plural ? term.slice(0, -1) : term) as GlossaryTerm
  const cap = /^[A-Z]/.test(base)
  const key = (cap ? base[0].toLowerCase() + base.slice(1) : base) as GlossaryTerm
  const entry = GLOSSARY[key]
  if (!entry) return undefined
  let word: string = plural ? (GLOSSARY_PLURAL[key]?.[lang] ?? entry[lang] + 's') : entry[lang]
  if (cap) word = word[0].toUpperCase() + word.slice(1)
  return word
}

export function interpolate(template: string, lang: Lang, vars?: Vars): string {
  return template.replace(/\{(@?[A-Za-z_][\w]*)\}/g, (whole, name: string) => {
    if (name.startsWith('@')) return glossaryWord(name.slice(1), lang) ?? whole
    const v = vars?.[name]
    return v == null ? '' : String(v)
  })
}

export interface Formatters {
  locale: string
  number: (n: number, digits?: number) => string
  integer: (n: number) => string
  percent: (fraction: number, digits?: number) => string
  /** metres in → best unit out; digits are fixed per unit (mm 0, cm 1, m 3) */
  length: (metres: number) => { value: string; unit: 'mm' | 'cm' | 'm' }
  lengthText: (metres: number) => string
  /** millimetres in (deviations, tolerances) */
  mm: (millimetres: number, digits?: number) => string
  degrees: (deg: number, digits?: number) => string
  area: (m2: number) => string
  volume: (m3: number) => string
  bytesMb: (mb: number) => string
  millions: (n: number) => string
  date: (d: Date | string | number) => string
  dateTime: (d: Date | string | number) => string
  time: (d: Date | string | number) => string
  duration: (seconds: number) => string
}

export function makeFormatters(lang: Lang): Formatters {
  const locale = LOCALES[lang]
  const nf = (digits: number) => new Intl.NumberFormat(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits })
  const int = new Intl.NumberFormat(locale, { maximumFractionDigits: 0 })
  const toDate = (d: Date | string | number) => (d instanceof Date ? d : new Date(d))
  const date = new Intl.DateTimeFormat(locale, { year: 'numeric', month: 'short', day: 'numeric' })
  const dateTime = new Intl.DateTimeFormat(locale, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  const time = new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit' })
  const safe = (f: Intl.DateTimeFormat, d: Date | string | number) => {
    const dt = toDate(d)
    return Number.isNaN(dt.getTime()) ? String(d) : f.format(dt)
  }
  const length = (m: number) => {
    const abs = Math.abs(m)
    if (abs < 0.01) return { value: nf(0).format(m * 1000), unit: 'mm' as const }
    if (abs < 1) return { value: nf(1).format(m * 100), unit: 'cm' as const }
    return { value: nf(3).format(m), unit: 'm' as const }
  }
  return {
    locale,
    number: (n, digits = 2) => nf(digits).format(n),
    integer: n => int.format(n),
    percent: (f, digits = 0) => `${nf(digits).format(f * 100)} %`,
    length,
    lengthText: m => { const l = length(m); return `${l.value} ${l.unit}` },
    mm: (v, digits = 1) => `${nf(digits).format(v)} mm`,
    degrees: (v, digits = 1) => `${nf(digits).format(v)}°`,
    area: v => `${nf(2).format(v)} m²`,
    volume: v => `${nf(2).format(v)} m³`,
    bytesMb: v => `${nf(v >= 100 ? 0 : 1).format(v)} MB`,
    millions: n => `${nf(n >= 10 ? 0 : 1).format(n / 1e6)} M`,
    date: d => safe(date, d),
    dateTime: d => safe(dateTime, d),
    time: d => safe(time, d),
    duration: s => {
      if (s < 60) return `${int.format(Math.round(s))} s`
      if (s < 3600) return `${int.format(Math.floor(s / 60))} min ${int.format(Math.round(s % 60))} s`
      return `${int.format(Math.floor(s / 3600))} h ${int.format(Math.round((s % 3600) / 60))} min`
    },
  }
}

export interface Translator {
  (key: string, vars?: Vars): string
  plural: (key: string, count: number, vars?: Vars) => string
  has: (key: string) => boolean
  lang: Lang
}

export function makeTranslator(lang: Lang): Translator {
  const dict = DICTS[lang]
  const rules = new Intl.PluralRules(LOCALES[lang])
  const t = ((key: string, vars?: Vars) => {
    const raw = lookup(dict, key) ?? lookup(DICTS.en, key)
    if (raw == null) {
      if (import.meta.env.DEV) console.warn(`[i18n] missing key "${key}" (${lang})`)
      return key
    }
    return interpolate(raw, lang, vars)
  }) as Translator
  t.plural = (key, count, vars) => {
    const cat = rules.select(count)
    const k = lookup(dict, `${key}_${cat}`) != null ? `${key}_${cat}` : `${key}_other`
    return t(k, { count, ...vars })
  }
  t.has = key => lookup(dict, key) != null
  t.lang = lang
  return t
}

interface I18nContextValue {
  lang: Lang
  setLang: (l: Lang) => void
  t: Translator
  fmt: Formatters
}

const I18nContext = createContext<I18nContextValue>({
  lang: 'es',
  setLang: () => {},
  t: makeTranslator('es'),
  fmt: makeFormatters('es'),
})

export function readStoredLang(): Lang {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'es' || v === 'en') return v
  } catch { /* storage unavailable */ }
  return 'es'
}

// Module-level accessors for non-React code (Three.js helpers, canvas
// labels). The provider keeps them in sync with the active language.
let currentT: Translator = makeTranslator(readStoredLang())
let currentFmt: Formatters = makeFormatters(readStoredLang())
export function getT(): Translator { return currentT }
export function getFmt(): Formatters { return currentFmt }

export function I18nProvider({ children, initialLang }: { children: ReactNode; initialLang?: Lang }) {
  const [lang, setLangState] = useState<Lang>(initialLang ?? readStoredLang)
  const setLang = useCallback((l: Lang) => {
    setLangState(l)
    try { localStorage.setItem(STORAGE_KEY, l) } catch { /* storage unavailable */ }
  }, [])
  useEffect(() => { document.documentElement.lang = LOCALES[lang] }, [lang])
  const value = useMemo<I18nContextValue>(() => {
    const t = makeTranslator(lang), fmt = makeFormatters(lang)
    currentT = t; currentFmt = fmt
    return { lang, setLang, t, fmt }
  }, [lang, setLang])
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}

export function useT(): Translator {
  return useContext(I18nContext).t
}

export function useFmt(): Formatters {
  return useContext(I18nContext).fmt
}
