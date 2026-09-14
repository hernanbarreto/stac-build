/**
 * Glossary — one concept, one word, in both languages (prompt_ui.txt §8).
 *
 * Messages in es.json / en.json reference a term as `{@term}` (lower case) or
 * `{@Term}` (capitalised); `check-strings.mjs` fails when a term is spelled
 * literally in either JSON, so the vocabulary cannot drift.
 *
 * Decision: the backend calls the top-level folder a "session" (`/sessions`)
 * and the multi-scan container a "project" (`/api/project/<sid>`); they are
 * the same object. The UI uses exactly ONE word for it — {@session} — and
 * never says "project". A {@scan} is one dated capture inside a session; a
 * {@campaign} is the set of scans compared over time.
 */
export const GLOSSARY = {
  session: { es: 'sesión', en: 'session' },
  scan: { es: 'escaneo', en: 'scan' },
  campaign: { es: 'campaña', en: 'campaign' },
  instance: { es: 'instancia', en: 'instance' },
  epoch: { es: 'época geométrica', en: 'geometric epoch' },
  acta: { es: 'acta', en: 'certificate' },
  correction: { es: 'corrección', en: 'correction' },
  fusion: { es: 'fusión', en: 'fusion' },
  witness: { es: 'testigo', en: 'witness' },
  deviation: { es: 'desviación', en: 'deviation' },
  tolerance: { es: 'tolerancia', en: 'tolerance' },
  primitive: { es: 'primitiva', en: 'primitive' },
} as const

export type GlossaryTerm = keyof typeof GLOSSARY

/** Plural forms that differ from "+s" (Spanish accents drop, English irregulars). */
export const GLOSSARY_PLURAL: Partial<Record<GlossaryTerm, { es: string; en: string }>> = {
  session: { es: 'sesiones', en: 'sessions' },
  correction: { es: 'correcciones', en: 'corrections' },
  fusion: { es: 'fusiones', en: 'fusions' },
  epoch: { es: 'épocas geométricas', en: 'geometric epochs' },
  deviation: { es: 'desviaciones', en: 'deviations' },
  acta: { es: 'actas', en: 'certificates' },
}
