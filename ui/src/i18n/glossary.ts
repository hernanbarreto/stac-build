/**
 * Glossary — one concept, one word, in the system's three languages (prompt_ui.txt §8; French
 * since 2026-11-14: USER "los 3 idiomas del sistema, ingles, español y frances").
 *
 * Messages in es.json / en.json reference a term as `{@term}` (lower case) or
 * `{@Term}` (capitalised); `check-strings.mjs` fails when a term is spelled
 * literally in the Spanish or the English JSON, so the vocabulary cannot drift.
 *
 * Decision: the backend calls the top-level folder a "session" (`/sessions`)
 * and the multi-scan container a "project" (`/api/project/<sid>`); they are
 * the same object. The UI uses exactly ONE word for it — {@session} — and
 * never says "project". A {@scan} is one dated capture inside a session; a
 * {@campaign} is the set of scans compared over time.
 */
export const GLOSSARY = {
  session: { es: 'sesión', en: 'session', fr: 'session' },
  scan: { es: 'escaneo', en: 'scan', fr: 'scan' },
  campaign: { es: 'campaña', en: 'campaign', fr: 'campagne' },
  instance: { es: 'instancia', en: 'instance', fr: 'instance' },
  epoch: { es: 'época geométrica', en: 'geometric epoch', fr: 'époque géométrique' },
  acta: { es: 'acta', en: 'certificate', fr: 'certificat' },
  correction: { es: 'corrección', en: 'correction', fr: 'correction' },
  fusion: { es: 'fusión', en: 'fusion', fr: 'fusion' },
  witness: { es: 'testigo', en: 'witness', fr: 'témoin' },
  deviation: { es: 'desviación', en: 'deviation', fr: 'écart' },
  tolerance: { es: 'tolerancia', en: 'tolerance', fr: 'tolérance' },
  primitive: { es: 'primitiva', en: 'primitive', fr: 'primitive' },
} as const

export type GlossaryTerm = keyof typeof GLOSSARY

/** Plural forms that differ from "+s" (Spanish accents drop, English irregulars). */
export const GLOSSARY_PLURAL: Partial<Record<GlossaryTerm, { es: string; en: string; fr: string }>> = {
  session: { es: 'sesiones', en: 'sessions', fr: 'sessions' },
  correction: { es: 'correcciones', en: 'corrections', fr: 'corrections' },
  fusion: { es: 'fusiones', en: 'fusions', fr: 'fusions' },
  epoch: { es: 'épocas geométricas', en: 'geometric epochs', fr: 'époques géométriques' },
  deviation: { es: 'desviaciones', en: 'deviations', fr: 'écarts' },
  acta: { es: 'actas', en: 'certificates', fr: 'certificats' },
}
