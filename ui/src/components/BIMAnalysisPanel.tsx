/**
 * BIM Analysis Panel — construction progress report shown in the Acta /
 * Quality tab when the deviation comparison is active: per-element quality,
 * advance and global progress. Tone colours come from the state tokens.
 */
import { Lock, CheckCircle2, Hammer, Award, Square } from 'lucide-react'
import type { ReactNode } from 'react'
import { Section, KeyValue, Row, Stack } from './ui/Panel'
import { Progress } from './ui/Progress'
import { Badge, type BadgeTone } from './ui/Badge'
import { useFmt, useT } from '../i18n'

interface ElementMeta {
  element_key: string
  label: string
  ifc_type: string
  status: string
  quality?: string        // good | regular | bad | not_built
  advance_pct?: number
  coverage_pct?: number
  coverage_cumulative?: number
  occluded_pct?: number
  element_state?: string   // NOT_STARTED | IN_PROGRESS | COMPLETED | VERIFIED | OCCLUDED_FROZEN
  correctness_pct?: number
  mean_mm?: number
  bim_surface_m2?: number
  total_points?: number
}

interface SabanaMeta {
  date: string
  tolerance_mm: number
  total_points: number
  global_advance_pct: number
  quality_thresholds?: { good_pct: number; regular_pct: number }
  summary: { total_elements: number; evaluated: number; unmatched: number; errors: number }
  elements: ElementMeta[]
}

const QUALITY_TONE: Record<string, BadgeTone> = { good: 'ok', regular: 'warn', bad: 'err', not_built: 'neutral' }
const STATE_TONE: Record<string, BadgeTone> = { NOT_STARTED: 'neutral', IN_PROGRESS: 'info', COMPLETED: 'ok', VERIFIED: 'brand', OCCLUDED_FROZEN: 'warn' }
const STATE_ICON: Record<string, ReactNode> = {
  NOT_STARTED: <Square aria-hidden />, IN_PROGRESS: <Hammer aria-hidden />, COMPLETED: <CheckCircle2 aria-hidden />, VERIFIED: <Award aria-hidden />, OCCLUDED_FROZEN: <Lock aria-hidden />,
}

function progressTone(pct: number): 'ok' | 'warn' | 'brand' | 'err' {
  if (pct >= 80) return 'ok'
  if (pct >= 50) return 'warn'
  if (pct > 0) return 'brand'
  return 'err'
}

export function BIMAnalysisPanel({ meta, sessionId }: { meta: SabanaMeta; sessionId: string }) {
  const t = useT()
  const fmt = useFmt()
  const evaluated = meta.elements.filter(e => e.status === 'evaluated')
  const unmatched = meta.elements.filter(e => e.status !== 'evaluated')
  const counts = { good: 0, regular: 0, bad: 0, not_built: unmatched.length }
  evaluated.forEach(e => { const q = (e.quality || 'bad') as keyof typeof counts; if (q in counts) counts[q]++ })

  return (
    <div className="stac-bap">
      <Section title={t('report.title')}>
        <KeyValue label={t('properties.session')} mono>{sessionId}</KeyValue>
        <KeyValue label={t('scans.date')} mono>{fmt.dateTime(meta.date)}</KeyValue>
        <KeyValue label={t('deviation.tolerance')} mono>{fmt.mm(meta.tolerance_mm, 0)}</KeyValue>
        <KeyValue label={t('report.scanPoints')} mono>{fmt.integer(meta.total_points)}</KeyValue>
      </Section>
      <Section title={t('report.globalProgress')}>
        <Progress value={Math.min(meta.global_advance_pct, 100)} tone={progressTone(meta.global_advance_pct)} size="md" />
        <Row wrap>
          {(['good', 'regular', 'bad', 'not_built'] as const).map(q => (
            <Badge key={q} tone={QUALITY_TONE[q]} mono>{fmt.integer(counts[q])} {t(`quality.${q}`)}</Badge>
          ))}
        </Row>
      </Section>
      <Section title={t('report.elements', { n: meta.elements.length })}>
        <Stack gap={2}>
          {evaluated.map(el => (
            <div key={el.element_key} className="stac-bap__el">
              <Row className="stac-bap__head">
                <span className="stac-bap__label" title={el.label}>{el.label.split(':').slice(0, -1).join(':') || el.label}</span>
                <span className="stac-bap__type">{el.ifc_type.replace('Ifc', '')}</span>
                <Badge tone={QUALITY_TONE[el.quality || 'bad']} size="sm">{t(`quality.${el.quality || 'bad'}`)}</Badge>
              </Row>
              <Progress value={Math.min(el.advance_pct || 0, 100)} tone={el.quality === 'good' ? 'ok' : el.quality === 'regular' ? 'warn' : 'err'} />
              <Row wrap className="stac-bap__meta">
                {el.element_state && <Badge tone={STATE_TONE[el.element_state] ?? 'neutral'} size="sm">{STATE_ICON[el.element_state]} {t(`elementState.${el.element_state}`)}</Badge>}
                {(el.occluded_pct || 0) > 0 && <span className="stac-mono">{t('report.occluded', { pct: fmt.percent((el.occluded_pct || 0) / 100) })}</span>}
                {el.coverage_cumulative != null && <span className="stac-mono">{t('report.cumulative', { pct: fmt.percent(el.coverage_cumulative / 100) })}</span>}
                <span className="stac-mono">{t('report.correctness', { pct: fmt.percent((el.correctness_pct || 0) / 100, 1) })}</span>
                <span className="stac-mono">{t('report.mean', { mm: fmt.mm(el.mean_mm || 0) })}</span>
                {el.bim_surface_m2 ? <span className="stac-mono">{t('report.surface', { area: fmt.area(el.bim_surface_m2) })}</span> : null}
              </Row>
            </div>
          ))}
          {unmatched.length > 0 && (
            <Section flush title={t('report.notBuilt', { n: unmatched.length })}>
              {unmatched.map(el => (
                <Row key={el.element_key} className="stac-bap__head stac-bap__el--unbuilt">
                  <span className="stac-bap__label" title={el.label}>{el.label.split(':').slice(0, -1).join(':') || el.label}</span>
                  <span className="stac-bap__type">{el.ifc_type.replace('Ifc', '')}</span>
                  <Badge size="sm">{t('quality.not_built')}</Badge>
                </Row>
              ))}
            </Section>
          )}
        </Stack>
      </Section>
    </div>
  )
}
