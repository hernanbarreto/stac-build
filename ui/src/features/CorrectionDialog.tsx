/**
 * CorrectionDialog — mark duplicated instances -> Analyze & correct; floor
 * alignment; revisit detection; derived artifacts with their epoch; the
 * append-only ledger. When the session holds more than one epoch it shows the
 * epoch selector (USER 2026-09-16: every epoch lives, you only choose which
 * one is on screen — there is no approving and no undoing).
 */
import { AlertTriangle, ArrowDownToLine, CheckCircle2, Repeat, Wrench, XCircle } from 'lucide-react'
import type { SegmentInstance } from '../components/Viewport'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Checkbox } from '../components/ui/Field'
import { Banner } from '../components/ui/Banner'
import { Badge } from '../components/ui/Badge'
import { Progress } from '../components/ui/Progress'
import { ColorDot } from '../components/ui/Badge'
import { Section, Stack, Row } from '../components/ui/Panel'
import { EmptyState } from '../components/ui/EmptyState'
import { useFmt, useT } from '../i18n'

export interface GateRow { name: string; group?: string; passed: boolean; advisory?: boolean; detail?: string }

interface CorrectionDialogProps {
  open: boolean
  onClose: () => void
  state: any
  report: any
  running: boolean
  progress: { pct: number; detail: string } | null
  segments: SegmentInstance[]
  selected: Set<number>
  onToggleSelected: (id: number) => void
  overrideScale: boolean
  onOverrideScale: (v: boolean) => void
  ledger: any[] | null
  artifacts: any[] | null
  onRun: () => void
  onFloor: () => void
  onRevisit: () => void
  /** Every epoch the session holds, and which one is on screen. */
  epochs?: { epoch: number; live: boolean; potree: boolean }[]
  onSelectEpoch: (epoch: number) => void
}

export function GateList({ gates, compact = false }: { gates: GateRow[]; compact?: boolean }) {
  const t = useT()
  return (
    <ul className={`stac-gates ${compact ? 'stac-gates--compact' : ''}`.trim()}>
      {gates.map(g => (
        <li key={g.name + String(g.group ?? '')} className="stac-gates__row">
          <span className={`stac-gates__icon stac-gates__icon--${g.advisory && !g.passed ? 'warn' : g.passed ? 'ok' : 'err'}`}>
            {g.advisory && !g.passed ? <AlertTriangle aria-hidden /> : g.passed ? <CheckCircle2 aria-hidden /> : <XCircle aria-hidden />}
          </span>
          <span className="stac-gates__name">{g.name}</span>
          <span className="stac-gates__detail">{g.detail}{g.advisory && !g.passed ? ` ${t('correction.advisoryNote')}` : ''}</span>
        </li>
      ))}
    </ul>
  )
}

export function CorrectionDialog(p: CorrectionDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const rep = p.state?.report
  const epochs = p.epochs || []
  return (
    <Dialog open={p.open} title={t('correction.title')} onClose={p.onClose} icon={<Wrench aria-hidden />} size="lg">
      {/* USER 2026-09-16: "todas viven, solo se seleccionan y la que se
          selecciona se muestra". Approve used to delete every other epoch and
          Undo the current one, so the session could only hold two states and
          one wrong click destroyed the other. Nothing is destroyed here: every
          epoch is a button, and the live one is marked. */}
      {epochs.length > 1 ? (
        <Stack gap={3}>
          <Banner tone="info" title={t('correction.epochsTitle', { n: epochs.length })}>
            {t('correction.epochsDesc')}
          </Banner>
          {(rep?.gates || []).length > 0 && <GateList gates={rep.gates} />}
          <ul className="stac-list stac-epochs">
            {epochs.map(e => (
              <li key={e.epoch} className={e.live ? 'is-live' : undefined}>
                <Button
                  block
                  variant={e.live ? 'primary' : undefined}
                  icon={e.live ? <CheckCircle2 aria-hidden /> : undefined}
                  disabled={p.running || e.live}
                  onClick={() => p.onSelectEpoch(e.epoch)}
                >
                  {e.epoch === 0
                    ? t('correction.epochOriginal')
                    : t('correction.epochN', { epoch: e.epoch })}
                  {e.live ? ` — ${t('correction.epochLive')}` : ''}
                </Button>
              </li>
            ))}
          </ul>
        </Stack>
      ) : (
        <Stack gap={4}>
          <p className="stac-dialog__message">{t('correction.description')}</p>
          {p.running && p.progress && <Progress value={p.progress.pct} label={p.progress.detail} tone="brand" size="md" />}
          {p.report?.status === 'rejected' && (
            <Banner tone="err" title={t('correction.rejectedTitle')}>
              <Stack gap={1}>
                <span>{p.report.rejection_reason}</span>
                {p.report.suggestion && <span>{p.report.suggestion}</span>}
                {(p.report.gates || []).length > 0 && <GateList gates={p.report.gates} compact />}
                {(p.report.gates || []).some((g: any) => g.name === 'scale_vs_da3' && !g.passed) && (
                  <Checkbox checked={p.overrideScale} onChange={p.onOverrideScale} label={t('correction.overrideScale')} />
                )}
              </Stack>
            </Banner>
          )}
          <Section title={t('correction.markTitle')} flush hint={t('correction.markHint')}>
            {p.segments.length === 0 && <EmptyState compact title={t('correction.noSegments')} description={t('instances.emptyDesc')} />}
            <ul className="stac-picklist">
              {p.segments.map(seg => (
                <li key={seg.key} className={`stac-picklist__item ${p.selected.has(seg.id) ? 'stac-picklist__item--on' : ''}`.trim()}>
                  <Checkbox checked={p.selected.has(seg.id)} disabled={p.running} onChange={() => p.onToggleSelected(seg.id)}
                    label={<Row><ColorDot color={seg.color} /><span>{seg.label}</span></Row>} />
                  <span className="stac-mono stac-picklist__meta">{fmt.integer(seg.totalPoints)}</span>
                </li>
              ))}
            </ul>
            <Button block variant="primary" icon={<Wrench aria-hidden />} loading={p.running} disabled={p.selected.size === 0} onClick={p.onRun}>{t('correction.analyzeCorrect')}</Button>
          </Section>
          <Section title={t('correction.floorTitle')} flush hint={t('correction.floorHint')}>
            <Button block icon={<ArrowDownToLine aria-hidden />} loading={p.running} onClick={p.onFloor}>{t('correction.alignFloor')}</Button>
          </Section>
          <Section title={t('correction.revisitTitle')} flush hint={t('correction.revisitHint')}>
            <Button block icon={<Repeat aria-hidden />} loading={p.running} onClick={p.onRevisit}>{t('correction.detectRevisits')}</Button>
          </Section>
          {p.artifacts && p.artifacts.length > 0 && (
            <Section title={t('correction.artifacts')} flush>
              <ul className="stac-artifacts">
                {p.artifacts.map((a: any) => (
                  <li key={a.path} className="stac-artifacts__row">
                    <span className="stac-artifacts__name">{a.kind}: {a.name}</span>
                    {a.stale
                      ? <Badge tone="warn" size="sm" title={t('correction.staleHint', { epoch: a.geometry_epoch })}>{t('correction.stale', { epoch: a.geometry_epoch })}</Badge>
                      : <Badge tone="ok" size="sm">{t('correction.epochBadge', { epoch: a.geometry_epoch })}</Badge>}
                  </li>
                ))}
              </ul>
            </Section>
          )}
          {p.ledger && p.ledger.length > 0 && (
            <Section title={t('correction.history')} flush>
              <ul className="stac-ledger">
                {p.ledger.slice().reverse().map((e: any) => (
                  <li key={e.correction_id} className="stac-ledger__row">
                    <span className="stac-ledger__kind">{e.kind}</span>
                    <span className="stac-mono">{t('correction.epochRange', { from: e.epoch_from, to: e.epoch_to })}</span>
                    <span className="stac-ledger__meta">{e.operator}</span>
                    <span className="stac-ledger__meta stac-mono">{fmt.dateTime(e.created_at)}</span>
                    {/* 'approved'/'undone' only appear in a ledger written before 2026-09-16 */}
                    <Badge size="sm" tone={e.verdict === 'rejected' ? 'err' : e.verdict === 'undone' ? 'warn' : 'ok'}>{t(`verdict.${e.verdict}`)}</Badge>
                    {e.overrides && Object.keys(e.overrides).length > 0 && <Badge tone="warn" size="sm">{t('correction.override')}</Badge>}
                  </li>
                ))}
              </ul>
            </Section>
          )}
        </Stack>
      )}
    </Dialog>
  )
}
