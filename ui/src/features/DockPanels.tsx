/**
 * Dock panels: ConsolePanel (backend log stream), JobsPanel (pipeline
 * stages, correction / meshing / resume / extraction progress),
 * PropertiesPanel (inspector: session, scan, cloud, epoch, selected
 * instance). Presentational.
 */
import { useEffect, useRef } from 'react'
import { Ban, Terminal, ListTodo } from 'lucide-react'
import type { SegmentInstance } from '../components/Viewport'
import { Progress, type ProgressStage } from '../components/ui/Progress'
import { Button } from '../components/ui/Button'
import { EmptyState } from '../components/ui/EmptyState'
import { KeyValue, Section, Stack } from '../components/ui/Panel'
import { Badge, ColorDot } from '../components/ui/Badge'
import { useFmt, useT } from '../i18n'

export interface LogEntry { ts: string; level: string; msg: string }

export function ConsolePanel({ logs }: { logs: LogEntry[] }) {
  const t = useT()
  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }) }, [logs])
  return (
    <div className="stac-console stac-selectable" role="log" aria-live="polite" aria-label={t('dock.console')}>
      {logs.length === 0 && <EmptyState compact icon={<Terminal aria-hidden />} title={t('console.empty')} />}
      {logs.map((e, i) => (
        <div key={i} className={`stac-console__line stac-console__line--${e.level}`}>
          <span className="stac-console__ts">{e.ts}</span>
          <span className="stac-console__msg">{e.msg}</span>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  )
}

export interface JobsState {
  pipeline: { session_id?: string; status: string; stages: ProgressStage[] } | null
  onCancelPipeline: () => void
  correction: { pct: number; detail: string } | null
  meshing: { phase: string; total?: number; done?: number } | null
  object: { phase: string; total?: number; done?: number } | null
  resume: { pct: number; detail: string } | null
  extracting: Record<string, number>
  tasks: Array<{ label: string; pct: number; detail: string }>
}

export function JobsPanel(j: JobsState) {
  const t = useT()
  const any = j.pipeline || j.correction || (j.meshing && j.meshing.phase !== 'idle') || (j.object && j.object.phase !== 'idle') || j.resume || Object.keys(j.extracting).length || j.tasks.length
  return (
    <div className="stac-jobs">
      {!any && <EmptyState compact icon={<ListTodo aria-hidden />} title={t('jobs.empty')} description={t('jobs.emptyDesc')} />}
      {j.pipeline && (
        <Section title={t('jobs.pipeline', { session: j.pipeline.session_id ?? '' })} flush actions={
          j.pipeline.status === 'running' ? <Button size="sm" variant="danger" icon={<Ban aria-hidden />} onClick={j.onCancelPipeline}>{t('common.cancel')}</Button>
            : <Badge tone={j.pipeline.status === 'done' ? 'ok' : j.pipeline.status === 'failed' ? 'err' : 'neutral'}>{t(`jobStatus.${j.pipeline.status}`)}</Badge>}>
          <Progress stages={j.pipeline.stages} tone="brand" />
        </Section>
      )}
      {j.correction && <Section title={t('jobs.correction')} flush><Progress value={j.correction.pct} label={j.correction.detail} /></Section>}
      {j.meshing && j.meshing.phase !== 'idle' && (
        <Section title={t('jobs.meshing')} flush><Progress value={j.meshing.total ? ((j.meshing.done || 0) / j.meshing.total) * 100 : 0} label={t(`meshPhase.${j.meshing.phase}`, { done: j.meshing.done || 0, total: j.meshing.total || 0 })} /></Section>
      )}
      {j.object && j.object.phase !== 'idle' && (
        <Section title={t('jobs.object')} flush><Progress value={j.object.total ? ((j.object.done || 0) / j.object.total) * 100 : 0} label={t(`shapePhase.${j.object.phase}`, { done: j.object.done || 0, total: j.object.total || 0 })} /></Section>
      )}
      {j.resume && <Section title={t('jobs.resume')} flush><Progress value={j.resume.pct} label={j.resume.detail} /></Section>}
      {Object.entries(j.extracting).map(([sid, pct]) => (
        <Section key={sid} title={t('jobs.extracting', { session: sid })} flush><Progress value={pct} /></Section>
      ))}
      {j.tasks.map((tk, i) => (
        <Section key={i} title={tk.label} flush><Progress value={tk.pct} label={tk.detail} /></Section>
      ))}
      <p className="stac-jobs__hint">{t('jobs.hint')}</p>
    </div>
  )
}

export interface PropertiesState {
  sessionId: string | null
  scanLabel: string | null
  isReference: boolean
  pointCount: number
  fps: number
  epoch: number | null
  pendingEpochs: number
  segments: SegmentInstance[]
  selected: SegmentInstance | null
  unsegmentedCount: number | null
  bimCount: number
  hasSabana: boolean
}

export function PropertiesPanel(s: PropertiesState) {
  const t = useT()
  const fmt = useFmt()
  if (!s.sessionId) return <EmptyState icon={<ListTodo aria-hidden />} title={t('properties.noSession')} description={t('welcome.pickSession')} />
  return (
    <Stack gap={1} className="stac-properties">
      <Section title={t('properties.session')}>
        <KeyValue label={t('properties.name')} mono>{s.sessionId}</KeyValue>
        <KeyValue label={t('header.scanMenu')}>{s.scanLabel ?? t('status.dash')}{s.isReference && <Badge tone="brand" size="sm">{t('header.reference')}</Badge>}</KeyValue>
        <KeyValue label={t('properties.pointsRendered')} mono>{s.pointCount > 0 ? fmt.integer(s.pointCount) : t('status.dash')}</KeyValue>
        <KeyValue label={t('status.epoch')} mono>{s.epoch != null ? fmt.integer(s.epoch) : t('status.dash')}{s.pendingEpochs > 0 && <Badge tone="warn" size="sm">{t('status.pending', { count: s.pendingEpochs })}</Badge>}</KeyValue>
        <KeyValue label={t('instances.title')} mono>{fmt.integer(s.segments.length)}</KeyValue>
        <KeyValue label={t('instances.unsegmented')} mono>{s.unsegmentedCount != null ? fmt.integer(s.unsegmentedCount) : t('status.dash')}</KeyValue>
        <KeyValue label={t('properties.bimModels')} mono>{fmt.integer(s.bimCount)}</KeyValue>
        <KeyValue label={t('properties.comparison')}>{s.hasSabana ? t('common.yes') : t('common.no')}</KeyValue>
      </Section>
      <Section title={t('properties.selectedInstance')}>
        {!s.selected && <p className="stac-section__hint">{t('properties.selectHint')}</p>}
        {s.selected && (
          <>
            <KeyValue label={t('properties.label')}><ColorDot color={s.selected.color} /> {s.selected.label}</KeyValue>
            <KeyValue label={t('properties.id')} mono>{s.selected.id}</KeyValue>
            <KeyValue label={t('scans.points')} mono>{fmt.integer(s.selected.totalPoints)}</KeyValue>
            <KeyValue label={t('properties.visible')}>{s.selected.visible ? t('common.yes') : t('common.no')}</KeyValue>
            {s.selected.excluded && <KeyValue label={t('properties.excluded')}>{t('common.yes')}</KeyValue>}
          </>
        )}
      </Section>
    </Stack>
  )
}
