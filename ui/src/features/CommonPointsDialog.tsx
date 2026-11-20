/**
 * CommonPointsDialog — the same physical point marked on two scans of the project, and what the
 * points say about how one scan lies on the other (USER 2026-11-17: "para comparar dos o más
 * épocas hay que emparejar sus escalas … deberían marcarse dos puntos y decir estos son los
 * mismos en ambas … y que se ajusten a uno de ellos, o a una medida externa … para que se puedan
 * posar uno sobre otro y comparar").
 *
 * A point is marked by a CLICK on the model, once on each scan (open one, mark; open the other,
 * mark the same point). From the points marked on both — two at least — the server gives the
 * factor, the turn about the vertical and the shift that bring the scan on screen onto the other
 * one, and what each point is left off by. With a scale GIVEN (a measurement from outside) the
 * points give the turn and the shift only.
 *
 * REPORTED, NOT APPLIED: nothing of either scan or of the project is changed from here.
 *
 * Presentational: the points, the marking state and every action come from App (the click on the
 * model arrives there).
 */
import { useMemo } from 'react'
import { Crosshair, LocateFixed, Plus, Trash2, X } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Badge } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { Field, NumberInput, Select } from '../components/ui/Field'
import { KeyValue, Row, Section, Stack } from '../components/ui/Panel'
import { Table, type Column } from '../components/ui/Table'
import { useFmt, useT } from '../i18n'

export interface CommonMark { position: number[]; geometry_epoch: number | null; stale: boolean }
export interface CommonPoint { id: string; name: string; marks: Record<string, CommonMark> }

/** the similarity that brings `scan` onto `reference`, as the server reports it */
export interface CommonFit {
  scan: string
  reference: string
  scale: number
  scale_source: 'points' | 'given'
  yaw_deg: number
  translation: number[]
  rms_m: number
  max_m: number
  baseline_m: number
  pairs: number
  points: Array<{ id: string; name: string; residual_m: number }>
  stale: string[]
  marked_on_one_only: string[]
}

interface Props {
  open: boolean
  sessionId: string | null
  /** the scan on screen: the one that is marked now, and brought onto the other */
  scanKey: string | null
  /** the project's other scans */
  others: Array<{ key: string; label: string }>
  other: string | null
  onOther: (key: string) => void
  points: CommonPoint[]
  /** the point waiting for its click on this scan */
  placing: string | null
  fit: CommonFit | null
  /** why there is no fit yet (fewer than two points on both, …), in the server's words */
  fitNote: string | null
  loadError: string | null
  /** a scale given from outside ('' = from the points) */
  givenScale: number | ''
  onGivenScale: (v: number | '') => void
  onNewPoint: () => void
  onMark: (pointId: string) => void
  onCancelMark: () => void
  onUnmark: (pointId: string) => void
  onDelete: (pointId: string) => void
  onFocus: (point: CommonPoint) => void
  onClose: () => void
}

export function CommonPointsDialog(p: Props) {
  const t = useT()
  const fmt = useFmt()
  const residual = useMemo(() => new Map((p.fit?.points ?? []).map(x => [x.id, x.residual_m])), [p.fit])
  const placingName = p.placing ? p.points.find(x => x.id === p.placing)?.name ?? '' : ''

  const columns: Column<CommonPoint>[] = useMemo(() => {
    const state = (m: CommonMark | undefined) => !m
      ? <Badge size="sm" tone="neutral">{t('commonPoints.notMarked')}</Badge>
      : m.stale ? <Badge size="sm" tone="warn" title={t('commonPoints.staleHint')}>{t('commonPoints.stale')}</Badge>
        : <Badge size="sm" tone="ok">{t('commonPoints.marked')}</Badge>
    return [
      // (the point's name takes what the other columns leave: it is what tells one point from another)
      { id: 'name', header: t('commonPoints.point'), width: '100%', sortValue: r => r.name, cell: r => <span title={r.name}>{r.name}</span> },
      { id: 'here', header: t('commonPoints.onThis'), cell: r => state(p.scanKey ? r.marks[p.scanKey] : undefined) },
      { id: 'other', header: t('commonPoints.onOther'), cell: r => state(p.other ? r.marks[p.other] : undefined) },
      { id: 'residual', header: t('commonPoints.residual'), align: 'right', mono: true, sortValue: r => residual.get(r.id) ?? null,
        cell: r => { const v = residual.get(r.id); return v == null ? '' : fmt.mm(v * 1000, 1) } },
      { id: 'actions', header: '', align: 'right', cell: r => (
          <Row justify="end">
            <IconButton size="sm" label={t('commonPoints.markHere')} icon={<Crosshair aria-hidden />} disabled={!p.scanKey}
              onClick={e => { e.stopPropagation(); p.onMark(r.id) }} />
            <IconButton size="sm" label={t('commonPoints.focus')} icon={<LocateFixed aria-hidden />} disabled={!p.scanKey || !r.marks[p.scanKey]}
              onClick={e => { e.stopPropagation(); p.onFocus(r) }} />
            {p.scanKey && r.marks[p.scanKey] &&
              <IconButton size="sm" label={t('commonPoints.unmark')} icon={<X aria-hidden />} onClick={e => { e.stopPropagation(); p.onUnmark(r.id) }} />}
            <IconButton size="sm" variant="danger" label={t('commonPoints.delete')} icon={<Trash2 aria-hidden />} onClick={e => { e.stopPropagation(); p.onDelete(r.id) }} />
          </Row>
        ) },
    ]
  }, [t, fmt, p, residual])

  return (
    <Dialog open={p.open} title={t('commonPoints.title')} subtitle={[p.sessionId, p.scanKey].filter(Boolean).join(', ') || undefined}
      size="md" nonBlocking position="right" onClose={p.onClose}>
      <Stack gap={3}>
        {p.loadError && <Banner tone="err">{t('commonPoints.loadFailed', { detail: p.loadError })}</Banner>}
        {p.placing && (
          <Banner tone="info" icon={<Crosshair aria-hidden />}
            action={<Button size="sm" icon={<X aria-hidden />} onClick={p.onCancelMark}>{t('common.cancel')}</Button>}>
            {t('commonPoints.placing', { name: placingName })}
          </Banner>
        )}
        <Section title={t('commonPoints.pairTitle')} hint={t('commonPoints.pairHint')}>
          <Row wrap>
            <Field label={t('commonPoints.other')}>
              <Select<string> value={p.other ?? ''} onChange={p.onOther} placeholder={t('commonPoints.other')} aria-label={t('commonPoints.other')}
                options={p.others.map(o => ({ value: o.key, label: o.label }))} />
            </Field>
            <Field label={t('commonPoints.givenScale')} hint={t('commonPoints.givenScaleHint')}>
              <NumberInput value={p.givenScale} onChange={p.onGivenScale} digits={5} min={0} step={0.001} aria-label={t('commonPoints.givenScale')} />
            </Field>
          </Row>
        </Section>
        <Section title={t('commonPoints.listTitle')}
          actions={<Button size="sm" variant="primary" icon={<Plus aria-hidden />} disabled={!p.scanKey || !!p.placing} title={t('commonPoints.newHint')} onClick={p.onNewPoint}>{t('commonPoints.new')}</Button>}>
          <Table<CommonPoint> columns={columns} rows={p.points} rowKey={r => r.id}
            onRowClick={r => { if (p.scanKey && r.marks[p.scanKey]) p.onFocus(r) }}
            emptyTitle={t('commonPoints.emptyTitle')} emptyDescription={t('commonPoints.emptyDesc')} />
        </Section>
        <Section title={t('commonPoints.resultTitle')}>
          {p.fit ? (
            <Stack gap={2}>
              <KeyValue label={t('commonPoints.factor')} mono>{`× ${fmt.number(p.fit.scale, 5)}`}{p.fit.scale_source === 'given' ? ` (${t('commonPoints.given')})` : ''}</KeyValue>
              <KeyValue label={t('commonPoints.yaw')} mono>{`${fmt.number(p.fit.yaw_deg, 2)}°`}</KeyValue>
              <KeyValue label={t('commonPoints.shift')} mono>{`${p.fit.translation.map(v => fmt.number(v, 3)).join(', ')} m`}</KeyValue>
              <KeyValue label={t('commonPoints.rms')} mono>{`${fmt.mm(p.fit.rms_m * 1000, 1)} / ${fmt.mm(p.fit.max_m * 1000, 1)}`}</KeyValue>
              <KeyValue label={t('commonPoints.used')}>{t('commonPoints.usedValue', { n: p.fit.pairs, baseline: fmt.number(p.fit.baseline_m, 2) })}</KeyValue>
              {p.fit.pairs === 2 && <Banner tone="info" compact>{t('commonPoints.twoPoints')}</Banner>}
              {p.fit.stale.length > 0 && <Banner tone="warn" compact>{t('commonPoints.staleLeftOut', { names: p.fit.stale.join(', ') })}</Banner>}
              <Banner tone="info" compact>{t('commonPoints.notApplied')}</Banner>
            </Stack>
          ) : (
            <Banner tone="info" compact>{p.fitNote || t('commonPoints.needTwo')}</Banner>
          )}
        </Section>
      </Stack>
    </Dialog>
  )
}
