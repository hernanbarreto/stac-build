/**
 * CalibrationDialog — the scan's calibration measurements, taken ON THE 3D MODEL
 * (USER 2026-11-13: "para calibrar debe hacerse tomando mediciones sobre el modelo 3d no sobre
 * las imagenes porque es dificil, se debe poder seleccionar que una medicion es de calibracion
 * y deben quedar almacenadas").
 *
 * The user measures a distance on the model with the distance tool and presses the pin of its
 * readout: the measurement arrives here as the DRAFT (its two picked points and the model's
 * length), the user states the real length, its role (gauge: calibrates the scale / test:
 * measures the error left), its instrument — or the other scan of the project the length was
 * taken from, to bring this scan to that scan's scale — and it is stored by the server
 * (`/api/calibration/{session}`; the model's length is computed there from the two points).
 *
 * The window lists what is stored with the scale factor the gauge measurements give and the
 * error on the test ones. The factor is REPORTED: nothing is rescaled from here.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Crosshair, Pin, Ruler, Save, Trash2 } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Badge } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { EmptyState } from '../components/ui/EmptyState'
import { Field, Input, NumberInput, SegmentedControl, Select } from '../components/ui/Field'
import { KeyValue, Row, Section, Stack } from '../components/ui/Panel'
import { Table, type Column } from '../components/ui/Table'
import { useToast } from '../components/ui/Toast'
import { useFmt, useT } from '../i18n'

export type CalRole = 'gauge' | 'test'
export type CalTool = 'tape' | 'laser' | 'total_station' | 'other_scan'

export interface CalMeasurement {
  id: string
  a: number[]
  b: number[]
  measured_m: number
  value_m: number
  sigma_m: number
  role: CalRole
  tool: CalTool
  note: string
  geometry_epoch: number
  created: string
  reference_scan?: string
}

interface CalRow {
  id: string
  role: CalRole
  tool: CalTool
  measured_m: number
  value_m: number
  sigma_m: number
  ratio: number
  geometry_epoch: number
  stale: boolean
  calibrated_m: number | null
  residual_m: number | null
}

export interface CalSummary {
  geometry_epoch: number
  n: number
  n_gauge: number
  n_test: number
  n_stale: number
  scale_factor: number | null
  weighting: string | null
  rows: CalRow[]
  gauge_fit_residual_rms_m?: number
  gauge_fit_residual_max_m?: number
  gauge_ratio_min?: number
  gauge_ratio_max?: number
  test_abs_error_m?: { median: number | null; max: number; n: number }
  test_rel_error?: { median: number | null; max: number; n: number }
}

export interface CalibrationDraft { readoutId: string; a: number[]; b: number[]; metres: number }

interface CalibrationDialogProps {
  open: boolean
  sessionId: string | null
  /** the active scan ("date/source") or null for the session's default */
  scanKey: string | null
  /** the project's other scans: a length can be taken from one of them */
  otherScans: Array<{ key: string; label: string }>
  /** a distance just measured on the model, waiting to be marked (null: the list only) */
  draft: CalibrationDraft | null
  onClose: () => void
  /** start the distance tool on the model (the window closes; the pin of the readout comes back here) */
  onMeasure: () => void
  /** the stored set changed (or was loaded): the viewer draws its marks */
  onChanged: (measurements: CalMeasurement[], summary: CalSummary) => void
  /** the draft was stored: its temporary measurement leaves the scene */
  onStored: (readoutId: string) => void
  /** fly to a stored measurement */
  onFrame: (m: CalMeasurement) => void
}

const scanQuery = (scanKey: string | null) => (scanKey ? `?scan_key=${encodeURIComponent(scanKey)}` : '')

export function CalibrationDialog({ open, sessionId, scanKey, otherScans, draft, onClose, onMeasure, onChanged, onStored, onFrame }: CalibrationDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const { notify } = useToast()

  const [measurements, setMeasurements] = useState<CalMeasurement[]>([])
  const [summary, setSummary] = useState<CalSummary | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // ── the draft's form ──
  const [value, setValue] = useState<number | ''>('')
  const [sigma, setSigma] = useState<number | ''>('')
  const [role, setRole] = useState<CalRole>('gauge')
  const [tool, setTool] = useState<CalTool>('tape')
  const [refScan, setRefScan] = useState<string>('')
  const [note, setNote] = useState('')

  const take = useCallback((d: { document?: { measurements?: CalMeasurement[] }; summary?: CalSummary }) => {
    const ms = d.document?.measurements ?? []
    setMeasurements(ms)
    setSummary(d.summary ?? null)
    if (d.summary) onChanged(ms, d.summary)
  }, [onChanged])

  useEffect(() => {
    if (!open || !sessionId) return
    let cancelled = false
    setLoadError(null)
    fetch(`/api/calibration/${encodeURIComponent(sessionId)}${scanQuery(scanKey)}`)
      .then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`); return d })
      .then(d => { if (!cancelled) take(d) })
      .catch(e => { if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e)) })
    return () => { cancelled = true }
  }, [open, sessionId, scanKey, take])

  // a new draft starts a clean form (the instrument and the role chosen last are kept)
  useEffect(() => { if (draft) { setValue(''); setSigma(''); setNote('') } }, [draft])

  const canSave = !!draft && typeof value === 'number' && value > 0 && (tool !== 'other_scan' || !!refScan) && !busy

  const store = useCallback(async () => {
    if (!draft || !sessionId || typeof value !== 'number') return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/measurements`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_key: scanKey ?? undefined, a: draft.a, b: draft.b, value_m: value, sigma_m: typeof sigma === 'number' ? sigma : 0,
                               role, tool, note, ...(tool === 'other_scan' ? { reference_scan: refScan } : {}) }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
      onStored(draft.readoutId)
      notify(t('calibration.saved', { id: d.measurement?.id ?? '' }), { tone: 'ok' })
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [draft, sessionId, scanKey, value, sigma, role, tool, note, refScan, take, onStored, notify, t])

  const patch = useCallback(async (id: string, body: Record<string, unknown>) => {
    if (!sessionId) return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/measurements/${encodeURIComponent(id)}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scan_key: scanKey ?? undefined, ...body }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [sessionId, scanKey, take, notify, t])

  const remove = useCallback(async (id: string) => {
    if (!sessionId) return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/measurements/${encodeURIComponent(id)}${scanQuery(scanKey)}`, { method: 'DELETE' })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
      notify(t('calibration.deleted', { id }), { tone: 'ok' })
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [sessionId, scanKey, take, notify, t])

  const byId = useMemo(() => new Map(measurements.map(m => [m.id, m])), [measurements])
  const scanLabel = useCallback((key?: string) => otherScans.find(s => s.key === key)?.label ?? key ?? '', [otherScans])

  const columns: Column<CalRow>[] = useMemo(() => [
    { id: 'id', header: t('calibration.colId'), mono: true, cell: r => r.id },
    { id: 'role', header: t('calibration.role'),
      cell: r => <Badge tone={r.stale ? 'neutral' : r.role === 'gauge' ? 'ok' : 'warn'} size="sm">{r.role === 'gauge' ? t('calibration.roleGauge') : t('calibration.roleTest')}</Badge> },
    { id: 'model', header: t('calibration.colModel'), align: 'right', mono: true, cell: r => fmt.lengthText(r.measured_m), sortValue: r => r.measured_m },
    { id: 'real', header: t('calibration.colReal'), align: 'right', mono: true, cell: r => fmt.lengthText(r.value_m), sortValue: r => r.value_m },
    { id: 'ratio', header: t('calibration.colRatio'), align: 'right', mono: true, cell: r => fmt.number(r.ratio, 4), sortValue: r => r.ratio },
    { id: 'residual', header: t('calibration.colResidual'), align: 'right', mono: true,
      cell: r => (r.residual_m == null ? '' : fmt.mm(r.residual_m * 1000, 1)), sortValue: r => (r.residual_m == null ? null : Math.abs(r.residual_m)) },
    { id: 'source', header: t('calibration.tool'),
      cell: r => { const m = byId.get(r.id); return m?.tool === 'other_scan' ? t('calibration.fromScan', { name: scanLabel(m.reference_scan) }) : t(`calibration.tool_${r.tool}`) } },
    { id: 'state', header: '', cell: r => (r.stale ? <Badge tone="neutral" size="sm" title={t('calibration.staleHint')}>{t('calibration.stale', { n: r.geometry_epoch })}</Badge> : byId.get(r.id)?.note ?? '') },
    { id: 'actions', header: '', align: 'right',
      cell: r => (
        <Row justify="end">
          <IconButton size="sm" label={t('calibration.frame')} icon={<Crosshair aria-hidden />} onClick={() => { const m = byId.get(r.id); if (m) onFrame(m) }} />
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => patch(r.id, { role: r.role === 'gauge' ? 'test' : 'gauge' })}>
            {r.role === 'gauge' ? t('calibration.makeTest') : t('calibration.makeGauge')}
          </Button>
          <IconButton size="sm" variant="danger" label={t('common.delete')} icon={<Trash2 aria-hidden />} disabled={busy} onClick={() => remove(r.id)} />
        </Row>
      ) },
  ], [t, fmt, byId, scanLabel, busy, patch, remove, onFrame])

  const factor = summary?.scale_factor ?? null

  return (
    <Dialog open={open} title={t('calibration.title')} subtitle={[sessionId, scanKey].filter(Boolean).join(', ') || undefined}
      icon={<Pin aria-hidden />} size="lg" onClose={onClose}
      footer={
        <>
          <Button icon={<Ruler aria-hidden />} disabled={!sessionId} onClick={onMeasure} title={t('calibration.measureHint')}>{t('calibration.measure')}</Button>
          <Button onClick={onClose}>{t('common.close')}</Button>
        </>
      }>
      <Stack gap={3}>
        {loadError && <Banner tone="err">{t('calibration.loadFailed', { detail: loadError })}</Banner>}

        {draft && (
          <Section title={t('calibration.draftTitle')}>
            <Stack gap={2}>
              <KeyValue label={t('calibration.modelLength')} mono>{fmt.lengthText(draft.metres)}</KeyValue>
              <Row wrap>
                <Field label={t('calibration.realLength')}>
                  <NumberInput value={value} onChange={setValue} unit="m" min={0} step={0.001} data-autofocus aria-label={t('calibration.realLength')} />
                </Field>
                <Field label={t('calibration.sigma')} hint={t('calibration.sigmaHint')}>
                  <NumberInput value={sigma} onChange={setSigma} unit="m" min={0} step={0.001} aria-label={t('calibration.sigma')} />
                </Field>
              </Row>
              <Field label={t('calibration.role')} hint={t('calibration.roleHint')}>
                <SegmentedControl<CalRole> value={role} onChange={setRole} ariaLabel={t('calibration.role')}
                  options={[{ value: 'gauge', label: t('calibration.roleGauge') }, { value: 'test', label: t('calibration.roleTest') }]} />
              </Field>
              <Row wrap>
                <Field label={t('calibration.tool')}>
                  <Select<CalTool> value={tool} onChange={setTool} aria-label={t('calibration.tool')}
                    options={[{ value: 'tape', label: t('calibration.tool_tape') }, { value: 'laser', label: t('calibration.tool_laser') },
                              { value: 'total_station', label: t('calibration.tool_total_station') },
                              { value: 'other_scan', label: t('calibration.tool_other_scan'), disabled: otherScans.length === 0 }]} />
                </Field>
                {tool === 'other_scan' && (
                  <Field label={t('calibration.referenceScan')} hint={t('calibration.referenceScanHint')}>
                    <Select<string> value={refScan} onChange={setRefScan} placeholder={t('calibration.referenceScan')} aria-label={t('calibration.referenceScan')}
                      options={otherScans.map(s => ({ value: s.key, label: s.label }))} />
                  </Field>
                )}
              </Row>
              <Field label={t('calibration.note')}>
                <Input value={note} onChange={e => setNote(e.target.value)} aria-label={t('calibration.note')} />
              </Field>
              <Row justify="end">
                <Button variant="primary" icon={<Save aria-hidden />} disabled={!canSave} onClick={store}>{t('calibration.save')}</Button>
              </Row>
            </Stack>
          </Section>
        )}

        <Section title={t('calibration.summaryTitle')}>
          {summary && summary.n > 0 ? (
            <Stack gap={1}>
              <KeyValue label={t('calibration.factor')} mono>{factor != null ? `× ${fmt.number(factor, 5)}` : t('calibration.noFactor')}</KeyValue>
              <KeyValue label={t('calibration.counts')}>{t('calibration.countsValue', { gauge: summary.n_gauge, test: summary.n_test, stale: summary.n_stale })}</KeyValue>
              {summary.gauge_fit_residual_rms_m != null && (
                <KeyValue label={t('calibration.fitResidual')} mono>{`${fmt.mm(summary.gauge_fit_residual_rms_m * 1000, 1)} rms, ${fmt.mm((summary.gauge_fit_residual_max_m ?? 0) * 1000, 1)} max`}</KeyValue>
              )}
              {summary.gauge_ratio_min != null && summary.n_gauge > 1 && (
                <KeyValue label={t('calibration.ratioRange')} mono>{`${fmt.number(summary.gauge_ratio_min, 4)} - ${fmt.number(summary.gauge_ratio_max ?? summary.gauge_ratio_min, 4)}`}</KeyValue>
              )}
              {summary.test_abs_error_m && (
                <KeyValue label={t('calibration.testError')} mono>{`${fmt.mm((summary.test_abs_error_m.median ?? 0) * 1000, 1)} ${t('calibration.median')}, ${fmt.mm(summary.test_abs_error_m.max * 1000, 1)} max (${summary.test_abs_error_m.n})`}</KeyValue>
              )}
              <p className="stac-muted">{t('calibration.notApplied')}</p>
            </Stack>
          ) : (
            <EmptyState compact title={t('calibration.empty')} description={t('calibration.emptyHint')} />
          )}
        </Section>

        {summary && summary.rows.length > 0 && (
          <Table columns={columns} rows={summary.rows} rowKey={r => r.id} caption={t('calibration.rows', { n: summary.rows.length })} />
        )}
      </Stack>
    </Dialog>
  )
}
